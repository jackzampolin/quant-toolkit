#!/usr/bin/env python3
"""Replay text calibration datasets through SGLang's chat completions API.

Examples:
    .venv/bin/python tools/sglang_text_calib_requests.py \
        --calib-config configs/calib_mimo_v25.toml \
        --model MiMo-V2.5 \
        --concurrency 32

    .venv/bin/python tools/sglang_text_calib_requests.py \
        --jsonl data/text/deep_calib.jsonl \
        --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    path: Path
    label: str
    limit: int | None
    max_len: int | None = None
    batch_size: int | None = None


@dataclass(frozen=True)
class RequestItem:
    dataset: str
    line_no: int
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--calib-config",
        help="TOML file with [[dataset]] entries, as used by quantize.py.",
    )
    input_group.add_argument(
        "--jsonl",
        action="append",
        help="Text calibration JSONL file. May be passed multiple times.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base directory for relative dataset paths in --calib-config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional explicit cap per dataset. By default, full datasets are replayed.",
    )
    parser.add_argument(
        "--total-limit",
        type=int,
        default=None,
        help="Optional explicit cap across all datasets.",
    )
    parser.add_argument(
        "--respect-config-limits",
        action="store_true",
        help="Honor TOML dataset limit fields. By default they are ignored.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model", default=os.environ.get("SGLANG_MODEL", "mimo-v2.5"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate for each request.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--extra-json",
        default=None,
        help="JSON object merged into every chat completions payload.",
    )
    parser.add_argument(
        "--drop-final-assistant",
        action="store_true",
        help="Drop a trailing assistant message before sending the request.",
    )
    parser.add_argument(
        "--skip-multimodal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip records whose message content contains non-text parts.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dry-run-count",
        type=int,
        default=3,
        help="Number of normalized payloads to print in --dry-run mode.",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--print-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print every response body as requests complete.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def resolve_dataset_path(path_value: str, data_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (data_dir / path).resolve(strict=False)


def effective_limit(
    dataset_limit: Any,
    cli_limit: int | None,
    respect_config_limits: bool,
) -> int | None:
    limit = int(dataset_limit) if respect_config_limits and dataset_limit is not None else None
    if cli_limit is None:
        return limit
    if limit is None:
        return cli_limit
    return min(limit, cli_limit)


def load_dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    data_dir = Path(args.data_dir).expanduser().resolve(strict=False)
    specs: list[DatasetSpec] = []

    if args.calib_config:
        config_path = Path(args.calib_config).expanduser()
        with config_path.open("rb") as f:
            config = tomllib.load(f)
        datasets = config.get("dataset", [])
        if not datasets:
            raise SystemExit(f"No [[dataset]] entries in {config_path}")

        for idx, dataset in enumerate(datasets):
            if "path" not in dataset:
                raise SystemExit(f"dataset[{idx}] missing 'path' in {config_path}")
            if dataset.get("multimodal", False):
                continue
            path = resolve_dataset_path(str(dataset["path"]), data_dir)
            specs.append(
                DatasetSpec(
                    path=path,
                    label=str(dataset["path"]),
                    limit=effective_limit(
                        dataset.get("limit"),
                        args.limit,
                        args.respect_config_limits,
                    ),
                    max_len=dataset.get("max_len"),
                    batch_size=dataset.get("batch_size"),
                )
            )
        if not specs:
            raise SystemExit(f"No text datasets found in {config_path}")
        return specs

    assert args.jsonl is not None
    for jsonl in args.jsonl:
        path = Path(jsonl).expanduser().resolve(strict=False)
        specs.append(
            DatasetSpec(
                path=path,
                label=str(path),
                limit=args.limit,
            )
        )
    return specs


def content_is_text_only(content: Any) -> bool:
    if isinstance(content, str):
        return True
    if not isinstance(content, list):
        return False
    return all(isinstance(part, dict) and part.get("type") == "text" for part in content)


def messages_are_text_only(messages: list[dict[str, Any]]) -> bool:
    return all(content_is_text_only(message.get("content", "")) for message in messages)


def normalize_messages(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if "messages" in record:
        messages = record["messages"]
    elif "prompt" in record or "text" in record:
        messages = [{"role": "user", "content": record.get("prompt") or record.get("text")}]
    else:
        return None

    if not isinstance(messages, list):
        return None

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        normalized.append(dict(message))

    if args.drop_final_assistant and normalized and normalized[-1].get("role") == "assistant":
        normalized = normalized[:-1]

    if args.skip_multimodal and not messages_are_text_only(normalized):
        return None

    return normalized


def load_request_items(args: argparse.Namespace, specs: list[DatasetSpec]) -> list[RequestItem]:
    extra_payload = json.loads(args.extra_json) if args.extra_json else {}
    if not isinstance(extra_payload, dict):
        raise SystemExit("--extra-json must be a JSON object")

    items: list[RequestItem] = []
    for spec in specs:
        if not spec.path.exists():
            raise SystemExit(f"Dataset not found: {spec.path}")

        sent_from_dataset = 0
        skipped = 0
        with spec.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if spec.limit is not None and sent_from_dataset >= spec.limit:
                    break
                if args.total_limit is not None and len(items) >= args.total_limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{spec.path}:{line_no}: invalid JSON: {exc}") from exc

                messages = normalize_messages(record, args)
                if messages is None:
                    skipped += 1
                    continue

                payload: dict[str, Any] = {
                    "model": args.model,
                    "messages": messages,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                }
                payload.update(extra_payload)
                items.append(RequestItem(spec.label, line_no, payload))
                sent_from_dataset += 1

        if skipped:
            print(f"  skipped {skipped} non-text/unsupported records from {spec.label}")
        if args.total_limit is not None and len(items) >= args.total_limit:
            break

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(items)

    return items


def request_url(args: argparse.Namespace) -> str:
    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    return f"{args.base_url.rstrip('/')}{endpoint}"


async def send_one(
    session: Any,
    semaphore: asyncio.Semaphore,
    url: str,
    item: RequestItem,
) -> tuple[RequestItem, int | str, str]:
    async with semaphore:
        try:
            async with session.post(url, json=item.payload) as response:
                body = await response.text()
                return item, response.status, body
        except Exception as exc:  # noqa: BLE001 - keep calibration replay running by default.
            return item, type(exc).__name__, str(exc)


async def send_all(args: argparse.Namespace, items: list[RequestItem]) -> None:
    try:
        import aiohttp
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "aiohttp is required for sending requests. Install it in this repo venv "
            "with: uv pip install aiohttp"
        ) from exc

    url = request_url(args)
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)

    print(f"Sending {len(items)} requests to {url} (concurrency={args.concurrency})")
    started = time.time()
    ok = 0
    failed = 0
    first_error: str | None = None

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        tasks = [
            asyncio.create_task(send_one(session, semaphore, url, item))
            for item in items
        ]

        for done_count, task in enumerate(asyncio.as_completed(tasks), start=1):
            item, status, body = await task
            if isinstance(status, int) and 200 <= status < 300:
                ok += 1
            else:
                failed += 1
                if first_error is None:
                    first_error = (
                        f"{item.dataset}:{item.line_no}: status={status}, "
                        f"body={body[:500]}"
                    )
                if args.fail_fast:
                    for pending in tasks:
                        pending.cancel()
                    raise SystemExit(first_error)

            if args.print_outputs:
                print(f"\n# response {done_count}: {item.dataset}:{item.line_no} status={status}")
                print(body)

            if done_count % args.progress_every == 0 or done_count == len(items):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"  {done_count}/{len(items)} "
                    f"ok={ok} failed={failed} rate={done_count / elapsed:.1f}/s"
                )

    elapsed = time.time() - started
    print(f"Done. ok={ok} failed={failed} elapsed={elapsed:.1f}s")
    if first_error:
        print(f"First error: {first_error}", file=sys.stderr)
        if failed:
            raise SystemExit(1)


def print_plan(specs: list[DatasetSpec], items: list[RequestItem], args: argparse.Namespace) -> None:
    print("Calibration replay plan:")
    for spec in specs:
        limit = "all" if spec.limit is None else spec.limit
        details = [f"limit={limit}"]
        if spec.max_len is not None:
            details.append(f"max_len={spec.max_len}")
        if spec.batch_size is not None:
            details.append(f"batch_size={spec.batch_size}")
        print(f"  {spec.label} -> {spec.path} ({', '.join(details)})")
    print(f"Prepared {len(items)} text request(s)")

    if args.dry_run:
        for item in items[: args.dry_run_count]:
            print(f"\n# {item.dataset}:{item.line_no}")
            print(json.dumps(item.payload, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    specs = load_dataset_specs(args)
    items = load_request_items(args, specs)
    print_plan(specs, items, args)

    if not items:
        raise SystemExit("No text requests prepared")
    if args.dry_run:
        return

    asyncio.run(send_all(args, items))


if __name__ == "__main__":
    main()
