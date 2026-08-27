#!/usr/bin/env python3
"""Record a sealed GLM candidate KLD runtime from live Docker state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "quant-toolkit.glm53-kld-runtime.v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _docker_inspect(container: str) -> dict:
    result = subprocess.run(
        ["docker", "inspect", container],
        check=True,
        capture_output=True,
        text=True,
    )
    values = json.loads(result.stdout)
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError(f"unexpected docker inspect result for {container}")
    return values[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--panel-dir", required=True, type=Path)
    parser.add_argument("--panel-role", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--coverage-receipt", required=True, type=Path)
    parser.add_argument("--scale-install-receipt", required=True, type=Path)
    parser.add_argument("--scale-construction-receipt", required=True, type=Path)
    parser.add_argument("--quant-toolkit-commit", required=True)
    parser.add_argument("--capture-tool", required=True, type=Path)
    parser.add_argument("--replay-tool", required=True, type=Path)
    parser.add_argument("--input-scale-scope", required=True)
    args = parser.parse_args(argv)

    candidate = args.candidate_model.resolve()
    panel_path = args.panel_dir / "panel.json"
    panel = _load_json(panel_path)
    selected = [
        record
        for record in panel.get("windows", [])
        if record.get("role") == args.panel_role
    ]
    if not selected:
        raise RuntimeError(f"panel has no role {args.panel_role!r}")
    predictions = {int(record["prediction_positions"]) for record in selected}
    if len(predictions) != 1:
        raise RuntimeError("panel role has nonuniform prediction lengths")
    prediction_positions = predictions.pop()

    inspect = _docker_inspect(args.container)
    state = inspect.get("State", {})
    if not state.get("Running"):
        raise RuntimeError(f"container is not running: {args.container}")
    config = inspect.get("Config", {})
    image_id = inspect.get("Image")
    coverage = _load_json(args.coverage_receipt)
    install = _load_json(args.scale_install_receipt)
    construction = _load_json(args.scale_construction_receipt)
    command = config.get("Cmd") or []

    runtime = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "run_name": args.run_name,
        "model": {
            "checkpoint_path": str(candidate),
            "bf16_source_revision": args.source_revision,
            "config_sha256": sha256_file(candidate / "config.json"),
            "index_sha256": sha256_file(
                candidate / "model.safetensors.index.json"
            ),
            "input_scale_sha256": sha256_file(
                candidate / "model-inputscales.safetensors"
            ),
            "coverage_receipt": {
                "path": str(args.coverage_receipt.resolve()),
                "file_sha256": sha256_file(args.coverage_receipt),
                "quantized_weight_tensors": coverage["coverage"][
                    "quantized_weight_tensors"
                ],
                "quantized_parameters": coverage["coverage"][
                    "quantized_parameters"
                ],
                "candidate_tensor_bytes": coverage["candidate"]["tensor_bytes"],
            },
            "scale_install_receipt": {
                "path": str(args.scale_install_receipt.resolve()),
                "file_sha256": sha256_file(args.scale_install_receipt),
                "receipt_sha256": install.get("receipt_sha256"),
            },
            "scale_construction_receipt": {
                "path": str(args.scale_construction_receipt.resolve()),
                "file_sha256": sha256_file(args.scale_construction_receipt),
                "receipt_sha256": construction.get("receipt_sha256"),
            },
            "input_scale_scope": args.input_scale_scope,
        },
        "runtime": {
            "host": os.uname().nodename,
            "container_name": args.container,
            "container_id": inspect.get("Id"),
            "container_created_at": inspect.get("Created"),
            "container_started_at": state.get("StartedAt"),
            "image": config.get("Image"),
            "image_id": image_id,
            "entrypoint": config.get("Entrypoint"),
            "command": command,
            "tensor_parallel_size": 4,
            "decode_context_parallel_size": 1,
            "kv_cache_requested": "fp8",
            "kv_cache_effective": "fp8_ds_mla",
            "max_model_len": 4096,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.90,
            "prefix_caching": False,
            "enforce_eager": True,
        },
        "suite": {
            "panel_path": str(panel_path.resolve()),
            "panel_sha256": sha256_file(panel_path),
            "dataset_revision": args.dataset_revision,
            "sealed_corpus_sha256": panel.get("sealed_corpus_sha256"),
            "role": args.panel_role,
            "windows": len(selected),
            "tokens_per_window": prediction_positions + 1,
            "predictions_per_window": prediction_positions,
            "total_prediction_positions": len(selected) * prediction_positions,
        },
        "tooling": {
            "quant_toolkit_commit": args.quant_toolkit_commit,
            "recorder_sha256": sha256_file(Path(__file__).resolve()),
            "capture_tool_sha256": sha256_file(args.capture_tool),
            "replay_tool_sha256": sha256_file(args.replay_tool),
        },
    }
    runtime["runtime_sha256"] = canonical_json_sha256(runtime)
    _write_json_atomic(args.output, runtime)
    print(json.dumps({"event": "glm53_candidate_runtime_recorded", **runtime}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
