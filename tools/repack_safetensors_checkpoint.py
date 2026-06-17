#!/usr/bin/env python3
"""Repack an indexed safetensors checkpoint into larger shards."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_SOURCE = Path("/models/GLM-5.2-NVFP4")
DEFAULT_OUTPUT = Path("/models/GLM-5.2-NVFP4-repacked")
DEFAULT_SHARD_SIZE = 5 * 1024**3
SKIP_AUXILIARY_NAMES = {".cache", ".git", "model.safetensors.index.json"}


def parse_size(text: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?\s*", text, re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError(f"Invalid size: {text!r}")
    value = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "t": 1000**4,
        "tb": 1000**4,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    return int(value * multipliers[unit])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repack an indexed safetensors checkpoint.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-size", type=parse_size, default=DEFAULT_SHARD_SIZE)
    parser.add_argument(
        "--keep-inputscales",
        action="store_true",
        default=True,
        help="Copy shards containing only *.input_scale tensors unchanged.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=4, sort_keys=False)
        f.write("\n")


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def keys_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        grouped[shard].append(key)
    return {shard: sorted(keys) for shard, keys in grouped.items()}


def inputscale_shards(weight_map: dict[str, str]) -> set[str]:
    grouped = keys_by_shard(weight_map)
    return {
        shard
        for shard, keys in grouped.items()
        if keys and all(key.endswith(".input_scale") for key in keys)
    }


def copy_auxiliary_files(source: Path, out: Path) -> list[str]:
    copied: list[str] = []
    for item in sorted(source.iterdir(), key=lambda p: p.name):
        if item.name in SKIP_AUXILIARY_NAMES or item.name.endswith(".safetensors"):
            continue
        dst = out / item.name
        if item.is_dir() and not item.is_symlink():
            shutil.copytree(item, dst, symlinks=True)
            copied.append(item.name + "/")
        else:
            shutil.copy2(item, dst, follow_symlinks=True)
            copied.append(item.name)
    return copied


def prepare_output(out: Path, overwrite: bool) -> None:
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def copy_kept_shards(source: Path, out: Path, kept_shards: set[str]) -> None:
    for shard in sorted(kept_shards):
        tmp = out / (shard + ".tmp")
        shutil.copy2(source / shard, tmp, follow_symlinks=True)
        os.replace(tmp, out / shard)


def repack_tensors(
    source: Path,
    out: Path,
    weight_map: dict[str, str],
    kept_shards: set[str],
    target_shard_size: int,
) -> dict[str, str]:
    grouped = keys_by_shard(weight_map)
    output_weight_map: dict[str, str] = {
        key: shard
        for key, shard in weight_map.items()
        if shard in kept_shards
    }

    current_tensors: dict[str, torch.Tensor] = {}
    current_size = 0
    shard_idx = 0
    written: list[tuple[str, int, int]] = []

    def flush() -> None:
        nonlocal current_tensors, current_size, shard_idx
        if not current_tensors:
            return
        shard_idx += 1
        shard_name = f"model-tmp-{shard_idx:05d}.safetensors"
        save_file(current_tensors, str(out / shard_name))
        for key in current_tensors:
            output_weight_map[key] = shard_name
        written.append((shard_name, len(current_tensors), (out / shard_name).stat().st_size))
        print(
            f"[{shard_idx:05d}] {shard_name}: "
            f"{len(current_tensors)} tensor(s), {(out / shard_name).stat().st_size / 1024**3:.2f} GiB",
            flush=True,
        )
        current_tensors = {}
        current_size = 0
        gc.collect()

    source_shards = sorted(shard for shard in grouped if shard not in kept_shards)
    for source_idx, shard in enumerate(source_shards, 1):
        with safe_open(str(source / shard), framework="pt", device="cpu") as f:
            for key in grouped[shard]:
                tensor = f.get_tensor(key).contiguous()
                size = tensor_nbytes(tensor)
                if current_tensors and current_size + size > target_shard_size:
                    flush()
                current_tensors[key] = tensor
                current_size += size
        print(f"read source shard [{source_idx:03d}/{len(source_shards):03d}] {shard}", flush=True)

    flush()

    total = len(written)
    rename_map = {
        shard_name: f"model-{idx:05d}-of-{total:05d}.safetensors"
        for idx, (shard_name, _tensor_count, _size) in enumerate(written, 1)
    }
    for old_name, new_name in rename_map.items():
        os.replace(out / old_name, out / new_name)
    output_weight_map = {
        key: rename_map.get(shard_name, shard_name)
        for key, shard_name in output_weight_map.items()
    }

    print(f"Repacked tensor shards written: {len(written)}", flush=True)
    return output_weight_map


def write_updated_index(out: Path, source_index: dict[str, Any], weight_map: dict[str, str]) -> None:
    metadata = dict(source_index.get("metadata", {}))
    metadata["total_size"] = sum((out / shard).stat().st_size for shard in set(weight_map.values()))
    index = {
        "metadata": metadata,
        "weight_map": dict(sorted(weight_map.items())),
    }
    write_json(out / "model.safetensors.index.json", index)


def main() -> None:
    args = parse_args()
    index = read_json(args.source / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    kept_shards = inputscale_shards(weight_map) if args.keep_inputscales else set()
    source_shards = set(weight_map.values())
    missing = sorted(shard for shard in source_shards if not (args.source / shard).exists())
    if missing:
        raise FileNotFoundError(f"Missing indexed shard(s): {missing[:10]}")

    print(f"source: {args.source}")
    print(f"output: {args.out}")
    print(f"target shard size: {args.shard_size / 1024**3:.2f} GiB")
    print(f"indexed source shards: {len(source_shards)}")
    print(f"input-scale shards kept unchanged: {', '.join(sorted(kept_shards)) or 'none'}")
    print(f"input-scale keys kept unchanged: {sum(key.endswith('.input_scale') for key in weight_map)}")
    print(f"tensor keys to repack: {sum(shard not in kept_shards for shard in weight_map.values())}")

    if args.dry_run:
        print("Dry run complete; no files written.")
        return

    prepare_output(args.out, args.overwrite)
    copied_aux = copy_auxiliary_files(args.source, args.out)
    print(f"Copied {len(copied_aux)} auxiliary file(s)/dir(s).")
    copy_kept_shards(args.source, args.out, kept_shards)
    output_weight_map = repack_tensors(
        args.source,
        args.out,
        weight_map,
        kept_shards,
        args.shard_size,
    )
    write_updated_index(args.out, index, output_weight_map)
    print(f"Wrote repacked checkpoint to {args.out}")


if __name__ == "__main__":
    main()
