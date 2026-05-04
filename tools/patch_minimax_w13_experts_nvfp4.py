#!/usr/bin/env python3
"""Repair a small set of MiniMax NVFP4 expert branches from BF16 source weights.

This patches w1/w3 for the known inconsistent experts by recomputing:
  - packed NVFP4 weights
  - per-block weight_scale
  - shared per-tensor weight_scale_2

The output checkpoint is written side-by-side. Unchanged files are symlinked.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_EXPERTS = [
    "model.layers.0.block_sparse_moe.experts.87",
    "model.layers.0.block_sparse_moe.experts.198",
    "model.layers.61.block_sparse_moe.experts.24",
    "model.layers.61.block_sparse_moe.experts.75",
    "model.layers.61.block_sparse_moe.experts.183",
    "model.layers.61.block_sparse_moe.experts.225",
]

E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
BLOCK_SIZE = 16
FP4_MAX = 6.0
FP8_MAX = 448.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-bf16", required=True, help="Source BF16 checkpoint dir")
    parser.add_argument("--src-nvfp4", required=True, help="Source NVFP4 checkpoint dir")
    parser.add_argument("--out", required=True, help="Output checkpoint dir")
    parser.add_argument(
        "--expert",
        action="append",
        dest="experts",
        help="Expert prefix to patch, e.g. model.layers.0.block_sparse_moe.experts.87",
    )
    return parser.parse_args()


def load_index(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def symlink_tree(src_dir: Path, dst_dir: Path, skip_names: set[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=False)
    for item in src_dir.iterdir():
        if item.name in skip_names:
            continue
        os.symlink(item.resolve(), dst_dir / item.name)


def reduce_block_amax(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    assert weight.shape[-1] % block_size == 0
    reshaped = weight.reshape(*weight.shape[:-1], -1, block_size)
    return reshaped.abs().amax(dim=-1).float()


def pack_nvfp4(weight: torch.Tensor, weight_scale_2: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    per_block_amax = reduce_block_amax(weight, block_size)
    per_block_scale = per_block_amax / (FP4_MAX * weight_scale_2.to(per_block_amax.device))
    per_block_scale[per_block_scale == 0] = 1.0
    per_block_scale_fp8 = per_block_scale.to(torch.float8_e4m3fn)

    reshaped = weight.reshape(*weight.shape[:-1], -1, block_size)
    scaled = reshaped / ((per_block_scale_fp8.to(torch.float32) * weight_scale_2.to(torch.float32)).unsqueeze(-1))
    packed = pack_fp4(scaled.reshape(weight.shape))
    return packed, per_block_scale_fp8


def pack_fp4(weight: torch.Tensor) -> torch.Tensor:
    device = weight.device
    bounds = E2M1_BOUNDS.to(device)

    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()
    ordinals = torch.searchsorted(bounds, weight_abs, out_int32=True).to(torch.uint8)
    odd_bounds = bounds[[1, 3, 5]]
    equals_odd_bounds = torch.any(weight_abs.unsqueeze(-1) == odd_bounds, dim=-1).to(torch.uint8)
    q = (sign_bit << 3) + ordinals + equals_odd_bounds
    return ((q[..., 1::2] << 4) | q[..., 0::2]).contiguous()


def shared_weight_scale_2(w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    max_abs = torch.maximum(w1.abs().amax().float(), w3.abs().amax().float())
    return max_abs / (FP4_MAX * FP8_MAX)


def load_bf16_tensors(src_dir: Path, weight_map: dict[str, str], keys: list[str]) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_shard[weight_map[key]].append(key)

    out: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(str(src_dir / shard), framework="pt", device="cpu") as f:
            for key in shard_keys:
                out[key] = f.get_tensor(key)
    return out


def load_shard_tensors(src_dir: Path, shard_names: set[str]) -> dict[str, dict[str, torch.Tensor]]:
    out = {}
    for shard in sorted(shard_names):
        tensors = {}
        with safe_open(str(src_dir / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        out[shard] = tensors
    return out


def main() -> None:
    args = parse_args()
    src_bf16 = Path(args.src_bf16)
    src_nvfp4 = Path(args.src_nvfp4)
    out_dir = Path(args.out)
    experts = args.experts or list(DEFAULT_EXPERTS)

    if out_dir.exists():
        raise SystemExit(f"Output directory already exists: {out_dir}")

    bf16_index = load_index(src_bf16 / "model.safetensors.index.json")
    nvfp4_index = load_index(src_nvfp4 / "model.safetensors.index.json")
    bf16_map = bf16_index["weight_map"]
    nvfp4_map = nvfp4_index["weight_map"]

    bf16_weight_keys = []
    affected_shards: set[str] = set()
    for expert in experts:
        for proj in ("w1", "w3"):
            bf16_weight_keys.append(f"{expert}.{proj}.weight")
            for suffix in ("weight", "weight_scale", "weight_scale_2"):
                key = f"{expert}.{proj}.{suffix}"
                affected_shards.add(nvfp4_map[key])

    bf16_weights = load_bf16_tensors(src_bf16, bf16_map, bf16_weight_keys)
    shard_tensors = load_shard_tensors(src_nvfp4, affected_shards)

    symlink_tree(
        src_nvfp4,
        out_dir,
        skip_names=affected_shards | {"model.safetensors.index.json"},
    )

    for expert in experts:
        w1_key = f"{expert}.w1.weight"
        w3_key = f"{expert}.w3.weight"
        w1 = bf16_weights[w1_key]
        w3 = bf16_weights[w3_key]
        scale2 = shared_weight_scale_2(w1, w3).reshape(())

        for proj, weight in (("w1", w1), ("w3", w3)):
            packed, scale = pack_nvfp4(weight, scale2, BLOCK_SIZE)
            shard = nvfp4_map[f"{expert}.{proj}.weight"]
            shard_tensors[shard][f"{expert}.{proj}.weight"] = packed
            shard_tensors[shard][f"{expert}.{proj}.weight_scale"] = scale
            shard_tensors[shard][f"{expert}.{proj}.weight_scale_2"] = scale2.clone()

    for shard, tensors in shard_tensors.items():
        save_file(tensors, str(out_dir / shard))

    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(nvfp4_index, f, indent=2)

    validate(out_dir, experts)
    print(f"Wrote patched checkpoint to {out_dir}")


def validate(out_dir: Path, experts: list[str]) -> None:
    index = load_index(out_dir / "model.safetensors.index.json")["weight_map"]
    for expert in experts:
        scale_keys = [f"{expert}.w1.weight_scale_2", f"{expert}.w3.weight_scale_2"]
        weight_keys = [f"{expert}.w1.weight", f"{expert}.w3.weight"]
        block_keys = [f"{expert}.w1.weight_scale", f"{expert}.w3.weight_scale"]
        tensors = {}
        shards = {index[k] for k in scale_keys + weight_keys + block_keys}
        for shard in shards:
            with safe_open(str(out_dir / shard), framework="pt", device="cpu") as f:
                for key in scale_keys + weight_keys + block_keys:
                    if key in f.keys():
                        tensors[key] = f.get_tensor(key)

        if not torch.equal(tensors[scale_keys[0]], tensors[scale_keys[1]]):
            raise RuntimeError(f"weight_scale_2 mismatch remains for {expert}")
        if tensors[weight_keys[0]].shape[-1] * 2 != BLOCK_SIZE * tensors[block_keys[0]].shape[-1]:
            raise RuntimeError(f"Unexpected packed layout for {expert}.w1")


if __name__ == "__main__":
    main()
