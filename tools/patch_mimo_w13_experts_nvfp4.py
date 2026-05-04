#!/usr/bin/env python3
"""Repair MiMo NVFP4 gate/up expert pairs with tied global weight scales.

SGLang's ModelOpt NVFP4 MoE path fuses gate_proj and up_proj into a single
w13 matrix and expects both branches to share the same weight_scale_2. This
script finds mismatched MiMo expert pairs, reloads their BF16 dequantized
weights, recomputes one shared NVFP4 global scale from those BF16 weights, and
re-packs both branches.

Unchanged files are symlinked into the output directory; affected shards are
rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
BLOCK_SIZE = 16
FP4_MAX = 6.0
FP8_MAX = 448.0
EXPERT_SCALE_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp\.experts\.\d+)\."
    r"(gate_proj|up_proj)\.weight_scale_2$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-bf16",
        default="/data/models/MiMo-V2.5-BF16-qkv-deinterleaved",
        help="Source BF16 dequantized MiMo checkpoint",
    )
    parser.add_argument(
        "--src-nvfp4",
        default="/data/models/MiMo-V2.5-NVFP4",
        help="Source NVFP4 MiMo checkpoint",
    )
    parser.add_argument("--out", help="Output checkpoint directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report mismatched expert pairs and affected shards",
    )
    parser.add_argument(
        "--expert",
        action="append",
        dest="experts",
        help=(
            "Expert prefix to patch, e.g. "
            "model.layers.7.mlp.experts.179. If omitted, auto-detect mismatches."
        ),
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for detecting existing scale mismatches",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for detecting existing scale mismatches",
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


def discover_mismatched_experts(
    src_nvfp4: Path, rtol: float, atol: float
) -> list[str]:
    scales: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for shard in sorted(src_nvfp4.glob("*.safetensors")):
        if shard.name == "amax_checkpoint.safetensors":
            continue
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                match = EXPERT_SCALE_RE.match(key)
                if match is None:
                    continue
                expert, proj = match.groups()
                scales[expert][proj] = f.get_tensor(key).float().reshape(())

    mismatched = []
    for expert, pair in sorted(scales.items(), key=_expert_sort_key_from_item):
        if "gate_proj" not in pair or "up_proj" not in pair:
            continue
        if not torch.allclose(pair["gate_proj"], pair["up_proj"], rtol=rtol, atol=atol):
            mismatched.append(expert)
    return mismatched


def _expert_sort_key_from_item(item: tuple[str, dict[str, torch.Tensor]]) -> tuple[int, int]:
    return _expert_sort_key(item[0])


def _expert_sort_key(expert: str) -> tuple[int, int]:
    match = re.search(r"layers\.(\d+)\.mlp\.experts\.(\d+)$", expert)
    if match is None:
        return (10**9, 10**9)
    return (int(match.group(1)), int(match.group(2)))


def reduce_block_amax(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    if weight.shape[-1] % block_size != 0:
        raise ValueError(
            f"Last dimension {weight.shape[-1]} is not divisible by block size {block_size}"
        )
    reshaped = weight.reshape(*weight.shape[:-1], -1, block_size)
    return reshaped.abs().amax(dim=-1).float()


def pack_fp4(weight: torch.Tensor) -> torch.Tensor:
    bounds = E2M1_BOUNDS.to(weight.device)
    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()
    ordinals = torch.searchsorted(bounds, weight_abs, out_int32=True).to(torch.uint8)
    odd_bounds = bounds[[1, 3, 5]]
    equals_odd_bounds = torch.any(weight_abs.unsqueeze(-1) == odd_bounds, dim=-1).to(
        torch.uint8
    )
    q = (sign_bit << 3) + ordinals + equals_odd_bounds
    return ((q[..., 1::2] << 4) | q[..., 0::2]).contiguous()


def shared_weight_scale_2(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    max_abs = torch.maximum(
        gate_weight.abs().amax().float(), up_weight.abs().amax().float()
    )
    return max_abs / (FP4_MAX * FP8_MAX)


def pack_nvfp4(
    weight: torch.Tensor, weight_scale_2: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    work = weight.float()
    per_block_amax = reduce_block_amax(work, block_size)
    per_block_scale = per_block_amax / (
        FP4_MAX * weight_scale_2.to(per_block_amax.device)
    )
    per_block_scale[per_block_scale == 0] = 1.0
    per_block_scale_fp8 = per_block_scale.to(torch.float8_e4m3fn)

    reshaped = work.reshape(*work.shape[:-1], -1, block_size)
    scaled = reshaped / (
        (per_block_scale_fp8.to(torch.float32) * weight_scale_2.to(torch.float32))
        .unsqueeze(-1)
    )
    packed = pack_fp4(scaled.reshape(work.shape))
    return packed, per_block_scale_fp8


def load_tensors(src_dir: Path, weight_map: dict[str, str], keys: list[str]) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_shard[weight_map[key]].append(key)

    tensors = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(src_dir / shard, framework="pt", device="cpu") as f:
            for key in shard_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def load_shards(src_dir: Path, shard_names: set[str]) -> dict[str, dict[str, torch.Tensor]]:
    out = {}
    for shard in sorted(shard_names):
        tensors = {}
        with safe_open(src_dir / shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        out[shard] = tensors
    return out


def main() -> None:
    args = parse_args()
    src_bf16 = Path(args.src_bf16)
    src_nvfp4 = Path(args.src_nvfp4)
    out_dir = Path(args.out) if args.out is not None else None

    if not args.dry_run and out_dir is None:
        raise SystemExit("--out is required unless --dry-run is set")
    if out_dir is not None and out_dir.exists():
        raise SystemExit(f"Output directory already exists: {out_dir}")

    bf16_index = load_index(src_bf16 / "model.safetensors.index.json")
    nvfp4_index = load_index(src_nvfp4 / "model.safetensors.index.json")
    bf16_map = bf16_index["weight_map"]
    nvfp4_map = nvfp4_index["weight_map"]

    experts = args.experts or discover_mismatched_experts(
        src_nvfp4, rtol=args.rtol, atol=args.atol
    )
    experts = sorted(set(experts), key=_expert_sort_key)
    if not experts:
        raise SystemExit("No mismatched MiMo gate/up weight_scale_2 pairs found.")

    bf16_weight_keys = []
    affected_shards: set[str] = set()
    for expert in experts:
        for proj in ("gate_proj", "up_proj"):
            bf16_weight_keys.append(f"{expert}.{proj}.weight")
            for suffix in ("weight", "weight_scale", "weight_scale_2"):
                affected_shards.add(nvfp4_map[f"{expert}.{proj}.{suffix}"])

    if args.dry_run:
        print(f"Would patch {len(experts)} expert pair(s):")
        for expert in experts:
            gate_key = f"{expert}.gate_proj.weight_scale_2"
            up_key = f"{expert}.up_proj.weight_scale_2"
            print(
                f"  {expert}: "
                f"gate shard={nvfp4_map[gate_key]}, up shard={nvfp4_map[up_key]}"
            )
        print("Affected shards:")
        for shard in sorted(affected_shards):
            print(f"  {shard}")
        return

    bf16_weights = load_tensors(src_bf16, bf16_map, bf16_weight_keys)
    shard_tensors = load_shards(src_nvfp4, affected_shards)

    symlink_tree(
        src_nvfp4,
        out_dir,
        skip_names=affected_shards | {"model.safetensors.index.json"},
    )

    print(f"Patching {len(experts)} expert pair(s) from BF16 dequantized weights:")
    for expert in experts:
        gate_key = f"{expert}.gate_proj.weight"
        up_key = f"{expert}.up_proj.weight"
        gate_weight = bf16_weights[gate_key]
        up_weight = bf16_weights[up_key]
        scale2 = shared_weight_scale_2(gate_weight, up_weight).reshape(())
        print(f"  {expert}: shared weight_scale_2={float(scale2):.9g}")

        for proj, weight in (("gate_proj", gate_weight), ("up_proj", up_weight)):
            packed, scale = pack_nvfp4(weight, scale2, BLOCK_SIZE)
            shard = nvfp4_map[f"{expert}.{proj}.weight"]
            shard_tensors[shard][f"{expert}.{proj}.weight"] = packed
            shard_tensors[shard][f"{expert}.{proj}.weight_scale"] = scale
            shard_tensors[shard][f"{expert}.{proj}.weight_scale_2"] = scale2.clone()

    for shard, tensors in shard_tensors.items():
        save_file(tensors, out_dir / shard)

    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(nvfp4_index, f, indent=2)

    validate(out_dir, experts)
    print(f"Wrote patched checkpoint to {out_dir}")


def validate(out_dir: Path, experts: list[str]) -> None:
    index = load_index(out_dir / "model.safetensors.index.json")["weight_map"]
    for expert in experts:
        keys = [
            f"{expert}.gate_proj.weight_scale_2",
            f"{expert}.up_proj.weight_scale_2",
            f"{expert}.gate_proj.weight",
            f"{expert}.up_proj.weight",
            f"{expert}.gate_proj.weight_scale",
            f"{expert}.up_proj.weight_scale",
        ]
        tensors = {}
        for shard in {index[k] for k in keys}:
            with safe_open(out_dir / shard, framework="pt", device="cpu") as f:
                for key in keys:
                    if key in f.keys():
                        tensors[key] = f.get_tensor(key)

        gate_scale = tensors[f"{expert}.gate_proj.weight_scale_2"]
        up_scale = tensors[f"{expert}.up_proj.weight_scale_2"]
        if not torch.equal(gate_scale, up_scale):
            raise RuntimeError(f"weight_scale_2 mismatch remains for {expert}")

        gate_weight = tensors[f"{expert}.gate_proj.weight"]
        gate_block_scale = tensors[f"{expert}.gate_proj.weight_scale"]
        if gate_weight.shape[-1] * 2 != BLOCK_SIZE * gate_block_scale.shape[-1]:
            raise RuntimeError(f"Unexpected packed layout for {expert}.gate_proj")


if __name__ == "__main__":
    main()
