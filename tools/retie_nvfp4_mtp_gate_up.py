#!/usr/bin/env python3
"""Retie NVFP4 gate/up weight_scale_2 values for a quantized MTP shard.

This rewrites only the tensor shard containing ``model.layers.78`` MTP weights.
It dequantizes each quantized gate/up expert pair, requantizes the pair with a
shared NVFP4 scalar scale, and atomically replaces the shard after the full temp
file has been written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from build_glm52_nvfp4_routed_experts import quantize_nvfp4_gate_up_pair


DEFAULT_TARGET = Path("/models/GLM-5.2-NVFP4")
BLOCK_SIZE = 16
E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
GATE_UP_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj)\.weight$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retie quantized NVFP4 MTP gate/up pairs by requantizing each pair."
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--mtp-layer", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used for MTP dequantize/requantize.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=4, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def dequantize_nvfp4_weight(
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    qweight = qweight.to(device=device, non_blocking=False)
    weight_scale = weight_scale.to(device=device, non_blocking=False)
    weight_scale_2 = weight_scale_2.to(device=device, non_blocking=False)

    unpacked_shape = (*qweight.shape[:-1], qweight.shape[-1] * 2)
    if unpacked_shape[-1] != weight_scale.shape[-1] * block_size:
        raise ValueError(
            "Packed weight/scale shape mismatch: "
            f"qweight={tuple(qweight.shape)} weight_scale={tuple(weight_scale.shape)}"
        )

    unpacked = torch.empty(unpacked_shape, dtype=torch.uint8, device=device)
    unpacked[..., 0::2] = qweight & 0x0F
    unpacked[..., 1::2] = qweight >> 4

    values = E2M1_VALUES.to(device=device)[unpacked.long()]
    scale = weight_scale.to(torch.float32) * weight_scale_2.to(torch.float32)
    values = values.reshape(*weight_scale.shape, block_size)
    values = values * scale.unsqueeze(-1)
    dequantized = values.reshape(unpacked_shape).to(torch.bfloat16)

    del qweight, weight_scale, weight_scale_2, unpacked, values, scale
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dequantized


def gate_up_pair_keys(layer: int, expert: int) -> tuple[str, str]:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    return f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"


def sidecar_keys(weight_key: str) -> tuple[str, str]:
    prefix = weight_key.removesuffix(".weight")
    return prefix + ".weight_scale", prefix + ".weight_scale_2"


def mtp_gate_up_weight_keys(weight_map: dict[str, str], mtp_layer: int) -> list[str]:
    keys: list[str] = []
    for key in weight_map:
        match = GATE_UP_WEIGHT_RE.match(key)
        if match is None:
            continue
        if int(match.group("layer")) == mtp_layer:
            keys.append(key)
    return sorted(keys)


def validate_mtp_shard(weight_map: dict[str, str], mtp_layer: int) -> tuple[str, list[int]]:
    gate_up_keys = mtp_gate_up_weight_keys(weight_map, mtp_layer)
    if not gate_up_keys:
        raise RuntimeError(f"No MTP gate/up weights found for layer {mtp_layer}")

    shards = {weight_map[key] for key in gate_up_keys}
    for key in gate_up_keys:
        weight_scale, weight_scale_2 = sidecar_keys(key)
        for sidecar in (weight_scale, weight_scale_2):
            if sidecar not in weight_map:
                raise RuntimeError(f"Missing sidecar for {key}: {sidecar}")
            shards.add(weight_map[sidecar])
    if len(shards) != 1:
        raise RuntimeError(f"MTP gate/up tensors span multiple shards: {sorted(shards)}")

    experts = sorted(
        {
            int(match.group("expert"))
            for key in gate_up_keys
            if (match := GATE_UP_WEIGHT_RE.match(key)) is not None
        }
    )
    for expert in experts:
        gate_key, up_key = gate_up_pair_keys(mtp_layer, expert)
        if gate_key not in weight_map or up_key not in weight_map:
            raise RuntimeError(f"Missing complete gate/up pair for layer {mtp_layer} expert {expert}")
    return next(iter(shards)), experts


def count_gate_up_mismatches(
    shard_path: Path,
    mtp_layer: int,
    experts: list[int],
) -> tuple[int, list[tuple[int, float, float]]]:
    mismatches: list[tuple[int, float, float]] = []
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        for expert in experts:
            gate_key, up_key = gate_up_pair_keys(mtp_layer, expert)
            gate_scale_2 = f.get_tensor(sidecar_keys(gate_key)[1])
            up_scale_2 = f.get_tensor(sidecar_keys(up_key)[1])
            if not torch.equal(gate_scale_2, up_scale_2):
                if len(mismatches) < 8:
                    mismatches.append((expert, float(gate_scale_2), float(up_scale_2)))
                else:
                    mismatches.append((expert, 0.0, 0.0))
    return len(mismatches), mismatches[:8]


def add_quantized_weight(
    tensors: dict[str, torch.Tensor],
    key: str,
    quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    weight_scale_key, weight_scale_2_key = sidecar_keys(key)
    qweight, weight_scale, weight_scale_2 = quantized
    tensors[key] = qweight.contiguous()
    tensors[weight_scale_key] = weight_scale.contiguous()
    tensors[weight_scale_2_key] = weight_scale_2.reshape(()).contiguous()


def rewrite_mtp_shard(
    target: Path,
    shard_name: str,
    mtp_layer: int,
    experts: list[int],
    block_size: int,
    device: torch.device,
) -> None:
    shard_path = target / shard_name
    tmp_path = target / f"{shard_name}.tmp"
    if tmp_path.exists():
        tmp_path.unlink()

    rewritten_keys: set[str] = set()
    tensors: dict[str, torch.Tensor] = {}

    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        shard_keys = sorted(f.keys())
        for key in shard_keys:
            if key in rewritten_keys:
                continue

            match = GATE_UP_WEIGHT_RE.match(key)
            if match is None or int(match.group("layer")) != mtp_layer:
                tensors[key] = f.get_tensor(key).contiguous()
                continue

            expert = int(match.group("expert"))
            gate_key, up_key = gate_up_pair_keys(mtp_layer, expert)
            if key != gate_key:
                continue

            gate_scale_key, gate_scale_2_key = sidecar_keys(gate_key)
            up_scale_key, up_scale_2_key = sidecar_keys(up_key)
            gate_weight = dequantize_nvfp4_weight(
                f.get_tensor(gate_key),
                f.get_tensor(gate_scale_key),
                f.get_tensor(gate_scale_2_key),
                block_size,
                device,
            )
            up_weight = dequantize_nvfp4_weight(
                f.get_tensor(up_key),
                f.get_tensor(up_scale_key),
                f.get_tensor(up_scale_2_key),
                block_size,
                device,
            )
            gate_quantized, up_quantized = quantize_nvfp4_gate_up_pair(
                gate_weight,
                up_weight,
                block_size,
                device,
            )
            add_quantized_weight(tensors, gate_key, gate_quantized)
            add_quantized_weight(tensors, up_key, up_quantized)
            rewritten_keys.update(
                {
                    gate_key,
                    gate_scale_key,
                    gate_scale_2_key,
                    up_key,
                    up_scale_key,
                    up_scale_2_key,
                }
            )
            del gate_weight, up_weight, gate_quantized, up_quantized
            if device.type == "cuda":
                torch.cuda.empty_cache()

    save_file(tensors, str(tmp_path))
    os.replace(tmp_path, shard_path)


def refresh_index_total_size(target: Path) -> None:
    index_path = target / "model.safetensors.index.json"
    index = read_json(index_path)
    weight_map = index["weight_map"]
    metadata = dict(index.get("metadata", {}))
    metadata["total_size"] = sum((target / shard).stat().st_size for shard in set(weight_map.values()))
    index["metadata"] = metadata
    write_json(index_path, index)


def main() -> None:
    args = parse_args()
    index = read_json(args.target / "model.safetensors.index.json")
    config = read_json(args.target / "config.json")
    mtp_layer = args.mtp_layer
    if mtp_layer is None:
        mtp_layer = int(config["num_hidden_layers"])

    weight_map = index["weight_map"]
    shard_name, experts = validate_mtp_shard(weight_map, mtp_layer)
    shard_path = args.target / shard_name
    before_count, before_sample = count_gate_up_mismatches(shard_path, mtp_layer, experts)

    print(f"target: {args.target}")
    print(f"MTP layer: {mtp_layer}")
    print(f"MTP shard: {shard_name}")
    print(f"gate/up pairs: {len(experts)}")
    print(f"pre-retie gate/up scale2 mismatches: {before_count}")
    if before_sample:
        print(f"first mismatches: {before_sample}")

    if args.dry_run:
        print("Dry run complete; no files written.")
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    rewrite_mtp_shard(
        args.target,
        shard_name,
        mtp_layer,
        experts,
        args.block_size,
        device,
    )
    refresh_index_total_size(args.target)

    after_count, after_sample = count_gate_up_mismatches(shard_path, mtp_layer, experts)
    print(f"post-retie gate/up scale2 mismatches: {after_count}")
    if after_sample:
        print(f"first remaining mismatches: {after_sample}")
    if after_count != 0:
        raise RuntimeError("MTP gate/up scale2 retie failed")


if __name__ == "__main__":
    main()
