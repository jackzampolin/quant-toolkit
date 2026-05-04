#!/usr/bin/env python3
"""Build a GLM NVFP4 checkpoint with layer-78 MTP weights quantized.

The existing GLM exports keep the next-token prediction/MTP block as raw BF16
under ``model.layers.78.*``. This tool writes a side-by-side checkpoint that
keeps the base shards as symlinks and overrides that block with:

  - model-mtp.safetensors
  - model-mtp-inputscales.safetensors

Activation scales are copied from the matching projection in the last normal
decoder layer, e.g. ``model.layers.77...input_scale`` becomes
``model.layers.78...input_scale``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt.torch.export.quant_utils import NVFP4QTensor


MTP_SHARD_NAME = "model-mtp.safetensors"
MTP_SCALE_SHARD_NAME = "model-mtp-inputscales.safetensors"
BASE_SKIP_NAMES = {
    "__pycache__",
    "config.json",
    "model.safetensors.index.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mtp-prefix", default="model.layers.78.")
    parser.add_argument("--scale-source-prefix", default="model.layers.77.")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


def tensor_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def copy_small_files_and_link_shards(src_dir: Path, dst_dir: Path, shard_names: set[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=False)
    for item in src_dir.iterdir():
        if item.name in BASE_SKIP_NAMES:
            continue
        if item.name.endswith(".safetensors"):
            if item.name in shard_names:
                os.symlink(item.resolve(), dst_dir / item.name)
            continue
        if item.is_dir():
            continue
        shutil.copy2(item.resolve() if item.is_symlink() else item, dst_dir / item.name)


def load_input_scales(base_dir: Path, source_prefix: str, mtp_prefix: str) -> dict[str, torch.Tensor]:
    scales_path = base_dir / "model-inputscales.safetensors"
    scales: dict[str, torch.Tensor] = {}
    with safe_open(str(scales_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(source_prefix):
                scales[mtp_prefix + key[len(source_prefix) :]] = f.get_tensor(key).contiguous()
    return scales


def quantize_nvfp4_weight(
    weight: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    work = weight.to(device=device, dtype=torch.bfloat16, non_blocking=False)
    qweight, weight_scale, weight_scale_2 = NVFP4QTensor.quantize(work, block_size=block_size)
    result = (
        qweight._quantized_data.detach().cpu().contiguous(),
        weight_scale.detach().cpu().contiguous(),
        weight_scale_2.detach().cpu().reshape(()).contiguous(),
    )
    del work, qweight, weight_scale, weight_scale_2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_and_quantize_mtp_tensors(
    base_dir: Path,
    base_index: dict,
    mtp_prefix: str,
    scale_source_prefix: str,
    block_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
    weight_map = base_index["weight_map"]
    mtp_raw_keys = sorted(key for key in weight_map if key.startswith(mtp_prefix))
    if not mtp_raw_keys:
        raise RuntimeError(f"No MTP keys found with prefix {mtp_prefix!r}")

    copied_scales = load_input_scales(base_dir, scale_source_prefix, mtp_prefix)
    quantizable_weights = {
        key
        for key in mtp_raw_keys
        if key.endswith(".weight") and key[:-len(".weight")] + ".input_scale" in copied_scales
    }

    mtp_tensors: dict[str, torch.Tensor] = {}
    mtp_scale_tensors: dict[str, torch.Tensor] = {}
    quantized_prefixes: list[str] = []

    shards = sorted(set(weight_map[key] for key in mtp_raw_keys))
    done = 0
    for shard_name in shards:
        with safe_open(str(base_dir / shard_name), framework="pt", device="cpu") as f:
            shard_keys = [key for key in mtp_raw_keys if weight_map[key] == shard_name]
            for key in shard_keys:
                tensor = f.get_tensor(key).contiguous()
                if key in quantizable_weights:
                    prefix = key[:-len(".weight")]
                    qweight, weight_scale, weight_scale_2 = quantize_nvfp4_weight(
                        tensor,
                        block_size=block_size,
                        device=device,
                    )
                    mtp_tensors[key] = qweight
                    mtp_tensors[prefix + ".weight_scale"] = weight_scale
                    mtp_tensors[prefix + ".weight_scale_2"] = weight_scale_2
                    mtp_scale_tensors[prefix + ".input_scale"] = copied_scales[
                        prefix + ".input_scale"
                    ].clone()
                    quantized_prefixes.append(prefix)
                    done += 1
                    if done == 1 or done % 25 == 0:
                        print(f"  quantized {done}/{len(quantizable_weights)}: {key}", flush=True)
                else:
                    mtp_tensors[key] = tensor

    return mtp_tensors, mtp_scale_tensors, sorted(quantized_prefixes)


def update_config_for_mtp(config: dict) -> None:
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        return

    ignore = quant_config.setdefault("ignore", [])
    if not isinstance(ignore, list):
        return

    additions = [
        "model.layers.78.eh_proj*",
        "model.layers.78.self_attn*",
        "model.layers.78.self_attn.indexer*",
        "model.layers.78.mlp.gate*",
        "model.layers.78.mlp.shared_experts*",
    ]
    seen = set(ignore)
    for pattern in additions:
        if pattern not in seen:
            ignore.append(pattern)
            seen.add(pattern)


def write_updated_index(
    out_dir: Path,
    base_index: dict,
    mtp_prefix: str,
    mtp_tensors: dict[str, torch.Tensor],
    mtp_scale_tensors: dict[str, torch.Tensor],
) -> None:
    weight_map = {
        key: shard
        for key, shard in base_index["weight_map"].items()
        if not key.startswith(mtp_prefix)
    }
    for key in mtp_tensors:
        weight_map[key] = MTP_SHARD_NAME
    for key in mtp_scale_tensors:
        weight_map[key] = MTP_SCALE_SHARD_NAME

    index = {
        "metadata": {
            "total_size": sum((out_dir / shard).stat().st_size for shard in set(weight_map.values()))
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    write_json(out_dir / "model.safetensors.index.json", index)


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Output directory already exists: {args.out}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    base_index = read_json(args.base_model / "model.safetensors.index.json")
    base_config = read_json(args.base_model / "config.json")

    non_mtp_shards = {
        shard_name
        for key, shard_name in base_index["weight_map"].items()
        if not key.startswith(args.mtp_prefix)
    }

    mtp_tensors, mtp_scale_tensors, quantized_prefixes = load_and_quantize_mtp_tensors(
        args.base_model,
        base_index,
        args.mtp_prefix,
        args.scale_source_prefix,
        args.block_size,
        device,
    )

    copy_small_files_and_link_shards(args.base_model, args.out, non_mtp_shards)
    save_file(mtp_tensors, str(args.out / MTP_SHARD_NAME))
    save_file(mtp_scale_tensors, str(args.out / MTP_SCALE_SHARD_NAME))

    update_config_for_mtp(base_config)
    write_json(args.out / "config.json", base_config)
    write_updated_index(args.out, base_index, args.mtp_prefix, mtp_tensors, mtp_scale_tensors)

    mtp_size = sum(tensor_size(t) for t in mtp_tensors.values())
    scale_size = sum(tensor_size(t) for t in mtp_scale_tensors.values())
    print(f"Wrote merged checkpoint to {args.out}")
    print(f"  base shards kept as symlinks: {len(non_mtp_shards)}")
    print(f"  MTP tensors: {len(mtp_tensors)} ({mtp_size / 1e9:.2f} GB)")
    print(f"  MTP input scales: {len(mtp_scale_tensors)} ({scale_size / 1e6:.2f} MB)")
    print(f"  quantized MTP modules: {len(quantized_prefixes)}")


if __name__ == "__main__":
    main()
