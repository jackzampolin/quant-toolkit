#!/usr/bin/env python3
"""Patch MiMo dense MLP tensors from original FP8 into a hybrid MXFP8 checkpoint.

This is for the MiMo-V2.5 NVFP4 + MXFP8-attention chimera.  The w13-tied base
contains BF16 dense MLP tensors for layer 0 and MTP.  The original HF checkpoint
serialized those tensors as FP8 E4M3 with 128x128 FP8 block scales.  This tool
converts those original FP8 tensors through BF16 into ModelOpt MXFP8 and writes
the MXFP8 weights and uint8 scale sidecars into the target checkpoint.
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from deinterleave_mimo_fp8_qkv import (
    dequantize_block_fp8_to_float,
    quantize_mxfp8_from_bf16,
)


DEFAULT_SOURCE = Path("/data/models/MiMo-V2.5-FP8-source")
DEFAULT_TARGET = Path("/data/models/MiMo-V2.5-NVFP4-w13-tied-MXFP8-attn")
INDEX_NAME = "model.safetensors.index.json"
FP8_BLOCK_SIZE = (128, 128)
MXFP8_GROUP_SIZE = 32
MXFP8_WEIGHT_BLOCK_SIZE = [1, 32]
DENSE_MLP_WEIGHT_RE = re.compile(
    r"^model\.(?:layers\.\d+|mtp\.layers\.\d+)\.mlp\."
    r"(?:gate|up|down)_proj\.weight$"
)
PACKED_MODULES_MAPPING = {"gate_up_proj": ["gate_proj", "up_proj"]}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        path.unlink()
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=path.name == INDEX_NAME)
        f.write("\n")
    tmp_path.replace(path)


def scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected a .weight tensor name, got {weight_name}")
    return weight_name.removesuffix(".weight") + ".weight_scale_inv"


def module_prefix(weight_name: str) -> str:
    return weight_name.removesuffix(".weight")


def sort_key(name: str) -> tuple[int, int, str]:
    mtp_match = re.match(r"^model\.mtp\.layers\.(\d+)\.", name)
    if mtp_match:
        return (1, int(mtp_match.group(1)), name)
    layer_match = re.match(r"^model\.layers\.(\d+)\.", name)
    if layer_match:
        return (0, int(layer_match.group(1)), name)
    return (2, 0, name)


def build_direct_weight_map(model_dir: Path) -> dict[str, str]:
    weight_map: dict[str, str] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_map:
                    raise RuntimeError(f"Duplicate tensor key {key} in {model_dir}")
                weight_map[key] = shard.name
    return weight_map


def selected_source_weights(source_weight_map: dict[str, str]) -> list[str]:
    weights = sorted(
        (key for key in source_weight_map if DENSE_MLP_WEIGHT_RE.match(key)),
        key=sort_key,
    )
    missing_scales = [scale_name(key) for key in weights if scale_name(key) not in source_weight_map]
    if missing_scales:
        raise RuntimeError(
            f"Source is missing {len(missing_scales)} FP8 scale tensor(s); first: {missing_scales[0]}"
        )
    return weights


def tensor_meta(model_dir: Path, weight_map: dict[str, str], key: str) -> tuple[list[int], str]:
    with safe_open(model_dir / weight_map[key], framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice(key)
        return tensor_slice.get_shape(), tensor_slice.get_dtype()


def validate_source_and_target(
    source_dir: Path,
    target_dir: Path,
    source_weight_map: dict[str, str],
    target_weight_map: dict[str, str],
    weights: list[str],
) -> None:
    missing_targets = [key for key in weights if key not in target_weight_map]
    if missing_targets:
        raise RuntimeError(
            f"Target is missing {len(missing_targets)} dense MLP tensor(s); first: {missing_targets[0]}"
        )

    for key in weights:
        src_shape, src_dtype = tensor_meta(source_dir, source_weight_map, key)
        scale_shape, scale_dtype = tensor_meta(source_dir, source_weight_map, scale_name(key))
        tgt_shape, _ = tensor_meta(target_dir, target_weight_map, key)
        if src_dtype != "F8_E4M3":
            raise RuntimeError(f"Expected original FP8 E4M3 for {key}, got {src_dtype}")
        if scale_dtype != "F32":
            raise RuntimeError(f"Expected original F32 FP8 scale for {scale_name(key)}, got {scale_dtype}")
        if src_shape != tgt_shape:
            raise RuntimeError(f"Shape mismatch for {key}: source {src_shape}, target {tgt_shape}")

        expected_scale_shape = [
            (src_shape[0] + FP8_BLOCK_SIZE[0] - 1) // FP8_BLOCK_SIZE[0],
            (src_shape[1] + FP8_BLOCK_SIZE[1] - 1) // FP8_BLOCK_SIZE[1],
        ]
        if scale_shape != expected_scale_shape:
            raise RuntimeError(
                f"Expected source FP8 scale shape {expected_scale_shape} for {scale_name(key)}, got {scale_shape}"
            )


def load_source_pair(
    source_dir: Path,
    source_weight_map: dict[str, str],
    key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_key = scale_name(key)
    weight_file = source_weight_map[key]
    scale_file = source_weight_map[scale_key]
    with safe_open(source_dir / weight_file, framework="pt", device="cpu") as f:
        weight = f.get_tensor(key).contiguous()
        if scale_file == weight_file:
            scale = f.get_tensor(scale_key).contiguous()
        else:
            with safe_open(source_dir / scale_file, framework="pt", device="cpu") as sf:
                scale = sf.get_tensor(scale_key).contiguous()
    return weight, scale


def convert_weight_to_mxfp8(
    source_dir: Path,
    source_weight_map: dict[str, str],
    key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight, scale = load_source_pair(source_dir, source_weight_map, key)
    dequantized = dequantize_block_fp8_to_float(weight, scale, FP8_BLOCK_SIZE)
    qweight, qscale = quantize_mxfp8_from_bf16(dequantized.to(torch.bfloat16))
    del weight, scale, dequantized
    gc.collect()
    return qweight.contiguous(), qscale.contiguous()


def convert_replacements_by_shard(
    source_dir: Path,
    target_weight_map: dict[str, str],
    source_weight_map: dict[str, str],
    weights: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    replacements_by_shard: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for idx, key in enumerate(weights, start=1):
        shard_name = target_weight_map[key]
        print(f"[{idx}/{len(weights)}] converting {key} -> {shard_name}", flush=True)
        qweight, qscale = convert_weight_to_mxfp8(source_dir, source_weight_map, key)
        replacements_by_shard[shard_name][key] = qweight
        replacements_by_shard[shard_name][scale_name(key)] = qscale
        print(
            f"    MXFP8 weight={tuple(qweight.shape)} {qweight.dtype}, "
            f"scale={tuple(qscale.shape)} {qscale.dtype}",
            flush=True,
        )
    return dict(replacements_by_shard)


def rewrite_target_shard(
    target_dir: Path,
    shard_name: str,
    replacements: dict[str, torch.Tensor],
) -> None:
    shard_path = target_dir / shard_name
    tmp_path = target_dir / f".{shard_name}.tmp"
    if tmp_path.exists():
        tmp_path.unlink()

    replacement_weights = {key for key in replacements if key.endswith(".weight")}
    replacement_scales = {scale_name(key) for key in replacement_weights}
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            if key in replacement_scales:
                continue
            if key in replacements:
                tensors[key] = replacements[key]
            else:
                tensors[key] = f.get_tensor(key).contiguous()

    for key in sorted(replacement_scales):
        tensors[key] = replacements[key]

    save_kwargs = {"metadata": metadata} if metadata else {}
    save_file(tensors, str(tmp_path), **save_kwargs)
    tmp_path.replace(shard_path)
    print(f"rewrote {shard_name}: {len(replacement_weights)} dense MLP weight(s)", flush=True)


def mxfp8_group(targets: list[str]) -> dict[str, Any]:
    return {
        "input_activations": {
            "dynamic": True,
            "num_bits": 8,
            "type": "float",
            "group_size": MXFP8_GROUP_SIZE,
        },
        "weights": {
            "dynamic": False,
            "num_bits": 8,
            "type": "float",
            "group_size": MXFP8_GROUP_SIZE,
            "weight_block_size": MXFP8_WEIGHT_BLOCK_SIZE,
        },
        "targets": targets,
    }


def merged_packed_modules_mapping(existing: Any) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    if isinstance(existing, dict):
        for key, value in existing.items():
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                merged[key] = value
    merged.update(PACKED_MODULES_MAPPING)
    return merged


def blocked_by_ignore(ignore: Any, prefixes: list[str]) -> tuple[str, str] | None:
    if not isinstance(ignore, list):
        return None
    for prefix in prefixes:
        for pattern in ignore:
            if isinstance(pattern, str) and fnmatch.fnmatchcase(prefix, pattern):
                return prefix, pattern
    return None


def update_config(config_path: Path, weights: list[str]) -> None:
    config = read_json(config_path)
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        raise RuntimeError(f"{config_path} is missing quantization_config")
    if quant.get("quant_algo") != "MIXED_PRECISION" or quant.get("quant_method") != "modelopt":
        raise RuntimeError("Expected a ModelOpt MIXED_PRECISION quantization_config")

    dense_prefixes = sorted(module_prefix(key) for key in weights)
    fused_prefixes = sorted(
        {
            f"{prefix.rsplit('.', 1)[0]}.gate_up_proj"
            for prefix in dense_prefixes
            if prefix.endswith((".gate_proj", ".up_proj"))
        }
    )
    blocker = blocked_by_ignore(quant.get("ignore"), sorted(set(dense_prefixes) | set(fused_prefixes)))
    if blocker is not None:
        prefix, pattern = blocker
        raise RuntimeError(f"Ignore pattern {pattern!r} masks newly quantized layer {prefix!r}")

    quantized_layers = quant.setdefault("quantized_layers", {})
    if not isinstance(quantized_layers, dict):
        raise RuntimeError("quantization_config.quantized_layers must be a dict")
    for prefix in dense_prefixes:
        quantized_layers[prefix] = {"quant_algo": "MXFP8", "group_size": MXFP8_GROUP_SIZE}

    config_groups = quant.setdefault("config_groups", {})
    if not isinstance(config_groups, dict):
        raise RuntimeError("quantization_config.config_groups must be a dict")
    config_groups["group_mxfp8_dense_mlp"] = mxfp8_group(dense_prefixes)
    quant["packed_modules_mapping"] = merged_packed_modules_mapping(quant.get("packed_modules_mapping"))
    quant["quantized_layers"] = dict(sorted(quantized_layers.items()))
    config["quantization_config"] = quant
    write_json(config_path, config)


def update_hf_quant_config(hf_quant_path: Path, weights: list[str], config_quant: dict[str, Any]) -> None:
    if not hf_quant_path.exists():
        return
    hf_quant = read_json(hf_quant_path)
    quantization = hf_quant.setdefault("quantization", {})
    if not isinstance(quantization, dict):
        raise RuntimeError("hf_quant_config.json quantization section must be a dict")
    quantized_layers = quantization.setdefault("quantized_layers", {})
    if not isinstance(quantized_layers, dict):
        raise RuntimeError("hf_quant_config.json quantized_layers must be a dict")
    for key in weights:
        quantized_layers[module_prefix(key)] = {
            "quant_algo": "MXFP8",
            "group_size": MXFP8_GROUP_SIZE,
        }
    quantization["quant_algo"] = "MIXED_PRECISION"
    quantization["quantized_layers"] = dict(sorted(quantized_layers.items()))
    quantization["packed_modules_mapping"] = config_quant.get("packed_modules_mapping")
    hf_quant["packed_modules_mapping"] = config_quant.get("packed_modules_mapping")
    hf_quant["producer"] = config_quant.get("producer", hf_quant.get("producer", {}))
    write_json(hf_quant_path, hf_quant)


def update_index(
    index_path: Path,
    weights: list[str],
    target_weight_map: dict[str, str],
    source_dir: Path,
) -> None:
    index = read_json(index_path)
    weight_map = dict(index["weight_map"])
    for key in weights:
        weight_map[key] = target_weight_map[key]
        weight_map[scale_name(key)] = target_weight_map[key]

    metadata = index.setdefault("metadata", {})
    target_dir = index_path.parent
    metadata["total_size"] = sum(
        (target_dir / shard).stat().st_size for shard in set(weight_map.values())
    )
    metadata["dense_mlp_mxfp8_source"] = str(source_dir)
    metadata["dense_mlp_mxfp8_tensors"] = len(weights)
    index["weight_map"] = dict(sorted(weight_map.items()))
    write_json(index_path, index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fp8", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_fp8
    target_dir = args.target
    index_path = target_dir / INDEX_NAME
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {source_dir}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing target safetensors index: {index_path}")

    source_weight_map = build_direct_weight_map(source_dir)
    target_index = read_json(index_path)
    target_weight_map = dict(target_index["weight_map"])
    weights = selected_source_weights(source_weight_map)
    validate_source_and_target(source_dir, target_dir, source_weight_map, target_weight_map, weights)

    affected_shards = sorted({target_weight_map[key] for key in weights})
    print(f"Source FP8 checkpoint: {source_dir}")
    print(f"Target hybrid checkpoint: {target_dir}")
    print(f"Dense MLP FP8 tensors selected: {len(weights)}")
    print(f"Affected target shards: {', '.join(affected_shards)}")
    if args.dry_run:
        for key in weights:
            print(f"  {key} -> {target_weight_map[key]}")
        print("Dry run; no files written.")
        return

    replacements_by_shard = convert_replacements_by_shard(
        source_dir=source_dir,
        target_weight_map=target_weight_map,
        source_weight_map=source_weight_map,
        weights=weights,
    )
    for shard_name in affected_shards:
        rewrite_target_shard(target_dir, shard_name, replacements_by_shard[shard_name])
        del replacements_by_shard[shard_name]
        gc.collect()

    update_config(target_dir / "config.json", weights)
    config_quant = read_json(target_dir / "config.json")["quantization_config"]
    update_hf_quant_config(target_dir / "hf_quant_config.json", weights, config_quant)
    update_index(index_path, weights, target_weight_map, source_dir)
    print("updated config.json, hf_quant_config.json, and model.safetensors.index.json")


if __name__ == "__main__":
    main()
