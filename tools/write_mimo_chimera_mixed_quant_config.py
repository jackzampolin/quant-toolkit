#!/usr/bin/env python3
"""Write mixed quantization metadata for the MiMo NVFP4 + FP8/MXFP8-attn chimera."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open


DEFAULT_MODEL_DIR = Path("/data/models/MiMo-V2.5-NVFP4-w13-tied-FP8-attn")
INDEX_NAME = "model.safetensors.index.json"
LANGUAGE_QKV_RE = re.compile(r"^model\.layers\.\d+\.self_attn\.qkv_proj\.weight$")
MTP_QKV_RE = re.compile(r"^model\.mtp\.layers\.\d+\.self_attn\.qkv_proj\.weight$")
NVFP4_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
FP8_WEIGHT_BLOCK_SIZE = [128, 128]
MXFP8_WEIGHT_BLOCK_SIZE = [1, 32]
MXFP8_GROUP_SIZE = 32


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        path.unlink()
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / INDEX_NAME
    if not index_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    index = read_json(index_path)
    return dict(index["weight_map"])


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def tensor_meta(model_dir: Path, weight_map: dict[str, str], key: str) -> tuple[list[int], str]:
    with safe_open(model_dir / weight_map[key], framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice(key)
        return tensor_slice.get_shape(), tensor_slice.get_dtype()


def infer_qkv_quant_format(weight_shape: list[int], scale_shape: list[int], scale_dtype: str) -> str:
    if scale_dtype == "F32":
        expected = [
            ceil_div(weight_shape[0], FP8_WEIGHT_BLOCK_SIZE[0]),
            ceil_div(weight_shape[1], FP8_WEIGHT_BLOCK_SIZE[1]),
        ]
        if scale_shape != expected:
            raise RuntimeError(f"Expected FP8_PB_WO scale shape {expected}, got {scale_shape}")
        return "fp8-pb"
    if scale_dtype == "U8":
        expected = [weight_shape[0], ceil_div(weight_shape[1], MXFP8_GROUP_SIZE)]
        if scale_shape != expected:
            raise RuntimeError(f"Expected MXFP8 scale shape {expected}, got {scale_shape}")
        return "mxfp8"
    raise RuntimeError(f"Unsupported QKV scale dtype {scale_dtype}; expected F32 or U8")


def detect_qkv_prefixes_by_format(model_dir: Path, weight_map: dict[str, str]) -> dict[str, list[str]]:
    prefixes_by_format: dict[str, list[str]] = {"fp8-pb": [], "mxfp8": []}
    for key in weight_map:
        if not (LANGUAGE_QKV_RE.match(key) or MTP_QKV_RE.match(key)):
            continue
        prefix = key.removesuffix(".weight")
        scale_name = f"{prefix}.weight_scale_inv"
        if scale_name not in weight_map:
            raise RuntimeError(f"Missing FP8 block scale sidecar for {key}: {scale_name}")
        weight_shape, weight_dtype = tensor_meta(model_dir, weight_map, key)
        scale_shape, scale_dtype = tensor_meta(model_dir, weight_map, scale_name)
        if weight_dtype != "F8_E4M3":
            raise RuntimeError(f"Expected FP8 E4M3 QKV weight for {key}, got {weight_dtype}")
        quant_format = infer_qkv_quant_format(weight_shape, scale_shape, scale_dtype)
        prefixes_by_format[quant_format].append(prefix)
    return {key: sorted(value) for key, value in prefixes_by_format.items() if value}


def detect_nvfp4_prefixes(weight_map: dict[str, str]) -> list[str]:
    scale_prefixes = set()
    for key in weight_map:
        for suffix in NVFP4_SUFFIXES:
            if key.endswith(suffix):
                scale_prefixes.add(key[: -len(suffix)])

    prefixes = []
    for prefix in sorted(scale_prefixes):
        missing = [
            f"{prefix}{suffix}"
            for suffix in (".weight", *NVFP4_SUFFIXES)
            if f"{prefix}{suffix}" not in weight_map
        ]
        if missing:
            raise RuntimeError(
                f"Incomplete NVFP4 tensor set for {prefix}; first missing tensor: {missing[0]}"
            )
        prefixes.append(prefix)
    return prefixes


def make_quantized_layers(
    nvfp4_prefixes: list[str],
    qkv_prefixes_by_format: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    all_qkv_prefixes = sorted({prefix for prefixes in qkv_prefixes_by_format.values() for prefix in prefixes})
    overlap = set(nvfp4_prefixes) & set(all_qkv_prefixes)
    if overlap:
        raise RuntimeError(f"Prefixes cannot be both NVFP4 and FP8; first: {sorted(overlap)[0]}")

    quantized_layers: dict[str, dict[str, Any]] = {}
    for prefix in nvfp4_prefixes:
        quantized_layers[prefix] = {"quant_algo": "NVFP4", "group_size": 16}
    for prefix in qkv_prefixes_by_format.get("fp8-pb", []):
        quantized_layers[prefix] = {
            "quant_algo": "FP8_PB_WO",
            "weight_block_size": FP8_WEIGHT_BLOCK_SIZE,
        }
    for prefix in qkv_prefixes_by_format.get("mxfp8", []):
        quantized_layers[prefix] = {
            "quant_algo": "MXFP8",
            "group_size": MXFP8_GROUP_SIZE,
        }
    return dict(sorted(quantized_layers.items()))


def fp8_group(prefixes: list[str]) -> dict[str, Any]:
    return {
        "input_activations": {
            "dynamic": True,
            "num_bits": 8,
            "type": "float",
        },
        "weights": {
            "dynamic": False,
            "num_bits": 8,
            "type": "float",
            "weight_block_size": FP8_WEIGHT_BLOCK_SIZE,
        },
        "targets": prefixes,
    }


def mxfp8_group(prefixes: list[str]) -> dict[str, Any]:
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
        "targets": prefixes,
    }


def nvfp4_group(prefixes: list[str]) -> dict[str, Any]:
    return {
        "input_activations": {
            "dynamic": False,
            "num_bits": 4,
            "type": "float",
            "group_size": 16,
        },
        "weights": {
            "dynamic": False,
            "num_bits": 4,
            "type": "float",
            "group_size": 16,
        },
        "targets": prefixes,
    }


def should_drop_ignore(pattern: object) -> bool:
    if not isinstance(pattern, str):
        return False
    if pattern == "model.mtp*":
        return True
    if pattern == "model.layers.0*":
        return True
    return bool(re.fullmatch(r"model\.layers\.\d+\.self_attn\*", pattern))


def filter_ignore_patterns(ignore: Any, quantized_prefixes: list[str]) -> list[str]:
    if not isinstance(ignore, list):
        return []

    kept = [pattern for pattern in ignore if not should_drop_ignore(pattern)]
    blockers = []
    for prefix in quantized_prefixes:
        for pattern in kept:
            if isinstance(pattern, str) and fnmatch.fnmatchcase(prefix, pattern):
                blockers.append((prefix, pattern))
                break
    if blockers:
        prefix, pattern = blockers[0]
        raise RuntimeError(f"Ignore pattern {pattern!r} still masks quantized layer {prefix!r}")
    return kept


def update_config_json(
    config: dict[str, Any],
    quantized_layers: dict[str, dict[str, Any]],
    nvfp4_prefixes: list[str],
    qkv_prefixes_by_format: dict[str, list[str]],
) -> dict[str, Any]:
    old_quant = config.get("quantization_config")
    if not isinstance(old_quant, dict):
        raise RuntimeError("config.json is missing quantization_config")

    quantized_prefixes = sorted(quantized_layers)
    config_groups = {"group_nvfp4": nvfp4_group(nvfp4_prefixes)}
    if qkv_prefixes_by_format.get("fp8-pb"):
        config_groups["group_fp8_qkv"] = fp8_group(qkv_prefixes_by_format["fp8-pb"])
    if qkv_prefixes_by_format.get("mxfp8"):
        config_groups["group_mxfp8_qkv"] = mxfp8_group(qkv_prefixes_by_format["mxfp8"])
    config["quantization_config"] = {
        "config_groups": config_groups,
        "quantized_layers": quantized_layers,
        "ignore": filter_ignore_patterns(old_quant.get("ignore", []), quantized_prefixes),
        "quant_algo": "MIXED_PRECISION",
        "producer": old_quant.get("producer", {}),
        "quant_method": "modelopt",
    }
    return config


def make_hf_quant_config(
    base_quant_config: dict[str, Any],
    quantized_layers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    producer = base_quant_config.get("producer", {})
    ignore = base_quant_config.get("ignore", [])
    return {
        "producer": producer,
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "exclude_modules": ignore,
            "quantized_layers": quantized_layers,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    weight_map = load_weight_map(model_dir)
    qkv_prefixes_by_format = detect_qkv_prefixes_by_format(model_dir, weight_map)
    nvfp4_prefixes = detect_nvfp4_prefixes(weight_map)
    quantized_layers = make_quantized_layers(nvfp4_prefixes, qkv_prefixes_by_format)

    config_path = model_dir / "config.json"
    config = read_json(config_path)
    config = update_config_json(config, quantized_layers, nvfp4_prefixes, qkv_prefixes_by_format)
    hf_quant_config = make_hf_quant_config(config["quantization_config"], quantized_layers)

    print(f"Model: {model_dir}")
    print(f"NVFP4 quantized modules: {len(nvfp4_prefixes)}")
    print(f"FP8_PB_WO QKV modules: {len(qkv_prefixes_by_format.get('fp8-pb', []))}")
    print(f"MXFP8 QKV modules: {len(qkv_prefixes_by_format.get('mxfp8', []))}")
    print(f"Total quantized_layers entries: {len(quantized_layers)}")
    print(f"Remaining ignore patterns: {len(config['quantization_config']['ignore'])}")

    if args.dry_run:
        print("Dry run; no files written.")
        return

    write_json(config_path, config)
    write_json(model_dir / "hf_quant_config.json", hf_quant_config)
    print("Wrote config.json and hf_quant_config.json")


if __name__ == "__main__":
    main()
