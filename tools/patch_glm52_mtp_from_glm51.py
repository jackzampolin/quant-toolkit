#!/usr/bin/env python3
"""Patch GLM-5.2-NVFP4 to use the quantized GLM-5.1 MTP layer."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_TARGET = Path("/models/GLM-5.2-NVFP4")
DEFAULT_MTP_SOURCE = Path("/models/GLM-5.1-NVFP4-MTP-NVFP4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace GLM-5.2 layer-78 MTP tensors with quantized GLM-5.1 MTP tensors."
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--mtp-source", type=Path, default=DEFAULT_MTP_SOURCE)
    parser.add_argument("--mtp-prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="Do not rewrite old target shards to remove stale BF16 MTP tensors.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=4, sort_keys=False)
        f.write("\n")


def is_quant_sidecar(key: str) -> bool:
    return key.endswith(".input_scale") or key.endswith(".weight_scale") or key.endswith(".weight_scale_2")


def layer_keys(weight_map: dict[str, str], prefix: str) -> set[str]:
    return {key for key in weight_map if key.startswith(prefix)}


def non_sidecar_layer_keys(weight_map: dict[str, str], prefix: str) -> set[str]:
    return {key for key in layer_keys(weight_map, prefix) if not is_quant_sidecar(key)}


def validate_mtp_keysets(
    target_weight_map: dict[str, str],
    source_weight_map: dict[str, str],
    prefix: str,
) -> None:
    target_base = non_sidecar_layer_keys(target_weight_map, prefix)
    source_base = non_sidecar_layer_keys(source_weight_map, prefix)
    if target_base != source_base:
        missing = sorted(target_base - source_base)
        extra = sorted(source_base - target_base)
        msg = [
            "MTP non-sidecar keyset mismatch.",
            f"  target={len(target_base)} source={len(source_base)}",
            f"  missing_in_source={len(missing)} extra_in_source={len(extra)}",
        ]
        if missing:
            msg.append("  first missing: " + ", ".join(missing[:8]))
        if extra:
            msg.append("  first extra: " + ", ".join(extra[:8]))
        raise RuntimeError("\n".join(msg))

    source_layer = layer_keys(source_weight_map, prefix)
    source_input_scales = [key for key in source_layer if key.endswith(".input_scale")]
    source_weight_scales = [key for key in source_layer if key.endswith(".weight_scale")]
    source_weight_scale_2 = [key for key in source_layer if key.endswith(".weight_scale_2")]
    if not source_input_scales or not source_weight_scales or not source_weight_scale_2:
        raise RuntimeError(f"{prefix} in MTP source does not look quantized")
    if not (
        len(source_input_scales) == len(source_weight_scales) == len(source_weight_scale_2)
    ):
        raise RuntimeError(
            "MTP source sidecar count mismatch: "
            f"input_scale={len(source_input_scales)} "
            f"weight_scale={len(source_weight_scales)} "
            f"weight_scale_2={len(source_weight_scale_2)}"
        )


def load_quantization_config(source_config: dict[str, Any]) -> dict[str, Any]:
    quant_config = source_config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise RuntimeError("MTP source config.json is missing quantization_config")
    return copy.deepcopy(quant_config)


def merge_target_config(
    target_config: dict[str, Any],
    source_quantization_config: dict[str, Any],
) -> dict[str, Any]:
    out_config = copy.deepcopy(target_config)
    out_config["quantization_config"] = copy.deepcopy(source_quantization_config)

    target_without_quant = copy.deepcopy(target_config)
    out_without_quant = copy.deepcopy(out_config)
    target_without_quant.pop("quantization_config", None)
    out_without_quant.pop("quantization_config", None)
    if target_without_quant != out_without_quant:
        raise RuntimeError("Config merge changed fields other than quantization_config")
    return out_config


def copy_source_mtp_shards(
    source: Path,
    target: Path,
    source_weight_map: dict[str, str],
    prefix: str,
) -> set[str]:
    shard_names = sorted({source_weight_map[key] for key in layer_keys(source_weight_map, prefix)})
    for shard_name in shard_names:
        tmp = target / (shard_name + ".tmp")
        shutil.copy2(source / shard_name, tmp, follow_symlinks=True)
        os.replace(tmp, target / shard_name)
    return set(shard_names)


def compact_stale_target_shards(
    target: Path,
    old_weight_map: dict[str, str],
    new_weight_map: dict[str, str],
    prefix: str,
    source_mtp_shards: set[str],
) -> list[str]:
    stale_shards = sorted(
        {
            old_weight_map[key]
            for key in layer_keys(old_weight_map, prefix)
            if old_weight_map[key] not in source_mtp_shards
        }
    )
    new_keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in new_weight_map.items():
        new_keys_by_shard[shard_name].append(key)

    rewritten: list[str] = []
    for shard_name in stale_shards:
        shard_path = target / shard_name
        if not shard_path.exists():
            continue
        keep_keys = sorted(new_keys_by_shard.get(shard_name, []))
        if not keep_keys:
            shard_path.unlink()
            rewritten.append(shard_name + " (removed)")
            continue

        tensors = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keep_keys:
                tensors[key] = f.get_tensor(key).contiguous()
        tmp = target / (shard_name + ".tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, shard_path)
        rewritten.append(shard_name)
    return rewritten


def write_updated_index(
    target: Path,
    old_index: dict[str, Any],
    new_weight_map: dict[str, str],
) -> None:
    metadata = dict(old_index.get("metadata", {}))
    metadata["total_size"] = sum((target / shard).stat().st_size for shard in set(new_weight_map.values()))
    index = {
        "metadata": metadata,
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    write_json(target / "model.safetensors.index.json", index)


def main() -> None:
    args = parse_args()
    target_index = read_json(args.target / "model.safetensors.index.json")
    target_config = read_json(args.target / "config.json")
    source_index = read_json(args.mtp_source / "model.safetensors.index.json")
    source_config = read_json(args.mtp_source / "config.json")

    prefix = args.mtp_prefix or f"model.layers.{target_config['num_hidden_layers']}."
    target_weight_map = target_index["weight_map"]
    source_weight_map = source_index["weight_map"]
    validate_mtp_keysets(target_weight_map, source_weight_map, prefix)

    source_layer_keys = layer_keys(source_weight_map, prefix)
    source_mtp_shards = sorted({source_weight_map[key] for key in source_layer_keys})
    quant_config = load_quantization_config(source_config)
    merge_target_config(target_config, quant_config)

    new_weight_map = {
        key: shard_name
        for key, shard_name in target_weight_map.items()
        if not key.startswith(prefix)
    }
    for key in source_layer_keys:
        new_weight_map[key] = source_weight_map[key]

    print(f"target: {args.target}")
    print(f"MTP source: {args.mtp_source}")
    print(f"MTP prefix: {prefix}")
    print(f"MTP source keys: {len(source_layer_keys)}")
    print(f"MTP source shards: {', '.join(source_mtp_shards)}")
    print(f"MTP input scales: {sum(key.endswith('.input_scale') for key in source_layer_keys)}")
    print(f"MTP weight scales: {sum(key.endswith('.weight_scale') for key in source_layer_keys)}")
    print(f"MTP weight_scale_2: {sum(key.endswith('.weight_scale_2') for key in source_layer_keys)}")
    print("config.json: target config with MTP-source quantization_config only")

    if args.dry_run:
        print("Dry run complete; no files written.")
        return

    copied_shards = copy_source_mtp_shards(args.mtp_source, args.target, source_weight_map, prefix)
    rewritten_shards: list[str] = []
    if not args.no_compact:
        rewritten_shards = compact_stale_target_shards(
            args.target,
            target_weight_map,
            new_weight_map,
            prefix,
            copied_shards,
        )

    write_json(args.target / "config.json", merge_target_config(target_config, quant_config))
    write_updated_index(args.target, target_index, new_weight_map)

    print(f"Copied MTP shard(s): {', '.join(sorted(copied_shards))}")
    if rewritten_shards:
        print(f"Compacted stale target shard(s): {', '.join(rewritten_shards)}")
    print(f"Patched quantized MTP into {args.target}")


if __name__ == "__main__":
    main()
