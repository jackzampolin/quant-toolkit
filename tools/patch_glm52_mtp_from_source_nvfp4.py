#!/usr/bin/env python3
"""Patch a GLM-5.2 NVFP4 checkpoint with GLM-5.2-source quantized MTP.

The target is expected to already contain the base GLM-5.2 model with routed
expert NVFP4 weights. This script replaces only ``model.layers.78``:

  - non-expert MTP tensors are copied from the GLM-5.2 BF16 source
  - MTP routed expert MLP weights are statically quantized to NVFP4
  - MTP gate/up ``weight_scale_2`` values are tied per expert
  - MTP activation scales are copied from an existing 5.1 MTP scale shard
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from build_glm52_nvfp4_routed_experts import (
    ROUTED_EXPERT_WEIGHT_RE,
    quantize_nvfp4_gate_up_pair,
    quantize_nvfp4_weight,
)


DEFAULT_TARGET = Path("/models/GLM-5.2-NVFP4-MTP")
DEFAULT_SOURCE = Path(
    "/home/luke/.cache/huggingface/hub/models--zai-org--GLM-5.2/"
    "snapshots/f6142f127a14b58dc602592e996cd7d8ff139351"
)
DEFAULT_MTP_SCALE_SOURCE = Path("/models/GLM-5.1-NVFP4-MTP-NVFP4/model-mtp-inputscales.safetensors")
DEFAULT_QUANT_CONFIG_TEMPLATE = Path("/models/GLM-5.1-NVFP4-MTP-NVFP4/config.json")
MTP_SHARD_NAME = "model-mtp.safetensors"
MTP_SCALE_SHARD_NAME = "model-mtp-inputscales.safetensors"
BLOCK_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace GLM-5.2 MTP with GLM-5.2-source static NVFP4 MoE MLP tensors."
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mtp-scale-source", type=Path, default=DEFAULT_MTP_SCALE_SOURCE)
    parser.add_argument("--quant-config-template", type=Path, default=DEFAULT_QUANT_CONFIG_TEMPLATE)
    parser.add_argument("--mtp-shard-name", default=MTP_SHARD_NAME)
    parser.add_argument("--mtp-scale-shard-name", default=MTP_SCALE_SHARD_NAME)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used for MTP NVFP4 packing.",
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


def load_scale_keys(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return set(f.keys())


def layer_keys(weight_map: dict[str, str], prefix: str) -> set[str]:
    return {key for key in weight_map if key.startswith(prefix)}


def routed_mtp_weight_keys(weight_map: dict[str, str], mtp_layer: int) -> list[str]:
    keys: list[str] = []
    for key in weight_map:
        match = ROUTED_EXPERT_WEIGHT_RE.match(key)
        if match is None:
            continue
        if int(match.group("layer")) == mtp_layer:
            keys.append(key)
    return sorted(keys)


def input_scale_key(weight_key: str) -> str:
    return weight_key.removesuffix(".weight") + ".input_scale"


def validate_mtp_scale_keys(scale_keys: set[str], mtp_weight_keys: list[str]) -> None:
    expected = {input_scale_key(key) for key in mtp_weight_keys}
    missing = sorted(expected - scale_keys)
    extra = sorted(scale_keys - expected)
    if missing or extra:
        msg = [
            "MTP input scale keyset does not exactly match selected MTP routed experts.",
            f"  expected={len(expected)} actual={len(scale_keys)}",
            f"  missing={len(missing)} extra={len(extra)}",
        ]
        if missing:
            msg.append("  first missing: " + ", ".join(missing[:8]))
        if extra:
            msg.append("  first extra: " + ", ".join(extra[:8]))
        raise RuntimeError("\n".join(msg))


def load_quantization_config(template_path: Path) -> dict[str, Any]:
    template = read_json(template_path)
    quant_config = template.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise RuntimeError(f"{template_path} does not contain quantization_config")
    return copy.deepcopy(quant_config)


def merge_target_config(target_config: dict[str, Any], quantization_config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(target_config)
    out["quantization_config"] = copy.deepcopy(quantization_config)

    target_without_quant = copy.deepcopy(target_config)
    out_without_quant = copy.deepcopy(out)
    target_without_quant.pop("quantization_config", None)
    out_without_quant.pop("quantization_config", None)
    if target_without_quant != out_without_quant:
        raise RuntimeError("Config merge changed fields other than quantization_config")
    return out


class TensorStore:
    def __init__(self, root: Path, weight_map: dict[str, str]) -> None:
        self.root = root
        self.weight_map = weight_map
        self._stack: ExitStack | None = None
        self._handles: dict[str, Any] = {}

    def __enter__(self) -> "TensorStore":
        self._stack = ExitStack()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._handles = {}

    def get_tensor(self, key: str) -> torch.Tensor:
        if self._stack is None:
            raise RuntimeError("TensorStore is not open")
        shard = self.weight_map[key]
        handle = self._handles.get(shard)
        if handle is None:
            handle = self._stack.enter_context(
                safe_open(str(self.root / shard), framework="pt", device="cpu")
            )
            self._handles[shard] = handle
        return handle.get_tensor(key)


def routed_key_parts(key: str) -> tuple[int, int, str]:
    match = ROUTED_EXPERT_WEIGHT_RE.match(key)
    if match is None:
        raise ValueError(f"Not a routed expert weight key: {key}")
    return int(match.group("layer")), int(match.group("expert")), match.group("proj")


def routed_weight_key(layer: int, expert: int, proj: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"


def add_quantized_weight(
    tensors: dict[str, torch.Tensor],
    key: str,
    quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    prefix = key.removesuffix(".weight")
    qweight, weight_scale, weight_scale_2 = quantized
    tensors[key] = qweight.contiguous()
    tensors[prefix + ".weight_scale"] = weight_scale.contiguous()
    tensors[prefix + ".weight_scale_2"] = weight_scale_2.reshape(()).contiguous()


def build_mtp_tensors(
    source: Path,
    source_weight_map: dict[str, str],
    prefix: str,
    mtp_weight_keys: set[str],
    block_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    processed_quantized: set[str] = set()

    with TensorStore(source, source_weight_map) as store:
        for key in sorted(layer_keys(source_weight_map, prefix)):
            if key in processed_quantized:
                continue
            if key not in mtp_weight_keys:
                tensors[key] = store.get_tensor(key).contiguous()
                continue

            layer, expert, proj = routed_key_parts(key)
            if proj in {"gate_proj", "up_proj"}:
                gate_key = routed_weight_key(layer, expert, "gate_proj")
                up_key = routed_weight_key(layer, expert, "up_proj")
                gate_quantized, up_quantized = quantize_nvfp4_gate_up_pair(
                    store.get_tensor(gate_key),
                    store.get_tensor(up_key),
                    block_size=block_size,
                    device=device,
                )
                add_quantized_weight(tensors, gate_key, gate_quantized)
                add_quantized_weight(tensors, up_key, up_quantized)
                processed_quantized.update({gate_key, up_key})
            else:
                quantized = quantize_nvfp4_weight(
                    store.get_tensor(key),
                    block_size=block_size,
                    device=device,
                )
                add_quantized_weight(tensors, key, quantized)
                processed_quantized.add(key)

    if processed_quantized != mtp_weight_keys:
        missing = sorted(mtp_weight_keys - processed_quantized)
        raise RuntimeError(f"Did not quantize {len(missing)} MTP weight(s): {missing[:8]}")
    return tensors


def keys_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        grouped[shard].append(key)
    return {shard: sorted(keys) for shard, keys in grouped.items()}


def rewrite_or_remove_stale_mtp_shards(
    target: Path,
    old_weight_map: dict[str, str],
    new_weight_map: dict[str, str],
    prefix: str,
    new_mtp_shards: set[str],
) -> list[str]:
    stale_shards = sorted(
        {
            old_weight_map[key]
            for key in layer_keys(old_weight_map, prefix)
            if old_weight_map[key] not in new_mtp_shards
        }
    )
    new_keys_by_shard = keys_by_shard(new_weight_map)
    rewritten: list[str] = []
    for shard_name in stale_shards:
        shard_path = target / shard_name
        if not shard_path.exists():
            continue
        keep_keys = new_keys_by_shard.get(shard_name, [])
        if not keep_keys:
            shard_path.unlink()
            rewritten.append(shard_name + " (removed)")
            continue
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            tensors = {key: f.get_tensor(key).contiguous() for key in keep_keys}
        tmp = target / (shard_name + ".tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, shard_path)
        rewritten.append(shard_name)
    return rewritten


def write_updated_index(target: Path, old_index: dict[str, Any], weight_map: dict[str, str]) -> None:
    metadata = dict(old_index.get("metadata", {}))
    metadata["total_size"] = sum((target / shard).stat().st_size for shard in set(weight_map.values()))
    index = {
        "metadata": metadata,
        "weight_map": dict(sorted(weight_map.items())),
    }
    write_json(target / "model.safetensors.index.json", index)


def main() -> None:
    args = parse_args()
    target_index = read_json(args.target / "model.safetensors.index.json")
    target_config = read_json(args.target / "config.json")
    source_index = read_json(args.source / "model.safetensors.index.json")
    source_config = read_json(args.source / "config.json")

    mtp_layer = int(source_config["num_hidden_layers"])
    prefix = f"model.layers.{mtp_layer}."
    target_weight_map = target_index["weight_map"]
    source_weight_map = source_index["weight_map"]
    mtp_weight_keys = routed_mtp_weight_keys(source_weight_map, mtp_layer)
    mtp_scale_keys = load_scale_keys(args.mtp_scale_source)
    validate_mtp_scale_keys(mtp_scale_keys, mtp_weight_keys)
    quantization_config = load_quantization_config(args.quant_config_template)
    merge_target_config(target_config, quantization_config)

    source_layer_keys = layer_keys(source_weight_map, prefix)
    new_mtp_keys = set(source_layer_keys)
    for key in mtp_weight_keys:
        base = key.removesuffix(".weight")
        new_mtp_keys.add(base + ".weight_scale")
        new_mtp_keys.add(base + ".weight_scale_2")
    new_mtp_keys.update(mtp_scale_keys)

    new_weight_map = {
        key: shard_name
        for key, shard_name in target_weight_map.items()
        if not key.startswith(prefix)
    }
    for key in sorted(source_layer_keys):
        new_weight_map[key] = args.mtp_shard_name
    for key in sorted(mtp_weight_keys):
        base = key.removesuffix(".weight")
        new_weight_map[base + ".weight_scale"] = args.mtp_shard_name
        new_weight_map[base + ".weight_scale_2"] = args.mtp_shard_name
    for key in sorted(mtp_scale_keys):
        new_weight_map[key] = args.mtp_scale_shard_name

    print(f"target: {args.target}")
    print(f"source: {args.source}")
    print(f"MTP prefix: {prefix}")
    print(f"MTP source layer keys: {len(source_layer_keys)}")
    print(f"MTP routed expert weights selected: {len(mtp_weight_keys)}")
    print(f"MTP activation scales copied: {len(mtp_scale_keys)} from {args.mtp_scale_source}")
    print(f"MTP output shard: {args.mtp_shard_name}")
    print(f"MTP scale shard: {args.mtp_scale_shard_name}")
    print(f"quantization_config template: {args.quant_config_template}")

    if args.dry_run:
        print("Dry run complete; no files written.")
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tensors = build_mtp_tensors(
        args.source,
        source_weight_map,
        prefix,
        set(mtp_weight_keys),
        args.block_size,
        device,
    )
    tmp_mtp = args.target / (args.mtp_shard_name + ".tmp")
    save_file(tensors, str(tmp_mtp))
    os.replace(tmp_mtp, args.target / args.mtp_shard_name)

    tmp_scale = args.target / (args.mtp_scale_shard_name + ".tmp")
    shutil.copy2(args.mtp_scale_source, tmp_scale, follow_symlinks=True)
    os.replace(tmp_scale, args.target / args.mtp_scale_shard_name)

    rewritten = rewrite_or_remove_stale_mtp_shards(
        args.target,
        target_weight_map,
        new_weight_map,
        prefix,
        {args.mtp_shard_name, args.mtp_scale_shard_name},
    )
    write_json(args.target / "config.json", merge_target_config(target_config, quantization_config))
    write_updated_index(args.target, target_index, new_weight_map)

    print(f"Wrote {args.target / args.mtp_shard_name}")
    print(f"Copied {args.target / args.mtp_scale_shard_name}")
    if rewritten:
        print(f"Removed/compacted stale MTP shard(s): {', '.join(rewritten)}")


if __name__ == "__main__":
    main()
