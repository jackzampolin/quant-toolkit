#!/usr/bin/env python3
"""Build a MiniMax-M3 NVFP4 checkpoint with MXFP8 attention/shared experts.

The input checkpoint is expected to already contain NVFP4 routed experts. This
tool leaves those tensors untouched, rewrites only language attention projection
weights and shared-expert weights as ModelOpt MXFP8, and emits mixed ModelOpt
quantization metadata.
"""

from __future__ import annotations

import argparse
import fnmatch
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


DEFAULT_BASE = Path("/models/MiniMax-M3-NVFP4")
DEFAULT_OUTPUT = Path("/models/MiniMax-M3-NVFP4-MXFP8-attn-shared")
INDEX_NAME = "model.safetensors.index.json"

MXFP8_GROUP_SIZE = 32
MXFP8_WEIGHT_BLOCK_SIZE = [1, 32]
NVFP4_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

ATTN_WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj)\.weight$"
)
SHARED_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.shared_experts\."
    r"(?:gate_up_proj|down_proj)\.weight$"
)


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
        raise ValueError(f"Expected a .weight key, got {weight_name}")
    return weight_name.removesuffix(".weight") + ".weight_scale_inv"


def module_prefix(weight_name: str) -> str:
    return weight_name.removesuffix(".weight")


def load_index(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / INDEX_NAME
    if not index_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    return read_json(index_path)


def keys_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        by_shard[shard].append(key)
    return {shard: sorted(keys) for shard, keys in by_shard.items()}


def tensor_meta(model_dir: Path, weight_map: dict[str, str], key: str) -> tuple[list[int], str]:
    with safe_open(model_dir / weight_map[key], framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice(key)
        return tensor_slice.get_shape(), tensor_slice.get_dtype()


def selected_weights(weight_map: dict[str, str]) -> tuple[list[str], list[str]]:
    attention = sorted(key for key in weight_map if ATTN_WEIGHT_RE.match(key))
    shared = sorted(key for key in weight_map if SHARED_EXPERT_WEIGHT_RE.match(key))
    if not attention:
        raise RuntimeError("No language attention projection weights found")
    if not shared:
        raise RuntimeError("No shared-expert weights found")
    return attention, shared


def validate_selected_weights(
    model_dir: Path,
    weight_map: dict[str, str],
    weights: list[str],
) -> None:
    for key in weights:
        if scale_name(key) in weight_map:
            raise RuntimeError(f"MXFP8 sidecar already exists for {key}: {scale_name(key)}")
        shape, dtype = tensor_meta(model_dir, weight_map, key)
        if dtype != "BF16":
            raise RuntimeError(f"Expected BF16 source weight for {key}, got {dtype}")
        if len(shape) != 2:
            raise RuntimeError(f"Expected 2D linear weight for {key}, got shape {shape}")
        if shape[-1] % MXFP8_GROUP_SIZE != 0:
            raise RuntimeError(
                f"MXFP8 group size {MXFP8_GROUP_SIZE} does not divide {key} "
                f"input dimension {shape[-1]}"
            )


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir} (use --force to replace it)"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def copy_metadata_entries(base_dir: Path, output_dir: Path) -> None:
    for entry in sorted(base_dir.iterdir()):
        if entry.name == INDEX_NAME or entry.name.endswith(".safetensors"):
            continue
        dest = output_dir / entry.name
        if entry.is_symlink():
            os.symlink(os.readlink(entry), dest)
        elif entry.is_dir():
            shutil.copytree(entry, dest, symlinks=True)
        else:
            shutil.copy2(entry, dest)


def link_or_copy(src: Path, dest: Path, mode: str) -> None:
    if mode == "symlink":
        os.symlink(src, dest)
    elif mode == "hardlink":
        os.link(src, dest)
    elif mode == "copy":
        shutil.copy2(src, dest)
    else:
        raise ValueError(f"Unknown unchanged-shards mode: {mode}")


def copy_unindexed_safetensors(
    base_dir: Path,
    output_dir: Path,
    indexed_shards: set[str],
    unchanged_shards: str,
) -> None:
    for entry in sorted(base_dir.glob("*.safetensors")):
        if entry.name in indexed_shards:
            continue
        link_or_copy(entry, output_dir / entry.name, unchanged_shards)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def quantize_mxfp8(weight: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    from modelopt.torch.quantization.qtensor import MXFP8QTensor

    work = weight.to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous()
    qweight, scale = MXFP8QTensor.quantize(work)
    qdata = qweight._quantized_data.detach().cpu().contiguous()
    scale = scale.detach().cpu().contiguous()
    del work, qweight
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return qdata, scale


def rewrite_shard(
    base_dir: Path,
    output_dir: Path,
    shard_name: str,
    shard_keys: list[str],
    replacements: set[str],
    device: torch.device,
) -> int:
    tensors: dict[str, torch.Tensor] = {}
    replaced = 0
    shard_path = base_dir / shard_name
    tmp_path = output_dir / f".{shard_name}.tmp"
    if tmp_path.exists():
        tmp_path.unlink()

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in shard_keys:
            tensor = f.get_tensor(key).contiguous()
            if key in replacements:
                qweight, qscale = quantize_mxfp8(tensor, device)
                tensors[key] = qweight
                tensors[scale_name(key)] = qscale
                replaced += 1
                del tensor, qweight, qscale
            else:
                tensors[key] = tensor

    save_kwargs = {"metadata": metadata} if metadata else {}
    save_file(tensors, str(tmp_path), **save_kwargs)
    tmp_path.replace(output_dir / shard_name)
    del tensors
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return replaced


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


def blocks_new_quantized_prefix(pattern: object, prefix: str) -> bool:
    if not isinstance(pattern, str):
        return False
    if fnmatch.fnmatchcase(prefix, pattern):
        return True

    # The original MiniMax-M3 config used source-checkpoint style names in its
    # ignore list. Drop those broad attention/shared-expert ignores as well.
    match = re.match(r"^model\.language_model\.layers\.(\d+)\.(.*)$", prefix)
    if match is None:
        return False
    layer, rest = match.groups()
    legacy_prefix = f"language_model.model.layers.{layer}.{rest}"
    if fnmatch.fnmatchcase(legacy_prefix, pattern):
        return True
    if ".self_attn." in prefix and pattern == f"language_model.model.layers.{layer}.self_attn.*":
        return True
    if ".mlp.shared_experts." in prefix and pattern == (
        f"language_model.model.layers.{layer}.block_sparse_moe.shared_experts.*"
    ):
        return True
    if pattern == f"language_model.model.layers.{layer}.*":
        return True
    return False


def filter_ignore_patterns(ignore: Any, quantized_prefixes: list[str]) -> list[str]:
    if not isinstance(ignore, list):
        return []
    kept = []
    for pattern in ignore:
        if any(blocks_new_quantized_prefix(pattern, prefix) for prefix in quantized_prefixes):
            continue
        kept.append(pattern)
    return kept


def make_quantization_config(
    base_quant: dict[str, Any],
    nvfp4_prefixes: list[str],
    attention_prefixes: list[str],
    shared_prefixes: list[str],
) -> dict[str, Any]:
    quantized_layers: dict[str, dict[str, Any]] = {}
    for prefix in nvfp4_prefixes:
        quantized_layers[prefix] = {"quant_algo": "NVFP4", "group_size": 16}
    for prefix in attention_prefixes + shared_prefixes:
        quantized_layers[prefix] = {
            "quant_algo": "MXFP8",
            "group_size": MXFP8_GROUP_SIZE,
        }

    quantized_layers = dict(sorted(quantized_layers.items()))
    new_mxfp8_prefixes = sorted(attention_prefixes + shared_prefixes)
    return {
        "config_groups": {
            "group_nvfp4_routed_experts": nvfp4_group(nvfp4_prefixes),
            "group_mxfp8_attention": mxfp8_group(sorted(attention_prefixes)),
            "group_mxfp8_shared_experts": mxfp8_group(sorted(shared_prefixes)),
        },
        "quantized_layers": quantized_layers,
        "ignore": filter_ignore_patterns(base_quant.get("ignore", []), new_mxfp8_prefixes),
        "quant_algo": "MIXED_PRECISION",
        "producer": base_quant.get("producer", {}),
        "quant_method": "modelopt",
    }


def update_metadata_files(
    output_dir: Path,
    base_index: dict[str, Any],
    output_weight_map: dict[str, str],
    base_dir: Path,
    attention_weights: list[str],
    shared_weights: list[str],
) -> None:
    metadata = base_index.setdefault("metadata", {})
    metadata["mxfp8_base"] = str(base_dir)
    metadata["mxfp8_attention_projection_tensors"] = len(attention_weights)
    metadata["mxfp8_shared_expert_tensors"] = len(shared_weights)
    metadata["total_size"] = sum(
        (output_dir / shard).stat().st_size for shard in set(output_weight_map.values())
    )
    base_index["weight_map"] = dict(sorted(output_weight_map.items()))
    write_json(output_dir / INDEX_NAME, base_index)

    config_path = output_dir / "config.json"
    config = read_json(config_path)
    base_quant = config.get("quantization_config")
    if not isinstance(base_quant, dict):
        raise RuntimeError(f"{config_path} is missing quantization_config")

    nvfp4_prefixes = detect_nvfp4_prefixes(output_weight_map)
    attention_prefixes = sorted(module_prefix(key) for key in attention_weights)
    shared_prefixes = sorted(module_prefix(key) for key in shared_weights)
    mixed_quant = make_quantization_config(
        base_quant=base_quant,
        nvfp4_prefixes=nvfp4_prefixes,
        attention_prefixes=attention_prefixes,
        shared_prefixes=shared_prefixes,
    )
    config["quantization_config"] = mixed_quant
    write_json(config_path, config)
    write_json(output_dir / "hf_quant_config.json", mixed_quant)


def validate_output(
    output_dir: Path,
    weight_map: dict[str, str],
    weights: list[str],
) -> None:
    for key in weights:
        sidecar = scale_name(key)
        if sidecar not in weight_map:
            raise RuntimeError(f"Missing MXFP8 sidecar in index: {sidecar}")
        weight_shape, weight_dtype = tensor_meta(output_dir, weight_map, key)
        scale_shape, scale_dtype = tensor_meta(output_dir, weight_map, sidecar)
        expected_scale_shape = [weight_shape[0], (weight_shape[1] + MXFP8_GROUP_SIZE - 1) // MXFP8_GROUP_SIZE]
        if weight_dtype != "F8_E4M3":
            raise RuntimeError(f"Expected F8_E4M3 MXFP8 weight for {key}, got {weight_dtype}")
        if scale_dtype != "U8":
            raise RuntimeError(f"Expected U8 MXFP8 scale for {sidecar}, got {scale_dtype}")
        if scale_shape != expected_scale_shape:
            raise RuntimeError(
                f"Unexpected MXFP8 scale shape for {sidecar}: "
                f"got {scale_shape}, expected {expected_scale_shape}"
            )


def build_checkpoint(
    base_dir: Path,
    output_dir: Path,
    unchanged_shards: str,
    device: str,
    force: bool,
    dry_run: bool,
) -> None:
    base_index = load_index(base_dir)
    base_weight_map = dict(base_index["weight_map"])
    by_shard = keys_by_shard(base_weight_map)
    attention_weights, shared_weights = selected_weights(base_weight_map)
    replacement_weights = sorted(set(attention_weights) | set(shared_weights))
    validate_selected_weights(base_dir, base_weight_map, replacement_weights)
    affected_shards = sorted({base_weight_map[key] for key in replacement_weights})

    print(f"Base checkpoint: {base_dir}")
    print(f"Output checkpoint: {output_dir}")
    print(f"Attention projection weights -> MXFP8: {len(attention_weights)}")
    print(f"Shared expert weights -> MXFP8: {len(shared_weights)}")
    print(f"Affected shards: {len(affected_shards)} / {len(by_shard)}")
    print(f"Unchanged safetensor handling: {unchanged_shards}")
    print(f"Quantization device: {resolve_device(device)}")

    if dry_run:
        for shard in affected_shards:
            count = sum(1 for key in replacement_weights if base_weight_map[key] == shard)
            print(f"  {shard}: {count} tensor(s)")
        print("Dry run; no files written.")
        return

    prepare_output_dir(output_dir, force)
    copy_metadata_entries(base_dir, output_dir)
    qdevice = resolve_device(device)
    replacement_set = set(replacement_weights)
    output_weight_map = dict(base_weight_map)
    for key in replacement_weights:
        output_weight_map[scale_name(key)] = base_weight_map[key]

    for shard in sorted(by_shard):
        if shard in affected_shards:
            replaced = rewrite_shard(
                base_dir=base_dir,
                output_dir=output_dir,
                shard_name=shard,
                shard_keys=by_shard[shard],
                replacements=replacement_set,
                device=qdevice,
            )
            print(f"rewrote {shard}: {replaced} MXFP8 tensor(s)", flush=True)
        else:
            link_or_copy(base_dir / shard, output_dir / shard, unchanged_shards)

    copy_unindexed_safetensors(
        base_dir=base_dir,
        output_dir=output_dir,
        indexed_shards=set(by_shard),
        unchanged_shards=unchanged_shards,
    )
    update_metadata_files(
        output_dir=output_dir,
        base_index=base_index,
        output_weight_map=output_weight_map,
        base_dir=base_dir,
        attention_weights=attention_weights,
        shared_weights=shared_weights,
    )
    validate_output(output_dir, output_weight_map, replacement_weights)
    print(f"updated {output_dir / INDEX_NAME}")
    print(f"wrote mixed ModelOpt config to {output_dir / 'config.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--unchanged-shards",
        choices=["symlink", "hardlink", "copy"],
        default="hardlink",
        help="How to place safetensor shards that do not contain rewritten tensors.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device used for MXFP8 quantization. Default: cuda:0 when available, else cpu.",
    )
    parser.add_argument("--force", action="store_true", help="Replace the output directory if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing files.")
    args = parser.parse_args()

    build_checkpoint(
        base_dir=args.base,
        output_dir=args.output,
        unchanged_shards=args.unchanged_shards,
        device=args.device,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
