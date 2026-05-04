#!/usr/bin/env python3
"""Build a MiMo NVFP4 checkpoint with FP8/MXFP8 fused-QKV attention weights.

The output is based on an existing NVFP4 export.  For each language-model QKV
projection, and optionally MTP QKV projection, this script replaces the base
BF16/dequantized ``qkv_proj.weight`` tensor with the quantized tensor from a
source checkpoint and adds the matching ``qkv_proj.weight_scale_inv`` sidecar.
All other tensors and config files come from the base checkpoint.
"""

from __future__ import annotations

import argparse
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


DEFAULT_BASE = Path("/data/models/MiMo-V2.5-NVFP4-w13-tied")
DEFAULT_FP8 = Path("/data/models/MiMo-V2.5-FP8-qkv-deinterleaved")
DEFAULT_OUTPUT = Path("/data/models/MiMo-V2.5-NVFP4-w13-tied-FP8-attn")
INDEX_NAME = "model.safetensors.index.json"
LANGUAGE_QKV_RE = re.compile(r"^model\.layers\.\d+\.self_attn\.qkv_proj\.weight$")
MTP_QKV_RE = re.compile(r"^model\.mtp\.layers\.\d+\.self_attn\.qkv_proj\.weight$")
FP8_PB_WEIGHT_BLOCK_SIZE = [128, 128]
MXFP8_WEIGHT_BLOCK_SIZE = [1, 32]


def _load_index(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / INDEX_NAME
    if not index_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    with index_path.open() as f:
        return json.load(f)


def _qkv_scale_name(weight_name: str) -> str:
    suffix = ".qkv_proj.weight"
    if not weight_name.endswith(suffix):
        raise ValueError(f"Not a qkv_proj.weight key: {weight_name}")
    return weight_name[: -len(suffix)] + ".qkv_proj.weight_scale_inv"


def _selected_qkv_keys(weight_map: dict[str, str], include_mtp: bool) -> list[str]:
    keys = []
    for key in weight_map:
        if LANGUAGE_QKV_RE.match(key) or (include_mtp and MTP_QKV_RE.match(key)):
            keys.append(key)
    return sorted(keys)


def _keys_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        by_shard[shard_name].append(key)
    return {shard: sorted(keys) for shard, keys in by_shard.items()}


def _tensor_meta(model_dir: Path, weight_map: dict[str, str], key: str) -> tuple[list[int], str]:
    with safe_open(model_dir / weight_map[key], framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice(key)
        return tensor_slice.get_shape(), tensor_slice.get_dtype()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _infer_attention_format(
    weight_shape: list[int],
    scale_shape: list[int],
    scale_dtype: str,
) -> str:
    if scale_dtype == "F32":
        expected = [
            _ceil_div(weight_shape[0], FP8_PB_WEIGHT_BLOCK_SIZE[0]),
            _ceil_div(weight_shape[1], FP8_PB_WEIGHT_BLOCK_SIZE[1]),
        ]
        if scale_shape != expected:
            raise RuntimeError(
                f"Expected FP8_PB_WO scale shape {expected}, got {scale_shape}"
            )
        return "fp8-pb"
    if scale_dtype == "U8":
        expected = [weight_shape[0], _ceil_div(weight_shape[1], MXFP8_WEIGHT_BLOCK_SIZE[1])]
        if scale_shape != expected:
            raise RuntimeError(f"Expected MXFP8 scale shape {expected}, got {scale_shape}")
        return "mxfp8"
    raise RuntimeError(f"Unsupported QKV scale dtype {scale_dtype}; expected F32 or U8")


def _validate_replacements(
    base_dir: Path,
    fp8_dir: Path,
    base_weight_map: dict[str, str],
    fp8_weight_map: dict[str, str],
    qkv_keys: list[str],
    requested_format: str,
) -> str:
    missing_weights = sorted(key for key in qkv_keys if key not in fp8_weight_map)
    missing_scales = sorted(_qkv_scale_name(key) for key in qkv_keys if _qkv_scale_name(key) not in fp8_weight_map)
    if missing_weights:
        raise RuntimeError(f"FP8 checkpoint is missing {len(missing_weights)} QKV weight(s); first: {missing_weights[0]}")
    if missing_scales:
        raise RuntimeError(f"FP8 checkpoint is missing {len(missing_scales)} QKV scale(s); first: {missing_scales[0]}")

    inferred_formats = set()
    for key in qkv_keys:
        base_shape, _ = _tensor_meta(base_dir, base_weight_map, key)
        fp8_shape, fp8_dtype = _tensor_meta(fp8_dir, fp8_weight_map, key)
        scale_shape, scale_dtype = _tensor_meta(fp8_dir, fp8_weight_map, _qkv_scale_name(key))
        if base_shape != fp8_shape:
            raise RuntimeError(f"Shape mismatch for {key}: base {base_shape}, fp8 {fp8_shape}")
        if fp8_dtype != "F8_E4M3":
            raise RuntimeError(f"Expected FP8 E4M3 weight for {key}, got {fp8_dtype}")
        if len(scale_shape) != 2:
            raise RuntimeError(f"Expected 2D scale for {_qkv_scale_name(key)}, got {scale_shape}")
        inferred_format = _infer_attention_format(fp8_shape, scale_shape, scale_dtype)
        if requested_format != "auto" and inferred_format != requested_format:
            raise RuntimeError(
                f"{key} is {inferred_format}, but --attention-format={requested_format}"
            )
        inferred_formats.add(inferred_format)
    if len(inferred_formats) != 1:
        raise RuntimeError(f"Expected one attention format, got {sorted(inferred_formats)}")
    return next(iter(inferred_formats))


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir} (use --force to replace it)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _copy_metadata_entries(base_dir: Path, output_dir: Path) -> None:
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


def _link_or_copy_safetensor(src: Path, dest: Path, mode: str) -> None:
    if mode == "symlink":
        os.symlink(src, dest)
    elif mode == "hardlink":
        os.link(src, dest)
    elif mode == "copy":
        shutil.copy2(src, dest)
    else:
        raise ValueError(f"Unknown unchanged-shards mode: {mode}")


def _copy_unindexed_safetensors(
    base_dir: Path,
    output_dir: Path,
    indexed_shards: set[str],
    unchanged_shards: str,
) -> None:
    for entry in sorted(base_dir.glob("*.safetensors")):
        if entry.name in indexed_shards:
            continue
        _link_or_copy_safetensor(entry, output_dir / entry.name, unchanged_shards)


def _load_fp8_tensor(fp8_dir: Path, fp8_weight_map: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(fp8_dir / fp8_weight_map[key], framework="pt", device="cpu") as f:
        return f.get_tensor(key).contiguous()


def _rewrite_shard(
    base_dir: Path,
    fp8_dir: Path,
    output_dir: Path,
    base_weight_map: dict[str, str],
    fp8_weight_map: dict[str, str],
    shard_name: str,
    shard_keys: list[str],
    replacements: set[str],
) -> int:
    tensors: dict[str, torch.Tensor] = {}
    replaced_count = 0
    base_shard_path = base_dir / shard_name

    with safe_open(base_shard_path, framework="pt", device="cpu") as f:
        for key in shard_keys:
            if key in replacements:
                tensors[key] = _load_fp8_tensor(fp8_dir, fp8_weight_map, key)
                tensors[_qkv_scale_name(key)] = _load_fp8_tensor(fp8_dir, fp8_weight_map, _qkv_scale_name(key))
                replaced_count += 1
            else:
                tensors[key] = f.get_tensor(key).contiguous()

    temp_path = output_dir / f".{shard_name}.tmp"
    final_path = output_dir / shard_name
    save_file(tensors, str(temp_path))
    temp_path.replace(final_path)
    return replaced_count


def _write_index(output_dir: Path, index: dict[str, Any], weight_map: dict[str, str]) -> None:
    index["weight_map"] = dict(sorted(weight_map.items()))
    metadata = index.setdefault("metadata", {})
    metadata["total_size"] = sum((output_dir / shard).stat().st_size for shard in set(weight_map.values()))
    with (output_dir / INDEX_NAME).open("w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")


def build_chimera(
    base_dir: Path,
    fp8_dir: Path,
    output_dir: Path,
    attention_format: str,
    include_mtp: bool,
    unchanged_shards: str,
    force: bool,
    dry_run: bool,
) -> None:
    base_index = _load_index(base_dir)
    fp8_index = _load_index(fp8_dir)
    base_weight_map = dict(base_index["weight_map"])
    fp8_weight_map = dict(fp8_index["weight_map"])
    qkv_keys = _selected_qkv_keys(base_weight_map, include_mtp)
    replacements = set(qkv_keys)
    affected_shards = sorted({base_weight_map[key] for key in qkv_keys})
    base_by_shard = _keys_by_shard(base_weight_map)

    inferred_attention_format = _validate_replacements(
        base_dir, fp8_dir, base_weight_map, fp8_weight_map, qkv_keys, attention_format
    )

    print(f"Base checkpoint: {base_dir}")
    print(f"Quantized attention source: {fp8_dir}")
    print(f"Attention format: {inferred_attention_format}")
    print(f"Output checkpoint: {output_dir}")
    print(f"Selected QKV weights: {len(qkv_keys)}")
    print(f"Affected base shards: {len(affected_shards)}")
    print(f"Unchanged safetensor handling: {unchanged_shards}")
    if include_mtp:
        print("MTP QKV replacement: enabled")
    else:
        print("MTP QKV replacement: disabled")

    if dry_run:
        print("\nDry run; no files written.")
        print("Affected shards:")
        for shard_name in affected_shards:
            count = sum(1 for key in qkv_keys if base_weight_map[key] == shard_name)
            print(f"  {shard_name}: {count} QKV tensor(s)")
        return

    _prepare_output_dir(output_dir, force)
    _copy_metadata_entries(base_dir, output_dir)

    output_weight_map = dict(base_weight_map)
    for key in qkv_keys:
        output_weight_map[_qkv_scale_name(key)] = base_weight_map[key]

    for shard_name in sorted(base_by_shard):
        if shard_name in affected_shards:
            replaced_count = _rewrite_shard(
                base_dir=base_dir,
                fp8_dir=fp8_dir,
                output_dir=output_dir,
                base_weight_map=base_weight_map,
                fp8_weight_map=fp8_weight_map,
                shard_name=shard_name,
                shard_keys=base_by_shard[shard_name],
                replacements=replacements,
            )
            print(f"rewrote {shard_name}: replaced {replaced_count} QKV tensor(s)")
        else:
            _link_or_copy_safetensor(base_dir / shard_name, output_dir / shard_name, unchanged_shards)

    _copy_unindexed_safetensors(
        base_dir=base_dir,
        output_dir=output_dir,
        indexed_shards=set(base_by_shard),
        unchanged_shards=unchanged_shards,
    )

    metadata = base_index.setdefault("metadata", {})
    metadata["chimera_base"] = str(base_dir)
    metadata["chimera_quantized_attention_source"] = str(fp8_dir)
    metadata["chimera_attention_format"] = inferred_attention_format
    metadata["chimera_qkv_tensors"] = len(qkv_keys)
    metadata["chimera_includes_mtp_qkv"] = include_mtp
    _write_index(output_dir, base_index, output_weight_map)
    print(f"updated {output_dir / INDEX_NAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--fp8-attn", type=Path, default=DEFAULT_FP8)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attention-format", choices=["auto", "fp8-pb", "mxfp8"], default="auto")
    parser.add_argument("--no-mtp", action="store_true", help="Do not replace model.mtp.* QKV tensors.")
    parser.add_argument(
        "--unchanged-shards",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to place safetensor shards that do not contain replaced QKV tensors.",
    )
    parser.add_argument("--force", action="store_true", help="Replace the output directory if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing files.")
    args = parser.parse_args()

    build_chimera(
        base_dir=args.base,
        fp8_dir=args.fp8_attn,
        output_dir=args.output,
        attention_format=args.attention_format,
        include_mtp=not args.no_mtp,
        unchanged_shards=args.unchanged_shards,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
