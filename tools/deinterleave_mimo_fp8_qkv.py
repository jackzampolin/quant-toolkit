#!/usr/bin/env python3
"""Deinterleave MiMo fused-QKV FP8 tensors.

This preserves the original FP8 values and scale values exactly for QKV layouts
whose per-rank Q/K/V row counts are aligned to the FP8 scale block height. If a
Q/K/V boundary cuts through a scale block, exact canonical FP8 is impossible
with the usual block scale grid; use ``--on-inexact requantize`` to dequantize
only those tensors, reorder them, and write fresh FP8 blocks.

Use ``--output-format mxfp8`` to always go through FP8 -> BF16 -> ModelOpt
MXFP8 for selected QKV tensors. In that mode, this script uses ModelOpt's
``MXFP8QTensor.quantize`` implementation for the final MXFP8 quantization.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from dequantize_fp8 import build_mimo_qkv_specs, build_weight_map_from_safetensors
from mimo_qkv_formats import (
    FP8_PB_WEIGHT_BLOCK_SIZE,
    MXFP8_GROUP_SIZE,
    MXFP8_WEIGHT_BLOCK_SIZE,
    infer_qkv_quant_format,
    qkv_config_group,
    qkv_quantized_layer_entry,
)


COPY_EXTS = {".json", ".txt", ".model", ".py", ".jinja", ".pt", ".md", ".png"}
COPY_NAMES = {".gitattributes"}
INDEX_NAME = "model.safetensors.index.json"
QKV_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$")
MTP_QKV_RE = re.compile(r"^model\.mtp\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$")


def resolve_model_dir(model_id: str) -> Path:
    path = Path(model_id)
    if path.exists():
        return path
    return Path(snapshot_download(model_id, local_files_only=True))


def load_index(model_dir: Path, ignore_index: bool) -> tuple[dict[str, str], dict]:
    index_path = model_dir / INDEX_NAME
    if not ignore_index and index_path.exists():
        with index_path.open() as f:
            index = json.load(f)
        return index["weight_map"], dict(index.get("metadata", {}))
    return build_weight_map_from_safetensors(model_dir), {}


def qkv_scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected a weight tensor name, got {weight_name}")
    return weight_name.removesuffix(".weight") + ".weight_scale_inv"


def qkv_shard_sizes(spec: dict[str, int]) -> tuple[int, int, int]:
    tp_size = spec["tp_size"]
    return (
        spec["q_size"] // tp_size,
        spec["k_size"] // tp_size,
        spec["v_size"] // tp_size,
    )


def exact_fp8_deinterleave_reason(spec: dict[str, int], block_m: int) -> str | None:
    q_shard, k_shard, v_shard = qkv_shard_sizes(spec)
    misaligned = [
        f"{name}_shard={rows}"
        for name, rows in (("q", q_shard), ("k", k_shard), ("v", v_shard))
        if rows % block_m != 0
    ]
    if misaligned:
        return (
            "Q/K/V shard boundary is not aligned to FP8 scale blocks "
            f"of {block_m} rows ({', '.join(misaligned)})"
        )
    return None


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _split_rank_local_qkv(tensor: torch.Tensor, spec: dict[str, int]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    tp_size = spec["tp_size"]
    q_shard, k_shard, v_shard = qkv_shard_sizes(spec)
    chunk_size = q_shard + k_shard + v_shard
    expected_rows = chunk_size * tp_size
    if tensor.shape[0] != expected_rows:
        raise ValueError(
            f"Cannot deinterleave {spec['name']}: expected {expected_rows} rows "
            f"from TP-packed QKV layout, got {tensor.shape[0]}"
        )

    q_chunks = []
    k_chunks = []
    v_chunks = []
    for rank in range(tp_size):
        chunk = tensor.narrow(0, rank * chunk_size, chunk_size)
        q, k, v = chunk.split([q_shard, k_shard, v_shard], dim=0)
        q_chunks.append(q)
        k_chunks.append(k)
        v_chunks.append(v)
    return q_chunks, k_chunks, v_chunks


def deinterleave_qkv_weight_rows(tensor: torch.Tensor, spec: dict[str, int]) -> torch.Tensor:
    q_chunks, k_chunks, v_chunks = _split_rank_local_qkv(tensor, spec)
    return torch.cat([*q_chunks, *k_chunks, *v_chunks], dim=0).contiguous()


def deinterleave_qkv_scale_rows(
    scale: torch.Tensor,
    spec: dict[str, int],
    block_m: int,
) -> torch.Tensor:
    reason = exact_fp8_deinterleave_reason(spec, block_m)
    if reason is not None:
        raise ValueError(f"Cannot exactly deinterleave {spec['name']} scales: {reason}")

    tp_size = spec["tp_size"]
    q_shard, k_shard, v_shard = qkv_shard_sizes(spec)
    q_scale_rows = q_shard // block_m
    k_scale_rows = k_shard // block_m
    v_scale_rows = v_shard // block_m
    scale_rows_per_chunk = q_scale_rows + k_scale_rows + v_scale_rows
    expected_rows = scale_rows_per_chunk * tp_size
    if scale.shape[0] < expected_rows:
        raise ValueError(
            f"Cannot deinterleave {spec['name']} scales: expected at least "
            f"{expected_rows} rows, got {scale.shape[0]}"
        )

    q_chunks = []
    k_chunks = []
    v_chunks = []
    core = scale.narrow(0, 0, expected_rows)
    for rank in range(tp_size):
        chunk = core.narrow(0, rank * scale_rows_per_chunk, scale_rows_per_chunk)
        q, k, v = chunk.split([q_scale_rows, k_scale_rows, v_scale_rows], dim=0)
        q_chunks.append(q)
        k_chunks.append(k)
        v_chunks.append(v)

    reordered = torch.cat([*q_chunks, *k_chunks, *v_chunks], dim=0)
    if scale.shape[0] > expected_rows:
        reordered = torch.cat([reordered, scale.narrow(0, expected_rows, scale.shape[0] - expected_rows)], dim=0)
    return reordered.contiguous()


def dequantize_block_fp8_to_float(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    weight_fp32 = weight.to(torch.float32)
    block_m, block_n = block_size
    expected_scale_shape = (
        ceil_div(weight_fp32.shape[0], block_m),
        ceil_div(weight_fp32.shape[1], block_n),
    )
    if scale.shape[0] < expected_scale_shape[0] or scale.shape[1] < expected_scale_shape[1]:
        raise ValueError(
            "FP8 scale grid is smaller than the weight requires: "
            f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}, expected={expected_scale_shape}"
        )
    scale = scale[: expected_scale_shape[0], : expected_scale_shape[1]].to(torch.float32)
    scale_expanded = scale.repeat_interleave(block_m, dim=0)
    scale_expanded = scale_expanded.repeat_interleave(block_n, dim=1)
    scale_expanded = scale_expanded[: weight_fp32.shape[0], : weight_fp32.shape[1]]
    return weight_fp32 * scale_expanded


def deinterleave_qkv_dequantized_rows(
    weight: torch.Tensor,
    scale: torch.Tensor,
    spec: dict[str, int],
    block_size: tuple[int, int],
) -> torch.Tensor:
    tp_size = spec["tp_size"]
    q_shard, k_shard, v_shard = qkv_shard_sizes(spec)
    chunk_size = q_shard + k_shard + v_shard
    block_m, _ = block_size
    scale_rows_per_chunk = ceil_div(chunk_size, block_m)
    expected_scale_rows = scale_rows_per_chunk * tp_size
    expected_weight_rows = chunk_size * tp_size
    if weight.shape[0] != expected_weight_rows:
        raise ValueError(
            f"Cannot deinterleave {spec['name']}: expected {expected_weight_rows} rows "
            f"from TP-packed QKV layout, got {weight.shape[0]}"
        )
    if scale.shape[0] < expected_scale_rows:
        raise ValueError(
            f"Cannot deinterleave {spec['name']}: expected at least {expected_scale_rows} "
            f"scale rows for TP-packed QKV layout, got {scale.shape[0]}"
        )

    q_chunks = []
    k_chunks = []
    v_chunks = []
    for rank in range(tp_size):
        weight_chunk = weight.narrow(0, rank * chunk_size, chunk_size)
        scale_chunk = scale.narrow(0, rank * scale_rows_per_chunk, scale_rows_per_chunk)
        dequant_chunk = dequantize_block_fp8_to_float(weight_chunk, scale_chunk, block_size)
        q, k, v = dequant_chunk.split([q_shard, k_shard, v_shard], dim=0)
        q_chunks.append(q)
        k_chunks.append(k)
        v_chunks.append(v)
    return torch.cat([*q_chunks, *k_chunks, *v_chunks], dim=0).contiguous()


def quantize_block_fp8_from_float(
    weight: torch.Tensor,
    block_size: tuple[int, int],
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp32 = weight.to(torch.float32)
    block_m, block_n = block_size
    scale_shape = (
        ceil_div(weight_fp32.shape[0], block_m),
        ceil_div(weight_fp32.shape[1], block_n),
    )
    qweight = torch.empty_like(weight_fp32, dtype=torch.float8_e4m3fn)
    scale = torch.empty(scale_shape, dtype=torch.float32)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    if weight_fp32.shape[0] % block_m == 0 and weight_fp32.shape[1] % block_n == 0:
        blocked = weight_fp32.reshape(
            weight_fp32.shape[0] // block_m,
            block_m,
            weight_fp32.shape[1] // block_n,
            block_n,
        ).permute(0, 2, 1, 3)
        scale = blocked.abs().amax(dim=(-1, -2)) / fp8_max
        scale = torch.where(
            (scale == 0) | ~torch.isfinite(scale),
            torch.ones((), dtype=torch.float32, device=scale.device),
            scale,
        )
        scaled = (blocked / scale.unsqueeze(-1).unsqueeze(-1)).clamp(min=-fp8_max, max=fp8_max)
        qweight = scaled.permute(0, 2, 1, 3).reshape(weight_fp32.shape).to(torch.float8_e4m3fn)
        return qweight.contiguous(), scale.to(scale_dtype).contiguous()

    for row_idx, row_start in enumerate(range(0, weight_fp32.shape[0], block_m)):
        row_end = min(row_start + block_m, weight_fp32.shape[0])
        for col_idx, col_start in enumerate(range(0, weight_fp32.shape[1], block_n)):
            col_end = min(col_start + block_n, weight_fp32.shape[1])
            block = weight_fp32[row_start:row_end, col_start:col_end]
            block_scale = block.abs().amax() / fp8_max
            if block_scale == 0 or not torch.isfinite(block_scale):
                block_scale = torch.ones((), dtype=torch.float32)
            scale[row_idx, col_idx] = block_scale
            qblock = (block / block_scale).clamp(min=-fp8_max, max=fp8_max)
            qweight[row_start:row_end, col_start:col_end] = qblock.to(torch.float8_e4m3fn)

    return qweight.contiguous(), scale.to(scale_dtype).contiguous()


def requantize_deinterleaved_qkv(
    weight: torch.Tensor,
    scale: torch.Tensor,
    spec: dict[str, int],
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    dequantized = deinterleave_qkv_dequantized_rows(weight, scale, spec, block_size)
    return quantize_block_fp8_from_float(dequantized, block_size, scale.dtype)


def quantize_mxfp8_from_bf16(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from modelopt.torch.quantization.qtensor import MXFP8QTensor

    qweight, scale = MXFP8QTensor.quantize(weight.to(torch.bfloat16).contiguous())
    return qweight._quantized_data.contiguous(), scale.contiguous()


def convert_deinterleaved_qkv_to_mxfp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    spec: dict[str, int],
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    dequantized = deinterleave_qkv_dequantized_rows(weight, scale, spec, block_size)
    return quantize_mxfp8_from_bf16(dequantized.to(torch.bfloat16))


def layer_sort_key(weight_name: str) -> int:
    match = QKV_RE.match(weight_name)
    if match:
        return int(match.group(1))
    match = MTP_QKV_RE.match(weight_name)
    return int(match.group(1)) if match else 10**9


def qkv_sort_key(weight_name: str) -> tuple[int, int]:
    if QKV_RE.match(weight_name):
        return (0, layer_sort_key(weight_name))
    if MTP_QKV_RE.match(weight_name):
        return (1, layer_sort_key(weight_name))
    return (2, layer_sort_key(weight_name))


def qkv_sizes_from_config(config: dict, is_swa: bool) -> tuple[int, int, int]:
    default_head_dim = config["hidden_size"] // config["num_attention_heads"]
    head_dim = config.get("head_dim", default_head_dim)
    v_head_dim = config.get("v_head_dim", head_dim)
    swa_head_dim = config.get("swa_head_dim", head_dim)
    swa_v_head_dim = config.get("swa_v_head_dim", v_head_dim)

    num_attention_heads = (
        config.get("swa_num_attention_heads", config["num_attention_heads"])
        if is_swa
        else config["num_attention_heads"]
    )
    num_key_value_heads = (
        config.get("swa_num_key_value_heads", config["num_key_value_heads"])
        if is_swa
        else config["num_key_value_heads"]
    )
    q_head_dim = swa_head_dim if is_swa else head_dim
    k_head_dim = q_head_dim
    v_dim = swa_v_head_dim if is_swa else v_head_dim
    return (
        num_attention_heads * q_head_dim,
        num_key_value_heads * k_head_dim,
        num_key_value_heads * v_dim,
    )


def tensor_shape(model_dir: Path, weight_map: dict[str, str], key: str) -> list[int]:
    with safe_open(model_dir / weight_map[key], framework="pt", device="cpu") as f:
        return f.get_slice(key).get_shape()


def tensor_meta(model_dir: Path, weight_map: dict[str, str], key: str) -> tuple[list[int], str]:
    with safe_open(model_dir / weight_map[key], framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice(key)
        return tensor_slice.get_shape(), tensor_slice.get_dtype()


def build_mimo_mtp_qkv_specs(
    config: dict,
    model_dir: Path,
    weight_map: dict[str, str],
    tp_size: int,
) -> dict[str, dict[str, int]]:
    specs = {}
    candidate_sizes = {
        is_swa: qkv_sizes_from_config(config, is_swa)
        for is_swa in (False, True)
    }
    unique_candidate_sizes = sorted(set(candidate_sizes.values()))
    for weight_name in sorted(weight_map):
        if MTP_QKV_RE.match(weight_name) is None:
            continue
        rows = tensor_shape(model_dir, weight_map, weight_name)[0]
        matches = [
            sizes
            for sizes in unique_candidate_sizes
            if sum(sizes) == rows
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot infer MTP QKV split for {weight_name}: rows={rows}, "
                f"candidate_splits={candidate_sizes}"
            )
        q_size, k_size, v_size = matches[0]
        if q_size % tp_size != 0 or k_size % tp_size != 0 or v_size % tp_size != 0:
            raise ValueError(
                f"{weight_name} cannot be evenly deinterleaved with TP size {tp_size}: "
                f"q={q_size}, k={k_size}, v={v_size}"
            )
        specs[weight_name] = {
            "name": weight_name,
            "tp_size": tp_size,
            "q_size": q_size,
            "k_size": k_size,
            "v_size": v_size,
        }
    return specs


def summarize_targets(
    specs: dict[str, dict[str, int]],
    block_m: int,
) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    exact = {}
    inexact = {}
    for weight_name, spec in sorted(specs.items(), key=lambda item: qkv_sort_key(item[0])):
        reason = exact_fp8_deinterleave_reason(spec, block_m)
        if reason is None:
            exact[weight_name] = spec
        else:
            inexact[weight_name] = reason
    return exact, inexact


def format_qkv_names(weight_names: list[str], limit: int = 24) -> str:
    labels = []
    for name in sorted(weight_names, key=qkv_sort_key):
        if QKV_RE.match(name):
            labels.append(f"L{layer_sort_key(name)}")
        elif MTP_QKV_RE.match(name):
            labels.append(f"MTP{layer_sort_key(name)}")
        else:
            labels.append(name)
    if len(labels) > limit:
        return ", ".join(labels[:limit]) + f", ... ({len(labels)} total)"
    return ", ".join(labels)


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        for item in output_dir.iterdir():
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_sidecars(source_dir: Path, output_dir: Path) -> None:
    for src in source_dir.rglob("*"):
        if not src.is_file() or (src.suffix not in COPY_EXTS and src.name not in COPY_NAMES):
            continue
        rel_path = src.relative_to(source_dir)
        dst = output_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def link_or_copy_unindexed_safetensors(
    source_dir: Path,
    output_dir: Path,
    indexed_shards: set[str],
    mode: str,
) -> None:
    for src in source_dir.rglob("*.safetensors"):
        rel_path = src.relative_to(source_dir)
        if len(rel_path.parts) == 1 and src.name in indexed_shards:
            continue
        dst = output_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(src, dst, mode)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown unchanged shard mode: {mode}")


def rewrite_modified_shard(
    source_dir: Path,
    output_dir: Path,
    file_name: str,
    weight_specs: dict[str, dict[str, int]],
    scale_specs: dict[str, dict[str, int]],
    requant_weight_specs: dict[str, dict[str, int]],
    requant_scale_names: set[str],
    mxfp8_qkv_specs: dict[str, dict[str, int]],
    mxfp8_scale_names: set[str],
    weight_map: dict[str, str],
    block_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    tensors = {}
    weights_rewritten = 0
    scales_rewritten = 0
    weights_requantized = 0
    weights_mxfp8 = 0
    generated_scales = {}
    with safe_open(source_dir / file_name, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key in requant_scale_names or key in mxfp8_scale_names:
                continue
            tensor = f.get_tensor(key)
            if key in weight_specs:
                if tensor.dtype != torch.float8_e4m3fn:
                    raise TypeError(
                        f"{key} is {tensor.dtype}, not torch.float8_e4m3fn; "
                        "this tool is for FP8-preserving rewrites"
                    )
                tensor = deinterleave_qkv_weight_rows(tensor, weight_specs[key])
                weights_rewritten += 1
            elif key in scale_specs:
                tensor = deinterleave_qkv_scale_rows(tensor, scale_specs[key], block_size[0])
                scales_rewritten += 1
            elif key in requant_weight_specs:
                if tensor.dtype != torch.float8_e4m3fn:
                    raise TypeError(
                        f"{key} is {tensor.dtype}, not torch.float8_e4m3fn; "
                        "this tool is for FP8-preserving rewrites"
                    )
                scale_name = qkv_scale_name(key)
                scale_file = weight_map[scale_name]
                if scale_file == file_name:
                    scale = f.get_tensor(scale_name)
                else:
                    with safe_open(source_dir / scale_file, framework="pt", device="cpu") as sf:
                        scale = sf.get_tensor(scale_name)
                tensor, generated_scales[scale_name] = requantize_deinterleaved_qkv(
                    tensor, scale, requant_weight_specs[key], block_size
                )
                weights_requantized += 1
            elif key in mxfp8_qkv_specs:
                if tensor.dtype != torch.float8_e4m3fn:
                    raise TypeError(
                        f"{key} is {tensor.dtype}, not torch.float8_e4m3fn; "
                        "MXFP8 conversion expects serialized FP8 source weights"
                    )
                scale_name = qkv_scale_name(key)
                scale_file = weight_map[scale_name]
                if scale_file == file_name:
                    scale = f.get_tensor(scale_name)
                else:
                    with safe_open(source_dir / scale_file, framework="pt", device="cpu") as sf:
                        scale = sf.get_tensor(scale_name)
                tensor, generated_scales[scale_name] = convert_deinterleaved_qkv_to_mxfp8(
                    tensor, scale, mxfp8_qkv_specs[key], block_size
                )
                weights_mxfp8 += 1
            tensors[key] = tensor.contiguous()

    tensors.update(generated_scales)
    if tensors:
        save_file(tensors, str(output_dir / file_name))
    return weights_rewritten, scales_rewritten, weights_requantized, weights_mxfp8


def write_index(output_dir: Path, weight_map: dict[str, str], metadata: dict) -> None:
    metadata = dict(metadata)
    metadata["total_size"] = sum(
        (output_dir / file_name).stat().st_size
        for file_name in set(weight_map.values())
    )
    with (output_dir / INDEX_NAME).open("w") as f:
        json.dump({"metadata": metadata, "weight_map": weight_map}, f, indent=2, sort_keys=True)
        f.write("\n")


def qkv_prefix(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected a QKV weight tensor, got {weight_name}")
    return weight_name.removesuffix(".weight")


def is_qkv_prefix(prefix: str) -> bool:
    return QKV_RE.match(f"{prefix}.weight") is not None or MTP_QKV_RE.match(f"{prefix}.weight") is not None


def detect_output_qkv_formats(
    output_dir: Path,
    weight_map: dict[str, str],
    specs: dict[str, dict[str, int]],
    expected_format: str,
) -> dict[str, list[str]]:
    prefixes_by_format: dict[str, list[str]] = {"fp8-pb": [], "mxfp8": []}
    for weight_name in sorted(specs, key=qkv_sort_key):
        scale_name = qkv_scale_name(weight_name)
        if weight_name not in weight_map:
            raise RuntimeError(f"Output index is missing QKV weight {weight_name}")
        if scale_name not in weight_map:
            raise RuntimeError(f"Output index is missing QKV scale {scale_name}")

        weight_shape, weight_dtype = tensor_meta(output_dir, weight_map, weight_name)
        scale_shape, scale_dtype = tensor_meta(output_dir, weight_map, scale_name)
        if weight_dtype != "F8_E4M3":
            raise RuntimeError(f"Expected FP8 E4M3 weight for {weight_name}, got {weight_dtype}")
        if len(scale_shape) != 2:
            raise RuntimeError(f"Expected 2D scale for {scale_name}, got {scale_shape}")

        inferred_format = infer_qkv_quant_format(weight_shape, scale_shape, scale_dtype)
        if inferred_format != expected_format:
            raise RuntimeError(
                f"{weight_name} is {inferred_format}, but output format is {expected_format}"
            )
        prefixes_by_format[inferred_format].append(qkv_prefix(weight_name))
    return {name: prefixes for name, prefixes in prefixes_by_format.items() if prefixes}


def update_qkv_quant_metadata(
    output_dir: Path,
    weight_map: dict[str, str],
    specs: dict[str, dict[str, int]],
    output_format: str,
    tp_size: int,
    include_mtp: bool,
) -> int:
    if not specs:
        return 0

    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Cannot write QKV quantization metadata; missing {config_path}")

    with config_path.open() as f:
        config = json.load(f)

    quant_config = config.get("quantization_config")
    if quant_config is None:
        quant_config = {}
        config["quantization_config"] = quant_config
    if not isinstance(quant_config, dict):
        raise RuntimeError("config.json quantization_config must be an object")

    prefixes_by_format = detect_output_qkv_formats(output_dir, weight_map, specs, output_format)
    qkv_quantized_layers: dict[str, dict] = {}
    for format_name, prefixes in prefixes_by_format.items():
        entry = qkv_quantized_layer_entry(format_name)
        for prefix in prefixes:
            qkv_quantized_layers[prefix] = dict(entry)

    config_groups = quant_config.get("config_groups")
    if not isinstance(config_groups, dict):
        config_groups = {}
    config_groups = dict(config_groups)
    config_groups.pop("group_fp8_qkv", None)
    config_groups.pop("group_mxfp8_qkv", None)
    for format_name, prefixes in prefixes_by_format.items():
        group_name = "group_fp8_qkv" if format_name == "fp8-pb" else "group_mxfp8_qkv"
        config_groups[group_name] = qkv_config_group(format_name, prefixes)
    quant_config["config_groups"] = dict(sorted(config_groups.items()))

    quantized_layers = quant_config.get("quantized_layers")
    if not isinstance(quantized_layers, dict):
        quantized_layers = {}
    quantized_layers = {
        prefix: value
        for prefix, value in quantized_layers.items()
        if not is_qkv_prefix(prefix)
    }
    quantized_layers.update(qkv_quantized_layers)
    quant_config["quantized_layers"] = dict(sorted(quantized_layers.items()))
    quant_config["qkv_quantized_layers"] = dict(sorted(qkv_quantized_layers.items()))

    formats = {}
    for format_name, prefixes in prefixes_by_format.items():
        format_metadata = dict(qkv_quantized_layer_entry(format_name))
        if format_name == "mxfp8":
            format_metadata["weight_block_size"] = MXFP8_WEIGHT_BLOCK_SIZE
        format_metadata["targets"] = prefixes
        formats[format_name] = format_metadata

    target_prefixes = [
        prefix
        for format_name in sorted(prefixes_by_format)
        for prefix in prefixes_by_format[format_name]
    ]
    quant_config["attention_projection_quantization"] = {
        "format": output_format,
        "formats": formats,
        "layout": config.get("attention_projection_layout", "fused_qkv"),
        "normalized_qkv_layout": "q_then_k_then_v",
        "scale_tensor_suffix": ".weight_scale_inv",
        "source_qkv_tp_size": tp_size,
        "targets": target_prefixes,
        "includes_mtp_qkv": include_mtp,
    }

    with config_path.open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    return len(target_prefixes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="MiMo model ID or local FP8 checkpoint directory.")
    parser.add_argument("-o", "--output-dir", type=Path, help="Output checkpoint directory.")
    parser.add_argument("--model-dir", type=Path, help="Override model directory instead of resolving model_id.")
    parser.add_argument("--ignore-index", action="store_true", help="Scan local safetensors instead of using the index.")
    parser.add_argument("--mimo-qkv-tp-size", type=int, default=4,
                        help="Tensor-parallel packing size used by MiMo fused QKV tensors.")
    parser.add_argument("--block-m", type=int, default=FP8_PB_WEIGHT_BLOCK_SIZE[0],
                        help="FP8 weight scale block height. MiMo uses 128.")
    parser.add_argument("--block-n", type=int, default=FP8_PB_WEIGHT_BLOCK_SIZE[1],
                        help="FP8 weight scale block width. MiMo uses 128.")
    parser.add_argument("--output-format", choices=("fp8-pb", "mxfp8"), default="fp8-pb",
                        help="Output QKV format. fp8-pb preserves/requants MiMo 128x128 FP8; "
                             "mxfp8 converts selected QKV tensors through BF16 with ModelOpt MXFP8.")
    parser.add_argument("--on-inexact", choices=("error", "skip", "requantize"), default="error",
                        help="What to do when exact FP8 deinterleaving is impossible in fp8-pb mode.")
    parser.add_argument("--no-mtp", action="store_true",
                        help="Do not also normalize model.mtp.* QKV tensors.")
    parser.add_argument("--unchanged-shards", choices=("symlink", "hardlink", "copy"), default="symlink",
                        help="How to place safetensors shards that do not need rewriting.")
    parser.add_argument("--dry-run", action="store_true", help="Report which layers can be rewritten exactly.")
    parser.add_argument("--force", action="store_true", help="Allow replacing contents of an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mimo_qkv_tp_size <= 0:
        raise SystemExit("--mimo-qkv-tp-size must be positive")
    if args.block_m <= 0:
        raise SystemExit("--block-m must be positive")
    if args.block_n <= 0:
        raise SystemExit("--block-n must be positive")

    model_dir = args.model_dir if args.model_dir is not None else resolve_model_dir(args.model_id)
    with (model_dir / "config.json").open() as f:
        config = json.load(f)
    source_weight_map, metadata = load_index(model_dir, args.ignore_index)
    output_weight_map = dict(source_weight_map)
    specs = build_mimo_qkv_specs(config, source_weight_map, args.mimo_qkv_tp_size)
    mtp_specs = (
        {}
        if args.no_mtp
        else build_mimo_mtp_qkv_specs(config, model_dir, source_weight_map, args.mimo_qkv_tp_size)
    )
    all_specs = {**specs, **mtp_specs}
    exact, inexact = summarize_targets(all_specs, args.block_m)
    block_size = (args.block_m, args.block_n)

    print(f"Source: {model_dir}")
    print(f"Output format: {args.output_format}")
    print(f"Main MiMo QKV tensors: {len(specs)}")
    print(f"MTP MiMo QKV tensors: {len(mtp_specs)}")
    print(f"Exactly FP8-deinterleavable: {len(exact)}")
    if exact:
        print(f"  tensors: {format_qkv_names(list(exact))}")
    print(f"Inexact with {args.block_m}-row FP8 scale blocks: {len(inexact)}")
    if inexact:
        print(f"  tensors: {format_qkv_names(list(inexact))}")
        first_name = next(iter(inexact))
        print(f"  first reason: {format_qkv_names([first_name])}: {inexact[first_name]}")
        if args.output_format == "mxfp8":
            print("  action: dequantize -> deinterleave -> ModelOpt MXFP8")
        elif args.on_inexact == "requantize":
            print("  action: dequantize -> deinterleave -> requantize these tensors")
    if not all_specs:
        raise SystemExit("No MiMo QKV tensors found in the source checkpoint.")

    if args.dry_run:
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --dry-run is set")

    target_weight_specs: dict[str, dict[str, int]] = {}
    target_scale_specs: dict[str, dict[str, int]] = {}
    requant_weight_specs: dict[str, dict[str, int]] = {}
    requant_scale_names: set[str] = set()
    mxfp8_qkv_specs: dict[str, dict[str, int]] = {}
    mxfp8_scale_names: set[str] = set()
    missing_scales = []
    if args.output_format == "mxfp8":
        mxfp8_qkv_specs = all_specs
        for weight_name in mxfp8_qkv_specs:
            scale_name = qkv_scale_name(weight_name)
            if scale_name not in source_weight_map:
                missing_scales.append(scale_name)
            else:
                output_weight_map[scale_name] = source_weight_map[weight_name]
                mxfp8_scale_names.add(scale_name)
    else:
        if inexact and args.on_inexact == "error":
            raise SystemExit(
                "Refusing to produce a mixed-layout checkpoint. Use --on-inexact requantize "
                "to rewrite all main QKV tensors, or --on-inexact skip to rewrite only exact layers."
            )
        target_weight_specs = exact
        requant_weight_specs = {
            weight_name: all_specs[weight_name]
            for weight_name in inexact
            if args.on_inexact == "requantize"
        }
        requant_scale_names = {qkv_scale_name(weight_name) for weight_name in requant_weight_specs}
        for weight_name, spec in {**target_weight_specs, **requant_weight_specs}.items():
            scale_name = qkv_scale_name(weight_name)
            if scale_name not in source_weight_map:
                missing_scales.append(scale_name)
            elif weight_name in target_weight_specs:
                target_scale_specs[scale_name] = spec
        for weight_name in requant_weight_specs:
            output_weight_map[qkv_scale_name(weight_name)] = source_weight_map[weight_name]
    if missing_scales:
        raise SystemExit(
            f"Missing {len(missing_scales)} FP8 scale sidecar(s); first missing: {missing_scales[0]}"
        )

    output_qkv_specs = (
        mxfp8_qkv_specs
        if args.output_format == "mxfp8"
        else {**target_weight_specs, **requant_weight_specs}
    )
    modified_files = {
        source_weight_map[name]
        for name in [
            *target_weight_specs,
            *target_scale_specs,
            *requant_weight_specs,
            *requant_scale_names,
            *mxfp8_qkv_specs,
            *mxfp8_scale_names,
        ]
    }
    safetensors_files = sorted(set(source_weight_map.values()))

    prepare_output_dir(args.output_dir, args.force)
    copy_sidecars(model_dir, args.output_dir)
    link_or_copy_unindexed_safetensors(
        model_dir,
        args.output_dir,
        indexed_shards=set(safetensors_files),
        mode=args.unchanged_shards,
    )

    total_weights = 0
    total_scales = 0
    total_requantized = 0
    total_mxfp8 = 0
    for file_name in safetensors_files:
        if file_name not in modified_files:
            link_or_copy(model_dir / file_name, args.output_dir / file_name, args.unchanged_shards)
            continue
        weights, scales, requantized, mxfp8 = rewrite_modified_shard(
            model_dir,
            args.output_dir,
            file_name,
            target_weight_specs,
            target_scale_specs,
            requant_weight_specs,
            requant_scale_names,
            mxfp8_qkv_specs,
            mxfp8_scale_names,
            source_weight_map,
            block_size,
        )
        total_weights += weights
        total_scales += scales
        total_requantized += requantized
        total_mxfp8 += mxfp8
        print(
            f"  rewrote {file_name}: {weights} exact QKV weight(s), "
            f"{scales} exact scale sidecar(s), {requantized} requantized QKV weight(s), "
            f"{mxfp8} MXFP8 QKV weight(s)"
        )

    index_metadata = dict(metadata)
    index_metadata["normalized_qkv_format"] = args.output_format
    index_metadata["normalized_qkv_tensors"] = len(output_qkv_specs)
    index_metadata["normalized_qkv_source_tp_size"] = args.mimo_qkv_tp_size
    index_metadata["normalized_qkv_includes_mtp_qkv"] = bool(mtp_specs)
    index_metadata["normalized_qkv_source"] = str(model_dir)
    write_index(args.output_dir, output_weight_map, index_metadata)
    metadata_targets = update_qkv_quant_metadata(
        args.output_dir,
        output_weight_map,
        output_qkv_specs,
        args.output_format,
        args.mimo_qkv_tp_size,
        include_mtp=bool(mtp_specs),
    )
    if args.output_format == "mxfp8":
        print(f"\nWrote ModelOpt MXFP8 MiMo checkpoint to {args.output_dir}")
    else:
        print(f"\nWrote FP8-preserving MiMo checkpoint to {args.output_dir}")
    print(f"  QKV quantization metadata targets: {metadata_targets}")
    print(f"  exact rewritten QKV weights: {total_weights}")
    print(f"  exact rewritten scale sidecars: {total_scales}")
    print(f"  dequantized/requantized QKV weights: {total_requantized}")
    print(f"  FP8 -> BF16 -> ModelOpt MXFP8 QKV weights: {total_mxfp8}")
    if inexact and args.on_inexact == "skip":
        print(f"  unchanged inexact QKV tensors: {len(inexact)}")


if __name__ == "__main__":
    main()
