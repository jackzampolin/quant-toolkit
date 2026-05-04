#!/usr/bin/env python3
"""Validate MiMo fused-QKV FP8 dequant/deinterleave against rank-local FP8 matmul."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dequantize_fp8 import build_mimo_qkv_specs, build_weight_map_from_safetensors, _ceil_div


def resolve_model_path(model: str) -> Path:
    path = Path(model)
    if path.exists():
        return path
    return Path(snapshot_download(model, local_files_only=True))


def load_weight_map(model_dir: Path, ignore_index: bool) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not ignore_index and index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    return build_weight_map_from_safetensors(model_dir)


def load_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model_dir / weight_map[name], framework="pt") as f:
        return f.get_tensor(name)


def load_triton_fp8():
    from transformers_compat import ensure_mimo_transformers_compat

    ensure_mimo_transformers_compat()
    from transformers.integrations import finegrained_fp8

    finegrained_fp8._load_triton_kernel()
    return finegrained_fp8


def dequantize_activation(qx: torch.Tensor, scales: torch.Tensor, block_k: int) -> torch.Tensor:
    expanded = scales.to(torch.float32).repeat_interleave(block_k, dim=-1)
    expanded = expanded[..., : qx.shape[-1]]
    return qx.to(torch.float32) * expanded


def rank_local_from_canonical(
    canonical: torch.Tensor,
    spec: dict[str, int],
    rank: int,
) -> torch.Tensor:
    tp_size = spec["tp_size"]
    q_shard = spec["q_size"] // tp_size
    k_shard = spec["k_size"] // tp_size
    v_shard = spec["v_size"] // tp_size

    q = canonical.narrow(0, rank * q_shard, q_shard)
    k = canonical.narrow(0, spec["q_size"] + rank * k_shard, k_shard)
    v = canonical.narrow(0, spec["q_size"] + spec["k_size"] + rank * v_shard, v_shard)
    return torch.cat([q, k, v], dim=0).contiguous()


def _cosine_summary(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    flat_cos = F.cosine_similarity(actual_f.flatten(), expected_f.flatten(), dim=0).item()
    row_cos = F.cosine_similarity(actual_f, expected_f, dim=-1)
    return flat_cos, row_cos.mean().item(), row_cos.min().item()


def summarize_diff(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
    segments: dict[str, tuple[int, int]] | None = None,
) -> bool:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-6)
    rel = diff / denom
    flat_cos, row_cos_mean, row_cos_min = _cosine_summary(actual, expected)
    print(
        f"{name}: max_abs={diff.max().item():.6g}, mean_abs={diff.mean().item():.6g}, "
        f"p99_abs={diff.flatten().quantile(0.99).item():.6g}, "
        f"max_rel={rel.max().item():.6g}, mean_rel={rel.mean().item():.6g}, "
        f"cos={flat_cos:.8f}, row_cos_mean={row_cos_mean:.8f}, row_cos_min={row_cos_min:.8f}"
    )
    if segments:
        segment_parts = []
        for segment_name, (start, length) in segments.items():
            actual_segment = actual.narrow(-1, start, length)
            expected_segment = expected.narrow(-1, start, length)
            segment_cos, segment_row_mean, segment_row_min = _cosine_summary(actual_segment, expected_segment)
            segment_parts.append(
                f"{segment_name}:cos={segment_cos:.8f},row_mean={segment_row_mean:.8f},row_min={segment_row_min:.8f}"
            )
        print(f"{name} segments: " + "; ".join(segment_parts))
    return torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare original MiMo TP-local FP8 QKV matmul to normalized BF16 QKV matmul."
    )
    parser.add_argument("--source-model", default="XiaomiMiMo/MiMo-V2.5",
                        help="Original FP8 model ID or local directory.")
    parser.add_argument("--bf16-model", required=True,
                        help="Dequantized/deinterleaved BF16 model directory.")
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer index whose self_attn.qkv_proj.weight to validate.")
    parser.add_argument("--tp-size", type=int, default=4,
                        help="Virtual TP size used by the source fused-QKV checkpoint.")
    parser.add_argument("--tokens", type=int, default=16,
                        help="Number of activation rows to test.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--ignore-source-index", action="store_true")
    parser.add_argument("--ignore-bf16-index", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton FP8 kernel, but torch.cuda.is_available() is false")

    source_dir = resolve_model_path(args.source_model)
    bf16_dir = resolve_model_path(args.bf16_model)
    source_weight_map = load_weight_map(source_dir, args.ignore_source_index)
    bf16_weight_map = load_weight_map(bf16_dir, args.ignore_bf16_index)

    with open(source_dir / "config.json") as f:
        config = json.load(f)
    specs = build_mimo_qkv_specs(config, source_weight_map, args.tp_size)

    weight_name = f"model.layers.{args.layer}.self_attn.qkv_proj.weight"
    scale_name = f"{weight_name}_scale_inv"
    if weight_name not in specs:
        raise KeyError(f"{weight_name} is not a MiMo fused-QKV tensor in the source weight map")
    if weight_name not in bf16_weight_map:
        raise KeyError(f"{weight_name} is missing from BF16 model")

    spec = specs[weight_name]
    tp_size = spec["tp_size"]
    q_shard = spec["q_size"] // tp_size
    k_shard = spec["k_size"] // tp_size
    v_shard = spec["v_size"] // tp_size
    chunk_rows = q_shard + k_shard + v_shard
    scale_rows_per_chunk = _ceil_div(chunk_rows, 128)

    print(f"Source: {source_dir}")
    print(f"BF16:   {bf16_dir}")
    print(f"Layer:  {args.layer}")
    print(f"Shard rows: q={q_shard}, k={k_shard}, v={v_shard}, fused={chunk_rows}")
    print(f"Scale rows per virtual rank: {scale_rows_per_chunk}")

    original_weight = load_tensor(source_dir, source_weight_map, weight_name)
    original_scale = load_tensor(source_dir, source_weight_map, scale_name)
    bf16_weight = load_tensor(bf16_dir, bf16_weight_map, weight_name)

    expected_weight_shape = (chunk_rows * tp_size, original_weight.shape[1])
    expected_scale_shape = (scale_rows_per_chunk * tp_size, _ceil_div(original_weight.shape[1], 128))
    if tuple(original_weight.shape) != expected_weight_shape:
        raise ValueError(f"Unexpected source weight shape {tuple(original_weight.shape)} != {expected_weight_shape}")
    if tuple(original_scale.shape) != expected_scale_shape:
        raise ValueError(f"Unexpected source scale shape {tuple(original_scale.shape)} != {expected_scale_shape}")
    if tuple(bf16_weight.shape) != expected_weight_shape:
        raise ValueError(f"Unexpected BF16 weight shape {tuple(bf16_weight.shape)} != {expected_weight_shape}")

    finegrained_fp8 = load_triton_fp8()
    block_size = [128, 128]
    hidden_size = original_weight.shape[1]

    torch.manual_seed(args.seed)
    activation = torch.randn(args.tokens, hidden_size, dtype=torch.bfloat16, device=args.device)
    qactivation, activation_scale = finegrained_fp8.triton_fp8_act_quant(activation, block_size[1])
    activation_dequant = dequantize_activation(qactivation, activation_scale, block_size[1])

    all_ok = True
    for rank in range(tp_size):
        weight_chunk = original_weight.narrow(0, rank * chunk_rows, chunk_rows).contiguous().to(args.device)
        scale_chunk = original_scale.narrow(0, rank * scale_rows_per_chunk, scale_rows_per_chunk)
        scale_chunk = scale_chunk.contiguous().to(args.device)
        bf16_local = rank_local_from_canonical(bf16_weight, spec, rank).to(args.device)

        fp8_out = finegrained_fp8.w8a8_fp8_matmul(
            qactivation,
            weight_chunk,
            activation_scale,
            scale_chunk,
            block_size,
            output_dtype=torch.float32,
        )
        bf16_out = activation_dequant @ bf16_local.float().T
        segments = {
            "q": (0, q_shard),
            "k": (q_shard, k_shard),
            "v": (q_shard + k_shard, v_shard),
        }
        ok = summarize_diff(f"rank {rank}", fp8_out, bf16_out, args.rtol, args.atol, segments=segments)
        all_ok = all_ok and ok

    if all_ok:
        print("PASS: BF16 deinterleaved QKV matches rank-local FP8 matmul within tolerance.")
    else:
        raise SystemExit("FAIL: BF16 deinterleaved QKV differs from rank-local FP8 matmul beyond tolerance.")


if __name__ == "__main__":
    main()
