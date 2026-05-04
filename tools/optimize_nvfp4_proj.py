#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BLOCK_SIZE = 16
FP4_MAX = 6.0
FP8_MAX = 448.0
FP8_MIN = 1.0 / FP8_MAX
E2M1_BOUNDS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
    dtype=torch.float32,
)
FP4_ABS_CODEBOOK = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
MODEL_PREFIXES = (
    "language_model.model.layers",
    "model.layers",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize one Kimi K2.6 expert gate/up projection from packed int4 to NVFP4."
    )
    parser.add_argument("--src-model", required=True, type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--expert", required=True, type=int)
    parser.add_argument("--proj", required=True, choices=["gate_proj", "up_proj"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["static", "activation"], default="static")
    parser.add_argument("--token-cap", type=int, default=2048)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--round-reg", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_snapshot_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "model.safetensors.index.json").exists():
        return path
    refs_main = path / "refs" / "main"
    if refs_main.exists():
        snapshot = path / "snapshots" / refs_main.read_text().strip()
        if (snapshot / "model.safetensors.index.json").exists():
            return snapshot
    snapshots = sorted((path / "snapshots").glob("*"))
    for snapshot in snapshots:
        if (snapshot / "model.safetensors.index.json").exists():
            return snapshot
    raise FileNotFoundError(f"Could not find model.safetensors.index.json under {path}")


def load_index(model_dir: Path) -> dict:
    with open(model_dir / "model.safetensors.index.json") as f:
        return json.load(f)


def load_tensors(model_dir: Path, weight_map: dict[str, str], keys: list[str]) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_shard[weight_map[key]].append(key)

    out: dict[str, torch.Tensor] = {}
    for shard_name, shard_keys in sorted(by_shard.items()):
        with safe_open(str(model_dir / shard_name), framework="pt", device="cpu") as f:
            for key in shard_keys:
                out[key] = f.get_tensor(key)
    return out


def find_proj_prefix(weight_map: dict[str, str], layer: int, expert: int, proj: str) -> str:
    for model_prefix in MODEL_PREFIXES:
        prefix = f"{model_prefix}.{layer}.mlp.experts.{expert}.{proj}"
        if f"{prefix}.weight_packed" in weight_map:
            return prefix
    raise KeyError(
        f"Could not find packed tensor keys for layer={layer}, expert={expert}, proj={proj}"
    )


def unpack_packed_int4(packed: torch.Tensor, out_features: int) -> torch.Tensor:
    shifts = (
        torch.arange(8, device=packed.device, dtype=torch.int64).view(1, 1, 8) * 4
    )
    nibbles = ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & 0xF).reshape(
        packed.shape[0], -1
    )
    nibbles = nibbles[:, :out_features].to(torch.int16)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    return signed.to(torch.float32)


def dequantize_ct_int4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
) -> torch.Tensor:
    rows = int(weight_shape.reshape(-1)[0].item())
    cols = int(weight_shape.reshape(-1)[1].item())
    q = unpack_packed_int4(weight_packed, cols)
    q = q[:rows, :cols]
    group_size = cols // weight_scale.shape[1]
    scale = weight_scale.to(torch.float32).repeat_interleave(group_size, dim=1)
    return q * scale[:, :cols]


def reduce_block_amax(weight: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    return weight.reshape(weight.shape[0], -1, block_size).abs().amax(dim=-1)


def project_block_scales(block_scales: torch.Tensor) -> torch.Tensor:
    projected = block_scales.clamp(min=FP8_MIN).to(torch.float8_e4m3fn)
    projected_f32 = projected.to(torch.float32)
    projected_f32[projected_f32 == 0] = FP8_MIN
    return projected_f32.to(torch.float8_e4m3fn)


def baseline_scales(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = weight.abs().amax().float().clamp_min(FP8_MIN)
    scale2 = (max_abs / (FP4_MAX * FP8_MAX)).reshape(())
    per_block = reduce_block_amax(weight)
    block_scale = per_block / (FP4_MAX * scale2)
    block_scale[per_block == 0] = 1.0
    block_scale = block_scale.clamp(min=FP8_MIN)
    return scale2, block_scale


def safe_logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def fp4_interval(
    normalized_abs: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codebook = FP4_ABS_CODEBOOK.to(device)
    hi_idx = torch.searchsorted(codebook, normalized_abs, right=False)
    hi_idx = hi_idx.clamp(max=codebook.numel() - 1)
    lo_idx = (hi_idx - 1).clamp(min=0)
    lo = codebook[lo_idx]
    hi = codebook[hi_idx]
    span = hi - lo
    return lo, hi, span


def nearest_fp4_abs(normalized_abs: torch.Tensor, device: torch.device) -> torch.Tensor:
    bounds = E2M1_BOUNDS.to(device)
    codebook = FP4_ABS_CODEBOOK.to(device)
    ordinals = torch.searchsorted(bounds, normalized_abs, out_int32=True)
    odd_bounds = bounds[[1, 3, 5]]
    equals_odd_bounds = torch.any(
        normalized_abs.unsqueeze(-1) == odd_bounds,
        dim=-1,
    ).to(ordinals.dtype)
    return codebook[(ordinals + equals_odd_bounds).clamp(max=codebook.numel() - 1)]


def build_scale_blocks(scale2: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    return block_scale.to(torch.float32) * scale2.to(torch.float32)


def soft_quantize_nvfp4(
    weight: torch.Tensor,
    scale2: torch.Tensor,
    block_scale: torch.Tensor,
    round_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_blocks = build_scale_blocks(scale2, block_scale).unsqueeze(-1)
    normalized = weight.reshape(weight.shape[0], -1, BLOCK_SIZE) / scale_blocks
    normalized_abs = normalized.abs()
    lo, hi, span = fp4_interval(normalized_abs, weight.device)
    base_ratio = torch.where(
        span > 0,
        ((normalized_abs - lo) / span).clamp(0.0, 1.0),
        torch.zeros_like(normalized_abs),
    )
    logits = safe_logit(base_ratio) + round_bias.reshape_as(base_ratio)
    alpha = torch.where(span > 0, torch.sigmoid(logits), torch.zeros_like(logits))
    q_abs = lo + alpha * span
    q_soft = torch.sign(normalized) * q_abs
    dequant = (q_soft * scale_blocks).reshape_as(weight)
    return dequant, alpha


def hard_quantize_nvfp4(
    weight: torch.Tensor,
    scale2: torch.Tensor,
    block_scale: torch.Tensor,
    round_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_fp8 = project_block_scales(block_scale)
    scale_blocks = build_scale_blocks(scale2, scale_fp8).unsqueeze(-1)
    normalized = weight.reshape(weight.shape[0], -1, BLOCK_SIZE) / scale_blocks
    normalized_abs = normalized.abs()
    lo, hi, span = fp4_interval(normalized_abs, weight.device)
    if round_bias is None:
        q_abs = nearest_fp4_abs(normalized_abs, weight.device)
    else:
        base_ratio = torch.where(
            span > 0,
            ((normalized_abs - lo) / span).clamp(0.0, 1.0),
            torch.zeros_like(normalized_abs),
        )
        logits = safe_logit(base_ratio) + round_bias.reshape_as(base_ratio)
        choose_hi = torch.sigmoid(logits) >= 0.5
        q_abs = torch.where(span > 0, torch.where(choose_hi, hi, lo), lo)
    q_hard = torch.sign(normalized) * q_abs
    dequant = (q_hard * scale_blocks).reshape_as(weight)
    packed = pack_fp4(q_hard.reshape_as(weight))
    return dequant, packed


def pack_fp4(weight: torch.Tensor) -> torch.Tensor:
    bounds = E2M1_BOUNDS.to(weight.device)
    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()
    ordinals = torch.searchsorted(bounds, weight_abs, out_int32=True).to(torch.uint8)
    odd_bounds = bounds[[1, 3, 5]]
    equals_odd_bounds = torch.any(
        weight_abs.unsqueeze(-1) == odd_bounds,
        dim=-1,
    ).to(torch.uint8)
    q = (sign_bit << 3) + ordinals + equals_odd_bounds
    return ((q[..., 1::2] << 4) | q[..., 0::2]).contiguous()


def collect_routed_tokens(
    dump_dir: Path,
    layer: int,
    expert: int,
    token_cap: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    files = sorted(dump_dir.rglob(f"layer_{layer}/pass_*.pt"))
    if not files:
        raise FileNotFoundError(f"No dump files found for layer_{layer} under {dump_dir}")

    selected = []
    total_tokens = 0
    matched_tokens = 0
    for file_path in files:
        payload = torch.load(file_path, map_location="cpu", weights_only=False)
        hidden_states = payload["hidden_states"].to(torch.float32)
        topk_ids = payload["topk_ids"].to(torch.int32)
        mask = (topk_ids == expert).any(dim=1)
        total_tokens += topk_ids.shape[0]
        if mask.any():
            picked = hidden_states[mask]
            selected.append(picked)
            matched_tokens += int(mask.sum().item())
        if sum(t.shape[0] for t in selected) >= token_cap:
            break

    if not selected:
        raise RuntimeError(
            f"No tokens routed to expert={expert} were found in {dump_dir} for layer={layer}"
        )

    x = torch.cat(selected, dim=0)[:token_cap].contiguous()
    return x, {
        "dump_files_scanned": len(files),
        "tokens_seen": total_tokens,
        "tokens_routed_to_expert": matched_tokens,
        "tokens_used": int(x.shape[0]),
    }


def split_train_val(x: torch.Tensor, train_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] < 2:
        return x, x
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=generator)
    train_count = max(1, min(x.shape[0] - 1, int(math.floor(x.shape[0] * train_fraction))))
    train_idx = perm[:train_count]
    val_idx = perm[train_count:]
    if val_idx.numel() == 0:
        val_idx = train_idx[: min(train_idx.numel(), 128)]
    return x[train_idx].contiguous(), x[val_idx].contiguous()


def metrics_path_for(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_suffix(".metrics.json")
    return out_path.parent / f"{out_path.name}.metrics.json"


def optimize_static(
    teacher_weight: torch.Tensor,
    *,
    steps: int,
    eval_every: int,
    lr: float,
    round_reg: float,
) -> tuple[dict, dict]:
    baseline_scale2_init, baseline_block_scale_init = baseline_scales(teacher_weight)
    baseline_weight, _ = hard_quantize_nvfp4(
        teacher_weight,
        baseline_scale2_init,
        baseline_block_scale_init,
        round_bias=None,
    )
    baseline_weight_mse = torch.mean((baseline_weight - teacher_weight) ** 2).item()

    log_scale2 = torch.nn.Parameter(torch.log(baseline_scale2_init.detach().clone()))
    log_block_scale = torch.nn.Parameter(
        torch.log(baseline_block_scale_init.detach().clone().clamp_min(FP8_MIN))
    )
    round_bias = torch.nn.Parameter(torch.zeros_like(teacher_weight))
    optimizer = torch.optim.Adam([log_scale2, log_block_scale, round_bias], lr=lr)

    best_state = None
    best_weight_mse = float("inf")

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        scale2 = torch.exp(log_scale2).clamp_min(FP8_MIN / FP4_MAX)
        block_scale = torch.exp(log_block_scale).clamp(min=FP8_MIN, max=FP8_MAX)
        soft_weight, alpha = soft_quantize_nvfp4(
            teacher_weight,
            scale2=scale2,
            block_scale=block_scale,
            round_bias=round_bias,
        )
        recon_loss = torch.mean((soft_weight - teacher_weight) ** 2)
        reg_loss = torch.mean(alpha * (1.0 - alpha))
        loss = recon_loss + round_reg * reg_loss
        loss.backward()
        optimizer.step()

        should_eval = step == 1 or step == steps or step % eval_every == 0
        if not should_eval:
            continue

        with torch.no_grad():
            scale2_eval = torch.exp(log_scale2).clamp_min(FP8_MIN / FP4_MAX)
            block_scale_eval = torch.exp(log_block_scale).clamp(min=FP8_MIN, max=FP8_MAX)
            hard_weight, packed = hard_quantize_nvfp4(
                teacher_weight,
                scale2=scale2_eval,
                block_scale=block_scale_eval,
                round_bias=round_bias,
            )
            weight_mse = torch.mean((hard_weight - teacher_weight) ** 2).item()
            if weight_mse < best_weight_mse:
                best_weight_mse = weight_mse
                best_state = {
                    "step": step,
                    "scale2": scale2_eval.detach().cpu().reshape(()),
                    "block_scale": project_block_scales(block_scale_eval).detach().cpu(),
                    "packed": packed.detach().cpu(),
                    "weight_mse": weight_mse,
                }
                print(f"[step {step:04d}] improved weight_mse={weight_mse:.6e}")

    if best_state is None:
        raise RuntimeError("Optimization did not produce any evaluated state")

    metrics = {
        "baseline": {
            "weight_mse": baseline_weight_mse,
        },
        "optimized": {
            "step": best_state["step"],
            "weight_mse": best_state["weight_mse"],
        },
    }
    return best_state, metrics


def optimize_activation(
    teacher_weight: torch.Tensor,
    *,
    dump_dir: Path,
    layer: int,
    expert: int,
    token_cap: int,
    train_fraction: float,
    batch_size: int,
    steps: int,
    eval_every: int,
    lr: float,
    round_reg: float,
    seed: int,
) -> tuple[dict, dict]:
    x_cpu, dump_stats = collect_routed_tokens(
        dump_dir,
        layer=layer,
        expert=expert,
        token_cap=token_cap,
    )
    if x_cpu.shape[1] != teacher_weight.shape[1]:
        raise RuntimeError(
            f"Activation width {x_cpu.shape[1]} does not match weight input dim {teacher_weight.shape[1]}"
        )
    train_x_cpu, val_x_cpu = split_train_val(
        x_cpu,
        train_fraction=train_fraction,
        seed=seed,
    )

    device = teacher_weight.device
    train_x = train_x_cpu.to(device=device, dtype=torch.float32)
    val_x = val_x_cpu.to(device=device, dtype=torch.float32)
    train_y = train_x @ teacher_weight.t()
    val_y = val_x @ teacher_weight.t()

    baseline_scale2_init, baseline_block_scale_init = baseline_scales(teacher_weight)
    baseline_weight, _ = hard_quantize_nvfp4(
        teacher_weight,
        baseline_scale2_init,
        baseline_block_scale_init,
        round_bias=None,
    )
    baseline_train_mse = torch.mean((train_x @ baseline_weight.t() - train_y) ** 2).item()
    baseline_val_mse = torch.mean((val_x @ baseline_weight.t() - val_y) ** 2).item()
    baseline_weight_mse = torch.mean((baseline_weight - teacher_weight) ** 2).item()

    log_scale2 = torch.nn.Parameter(torch.log(baseline_scale2_init.detach().clone()))
    log_block_scale = torch.nn.Parameter(
        torch.log(baseline_block_scale_init.detach().clone().clamp_min(FP8_MIN))
    )
    round_bias = torch.nn.Parameter(torch.zeros_like(teacher_weight))
    optimizer = torch.optim.Adam([log_scale2, log_block_scale, round_bias], lr=lr)

    best_state = None
    best_val_mse = float("inf")
    batch_size = min(batch_size, train_x.shape[0])

    for step in range(1, steps + 1):
        batch_idx = torch.randperm(train_x.shape[0], device=device)[:batch_size]
        x_batch = train_x[batch_idx]
        y_batch = train_y[batch_idx]

        optimizer.zero_grad(set_to_none=True)
        scale2 = torch.exp(log_scale2).clamp_min(FP8_MIN / FP4_MAX)
        block_scale = torch.exp(log_block_scale).clamp(min=FP8_MIN, max=FP8_MAX)
        soft_weight, alpha = soft_quantize_nvfp4(
            teacher_weight,
            scale2=scale2,
            block_scale=block_scale,
            round_bias=round_bias,
        )
        pred = x_batch @ soft_weight.t()
        recon_loss = torch.mean((pred - y_batch) ** 2)
        reg_loss = torch.mean(alpha * (1.0 - alpha))
        loss = recon_loss + round_reg * reg_loss
        loss.backward()
        optimizer.step()

        should_eval = step == 1 or step == steps or step % eval_every == 0
        if not should_eval:
            continue

        with torch.no_grad():
            scale2_eval = torch.exp(log_scale2).clamp_min(FP8_MIN / FP4_MAX)
            block_scale_eval = torch.exp(log_block_scale).clamp(min=FP8_MIN, max=FP8_MAX)
            hard_weight, packed = hard_quantize_nvfp4(
                teacher_weight,
                scale2=scale2_eval,
                block_scale=block_scale_eval,
                round_bias=round_bias,
            )
            val_mse = torch.mean((val_x @ hard_weight.t() - val_y) ** 2).item()
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = {
                    "step": step,
                    "scale2": scale2_eval.detach().cpu().reshape(()),
                    "block_scale": project_block_scales(block_scale_eval).detach().cpu(),
                    "packed": packed.detach().cpu(),
                    "weight_mse": torch.mean((hard_weight - teacher_weight) ** 2).item(),
                    "train_mse": torch.mean((train_x @ hard_weight.t() - train_y) ** 2).item(),
                    "val_mse": val_mse,
                    "tokens": dump_stats,
                    "train_tokens": int(train_x.shape[0]),
                    "val_tokens": int(val_x.shape[0]),
                }
                print(
                    f"[step {step:04d}] improved val_mse={val_mse:.6e} "
                    f"train_mse={best_state['train_mse']:.6e}"
                )

    if best_state is None:
        raise RuntimeError("Optimization did not produce any evaluated state")

    metrics = {
        "tokens": dump_stats,
        "train_tokens": int(train_x.shape[0]),
        "val_tokens": int(val_x.shape[0]),
        "baseline": {
            "train_mse": baseline_train_mse,
            "val_mse": baseline_val_mse,
            "weight_mse": baseline_weight_mse,
        },
        "optimized": {
            "step": best_state["step"],
            "train_mse": best_state["train_mse"],
            "val_mse": best_state["val_mse"],
            "weight_mse": best_state["weight_mse"],
        },
    }
    return best_state, metrics


def main() -> None:
    args = parse_args()
    if args.proj == "down_proj":
        raise SystemExit("down_proj is intentionally out of scope for this prototype")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but CUDA is not available")

    model_dir = resolve_snapshot_dir(args.src_model)
    index = load_index(model_dir)
    weight_map = index["weight_map"]
    prefix = find_proj_prefix(weight_map, args.layer, args.expert, args.proj)
    keys = [f"{prefix}.weight_packed", f"{prefix}.weight_scale", f"{prefix}.weight_shape"]
    tensors = load_tensors(model_dir, weight_map, keys)
    teacher_weight = dequantize_ct_int4(
        tensors[f"{prefix}.weight_packed"],
        tensors[f"{prefix}.weight_scale"],
        tensors[f"{prefix}.weight_shape"],
    )
    teacher_weight = teacher_weight.to(device=device, dtype=torch.float32)

    if args.mode == "activation":
        if args.dump_dir is None:
            raise SystemExit("--dump-dir is required when --mode activation")
        best_state, objective_metrics = optimize_activation(
            teacher_weight,
            dump_dir=args.dump_dir,
            layer=args.layer,
            expert=args.expert,
            token_cap=args.token_cap,
            train_fraction=args.train_fraction,
            batch_size=args.batch_size,
            steps=args.steps,
            eval_every=args.eval_every,
            lr=args.lr,
            round_reg=args.round_reg,
            seed=args.seed,
        )
    else:
        best_state, objective_metrics = optimize_static(
            teacher_weight,
            steps=args.steps,
            eval_every=args.eval_every,
            lr=args.lr,
            round_reg=args.round_reg,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            f"{prefix}.weight": best_state["packed"],
            f"{prefix}.weight_scale": best_state["block_scale"],
            f"{prefix}.weight_scale_2": best_state["scale2"],
        },
        str(args.out),
    )

    metrics = {
        "source_model": str(model_dir),
        "layer": args.layer,
        "expert": args.expert,
        "proj": args.proj,
        "mode": args.mode,
        "prefix": prefix,
        "device": str(device),
        "seed": args.seed,
    }
    if args.dump_dir is not None:
        metrics["dump_dir"] = str(args.dump_dir.resolve())
    metrics.update(objective_metrics)
    metrics_path = metrics_path_for(args.out)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote optimized shard to {args.out}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
