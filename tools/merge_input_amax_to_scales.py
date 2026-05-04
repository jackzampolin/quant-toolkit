#!/usr/bin/env python3
"""Merge expert input amax checkpoints and emit NVFP4 model-inputscales.

Rules:
  1. Start from the exported checkpoint's input quantizer amaxes.
  2. Merge with another amax checkpoint via elementwise max.
  3. Optionally cap merged amax at `max_multiplier * exported_amax`.
  4. Optionally clamp any remaining spikes to `amax_clamp`.
  5. Tie w1/w3 per expert by taking a shared max, respecting cap/clamp if enabled.
  6. Convert merged amax to NVFP4 `input_scale` via `amax / (6 * 448)`.

The output uses the same key format as `model-inputscales.safetensors`.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT_Q_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w[13])\.input_quantizer$"
)

MAXBOUND = 6.0
NVFP4_ACT_DENOM = MAXBOUND * 448.0


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k).float().cpu() for k in f.keys()}


def sanitize_export_amax(v: torch.Tensor) -> torch.Tensor:
    out = v.clone().float()
    out[out == 0] = MAXBOUND
    out = torch.nan_to_num(out, nan=MAXBOUND)
    clamp_min, clamp_max = torch.finfo(out.dtype).tiny, torch.finfo(out.dtype).max
    return out.clamp(min=clamp_min, max=clamp_max)


def amax_to_input_scale(v: torch.Tensor) -> torch.Tensor:
    out = sanitize_export_amax(v) / NVFP4_ACT_DENOM
    return out.squeeze()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-amax", required=True, type=Path)
    parser.add_argument("--export-amax", required=True, type=Path)
    parser.add_argument("--export-scales", required=True, type=Path)
    parser.add_argument("--output-scales", required=True, type=Path)
    parser.add_argument("--output-amax", type=Path, default=None)
    parser.add_argument("--max-multiplier", type=float, default=10.0)
    parser.add_argument("--amax-clamp", type=float, default=1e4)
    parser.add_argument("--no-cap", action="store_true")
    parser.add_argument("--no-clamp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ours = load_tensors(args.ours_amax)
    export_amax = load_tensors(args.export_amax)
    export_scales = load_tensors(args.export_scales)

    merged_amax_for_scales: dict[str, torch.Tensor] = {}
    merged_amax_full: dict[str, torch.Tensor] = {}
    missing_in_ours = 0
    capped = 0
    clamped = 0
    raised = 0

    def merge_value(q_key: str) -> torch.Tensor:
        nonlocal missing_in_ours, capped, clamped, raised

        if q_key not in export_amax:
            raise KeyError(f"Missing exported amax for {q_key}")

        export_v = sanitize_export_amax(export_amax[q_key]).reshape(-1)[0]
        ours_v = ours.get(q_key)
        if ours_v is None:
            ours_v = export_v.clone()
            missing_in_ours += 1
        else:
            ours_v = ours_v.float().reshape(-1)[0]

        merged_v = torch.maximum(export_v, ours_v)
        if not args.no_cap:
            cap_v = export_v * args.max_multiplier
            if merged_v > cap_v:
                merged_v = cap_v
                capped += 1
        if not args.no_clamp and merged_v > args.amax_clamp:
            merged_v = torch.tensor(args.amax_clamp, dtype=torch.float32)
            clamped += 1
        if merged_v > export_v:
            raised += 1
        return merged_v

    # Use the exported scale shard as the authoritative output keyset.
    for scale_key in sorted(export_scales):
        if not scale_key.endswith(".input_scale"):
            continue
        q_key = scale_key.replace(".input_scale", ".input_quantizer")
        merged_amax_for_scales[q_key] = merge_value(q_key)

    # Preserve the full merged expert-input keyset for output_amax. This can be
    # larger than the export scale shard when the amax sources include experts
    # that are missing from the current checkpoint's model-inputscales file.
    full_input_keys = sorted(
        k
        for k in (set(export_amax) | set(ours))
        if k.endswith(".input_quantizer") and (k in export_amax or k in ours)
    )
    counted_keys = set(merged_amax_for_scales)
    for q_key in full_input_keys:
        if q_key in export_amax:
            if q_key in counted_keys:
                merged_amax_full[q_key] = merged_amax_for_scales[q_key]
            else:
                merged_amax_full[q_key] = merge_value(q_key)
        else:
            merged_amax_full[q_key] = ours[q_key].float().reshape(-1)[0]

    # Tie w1/w3 input amaxes per expert by a shared max that respects cap/clamp if enabled.
    tied_groups = 0
    tied_changed = 0
    grouped: dict[tuple[int, int], dict[str, str]] = defaultdict(dict)
    for q_key in merged_amax_full:
        m = INPUT_Q_RE.match(q_key)
        if m is None:
            continue
        grouped[(int(m.group("layer")), int(m.group("expert")))] [m.group("proj")] = q_key

    for _, proj_map in grouped.items():
        if "w1" not in proj_map or "w3" not in proj_map:
            continue
        k1 = proj_map["w1"]
        k3 = proj_map["w3"]
        shared = torch.maximum(merged_amax_full[k1], merged_amax_full[k3])
        if not args.no_cap and k1 in export_amax and k3 in export_amax:
            export1 = sanitize_export_amax(export_amax[k1]).reshape(-1)[0]
            export3 = sanitize_export_amax(export_amax[k3]).reshape(-1)[0]
            shared_cap = torch.minimum(export1, export3) * args.max_multiplier
            shared = torch.minimum(shared, shared_cap)
        if not args.no_clamp:
            shared = torch.minimum(shared, torch.tensor(args.amax_clamp, dtype=torch.float32))
        tied_groups += 1
        if shared != merged_amax_full[k1] or shared != merged_amax_full[k3]:
            tied_changed += 1
        merged_amax_full[k1] = shared
        merged_amax_full[k3] = shared
        if k1 in merged_amax_for_scales:
            merged_amax_for_scales[k1] = shared
        if k3 in merged_amax_for_scales:
            merged_amax_for_scales[k3] = shared

    output_scales = {
        q_key.replace(".input_quantizer", ".input_scale"): amax_to_input_scale(v)
        for q_key, v in merged_amax_for_scales.items()
    }

    args.output_scales.parent.mkdir(parents=True, exist_ok=True)
    save_file(output_scales, str(args.output_scales))

    if args.output_amax is not None:
        args.output_amax.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.reshape(1).clone() for k, v in merged_amax_full.items()}, str(args.output_amax))

    changed_vs_export = 0
    tied_scale_pairs_equal = 0
    for scale_key, scale_v in output_scales.items():
        if not torch.equal(scale_v, export_scales[scale_key].float().cpu().squeeze()):
            changed_vs_export += 1
        m = INPUT_Q_RE.match(scale_key.replace(".input_scale", ".input_quantizer"))
        if m is None or m.group("proj") != "w1":
            continue
        peer = scale_key.replace(".w1.input_scale", ".w3.input_scale")
        if peer in output_scales and torch.equal(scale_v, output_scales[peer]):
            tied_scale_pairs_equal += 1

    print("MERGE SUMMARY")
    print(f"  output_keys: {len(output_scales)}")
    print(f"  output_amax_keys: {len(merged_amax_full)}")
    print(f"  missing_in_ours_fell_back_to_export: {missing_in_ours}")
    print(f"  cap_enabled: {not args.no_cap}")
    print(f"  clamp_enabled: {not args.no_clamp}")
    print(f"  raised_vs_export_before_tying: {raised}")
    print(f"  hit_10x_cap_before_tying: {capped}")
    print(f"  hit_1e4_clamp_before_tying: {clamped}")
    print(f"  tied_w1_w3_groups: {tied_groups}")
    print(f"  tied_groups_changed: {tied_changed}")
    print(f"  changed_vs_export_scale_file: {changed_vs_export}")
    print(f"  tied_w1_w3_scale_pairs_equal: {tied_scale_pairs_equal}")
    print(f"  output_scales: {args.output_scales}")
    if args.output_amax is not None:
        print(f"  output_amax: {args.output_amax}")


if __name__ == "__main__":
    main()
