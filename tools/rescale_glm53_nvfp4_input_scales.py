#!/usr/bin/env python3
"""Create a sealed multiplicative variant of GLM routed-expert input scales."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


SCALE_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.input_scale$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scales", type=Path, required=True)
    parser.add_argument("--output-scales", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gate-up-factor", type=float, required=True)
    parser.add_argument("--down-factor", type=float, required=True)
    args = parser.parse_args(argv)

    for label, factor in (
        ("gate/up", args.gate_up_factor),
        ("down", args.down_factor),
    ):
        if not math.isfinite(factor) or factor <= 0:
            parser.error(f"{label} factor must be finite and positive")
    for destination in (args.output_scales, args.receipt):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite: {destination}")

    source = load_file(args.source_scales)
    output: dict[str, torch.Tensor] = {}
    projections: dict[tuple[int, int], dict[str, float]] = {}
    for key, tensor in source.items():
        match = SCALE_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"unexpected input-scale key: {key}")
        if tensor.dtype != torch.float32 or tensor.numel() != 1:
            raise ValueError(f"input scale is not scalar F32: {key}")
        layer, expert, projection = match.groups()
        value = float(tensor.item())
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"input scale is not finite and positive: {key}")
        factor = args.down_factor if projection == "down_proj" else args.gate_up_factor
        scaled = tensor * factor
        scaled_value = float(scaled.item())
        if not math.isfinite(scaled_value) or scaled_value <= 0:
            raise ValueError(f"scaled input scale is invalid: {key}")
        output[key] = scaled
        projections.setdefault((int(layer), int(expert)), {})[projection] = scaled_value

    for identity, values in projections.items():
        if set(values) != {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"incomplete expert input scales: {identity}")
        if values["gate_proj"] != values["up_proj"]:
            raise ValueError(f"gate/up scales are not tied: {identity}")

    args.output_scales.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_scales.with_name(f".{args.output_scales.name}.incomplete")
    save_file(dict(sorted(output.items())), temporary)
    os.replace(temporary, args.output_scales)

    receipt = {
        "schema": "quant-toolkit.glm53-nvfp4-input-scale-rescale.v1",
        "source": {
            "path": str(args.source_scales.resolve()),
            "bytes": args.source_scales.stat().st_size,
            "sha256": _sha256(args.source_scales),
        },
        "method": {
            "gate_up_factor": args.gate_up_factor,
            "down_factor": args.down_factor,
            "operation": "float32 scalar multiplication",
        },
        "topology": {
            "experts": len(projections),
            "input_scale_tensors": len(output),
            "gate_up_tied": True,
        },
        "output": {
            "path": str(args.output_scales.resolve()),
            "bytes": args.output_scales.stat().st_size,
            "sha256": _sha256(args.output_scales),
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = args.receipt.with_name(f".{args.receipt.name}.incomplete")
    temporary_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_receipt, args.receipt)
    print(json.dumps({"event": "nvfp4_input_scales_rescaled", **receipt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
