#!/usr/bin/env python3
"""Combine per-rank amax safetensors into one elementwise-max amax file."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default="/tmp/glm51-mtp-amax.rank*.safetensors",
        help="Glob for per-rank amax safetensors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/glm51-mtp-amax.safetensors"),
        help="Merged output safetensors path.",
    )
    parser.add_argument(
        "--allow-missing-keys",
        action="store_true",
        help="Allow ranks to have different keysets and merge the union.",
    )
    return parser.parse_args()


def merge_rank_amaxes(paths: list[Path], allow_missing_keys: bool) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    expected_keys: set[str] | None = None

    for path in paths:
        tensors = {
            name: value.detach().cpu().float()
            for name, value in load_file(str(path)).items()
        }
        keys = set(tensors)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys and not allow_missing_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            details = []
            if missing:
                details.append(f"missing={missing[:5]}")
            if extra:
                details.append(f"extra={extra[:5]}")
            raise ValueError(f"{path} keyset differs from previous ranks: {' '.join(details)}")

        for name, value in tensors.items():
            value = value.squeeze()
            current = merged.get(name)
            if current is None:
                merged[name] = value.clone()
                continue
            if current.shape != value.shape:
                raise ValueError(
                    f"{name} shape mismatch while reading {path}: "
                    f"{tuple(current.shape)} vs {tuple(value.shape)}"
                )
            merged[name] = torch.maximum(current, value)

    return merged


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise FileNotFoundError(f"No files matched {args.input_glob!r}")

    merged = merge_rank_amaxes(paths, args.allow_missing_keys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.clone() for k, v in sorted(merged.items())}, str(args.output))

    print("COMBINE RANK AMAX SUMMARY")
    print(f"  input_files: {len(paths)}")
    print(f"  input_glob: {args.input_glob}")
    print(f"  output_keys: {len(merged)}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
