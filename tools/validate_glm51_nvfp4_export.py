#!/usr/bin/env python3
import argparse
import fnmatch
import json
from pathlib import Path

import torch
from safetensors.torch import safe_open


def excluded_by_vllm_legacy(ignore: list[str], prefix: str) -> tuple[bool, str, str]:
    if prefix in ignore:
        return True, "exact", prefix
    for entry in ignore:
        if entry != prefix and entry in prefix:
            return True, "substring", entry
    for entry in ignore:
        if fnmatch.fnmatch(prefix, entry):
            return True, "fnmatch", entry
    return False, "", ""


def load_tensor(ckpt: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = weight_map[key]
    with safe_open(ckpt / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layers", default="0,1,2,3,9,10,11,19,20,21,29,30,77")
    parser.add_argument("--scale-samples", default="3:0,10:17,77:255")
    args = parser.parse_args()

    config_path = args.checkpoint / "config.json"
    index_path = args.checkpoint / "model.safetensors.index.json"
    config = json.load(open(config_path))
    ignore = config.get("quantization_config", {}).get("ignore", [])

    ok = True
    print("IGNORE_CHECK")
    for layer in [int(x) for x in args.layers.split(",") if x]:
        prefix = f"model.layers.{layer}.mlp"
        matched, mode, entry = excluded_by_vllm_legacy(ignore, prefix)
        should_match = layer < int(config.get("first_k_dense_replace", 0) or 0)
        ok = ok and matched == should_match
        print(
            f"{prefix:24s} excluded={matched!s:5s} "
            f"expected={should_match!s:5s} mode={mode or '-':9s} entry={entry or '-'}"
        )

    weight_map = json.load(open(index_path))["weight_map"]
    expert_scale2 = [
        k for k in weight_map
        if ".mlp.experts." in k and k.endswith(".weight_scale_2")
    ]
    expert_input = [
        k for k in weight_map
        if ".mlp.experts." in k and k.endswith(".input_scale")
    ]
    input_count_ok = len(expert_input) == len(expert_scale2)
    ok = ok and input_count_ok
    print("INPUT_SCALE_CHECK")
    print(
        f"expert_weight_scale_2={len(expert_scale2)} "
        f"expert_input_scale={len(expert_input)} equal={input_count_ok}"
    )

    print("W13_SCALE_CHECK")
    for sample in [x for x in args.scale_samples.split(",") if x]:
        layer_s, expert_s = sample.split(":", 1)
        layer = int(layer_s)
        expert = int(expert_s)
        gate_key = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight_scale_2"
        up_key = f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight_scale_2"
        gate = load_tensor(args.checkpoint, weight_map, gate_key)
        up = load_tensor(args.checkpoint, weight_map, up_key)
        equal = torch.equal(gate, up)
        diff = (gate - up).abs().max().item()
        ok = ok and equal
        print(
            f"layer={layer:02d} expert={expert:03d} equal={equal!s:5s} "
            f"max_abs_diff={diff:.10g} gate={float(gate.flatten()[0]):.10g} "
            f"up={float(up.flatten()[0]):.10g}"
        )

    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
