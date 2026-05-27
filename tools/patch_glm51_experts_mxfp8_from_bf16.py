#!/usr/bin/env python3
"""Build a GLM-5.1 mixed MXFP8 expert checkpoint from BF16 expert weights.

The intended input is an existing GLM-5.1 mixed checkpoint whose selected MoE
expert layers are already isolated into per-layer expert shards.  This tool
hardlinks all unchanged files into a new output directory and rewrites only the
selected sparse expert layer shards with ModelOpt MXFP8 tensors quantized from a
BF16 source checkpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INDEX_NAME = "model.safetensors.index.json"
MXFP8_GROUP_SIZE = 32
MXFP8_WEIGHT_BLOCK_SIZE = [1, 32]

EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=path.name == INDEX_NAME)
        f.write("\n")
    tmp_path.replace(path)


def parse_layers(spec: str) -> list[int]:
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise ValueError("No layers selected")
    return sorted(layers)


def expert_sort_key(name: str) -> tuple[int, int, int]:
    match = EXPERT_WEIGHT_RE.match(name)
    if not match:
        return (10**9, 10**9, 10**9)
    proj_order = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
    return (
        int(match.group("layer")),
        int(match.group("expert")),
        proj_order[match.group("proj")],
    )


def scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected a .weight tensor name, got {weight_name}")
    return weight_name.removesuffix(".weight") + ".weight_scale"


def layer_prefix(layer: int) -> str:
    return f"model.layers.{layer}.mlp.experts"


def selected_expert_weights(weight_map: dict[str, str], layers: Iterable[int]) -> list[str]:
    selected_layers = set(layers)
    weights: list[str] = []
    for key in weight_map:
        match = EXPERT_WEIGHT_RE.match(key)
        if match and int(match.group("layer")) in selected_layers:
            weights.append(key)
    return sorted(weights, key=expert_sort_key)


def tensor_meta(model_dir: Path, weight_map: dict[str, str], key: str) -> tuple[list[int], str]:
    with safe_open(model_dir / weight_map[key], framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice(key)
        return tensor_slice.get_shape(), tensor_slice.get_dtype()


def validate_inputs(
    base_dir: Path,
    bf16_dir: Path,
    base_weight_map: dict[str, str],
    bf16_weight_map: dict[str, str],
    weights: list[str],
) -> None:
    if not weights:
        raise RuntimeError("No selected sparse expert weights found in base checkpoint")

    missing_bf16 = [key for key in weights if key not in bf16_weight_map]
    if missing_bf16:
        raise RuntimeError(f"BF16 source is missing {len(missing_bf16)} tensor(s); first: {missing_bf16[0]}")

    missing_scales = [scale_name(key) for key in weights if scale_name(key) not in base_weight_map]
    if missing_scales:
        raise RuntimeError(f"Base checkpoint is missing {len(missing_scales)} scale tensor(s); first: {missing_scales[0]}")

    for key in weights:
        base_shape, base_dtype = tensor_meta(base_dir, base_weight_map, key)
        base_scale_shape, base_scale_dtype = tensor_meta(base_dir, base_weight_map, scale_name(key))
        bf16_shape, bf16_dtype = tensor_meta(bf16_dir, bf16_weight_map, key)
        if base_shape != bf16_shape:
            raise RuntimeError(f"Shape mismatch for {key}: base {base_shape}, BF16 {bf16_shape}")
        if bf16_dtype != "BF16":
            raise RuntimeError(f"Expected BF16 source for {key}, got {bf16_dtype}")
        if base_dtype != "F8_E4M3":
            raise RuntimeError(f"Expected FP8 base tensor for {key}, got {base_dtype}")
        if base_scale_dtype not in {"F32", "U8"}:
            raise RuntimeError(f"Unexpected base scale dtype for {scale_name(key)}: {base_scale_dtype}")
        expected_mxfp8_scale = [bf16_shape[0], (bf16_shape[1] + MXFP8_GROUP_SIZE - 1) // MXFP8_GROUP_SIZE]
        if len(base_scale_shape) != 2:
            raise RuntimeError(f"Expected 2D base scale for {scale_name(key)}, got {base_scale_shape}")
        if expected_mxfp8_scale[0] != bf16_shape[0]:
            raise RuntimeError(f"Internal MXFP8 scale check failed for {key}: {expected_mxfp8_scale}")


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if src.is_symlink():
        real_src = src.resolve()
    else:
        real_src = src
    try:
        os.link(real_src, dst)
    except OSError:
        shutil.copy2(real_src, dst)


def initialize_output(base_dir: Path, output_dir: Path, affected_shards: set[str], force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output already exists; use --force to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for item in sorted(base_dir.iterdir(), key=lambda p: p.name):
        dst = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, symlinks=True)
        elif item.name in affected_shards:
            continue
        elif item.name in {"config.json", INDEX_NAME, "hf_quant_config.json"}:
            shutil.copy2(item, dst)
        else:
            hardlink_or_copy(item, dst)


def copy_referenced_source_shards(source_dir: Path, output_dir: Path, source_shards: Iterable[str]) -> None:
    for shard_name in sorted(set(source_shards)):
        hardlink_or_copy(source_dir / shard_name, output_dir / shard_name)


def transplant_glm51_layer78_mtp(
    output_dir: Path,
    source_dir: Path,
    mixed_quant_config: dict[str, Any],
) -> None:
    """Replace GLM-5.1 layer 78 with the NVFP4 MTP layer from source_dir.

    GLM-5.1 stores its next-token-prediction/MTP layer as model.layers.78.*
    rather than as model.mtp.*.  The FP8-PB-WO mixed base used for this chimera
    keeps that layer in BF16, so a production MTP-capable checkpoint must
    transplant the already-quantized NVFP4 layer 78 tensors from the MTP source
    checkpoint and point the output index at those shards.
    """

    source_index = read_json(source_dir / INDEX_NAME)
    source_weight_map = dict(source_index["weight_map"])
    mtp_keys = sorted(key for key in source_weight_map if key.startswith("model.layers.78."))
    if not mtp_keys:
        raise RuntimeError(f"No model.layers.78.* MTP tensors found in {source_dir}")

    source_shards = {source_weight_map[key] for key in mtp_keys}
    copy_referenced_source_shards(source_dir, output_dir, source_shards)

    index_path = output_dir / INDEX_NAME
    index = read_json(index_path)
    weight_map = dict(index["weight_map"])
    for key in mtp_keys:
        weight_map[key] = source_weight_map[key]

    metadata = index.setdefault("metadata", {})
    metadata["glm51_mtp_layer78_patch"] = "replaced BF16/unquantized layer78 with NVFP4 MTP tensors"
    metadata["glm51_mtp_layer78_source"] = str(source_dir)
    index["weight_map"] = dict(sorted(weight_map.items()))
    write_json(index_path, index)

    quantized_layers = mixed_quant_config.setdefault("quantized_layers", {})
    if not isinstance(quantized_layers, dict):
        raise RuntimeError("quantization_config.quantized_layers must be a dict")
    quantized_layers["model.layers.78.mlp.experts"] = {
        "quant_algo": "NVFP4",
        "group_size": 16,
    }

    producer = mixed_quant_config.setdefault("producer", {})
    if isinstance(producer, dict):
        producer["glm51_mtp_layer78_patch"] = "NVFP4 MTP layer transplanted"
        producer["glm51_mtp_layer78_source"] = str(source_dir)


def quantize_mxfp8_from_bf16(weight: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    from modelopt.torch.quantization.qtensor import MXFP8QTensor

    if device != "cpu":
        weight = weight.to(device=device, dtype=torch.bfloat16, non_blocking=False).contiguous()
    else:
        weight = weight.to(torch.bfloat16).contiguous()
    with torch.inference_mode():
        qweight, scale = MXFP8QTensor.quantize(weight)
    quantized_data = qweight._quantized_data.detach().cpu().contiguous()
    scale_data = scale.detach().cpu().contiguous()
    del qweight, scale, weight
    return quantized_data, scale_data


def load_bf16_tensor(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(model_dir / weight_map[key], framework="pt", device="cpu") as f:
        return f.get_tensor(key).contiguous()


def rewrite_layer_shard(
    output_dir: Path,
    shard_name: str,
    layer_weights: list[str],
    bf16_dir: Path,
    bf16_weight_map: dict[str, str],
    device: str,
) -> None:
    tmp_path = output_dir / f".{shard_name}.tmp"
    final_path = output_dir / shard_name
    if tmp_path.exists():
        tmp_path.unlink()

    tensors: dict[str, torch.Tensor] = {}
    for idx, key in enumerate(layer_weights, start=1):
        bf16 = load_bf16_tensor(bf16_dir, bf16_weight_map, key)
        qweight, qscale = quantize_mxfp8_from_bf16(bf16, device)
        del bf16
        tensors[key] = qweight
        tensors[scale_name(key)] = qscale
        if idx % 64 == 0 or idx == len(layer_weights):
            print(f"    [{device}] {idx}/{len(layer_weights)} tensors converted for {shard_name}", flush=True)
            gc.collect()

    save_file(tensors, str(tmp_path))
    tmp_path.replace(final_path)
    print(f"wrote {shard_name}: {len(layer_weights)} MXFP8 expert weight tensor(s) on {device}", flush=True)


def rewrite_layers_on_device(
    gpu: int | None,
    jobs: list[tuple[int, str, list[str]]],
    output_dir: Path,
    bf16_dir: Path,
    bf16_weight_map: dict[str, str],
) -> None:
    device = "cpu" if gpu is None else f"cuda:{gpu}"
    if gpu is not None:
        torch.cuda.set_device(gpu)
    for layer, output_shard, layer_weights in jobs:
        print(f"[layer {layer}] -> {output_shard} on {device}", flush=True)
        rewrite_layer_shard(
            output_dir=output_dir,
            shard_name=output_shard,
            layer_weights=layer_weights,
            bf16_dir=bf16_dir,
            bf16_weight_map=bf16_weight_map,
            device=device,
        )
        gc.collect()
        if gpu is not None:
            torch.cuda.empty_cache()


def parse_gpus(spec: str | None) -> list[int]:
    if spec is None or spec.strip().lower() in {"", "auto"}:
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    if spec.strip().lower() in {"cpu", "none"}:
        return []
    gpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        gpus.append(int(part))
    return gpus


def rewrite_layers_parallel(
    layers: list[int],
    layer_to_weights: dict[int, list[str]],
    output_dir: Path,
    bf16_dir: Path,
    bf16_weight_map: dict[str, str],
    gpus: list[int],
) -> None:
    if not gpus:
        for layer in layers:
            output_shard = f"model-mixed-mxfp8-layer{layer}.safetensors"
            rewrite_layers_on_device(
                gpu=None,
                jobs=[(layer, output_shard, layer_to_weights[layer])],
                output_dir=output_dir,
                bf16_dir=bf16_dir,
                bf16_weight_map=bf16_weight_map,
            )
        return

    jobs_by_gpu: dict[int, list[tuple[int, str, list[str]]]] = {gpu: [] for gpu in gpus}
    for idx, layer in enumerate(layers):
        gpu = gpus[idx % len(gpus)]
        output_shard = f"model-mixed-mxfp8-layer{layer}.safetensors"
        jobs_by_gpu[gpu].append((layer, output_shard, layer_to_weights[layer]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(
                rewrite_layers_on_device,
                gpu,
                jobs,
                output_dir,
                bf16_dir,
                bf16_weight_map,
            )
            for gpu, jobs in jobs_by_gpu.items()
            if jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def mxfp8_group(targets: list[str]) -> dict[str, Any]:
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
        "targets": targets,
    }


def update_config(
    output_dir: Path,
    layers: list[int],
    bf16_dir: Path,
    mtp_nvfp4_source: Path | None,
) -> None:
    config_path = output_dir / "config.json"
    config = read_json(config_path)
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        raise RuntimeError("config.json is missing quantization_config")
    if quant.get("quant_method") != "modelopt":
        raise RuntimeError(f"Expected ModelOpt quantization_config, got {quant.get('quant_method')}")

    quant["quant_algo"] = "MIXED_PRECISION"
    quantized_layers = quant.setdefault("quantized_layers", {})
    if not isinstance(quantized_layers, dict):
        raise RuntimeError("quantization_config.quantized_layers must be a dict")

    targets = [layer_prefix(layer) for layer in layers]
    for target in targets:
        quantized_layers[target] = {
            "quant_algo": "MXFP8",
            "group_size": MXFP8_GROUP_SIZE,
        }

    config_groups = quant.setdefault("config_groups", {})
    if not isinstance(config_groups, dict):
        raise RuntimeError("quantization_config.config_groups must be a dict")
    config_groups["group_mxfp8_sparse_experts"] = mxfp8_group(targets)

    producer = quant.setdefault("producer", {})
    if isinstance(producer, dict):
        producer["glm51_mxfp8_bf16_source"] = str(bf16_dir)
        producer["glm51_mxfp8_layers"] = ",".join(str(layer) for layer in layers)

    if mtp_nvfp4_source is not None:
        transplant_glm51_layer78_mtp(
            output_dir=output_dir,
            source_dir=mtp_nvfp4_source,
            mixed_quant_config=quant,
        )

    quant["quantized_layers"] = dict(sorted(quantized_layers.items()))
    config["quantization_config"] = quant
    write_json(config_path, config)


def update_index(output_dir: Path, layers: list[int], weights: list[str], source_dir: Path) -> None:
    index_path = output_dir / INDEX_NAME
    index = read_json(index_path)
    weight_map = dict(index["weight_map"])

    selected_layers = set(layers)
    for key in weights:
        match = EXPERT_WEIGHT_RE.match(key)
        if not match:
            continue
        layer = int(match.group("layer"))
        if layer not in selected_layers:
            continue
        shard_name = f"model-mixed-mxfp8-layer{layer}.safetensors"
        weight_map[key] = shard_name
        weight_map[scale_name(key)] = shard_name

    metadata = index.setdefault("metadata", {})
    metadata["total_size"] = sum((output_dir / shard).stat().st_size for shard in set(weight_map.values()))
    metadata["glm51_mxfp8_bf16_source"] = str(source_dir)
    metadata["glm51_mxfp8_layers"] = ",".join(str(layer) for layer in layers)
    metadata["glm51_mxfp8_weight_tensors"] = len(weights)
    index["weight_map"] = dict(sorted(weight_map.items()))
    metadata["total_size"] = sum((output_dir / shard).stat().st_size for shard in set(weight_map.values()))
    write_json(index_path, index)


def inspect_output(output_dir: Path, layers: list[int]) -> None:
    index = read_json(output_dir / INDEX_NAME)
    weight_map = index["weight_map"]
    sample_layer = layers[0]
    sample_key = f"model.layers.{sample_layer}.mlp.experts.0.gate_proj.weight"
    sample_scale = scale_name(sample_key)
    with safe_open(output_dir / weight_map[sample_key], framework="pt", device="cpu") as f:
        weight_slice = f.get_slice(sample_key)
        scale_slice = f.get_slice(sample_scale)
        print(
            "sample:",
            sample_key,
            weight_slice.get_shape(),
            weight_slice.get_dtype(),
            sample_scale,
            scale_slice.get_shape(),
            scale_slice.get_dtype(),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Existing mixed GLM checkpoint")
    parser.add_argument("--bf16-source", type=Path, required=True, help="BF16 GLM checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint directory")
    parser.add_argument("--layers", default="42-62", help="Layer selection, e.g. 42-62 or 42-47,51-62")
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU IDs, auto, or cpu")
    parser.add_argument(
        "--mtp-nvfp4-source",
        type=Path,
        help="Optional GLM-5.1 NVFP4-MTP checkpoint used to replace BF16 model.layers.78.*",
    )
    parser.add_argument("--force", action="store_true", help="Replace output directory if it exists")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned changes only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base
    bf16_dir = args.bf16_source
    output_dir = args.output
    layers = parse_layers(args.layers)
    gpus = parse_gpus(args.gpus)
    mtp_nvfp4_source = args.mtp_nvfp4_source

    base_index = read_json(base_dir / INDEX_NAME)
    bf16_index = read_json(bf16_dir / INDEX_NAME)
    base_weight_map = dict(base_index["weight_map"])
    bf16_weight_map = dict(bf16_index["weight_map"])
    weights = selected_expert_weights(base_weight_map, layers)
    validate_inputs(base_dir, bf16_dir, base_weight_map, bf16_weight_map, weights)

    layer_to_weights: dict[int, list[str]] = {layer: [] for layer in layers}
    for key in weights:
        match = EXPERT_WEIGHT_RE.match(key)
        assert match is not None
        layer_to_weights[int(match.group("layer"))].append(key)

    affected_shards = {base_weight_map[key] for key in weights}
    expected_one_shard_per_layer = {
        layer: sorted({base_weight_map[key] for key in layer_to_weights[layer]})
        for layer in layers
    }
    for layer, shards in expected_one_shard_per_layer.items():
        if len(shards) != 1:
            raise RuntimeError(f"Expected one expert shard for layer {layer}, got {shards}")

    print(f"Base checkpoint: {base_dir}")
    print(f"BF16 source:     {bf16_dir}")
    print(f"Output:          {output_dir}")
    print(f"Layers:          {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Expert weights:  {len(weights)}")
    print(f"Affected shards: {len(affected_shards)}")
    print(f"Devices:         {','.join(f'cuda:{gpu}' for gpu in gpus) if gpus else 'cpu'}")
    for layer in layers:
        print(f"  layer {layer}: {len(layer_to_weights[layer])} weights")

    if args.dry_run:
        if mtp_nvfp4_source is not None:
            source_index = read_json(mtp_nvfp4_source / INDEX_NAME)
            mtp_keys = [key for key in source_index["weight_map"] if key.startswith("model.layers.78.")]
            print(f"MTP layer78 source: {mtp_nvfp4_source} ({len(mtp_keys)} tensors)")
        print("Dry run; no files written.")
        return

    initialize_output(base_dir, output_dir, affected_shards, args.force)

    rewrite_layers_parallel(
        layers=layers,
        layer_to_weights=layer_to_weights,
        output_dir=output_dir,
        bf16_dir=bf16_dir,
        bf16_weight_map=bf16_weight_map,
        gpus=gpus,
    )

    update_config(output_dir, layers, bf16_dir, mtp_nvfp4_source)
    update_index(output_dir, layers, weights, bf16_dir)
    inspect_output(output_dir, layers)
    print("done")


if __name__ == "__main__":
    main()
