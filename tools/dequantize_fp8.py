#!/usr/bin/env python3
"""Dequantize FP8 checkpoint to bfloat16 by processing safetensors files directly."""
import argparse
import json
import multiprocessing
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

_MIMO_IMAGE_MEAN = [0.485, 0.456, 0.406]
_MIMO_IMAGE_STD = [0.229, 0.224, 0.225]


def resolve_model_dir(model_id: str) -> Path:
    """Resolve a HuggingFace model ID to its local cache directory."""
    path = snapshot_download(model_id, local_files_only=True)
    return Path(path)


def build_weight_map_from_safetensors(model_dir: Path) -> dict[str, str]:
    """Build a weight map from the safetensors files present on disk."""
    weight_map = {}
    for file_path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                weight_map[key] = file_path.name
    return weight_map


def _tensor_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tmp_shard_name(file_name: str, shard_idx: int) -> str:
    return f".tmp-{Path(file_name).stem}-{shard_idx:05d}.safetensors"


def _dequantize_block_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    weight_fp32 = weight.to(torch.float32)
    block_m, block_n = block_size
    expected_scale_shape = (
        _ceil_div(weight_fp32.shape[0], block_m),
        _ceil_div(weight_fp32.shape[1], block_n),
    )
    if scale.shape[0] < expected_scale_shape[0] or scale.shape[1] < expected_scale_shape[1]:
        raise ValueError(
            "FP8 scale grid is smaller than the weight requires: "
            f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}, expected={expected_scale_shape}"
        )
    scale = scale[: expected_scale_shape[0], : expected_scale_shape[1]]
    scale_expanded = scale.repeat_interleave(block_m, dim=0)
    scale_expanded = scale_expanded.repeat_interleave(block_n, dim=1)
    scale_expanded = scale_expanded[: weight_fp32.shape[0], : weight_fp32.shape[1]]

    # Despite the name "weight_scale_inv", these are actually scales (not inverses).
    # The Triton kernel multiplies by them, so we do too.
    return (weight_fp32 * scale_expanded).to(torch.bfloat16)


def _deinterleave_mimo_qkv_rows(tensor: torch.Tensor, spec: dict[str, int]) -> torch.Tensor:
    tp_size = spec["tp_size"]
    q_shard = spec["q_size"] // tp_size
    k_shard = spec["k_size"] // tp_size
    v_shard = spec["v_size"] // tp_size
    chunk_size = q_shard + k_shard + v_shard

    if tensor.shape[0] != chunk_size * tp_size:
        raise ValueError(
            f"Cannot deinterleave {spec['name']}: expected {chunk_size * tp_size} rows "
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

    return torch.cat([*q_chunks, *k_chunks, *v_chunks], dim=0)


def _dequantize_and_deinterleave_mimo_qkv(
    weight: torch.Tensor,
    scale: torch.Tensor,
    spec: dict[str, int],
    block_size: tuple[int, int],
) -> torch.Tensor:
    tp_size = spec["tp_size"]
    q_shard = spec["q_size"] // tp_size
    k_shard = spec["k_size"] // tp_size
    v_shard = spec["v_size"] // tp_size
    chunk_size = q_shard + k_shard + v_shard
    block_m, _ = block_size
    # The public MiMo fused-QKV checkpoint stores each TP rank as one local
    # fused [q_shard][k_shard][v_shard] matrix. Its FP8 scales are therefore
    # block rows over that local fused matrix, not separately reset at Q/K/V
    # boundaries. This intentionally allows a block to straddle K/V when the
    # local K shard is not a multiple of block_m, matching native FP8 loading.
    scale_rows_per_chunk = _ceil_div(chunk_size, block_m)
    expected_scale_rows = scale_rows_per_chunk * tp_size

    if weight.shape[0] != chunk_size * tp_size:
        raise ValueError(
            f"Cannot deinterleave {spec['name']}: expected {chunk_size * tp_size} rows "
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
        dequant_chunk = _dequantize_block_fp8_weight(weight_chunk, scale_chunk, block_size)
        q, k, v = dequant_chunk.split([q_shard, k_shard, v_shard], dim=0)
        q_chunks.append(q)
        k_chunks.append(k)
        v_chunks.append(v)

    return torch.cat([*q_chunks, *k_chunks, *v_chunks], dim=0)


def _load_tensor_from_weight_map(model_dir: str | Path, weight_map: dict[str, str], tensor_name: str) -> torch.Tensor | None:
    file_name = weight_map.get(tensor_name)
    if file_name is None:
        return None
    with safe_open(Path(model_dir) / file_name, framework="pt") as f:
        return f.get_tensor(tensor_name)


def build_mimo_qkv_specs(config: dict, weight_map: dict[str, str], tp_size: int) -> dict[str, dict[str, int]]:
    """Return per-layer QKV deinterleave specs for MiMo's TP-packed fused QKV tensors."""
    pattern = config.get("hybrid_layer_pattern")
    if not isinstance(pattern, list):
        raise ValueError("MiMo QKV deinterleave requires hybrid_layer_pattern in config.json")

    default_head_dim = config["hidden_size"] // config["num_attention_heads"]
    head_dim = config.get("head_dim", default_head_dim)
    v_head_dim = config.get("v_head_dim", head_dim)
    swa_head_dim = config.get("swa_head_dim", head_dim)
    swa_v_head_dim = config.get("swa_v_head_dim", v_head_dim)

    specs = {}
    key_re = re.compile(r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$")
    for weight_name in weight_map:
        match = key_re.match(weight_name)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        if layer_idx >= len(pattern):
            raise ValueError(f"No hybrid_layer_pattern entry for {weight_name}")

        is_swa = pattern[layer_idx] == 1
        num_attention_heads = config.get("swa_num_attention_heads", config["num_attention_heads"]) if is_swa else config["num_attention_heads"]
        num_key_value_heads = config.get("swa_num_key_value_heads", config["num_key_value_heads"]) if is_swa else config["num_key_value_heads"]
        q_head_dim = swa_head_dim if is_swa else head_dim
        k_head_dim = q_head_dim
        v_dim = swa_v_head_dim if is_swa else v_head_dim

        q_size = num_attention_heads * q_head_dim
        k_size = num_key_value_heads * k_head_dim
        v_size = num_key_value_heads * v_dim
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


def process_resharded_file(args):
    """Process one safetensors file while flushing smaller output shards."""
    file_name, weight_names, model_dir, output_dir, max_shard_bytes, qkv_specs, weight_map = args
    block_size = (128, 128)

    try:
        file_path = Path(model_dir) / file_name
        output_dir = Path(output_dir)

        processed = {}
        processed_entries = []
        fp8_dequantized = 0
        shard_idx = 1
        shard_size = 0

        def flush():
            nonlocal processed, shard_idx, shard_size
            if not processed:
                return
            output_name = _tmp_shard_name(file_name, shard_idx)
            save_file(processed, str(output_dir / output_name))
            processed_entries.extend((key, output_name) for key in processed)
            processed = {}
            shard_idx += 1
            shard_size = 0

        with safe_open(file_path, framework="pt") as f:
            available = set(f.keys())
            for weight_name in weight_names:
                if weight_name.endswith(".weight_scale_inv"):
                    continue

                tensor = f.get_tensor(weight_name)

                qkv_spec = qkv_specs.get(weight_name)

                if tensor.dtype == torch.float8_e4m3fn and weight_name.endswith(".weight"):
                    scale_name = weight_name.replace(".weight", ".weight_scale_inv")

                    scale_inv = f.get_tensor(scale_name) if scale_name in available else _load_tensor_from_weight_map(
                        model_dir, weight_map, scale_name
                    )
                    if scale_inv is None:
                        raise ValueError(f"Missing FP8 scale sidecar for {weight_name}: expected {scale_name}")

                    fp8_dequantized += 1
                    if qkv_spec is not None:
                        out_tensor = _dequantize_and_deinterleave_mimo_qkv(
                            tensor, scale_inv, qkv_spec, block_size
                        )
                    else:
                        out_tensor = _dequantize_block_fp8_weight(tensor, scale_inv, block_size)

                    if fp8_dequantized <= 3:
                        weight_fp32 = tensor.to(torch.float32)
                        print(f"    {weight_name}:")
                        print(f"      FP8 weight: min={weight_fp32.min():.2f}, max={weight_fp32.max():.2f}")
                        print(f"      Scale: min={scale_inv.min():.6f}, max={scale_inv.max():.6f}")
                        print(f"      Dequantized: min={out_tensor.min():.2f}, max={out_tensor.max():.2f}, mean={out_tensor.mean():.4f}, std={out_tensor.std():.4f}")
                elif tensor.dtype == torch.float8_e4m3fn:
                    raise ValueError(f"Unexpected FP8 tensor without scale handling: {weight_name}")
                else:
                    out_tensor = tensor

                if qkv_spec is not None and tensor.dtype != torch.float8_e4m3fn:
                    out_tensor = _deinterleave_mimo_qkv_rows(out_tensor, qkv_spec)

                out_tensor = out_tensor.contiguous()
                tensor_size = _tensor_size(out_tensor)
                if processed and shard_size + tensor_size > max_shard_bytes:
                    flush()
                processed[weight_name] = out_tensor
                shard_size += tensor_size

        flush()

        return (file_name, processed_entries, fp8_dequantized, None)

    except Exception as e:
        import traceback
        error_str = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return (file_name, [], 0, error_str)


def process_single_file(args):
    """Process a single safetensors file - designed to run in parallel."""
    file_name, weight_names, model_dir, output_dir, qkv_specs, weight_map = args
    block_size = (128, 128)

    try:
        file_path = Path(model_dir) / file_name

        tensors = {}
        with safe_open(file_path, framework="pt") as f:
            for weight_name in weight_names:
                tensors[weight_name] = f.get_tensor(weight_name)

        processed = {}
        fp8_dequantized = 0

        for weight_name, tensor in tensors.items():
            if tensor is None:
                continue

            qkv_spec = qkv_specs.get(weight_name)

            if tensor.dtype == torch.float8_e4m3fn and weight_name.endswith('.weight'):
                scale_name = weight_name.replace('.weight', '.weight_scale_inv')

                if scale_name in tensors:
                    scale_inv = tensors[scale_name]
                    tensors[scale_name] = None
                else:
                    scale_inv = _load_tensor_from_weight_map(model_dir, weight_map, scale_name)

                if scale_inv is not None:
                    fp8_dequantized += 1

                    if qkv_spec is not None:
                        dequant = _dequantize_and_deinterleave_mimo_qkv(tensor, scale_inv, qkv_spec, block_size)
                    else:
                        dequant = _dequantize_block_fp8_weight(tensor, scale_inv, block_size)
                    processed[weight_name] = dequant

                    if fp8_dequantized <= 3:
                        weight_fp32 = tensor.to(torch.float32)
                        print(f"    {weight_name}:")
                        print(f"      FP8 weight: min={weight_fp32.min():.2f}, max={weight_fp32.max():.2f}")
                        print(f"      Scale: min={scale_inv.min():.6f}, max={scale_inv.max():.6f}")
                        print(f"      Dequantized: min={dequant.min():.2f}, max={dequant.max():.2f}, mean={dequant.mean():.4f}, std={dequant.std():.4f}")

                else:
                    raise ValueError(f"Missing FP8 scale sidecar for {weight_name}: expected {scale_name}")
            elif weight_name.endswith('.weight_scale_inv'):
                pass
            else:
                if tensor.dtype == torch.float8_e4m3fn:
                    raise ValueError(f"Unexpected FP8 tensor without scale handling: {weight_name}")
                processed[weight_name] = tensor
                if qkv_spec is not None:
                    processed[weight_name] = _deinterleave_mimo_qkv_rows(processed[weight_name], qkv_spec)

        output_file = Path(output_dir) / file_name
        save_file(processed, output_file)

        return (file_name, list(processed.keys()), fp8_dequantized, None)

    except Exception as e:
        import traceback
        error_str = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return (file_name, [], 0, error_str)


def normalize_shard_names(output_dir: Path, weight_map: dict[str, str]) -> None:
    """Rename temporary output shards to canonical HF shard names."""
    old_names = sorted(set(weight_map.values()))
    total = len(old_names)
    rename_map = {
        old_name: f"model-{idx:05d}-of-{total:05d}.safetensors"
        for idx, old_name in enumerate(old_names, 1)
    }

    for old_name, new_name in rename_map.items():
        if old_name == new_name:
            continue
        (output_dir / old_name).rename(output_dir / new_name)

    for key in list(weight_map):
        weight_map[key] = rename_map[weight_map[key]]


def _build_mimo_processor_configs(config: dict) -> tuple[dict, dict, dict] | None:
    vision_config = config.get("vision_config")
    if not isinstance(vision_config, dict):
        return None

    mimo_processor_config = config.get("processor_config") or {}
    patch_size = vision_config.get("patch_size", vision_config.get("spatial_patch_size", 16))
    temporal_patch_size = vision_config.get("temporal_patch_size", 2)
    merge_size = vision_config.get("spatial_merge_size", 2)

    image_config = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": _MIMO_IMAGE_MEAN,
        "image_processor_type": "Qwen2VLImageProcessor",
        "image_std": _MIMO_IMAGE_STD,
        "merge_size": merge_size,
        "patch_size": patch_size,
        "resample": 3,
        "rescale_factor": 1 / 255,
        "size": {
            "longest_edge": mimo_processor_config.get("image_max_pixels", 8388608),
            "shortest_edge": mimo_processor_config.get("image_min_pixels", 8192),
        },
        "temporal_patch_size": temporal_patch_size,
    }
    video_config = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "do_sample_frames": False,
        "image_mean": _MIMO_IMAGE_MEAN,
        "image_std": _MIMO_IMAGE_STD,
        "fps": mimo_processor_config.get("fps", 1.0),
        "max_frames": mimo_processor_config.get("max_frames", 3600),
        "merge_size": merge_size,
        "min_frames": mimo_processor_config.get("min_frames"),
        "num_frames": mimo_processor_config.get("num_frames"),
        "patch_size": patch_size,
        "resample": 3,
        "rescale_factor": 1 / 255,
        "return_metadata": False,
        "size": {
            "longest_edge": mimo_processor_config.get("video_max_pixels", 8388608),
            "shortest_edge": mimo_processor_config.get("video_min_pixels", 8192),
        },
        "temporal_patch_size": temporal_patch_size,
        "video_processor_type": "Qwen2VLVideoProcessor",
    }
    processor_config = {
        "image_processor": image_config,
        "processor_class": "Qwen2_5_VLProcessor",
        "video_processor": video_config,
    }
    return image_config, video_config, processor_config


def write_mimo_processor_configs(output_dir: Path, config: dict) -> None:
    configs = _build_mimo_processor_configs(config)
    if configs is None:
        return

    for file_name, data in zip(
        ("preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json"),
        configs,
    ):
        with open(output_dir / file_name, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"  {file_name}")


def add_mimo_zero_visual_biases(output_dir: Path, config: dict, weight_map: dict[str, str]) -> None:
    """Add zero-initialized MiMo visual merger biases omitted by the upstream checkpoint."""
    if config.get("model_type") != "mimo_v2":
        return

    vision_config = config.get("vision_config")
    if not isinstance(vision_config, dict):
        return

    context_dim = vision_config.get("hidden_size")
    output_dim = vision_config.get("out_hidden_size", config.get("hidden_size"))
    spatial_merge_size = vision_config.get("spatial_merge_size", 2)
    if context_dim is None or output_dim is None:
        return

    merged_dim = context_dim * (spatial_merge_size**2)
    bias_specs = {
        "visual.merger.mlp.0.bias": (merged_dim,),
        "visual.merger.mlp.2.bias": (output_dim,),
    }
    missing = {name: shape for name, shape in bias_specs.items() if name not in weight_map}
    if not missing:
        return

    shard_name = "model-mimo-visual-biases.safetensors"
    tensors = {
        name: torch.zeros(shape, dtype=torch.bfloat16)
        for name, shape in missing.items()
    }
    save_file(tensors, output_dir / shard_name)
    for name in tensors:
        weight_map[name] = shard_name
    print(f"Added zero MiMo visual biases to {shard_name}: {', '.join(sorted(tensors))}")


def main():
    parser = argparse.ArgumentParser(description="Dequantize FP8 checkpoint to bfloat16")
    parser.add_argument("model_id", help="HuggingFace model ID (e.g. MiniMaxAI/MiniMax-M2.5)")
    parser.add_argument("-o", "--output-dir", help="Output directory for dequantized checkpoint.")
    parser.add_argument("--model-dir", help="Override model directory instead of resolving from HF cache")
    parser.add_argument("--config-source",
                        help="Alternative model ID or directory to copy config/tokenizer files from")
    parser.add_argument("-j", "--workers", type=int, default=min(8, multiprocessing.cpu_count()),
                        help="Number of parallel workers (default: min(8, cpu_count))")
    parser.add_argument("--ignore-index", action="store_true",
                        help="Ignore model.safetensors.index.json and scan local safetensors files instead.")
    parser.add_argument("--output-shard-size-gib", type=float, default=None,
                        help="Stream and normalize output into shards of approximately this size in GiB.")
    parser.add_argument("--mimo-deinterleave-qkv", action="store_true",
                        help="Normalize MiMo-V2 TP-packed fused QKV rows to canonical [Q][K][V] order.")
    parser.add_argument("--mimo-qkv-tp-size", type=int, default=4,
                        help="Tensor-parallel packing size used by MiMo fused QKV tensors (default: 4).")
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        print(f"Resolving {args.model_id} from HuggingFace cache...")
        model_dir = resolve_model_dir(args.model_id)

    model_name = args.model_id.split("/")[-1]
    if not args.output_dir:
        parser.error("--output-dir is required")
    output_dir = args.output_dir

    print(f"Dequantizing {model_dir} to {output_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ignore_index:
        print("Ignoring index; scanning safetensors files...")
        weight_map = build_weight_map_from_safetensors(model_dir)
        metadata = {}
    else:
        print("Loading index...")
        index_path = model_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        weight_map = index['weight_map']
        metadata = index.get('metadata', {})

    safetensors_files = sorted(set(weight_map.values()))
    print(f"Found {len(safetensors_files)} safetensors files to process")

    qkv_specs = {}
    if args.mimo_deinterleave_qkv:
        if args.mimo_qkv_tp_size <= 0:
            parser.error("--mimo-qkv-tp-size must be positive")
        with open(model_dir / "config.json") as f:
            config = json.load(f)
        qkv_specs = build_mimo_qkv_specs(config, weight_map, args.mimo_qkv_tp_size)
        print(
            f"Will deinterleave {len(qkv_specs)} MiMo fused QKV tensors "
            f"from TP-packed layout with TP size {args.mimo_qkv_tp_size}"
        )

    file_to_weights = {}
    for weight_name, file_name in weight_map.items():
        if file_name not in file_to_weights:
            file_to_weights[file_name] = []
        file_to_weights[file_name].append(weight_name)

    num_workers = args.workers
    print(f"\nProcessing {len(safetensors_files)} files with {num_workers} workers...")

    new_weight_map = {}
    total_dequantized = 0
    errors = []

    if args.output_shard_size_gib is None:
        process_fn = process_single_file
        process_args = [
            (file_name, file_to_weights[file_name], str(model_dir), str(output_dir), qkv_specs, weight_map)
            for file_name in safetensors_files
        ]
    else:
        process_fn = process_resharded_file
        max_shard_bytes = int(args.output_shard_size_gib * 1024**3)
        if max_shard_bytes <= 0:
            parser.error("--output-shard-size-gib must be positive")
        for old_tmp in output_dir.glob(".tmp-*.safetensors"):
            old_tmp.unlink()
        process_args = [
            (
                file_name,
                file_to_weights[file_name],
                str(model_dir),
                str(output_dir),
                max_shard_bytes,
                qkv_specs,
                weight_map,
            )
            for file_name in safetensors_files
        ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(process_fn, a): a[0]
            for a in process_args
        }

        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            returned_file, weight_names, fp8_count, error = future.result()

            if error is not None:
                print(f"  ERROR processing {file_name}:")
                print(f"    {error[:200]}...")
                errors.append((file_name, error))
            else:
                if args.output_shard_size_gib is None:
                    for weight_name in weight_names:
                        new_weight_map[weight_name] = returned_file
                    output_shards = 1
                else:
                    for weight_name, output_file in weight_names:
                        new_weight_map[weight_name] = output_file
                    output_shards = len(set(output_file for _, output_file in weight_names))

                total_dequantized += fp8_count
                completed = len([f for f in future_to_file if f.done()])
                print(f"  [{completed}/{len(safetensors_files)}] Completed {returned_file} "
                      f"({fp8_count} FP8 weights, {output_shards} output shard(s))")

    if errors:
        print(f"\nERROR: dequantization failed in {len(errors)} file(s); not writing index/config.")
        for file_name, error in errors[:10]:
            print(f"\n--- {file_name} ---")
            print(error)
        if len(errors) > 10:
            print(f"\n... {len(errors) - 10} more error(s) omitted")
        raise SystemExit(1)

    print(f"\n{'='*80}")
    print(f"Parallel processing complete!")
    print(f"  Total files processed: {len(safetensors_files)}")
    print(f"  Total FP8 weights dequantized: {total_dequantized}")

    print("\nCreating new index...")
    if args.output_shard_size_gib is not None:
        print("Normalizing shard names...")
        normalize_shard_names(output_dir, new_weight_map)

    source_config_path = model_dir / "config.json"
    if source_config_path.exists():
        with open(source_config_path) as f:
            add_mimo_zero_visual_biases(output_dir, json.load(f), new_weight_map)

    metadata = dict(metadata)
    metadata["total_size"] = sum(
        (output_dir / file_name).stat().st_size
        for file_name in set(new_weight_map.values())
    )
    new_index = {
        "metadata": metadata,
        "weight_map": new_weight_map
    }

    index_output = output_dir / "model.safetensors.index.json"
    with open(index_output, 'w') as f:
        json.dump(new_index, f, indent=2)

    # Copy config/tokenizer files from --config-source if given, else from model_dir.
    config_dir = model_dir
    if args.config_source:
        if Path(args.config_source).is_dir():
            config_dir = Path(args.config_source)
        else:
            print(f"Resolving config source {args.config_source} from HuggingFace cache...")
            config_dir = resolve_model_dir(args.config_source)

    COPY_EXTS = {'.json', '.txt', '.model', '.py', '.jinja'}
    print(f"Copying config and tokenizer files from {config_dir}...")
    output_config = None
    for src in config_dir.rglob('*'):
        if not src.is_file() or src.suffix not in COPY_EXTS or 'safetensors' in src.name:
            continue
        try:
            rel_path = src.relative_to(config_dir)
            dst = Path(output_dir) / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.name == 'config.json':
                with open(src) as f:
                    config = json.load(f)
                if 'quantization_config' in config:
                    del config['quantization_config']
                output_config = config
                with open(dst, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                shutil.copy(src, dst)
            print(f"  {rel_path}")
        except Exception as e:
            print(f"  Skipping {src.name}: {e}")

    if output_config is None:
        config_path = Path(output_dir) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                output_config = json.load(f)
    if output_config is not None and output_config.get("model_type") == "mimo_v2":
        write_mimo_processor_configs(Path(output_dir), output_config)
    elif output_config is not None and not (Path(output_dir) / "preprocessor_config.json").exists():
        write_mimo_processor_configs(Path(output_dir), output_config)

    print(f"\n{'='*80}")
    print(f"Dequantized checkpoint saved to: {output_dir}")
    print(f"  Total weights: {len(new_weight_map)}")
    print(f"  Total FP8 weights dequantized: {total_dequantized}")


if __name__ == "__main__":
    main()
