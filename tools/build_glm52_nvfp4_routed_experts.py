#!/usr/bin/env python3
"""Build a GLM-5.2 checkpoint with only base routed experts quantized to NVFP4.

This is an offline shard-by-shard converter for the initial GLM-5.2 path:

  - read a downloaded BF16 GLM-5.2 checkpoint
  - quantize only ``model.layers.{3..77}.mlp.experts.*.{gate,up,down}_proj.weight``
    using max-method NVFP4 packing
  - keep dense layers, attention, shared experts, lm_head, and MTP layer 78 as BF16
  - copy the GLM-5.1 ``model-inputscales.safetensors`` as the initial activation
    scale shard
  - tie gate/up ``weight_scale_2`` per expert so fused W1/W3 consumers dequantize
    both projections with the same global scale
  - write the model tensors directly into large shards
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import re
import shutil
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_SOURCE = Path("/models/GLM-5.2")
DEFAULT_OUTPUT = Path("/models/GLM-5.2-NVFP4")
DEFAULT_SCALE_SOURCE = Path("/models/GLM-5.1-NVFP4/model-inputscales.safetensors")
DEFAULT_QUANT_CONFIG_TEMPLATE = Path("/models/GLM-5.1-NVFP4/config.json")
DEFAULT_SHARD_SIZE = 5 * 1024**3
SCALE_SHARD_NAME = "model-inputscales.safetensors"
GENERATED_TOP_LEVEL_FILES = {"config.json", "model.safetensors.index.json"}
SKIP_AUXILIARY_NAMES = {".cache", ".git", *GENERATED_TOP_LEVEL_FILES}
BLOCK_SIZE = 16
FP4_MAX = 6.0
FP8_MAX = 448.0
FP8_MIN = 2.0 ** -9
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)

ROUTED_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


def parse_size(text: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?\s*", text, re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError(f"Invalid size: {text!r}")
    value = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "t": 1000**4,
        "tb": 1000**4,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    return int(value * multipliers[unit])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize GLM-5.2 base routed experts to NVFP4 shard by shard."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scale-source", type=Path, default=DEFAULT_SCALE_SOURCE)
    parser.add_argument("--quant-config-template", type=Path, default=DEFAULT_QUANT_CONFIG_TEMPLATE)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--shard-size", type=parse_size, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Validate keysets without writing output.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used for per-weight NVFP4 packing.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=4, sort_keys=False)
        f.write("\n")


def load_scale_keys(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return set(f.keys())


def routed_expert_weight_keys(weight_map: dict[str, str], config: dict[str, Any]) -> list[str]:
    first_sparse_layer = int(config.get("first_k_dense_replace", 3))
    num_hidden_layers = int(config["num_hidden_layers"])
    keys: list[str] = []
    for key in weight_map:
        match = ROUTED_EXPERT_WEIGHT_RE.match(key)
        if match is None:
            continue
        layer = int(match.group("layer"))
        # GLM stores the MTP block as model.layers.78 when num_hidden_layers is
        # 78. Match GLM-5.1-NVFP4 and leave that block BF16.
        if first_sparse_layer <= layer < num_hidden_layers:
            keys.append(key)
    return sorted(keys)


def input_scale_key(weight_key: str) -> str:
    return weight_key.removesuffix(".weight") + ".input_scale"


def validate_scales(scale_keys: set[str], quantized_weight_keys: list[str]) -> None:
    expected = {input_scale_key(key) for key in quantized_weight_keys}
    missing = sorted(expected - scale_keys)
    extra = sorted(scale_keys - expected)
    if missing or extra:
        msg = [
            "Input scale keyset does not exactly match selected routed expert weights.",
            f"  expected={len(expected)} actual={len(scale_keys)}",
            f"  missing={len(missing)} extra={len(extra)}",
        ]
        if missing:
            msg.append("  first missing: " + ", ".join(missing[:8]))
        if extra:
            msg.append("  first extra: " + ", ".join(extra[:8]))
        raise RuntimeError("\n".join(msg))


def validate_source_files(source: Path, index: dict[str, Any]) -> None:
    required = {"config.json", "model.safetensors.index.json"}
    required.update(index["weight_map"].values())
    missing = sorted(name for name in required if not (source / name).exists())
    if missing:
        sample = "\n    ".join(missing[:20])
        raise FileNotFoundError(
            f"Source checkpoint is missing {len(missing)} required file(s):\n    {sample}"
        )


def is_auxiliary_item(item: Path) -> bool:
    if item.name in SKIP_AUXILIARY_NAMES:
        return False
    # Checkpoint tensors are regenerated shard-by-shard, and the activation scale
    # tensor is copied from --scale-source.
    if item.name.endswith(".safetensors"):
        return False
    return True


def auxiliary_items(source: Path) -> list[Path]:
    return sorted((item for item in source.iterdir() if is_auxiliary_item(item)), key=lambda p: p.name)


def describe_auxiliary_items(items: list[Path], limit: int = 32) -> str:
    labels = [item.name + ("/" if item.is_dir() and not item.is_symlink() else "") for item in items]
    if not labels:
        return "none"
    if len(labels) > limit:
        return ", ".join(labels[:limit]) + f", ... ({len(labels)} total)"
    return ", ".join(labels)


def copy_auxiliary_files(source: Path, out: Path) -> list[str]:
    copied: list[str] = []
    for item in auxiliary_items(source):
        dst = out / item.name
        if item.is_dir() and not item.is_symlink():
            shutil.copytree(item, dst, symlinks=True)
        else:
            # HF cache snapshots usually contain relative symlinks into a
            # repo-local blobs directory. Dereference them so the exported
            # checkpoint is standalone under /models.
            shutil.copy2(item, dst, follow_symlinks=True)
        copied.append(item.name + ("/" if item.is_dir() and not item.is_symlink() else ""))
    return copied


def copy_input_scales(scale_source: Path, out: Path) -> None:
    shutil.copy2(scale_source, out / SCALE_SHARD_NAME)


def load_quantization_config(template_path: Path) -> dict[str, Any]:
    if not template_path.exists():
        raise FileNotFoundError(f"Quantization config template does not exist: {template_path}")
    template = read_json(template_path)
    quant_config = template.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise RuntimeError(f"{template_path} does not contain a quantization_config object")
    return copy.deepcopy(quant_config)


def adapt_quantization_config_for_glm52_mtp(
    quantization_config: dict[str, Any],
    source_config: dict[str, Any],
) -> dict[str, Any]:
    quantization_config = copy.deepcopy(quantization_config)
    mtp_layer = int(source_config["num_hidden_layers"])
    ignore = list(quantization_config.get("ignore", []))
    mtp_ignore = f"model.layers.{mtp_layer}.*"
    if mtp_ignore not in ignore:
        ignore.append(mtp_ignore)
    quantization_config["ignore"] = ignore
    return quantization_config


def validate_only_quantization_config_changed(
    source_config: dict[str, Any],
    output_config: dict[str, Any],
) -> None:
    source_without_quant = copy.deepcopy(source_config)
    output_without_quant = copy.deepcopy(output_config)
    source_without_quant.pop("quantization_config", None)
    output_without_quant.pop("quantization_config", None)
    if output_without_quant != source_without_quant:
        changed_keys = sorted(
            key
            for key in set(source_without_quant) | set(output_without_quant)
            if source_without_quant.get(key) != output_without_quant.get(key)
        )
        sample = ", ".join(changed_keys[:20])
        raise RuntimeError(
            "Output config changed source fields other than quantization_config: "
            f"{sample}"
        )


def merge_config_with_quantization_config(
    source_config: dict[str, Any],
    quantization_config: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(source_config)
    config["quantization_config"] = copy.deepcopy(quantization_config)
    validate_only_quantization_config_changed(source_config, config)
    return config


def quantize_nvfp4_weight(
    weight: torch.Tensor,
    block_size: int,
    device: torch.device,
    weight_scale_2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.shape[-1] % block_size != 0:
        raise ValueError(
            f"Last dimension {weight.shape[-1]} is not divisible by NVFP4 block size {block_size}"
        )
    work = weight.to(device=device, dtype=torch.bfloat16, non_blocking=False)
    qweight, weight_scale, packed_weight_scale_2 = pack_nvfp4_max(
        work,
        block_size,
        weight_scale_2=weight_scale_2,
    )
    result = (
        qweight.detach().cpu().contiguous(),
        weight_scale.detach().cpu().contiguous(),
        packed_weight_scale_2.detach().cpu().reshape(()).contiguous(),
    )
    del work, qweight, weight_scale, packed_weight_scale_2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def quantize_nvfp4_gate_up_pair(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if gate_weight.shape[-1] % block_size != 0:
        raise ValueError(
            f"Last dimension {gate_weight.shape[-1]} is not divisible by NVFP4 block size {block_size}"
        )
    if up_weight.shape[-1] % block_size != 0:
        raise ValueError(
            f"Last dimension {up_weight.shape[-1]} is not divisible by NVFP4 block size {block_size}"
        )
    gate_work = gate_weight.to(device=device, dtype=torch.bfloat16, non_blocking=False)
    up_work = up_weight.to(device=device, dtype=torch.bfloat16, non_blocking=False)
    max_abs = torch.maximum(
        gate_work.abs().amax().float(),
        up_work.abs().amax().float(),
    )
    if max_abs.item() == 0:
        shared_weight_scale_2 = torch.ones((), device=device, dtype=torch.float32) / (FP4_MAX * FP8_MAX)
    else:
        shared_weight_scale_2 = max_abs / (FP4_MAX * FP8_MAX)

    gate_qweight, gate_weight_scale, gate_weight_scale_2 = pack_nvfp4_max(
        gate_work,
        block_size,
        weight_scale_2=shared_weight_scale_2,
    )
    up_qweight, up_weight_scale, up_weight_scale_2 = pack_nvfp4_max(
        up_work,
        block_size,
        weight_scale_2=shared_weight_scale_2,
    )
    gate_result = (
        gate_qweight.detach().cpu().contiguous(),
        gate_weight_scale.detach().cpu().contiguous(),
        gate_weight_scale_2.detach().cpu().reshape(()).contiguous(),
    )
    up_result = (
        up_qweight.detach().cpu().contiguous(),
        up_weight_scale.detach().cpu().contiguous(),
        up_weight_scale_2.detach().cpu().reshape(()).contiguous(),
    )
    del (
        gate_work,
        up_work,
        max_abs,
        shared_weight_scale_2,
        gate_qweight,
        gate_weight_scale,
        gate_weight_scale_2,
        up_qweight,
        up_weight_scale,
        up_weight_scale_2,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return gate_result, up_result


def reduce_block_amax(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    reshaped = weight.reshape(*weight.shape[:-1], -1, block_size)
    return reshaped.abs().amax(dim=-1).float()


def pack_fp4_e2m1(weight: torch.Tensor) -> torch.Tensor:
    bounds = E2M1_BOUNDS.to(weight.device)
    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()
    ordinals = torch.searchsorted(bounds, weight_abs, out_int32=True).to(torch.uint8)
    odd_bounds = bounds[[1, 3, 5]]
    equals_odd_bounds = torch.any(weight_abs.unsqueeze(-1) == odd_bounds, dim=-1).to(
        torch.uint8
    )
    q = (sign_bit << 3) + ordinals + equals_odd_bounds
    return ((q[..., 1::2] << 4) | q[..., 0::2]).contiguous()


def pack_nvfp4_max(
    weight: torch.Tensor,
    block_size: int,
    weight_scale_2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight_scale_2 is None:
        max_abs = weight.abs().amax().float()
        if max_abs.item() == 0:
            weight_scale_2 = torch.ones((), device=weight.device, dtype=torch.float32) / (FP4_MAX * FP8_MAX)
        else:
            weight_scale_2 = max_abs / (FP4_MAX * FP8_MAX)
    else:
        weight_scale_2 = weight_scale_2.to(device=weight.device, dtype=torch.float32).reshape(())

    per_block_amax = reduce_block_amax(weight, block_size)
    per_block_scale = per_block_amax / (FP4_MAX * weight_scale_2.to(per_block_amax.device))
    per_block_scale = per_block_scale.clamp(min=FP8_MIN, max=FP8_MAX)
    per_block_scale_fp8 = per_block_scale.to(torch.float8_e4m3fn)

    reshaped = weight.reshape(*weight.shape[:-1], -1, block_size)
    scaled = reshaped / (
        (per_block_scale_fp8.to(torch.float32) * weight_scale_2.to(torch.float32))
        .unsqueeze(-1)
    )
    packed = pack_fp4_e2m1(scaled.reshape(weight.shape))
    return packed, per_block_scale_fp8, weight_scale_2.reshape(())


def keys_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        out[shard].append(key)
    return {shard: sorted(keys) for shard, keys in out.items()}


def routed_key_parts(key: str) -> tuple[int, int, str]:
    match = ROUTED_EXPERT_WEIGHT_RE.match(key)
    if match is None:
        raise ValueError(f"Not a routed expert weight key: {key}")
    return int(match.group("layer")), int(match.group("expert")), match.group("proj")


def routed_weight_key(layer: int, expert: int, proj: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class TensorStore:
    def __init__(self, source: Path, weight_map: dict[str, str]) -> None:
        self.source = source
        self.weight_map = weight_map
        self._stack: ExitStack | None = None
        self._handles: dict[str, Any] = {}

    def __enter__(self) -> "TensorStore":
        self._stack = ExitStack()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._handles = {}

    def get_tensor(self, key: str) -> torch.Tensor:
        if self._stack is None:
            raise RuntimeError("TensorStore is not open")
        shard = self.weight_map[key]
        handle = self._handles.get(shard)
        if handle is None:
            handle = self._stack.enter_context(
                safe_open(str(self.source / shard), framework="pt", device="cpu")
            )
            self._handles[shard] = handle
        return handle.get_tensor(key)


class ShardWriter:
    def __init__(self, out: Path, target_shard_size: int) -> None:
        self.out = out
        self.target_shard_size = target_shard_size
        self.current_tensors: dict[str, torch.Tensor] = {}
        self.current_storage_keys: dict[tuple[str, int], str] = {}
        self.current_size = 0
        self.shard_idx = 0
        self.output_weight_map: dict[str, str] = {}
        self.written: list[tuple[str, int, int]] = []

    @staticmethod
    def _storage_key(tensor: torch.Tensor) -> tuple[str, int] | None:
        if tensor.device.type == "meta" or tensor.numel() == 0:
            return None
        return str(tensor.device), tensor.untyped_storage().data_ptr()

    def add(self, key: str, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        size = tensor_nbytes(tensor)
        if self.current_tensors and self.current_size + size > self.target_shard_size:
            self.flush()
        storage_key = self._storage_key(tensor)
        if storage_key is not None and storage_key in self.current_storage_keys:
            tensor = tensor.clone()
            storage_key = self._storage_key(tensor)
        self.current_tensors[key] = tensor
        if storage_key is not None:
            self.current_storage_keys[storage_key] = key
        self.current_size += size

    def flush(self) -> None:
        if not self.current_tensors:
            return
        self.shard_idx += 1
        shard_name = f"model-tmp-{self.shard_idx:05d}.safetensors"
        save_file(self.current_tensors, str(self.out / shard_name))
        shard_size = (self.out / shard_name).stat().st_size
        for key in self.current_tensors:
            self.output_weight_map[key] = shard_name
        self.written.append((shard_name, len(self.current_tensors), shard_size))
        print(
            f"[out {self.shard_idx:05d}] {shard_name}: "
            f"{len(self.current_tensors)} tensor(s), {shard_size / 1024**3:.2f} GiB",
            flush=True,
        )
        self.current_tensors = {}
        self.current_storage_keys = {}
        self.current_size = 0
        gc.collect()

    def finalize(self) -> dict[str, str]:
        self.flush()
        total = len(self.written)
        rename_map = {
            shard_name: f"model-{idx:05d}-of-{total:05d}.safetensors"
            for idx, (shard_name, _tensor_count, _size) in enumerate(self.written, 1)
        }
        for old_name, new_name in rename_map.items():
            os.replace(self.out / old_name, self.out / new_name)
        self.output_weight_map = {
            key: rename_map[shard_name]
            for key, shard_name in self.output_weight_map.items()
        }
        return self.output_weight_map


def add_quantized_weight(
    writer: ShardWriter,
    key: str,
    quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    prefix = key.removesuffix(".weight")
    qweight, weight_scale, weight_scale_2 = quantized
    writer.add(key, qweight)
    writer.add(prefix + ".weight_scale", weight_scale)
    writer.add(prefix + ".weight_scale_2", weight_scale_2)


def process_shards(
    source: Path,
    out: Path,
    weight_map: dict[str, str],
    quantized_weight_keys: set[str],
    block_size: int,
    device: torch.device,
    target_shard_size: int,
) -> dict[str, str]:
    by_shard = keys_by_shard(weight_map)
    shard_names = sorted(by_shard)
    writer = ShardWriter(out, target_shard_size)
    processed_quantized: set[str] = set()

    with TensorStore(source, weight_map) as store:
        for shard_idx, shard_name in enumerate(shard_names, 1):
            shard_quantized = 0
            for key in by_shard[shard_name]:
                if key in processed_quantized:
                    continue
                if key not in quantized_weight_keys:
                    writer.add(key, store.get_tensor(key))
                    continue

                layer, expert, proj = routed_key_parts(key)
                if proj in {"gate_proj", "up_proj"}:
                    gate_key = routed_weight_key(layer, expert, "gate_proj")
                    up_key = routed_weight_key(layer, expert, "up_proj")
                    if gate_key not in quantized_weight_keys or up_key not in quantized_weight_keys:
                        raise RuntimeError(f"Missing gate/up pair for {key}")
                    gate_quantized, up_quantized = quantize_nvfp4_gate_up_pair(
                        store.get_tensor(gate_key),
                        store.get_tensor(up_key),
                        block_size=block_size,
                        device=device,
                    )
                    add_quantized_weight(writer, gate_key, gate_quantized)
                    add_quantized_weight(writer, up_key, up_quantized)
                    processed_quantized.update({gate_key, up_key})
                    shard_quantized += 2
                else:
                    qweight, weight_scale, weight_scale_2 = quantize_nvfp4_weight(
                        store.get_tensor(key),
                        block_size=block_size,
                        device=device,
                    )
                    add_quantized_weight(writer, key, (qweight, weight_scale, weight_scale_2))
                    processed_quantized.add(key)
                    shard_quantized += 1

            print(
                f"[src {shard_idx:03d}/{len(shard_names):03d}] {shard_name}: "
                f"quantized {shard_quantized} routed expert weight(s)",
                flush=True,
            )

    if processed_quantized != quantized_weight_keys:
        missing = sorted(quantized_weight_keys - processed_quantized)
        raise RuntimeError(f"Did not quantize {len(missing)} selected weight(s): {missing[:8]}")

    output_weight_map = writer.finalize()
    print(f"Quantized routed expert weights: {len(processed_quantized)}", flush=True)
    print(f"Model tensor shards written: {len(set(output_weight_map.values()))}", flush=True)
    return output_weight_map


def add_input_scales_to_weight_map(
    weight_map: dict[str, str],
    scale_keys: set[str],
) -> None:
    for key in sorted(scale_keys):
        weight_map[key] = SCALE_SHARD_NAME


def write_updated_config(
    source_config: dict[str, Any],
    quantization_config: dict[str, Any],
    out: Path,
) -> None:
    config = merge_config_with_quantization_config(source_config, quantization_config)
    write_json(out / "config.json", config)


def write_updated_index(out: Path, source_index: dict[str, Any], weight_map: dict[str, str]) -> None:
    metadata = dict(source_index.get("metadata", {}))
    metadata["total_size"] = sum((out / shard).stat().st_size for shard in set(weight_map.values()))
    index = {
        "metadata": metadata,
        "weight_map": dict(sorted(weight_map.items())),
    }
    write_json(out / "model.safetensors.index.json", index)


def prepare_output(out: Path, overwrite: bool) -> None:
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise SystemExit("--block-size must be positive")

    source_index = read_json(args.source / "model.safetensors.index.json")
    source_config = read_json(args.source / "config.json")
    source_weight_map = source_index["weight_map"]

    validate_source_files(args.source, source_index)
    scale_keys = load_scale_keys(args.scale_source)
    quantization_config = load_quantization_config(args.quant_config_template)
    quantization_config = adapt_quantization_config_for_glm52_mtp(
        quantization_config,
        source_config,
    )
    merge_config_with_quantization_config(source_config, quantization_config)
    quantized_weight_keys = routed_expert_weight_keys(source_weight_map, source_config)
    validate_scales(scale_keys, quantized_weight_keys)

    mtp_selected = [
        key for key in source_weight_map
        if key.startswith(f"model.layers.{source_config['num_hidden_layers']}.")
        and key in quantized_weight_keys
    ]
    if mtp_selected:
        raise RuntimeError("MTP layer was selected unexpectedly")

    shard_count = len(set(source_weight_map.values()))
    aux_items = auxiliary_items(args.source)
    print(f"source: {args.source}")
    print(f"output: {args.out}")
    print(f"source shards: {shard_count}")
    print(f"target model shard size: {args.shard_size / 1024**3:.2f} GiB")
    print(f"routed expert weights selected: {len(quantized_weight_keys)}")
    print(f"input scales copied: {len(scale_keys)} from {args.scale_source}")
    print(f"MTP layer kept BF16: model.layers.{source_config['num_hidden_layers']}")
    print(f"source auxiliary files/dirs copied: {describe_auxiliary_items(aux_items)}")
    print(
        "config.json: source config with NVFP4 quantization_config "
        f"from {args.quant_config_template}"
    )
    print("model.safetensors.index.json: regenerated for the output tensors")

    if args.dry_run:
        print("Dry run complete; no files written.")
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    prepare_output(args.out, args.overwrite)
    copied_aux = copy_auxiliary_files(args.source, args.out)
    print(f"Copied {len(copied_aux)} source auxiliary file(s)/dir(s).")
    copy_input_scales(args.scale_source, args.out)

    output_weight_map = process_shards(
        args.source,
        args.out,
        source_weight_map,
        set(quantized_weight_keys),
        args.block_size,
        device,
        args.shard_size,
    )
    add_input_scales_to_weight_map(output_weight_map, scale_keys)
    write_updated_config(source_config, quantization_config, args.out)
    write_updated_index(args.out, source_index, output_weight_map)

    print(f"Wrote GLM-5.2 routed-expert NVFP4 checkpoint to {args.out}")


if __name__ == "__main__":
    main()
