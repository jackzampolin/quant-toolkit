#!/usr/bin/env python3
"""Post-export MiMo-V2.5 NVFP4 fixups.

This is intentionally narrow:
  - report whether the visual/audio towers are present in the exported index
  - copy the BF16 dequantized MTP tensors into model-mtp.safetensors
  - repoint the exported safetensors index at that dedicated MTP shard
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_SOURCE = Path("/data/models/MiMo-V2.5-BF16-qkv-deinterleaved")
DEFAULT_EXPORT = Path("/data/models/MiMo-V2.5-NVFP4")
MTP_SHARD_NAME = "model-mtp.safetensors"
MEDIA_PREFIXES = ("visual.", "audio_encoder.", "speech_embeddings.")
MTP_PREFIX = "model.mtp."


def _load_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    with index_path.open() as f:
        return json.load(f)


def _prefix_keys(weight_map: dict[str, str], prefix: str) -> list[str]:
    return sorted(k for k in weight_map if k.startswith(prefix))


def _report_prefixes(label: str, weight_map: dict[str, str], prefixes: tuple[str, ...]) -> None:
    print(f"\n{label}:")
    for prefix in prefixes:
        keys = _prefix_keys(weight_map, prefix)
        shards = sorted({weight_map[k] for k in keys})
        shard_text = ", ".join(shards[:6])
        if len(shards) > 6:
            shard_text += f", ... ({len(shards)} shards)"
        print(f"  {prefix:<18} {len(keys):>5} tensor(s)  [{shard_text}]")


def _load_source_tensors(source_dir: Path, source_weight_map: dict[str, str], keys: list[str]) -> dict[str, torch.Tensor]:
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        keys_by_shard[source_weight_map[key]].append(key)

    tensors: dict[str, torch.Tensor] = {}
    for shard_name, shard_keys in sorted(keys_by_shard.items()):
        shard_path = source_dir / shard_name
        print(f"  loading {len(shard_keys)} tensor(s) from {shard_name}")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in sorted(shard_keys):
                tensors[key] = f.get_tensor(key).contiguous()
    return tensors


def _rewrite_index(export_dir: Path, index: dict, weight_map: dict[str, str]) -> None:
    index["weight_map"] = dict(sorted(weight_map.items()))
    metadata = index.setdefault("metadata", {})
    metadata["total_size"] = sum((export_dir / shard).stat().st_size for shard in set(weight_map.values()))
    with (export_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")


def _copy_mtp(source_dir: Path, export_dir: Path, source_weight_map: dict[str, str], export_index: dict) -> None:
    export_weight_map = export_index["weight_map"]
    mtp_keys = _prefix_keys(source_weight_map, MTP_PREFIX)
    if not mtp_keys:
        raise RuntimeError(f"No source MTP tensors found with prefix {MTP_PREFIX!r}")

    print(f"\nCopying {len(mtp_keys)} source BF16 MTP tensor(s) to {MTP_SHARD_NAME}")
    tensors = _load_source_tensors(source_dir, source_weight_map, mtp_keys)
    missing = sorted(set(mtp_keys) - set(tensors))
    if missing:
        raise RuntimeError(f"Failed to load {len(missing)} MTP tensors; first missing: {missing[0]}")

    shard_path = export_dir / MTP_SHARD_NAME
    if shard_path.exists():
        shard_path.unlink()
    save_file(tensors, str(shard_path))

    for key in mtp_keys:
        export_weight_map[key] = MTP_SHARD_NAME
    _rewrite_index(export_dir, export_index, export_weight_map)

    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"  wrote {len(tensors)} tensor(s), {total_bytes / (1024 ** 3):.2f} GiB payload")
    print(f"  updated {export_dir / 'model.safetensors.index.json'}")


def _describe_mtp(source_dir: Path, source_weight_map: dict[str, str]) -> None:
    mtp_keys = _prefix_keys(source_weight_map, MTP_PREFIX)
    layer_re = re.compile(r"^model\.mtp\.layers\.(\d+)\.")
    layers = sorted({int(m.group(1)) for key in mtp_keys if (m := layer_re.match(key))})

    print("\nMTP architecture from source tensors:")
    print(f"  layers: {layers}")
    print(f"  tensors: {len(mtp_keys)}")
    for layer in layers:
        interesting = [
            f"model.mtp.layers.{layer}.self_attn.qkv_proj.weight",
            f"model.mtp.layers.{layer}.self_attn.o_proj.weight",
            f"model.mtp.layers.{layer}.self_attn.attention_sink_bias",
            f"model.mtp.layers.{layer}.mlp.gate_proj.weight",
            f"model.mtp.layers.{layer}.mlp.up_proj.weight",
            f"model.mtp.layers.{layer}.mlp.down_proj.weight",
            f"model.mtp.layers.{layer}.eh_proj.weight",
        ]
        print(f"  layer {layer}:")
        opened: dict[str, safe_open] = {}
        try:
            for key in interesting:
                shard_name = source_weight_map.get(key)
                if shard_name is None:
                    continue
                if shard_name not in opened:
                    opened[shard_name] = safe_open(source_dir / shard_name, framework="pt", device="cpu")
                    opened[shard_name].__enter__()
                tensor_slice = opened[shard_name].get_slice(key)
                short_key = key.removeprefix(f"model.mtp.layers.{layer}.")
                print(f"    {short_key:<36} {tensor_slice.get_shape()} {tensor_slice.get_dtype()}")
        finally:
            for handle in opened.values():
                handle.__exit__(None, None, None)

    print("  note: qkv_proj is full-attention-shaped and each MTP layer has attention_sink_bias.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write model-mtp.safetensors.")
    args = parser.parse_args()

    source_dir = args.source_dir
    export_dir = args.export_dir
    source_index = _load_index(source_dir)
    export_index = _load_index(export_dir)
    source_weight_map = source_index["weight_map"]
    export_weight_map = export_index["weight_map"]

    _report_prefixes("Source checkpoint", source_weight_map, MEDIA_PREFIXES + (MTP_PREFIX,))
    _report_prefixes("Export checkpoint before fixup", export_weight_map, MEDIA_PREFIXES + (MTP_PREFIX,))
    _describe_mtp(source_dir, source_weight_map)

    if args.dry_run:
        print("\nDry run; no files written.")
        return

    _copy_mtp(source_dir, export_dir, source_weight_map, export_index)
    fixed_index = _load_index(export_dir)
    _report_prefixes("Export checkpoint after fixup", fixed_index["weight_map"], MEDIA_PREFIXES + (MTP_PREFIX,))


if __name__ == "__main__":
    main()
