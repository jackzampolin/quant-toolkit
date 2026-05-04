#!/usr/bin/env python3
"""Run MiMo's vision tower only and print tensor stats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mimo_v25_remote.configuration_mimo_v2 import MiMoV2Config  # noqa: E402
from models.mimo_v25_remote.modeling_mimo_v2 import (  # noqa: E402
    MiMoVisionTransformer,
    _as_namespace,
)
from tools.mimo_forward_smoke import (  # noqa: E402
    _deinterleave_grouped_qkv_rows,
    _mimo_image_pixel_values,
    _tensor_stats,
)


def _print_stats(stage: str, **values) -> None:
    stats = []
    for name, value in values.items():
        if torch.is_tensor(value):
            stats.append(_tensor_stats(name, value))
    print(f"[MiMo vision stats] {stage}: {json.dumps(stats)}")


def _checkpoint_files(model_id: Path) -> list[Path]:
    index_path = model_id / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        files = sorted({model_id / filename for key, filename in weight_map.items() if key.startswith("visual.")})
        files = [path for path in files if path.exists()]
        if files:
            return files
    return sorted(model_id.glob("*.safetensors"))


def _deinterleave_tp_qkv_rows(
    tensor: torch.Tensor,
    *,
    tp_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    q_rows = num_heads * head_dim
    kv_rows = num_kv_heads * head_dim
    expected_rows = q_rows + 2 * kv_rows
    if tensor.shape[0] != expected_rows:
        raise ValueError(f"Visual QKV shape mismatch: got {tensor.shape[0]}, expected {expected_rows}")
    if num_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
        raise ValueError(f"Visual QKV TP{tp_size} layout requires divisible heads/KV heads")

    q_rank_rows = q_rows // tp_size
    kv_rank_rows = kv_rows // tp_size
    rank_rows = q_rank_rows + 2 * kv_rank_rows
    rest_shape = tensor.shape[1:]
    ranked = tensor.reshape(tp_size, rank_rows, *rest_shape)
    q = ranked[:, :q_rank_rows].reshape(q_rows, *rest_shape)
    k = ranked[:, q_rank_rows : q_rank_rows + kv_rank_rows].reshape(kv_rows, *rest_shape)
    v = ranked[:, q_rank_rows + kv_rank_rows :].reshape(kv_rows, *rest_shape)
    return torch.cat([q, k, v], dim=0).contiguous()


def _normalize_visual_state_dict_qkv(state_dict: dict[str, torch.Tensor], config, layout: str, tp_size: int) -> None:
    if layout == "canonical":
        return

    num_heads = int(getattr(config, "num_heads"))
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = int(getattr(config, "qk_channels", getattr(config, "kv_channels", 64)))

    for key in list(state_dict):
        if ".attn.qkv." not in key:
            continue
        if layout == "tp":
            state_dict[key] = _deinterleave_tp_qkv_rows(
                state_dict[key],
                tp_size=tp_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        elif layout == "grouped":
            state_dict[key] = _deinterleave_grouped_qkv_rows(
                state_dict[key],
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        else:
            raise ValueError(f"Unsupported visual QKV layout: {layout}")


def _load_visual_weights(
    visual: MiMoVisionTransformer,
    model_id: Path,
    *,
    visual_qkv_layout: str,
    visual_qkv_tp_size: int,
) -> None:
    state_dict = {}
    loaded_keys = []
    for filename in _checkpoint_files(model_id):
        with safe_open(filename, framework="pt", device="cpu") as sf:
            visual_keys = [key for key in sf.keys() if key.startswith("visual.")]
            if not visual_keys:
                continue
            for key in visual_keys:
                state_dict[key.removeprefix("visual.")] = sf.get_tensor(key)
                loaded_keys.append(key)

    if not state_dict:
        raise RuntimeError(f"No visual.* tensors found under {model_id}")

    _normalize_visual_state_dict_qkv(
        state_dict,
        visual.config,
        layout=visual_qkv_layout,
        tp_size=visual_qkv_tp_size,
    )

    with torch.no_grad():
        for name, param in visual.named_parameters():
            if name.endswith(".bias"):
                param.zero_()

    missing, unexpected = visual.load_state_dict(state_dict, strict=False)
    allowed_unexpected = {
        "merger.ln_q.bias",
    }
    unhandled_unexpected = sorted(set(unexpected) - allowed_unexpected)
    if unhandled_unexpected:
        raise RuntimeError(f"Unexpected visual checkpoint keys: {unhandled_unexpected[:8]}")

    allowed_missing = {
        "merger.mlp.0.bias",
        "merger.mlp.2.bias",
    }
    unhandled_missing = sorted(set(missing) - allowed_missing)
    if unhandled_missing:
        raise RuntimeError(f"Missing visual checkpoint keys: {unhandled_missing[:16]}")

    if hasattr(visual.patch_embed, "sync_proj_weight_linear_format"):
        visual.patch_embed.sync_proj_weight_linear_format()

    print(
        f"Loaded {len(loaded_keys)} visual tensors from "
        f"{', '.join(path.name for path in _checkpoint_files(model_id))}",
        file=sys.stderr,
    )
    if missing:
        print(f"Zero-filled missing visual tensors: {', '.join(sorted(missing))}", file=sys.stderr)
    ignored = sorted(set(unexpected) & allowed_unexpected)
    if ignored:
        print(f"Ignored unused visual checkpoint tensors: {', '.join(ignored)}", file=sys.stderr)


def _apply_vision_attention_ablation(visual: MiMoVisionTransformer, mode: str) -> None:
    if mode == "correct":
        return
    if mode == "no-sinks":
        for block in visual.blocks:
            block.attn.sinks = None
        return
    if mode == "full-no-sinks":
        visual.fullatt_block_indexes = list(range(len(visual.blocks)))
        for block in visual.blocks:
            block.attn.sinks = None
        return
    raise ValueError(f"Unsupported vision attention mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="/data/models/MiMo-V2.5-BF16-qkv-deinterleaved")
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--visual-qkv-layout", choices=["canonical", "tp", "grouped"], default="canonical")
    parser.add_argument("--visual-qkv-tp-size", type=int, default=4)
    parser.add_argument(
        "--vision-attn-ablation",
        choices=["correct", "no-sinks", "full-no-sinks"],
        default="correct",
    )
    args = parser.parse_args()

    model_id = Path(args.model_id)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    config = MiMoV2Config.from_pretrained(model_id)
    visual = MiMoVisionTransformer(_as_namespace(config.vision_config))
    _load_visual_weights(
        visual,
        model_id,
        visual_qkv_layout=args.visual_qkv_layout,
        visual_qkv_tp_size=args.visual_qkv_tp_size,
    )
    visual.to(device=device, dtype=dtype)
    if hasattr(visual.patch_embed, "sync_proj_weight_linear_format"):
        visual.patch_embed.sync_proj_weight_linear_format()
    _apply_vision_attention_ablation(visual, args.vision_attn_ablation)
    visual.eval()

    image = Image.open(args.image).convert("RGB")
    pixel_values, image_grid_thw = _mimo_image_pixel_values(image, config)
    pixel_values = pixel_values.to(device=device)
    image_grid_thw = image_grid_thw.to(device=device)

    _print_stats(
        "model.image",
        pixel_values=pixel_values.to(dtype=visual.dtype),
        image_grid_thw=image_grid_thw,
    )
    with torch.inference_mode():
        image_embeds = visual(pixel_values=pixel_values, grid_thw=image_grid_thw)
    _print_stats("model.image_embeds", image_embeds=image_embeds)


if __name__ == "__main__":
    main()
