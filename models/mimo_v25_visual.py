import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor

from .mimo_v25_media import (
    MIMO_VIDEO_FPS,
    MIMO_VIDEO_MAX_FRAMES,
    MIMO_VIDEO_MIN_FRAMES,
)


_MIMO_IMAGE_MEAN = [0.485, 0.456, 0.406]
_MIMO_IMAGE_STD = [0.229, 0.224, 0.225]


def resolve_model_dir(model_id: str | Path) -> Path:
    path = Path(model_id).expanduser()
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(model_id),
            allow_patterns=["*.json", "*.safetensors"],
            local_files_only=True,
        )
    )


def mimo_visual_checkpoint_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        files = sorted(
            {
                model_dir / filename
                for key, filename in weight_map.items()
                if key.startswith("visual.")
            }
        )
        files = [path for path in files if path.exists()]
        if files:
            return files
    return sorted(model_dir.glob("*.safetensors"))


def load_mimo_standalone_visual(
    model_id: str | Path,
    model_config,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    *,
    log_file=None,
):
    from .mimo_v25_remote.modeling_mimo_v2 import MiMoVisionTransformer, _as_namespace

    model_dir = resolve_model_dir(model_id)
    visual = MiMoVisionTransformer(_as_namespace(model_config.vision_config))
    state_dict = {}
    loaded_files = []

    for filename in mimo_visual_checkpoint_files(model_dir):
        with safe_open(filename, framework="pt", device="cpu") as sf:
            visual_keys = [key for key in sf.keys() if key.startswith("visual.")]
            if not visual_keys:
                continue
            loaded_files.append(filename.name)
            for key in visual_keys:
                state_dict[key.removeprefix("visual.")] = sf.get_tensor(key)

    if not state_dict:
        raise RuntimeError(f"No visual.* tensors found under {model_dir}")

    with torch.no_grad():
        for name, param in visual.named_parameters():
            if name.endswith(".bias"):
                param.zero_()

    missing, unexpected = visual.load_state_dict(state_dict, strict=False)
    allowed_missing = {"merger.mlp.0.bias", "merger.mlp.2.bias"}
    allowed_unexpected = {"merger.ln_q.bias"}
    bad_missing = sorted(set(missing) - allowed_missing)
    bad_unexpected = sorted(set(unexpected) - allowed_unexpected)
    if bad_missing:
        raise RuntimeError(f"Missing MiMo visual checkpoint keys: {bad_missing[:16]}")
    if bad_unexpected:
        raise RuntimeError(f"Unexpected MiMo visual checkpoint keys: {bad_unexpected[:16]}")

    visual.to(device=device, dtype=dtype)
    if hasattr(visual.patch_embed, "sync_proj_weight_linear_format"):
        visual.patch_embed.sync_proj_weight_linear_format()
    visual.eval()

    if log_file is not None:
        print(
            f"Loaded standalone MiMo visual tower from {', '.join(sorted(set(loaded_files)))}",
            file=log_file,
        )
    return visual


def build_mimo_processor(tokenizer, model_config):
    vision_config = getattr(model_config, "vision_config", None) or {}
    processor_config = getattr(model_config, "processor_config", None) or {}
    patch_size = vision_config.get("patch_size", vision_config.get("spatial_patch_size", 16))
    temporal_patch_size = vision_config.get("temporal_patch_size", 2)
    merge_size = vision_config.get("spatial_merge_size", 2)

    image_processor = Qwen2VLImageProcessor(
        image_mean=_MIMO_IMAGE_MEAN,
        image_std=_MIMO_IMAGE_STD,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
        size={
            "shortest_edge": processor_config.get("image_min_pixels", 8192),
            "longest_edge": processor_config.get("image_max_pixels", 8388608),
        },
    )
    video_processor = Qwen2VLVideoProcessor(
        image_mean=_MIMO_IMAGE_MEAN,
        image_std=_MIMO_IMAGE_STD,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
        do_sample_frames=True,
        fps=MIMO_VIDEO_FPS,
        max_frames=MIMO_VIDEO_MAX_FRAMES,
        min_frames=MIMO_VIDEO_MIN_FRAMES,
        num_frames=processor_config.get("num_frames"),
        size={
            "shortest_edge": processor_config.get("video_min_pixels", 8192),
            "longest_edge": processor_config.get("video_max_pixels", 8388608),
        },
    )
    return Qwen2_5_VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    if min(height, width) < factor:
        scale = factor / min(height, width)
        height = int(round(height * scale))
        width = int(round(width * scale))
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / beta / factor) * factor
        resized_width = math.floor(width / beta / factor) * factor
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return int(resized_height), int(resized_width)


def mimo_image_pixel_values(image, model_config) -> tuple[torch.Tensor, torch.Tensor]:
    vision_config = getattr(model_config, "vision_config", None) or {}
    processor_config = getattr(model_config, "processor_config", None) or {}
    patch_size = int(
        vision_config.get(
            "patch_size",
            vision_config.get("spatial_patch_size", processor_config.get("patch_size", 16)),
        )
    )
    temporal_patch_size = int(vision_config.get("temporal_patch_size", processor_config.get("temporal_patch_size", 2)))
    merge_size = int(vision_config.get("spatial_merge_size", processor_config.get("merge_size", 2)))
    min_pixels = int(processor_config.get("image_min_pixels", 8192))
    max_pixels = int(processor_config.get("image_max_pixels", 8388608))

    image = image.convert("RGB")
    width, height = image.size
    resized_height, resized_width = _smart_resize(
        height=height,
        width=width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    image_tensor = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    mean = torch.tensor([123.675, 116.28, 103.53], dtype=image_tensor.dtype).view(-1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=image_tensor.dtype).view(-1, 1, 1)
    image_tensor = (image_tensor - mean) / std

    patches = image_tensor.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)
    channels = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    patches = patches.contiguous().view(
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
    pixel_values = patches.view(
        grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    )
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int32)
    return pixel_values, image_grid_thw


def replace_mimo_image_pixels(batch: dict, images: list, model_config) -> None:
    if not images:
        return

    pixel_values = []
    grids = []
    for image in images:
        image_pixels, image_grid = mimo_image_pixel_values(image, model_config)
        pixel_values.append(image_pixels)
        grids.append(image_grid)

    pixel_values = torch.cat(pixel_values, dim=0)
    image_grid_thw = torch.cat(grids, dim=0)
    if "image_grid_thw" in batch and not torch.equal(batch["image_grid_thw"].cpu().to(image_grid_thw.dtype), image_grid_thw):
        raise ValueError(
            f"MiMo image preprocessing grid mismatch: processor grid {batch['image_grid_thw'].tolist()} "
            f"!= MiMo grid {image_grid_thw.tolist()}"
        )
    batch["pixel_values"] = pixel_values
    batch["image_grid_thw"] = image_grid_thw


def compute_mimo_visual_embeds(
    model_id: str | Path,
    model_config,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    *,
    log_file=sys.stderr,
) -> torch.Tensor:
    visual = load_mimo_standalone_visual(
        model_id,
        model_config,
        device=device,
        dtype=dtype,
        log_file=log_file,
    )
    try:
        with torch.inference_mode():
            return visual(
                pixel_values=pixel_values.to(device=device),
                grid_thw=grid_thw.to(device=device),
            )
    finally:
        del visual
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _modal_token_id(model_config, attr_name: str):
    token_id = getattr(model_config, attr_name, None)
    if token_id is not None:
        return token_id
    processor_config = getattr(model_config, "processor_config", None) or {}
    return processor_config.get(attr_name)


def _validate_modal_embed_count(batch: dict, token_id, embeds: torch.Tensor, name: str) -> None:
    if token_id is None or "input_ids" not in batch:
        return
    expected = int(batch["input_ids"].eq(int(token_id)).sum().item())
    actual = int(embeds.shape[0])
    if expected != actual:
        raise RuntimeError(
            f"MiMo {name} calibration mismatch: prompt has {expected} placeholder tokens, "
            f"but visual tower produced {actual} embeddings"
        )


def precompute_mimo_visual_embeds_for_batches(
    model_id: str | Path,
    model_config,
    batches: list[tuple[int, dict]],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    log_file=sys.stdout,
) -> int:
    needs_visual = any(
        ("pixel_values" in batch and "image_embeds" not in batch)
        or ("video_pixel_values" in batch and "video_embeds" not in batch)
        for _, batch in batches
    )
    if not needs_visual:
        return 0

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    visual = load_mimo_standalone_visual(
        model_id,
        model_config,
        device=device,
        dtype=dtype,
        log_file=log_file,
    )
    converted = 0

    try:
        with torch.inference_mode():
            for _, batch in batches:
                if "pixel_values" in batch and "image_embeds" not in batch:
                    if "image_grid_thw" not in batch:
                        raise RuntimeError("MiMo image batch has pixel_values but no image_grid_thw")
                    image_embeds = visual(
                        pixel_values=batch["pixel_values"].to(device=device),
                        grid_thw=batch["image_grid_thw"].to(device=device),
                    )
                    _validate_modal_embed_count(
                        batch,
                        _modal_token_id(model_config, "image_token_id"),
                        image_embeds,
                        "image",
                    )
                    batch["image_embeds"] = image_embeds.detach().cpu()
                    del batch["pixel_values"]
                    converted += 1

                if "video_pixel_values" in batch and "video_embeds" not in batch:
                    if "video_grid_thw" not in batch:
                        raise RuntimeError("MiMo video batch has video_pixel_values but no video_grid_thw")
                    video_embeds = visual(
                        pixel_values=batch["video_pixel_values"].to(device=device),
                        grid_thw=batch["video_grid_thw"].to(device=device),
                    )
                    _validate_modal_embed_count(
                        batch,
                        _modal_token_id(model_config, "video_token_id"),
                        video_embeds,
                        "video",
                    )
                    batch["video_embeds"] = video_embeds.detach().cpu()
                    del batch["video_pixel_values"]
                    converted += 1
    finally:
        del visual
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for _, batch in batches:
        if "pixel_values" in batch or "video_pixel_values" in batch:
            raise RuntimeError("MiMo visual precompute failed: raw pixel tensors remain in calibration batch")

    return converted
