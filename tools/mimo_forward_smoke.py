#!/usr/bin/env python3
"""Run a simple MiMo generation smoke test."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import load_config  # noqa: E402
from models.mimo_v25_visual import (  # noqa: E402
    build_mimo_processor,
    compute_mimo_visual_embeds,
    mimo_image_pixel_values,
)
from models.mimo_v25_media import (  # noqa: E402
    expand_audio_placeholders,
    has_audio_track,
    mimo_video_processor_kwargs,
    normalize_processor_inputs,
    prepare_audio_codes,
    processor_config_value,
)


def _build_messages(prompt: str, image_path: str | None, video_path: str | None, audio_paths: list[str]):
    if image_path is None and video_path is None and not audio_paths:
        content = prompt
    else:
        content = []
        if image_path is not None:
            content.append({"type": "image", "image": image_path})
        if video_path is not None:
            content.append({"type": "video", "video": video_path})
        for audio_path in audio_paths:
            content.append({"type": "audio", "audio": audio_path})
        content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _build_prompt(tokenizer_or_processor, messages, enable_thinking):
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer_or_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def _move_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _resolve_default_audio_path() -> str:
    audio_dir_env = os.environ.get("MIMO_AUDIO_CALIB_DIR")
    if audio_dir_env:
        audio_dir = Path(audio_dir_env).expanduser()
        for suffix in ("*.wav", "*.flac", "*.mp3"):
            matches = sorted(audio_dir.glob(suffix))
            if matches:
                return str(matches[0])
        raise FileNotFoundError(f"No audio files found under {audio_dir}")
    raise FileNotFoundError("Pass --audio PATH or set MIMO_AUDIO_CALIB_DIR")


def _tensor_stats(name: str, value: torch.Tensor) -> dict:
    flat = value.detach().flatten()
    stats = {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": value.numel(),
    }
    if flat.numel() == 0:
        return stats

    stats["head"] = flat[:8].cpu().tolist()
    if torch.is_floating_point(value):
        work = flat.float()
        stats.update(
            {
                "min": float(work.min().item()),
                "max": float(work.max().item()),
                "mean": float(work.mean().item()),
                "std": float(work.std(unbiased=False).item()),
                "sum": float(work.sum().item()),
            }
        )
    else:
        work = flat.to(torch.int64)
        stats.update(
            {
                "min": int(work.min().item()),
                "max": int(work.max().item()),
                "sum": int(work.sum().item()),
            }
        )
    return stats


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _print_preprocess_stats(stage: str, **values):
    stats = []
    for name, value in values.items():
        if torch.is_tensor(value):
            stats.append(_tensor_stats(name, value))
        elif value is None:
            continue
        elif isinstance(value, (int, float, str, bool)):
            stats.append({"name": name, "value": value})
        elif isinstance(value, (list, tuple)):
            stats.append({"name": name, "len": len(value), "head": _jsonable(value[:16])})
    print(f"[MiMo preprocess stats] {stage}: {json.dumps(stats)}", file=sys.stderr)


def _mm_token_offsets(input_ids: torch.Tensor, token_id: int) -> list[tuple[int, int]]:
    flat = input_ids.flatten().cpu()
    mask = flat == token_id
    start_positions = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
    end_positions = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]
    return list(zip(start_positions.tolist(), end_positions.tolist()))


def _replace_modal_embeddings_for_stats(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    token_id: int,
    modal_embeds: torch.Tensor,
) -> torch.Tensor:
    mask = input_ids.eq(token_id)
    num_slots = int(mask.sum().item())
    if num_slots != modal_embeds.shape[0]:
        raise ValueError(
            f"Modal embedding count mismatch for token_id={token_id}: "
            f"found {num_slots} placeholders but got {modal_embeds.shape[0]} embeddings."
        )
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[mask] = modal_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    return inputs_embeds


def _log_local_backbone_input_stats(model, prompt, model_config):
    input_ids = prompt["input_ids"]
    attention_mask = prompt["attention_mask"]
    position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.long).unsqueeze(0)

    inputs_embeds = model.model.get_input_embeddings()(input_ids)
    image_token_id = getattr(model_config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor_config_value(model_config, "image_token_id")
    video_token_id = getattr(model_config, "video_token_id", None)
    if video_token_id is None:
        video_token_id = processor_config_value(model_config, "video_token_id")

    values = {
        "input_ids": input_ids.flatten(),
        "attention_mask": attention_mask.flatten(),
        "position_ids": position_ids.flatten(),
    }
    if prompt.get("mrope_positions") is not None:
        values["mrope_positions"] = prompt["mrope_positions"]
    if prompt.get("mrope_position_delta") is not None:
        values["mrope_position_delta"] = prompt["mrope_position_delta"]

    if prompt["image_embeds"] is not None:
        if image_token_id is None:
            raise ValueError("Cannot log image backbone stats without image_token_id")
        offsets = _mm_token_offsets(input_ids, int(image_token_id))
        image_embeds = prompt["image_embeds"]
        inputs_embeds = _replace_modal_embeddings_for_stats(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            token_id=int(image_token_id),
            modal_embeds=image_embeds,
        )
        values["image_offsets"] = offsets
        values["image_embeds"] = image_embeds
        if offsets:
            image_slices = [inputs_embeds.reshape(-1, inputs_embeds.shape[-1])[start : end + 1] for start, end in offsets]
            values["input_embeds.image_spans"] = torch.cat(image_slices, dim=0)

    if prompt.get("video_embeds") is not None:
        if video_token_id is None:
            raise ValueError("Cannot log video backbone stats without video_token_id")
        offsets = _mm_token_offsets(input_ids, int(video_token_id))
        video_embeds = prompt["video_embeds"]
        inputs_embeds = _replace_modal_embeddings_for_stats(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            token_id=int(video_token_id),
            modal_embeds=video_embeds,
        )
        values["video_offsets"] = offsets
        values["video_embeds"] = video_embeds
        if offsets:
            video_slices = [inputs_embeds.reshape(-1, inputs_embeds.shape[-1])[start : end + 1] for start, end in offsets]
            values["input_embeds.video_spans"] = torch.cat(video_slices, dim=0)

    if prompt.get("audio_codes") is not None or prompt.get("audio_embeds") is not None:
        audio_token_id = getattr(model_config, "audio_token_id", None)
        if audio_token_id is None:
            audio_token_id = processor_config_value(model_config, "audio_token_id")
        if audio_token_id is None:
            raise ValueError("Cannot log audio backbone stats without audio_token_id")

        with torch.inference_mode():
            if prompt.get("audio_embeds") is not None:
                audio_embeds = prompt["audio_embeds"]
            else:
                audio_embeds = model.audio_encoder(
                    speech_embeddings=model.speech_embeddings,
                    audio_codes=prompt["audio_codes"],
                )
        offsets = _mm_token_offsets(input_ids, int(audio_token_id))
        inputs_embeds = _replace_modal_embeddings_for_stats(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            token_id=int(audio_token_id),
            modal_embeds=audio_embeds,
        )
        values["audio_offsets"] = offsets
        values["audio_embeds"] = audio_embeds
        if prompt.get("audio_codes") is not None:
            values["audio_codes"] = prompt["audio_codes"]
        if offsets:
            audio_slices = [inputs_embeds.reshape(-1, inputs_embeds.shape[-1])[start : end + 1] for start, end in offsets]
            values["input_embeds.audio_spans"] = torch.cat(audio_slices, dim=0)

    values["input_embeds"] = inputs_embeds
    _print_preprocess_stats("backbone.input", **values)


def _input_device(model):
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _zero_known_mimo_missing_biases(model):
    names = [
        "visual.merger.mlp.0.bias",
        "visual.merger.mlp.2.bias",
    ]
    zeroed = []
    for name in names:
        try:
            param = model.get_parameter(name)
        except (AttributeError, ValueError):
            continue
        param.data.zero_()
        zeroed.append(name)


def _deinterleave_grouped_qkv_rows(tensor, num_heads: int, num_kv_heads: int, head_dim: int):
    q_per_group = num_heads // num_kv_heads
    q_group = q_per_group * head_dim
    kv_group = head_dim
    group = q_group + 2 * kv_group
    expected_rows = num_kv_heads * group
    if tensor.shape[0] != expected_rows:
        raise ValueError(
            f"Grouped visual QKV shape mismatch: got first dim {tensor.shape[0]}, expected {expected_rows}"
        )

    rest_shape = tensor.shape[1:]
    grouped = tensor.reshape(num_kv_heads, group, *rest_shape)
    q = grouped[:, :q_group].reshape(num_heads * head_dim, *rest_shape)
    k = grouped[:, q_group : q_group + kv_group].reshape(num_kv_heads * head_dim, *rest_shape)
    v = grouped[:, q_group + kv_group :].reshape(num_kv_heads * head_dim, *rest_shape)
    return torch.cat([q, k, v], dim=0).contiguous()


def _normalize_visual_qkv_layout(model, model_config, layout: str):
    if layout == "canonical":
        return
    if layout != "grouped":
        raise ValueError(f"Unsupported visual QKV layout: {layout}")

    vision_config = getattr(model_config, "vision_config", None) or {}
    num_heads = int(vision_config.get("num_heads", 32))
    num_kv_heads = int(vision_config.get("num_key_value_heads", num_heads))
    head_dim = int(vision_config.get("qk_channels", vision_config.get("kv_channels", 64)))
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")

    visual = getattr(model, "visual", None)
    if visual is None:
        return
    for block in visual.blocks:
        qkv = block.attn.qkv
        with torch.no_grad():
            qkv.weight.copy_(_deinterleave_grouped_qkv_rows(qkv.weight.data, num_heads, num_kv_heads, head_dim))
            if qkv.bias is not None:
                qkv.bias.copy_(_deinterleave_grouped_qkv_rows(qkv.bias.data, num_heads, num_kv_heads, head_dim))


def _apply_vision_attention_ablation(model, mode: str):
    if mode == "correct":
        return
    if mode != "full-no-sinks":
        raise ValueError(f"Unsupported vision attention ablation: {mode}")

    visual = getattr(model, "visual", None)
    if visual is None:
        return

    for block in visual.blocks:
        attn = block.attn
        attn.window_size = -1
        attn.sinks = None


def _prepare_prompt(
    model,
    model_config,
    tokenizer,
    processor,
    prompt_text,
    image_path,
    video_path,
    audio_codes,
    max_input_tokens,
    device,
    log_preprocess_stats,
    model_id,
):
    if image_path is None and video_path is None:
        batch = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        return {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "pixel_values": None,
            "image_grid_thw": None,
            "image_embeds": None,
            "video_pixel_values": None,
            "video_grid_thw": None,
            "video_embeds": None,
            "audio_codes": audio_codes.to(device) if audio_codes is not None else None,
            "audio_embeds": None,
            "mrope_positions": None,
            "mrope_position_delta": None,
        }

    images = [Image.open(image_path).convert("RGB")] if image_path is not None else None
    videos = [str(Path(video_path).expanduser())] if video_path is not None else None
    processor_kwargs = {
        "text": [prompt_text],
        "return_tensors": "pt",
        "truncation": True,
        "max_length": max_input_tokens,
    }
    if images is not None:
        processor_kwargs["images"] = images
    if videos is not None:
        processor_kwargs["videos"] = videos
        processor_kwargs.update(mimo_video_processor_kwargs())
    batch = processor(**processor_kwargs)
    batch = normalize_processor_inputs(batch)

    if images is not None:
        pixel_values, image_grid_thw = mimo_image_pixel_values(images[0], model_config)
        if "image_grid_thw" in batch and not torch.equal(batch["image_grid_thw"].cpu().to(image_grid_thw.dtype), image_grid_thw):
            raise ValueError(
                f"MiMo image preprocessing grid mismatch: tokenizer grid {batch['image_grid_thw'].tolist()} "
                f"!= MiMo grid {image_grid_thw.tolist()}"
            )
        batch["pixel_values"] = pixel_values
        batch["image_grid_thw"] = image_grid_thw

    if log_preprocess_stats:
        image_token_id = getattr(model_config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = processor_config_value(model_config, "image_token_id")
        video_token_id = getattr(model_config, "video_token_id", None)
        if video_token_id is None:
            video_token_id = processor_config_value(model_config, "video_token_id")
        _print_preprocess_stats(
            "processor.output",
            input_ids=batch.get("input_ids").flatten(),
            mrope_positions=batch.get("mrope_positions"),
            mrope_position_delta=batch.get("mrope_position_delta"),
        )
        if images is not None:
            _print_preprocess_stats(
                "processor.image",
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                image_offsets=_mm_token_offsets(batch["input_ids"], int(image_token_id)),
            )
        if videos is not None:
            _print_preprocess_stats(
                "processor.video",
                video_pixel_values=batch.get("video_pixel_values"),
                video_grid_thw=batch.get("video_grid_thw"),
                video_offsets=_mm_token_offsets(batch["input_ids"], int(video_token_id)),
            )
    batch = _move_to_device(batch, device)
    image_embeds = None
    video_embeds = None
    if images is not None:
        image_embeds = compute_mimo_visual_embeds(
            model_id=model_id,
            model_config=model_config,
            pixel_values=batch["pixel_values"],
            grid_thw=batch["image_grid_thw"],
            device=device,
            log_file=sys.stderr,
        )
    if videos is not None:
        if "video_pixel_values" not in batch or "video_grid_thw" not in batch:
            raise ValueError("Video preprocessing did not produce video_pixel_values and video_grid_thw")
        video_embeds = compute_mimo_visual_embeds(
            model_id=model_id,
            model_config=model_config,
            pixel_values=batch["video_pixel_values"],
            grid_thw=batch["video_grid_thw"],
            device=device,
            log_file=sys.stderr,
        )
    if log_preprocess_stats:
        if images is not None:
            _print_preprocess_stats(
                "model.image",
                pixel_values=batch["pixel_values"].to(dtype=torch.bfloat16),
                image_grid_thw=batch["image_grid_thw"],
            )
            _print_preprocess_stats("model.image_embeds", image_embeds=image_embeds)
        if videos is not None:
            _print_preprocess_stats(
                "model.video",
                video_pixel_values=batch["video_pixel_values"].to(dtype=torch.bfloat16),
                video_grid_thw=batch["video_grid_thw"],
            )
            _print_preprocess_stats("model.video_embeds", video_embeds=video_embeds)

    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "pixel_values": batch.get("pixel_values"),
        "image_grid_thw": batch.get("image_grid_thw"),
        "image_embeds": image_embeds,
        "video_pixel_values": batch.get("video_pixel_values"),
        "video_grid_thw": batch.get("video_grid_thw"),
        "video_embeds": video_embeds,
        "audio_codes": audio_codes.to(device) if audio_codes is not None else None,
        "audio_embeds": None,
        "mrope_positions": batch.get("mrope_positions"),
        "mrope_position_delta": batch.get("mrope_position_delta"),
    }


def _decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _add_prefill_modal_inputs(model_inputs: dict, prompt: dict) -> None:
    if prompt["image_embeds"] is not None:
        model_inputs["image_embeds"] = prompt["image_embeds"]
    elif prompt["pixel_values"] is not None:
        model_inputs["pixel_values"] = prompt["pixel_values"]
        model_inputs["image_grid_thw"] = prompt["image_grid_thw"]
    if prompt.get("video_embeds") is not None:
        model_inputs["video_embeds"] = prompt["video_embeds"]
    elif prompt.get("video_pixel_values") is not None:
        model_inputs["video_pixel_values"] = prompt["video_pixel_values"]
        model_inputs["video_grid_thw"] = prompt["video_grid_thw"]
    if prompt.get("audio_codes") is not None:
        model_inputs["audio_codes"] = prompt["audio_codes"]
    if prompt.get("audio_embeds") is not None:
        model_inputs["audio_embeds"] = prompt["audio_embeds"]


def _greedy_full_context_loop(model, tokenizer, prompt, max_new_tokens):
    input_ids = prompt["input_ids"]
    attention_mask = prompt["attention_mask"]
    generated = []

    for _ in range(max_new_tokens):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "logits_to_keep": 1,
        }
        _add_prefill_modal_inputs(model_inputs, prompt)

        with torch.inference_mode():
            logits = model(**model_inputs).logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated.append(token_id)

        sys.stdout.write(_decode_token(tokenizer, token_id))
        sys.stdout.flush()

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        if token_id == tokenizer.eos_token_id:
            break

    return generated


def _greedy_kv_cache_loop(model, tokenizer, prompt, max_new_tokens):
    if max_new_tokens <= 0:
        return []

    input_ids = prompt["input_ids"]
    attention_mask = prompt["attention_mask"]
    generated = []

    prefill_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "logits_to_keep": 1,
    }
    _add_prefill_modal_inputs(prefill_inputs, prompt)

    with torch.inference_mode():
        outputs = model(**prefill_inputs)

    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        sys.stdout.write(_decode_token(tokenizer, token_id))
        sys.stdout.flush()

        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        if token_id == tokenizer.eos_token_id:
            break

        with torch.inference_mode():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return generated


def _compare_logits(cache_logits: torch.Tensor, full_logits: torch.Tensor) -> dict:
    cache = cache_logits.float().flatten()
    full = full_logits.float().flatten()
    diff = cache - full
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "cosine": float(F.cosine_similarity(cache, full, dim=0).item()),
        "cache_argmax": int(cache.argmax().item()),
        "full_argmax": int(full.argmax().item()),
    }


def _greedy_compare_cache_loop(model, tokenizer, prompt, max_new_tokens):
    if max_new_tokens <= 0:
        return []

    full_input_ids = prompt["input_ids"]
    full_attention_mask = prompt["attention_mask"]
    cache_attention_mask = prompt["attention_mask"]
    generated = []

    prefill_inputs = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "use_cache": True,
        "logits_to_keep": 1,
    }
    full_inputs = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "use_cache": False,
        "logits_to_keep": 1,
    }
    _add_prefill_modal_inputs(prefill_inputs, prompt)
    _add_prefill_modal_inputs(full_inputs, prompt)

    with torch.inference_mode():
        cache_outputs = model(**prefill_inputs)
        full_outputs = model(**full_inputs)

    past_key_values = cache_outputs.past_key_values
    cache_logits = cache_outputs.logits[:, -1, :]
    full_logits = full_outputs.logits[:, -1, :]

    for step in range(max_new_tokens):
        stats = _compare_logits(cache_logits, full_logits)
        print(
            "step={step} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
            "cosine={cosine:.9f} cache_argmax={cache_argmax} full_argmax={full_argmax}".format(
                step=step,
                **stats,
            ),
            file=sys.stderr,
        )

        next_token = cache_logits.argmax(dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated.append(token_id)
        sys.stdout.write(_decode_token(tokenizer, token_id))
        sys.stdout.flush()

        full_input_ids = torch.cat([full_input_ids, next_token], dim=-1)
        full_attention_mask = torch.cat([full_attention_mask, torch.ones_like(next_token)], dim=-1)
        cache_attention_mask = torch.cat([cache_attention_mask, torch.ones_like(next_token)], dim=-1)
        if token_id == tokenizer.eos_token_id:
            break

        with torch.inference_mode():
            cache_outputs = model(
                input_ids=next_token,
                attention_mask=cache_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            full_inputs = {
                "input_ids": full_input_ids,
                "attention_mask": full_attention_mask,
                "use_cache": False,
                "logits_to_keep": 1,
            }
            _add_prefill_modal_inputs(full_inputs, prompt)
            full_outputs = model(**full_inputs)

        past_key_values = cache_outputs.past_key_values
        cache_logits = cache_outputs.logits[:, -1, :]
        full_logits = full_outputs.logits[:, -1, :]

    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Explain why large language model checkpoint integrity matters.")
    parser.add_argument("--model", default="mimo_v25", help="Repo model config name.")
    parser.add_argument("--model-id", default="/data/models/MiMo-V2.5-BF16-qkv-deinterleaved",
                        help="HF model id or local checkpoint path.")
    parser.add_argument("--image", default=None, help="Optional image path to include in the user message.")
    parser.add_argument("--video", default=None, help="Optional video path to include in the user message.")
    parser.add_argument("--video-use-audio", choices=["auto", "yes", "no"], default="auto",
                        help="Whether to include a video's audio track as MiMo audio tokens.")
    parser.add_argument("--audio", nargs="?", const="__default__", default=None,
                        help="Optional audio path to include in the user message. With no value, uses the first calibration WAV.")
    parser.add_argument("--audio-tokenizer-dir", default=None,
                        help="Optional path to MiMo's audio_tokenizer sidecar directory.")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--decode-mode", choices=["full", "kv-cache", "compare-cache"], default="full",
                        help="Generation loop to use. compare-cache prints cached-vs-full logits stats to stderr.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="auto",
                        help="Transformers attention backend override. Defaults to auto, which omits the override.")
    parser.add_argument("--visual-qkv-layout", choices=["canonical", "grouped"], default="canonical",
                        help="How visual attn.qkv rows are stored on disk before the vendored model splits Q/K/V.")
    parser.add_argument("--vision-attn-ablation", choices=["correct", "full-no-sinks"], default="correct",
                        help="Optionally force MiMo vision attention to full attention without sink logits.")
    parser.add_argument("--log-preprocess-stats", action="store_true",
                        help="Print multimodal preprocessing tensor metadata/stats before generation.")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking_group.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.model)
    trust_remote = cfg.trust_remote_code

    model_cls = cfg.get_model_cls()
    cls = model_cls
    model_config = cls.config_class.from_pretrained(args.model_id)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=trust_remote,
        config=model_config,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = None
    prompt_renderer = tokenizer
    if args.image is not None or args.video is not None:
        try:
            processor = AutoProcessor.from_pretrained(
                args.model_id,
                trust_remote_code=trust_remote,
                config=model_config,
            )
        except OSError:
            processor = build_mimo_processor(tokenizer, model_config)
        if getattr(processor, "chat_template", None) is None:
            processor.chat_template = tokenizer.chat_template

    no_split = ["MiMoV2DecoderLayer", "MiMoV2MoE"]
    existing = list(getattr(cls, "_no_split_modules", []) or [])
    cls._no_split_modules = sorted(set(existing + no_split))
    model_kwargs = {
        "config": model_config,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": trust_remote,
        "device_map": args.device_map,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = cls.from_pretrained(
        args.model_id,
        **model_kwargs,
    )
    _zero_known_mimo_missing_biases(model)
    _normalize_visual_qkv_layout(model, model_config, args.visual_qkv_layout)
    _apply_vision_attention_ablation(model, args.vision_attn_ablation)
    model.eval()

    audio_paths = []
    if args.video is not None:
        video_path = str(Path(args.video).expanduser())
        if args.video_use_audio == "yes":
            audio_paths.append(video_path)
        elif args.video_use_audio == "auto" and has_audio_track(video_path):
            audio_paths.append(video_path)
    if args.audio == "__default__":
        audio_paths.append(_resolve_default_audio_path())
    elif args.audio is not None:
        audio_paths.append(args.audio)

    device = _input_device(model)
    audio_codes, audio_counts = prepare_audio_codes(
        model=model,
        model_config=model_config,
        model_id=args.model_id,
        audio_paths=audio_paths,
        audio_tokenizer_dir=args.audio_tokenizer_dir,
        device=device,
        log_fn=_print_preprocess_stats if args.log_preprocess_stats else None,
    )

    messages = _build_messages(args.prompt, args.image, args.video, audio_paths)
    prompt_text = _build_prompt(prompt_renderer, messages, args.enable_thinking)
    if audio_counts:
        prompt_text = expand_audio_placeholders(prompt_text, audio_counts)

    prompt = _prepare_prompt(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        processor=processor,
        prompt_text=prompt_text,
        image_path=args.image,
        video_path=args.video,
        audio_codes=audio_codes,
        max_input_tokens=args.max_input_tokens,
        device=device,
        log_preprocess_stats=args.log_preprocess_stats,
        model_id=args.model_id,
    )
    if args.log_preprocess_stats:
        _log_local_backbone_input_stats(model, prompt, model_config)
    if args.decode_mode == "full":
        _greedy_full_context_loop(model, tokenizer, prompt, args.max_new_tokens)
    elif args.decode_mode == "kv-cache":
        _greedy_kv_cache_loop(model, tokenizer, prompt, args.max_new_tokens)
    elif args.decode_mode == "compare-cache":
        _greedy_compare_cache_loop(model, tokenizer, prompt, args.max_new_tokens)
    else:
        raise ValueError(f"Unsupported decode mode: {args.decode_mode}")


if __name__ == "__main__":
    main()
