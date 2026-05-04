"""Shared MiMo-V2.5 multimodal preparation helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch


BASE_MIMO_MODEL_ID = "XiaomiMiMo/MiMo-V2.5"
MIMO_VIDEO_FPS = 2.0
MIMO_VIDEO_MAX_FRAMES = 768
MIMO_VIDEO_MIN_FRAMES = 4


def mimo_video_processor_kwargs() -> dict:
    """Match SGLang's MiMo video sampling defaults."""
    return {
        "do_sample_frames": True,
        "fps": MIMO_VIDEO_FPS,
        "max_frames": MIMO_VIDEO_MAX_FRAMES,
        "min_frames": MIMO_VIDEO_MIN_FRAMES,
    }


def normalize_processor_inputs(batch):
    """Normalize processor output keys to the MiMo remote-model forward names."""
    if "pixel_values_videos" in batch and "video_pixel_values" not in batch:
        batch["video_pixel_values"] = batch.pop("pixel_values_videos")
    return batch


def processor_config_value(model_config, key: str, default=None):
    processor_config = getattr(model_config, "processor_config", None) or {}
    if isinstance(processor_config, dict):
        return processor_config.get(key, default)
    return getattr(processor_config, key, default)


def preprocess_kwargs(part: dict) -> dict:
    kwargs = part.get("preprocess_kwargs") or {}
    if not isinstance(kwargs, dict):
        raise TypeError(f"preprocess_kwargs must be a dict, got {type(kwargs).__name__}")
    return kwargs


def has_audio_track(media_ref) -> bool:
    if not isinstance(media_ref, str):
        return False
    path = Path(media_ref).expanduser()
    probe_target = str(path) if path.exists() else media_ref
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "a",
                probe_target,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found; install ffmpeg to auto-detect video audio tracks") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out while checking audio track for {media_ref}") from exc

    if result.returncode != 0:
        return False
    try:
        return bool(json.loads(result.stdout).get("streams"))
    except json.JSONDecodeError:
        return False


def video_audio_source(part: dict, video_path: str):
    return part.get("audio") or part.get("audio_file") or part.get("audio_url") or video_path


def video_uses_audio(part: dict, video_path: str) -> bool:
    kwargs = preprocess_kwargs(part)
    if "use_audio" in kwargs:
        return bool(kwargs["use_audio"])
    if part.get("audio") or part.get("audio_file") or part.get("audio_url"):
        return True
    return has_audio_track(video_path)


def audio_group_size(model_config) -> int:
    audio_config = getattr(model_config, "audio_config", None) or {}
    if isinstance(audio_config, dict):
        return int(audio_config.get("group_size", 4))
    return int(getattr(audio_config, "group_size", 4))


def audio_placeholder_count_from_codes(audio_codes: torch.Tensor, model_config) -> int:
    if audio_codes.dim() == 1:
        audio_codes = audio_codes.unsqueeze(-1)
    group_size = audio_group_size(model_config)
    return (audio_codes.shape[0] + group_size - 1) // group_size


def pad_audio_codes_to_group_boundary(audio_codes: torch.Tensor, model_config) -> torch.Tensor:
    if audio_codes.dim() == 1:
        audio_codes = audio_codes.unsqueeze(-1)
    if audio_codes.dim() != 2:
        raise ValueError(f"audio_codes must be 2D [T, C], got {tuple(audio_codes.shape)}")
    if audio_codes.shape[0] == 0:
        raise ValueError("audio_codes must not be empty")

    group_size = audio_group_size(model_config)
    padded_len = ((audio_codes.shape[0] + group_size - 1) // group_size) * group_size
    if padded_len == audio_codes.shape[0]:
        return audio_codes
    pad = audio_codes[-1:].expand(padded_len - audio_codes.shape[0], -1)
    return torch.cat([audio_codes, pad], dim=0)


def expand_audio_placeholders(text: str, counts: list[int]) -> str:
    audio_pad = "<|audio_pad|>"
    for count in counts:
        if count <= 0:
            continue
        if audio_pad not in text:
            raise ValueError("Audio sample has codes/embeds but chat template produced no <|audio_pad|> token")
        text = text.replace(audio_pad, audio_pad * count, 1)
    return text


def resolve_mimo_audio_tokenizer_dir(
    model_id: str,
    override: str | None = None,
    fallback_model_id: str = BASE_MIMO_MODEL_ID,
) -> Path:
    if override is not None:
        path = Path(override).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Audio tokenizer directory does not exist: {path}")
        return path

    from huggingface_hub import snapshot_download

    candidates = []
    errors = []
    model_path = Path(model_id).expanduser()
    if model_path.is_dir():
        candidates.append(model_path / "audio_tokenizer")
    else:
        try:
            candidates.append(
                Path(
                    snapshot_download(
                        model_id,
                        allow_patterns=["audio_tokenizer/*"],
                        local_files_only=True,
                    )
                )
                / "audio_tokenizer"
            )
        except Exception as exc:  # noqa: BLE001 - keep the final error actionable.
            errors.append(f"{model_id}: {exc}")

    # Local dequantized MiMo checkpoints usually omit this sidecar. It is
    # architecture metadata, so the base MiMo snapshot is a valid fallback.
    if fallback_model_id != model_id:
        try:
            candidates.append(
                Path(
                    snapshot_download(
                        fallback_model_id,
                        allow_patterns=["audio_tokenizer/*"],
                        local_files_only=True,
                    )
                )
                / "audio_tokenizer"
            )
        except Exception as exc:  # noqa: BLE001 - keep the final error actionable.
            errors.append(f"{fallback_model_id}: {exc}")

    audio_tokenizer_dir = next((path for path in candidates if path.exists()), None)
    if audio_tokenizer_dir is None:
        searched = ", ".join(str(path) for path in candidates) or model_id
        detail = f" Errors: {'; '.join(errors)}" if errors else ""
        raise FileNotFoundError(f"No audio_tokenizer directory found. Searched: {searched}.{detail}")
    return audio_tokenizer_dir


def ensure_mimo_audio_tokenizer(
    model,
    model_id: str,
    audio_tokenizer_dir: str | None = None,
    device: torch.device | None = None,
    fallback_model_id: str = BASE_MIMO_MODEL_ID,
    log_file=sys.stderr,
):
    if not hasattr(model, "load_audio_tokenizer"):
        raise RuntimeError("This model does not expose load_audio_tokenizer(); provide audio_codes or audio_embeds instead")
    if getattr(model, "audio_tokenizer", None) is not None:
        return model.audio_tokenizer

    resolved = resolve_mimo_audio_tokenizer_dir(
        model_id=model_id,
        override=audio_tokenizer_dir,
        fallback_model_id=fallback_model_id,
    )
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_audio_tokenizer(str(resolved), device=device, dtype=torch.bfloat16)
    print(f"Loaded MiMo audio tokenizer from {resolved}", file=log_file)
    return model.audio_tokenizer


def load_audio_mel(audio_path: str, audio_tokenizer) -> torch.Tensor:
    import torchaudio

    cfg = audio_tokenizer.config
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != cfg.sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, cfg.sampling_rate)

    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sampling_rate,
        n_fft=cfg.nfft,
        win_length=cfg.window_size,
        hop_length=cfg.hop_length,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    mel = mel_fn(waveform).squeeze(0).transpose(0, 1)
    return torch.log(torch.clamp(mel, min=1e-5))


def audio_codes_from_mels(model, mels: list[torch.Tensor]) -> list[torch.Tensor]:
    audio_tokenizer = model.audio_tokenizer
    modeling_module = sys.modules[type(model).__module__]
    tokenize_audio_batch = getattr(modeling_module, "tokenize_audio_batch")
    segment_size = getattr(getattr(model, "audio_encoder", None), "audio_segment_size", 6000)
    codes = tokenize_audio_batch(
        mels,
        audio_tokenizer.encoder,
        segment_size=segment_size,
        device=next(audio_tokenizer.parameters()).device,
    )
    return [code.detach().cpu().long() for code in codes]


def prepare_audio_codes(
    model,
    model_config,
    model_id: str,
    audio_paths: list[str],
    audio_tokenizer_dir: str | None = None,
    device: torch.device | None = None,
    fallback_model_id: str = BASE_MIMO_MODEL_ID,
    log_fn: Callable[..., None] | None = None,
) -> tuple[torch.Tensor | None, list[int]]:
    if not audio_paths:
        return None, []

    audio_tokenizer = ensure_mimo_audio_tokenizer(
        model=model,
        model_id=model_id,
        audio_tokenizer_dir=audio_tokenizer_dir,
        device=device,
        fallback_model_id=fallback_model_id,
    )
    mels = [load_audio_mel(str(Path(audio_path).expanduser()), audio_tokenizer) for audio_path in audio_paths]
    codes = [pad_audio_codes_to_group_boundary(c, model_config) for c in audio_codes_from_mels(model, mels)]
    counts = [audio_placeholder_count_from_codes(audio_codes, model_config) for audio_codes in codes]
    if log_fn is not None:
        for idx, (mel, audio_codes) in enumerate(zip(mels, codes)):
            log_fn(f"processor.audio[{idx}]", audio_mel=mel, audio_codes=audio_codes)
    return torch.cat(codes, dim=0), counts
