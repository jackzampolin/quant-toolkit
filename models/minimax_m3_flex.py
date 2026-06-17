"""MiniMax M3 sparse-attention FlexAttention backend.

This registers a custom Transformers attention implementation name that keeps
MiniMax M3's upstream sparse-indexer semantics but avoids expanding the selected
KV blocks into a dense block-selection mask before attention.
"""

from __future__ import annotations

import os

import torch


MINIMAX_M3_FLEX_ATTN = "minimax_m3_flex"
_COMPILED_FLEX_ATTENTION_CACHE = {}


def _compiled_flex_attention_for(query: torch.Tensor, training: bool):
    from torch.nn.attention.flex_attention import flex_attention
    from transformers.utils.import_utils import is_torchdynamo_compiling

    if is_torchdynamo_compiling():
        return flex_attention

    device_index = query.device.index if query.device.index is not None else -1
    cache_key = (query.device.type, device_index, bool(training))
    compiled = _COMPILED_FLEX_ATTENTION_CACHE.get(cache_key)
    if compiled is None:
        compiled = torch.compile(flex_attention)
        _COMPILED_FLEX_ATTENTION_CACHE[cache_key] = compiled
    return compiled


def _expand_position_ids(
    position_ids: torch.Tensor | None,
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
) -> torch.Tensor:
    if position_ids is None:
        start = key_length - query_length
        position_ids = torch.arange(start, key_length, device=device).unsqueeze(0)
    elif position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)
    return position_ids.to(device=device).expand(batch_size, -1)


def _minimax_m3_flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    block_indices: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """AttentionInterface callback for MiniMax M3 sparse layers.

    ``block_indices`` is MiniMax M3's per-query-token list of selected KV block
    IDs. Flex schedules work per query tile, so we schedule the union of selected
    KV blocks for every query tile, then ``score_mod`` filters each token back to
    its exact selected blocks plus the model-level causal/padding mask.
    """

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if block_indices is None:
        sdpa_attention_forward = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    if dropout > 0:
        raise ValueError("MiniMax M3 FlexAttention calibration path expects inference/eval dropout=0.")

    from transformers.integrations.flex_attention import repeat_kv
    from torch.nn.attention.flex_attention import BlockMask

    batch_size, num_query_heads, query_length, _ = query.shape
    key_length = key.shape[-2]
    block_size = int(getattr(module.indexer, "block_size", 128))
    query_block_size = block_size
    num_key_blocks = -(-key_length // block_size)

    valid_blocks = block_indices >= 0
    safe_blocks = block_indices.masked_fill(~valid_blocks, num_key_blocks)
    token_block_keep = block_indices.new_zeros(
        (batch_size, query_length, num_key_blocks + 1),
        dtype=torch.bool,
    )
    token_block_keep.scatter_(-1, safe_blocks, True)
    token_block_keep = token_block_keep[..., :num_key_blocks].contiguous()

    num_query_blocks = -(-query_length // query_block_size)
    query_pad = num_query_blocks * query_block_size - query_length
    if query_pad:
        token_block_keep_for_tiles = torch.nn.functional.pad(token_block_keep, (0, 0, 0, query_pad))
    else:
        token_block_keep_for_tiles = token_block_keep
    tile_block_keep = token_block_keep_for_tiles.view(
        batch_size,
        num_query_blocks,
        query_block_size,
        num_key_blocks,
    ).any(dim=2)

    kv_num_blocks = tile_block_keep.sum(dim=-1, dtype=torch.int32).unsqueeze(1).contiguous()
    kv_indices = torch.argsort(
        tile_block_keep.to(torch.int32),
        dim=-1,
        descending=True,
        stable=True,
    ).to(torch.int32).unsqueeze(1).contiguous()
    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        BLOCK_SIZE=(query_block_size, block_size),
        seq_lengths=(query_length, key_length),
        compute_q_blocks=False,
    )

    score_mask = None
    if attention_mask is not None:
        score_mask = attention_mask[:, :, :, :key_length]
    position_ids = None
    if score_mask is None:
        position_ids = _expand_position_ids(
            kwargs.get("position_ids"),
            batch_size=batch_size,
            query_length=query_length,
            key_length=key_length,
            device=query.device,
        )

    def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        selected = token_block_keep[batch_idx][q_idx][kv_idx // block_size]
        if score_mask is not None:
            mask_value = score_mask[batch_idx][0][q_idx][kv_idx]
            if score_mask.dtype == torch.bool:
                return torch.where(selected & mask_value, score, torch.finfo(score.dtype).min)
            return torch.where(selected, score + mask_value, torch.finfo(score.dtype).min)
        allowed = kv_idx <= position_ids[batch_idx][q_idx]
        return torch.where(selected & allowed, score, torch.finfo(score.dtype).min)

    enable_gqa = True
    if (num_query_heads & (num_query_heads - 1)) != 0:
        key = repeat_kv(key, query.shape[1] // key.shape[1])
        value = repeat_kv(value, query.shape[1] // value.shape[1])
        enable_gqa = False

    flex_attention = _compiled_flex_attention_for(query, module.training)
    attn_output = flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=kwargs.get("kernel_options"),
    )
    return attn_output.transpose(1, 2).contiguous(), None


def register_minimax_m3_flex_attention() -> None:
    _set_dynamo_recompile_limits()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if MINIMAX_M3_FLEX_ATTN not in ALL_ATTENTION_FUNCTIONS.valid_keys():
        ALL_ATTENTION_FUNCTIONS.register(MINIMAX_M3_FLEX_ATTN, _minimax_m3_flex_attention_forward)


def _set_dynamo_recompile_limits() -> None:
    """Keep per-device/per-shape Flex recompiles from disabling the backend."""

    try:
        import torch._dynamo.config as dynamo_config
    except ImportError:
        return

    recompile_limit = int(os.environ.get("TORCHDYNAMO_RECOMPILE_LIMIT", "100000"))
    accumulated_limit = int(os.environ.get("TORCHDYNAMO_ACCUMULATED_RECOMPILE_LIMIT", str(recompile_limit)))
    dynamo_config.recompile_limit = recompile_limit
    dynamo_config.cache_size_limit = recompile_limit
    dynamo_config.accumulated_recompile_limit = accumulated_limit
    dynamo_config.accumulated_cache_size_limit = accumulated_limit


def assert_attention_implementation(model: torch.nn.Module, expected: str) -> None:
    """Fail fast if a loaded model did not keep the requested attention backend."""

    configs = []
    seen = set()

    def add_config(label: str, config) -> None:
        if config is None or id(config) in seen:
            return
        seen.add(id(config))
        configs.append((label, config))
        for sub_name in getattr(config, "sub_configs", {}) or {}:
            add_config(f"{label}.{sub_name}", getattr(config, sub_name, None))

    add_config("model.config", getattr(model, "config", None))
    base = getattr(model, "model", None)
    add_config("model.model.config", getattr(base, "config", None))
    add_config("model.model.language_model.config", getattr(getattr(base, "language_model", None), "config", None))
    add_config("model.language_model.config", getattr(getattr(model, "language_model", None), "config", None))

    mismatches = []
    for label, config in configs:
        actual = getattr(config, "_attn_implementation", None)
        if actual is not None and actual != expected:
            mismatches.append(f"{label}={actual!r}")

    if mismatches:
        raise RuntimeError(
            f"Expected attention implementation {expected!r}, but loaded: "
            + ", ".join(mismatches)
        )
