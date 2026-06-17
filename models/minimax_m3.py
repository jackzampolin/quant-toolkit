from .base import ModelQuantConfig


class _MinimaxM3Config(ModelQuantConfig):
    def get_model_cls(self):
        try:
            from transformers import MiniMaxM3SparseForConditionalGeneration
        except ImportError:
            from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
                MiniMaxM3SparseForConditionalGeneration,
            )

        return MiniMaxM3SparseForConditionalGeneration

    def register_moe(self):
        from moe_registry import register_minimax_m3_moe_for_quantization

        register_minimax_m3_moe_for_quantization()


MinimaxM3Config = _MinimaxM3Config(
    model_id="MiniMaxAI/MiniMax-M3",
    # The model class comes from upstream Transformers. The published HF repo
    # still uses remote processor code for image/video token expansion.
    trust_remote_code=False,
    processor_trust_remote_code=True,
    streaming=True,
    preserve_remote_code=True,
    extra_quant_overrides={
        # Expert-only NVFP4: disable everything first, then re-enable only the
        # routed expert linears after ModelOpt unfuses MiniMaxM3VLExperts into
        # projection-first ModuleLists.
        "*weight_quantizer": {"enable": False},
        "*input_quantizer": {"enable": False},
        "model.language_model.layers.*.mlp.experts.gate_proj.*.weight_quantizer": {"enable": True},
        "model.language_model.layers.*.mlp.experts.gate_proj.*.input_quantizer": {"enable": True},
        "model.language_model.layers.*.mlp.experts.up_proj.*.weight_quantizer": {"enable": True},
        "model.language_model.layers.*.mlp.experts.up_proj.*.input_quantizer": {"enable": True},
        "model.language_model.layers.*.mlp.experts.down_proj.*.weight_quantizer": {"enable": True},
        "model.language_model.layers.*.mlp.experts.down_proj.*.input_quantizer": {"enable": True},
        # Keep all non-routed-expert paths in source precision, including media
        # towers/projectors, attention/indexers, routers, dense MLPs, shared
        # experts, embeddings, and the LM head.
        "*[kv]_bmm_quantizer": {"enable": False},
        "*vision_tower*": {"enable": False},
        "*multi_modal_projector*": {"enable": False},
        "*self_attn*": {"enable": False},
        "*indexer*": {"enable": False},
        "*embed_tokens*": {"enable": False},
        "*lm_head*": {"enable": False},
        "*shared_expert*": {"enable": False},
        "*shared_experts*": {"enable": False},
        "model.language_model.layers.*.mlp.gate*": {"enable": False},
        "model.language_model.layers.*.mlp.gate_up_proj*": {"enable": False},
        "model.language_model.layers.*.mlp.down_proj*": {"enable": False},
    },
)
