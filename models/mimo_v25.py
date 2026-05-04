from .base import ModelQuantConfig


class _MimoV25Config(ModelQuantConfig):
    def get_model_cls(self):
        from transformers_compat import ensure_mimo_transformers_compat

        ensure_mimo_transformers_compat()

        from .mimo_v25_remote import MiMoV2ForCausalLM

        return MiMoV2ForCausalLM


MimoV25Config = _MimoV25Config(
    model_id="XiaomiMiMo/MiMo-V2.5",
    trust_remote_code=False,
    streaming=True,
    preserve_remote_code=True,
    extra_quant_overrides={
        # MiMo-V2.5 calibration should only quantize the routed main-backbone
        # expert MLP linears. Disable globally first, then re-enable exactly
        # the per-expert gate/up/down projections below.
        "*weight_quantizer": {"enable": False},
        "*input_quantizer": {"enable": False},
        "model.layers.*.mlp.experts.*.gate_proj.weight_quantizer": {"enable": True},
        "model.layers.*.mlp.experts.*.gate_proj.input_quantizer": {"enable": True},
        "model.layers.*.mlp.experts.*.up_proj.weight_quantizer": {"enable": True},
        "model.layers.*.mlp.experts.*.up_proj.input_quantizer": {"enable": True},
        "model.layers.*.mlp.experts.*.down_proj.weight_quantizer": {"enable": True},
        "model.layers.*.mlp.experts.*.down_proj.input_quantizer": {"enable": True},
        # MiMo-V2.5 is omnimodal; keep all media towers/tokenizers/projections
        # in source precision. Video uses the visual tower, so visual covers it.
        "*visual*": {"enable": False},
        "*audio_encoder*": {"enable": False},
        "*audio_tokenizer*": {"enable": False},
        "*speech_embeddings*": {"enable": False},
        "*video*": {"enable": False},
        # Keep embeddings, head, routers, dense MLPs, and any shared experts in
        # source precision. MiMo currently has no shared expert module, but this
        # makes the intent explicit if the class changes.
        "*embed_tokens*": {"enable": False},
        "*lm_head*": {"enable": False},
        "*shared_expert*": {"enable": False},
        "*shared_experts*": {"enable": False},
        "model.layers.*.mlp.gate*": {"enable": False},
        "model.layers.*.mlp.gate_proj*": {"enable": False},
        "model.layers.*.mlp.up_proj*": {"enable": False},
        "model.layers.*.mlp.down_proj*": {"enable": False},
        # Router parameters are not Linear modules, but keep any matched gate
        # quantizers disabled if ModelOpt starts wrapping them.
        "*mlp.gate*weight_quantizer": {"enable": False},
        "*mlp.gate*input_quantizer": {"enable": False},
        # Custom MiMo attention has separate full/SWA head dimensions and sink
        # bias. Leave KV cache quantization off unless it is validated directly.
        "*[kv]_bmm_quantizer": {"enable": False},
    },
)
