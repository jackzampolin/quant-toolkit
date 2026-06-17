from .base import ModelQuantConfig


class _Glm52Config(ModelQuantConfig):
    def register_moe(self):
        from moe_registry import register_glm5_moe_for_quantization
        register_glm5_moe_for_quantization()


Glm52Config = _Glm52Config(
    model_id="zai-org/GLM-5.2",
    trust_remote_code=True,
    streaming=True,
    extra_quant_overrides={
        "*indexer*": {"enable": False},
        "*shared_experts.gate_proj*weight_quantizer": {"enable": False},
        "*shared_experts.gate_proj*input_quantizer": {"enable": False},
        "*shared_experts.up_proj*weight_quantizer": {"enable": False},
        "*shared_experts.up_proj*input_quantizer": {"enable": False},
        "*shared_experts.down_proj*weight_quantizer": {"enable": False},
        "*shared_experts.down_proj*input_quantizer": {"enable": False},
        # Disable KV cache quantization.
        "*[kv]_bmm_quantizer": {"enable": False},
        # Dense MLP layers (0-2 have no MoE).
        "*layers.0.mlp*weight_quantizer": {"enable": False},
        "*layers.0.mlp*input_quantizer": {"enable": False},
        "*layers.1.mlp*weight_quantizer": {"enable": False},
        "*layers.1.mlp*input_quantizer": {"enable": False},
        "*layers.2.mlp*weight_quantizer": {"enable": False},
        "*layers.2.mlp*input_quantizer": {"enable": False},
    },
    extra_mtp_prefixes=["model.layers.78."],
)
