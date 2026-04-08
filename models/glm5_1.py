from .base import ModelQuantConfig


class _Glm51Config(ModelQuantConfig):
    def register_moe(self):
        from moe_registry import register_glm5_moe_for_quantization
        register_glm5_moe_for_quantization()


Glm51Config = _Glm51Config(
    model_id="zai-org/GLM-5.1",
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
    },
    extra_mtp_prefixes=["model.layers.78."],
)
