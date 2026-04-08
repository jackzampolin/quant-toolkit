from .base import ModelQuantConfig


Glm5Config = ModelQuantConfig(
    model_id="zai-org/GLM-5",
    trust_remote_code=True,
    streaming=True,
    extra_quant_overrides={
        "*indexer*": {"enable": False},
    },
    extra_mtp_prefixes=["model.layers.78."],
)


def _register_moe():
    from moe_registry import register_glm5_moe_for_quantization
    register_glm5_moe_for_quantization()


Glm5Config.register_moe = _register_moe
