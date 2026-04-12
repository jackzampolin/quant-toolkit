from .base import ModelQuantConfig


MinimaxM27Config = ModelQuantConfig(
    model_id="/data/models/MiniMax-M2.7-BF16",
    trust_remote_code=True,
    streaming=True,
    extra_quant_overrides={
        "*[kv]_bmm_quantizer": {"enable": False},
        "*gate*weight_quantizer": {"enable": False},
        "*gate*input_quantizer": {"enable": False},
        "*lm_head*weight_quantizer": {"enable": False},
        "*lm_head*input_quantizer": {"enable": False},
    },
)


def _register_moe():
    from transformers_compat import ensure_minimax_transformers_compat

    ensure_minimax_transformers_compat()


def _get_model_cls():
    from transformers_compat import load_minimax_model_cls

    return load_minimax_model_cls(MinimaxM27Config.model_id)


MinimaxM27Config.register_moe = _register_moe
MinimaxM27Config.get_model_cls = _get_model_cls
