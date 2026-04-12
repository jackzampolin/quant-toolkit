"""Compatibility shims for model repos that import moved HF internals."""


def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
    import torch

    config.standardize_rope_params()
    rope_parameters_dict = (
        config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
    )
    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim
        )
    )
    return inv_freq, 1.0


def ensure_minimax_transformers_compat():
    """Patch Transformers internals expected by MiniMax remote code."""
    from transformers.utils import generic as generic_utils
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    patched = False

    if not hasattr(generic_utils, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        generic_utils.OutputRecorder = OutputRecorder
        patched = True

    if not hasattr(generic_utils, "check_model_inputs"):
        def check_model_inputs(fn=None, *args, **kwargs):
            if fn is None:
                def decorator(inner_fn):
                    return inner_fn

                return decorator
            return fn

        generic_utils.check_model_inputs = check_model_inputs
        patched = True

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
        patched = True

    if patched:
        print("✓ Patched MiniMax transformers compatibility")


def load_minimax_model_cls(model_id: str):
    """Import and patch the MiniMax causal LM class from a local/remote repo."""
    import sys

    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    ensure_minimax_transformers_compat()

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    class_ref = config.auto_map["AutoModelForCausalLM"]
    model_cls = get_class_from_dynamic_module(class_ref, model_id)

    module = sys.modules[model_cls.__module__]
    rotary_cls = getattr(module, "MiniMaxM2RotaryEmbedding", None)
    if rotary_cls is not None and not hasattr(rotary_cls, "compute_default_rope_parameters"):
        rotary_cls.compute_default_rope_parameters = staticmethod(_compute_default_rope_parameters)

    return model_cls
