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


def load_mimo_model_cls(model_id: str):
    """Import the MiMo-V2 causal LM class from its remote-code repo."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    class_ref = config.auto_map["AutoModelForCausalLM"]
    return get_class_from_dynamic_module(class_ref, model_id)


def ensure_mimo_transformers_compat():
    """Patch Transformers FP8 loading for MiMo's Qwen2-style MoE checkpoint."""
    from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )
    from transformers.integrations import finegrained_fp8
    from transformers.quantizers.quantizers_utils import should_convert_module
    from transformers.utils.quantization_config import FineGrainedFP8Config

    mimo_modules_to_not_convert = [
        "visual",
        "audio_encoder",
        "audio_tokenizer",
        "speech_embeddings",
        "lm_head",
    ]

    if not getattr(FineGrainedFP8Config, "_mimo_ignored_layers_patch", False):
        original_init = FineGrainedFP8Config.__init__

        def __init__(
            self,
            activation_scheme: str = "dynamic",
            weight_block_size: tuple[int, int] = (128, 128),
            dequantize: bool = False,
            modules_to_not_convert: list | None = None,
            ignored_layers: list | None = None,
            **kwargs,
        ):
            if modules_to_not_convert is None and ignored_layers is not None:
                modules_to_not_convert = ignored_layers
            original_init(
                self,
                activation_scheme=activation_scheme,
                weight_block_size=weight_block_size,
                dequantize=dequantize,
                modules_to_not_convert=modules_to_not_convert,
                **kwargs,
            )

        FineGrainedFP8Config.__init__ = __init__
        FineGrainedFP8Config._mimo_ignored_layers_patch = True

    if not getattr(finegrained_fp8, "_mimo_replace_patch", False):

        def replace_with_fp8_linear(
            model, modules_to_not_convert: list[str] | None = None, quantization_config=None, pre_quantized=False
        ):
            if quantization_config.dequantize:
                return model

            import torch
            import torch.nn as nn

            if getattr(model.config, "model_type", None) == "mimo_v2":
                modules_to_not_convert = list(modules_to_not_convert or [])
                for module_name in mimo_modules_to_not_convert:
                    if module_name not in modules_to_not_convert:
                        modules_to_not_convert.append(module_name)

            has_been_replaced = False
            replaced_prefixes = []
            for module_name, module in list(model.named_modules()):
                if any(module_name.startswith(prefix + ".") for prefix in replaced_prefixes):
                    continue
                if not should_convert_module(module_name, modules_to_not_convert):
                    continue

                module_kwargs = {} if pre_quantized else {"dtype": None}
                new_module = None
                with torch.device("meta"):
                    if module_name.endswith(".experts"):
                        has_gate = getattr(module, "has_gate", True)
                        has_bias = getattr(module, "has_bias", False)
                        config = getattr(module, "config", model.config.get_text_config())
                        new_class = finegrained_fp8.use_experts_implementation(
                            experts_class=finegrained_fp8.FP8Experts,
                            experts_interface=finegrained_fp8.ALL_FP8_EXPERTS_FUNCTIONS,
                            has_bias=has_bias,
                            has_gate=has_gate,
                        )
                        new_module = new_class(
                            config=config,
                            block_size=quantization_config.weight_block_size,
                            activation_scheme=quantization_config.activation_scheme,
                            has_bias=has_bias,
                            has_gate=has_gate,
                            **module_kwargs,
                        )
                    elif isinstance(module, nn.Linear):
                        new_module = finegrained_fp8.FP8Linear(
                            in_features=module.in_features,
                            out_features=module.out_features,
                            block_size=quantization_config.weight_block_size,
                            activation_scheme=quantization_config.activation_scheme,
                            has_bias=module.bias is not None,
                            **module_kwargs,
                        )
                    if new_module is not None:
                        model.set_submodule(module_name, new_module)
                        has_been_replaced = True
                        if module_name.endswith(".experts"):
                            replaced_prefixes.append(module_name)

            if not has_been_replaced:
                finegrained_fp8.logger.warning(
                    "You are loading your model using fp8 but no linear modules were found in your model. "
                    "Please double check your model architecture."
                )
            return model

        finegrained_fp8.replace_with_fp8_linear = replace_with_fp8_linear
        finegrained_fp8._mimo_replace_patch = True

    if not getattr(finegrained_fp8, "_mimo_triton_only_patch", False):
        import torch

        def w8a8_fp8_matmul(
            A: torch.Tensor,
            B: torch.Tensor,
            As: torch.Tensor,
            Bs: torch.Tensor,
            block_size: list[int],
            output_dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            if block_size is not None:
                block_n, block_k = block_size
                expected_scale_shape = (
                    (B.shape[-2] + block_n - 1) // block_n,
                    (B.shape[-1] + block_k - 1) // block_k,
                )
                if Bs.shape[-2:] != expected_scale_shape:
                    if Bs.shape[-2] < expected_scale_shape[0] or Bs.shape[-1] < expected_scale_shape[1]:
                        raise ValueError(
                            "FP8 weight scale grid is smaller than the weight requires: "
                            f"weight={tuple(B.shape)}, scale={tuple(Bs.shape)}, expected={expected_scale_shape}"
                        )
                    Bs = Bs[..., : expected_scale_shape[0], : expected_scale_shape[1]].contiguous()
            finegrained_fp8._load_triton_kernel()
            return finegrained_fp8.triton_fp8_matmul(A, B, As, Bs, block_size, output_dtype)

        finegrained_fp8.w8a8_fp8_matmul = w8a8_fp8_matmul
        finegrained_fp8._mimo_triton_only_patch = True

    if get_checkpoint_conversion_mapping("mimo_v2") is None:
        register_checkpoint_conversion_mapping(
            "mimo_v2",
            [
                WeightConverter(
                    source_patterns=[
                        "mlp.experts.*.gate_proj.weight",
                        "mlp.experts.*.up_proj.weight",
                    ],
                    target_patterns="mlp.experts.gate_up_proj",
                    operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
                ),
                WeightConverter(
                    source_patterns=[
                        "mlp.experts.*.gate_proj.weight_scale_inv",
                        "mlp.experts.*.up_proj.weight_scale_inv",
                    ],
                    target_patterns="mlp.experts.gate_up_proj_scale_inv",
                    operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
                ),
                WeightConverter(
                    source_patterns="mlp.experts.*.down_proj.weight",
                    target_patterns="mlp.experts.down_proj",
                    operations=[MergeModulelist(dim=0)],
                ),
                WeightConverter(
                    source_patterns="mlp.experts.*.down_proj.weight_scale_inv",
                    target_patterns="mlp.experts.down_proj_scale_inv",
                    operations=[MergeModulelist(dim=0)],
                ),
            ],
        )
