from dataclasses import dataclass, field


# Shared quant overrides: disable attention weights/inputs, keep FP8 KV cache.
COMMON_QUANT_OVERRIDES = {
    "*self_attn*weight_quantizer": {"enable": False},
    "*self_attn*input_quantizer": {"enable": False},
    "*self_attn*q_bmm_quantizer": {"enable": False},
    "*self_attn*softmax_quantizer": {"enable": False},
    "*[kv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
}


@dataclass
class ModelQuantConfig:
    model_id: str
    trust_remote_code: bool = False
    streaming: bool = False
    extra_quant_overrides: dict = field(default_factory=dict)
    extra_mtp_prefixes: list = field(default_factory=list)
    preserve_remote_code: bool = False
    processor_trust_remote_code: bool | None = None

    def get_model_cls(self):
        """Return explicit model class, or None for AutoModelForCausalLM."""
        return None

    def register_moe(self):
        """Register MoE QuantModules if needed. Override in subclasses."""
        pass

    def get_processor_trust_remote_code(self) -> bool:
        if self.processor_trust_remote_code is None:
            return self.trust_remote_code
        return self.processor_trust_remote_code

    def get_all_quant_overrides(self) -> dict:
        overrides = dict(COMMON_QUANT_OVERRIDES)
        overrides.update(self.extra_quant_overrides)
        return overrides
