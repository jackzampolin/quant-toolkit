"""Shared MiMo fused-QKV quantization format metadata."""

from __future__ import annotations

from typing import Any


FP8_PB_WEIGHT_BLOCK_SIZE = [128, 128]
MXFP8_WEIGHT_BLOCK_SIZE = [1, 32]
MXFP8_GROUP_SIZE = MXFP8_WEIGHT_BLOCK_SIZE[1]


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def infer_qkv_quant_format(
    weight_shape: list[int],
    scale_shape: list[int],
    scale_dtype: str,
) -> str:
    if scale_dtype == "F32":
        expected = [
            ceil_div(weight_shape[0], FP8_PB_WEIGHT_BLOCK_SIZE[0]),
            ceil_div(weight_shape[1], FP8_PB_WEIGHT_BLOCK_SIZE[1]),
        ]
        if scale_shape != expected:
            raise RuntimeError(
                f"Expected FP8_PB_WO scale shape {expected}, got {scale_shape}"
            )
        return "fp8-pb"
    if scale_dtype == "U8":
        expected = [
            weight_shape[0],
            ceil_div(weight_shape[1], MXFP8_WEIGHT_BLOCK_SIZE[1]),
        ]
        if scale_shape != expected:
            raise RuntimeError(f"Expected MXFP8 scale shape {expected}, got {scale_shape}")
        return "mxfp8"
    raise RuntimeError(f"Unsupported QKV scale dtype {scale_dtype}; expected F32 or U8")


def qkv_quantized_layer_entry(format_name: str) -> dict[str, Any]:
    if format_name == "fp8-pb":
        return {
            "quant_algo": "FP8_PB_WO",
            "weight_block_size": FP8_PB_WEIGHT_BLOCK_SIZE,
        }
    if format_name == "mxfp8":
        return {
            "quant_algo": "MXFP8",
            "group_size": MXFP8_GROUP_SIZE,
        }
    raise ValueError(f"Unknown QKV quantization format: {format_name}")


def qkv_config_group(format_name: str, prefixes: list[str]) -> dict[str, Any]:
    if format_name == "fp8-pb":
        return {
            "input_activations": {
                "dynamic": True,
                "num_bits": 8,
                "type": "float",
            },
            "weights": {
                "dynamic": False,
                "num_bits": 8,
                "type": "float",
                "weight_block_size": FP8_PB_WEIGHT_BLOCK_SIZE,
            },
            "targets": prefixes,
        }
    if format_name == "mxfp8":
        return {
            "input_activations": {
                "dynamic": True,
                "num_bits": 8,
                "type": "float",
                "group_size": MXFP8_GROUP_SIZE,
            },
            "weights": {
                "dynamic": False,
                "num_bits": 8,
                "type": "float",
                "group_size": MXFP8_GROUP_SIZE,
                "weight_block_size": MXFP8_WEIGHT_BLOCK_SIZE,
            },
            "targets": prefixes,
        }
    raise ValueError(f"Unknown QKV quantization format: {format_name}")
