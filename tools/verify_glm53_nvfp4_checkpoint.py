#!/usr/bin/env python3
"""Verify and seal GLM-5.3 routed-expert NVFP4 checkpoint coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

SCHEMA = "quant-toolkit.glm53-nvfp4-coverage.v1"
BLOCK_SIZE = 16
ROUTED_WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def tensor_nbytes(info: dict) -> int:
    try:
        element_bytes = DTYPE_BYTES[info["dtype"]]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype: {info['dtype']}") from exc
    return numel(tuple(info["shape"])) * element_bytes


def read_checkpoint(
    directory: Path,
    *,
    scalar_keys: set[str] | None = None,
) -> tuple[dict, dict[str, dict], dict[str, float], list[dict]]:
    index_path = directory / "model.safetensors.index.json"
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"checkpoint index has no weight_map: {index_path}")

    keys_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not isinstance(shard, str):
            raise TypeError("checkpoint weight_map keys and values must be strings")
        keys_by_shard[shard].add(key)

    info: dict[str, dict] = {}
    scalars: dict[str, float] = {}
    files: list[dict] = []
    scalar_keys = scalar_keys or set()
    for shard, indexed_keys in sorted(keys_by_shard.items()):
        path = directory / shard
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint shard does not exist: {path}")
        files.append({"path": shard, "bytes": path.stat().st_size})
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            if actual_keys != indexed_keys:
                missing = sorted(indexed_keys - actual_keys)
                extra = sorted(actual_keys - indexed_keys)
                raise ValueError(
                    f"index/shard key mismatch for {path}: "
                    f"missing={missing[:8]} extra={extra[:8]}"
                )
            for key in sorted(actual_keys):
                tensor_slice = handle.get_slice(key)
                info[key] = {
                    "dtype": tensor_slice.get_dtype(),
                    "shape": list(tensor_slice.get_shape()),
                    "shard": shard,
                }
                if key in scalar_keys:
                    tensor = handle.get_tensor(key).reshape(-1)
                    if tensor.numel() != 1:
                        raise ValueError(f"expected scalar tensor: {key}")
                    scalars[key] = float(tensor.float().item())

    if set(info) != set(weight_map):
        raise AssertionError("checkpoint header scan did not cover the complete index")
    return index, info, scalars, files


def source_routed_keys(source_info: dict[str, dict], text_config: dict) -> list[str]:
    first_sparse_layer = int(text_config["first_k_dense_replace"])
    num_hidden_layers = int(text_config["num_hidden_layers"])
    num_experts = int(text_config["n_routed_experts"])
    selected: list[str] = []
    observed: dict[tuple[int, str], set[int]] = defaultdict(set)
    for key in source_info:
        match = ROUTED_WEIGHT_RE.fullmatch(key)
        if match is None:
            continue
        layer = int(match.group("layer"))
        if first_sparse_layer <= layer < num_hidden_layers:
            selected.append(key)
            observed[(layer, match.group("proj"))].add(int(match.group("expert")))

    expected_experts = set(range(num_experts))
    for layer in range(first_sparse_layer, num_hidden_layers):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            actual = observed.get((layer, projection), set())
            if actual != expected_experts:
                raise ValueError(
                    "source routed-expert topology mismatch: "
                    f"layer={layer} projection={projection} "
                    f"expected={num_experts} actual={len(actual)}"
                )
    expected_count = (num_hidden_layers - first_sparse_layer) * num_experts * 3
    if len(selected) != expected_count:
        raise AssertionError("selected routed tensor count does not match topology")
    return sorted(selected)


def sidecar_keys(weight_key: str) -> tuple[str, str, str]:
    prefix = weight_key.removesuffix(".weight")
    return (
        prefix + ".weight_scale",
        prefix + ".weight_scale_2",
        prefix + ".input_scale",
    )


def validate_quantized_tensor(
    key: str,
    source: dict,
    candidate_info: dict[str, dict],
) -> None:
    source_shape = tuple(source["shape"])
    if len(source_shape) != 2 or source_shape[-1] % BLOCK_SIZE:
        raise ValueError(
            f"unsupported source routed weight shape for {key}: {source_shape}"
        )
    packed = candidate_info[key]
    expected_packed_shape = (*source_shape[:-1], source_shape[-1] // 2)
    if packed["dtype"] != "U8" or tuple(packed["shape"]) != expected_packed_shape:
        raise ValueError(
            f"invalid NVFP4 packed weight {key}: "
            f"dtype={packed['dtype']} shape={packed['shape']} "
            f"expected=U8/{expected_packed_shape}"
        )

    weight_scale_key, weight_scale_2_key, input_scale_key = sidecar_keys(key)
    expected_scale_shape = (*source_shape[:-1], source_shape[-1] // BLOCK_SIZE)
    weight_scale = candidate_info[weight_scale_key]
    if (
        weight_scale["dtype"] != "F8_E4M3"
        or tuple(weight_scale["shape"]) != expected_scale_shape
    ):
        raise ValueError(
            f"invalid NVFP4 block scale {weight_scale_key}: "
            f"dtype={weight_scale['dtype']} shape={weight_scale['shape']} "
            f"expected=F8_E4M3/{expected_scale_shape}"
        )
    for scalar_key in (weight_scale_2_key, input_scale_key):
        scalar = candidate_info[scalar_key]
        if scalar["dtype"] != "F32" or numel(tuple(scalar["shape"])) != 1:
            raise ValueError(
                f"invalid NVFP4 scalar {scalar_key}: "
                f"dtype={scalar['dtype']} shape={scalar['shape']}"
            )


def write_report(path: Path, report: dict) -> None:
    report = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = canonical_json_sha256(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--hash-candidate-shards",
        action="store_true",
        help="Include full SHA-256 hashes for every candidate tensor shard.",
    )
    args = parser.parse_args(argv)

    source_dir = Path(args.source_model).resolve()
    candidate_dir = Path(args.candidate_model).resolve()
    source_config_path = source_dir / "config.json"
    candidate_config_path = candidate_dir / "config.json"
    source_config = load_json(source_config_path)
    candidate_config = load_json(candidate_config_path)
    text_config = source_config.get("text_config")
    if not isinstance(text_config, dict):
        raise TypeError("source GLM-5.3 config has no text_config object")
    architectures = source_config.get("architectures", [])
    if "Glm5NextForConditionalGeneration" not in architectures:
        raise ValueError("source checkpoint is not GLM-5.3-Flash")

    _source_index, source_info, _unused, source_files = read_checkpoint(source_dir)
    selected = source_routed_keys(source_info, text_config)
    selected_set = set(selected)
    scalar_keys = {scalar for key in selected for scalar in sidecar_keys(key)[1:]}
    _candidate_index, candidate_info, scalars, candidate_files = read_checkpoint(
        candidate_dir,
        scalar_keys=scalar_keys,
    )

    expected_candidate_keys = set(source_info)
    for key in selected:
        expected_candidate_keys.update(sidecar_keys(key))
    actual_candidate_keys = set(candidate_info)
    if actual_candidate_keys != expected_candidate_keys:
        missing = sorted(expected_candidate_keys - actual_candidate_keys)
        extra = sorted(actual_candidate_keys - expected_candidate_keys)
        raise ValueError(
            "candidate tensor keyset is not routed-expert-only NVFP4: "
            f"missing={len(missing)} extra={len(extra)} "
            f"first_missing={missing[:8]} first_extra={extra[:8]}"
        )

    for key, source_tensor in source_info.items():
        if key in selected_set:
            validate_quantized_tensor(key, source_tensor, candidate_info)
        elif (
            candidate_info[key]["dtype"] != source_tensor["dtype"]
            or candidate_info[key]["shape"] != source_tensor["shape"]
        ):
            raise ValueError(f"preserved tensor dtype/shape changed: {key}")

    for key in selected:
        _weight_scale, weight_scale_2, input_scale = sidecar_keys(key)
        for scalar_key in (weight_scale_2, input_scale):
            value = scalars[scalar_key]
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"non-positive or non-finite quant scalar: {scalar_key}"
                )

    first_sparse_layer = int(text_config["first_k_dense_replace"])
    num_hidden_layers = int(text_config["num_hidden_layers"])
    num_experts = int(text_config["n_routed_experts"])
    for layer in range(first_sparse_layer, num_hidden_layers):
        for expert in range(num_experts):
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
            gate = scalars[prefix + ".gate_proj.weight_scale_2"]
            up = scalars[prefix + ".up_proj.weight_scale_2"]
            if gate != up:
                raise ValueError(
                    "gate/up weight_scale_2 is not tied: "
                    f"layer={layer} expert={expert} gate={gate} up={up}"
                )

    quant_config = candidate_config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise TypeError("candidate config has no quantization_config object")
    if "NVFP4" not in json.dumps(quant_config, sort_keys=True).upper():
        raise ValueError("candidate quantization_config does not declare NVFP4")

    source_tensor_bytes = sum(tensor_nbytes(value) for value in source_info.values())
    selected_params = sum(numel(tuple(source_info[key]["shape"])) for key in selected)
    candidate_tensor_bytes = sum(
        tensor_nbytes(value) for value in candidate_info.values()
    )
    preserved_tensor_bytes = sum(
        tensor_nbytes(source_info[key])
        for key in source_info
        if key not in selected_set
    )
    expected_candidate_tensor_bytes = (
        preserved_tensor_bytes
        + selected_params // 2
        + selected_params // BLOCK_SIZE
        + len(selected) * 8
    )
    if candidate_tensor_bytes != expected_candidate_tensor_bytes:
        raise ValueError(
            "candidate tensor-byte total disagrees with exact NVFP4 layout: "
            f"actual={candidate_tensor_bytes} expected={expected_candidate_tensor_bytes}"
        )

    if args.hash_candidate_shards:
        for item in candidate_files:
            item["sha256"] = sha256_file(candidate_dir / item["path"])

    selected_key_sha256 = canonical_json_sha256(selected)
    report = {
        "schema": SCHEMA,
        "source": {
            "directory": str(source_dir),
            "revision": args.source_revision,
            "config_sha256": sha256_file(source_config_path),
            "index_sha256": sha256_file(source_dir / "model.safetensors.index.json"),
            "tensor_count": len(source_info),
            "tensor_bytes": source_tensor_bytes,
            "shards": source_files,
        },
        "candidate": {
            "name": args.candidate_name,
            "directory": str(candidate_dir),
            "config_sha256": sha256_file(candidate_config_path),
            "index_sha256": sha256_file(candidate_dir / "model.safetensors.index.json"),
            "tensor_count": len(candidate_info),
            "tensor_bytes": candidate_tensor_bytes,
            "shards_hashed": args.hash_candidate_shards,
            "shards": candidate_files,
        },
        "coverage": {
            "first_sparse_layer": first_sparse_layer,
            "last_main_layer": num_hidden_layers - 1,
            "protected_mtp_layer": num_hidden_layers,
            "experts_per_layer": num_experts,
            "quantized_weight_tensors": len(selected),
            "quantized_parameters": selected_params,
            "source_parameters": sum(
                numel(tuple(value["shape"])) for value in source_info.values()
            ),
            "quantized_parameter_fraction": selected_params
            / sum(numel(tuple(value["shape"])) for value in source_info.values()),
            "selected_keyset_sha256": selected_key_sha256,
            "block_size": BLOCK_SIZE,
            "packed_weight_dtype": "U8",
            "block_scale_dtype": "F8_E4M3",
            "global_and_input_scale_dtype": "F32",
            "gate_up_weight_scale_2_tied": True,
            "unexpected_quantized_tensors": 0,
        },
    }
    write_report(Path(args.output).resolve(), report)
    print(
        json.dumps({"event": "glm53_nvfp4_checkpoint_verified", **report["coverage"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
