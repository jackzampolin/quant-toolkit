#!/usr/bin/env python3
"""Score a vLLM candidate against sealed dense BF16 prefill logits."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from kld_common import (
    dense_prompt_logits,
    sampling_params_for_prompt_logits,
    summarize_kld,
    tokenwise_kld,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-logits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="Use auto/BF16 for weight-only KLD; pass fp8 for deployment KLD.",
    )
    parser.add_argument("--load-format", default="safetensors")
    parser.add_argument("--quantization", default="auto")
    parser.add_argument("--attention-backend")
    parser.add_argument("--distributed-executor-backend", choices=("mp", "ray"))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--chunk-rows", type=int, default=8)
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--llm-extra-json", default="{}")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    reference_dir = Path(args.reference_logits)
    with open(reference_dir / "manifest.json") as handle:
        reference_manifest = json.load(handle)
    if reference_manifest.get("schema") != "quant-toolkit.prefill-logits.v1":
        raise ValueError("unsupported reference manifest schema")
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError(f"output directory is not empty: {destination}")

    llm_values = {
        "model": args.model,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "load_format": args.load_format,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "max_logprobs": -1,
    }
    for key in (
        "tokenizer",
        "revision",
        "tokenizer_revision",
        "attention_backend",
        "distributed_executor_backend",
    ):
        value = getattr(args, key)
        if value:
            llm_values[key] = value
    if args.quantization.lower() not in ("", "auto", "none", "null"):
        llm_values["quantization"] = args.quantization
    llm_values.update(json.loads(args.llm_extra_json))

    params, supports_prompt_logits = sampling_params_for_prompt_logits(SamplingParams)
    model = LLM(**llm_values)
    vocab_size = int(model.llm_engine.model_config.get_vocab_size())
    compute_dtype = torch.float64 if args.compute_dtype == "float64" else torch.float32
    all_kld = []
    all_ref_top1 = []
    all_candidate_top1 = []
    windows = []
    started = time.time()
    for record in reference_manifest["windows"]:
        tensors = load_file(reference_dir / record["file"])
        reference = tensors["logits"]
        input_ids = tensors["input_ids"].to(torch.int64).tolist()
        if reference.shape[1] != vocab_size:
            raise ValueError(
                f"vocabulary mismatch: reference={reference.shape[1]} candidate={vocab_size}"
            )
        prompt: TokensPrompt = {
            "prompt_token_ids": input_ids,
            "target_token_ids": input_ids[1:],
        }
        output = model.generate([prompt], sampling_params=params)[0]
        candidate, _ = dense_prompt_logits(
            output,
            supports_prompt_logits,
            reference.shape[0],
            vocab_size,
        )
        kld, ref_top1, candidate_top1 = tokenwise_kld(
            reference,
            candidate,
            chunk_rows=args.chunk_rows,
            compute_dtype=compute_dtype,
        )
        window_summary = summarize_kld(kld, ref_top1, candidate_top1)
        window_summary["index"] = record["index"]
        windows.append(window_summary)
        all_kld.append(kld)
        all_ref_top1.append(ref_top1)
        all_candidate_top1.append(candidate_top1)
        print(json.dumps({"event": "candidate_window_scored", **window_summary}), flush=True)
        del output, candidate, reference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_kld = torch.cat(all_kld)
    all_ref_top1 = torch.cat(all_ref_top1)
    all_candidate_top1 = torch.cat(all_candidate_top1)
    aggregate = summarize_kld(all_kld, all_ref_top1, all_candidate_top1)
    result = {
        "schema": "quant-toolkit.prefill-kld.v1",
        "candidate_model": args.model,
        "candidate_revision": args.revision,
        "reference": str(reference_dir.resolve()),
        "reference_manifest": reference_manifest,
        "candidate_engine_request": llm_values,
        "prompt_logits_mode": (
            "return_prompt_logits" if supports_prompt_logits else "flat_prompt_logprobs"
        ),
        "compute_dtype": args.compute_dtype,
        "aggregate": aggregate,
        "windows": windows,
        "elapsed_sec": time.time() - started,
    }
    save_file(
        {
            "kld_nats": all_kld,
            "kld_bits": all_kld / math.log(2.0),
            "reference_top1": all_ref_top1,
            "candidate_top1": all_candidate_top1,
        },
        str(destination / "tokenwise.safetensors"),
    )
    with open(destination / "summary.json", "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"event": "candidate_score_complete", **aggregate}), flush=True)


if __name__ == "__main__":
    main()
