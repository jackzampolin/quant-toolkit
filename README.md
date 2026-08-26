Ask your friendly neighborhood AI agent how to use this.

## GLM-5.3-Flash routed-expert NVFP4

The first GLM-5.3 profile is deliberately conservative: only routed expert
gate/up/down linears are NVFP4. KDA linear attention, NoPE sparse attention and
its indexer, vision, hyper-connections, routers, shared experts, dense MLPs,
embeddings, the LM head, and MTP remain in source precision.

```bash
MODEL_ID=/path/to/GLM-5.3-Flash-BF16 \
EXPORT_DIR=/data/models/GLM-5.3-Flash-NVFP4-routed \
scripts/quantize_glm5_3_flash.sh
```

This covers about 304.4B of 321.3B parameters and is expected to produce a
checkpoint near 195-205 GB after NVFP4 block scales and preserved tensors. Do
not widen coverage based only on local reconstruction error; use BF16-teacher
KLD to qualify each candidate.

## Dense-prefill KLD

The campaign workflow is capture once, compare many. Every weight/topology/KV
combination gets one sealed full-vocabulary logit capture. Comparisons are then
computed offline, so canonical, total-deployment, and within-weight KV-only KLD
do not require another model load. Publishable captures use float32 storage;
do not downcast them when qualifying a checkpoint.

For the two-workstation deployment, keep CPU-heavy work off the GPU hosts. Use
the operator Mac and `uv` for corpus preparation, hashing, offline comparison,
independent replay, and reporting. Workstation-1 and workstation-2 are reserved
for distributed model inference and logit capture; copy sealed captures away
before running the offline tools. Do not overlap a workstation CPU analysis job
with a GPU model run.

Capture the canonical BF16-weights/BF16-KV run:

```bash
python tools/collect_prefill_logits.py \
  --model /path/to/GLM-5.3-Flash-BF16 \
  --revision b1967181a3917ae70a437f4884748f6b8e3a1f4d \
  --output-dir /data/kld/captures/bf16-tp8-bf16-kv \
  --role canonical --run-label bf16-tp8-bf16-kv \
  --tensor-parallel-size 8 \
  --distributed-executor-backend ray \
  --kv-cache-dtype bfloat16 \
  --storage-dtype float32 \
  --context-length 2048 --stride 512 --max-windows 25
```

Capture every other run with `--role candidate`, the same pinned corpus,
context length, stride, and window count. For example:

```bash
python tools/collect_prefill_logits.py \
  --model /path/to/GLM-5.3-Flash \
  --output-dir /data/kld/captures/fp8-tp4-fp8-ds-mla-kv \
  --role candidate --run-label fp8-tp4-fp8-ds-mla-kv \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --storage-dtype float32 \
  --context-length 2048 --stride 512 --max-windows 25
```

Compare any two captures without loading either model:

```bash
uv run --no-project --with torch --with 'numpy>=2.0' --with safetensors \
  python tools/compare_captured_prefill_kld.py \
  --reference-logits /data/kld/captures/bf16-tp8-bf16-kv \
  --candidate-logits /data/kld/captures/fp8-tp4-fp8-ds-mla-kv \
  --output-dir /data/kld/reports/bf16-bf16__fp8-fp8-ds-mla
```

Independently replay that report. The verifier uses NumPy and an explicit
log-sum-exp implementation rather than the producer's PyTorch `kl_div` path,
checks both manifests, every artifact hash, and exact input-token equality, and
rejects a tokenwise discrepancy above `1e-12` by default.

```bash
uv run --no-project --with 'numpy>=2.0' --with safetensors \
  python tools/replay_captured_prefill_kld.py \
  --reference-logits /data/kld/captures/bf16-tp8-bf16-kv \
  --candidate-logits /data/kld/captures/fp8-tp4-fp8-ds-mla-kv \
  --report /data/kld/reports/bf16-bf16__fp8-fp8-ds-mla \
  --output-dir /data/kld/verifications/bf16-bf16__fp8-fp8-ds-mla
```

Reports contain tokenwise KLD plus mean, median, p95, p99, maximum, and top-1
agreement. The direction is always `KL(reference || candidate)`, in both nats
and bits. The vLLM build must expose `return_prompt_logits` or the lab's flat
full-logprob fallback. `score_prefill_kld.py` and `replay_prefill_kld.py` remain
available for one-off live comparisons, but they are not the preferred campaign
path.

### GLM-5.3 NoPE KV policy gate

Do not import DeepSeek's mixed-RoPE cache names into GLM-5.3-Flash. The pinned
model has `qk_rope_head_dim=0`, `qk_nope_head_dim=256`, and
`kv_lora_rank=512`; there is no model RoPE tensor to preserve or quantize.

The pinned cstech SM120 image currently resolves the intended policies as
follows:

| Intended tier | Requested dtype | Effective GLM layout | Current status |
|---|---|---|---|
| BF16 throughout | `bfloat16` | 512-wide BF16 latent cache | No compatible sparse-MLA SM120 backend in the pinned image |
| BF16 RoPE + FP8 NoPE | `fp8` | `fp8_ds_mla`: 512 FP8 NoPE values, four FP32 scales, and a padded/reserved 64-wide BF16 RoPE region | Supported; this is the current production layout |
| FP8 RoPE + NVFP4 NoPE | no valid flag yet | Must be a genuine NVFP4 latent-cache kernel; GLM has no actual RoPE part | Unsupported by the pinned sparse-MLA SM120 backends |

`nvfp4_4over6` is not the third mixed layout. It selects max/6 versus max/4
scaling per 16 NVFP4 values by reconstruction error. Do not record an NVFP4 KV
result until backend selection and the startup log prove a distinct supported
layout.

For the TP4 campaign, use this order:

1. Once supported, capture BF16 weights on TP8 under BF16, production FP8, and
   genuine NVFP4 KV policies.
2. Unload BF16, then capture official FP8 weights on TP4 under the same three
   policies.
3. Compare all five non-canonical captures with BF16-weights/BF16-KV. Also
   compare BF16 policy pairs and official-FP8 policy pairs for within-weight
   KV-only deltas.
4. Use official FP8 TP4 KLD, cache capacity, prefill, and decode measurements
   to eliminate dominated KV policies.
5. Quantize routed experts, then capture NVFP4 weights on TP4 under the same
   policies. Compare against the canonical capture and within the NVFP4 weight
   family before running expensive concurrency and Estonia gates.

Keep corpus, tokenizer, context windows, attention backend, and arithmetic
settings fixed. BF16 requires TP8 while the deployment target is TP4, so its
canonical comparison intentionally includes the small topology/runtime numeric
delta. Official FP8 and NVFP4 candidates remain TP4 throughout; a matched-TP8
FP8 control is optional and should only be run if the topology delta needs to be
separated.
