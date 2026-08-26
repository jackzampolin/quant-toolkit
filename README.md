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

Capture the BF16 teacher once. The artifact stores exact input token IDs with
every dense-logit window, avoiding tokenizer or corpus drift during candidate
comparisons.

```bash
python tools/collect_prefill_logits.py \
  --model /path/to/GLM-5.3-Flash-BF16 \
  --revision b1967181a3917ae70a437f4884748f6b8e3a1f4d \
  --output-dir /data/kld/glm53-bf16 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend ray \
  --kv-cache-dtype auto \
  --context-length 2048 --stride 512 --max-windows 25
```

Score the official FP8 checkpoint first to establish the acceptance baseline,
then score each NVFP4 candidate against the same reference artifact.

```bash
python tools/score_prefill_kld.py \
  --model /path/to/GLM-5.3-Flash \
  --reference-logits /data/kld/glm53-bf16 \
  --output-dir /data/kld/glm53-official-fp8 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend ray \
  --kv-cache-dtype auto
```

The scorer emits tokenwise values and aggregate/tail summaries in both nats and
bits. Its direction is always `KL(BF16 reference || candidate)`. The vLLM build
must expose `return_prompt_logits` or the lab's flat full-logprob fallback.
Keep KV dtype, attention backend, tensor parallelism, corpus, and tokenizer
fixed between the reference and candidate runs. Use `--kv-cache-dtype fp8` on
both sides only when measuring the production FP8-KV stack rather than isolated
weight error.
