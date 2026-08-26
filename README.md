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

Capture the BF16 teacher with BF16 KV first. The artifact stores exact input
token IDs with every dense-logit window, avoiding tokenizer or corpus drift
during candidate comparisons. Publishable captures store float32 logits by
default; do not downcast them when qualifying a checkpoint.

```bash
python tools/collect_prefill_logits.py \
  --model /path/to/GLM-5.3-Flash-BF16 \
  --revision b1967181a3917ae70a437f4884748f6b8e3a1f4d \
  --output-dir /data/kld/glm53-bf16 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend ray \
  --kv-cache-dtype bfloat16 \
  --storage-dtype float32 \
  --context-length 2048 --stride 512 --max-windows 25
```

Score the official FP8 checkpoint first to establish the acceptance baseline,
then score each NVFP4 candidate against the same reference artifact.

```bash
python tools/score_prefill_kld.py \
  --model /path/to/GLM-5.3-Flash \
  --reference-logits /data/kld/glm53-bf16 \
  --output-dir /data/kld/glm53-official-fp8 \
  --tensor-parallel-size 4 \
  --kv-cache-dtype bfloat16 \
  --storage-dtype float32
```

Independently replay the saved teacher and candidate logits. This verifier uses
NumPy and an explicit log-sum-exp implementation rather than the producer's
PyTorch `kl_div` path, checks every artifact hash and exact input-token match,
and rejects a tokenwise discrepancy above `1e-12` by default.

```bash
python tools/replay_prefill_kld.py \
  --reference-logits /data/kld/glm53-bf16 \
  --candidate-logits /data/kld/glm53-official-fp8 \
  --output-dir /data/kld/glm53-official-fp8-replay
```

The scorer emits tokenwise values and aggregate/tail summaries in both nats and
bits. Its direction is always `KL(BF16 reference || candidate)`. The vLLM build
must expose `return_prompt_logits` or the lab's flat full-logprob fallback.

For the TP4 campaign, use this order:

1. Capture BF16 weights on the required TP8 topology with BF16 KV. This is the
   canonical quality reference.
2. Capture BF16 weights once more with production FP8 KV. This isolates KV
   error without reloading the teacher for every candidate format.
3. Score official FP8 weights on TP4 with BF16 KV, establishing the target
   topology/runtime and weight-quantization baseline.
4. Sweep KV formats on the official FP8 TP4 checkpoint. Compare each format to
   the official-FP8/BF16-KV anchor for its KV-only delta, and to the canonical
   BF16 capture for total deployment KLD.
5. Run the eventual NVFP4 TP4 checkpoint only with KV modes that survive the
   FP8 KLD, capacity, and speed screen.

Keep corpus, tokenizer, context windows, attention backend, and arithmetic
settings fixed. BF16 requires TP8 while the deployment target is TP4, so its
canonical comparison intentionally includes the small topology/runtime numeric
delta. Official FP8 and NVFP4 candidates remain TP4 throughout; a matched-TP8
FP8 control is optional and should only be run if the topology delta needs to be
separated.
