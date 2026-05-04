#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_ID="${MODEL_ID:-/data/models/MiMo-V2.5-BF16-qkv-deinterleaved}"
EXPORT_DIR="${EXPORT_DIR:-/data/models/MiMo-V2.5-NVFP4}"
RESUME_AMAX="${RESUME_AMAX-$EXPORT_DIR/amax_checkpoint.safetensors}"
RESUME_BATCH="${RESUME_BATCH-246}"

mkdir -p "$EXPORT_DIR"

RESUME_ARGS=()
if [[ -n "$RESUME_AMAX" ]]; then
    if [[ ! -f "$RESUME_AMAX" ]]; then
        echo "Missing RESUME_AMAX checkpoint: $RESUME_AMAX" >&2
        exit 1
    fi
    RESUME_ARGS=(--resume-amax "$RESUME_AMAX" --resume-batch "$RESUME_BATCH")
fi

python quantize.py \
    --model mimo_v25 \
    --model-id "$MODEL_ID" \
    --export-dir "$EXPORT_DIR" \
    --calib-config configs/calib_mimo_v25.toml \
    --save-amax "$EXPORT_DIR/MiMo-V2.5-NVFP4/amax.safetensors" \
    --batch-tokens 100000 \
    --streaming \
    --cpu-capacity 200GiB \
    --floor-amaxes \
    "${RESUME_ARGS[@]}"
