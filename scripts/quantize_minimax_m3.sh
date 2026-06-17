#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING:-1}"
export TORCHDYNAMO_RECOMPILE_LIMIT="${TORCHDYNAMO_RECOMPILE_LIMIT:-100000}"
export TORCHDYNAMO_ACCUMULATED_RECOMPILE_LIMIT="${TORCHDYNAMO_ACCUMULATED_RECOMPILE_LIMIT:-100000}"
# Default to all 10 visible GPUs. GPU 0 executes streamed layers; CUDA layer
# moves from GPU 9 trampoline through GPU 8 because GPU 9 cannot peer directly
# with GPU 0 on this box.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9}"

MODEL_ID="${MODEL_ID:-MiniMaxAI/MiniMax-M3}"
EXPORT_DIR="${EXPORT_DIR:-./output/MiniMax-M3-NVFP4}"
RESUME_AMAX="${RESUME_AMAX-$EXPORT_DIR/amax_checkpoint.safetensors}"
RESUME_BATCH="${RESUME_BATCH-0}"
STREAMING="${STREAMING:-1}"
STREAMING_GPU0_STORAGE_CAPACITY="${STREAMING_GPU0_STORAGE_CAPACITY:-0GiB}"
STREAMING_GPU_CAPACITY="${STREAMING_GPU_CAPACITY:-}"
CPU_CAPACITY="${CPU_CAPACITY:-128GiB}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-minimax_m3_flex}"
CUDA_STAGING_DEVICE="${CUDA_STAGING_DEVICE:-cuda:8}"
CUDA_STAGING_SOURCE_DEVICE="${CUDA_STAGING_SOURCE_DEVICE:-cuda:9}"
CUDA_STAGING_RESERVE="${CUDA_STAGING_RESERVE:-12GiB}"

mkdir -p "$EXPORT_DIR"

RESUME_ARGS=()
if [[ -n "$RESUME_AMAX" && -f "$RESUME_AMAX" ]]; then
    RESUME_ARGS=(--resume-amax "$RESUME_AMAX" --resume-batch "$RESUME_BATCH")
elif [[ -n "$RESUME_AMAX" && "$RESUME_BATCH" != "0" ]]; then
    echo "Missing RESUME_AMAX checkpoint: $RESUME_AMAX" >&2
    exit 1
fi

STREAMING_ARGS=()
case "$STREAMING" in
    0|false|False|no|No)
        STREAMING_ARGS=(--no-streaming)
        ;;
    *)
        STREAMING_ARGS=(
            --streaming
            --cpu-capacity "$CPU_CAPACITY"
            --streaming-gpu0-storage-capacity "$STREAMING_GPU0_STORAGE_CAPACITY"
            --cuda-staging-device "$CUDA_STAGING_DEVICE"
            --cuda-staging-source-device "$CUDA_STAGING_SOURCE_DEVICE"
            --cuda-staging-reserve "$CUDA_STAGING_RESERVE"
        )
        if [[ -n "$STREAMING_GPU_CAPACITY" ]]; then
            STREAMING_ARGS+=(--streaming-gpu-capacity "$STREAMING_GPU_CAPACITY")
        fi
        ;;
esac

python quantize.py \
  --model minimax_m3 \
  --model-id "$MODEL_ID" \
  --export-dir "$EXPORT_DIR" \
  --calib-config configs/calib_minimax_m3.toml \
  --save-amax "$EXPORT_DIR/amax.safetensors" \
  --batch-tokens 100000 \
  --attn-implementation "$ATTN_IMPLEMENTATION" \
  "${STREAMING_ARGS[@]}" \
  --floor-amaxes \
  "${RESUME_ARGS[@]}"
