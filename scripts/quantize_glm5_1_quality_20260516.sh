#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

BF16_MODEL="${BF16_MODEL:-/root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d}"
EXPORT_DIR="${EXPORT_DIR:-/root/kld/checkpoints/GLM-5.1-NVFP4-quality-20260516}"
CALIB_CONFIG="${CALIB_CONFIG:-configs/calib_glm5_1_quality_max_20260516.toml}"
CPU_CAPACITY="${CPU_CAPACITY:-750GiB}"

python3 -u quantize.py \
    --model glm5_1 \
    --model-id "${BF16_MODEL}" \
    --export-dir "${EXPORT_DIR}" \
    --batch-tokens 65536 \
    --calib-config "${CALIB_CONFIG}" \
    --save-amax "${EXPORT_DIR}/amax_quality_20260516.safetensors" \
    --streaming \
    --cpu-capacity "${CPU_CAPACITY}" \
    --floor-amaxes
