#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python quantize.py \
    --model glm5_1 \
    --export-dir ./output/GLM-5.1-NVFP4 \
    --batch-tokens 85000 \
    --calib-config configs/calib_glm5_1.toml \
    --save-amax ./output/GLM-5.1-NVFP4/amax.safetensors \
    --streaming \
    --cpu-capacity 160GiB \
    --floor-amaxes
