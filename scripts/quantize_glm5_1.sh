#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python quantize.py \
    --model glm5_1 \
    --export-dir /data/models/GLM-5.1-NVFP4 \
    --batch-tokens 65536 \
    --calib-config configs/calib_glm5_1.toml \
    --save-amax ./output/GLM-5.1-NVFP4/amax_new.safetensors \
    --resume-amax ./output/GLM-5.1-NVFP4/amax_checkpoint_corrected.safetensors \
    --save-quantiles ./output/GLM-5.1-NVFP4/quantile_data_new.json \
    --streaming \
    --cpu-capacity 100GiB \
    --floor-amaxes
