#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python quantize.py \
    --model minimax_m27 \
    --export-dir ./output/MiniMax-M2.7-NVFP4 \
    --calib-config configs/calib_minimax_m27.toml \
    --batch-tokens 225000 \
    --save-amax ./output/MiniMax-M2.7-NVFP4/amax_new.safetensors \
    --resume-amax ./output/MiniMax-M2.7-NVFP4/amax_checkpoint.safetensors \
    --resume-batch 200 \
    --save-quantiles ./output/MiniMax-M2.7-NVFP4/quantile_data_new.json \
    --streaming \
    --floor-amaxes
