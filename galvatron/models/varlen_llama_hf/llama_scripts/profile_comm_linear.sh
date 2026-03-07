#!/bin/bash
# Communication profiling with linear fitting y = a*x + b
# 8 GPU distributed
cd "$(dirname "$0")/../"

echo "=============================================="
echo " Communication Profiling with Linear Fitting"
echo "=============================================="

torchrun --nproc_per_node=8 profile_and_validate.py \
    --mode comm \
    --n_heads 32 \
    --n_kv_heads 32 \
    --head_dim 128 \
    --hidden_size 4096 \
    --num_layers 32 \
    --model_name llama-7b \
    --warmup 5 \
    --iters 50 \
    --save_dir ./configs

