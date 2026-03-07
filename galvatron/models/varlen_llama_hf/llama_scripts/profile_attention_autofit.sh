#!/bin/bash
# Attention profiling with automatic breakpoint detection
# Single GPU - no distributed required
cd "$(dirname "$0")/../"

export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo " Attention Profiling with Auto Breakpoint Detection"
echo "=============================================="

python3 profile_and_validate.py \
    --mode attention \
    --n_heads 32 \
    --n_kv_heads 32 \
    --head_dim 128 \
    --hidden_size 4096 \
    --num_layers 32 \
    --model_name llama-7b \
    --attn_step 64 \
    --attn_max 32768 \
    --warmup 5 \
    --iters 30 \
    --use_varlen \
    --bp_window 5 \
    --bp_threshold 3.0 \
    --bp_min_segment 8 \
    --save_dir ./configs

