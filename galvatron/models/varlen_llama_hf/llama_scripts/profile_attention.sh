#!/bin/bash
# Profile Flash Attention Computation + Piecewise Quadratic Fitting for AdaCPSP
# This is a single-GPU script (no distributed needed)

export CUDA_VISIBLE_DEVICES=0

# llama-7b config (MHA: n_kv_heads = n_heads)
python3 profile_attention_fit.py \
    --n_heads 32 \
    --n_kv_heads 32 \
    --head_dim 128 \
    --warmup 5 \
    --iters 20 \
    --model_name llama-7b \
    --save_dir ./configs \
    --skip_very_long

