#!/bin/bash
# Full profiling + validation pipeline (all-in-one)
# 8 GPU distributed
cd "$(dirname "$0")/../"

echo "=============================================="
echo " AdaCPSP Full Profiling & Validation Pipeline"
echo " Mode: ALL (attention + comm + cost model + memory)"
echo "=============================================="

torchrun --nproc_per_node=8 profile_and_validate.py \
    --mode all \
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
    --save_dir ./configs

