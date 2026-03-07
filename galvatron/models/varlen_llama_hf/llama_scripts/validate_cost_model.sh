#!/bin/bash
# Full CostModel validation: compute + comm + memory
# 8 GPU distributed
cd "$(dirname "$0")/../"

echo "=============================================="
echo " Full CostModel Validation"
echo "=============================================="

# Use existing profile files if available, or run fresh
ATTN_JSON=""
A2A_JSON=""
P2P_JSON=""

# Auto-detect latest profile files
for f in configs/profile_validate_*.json; do
    if [ -f "$f" ]; then
        echo "Found existing profile: $f"
    fi
done

# Check for individual profile files
if [ -f "configs/attention_fit_llama-7b_20260303_231108.json" ]; then
    ATTN_JSON="--attn_json configs/attention_fit_llama-7b_20260303_231108.json"
fi
if [ -f "configs/alltoall_profile_8gpus_20260303_231306.json" ]; then
    A2A_JSON="--alltoall_json configs/alltoall_profile_8gpus_20260303_231306.json"
fi
if [ -f "configs/p2p_ring_profile_8gpus_20260303_231358.json" ]; then
    P2P_JSON="--p2p_json configs/p2p_ring_profile_8gpus_20260303_231358.json"
fi

torchrun --nproc_per_node=8 profile_and_validate.py \
    --mode validate_cost_model \
    --n_heads 32 \
    --n_kv_heads 32 \
    --head_dim 128 \
    --hidden_size 4096 \
    --num_layers 32 \
    --model_name llama-7b \
    --warmup 5 \
    --iters 30 \
    --attn_max 32768 \
    --save_dir ./configs \
    $ATTN_JSON $A2A_JSON $P2P_JSON

