#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
# Test: Combined Ulysses(sp=2) + Ring Attention(cp=4) with varlen packing
# 8 GPUs total: tp=2 (Ulysses sp_size=2), cp_size=4, dp=1
export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29503
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_HCA=mlx5_2,mlx5_5

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist_adacpsp.py"

# llama-7b: hidden=4096, heads=32, kv_heads=32 (MHA)
# Use 2 layers for fast testing, seq_length=4096
# Combined: tp_deg=2 (with use-ulysses → sp_size=2), cp_deg=4 (Ring Attention)
# heads must be divisible by sp_size: 32/2=16 ✓
# seq must be divisible by 2*cp_size=8: 4096/8=512 ✓
MODEL_ARGS="
    --model_size llama-7b \
    --set_model_config_manually 0 \
    --set_layernum_manually 1 \
    --set_seqlen_manually 1 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_hidden_layers 2 \
    --num_attention_heads 32 \
    --seq_length 4096"

TRAIN_ARGS="
    --global_train_batch_size 8 \
    --train-iters 5 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.0 \
    --check_loss 0 \
    --profile 1 \
    --save_profiled_memory 0"

# Combined: tp=2 (Ulysses), cp=4 (Ring), dp=1
# vocab_tp should match tp_deg, vocab_cp should match cp_deg
PARALLEL_ARGS="
    --pp_deg 1 \
    --global_tp_deg 2 \
    --global_tp_consec 1 \
    --sdp 0 \
    --global_checkpoint 0 \
    --vocab_tp 2 \
    --vocab_cp 4 \
    --chunks 1 \
    --global_cp_deg 4 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --sequence-parallel \
    --use-ulysses \
    --use-flash-attn \
    --initialize_on_meta 1 \
    --use-packing"

echo "=============================================="
echo "Test Combined Ulysses(sp=2) + Ring(cp=4) + Varlen"
echo "=============================================="

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS}
