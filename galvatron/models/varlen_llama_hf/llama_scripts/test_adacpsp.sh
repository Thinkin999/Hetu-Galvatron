#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
# ═══════════════════════════════════════════════════════════════
# Test: AdaCPSP — Solver-driven adaptive heterogeneous groups
# ═══════════════════════════════════════════════════════════════
# Key design:
#   - tp_deg=1 (no tensor parallelism, full weights on every GPU)
#   - FSDP dp=world_size (all GPUs share model via FSDP)
#   - sp_size and cp_size are BOTH dynamic per group
#   - Each microbatch can have HETEROGENEOUS groups:
#     e.g., ranks 0-3: Ulysses×4, ranks 4-7: Ring×4
#   - force_all_modules ensures Flash, Ulysses, Ring modules all created

export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29504
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
    --global_train_batch_size 16 \
    --train-iters 10 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.0 \
    --check_loss 0 \
    --profile 1 \
    --save_profiled_memory 0"

# AdaCPSP: tp=1, dp=world_size
# --use-adaCPSP triggers:
#   1. force_all_attn_modules in Attention.__init__
#   2. Override tp=1, sp=1, cp=1 in train_dist_adacpsp.py
#   3. Solver-driven heterogeneous groups in collate_fn
PARALLEL_ARGS="
    --pp_deg 1 \
    --global_tp_deg 1 \
    --global_tp_consec 1 \
    --sdp 0 \
    --global_checkpoint 0 \
    --vocab_tp 1 \
    --chunks 1 \
    --global_cp_deg 1 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --use-flash-attn \
    --initialize_on_meta 1 \
    --use-packing \
    --use-adaCPSP"

echo "=============================================="
echo "Test: AdaCPSP Heterogeneous Groups (tp=1)"
echo "  - FSDP dp=world_size"
echo "  - Dynamic sp_size + cp_size per group"
echo "  - Solver determines heterogeneous strategy"
echo "=============================================="

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS}
