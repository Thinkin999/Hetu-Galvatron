#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Experiment: E2_mixed — AdaCPSP (Ulysses+Ring+USP)
# Description: Mixed lengths (<=32k), moderate GBS
# Config: max_seq=32768, GBS=32, layers=32
# ═══════════════════════════════════════════════════════════════════════
cd "$(dirname "$0")/.." || exit 1

export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29567
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_HCA=mlx5_2,mlx5_5

mkdir -p logs

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║ Experiment: E2_mixed     — ADACPSP                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ max_seq= 32768 | GBS= 32 | layers=32 | iters=20       ║"
echo "║ attn_types: ulysses ring usp                              ║"
echo "║ dataset: wikipedia                                        ║"
echo "║ memory_limit: 36 GB                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

START_TIME=$(date +%s)

torchrun \
    --nnodes ${NUM_NODES} \
    --nproc_per_node ${NUM_GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --node_rank ${NODE_RANK} \
    train_dist_adacpsp.py \
    --model_size llama-7b \
    --set_model_config_manually 0 \
    --set_layernum_manually 1 \
    --set_seqlen_manually 1 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_hidden_layers 32 \
    --num_attention_heads 32 \
    --seq_length 32768 \
    --global_train_batch_size 32 \
    --train-iters 20 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.0 \
    --check_loss 0 \
    --profile 1 \
    --save_profiled_memory 0 \
    --dataset wikipedia \
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
    --use-adaCPSP \
    --adaCPSP-attn-types ulysses ring usp \
    --memory-limit-gb 36 \
    2>&1 | tee logs/exp_E2_mixed_adacpsp.log

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Experiment E2_mixed (adacpsp) completed in ${ELAPSED}s"
echo "Log: logs/exp_E2_mixed_adacpsp.log"
echo "═══════════════════════════════════════════════════════════════"
