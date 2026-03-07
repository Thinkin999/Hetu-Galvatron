#!/bin/bash
# End-to-end AdaCPSP test with real varlen dataset (wikipedia/common_crawl/github)
# 8 GPU distributed
cd "$(dirname "$0")/../"

NUM_NODES=1
NUM_GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=9992
NODE_RANK=0

# Dataset: wikipedia / common_crawl / github / random / fix_length
DATASET=${1:-"wikipedia"}

TRAINER="train_dist_adacpsp.py"

torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank=${NODE_RANK} \
    ${TRAINER} \
    --model_size llama-7b \
    --hidden_size 4096 \
    --num_attention_heads 32 \
    --seq_length 32768 \
    --global_train_batch_size 16 \
    --epochs 1 \
    --lr 0.00015 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 0 \
    --profile 0 \
    --save_profiled_memory 0 \
    --global_tp_deg 1 \
    --global_tp_consec 1 \
    --vocab_tp 1 \
    --chunks 1 \
    --pp_deg 1 \
    --use-flash-attn \
    --mixed_precision bf16 \
    --use-adaCPSP \
    --adaCPSP-strategy adaptive \
    --dataset ${DATASET} \
    --train-iters 5 \
    --global_cp_deg 1
