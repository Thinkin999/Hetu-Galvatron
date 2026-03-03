#!/bin/bash
# Profile All-to-All Communication Bandwidth for AdaCPSP (Ulysses SP)
export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29510
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_HCA=mlx5_2,mlx5_5

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

# llama-7b config
PROFILE_ARGS="
    --hidden_size 4096 \
    --num_attention_heads 32 \
    --num_layers 32 \
    --warmup 5 \
    --iters 20 \
    --save_dir ./configs \
    --mode both"

${LAUNCHER} profile_alltoall.py ${PROFILE_ARGS}

