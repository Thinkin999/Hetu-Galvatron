#!/bin/bash
set -euo pipefail

# ============================================
# Worker node launcher for align_costmodel
# Target setup: 2 nodes x 8 GPUs = 16 GPUs
# Model: qwen2.5-7b
# ============================================

cd /mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron

# ---------- Required / recommended environment ----------
export ROOT="${ROOT:-/mnt/bn/wyj-data0-hl/lqs}"
export ADACPSP_ENV="${ADACPSP_ENV:-$ROOT/envs/galvatron-adacpsp-py39-torch21-cu121}"
export MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# IMPORTANT:
# For a 2-node run, this should be 1 on the non-master node.
export NODE_RANK="${NODE_RANK:-1}"

# Must be exactly the same as master node.
export MASTER_ADDR="${MASTER_ADDR:-${METIS_WORKER_0_HOST:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Must be exactly the same as master node.
export ALIGN_RUN_ID="${ALIGN_RUN_ID:-align_qwen25_7b_16gpu_001}"

# Must be exactly the same as master node.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export ALIGN_SEQ_LENGTHS="${ALIGN_SEQ_LENGTHS:-2048 4096 8192 16384}"
export WARMUP_ITERS="${WARMUP_ITERS:-5}"
export MEASURE_ITERS="${MEASURE_ITERS:-20}"
export ACROSS_GROUP_AGG="${ACROSS_GROUP_AGG:-p90}"
export BENCH_GROUP_SIZES="${BENCH_GROUP_SIZES:-1 2 4 8 16}"

echo "=========================================="
echo "align_costmodel worker launcher"
echo "=========================================="
echo "MODEL_NAME=${MODEL_NAME}"
echo "NNODES=${NNODES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "ALIGN_RUN_ID=${ALIGN_RUN_ID}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "ALIGN_SEQ_LENGTHS=${ALIGN_SEQ_LENGTHS}"
echo "BENCH_GROUP_SIZES=${BENCH_GROUP_SIZES}"
echo "=========================================="

bash galvatron/models/varlen_llama_hf/align_costmodel/05_run_all.sh
