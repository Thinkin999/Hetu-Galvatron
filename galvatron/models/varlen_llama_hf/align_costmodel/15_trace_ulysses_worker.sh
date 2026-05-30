#!/bin/bash
# Worker-side launcher for Ulysses trace. Mirrors 08_trace_ring_attention.sh.

set -euo pipefail

source "$(dirname "$0")/00_common.sh"
load_model_meta >/dev/null

# A800/IDC NCCL settings
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-23}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-2}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=eth0}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

TRACE_RUN_ID="${TRACE_RUN_ID:-uly_trace_$(date +%Y%m%d_%H%M%S)}"
TRACE_ROOT="${ALIGN_DIR}/results/${TRACE_RUN_ID}"
TRACE_DIR="${TRACE_DIR:-${TRACE_ROOT}/traces}"
TRACE_LOG_DIR="${TRACE_ROOT}/logs"
mkdir -p "${TRACE_DIR}" "${TRACE_LOG_DIR}"

TRACE_CASES="${TRACE_CASES:-8192:16 32768:16}"
TRACE_NUM_SEQS="${TRACE_NUM_SEQS:-16}"
TRACE_PROFILE_RANKS="${TRACE_PROFILE_RANKS:-0 8}"
TRACE_WARMUP="${TRACE_WARMUP:-3}"
TRACE_ACTIVE="${TRACE_ACTIVE:-3}"

export GALVATRON_ULYSSES_TRACE=1

CMD="$(torchrun_prefix) \"${ALIGN_DIR}/15_trace_ulysses.py\" \
  --trace-dir \"${TRACE_DIR}\" \
  --cases \"${TRACE_CASES}\" \
  --num-seqs \"${TRACE_NUM_SEQS}\" \
  --n-heads \"${N_HEADS}\" \
  --n-kv-heads \"${N_KV_HEADS}\" \
  --head-dim \"${HEAD_DIM}\" \
  --warmup \"${TRACE_WARMUP}\" \
  --active \"${TRACE_ACTIVE}\" \
  --profile-ranks ${TRACE_PROFILE_RANKS}"

log "Ulysses trace sweep"
log "TRACE_RUN_ID=${TRACE_RUN_ID}"
log "TRACE_DIR=${TRACE_DIR}"
log "TRACE_CASES=${TRACE_CASES}"
log "NODE_RANK=${NODE_RANK}  MASTER=${MASTER_ADDR}:${MASTER_PORT}"
log "Command: ${CMD}"

LOG_FILE="${TRACE_LOG_DIR}/trace_node${NODE_RANK}.log"
bash -c "${CMD}" 2>&1 | tee "${LOG_FILE}"
