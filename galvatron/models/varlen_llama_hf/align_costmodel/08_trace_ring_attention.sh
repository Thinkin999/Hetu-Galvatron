#!/bin/bash
# Multi-case Ring-attention trace runner. Must be launched on BOTH worker-0
# (NODE_RANK=0) and worker-1 (NODE_RANK=1) with the SAME MASTER_ADDR/PORT
# and SAME TRACE_RUN_ID, e.g. via 08_trace_ring_dispatch.sh.

set -euo pipefail

source "$(dirname "$0")/00_common.sh"
load_model_meta >/dev/null

# NCCL settings for byted A800 IDC. Mirror arnold/init_env/nvidia.sh defaults
# (we are not started via Arnold entrypoint so the env is NOT auto-injected).
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-23}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-2}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=eth0}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# A800-SXM-80 (CN region) wants relaxed PCI ordering.
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
# Make cross-node hangs visible within ~5 min instead of 30.
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

# Trace runs share a result dir per run id (separate from benchmark runs).
TRACE_RUN_ID="${TRACE_RUN_ID:-trace_$(date +%Y%m%d_%H%M%S)}"
TRACE_ROOT="${ALIGN_DIR}/results/${TRACE_RUN_ID}"
TRACE_DIR="${TRACE_DIR:-${TRACE_ROOT}/traces}"
TRACE_LOG_DIR="${TRACE_ROOT}/logs"
mkdir -p "${TRACE_DIR}" "${TRACE_LOG_DIR}"

# Default sweep: seq lengths × cp sizes for 16 GPU.
TRACE_CASES="${TRACE_CASES:-4096:2 4096:8 4096:16 8192:2 8192:8 8192:16 16384:2 16384:8 16384:16 32768:2 32768:8 32768:16}"
TRACE_NUM_SEQS="${TRACE_NUM_SEQS:-1}"
TRACE_TOPOLOGY="${TRACE_TOPOLOGY:-consecutive}"
TRACE_PROFILE_RANKS="${TRACE_PROFILE_RANKS:-0 8 15}"
TRACE_WARMUP="${TRACE_WARMUP:-5}"
TRACE_ACTIVE="${TRACE_ACTIVE:-3}"

# Mark every ring step with record_function inside attention_impl.
export GALVATRON_RING_TRACE_PER_STEP=1

CMD="$(torchrun_prefix) \"${ALIGN_DIR}/08_trace_ring_attention.py\" \
  --trace-dir \"${TRACE_DIR}\" \
  --cases \"${TRACE_CASES}\" \
  --num-seqs \"${TRACE_NUM_SEQS}\" \
  --topology \"${TRACE_TOPOLOGY}\" \
  --n-heads \"${N_HEADS}\" \
  --n-kv-heads \"${N_KV_HEADS}\" \
  --head-dim \"${HEAD_DIM}\" \
  --warmup \"${TRACE_WARMUP}\" \
  --active \"${TRACE_ACTIVE}\" \
  --profile-ranks ${TRACE_PROFILE_RANKS}"

log "Ring trace sweep"
log "TRACE_RUN_ID=${TRACE_RUN_ID}"
log "TRACE_DIR=${TRACE_DIR}"
log "TRACE_CASES=${TRACE_CASES}"
log "NODE_RANK=${NODE_RANK}  MASTER=${MASTER_ADDR}:${MASTER_PORT}"
log "Command: ${CMD}"

LOG_FILE="${TRACE_LOG_DIR}/trace_node${NODE_RANK}.log"
bash -c "${CMD}" 2>&1 | tee "${LOG_FILE}"
