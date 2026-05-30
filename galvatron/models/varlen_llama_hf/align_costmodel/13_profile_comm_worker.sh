#!/bin/bash
# Worker-side launcher for comm profile. Must be invoked on BOTH worker-0
# (NODE_RANK=0) and worker-1 (NODE_RANK=1) with the same MASTER_ADDR/PORT and
# COMM_RUN_ID, e.g. via 13_profile_comm_dispatch.sh.

set -euo pipefail

source "$(dirname "$0")/00_common.sh"
load_model_meta >/dev/null

# A800/IDC NCCL settings; same set as 08_trace_ring_attention.sh so cross-node
# IB works when we are NOT launched via Arnold entrypoint.
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

COMM_RUN_ID="${COMM_RUN_ID:-comm_v2_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${ALIGN_DIR}/results/${COMM_RUN_ID}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

PRIMITIVES="${COMM_V2_PRIMITIVES:-alltoall_single p2p_sendrecv p2p_kv_pair}"
MESSAGE_SIZES_MB="${COMM_V2_MESSAGE_SIZES_MB:-0.25 0.5 1 2 4 8 16 32 64 128 256}"
TOPOLOGY="${COMM_V2_TOPOLOGY:-both}"

CMD="$(torchrun_prefix) \"${MODEL_DIR}/profile_comm_v2.py\" \
  --model-name \"${MODEL_NAME}\" \
  --save-dir \"${CONFIGS_DIR}\" \
  --primitives ${PRIMITIVES} \
  --message-sizes-mb \"${MESSAGE_SIZES_MB}\" \
  --topology \"${TOPOLOGY}\" \
  --warmup \"${WARMUP_ITERS}\" \
  --iters \"${MEASURE_ITERS}\" \
  --across-group-agg \"${ACROSS_GROUP_AGG:-p90}\""

log "Comm profile v2 ${COMM_RUN_ID}"
log "primitives=${PRIMITIVES}"
log "message_sizes_MB=${MESSAGE_SIZES_MB}"
log "NODE_RANK=${NODE_RANK}  MASTER=${MASTER_ADDR}:${MASTER_PORT}"
log "Command: ${CMD}"

LOG_FILE="${LOG_DIR}/comm_worker_node${NODE_RANK}.log"
bash -c "${CMD}" 2>&1 | tee "${LOG_FILE}"
