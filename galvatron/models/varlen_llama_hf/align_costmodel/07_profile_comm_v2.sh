#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/00_common.sh"
write_run_metadata
load_model_meta >/dev/null

COMM_V2_LOG_BASE="${LOG_DIR}/07_profile_comm_v2_${MODEL_NAME}"
if [ "${NODE_RANK}" = "0" ]; then
    COMM_V2_LOG="${COMM_V2_LOG_BASE}.log"
else
    COMM_V2_LOG="${COMM_V2_LOG_BASE}.node${NODE_RANK}.log"
fi

PRIMITIVES="${COMM_V2_PRIMITIVES:-alltoall_single p2p_sendrecv}"
MESSAGE_SIZES_MB="${COMM_V2_MESSAGE_SIZES_MB:-1 2 4 8 16 32 64 128 256 512 1024}"
TOPOLOGY="${COMM_V2_TOPOLOGY:-both}"

log "Starting primitive communication profiling v2 for ${MODEL_NAME}"
log "World size: ${WORLD_SIZE} (${NNODES} x ${NPROC_PER_NODE})"
log "Primitives: ${PRIMITIVES}"
log "Message sizes MB: ${MESSAGE_SIZES_MB}"

CMD="$(torchrun_prefix) \"${MODEL_DIR}/profile_comm_v2.py\" \
  --model-name \"${MODEL_NAME}\" \
  --save-dir \"${CONFIGS_DIR}\" \
  --primitives ${PRIMITIVES} \
  --message-sizes-mb \"${MESSAGE_SIZES_MB}\" \
  --topology \"${TOPOLOGY}\" \
  --warmup \"${WARMUP_ITERS}\" \
  --iters \"${MEASURE_ITERS}\" \
  --across-group-agg \"${ACROSS_GROUP_AGG}\""

if [ "${NODE_RANK}" = "0" ]; then
    {
        echo "=== Command ==="
        echo "${CMD}"
        echo "=== Start: $(date) ==="
        echo
    } > "${COMM_V2_LOG}"
fi

bash -c "${CMD}" 2>&1 | tee -a "${COMM_V2_LOG}"
