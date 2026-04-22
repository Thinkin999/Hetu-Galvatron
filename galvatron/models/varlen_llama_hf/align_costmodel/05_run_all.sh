#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/00_common.sh"

log "ALIGN_RUN_ID=${ALIGN_RUN_ID}"
log "MODEL_NAME=${MODEL_NAME}"
log "WORLD_SIZE=${WORLD_SIZE} (${NNODES} x ${NPROC_PER_NODE})"
log "MASTER=${MASTER_ADDR}:${MASTER_PORT}"

bash "${ALIGN_DIR}/01_profile_attention.sh"
bash "${ALIGN_DIR}/02_profile_comm.sh"
bash "${ALIGN_DIR}/03_run_real_strategies.sh"

if [ "${NODE_RANK}" = "0" ]; then
    python3 "${ALIGN_DIR}/04_align_costmodel.py" \
      --result-dir "${RESULT_ROOT}" \
      --world-size "${WORLD_SIZE}" \
      --gpus-per-node "${NPROC_PER_NODE}"
fi
