#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/00_common.sh"
write_run_metadata
load_model_meta >/dev/null

COMM_LOG_BASE="${LOG_DIR}/02_profile_comm_${MODEL_NAME}"
if [ "${NODE_RANK}" = "0" ]; then
    COMM_LOG="${COMM_LOG_BASE}.log"
else
    COMM_LOG="${COMM_LOG_BASE}.node${NODE_RANK}.log"
fi

log "Starting topology-aware communication profiling for ${MODEL_NAME}"
log "World size: ${WORLD_SIZE} (${NNODES} x ${NPROC_PER_NODE})"
log "Log file: ${COMM_LOG}"

CMD="$(torchrun_prefix) \"${COMM_PROFILE_SCRIPT}\" \
  --hidden_size \"${HIDDEN}\" \
  --num_attention_heads \"${N_HEADS}\" \
  --num_kv_heads \"${N_KV_HEADS}\" \
  --num_layers \"${N_LAYERS}\" \
  --model_name \"${MODEL_NAME}\" \
  --save_dir \"${CONFIGS_DIR}\" \
  --mode both \
  --topology both \
  --gpus_per_node \"${NPROC_PER_NODE}\" \
  --across-group-agg \"${ACROSS_GROUP_AGG}\""

if [ "${NODE_RANK}" = "0" ]; then
    {
        echo "=== Command ==="
        echo "${CMD}"
        echo "=== Start: $(date) ==="
        echo
    } > "${COMM_LOG}"
fi

bash -c "${CMD}" 2>&1 | tee -a "${COMM_LOG}"

if [ "${NODE_RANK}" = "0" ]; then
    LATEST_COMM="$(latest_comm_profile)"
    if [ -n "${LATEST_COMM}" ]; then
        echo "${LATEST_COMM}" > "${SUMMARY_DIR}/comm_profile_path.txt"
        log "Latest communication profile: ${LATEST_COMM}"
    else
        log "WARNING: could not auto-detect communication profile output"
    fi
fi
