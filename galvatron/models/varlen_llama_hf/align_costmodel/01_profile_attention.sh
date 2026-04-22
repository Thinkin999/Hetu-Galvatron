#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/00_common.sh"
write_run_metadata
load_model_meta >/dev/null

if [ "${NODE_RANK}" != "0" ]; then
    log "Attention profiling only runs on node rank 0. Skipping on rank ${NODE_RANK}."
    exit 0
fi

ATTN_LOG="${LOG_DIR}/01_profile_attention_${MODEL_NAME}.log"

log "Starting attention profiling for ${MODEL_NAME}"
log "Log file: ${ATTN_LOG}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 "${ATTN_PROFILE_SCRIPT}" \
  --mode attention \
  --n_heads "${N_HEADS}" \
  --n_kv_heads "${N_KV_HEADS}" \
  --head_dim "${HEAD_DIM}" \
  --hidden_size "${HIDDEN}" \
  --num_layers "${N_LAYERS}" \
  --model_name "${MODEL_NAME}" \
  --save_dir "${CONFIGS_DIR}" \
  --warmup "${WARMUP_ITERS}" \
  --iters "${MEASURE_ITERS}" \
  --attn_step "${ATTN_STEP}" \
  --attn_max "${ATTN_MAX}" \
  --attn_timing_groups 3 \
  --attn_timing_stat median \
  --attn_min_group_elapsed_ms 1.0 \
  --attn_max_iters_per_group 512 \
  --skip_head_scaling_check \
  2>&1 | tee "${ATTN_LOG}"

LATEST_ATTN="$(latest_attention_profile)"
if [ -n "${LATEST_ATTN}" ]; then
    echo "${LATEST_ATTN}" > "${SUMMARY_DIR}/attention_profile_path.txt"
    log "Latest attention profile: ${LATEST_ATTN}"
else
    log "WARNING: could not auto-detect attention profile output"
fi
