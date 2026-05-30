#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

# Multi-node defaults follow the 2x8 setup used for AdaCPSP profiling.
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29631}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"
SEQ_LENGTH="${SEQ_LENGTH:-131072}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
PROFILE_START_ITER="${PROFILE_START_ITER:-2}"
PROFILE_END_ITER="${PROFILE_END_ITER:-9}"
PROFILE_RANKS="${PROFILE_RANKS:-0 8 15}"
TIMELINE_RANKS="${TIMELINE_RANKS:-0 8 15}"
TIMELINE_START_ITER="${TIMELINE_START_ITER:-5}"
TIMELINE_END_ITER="${TIMELINE_END_ITER:-6}"
ENABLE_TIMELINE="${ENABLE_TIMELINE:-0}"
STRATEGIES="${STRATEGIES:-ulysses:16 ring:16 usp:4x4}"
RUN_ID="${RUN_ID:-adacpsp_end2end_$(date +%Y%m%d_%H%M%S)}"

PROFILE_DIR="${PROFILE_DIR:-configs/end2end_profiles/${RUN_ID}}"
TRACE_ROOT="${TRACE_ROOT:-configs/end2end_traces/${RUN_ID}}"
mkdir -p "${PROFILE_DIR}" "${TRACE_ROOT}"

LAUNCHER=(
  torchrun
  --nnodes "${NNODES}"
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  --node_rank "${NODE_RANK}"
)

COMMON_ARGS=(
  --model_size "${MODEL_NAME}"
  --set_model_config_manually 0
  --set_layernum_manually 0
  --set_seqlen_manually 1
  --seq_length "${SEQ_LENGTH}"
  --global_train_batch_size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS}"
  --lr 1e-4
  --adam_weight_decay 0.01
  --dropout_prob 0.0
  --check_loss 0
  --profile 1
  --profile_start_iter "${PROFILE_START_ITER}"
  --profile_end_iter "${PROFILE_END_ITER}"
  --save_profiled_memory 0
  --pp_deg 1
  --global_tp_deg 1
  --global_tp_consec 1
  --sdp 0
  --global_checkpoint 0
  --vocab_tp 1
  --chunks 1
  --global_cp_deg 1
  --pipeline_type pipedream_flush
  --default_dp_type zero2
  --mixed_precision bf16
  --use-flash-attn
  --initialize_on_meta 1
  --use-packing
  --use-adaCPSP
  --adaCPSP-sync-solver
  --adaCPSP-end2end-profile
  --adaCPSP-end2end-profile-dir "${PROFILE_DIR}"
  --adaCPSP-end2end-profile-ranks ${PROFILE_RANKS}
)

echo "============================================================"
echo "AdaCPSP end-to-end profile"
echo "  NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}, NODE_RANK=${NODE_RANK}"
echo "  MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "  MODEL=${MODEL_NAME}, SEQ_LENGTH=${SEQ_LENGTH}, GBS=${GLOBAL_BATCH_SIZE}"
echo "  STRATEGIES=${STRATEGIES}"
echo "  PROFILE_DIR=${PROFILE_DIR}"
echo "  ENABLE_TIMELINE=${ENABLE_TIMELINE}"
echo "============================================================"

for strategy in ${STRATEGIES}; do
  echo "============================================================"
  echo "Running forced strategy: ${strategy}"
  echo "============================================================"

  EXTRA_ARGS=(
    --adaCPSP-forced-strategy "${strategy}"
  )

  if [ "${ENABLE_TIMELINE}" = "1" ]; then
    safe_strategy="$(echo "${strategy}" | tr ':,x' '___')"
    EXTRA_ARGS+=(
      --adaCPSP-timeline-profile
      --adaCPSP-timeline-dir "${TRACE_ROOT}/${safe_strategy}"
      --adaCPSP-timeline-ranks ${TIMELINE_RANKS}
      --adaCPSP-timeline-start-iter "${TIMELINE_START_ITER}"
      --adaCPSP-timeline-end-iter "${TIMELINE_END_ITER}"
    )
  fi

  "${LAUNCHER[@]}" train_dist_adacpsp.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
done

if [ "${NODE_RANK}" = "0" ]; then
  python3 analyze_end2end_profile.py \
    --profile-dir "${PROFILE_DIR}" \
    --output-dir "${PROFILE_DIR}/analysis"
fi
