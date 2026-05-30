#!/usr/bin/env bash
# Worker-side launcher for FSDP step tracing. Invoked by
# 20_trace_fsdp_step_dispatch.sh on both nodes with NODE_RANK={0,1}.
#
# For each seq length in FSDP_SEQ_LENGTHS, launches torchrun train_dist_adacpsp
# with the timeline profiler enabled around a small iter window. Output traces
# land under ${TRACE_ROOT}/seq<L>/rank<R>.*.json.

set -euo pipefail

source "$(dirname "$0")/00_common.sh"
load_model_meta >/dev/null

# A800 / IDC NCCL settings (mirrors comm worker)
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# Avoid the 5-minute elastic exit barrier between seq cases
export TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT="${TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT:-30}"

FSDP_SEQ_LENGTHS_LIST=(${FSDP_SEQ_LENGTHS:-4096 16384})
FSDP_STRATEGY="${FSDP_STRATEGY:-ulysses:8}"
FSDP_GBS="${FSDP_GBS:-16}"
FSDP_TRAIN_ITERS="${FSDP_TRAIN_ITERS:-6}"
FSDP_TIMELINE_START_ITER="${FSDP_TIMELINE_START_ITER:-3}"
FSDP_TIMELINE_END_ITER="${FSDP_TIMELINE_END_ITER:-5}"
FSDP_TIMELINE_RANKS="${FSDP_TIMELINE_RANKS:-0 8}"
FSDP_SDP="${FSDP_SDP:-1}"
FSDP_DEFAULT_DP="${FSDP_DEFAULT_DP:-zero2}"
FSDP_USE_CKPT="${FSDP_USE_CKPT:-0}"
TRACE_ROOT="${TRACE_ROOT:-${RUN_DIR:-.}/traces}"
mkdir -p "${TRACE_ROOT}"

safe_strategy="$(echo "${FSDP_STRATEGY}" | tr ':,x' '___')"

for seq_len in "${FSDP_SEQ_LENGTHS_LIST[@]}"; do
    log "FSDP step trace seq=${seq_len} strategy=${FSDP_STRATEGY} NODE_RANK=${NODE_RANK}"

    SEQ_TRACE_DIR="${TRACE_ROOT}/seq${seq_len}_${safe_strategy}"
    mkdir -p "${SEQ_TRACE_DIR}"

    # Mirror H20 production launcher: model_size + explicit dims (some megatron
    # args check hidden_size etc. directly before the meta-config dispatch).
    $(torchrun_prefix) "${MODEL_DIR}/train_dist_adacpsp.py" \
        --model_size "${MODEL_NAME}" \
        --set_model_config_manually 0 \
        --set_layernum_manually 0 \
        --set_seqlen_manually 1 \
        --vocab_size "${VOCAB_SIZE}" \
        --hidden_size "${HIDDEN}" \
        --num_hidden_layers "${N_LAYERS}" \
        --num_attention_heads "${N_HEADS}" \
        --seq_length "${seq_len}" \
        --global_train_batch_size "${FSDP_GBS}" \
        --train-iters "${FSDP_TRAIN_ITERS}" \
        --lr 1e-4 \
        --adam_weight_decay 0.01 \
        --dropout_prob 0.0 \
        --check_loss 0 \
        --profile 1 \
        --profile_start_iter "${FSDP_TIMELINE_START_ITER}" \
        --profile_end_iter "${FSDP_TIMELINE_END_ITER}" \
        --save_profiled_memory 0 \
        --pp_deg 1 \
        --global_tp_deg 1 \
        --global_tp_consec 1 \
        --sdp "${FSDP_SDP}" \
        --global_checkpoint "${FSDP_USE_CKPT}" \
        --vocab_tp 1 \
        --chunks 1 \
        --global_cp_deg 1 \
        --pipeline_type pipedream_flush \
        --default_dp_type "${FSDP_DEFAULT_DP}" \
        --mixed_precision bf16 \
        --use-flash-attn \
        --initialize_on_meta 1 \
        --use-packing \
        --use-adaCPSP \
        --adaCPSP-sync-solver \
        --adaCPSP-forced-strategy "${FSDP_STRATEGY}" \
        --adaCPSP-timeline-profile \
        --adaCPSP-timeline-dir "${SEQ_TRACE_DIR}" \
        --adaCPSP-timeline-ranks ${FSDP_TIMELINE_RANKS} \
        --adaCPSP-timeline-start-iter "${FSDP_TIMELINE_START_ITER}" \
        --adaCPSP-timeline-end-iter "${FSDP_TIMELINE_END_ITER}" \
        --dataset fix_length
done

log "FSDP step trace worker done."
