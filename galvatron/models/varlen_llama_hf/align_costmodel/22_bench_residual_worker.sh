#!/usr/bin/env bash
# Worker-side launcher for residual benchmark. Invoked by
# 22_bench_residual_dispatch.sh on both nodes with NODE_RANK={0,1}.
# Sweeps over (sp, seq_length, ckpt) cells; per cell, launches one torchrun
# that records `--adaCPSP-end2end-profile` JSONL.

set -euo pipefail

source "$(dirname "$0")/00_common.sh"
load_model_meta >/dev/null

# NCCL / cluster knobs (same set as 13_/20_).
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
export TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT="${TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT:-30}"

RESID_CELLS="${RESID_CELLS:-1:4096:0 8:4096:0 8:16384:1}"
RESID_GBS="${RESID_GBS:-16}"
RESID_TRAIN_ITERS="${RESID_TRAIN_ITERS:-20}"
RESID_PROFILE_START_ITER="${RESID_PROFILE_START_ITER:-5}"
RESID_PROFILE_END_ITER="${RESID_PROFILE_END_ITER:-19}"
RESID_PROFILE_RANKS="${RESID_PROFILE_RANKS:-0 8 15}"
PROFILE_DIR="${PROFILE_DIR:-${RUN_DIR:-.}/end2end}"
mkdir -p "${PROFILE_DIR}"

for cell in ${RESID_CELLS}; do
    sp="${cell%%:*}"
    rest="${cell#*:}"
    seq_len="${rest%%:*}"
    ckpt="${cell##*:}"

    cell_label="sp${sp}_seq${seq_len}_ckpt${ckpt}"
    cell_dir="${PROFILE_DIR}/${cell_label}"
    mkdir -p "${cell_dir}"

    log "RESID cell ${cell_label}  NODE_RANK=${NODE_RANK}"

    # Pick strategy: sp=1 → local attention; sp>1 → ulysses (cheapest correct
    # baseline; A2A is in the existing attention model so subtracting it gives
    # the pure residual).
    if [ "${sp}" = "1" ]; then
        strategy="ulysses:1"
    else
        strategy="ulysses:${sp}"
    fi

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
        --global_train_batch_size "${RESID_GBS}" \
        --train-iters "${RESID_TRAIN_ITERS}" \
        --lr 1e-4 \
        --adam_weight_decay 0.01 \
        --dropout_prob 0.0 \
        --check_loss 0 \
        --profile 1 \
        --profile_start_iter "${RESID_PROFILE_START_ITER}" \
        --profile_end_iter "${RESID_PROFILE_END_ITER}" \
        --save_profiled_memory 0 \
        --pp_deg 1 \
        --global_tp_deg 1 \
        --global_tp_consec 1 \
        --sdp 1 \
        --global_checkpoint "${ckpt}" \
        --vocab_tp 1 \
        --chunks 1 \
        --global_cp_deg 1 \
        --pipeline_type pipedream_flush \
        --default_dp_type zero2 \
        --mixed_precision bf16 \
        --use-flash-attn \
        --initialize_on_meta 1 \
        --use-packing \
        --use-adaCPSP \
        --adaCPSP-sync-solver \
        --adaCPSP-forced-strategy "${strategy}" \
        --adaCPSP-end2end-profile \
        --adaCPSP-end2end-profile-dir "${cell_dir}" \
        --adaCPSP-end2end-profile-ranks ${RESID_PROFILE_RANKS} \
        --dataset fix_length \
        || log "RESID cell ${cell_label} EXITED (rc=$?) — continuing sweep"
done

log "RESID benchmark worker done."
