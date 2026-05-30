#!/usr/bin/env bash
# Worker-side launcher for github multi-mb validation benchmark.
# Invoked by 26_bench_github_multimb_dispatch.sh with NODE_RANK={0,1}.
#
# Each cell encodes "<config>:<chunks>":
#   ulysses8:1 / ulysses8:8 → forced ulysses with sp=8 (2 groups across 16 GPUs)
#   ring8:1   / ring8:8     → forced ring with cp=8 (2 groups)
#   usp2x4:1  / usp2x4:8    → forced usp sp=2 cp=4 (2 groups)
#   adacpsp:auto            → no forced strategy (solver decides)

set -euo pipefail

source "$(dirname "$0")/00_common.sh"
load_model_meta >/dev/null

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

GHMB_CELLS="${GHMB_CELLS:-ulysses8:1 adacpsp:auto}"
GHMB_GBS="${GHMB_GBS:-16}"
GHMB_SEQ_LENGTH="${GHMB_SEQ_LENGTH:-65536}"
GHMB_TRAIN_ITERS="${GHMB_TRAIN_ITERS:-22}"
GHMB_PROFILE_START_ITER="${GHMB_PROFILE_START_ITER:-5}"
GHMB_PROFILE_END_ITER="${GHMB_PROFILE_END_ITER:-21}"
GHMB_PROFILE_RANKS="${GHMB_PROFILE_RANKS:-0 8 15}"
PROFILE_DIR="${PROFILE_DIR:-${RUN_DIR:-.}/end2end}"
mkdir -p "${PROFILE_DIR}"

# Translate a config token to (forced_strategy, extra_args).
resolve_config() {
    local cfg="$1"
    case "${cfg}" in
        ulysses8) echo "ulysses:8" ;;
        ring8)    echo "ring:8" ;;
        usp2x4)   echo "usp:2x4" ;;
        adacpsp)  echo "" ;;
        *) echo "UNKNOWN" ;;
    esac
}

for cell in ${GHMB_CELLS}; do
    cfg="${cell%%:*}"
    chunks="${cell##*:}"
    cell_label="${cfg}_chunks${chunks}"
    cell_dir="${PROFILE_DIR}/${cell_label}"
    mkdir -p "${cell_dir}"

    forced_strategy="$(resolve_config "${cfg}")"
    if [ "${forced_strategy}" = "UNKNOWN" ]; then
        log "GHMB SKIP unknown config ${cfg}"
        continue
    fi

    forced_chunks_arg=""
    forced_strategy_arg=""
    if [ -n "${forced_strategy}" ]; then
        forced_strategy_arg="--adaCPSP-forced-strategy ${forced_strategy}"
        if [ "${chunks}" != "auto" ]; then
            forced_chunks_arg="--adaCPSP-forced-chunks ${chunks}"
        fi
    fi

    log "GHMB cell ${cell_label}  strategy='${forced_strategy:-auto}' chunks=${chunks}  NODE_RANK=${NODE_RANK}"

    $(torchrun_prefix) "${MODEL_DIR}/train_dist_adacpsp.py" \
        --model_size "${MODEL_NAME}" \
        --set_model_config_manually 0 \
        --set_layernum_manually 0 \
        --set_seqlen_manually 1 \
        --vocab_size "${VOCAB_SIZE}" \
        --hidden_size "${HIDDEN}" \
        --num_hidden_layers "${N_LAYERS}" \
        --num_attention_heads "${N_HEADS}" \
        --seq_length "${GHMB_SEQ_LENGTH}" \
        --global_train_batch_size "${GHMB_GBS}" \
        --train-iters "${GHMB_TRAIN_ITERS}" \
        --lr 1e-4 \
        --adam_weight_decay 0.01 \
        --dropout_prob 0.0 \
        --check_loss 0 \
        --profile 1 \
        --profile_start_iter "${GHMB_PROFILE_START_ITER}" \
        --profile_end_iter "${GHMB_PROFILE_END_ITER}" \
        --save_profiled_memory 0 \
        --pp_deg 1 \
        --global_tp_deg 1 \
        --global_tp_consec 1 \
        --sdp 1 \
        --global_checkpoint 1 \
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
        ${forced_strategy_arg} \
        ${forced_chunks_arg} \
        --adaCPSP-end2end-profile \
        --adaCPSP-end2end-profile-dir "${cell_dir}" \
        --adaCPSP-end2end-profile-ranks ${GHMB_PROFILE_RANKS} \
        --dataset github \
        || log "GHMB cell ${cell_label} EXITED (rc=$?) — continuing sweep"
done

log "GHMB benchmark worker done."
