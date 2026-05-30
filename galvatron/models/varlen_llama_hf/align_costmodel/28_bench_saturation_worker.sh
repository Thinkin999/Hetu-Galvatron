#!/usr/bin/env bash
# Worker-side launcher for the saturation benchmark.
# Invoked by 28_bench_saturation_dispatch.sh with NODE_RANK={0,1}.
#
# Each cell encodes "<config>:<chunks>:<seq_label>".
#   seq_label ∈ {65k, 131k}  →  seq_length = 65536 / 131072
#   chunks   ∈ {1, 2, 4, 8}
#   config   ∈ {ulysses8, ring8, usp2x4}
#
# Key differences vs 26_bench_github_multimb_worker.sh:
#   - --sdp 1 --default_dp_type zero3   (ZeRO-3, production memory profile)
#   - --global_checkpoint 0             (ckpt=0, matches calibration & production)
#   - seq_length varies per cell

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

SAT_CELLS="${SAT_CELLS:-ulysses8:1:65k}"
SAT_GBS="${SAT_GBS:-16}"
SAT_TRAIN_ITERS="${SAT_TRAIN_ITERS:-24}"
SAT_PROFILE_START_ITER="${SAT_PROFILE_START_ITER:-5}"
SAT_PROFILE_END_ITER="${SAT_PROFILE_END_ITER:-23}"
SAT_PROFILE_RANKS="${SAT_PROFILE_RANKS:-0 8 15}"
PROFILE_DIR="${PROFILE_DIR:-${RUN_DIR:-.}/end2end}"
mkdir -p "${PROFILE_DIR}"

# Translate a config token to (forced_strategy).
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

# Translate seq label to integer seq_length.
resolve_seq() {
    case "$1" in
        65k)  echo 65536 ;;
        131k) echo 131072 ;;
        256k) echo 262144 ;;
        *) echo 0 ;;
    esac
}

for cell in ${SAT_CELLS}; do
    IFS=':' read -r cfg chunks seq_label <<<"${cell}"
    cell_label="${cfg}_chunks${chunks}_seq${seq_label}"
    cell_dir="${PROFILE_DIR}/${cell_label}"
    mkdir -p "${cell_dir}"

    forced_strategy="$(resolve_config "${cfg}")"
    seq_length="$(resolve_seq "${seq_label}")"
    if [ "${forced_strategy}" = "UNKNOWN" ] || [ "${seq_length}" = "0" ]; then
        log "SAT SKIP unknown cell ${cell}"
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

    log "SAT cell ${cell_label}  strategy='${forced_strategy:-auto}' chunks=${chunks} seq=${seq_length}  NODE_RANK=${NODE_RANK}"

    $(torchrun_prefix) "${MODEL_DIR}/train_dist_adacpsp.py" \
        --model_size "${MODEL_NAME}" \
        --set_model_config_manually 0 \
        --set_layernum_manually 0 \
        --set_seqlen_manually 1 \
        --vocab_size "${VOCAB_SIZE}" \
        --hidden_size "${HIDDEN}" \
        --num_hidden_layers "${N_LAYERS}" \
        --num_attention_heads "${N_HEADS}" \
        --seq_length "${seq_length}" \
        --global_train_batch_size "${SAT_GBS}" \
        --train-iters "${SAT_TRAIN_ITERS}" \
        --lr 1e-4 \
        --adam_weight_decay 0.01 \
        --dropout_prob 0.0 \
        --check_loss 0 \
        --profile 1 \
        --profile_start_iter "${SAT_PROFILE_START_ITER}" \
        --profile_end_iter "${SAT_PROFILE_END_ITER}" \
        --save_profiled_memory 0 \
        --pp_deg 1 \
        --global_tp_deg 1 \
        --global_tp_consec 1 \
        --sdp 1 \
        --global_checkpoint 0 \
        --vocab_tp 1 \
        --chunks 1 \
        --global_cp_deg 1 \
        --pipeline_type pipedream_flush \
        --default_dp_type zero3 \
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
        --adaCPSP-end2end-profile-ranks ${SAT_PROFILE_RANKS} \
        --dataset github \
        || log "SAT cell ${cell_label} EXITED (rc=$?) — continuing sweep"
done

log "SAT benchmark worker done."
