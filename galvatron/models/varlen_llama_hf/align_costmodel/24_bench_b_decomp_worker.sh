#!/usr/bin/env bash
# Worker-side launcher for b-decomposition benchmark. Invoked by
# 24_bench_b_decomp_dispatch.sh on both nodes with NODE_RANK={0,1}.
# Sweeps over (sp, seq_length, chunks) cells; per cell, launches one torchrun
# that records `--adaCPSP-end2end-profile` JSONL.
#
# Per cell:
#   strategy   = ulysses:${sp}              (so attention model has Ulysses A2A built-in)
#   num_groups = world_size / sp            (parallel forced groups per microbatch)
#   GBS        = world_size * chunks        (Megatron requires GBS % world_size == 0;
#                                            yields seqs_per_group = sp, so
#                                            tokens_per_GPU = sp * seq_len / sp = seq_len
#                                            -- INDEPENDENT of sp, perfect for sweep)
#   chunks     = sequential microbatches via --adaCPSP-forced-chunks

set -euo pipefail

source "$(dirname "$0")/00_common.sh"
load_model_meta >/dev/null

# NCCL / cluster knobs (same set as 13_/20_/22_).
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

BDEC_CELLS="${BDEC_CELLS:-1:8192:1 1:8192:2 8:8192:1 8:8192:2}"
BDEC_TRAIN_ITERS="${BDEC_TRAIN_ITERS:-22}"
BDEC_PROFILE_START_ITER="${BDEC_PROFILE_START_ITER:-5}"
BDEC_PROFILE_END_ITER="${BDEC_PROFILE_END_ITER:-21}"
BDEC_PROFILE_RANKS="${BDEC_PROFILE_RANKS:-0 8 15}"
PROFILE_DIR="${PROFILE_DIR:-${RUN_DIR:-.}/end2end}"
mkdir -p "${PROFILE_DIR}"

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

for cell in ${BDEC_CELLS}; do
    sp="${cell%%:*}"
    rest="${cell#*:}"
    seq_len="${rest%%:*}"
    chunks="${cell##*:}"

    if (( WORLD_SIZE % sp != 0 )); then
        log "BDEC cell ${cell} SKIP: world_size=${WORLD_SIZE} not divisible by sp=${sp}"
        continue
    fi
    num_groups=$(( WORLD_SIZE / sp ))
    # GBS = WORLD_SIZE * chunks satisfies Megatron's "GBS % world_size == 0".
    # It also yields seqs_per_group = WORLD_SIZE * chunks / num_groups / chunks = sp
    # sequences per group per microbatch, giving tokens_per_GPU = sp * seq_len / sp
    # = seq_len, INDEPENDENT of sp. The chunks-vs-fb fit therefore isolates the
    # per-microbatch and per-step components cleanly.
    gbs=$(( WORLD_SIZE * chunks ))

    cell_label="sp${sp}_seq${seq_len}_chunks${chunks}"
    cell_dir="${PROFILE_DIR}/${cell_label}"
    mkdir -p "${cell_dir}"

    log "BDEC cell ${cell_label}  num_groups=${num_groups}  gbs=${gbs}  NODE_RANK=${NODE_RANK}"

    strategy="ulysses:${sp}"

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
        --global_train_batch_size "${gbs}" \
        --train-iters "${BDEC_TRAIN_ITERS}" \
        --lr 1e-4 \
        --adam_weight_decay 0.01 \
        --dropout_prob 0.0 \
        --check_loss 0 \
        --profile 1 \
        --profile_start_iter "${BDEC_PROFILE_START_ITER}" \
        --profile_end_iter "${BDEC_PROFILE_END_ITER}" \
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
        --default_dp_type zero2 \
        --mixed_precision bf16 \
        --use-flash-attn \
        --initialize_on_meta 1 \
        --use-packing \
        --use-adaCPSP \
        --adaCPSP-sync-solver \
        --adaCPSP-forced-strategy "${strategy}" \
        --adaCPSP-forced-chunks "${chunks}" \
        --adaCPSP-end2end-profile \
        --adaCPSP-end2end-profile-dir "${cell_dir}" \
        --adaCPSP-end2end-profile-ranks ${BDEC_PROFILE_RANKS} \
        --dataset fix_length \
        || log "BDEC cell ${cell_label} EXITED (rc=$?) — continuing sweep"
done

log "BDEC benchmark worker done."
