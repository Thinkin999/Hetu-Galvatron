#!/bin/bash
###############################################################################
# AdaCPSP experiment runner for ByteDance Merlin Seed.
#
# This script mirrors experiments/run_all_experiments.sh, but:
#   1. torchrun derives node information from platform-injected env vars
#   2. dataset mount can be wired through DATASET_MOUNT_DIR
#   3. the default sweep uses the currently allocated platform resources
###############################################################################

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_DIR}/galvatron/models/varlen_llama_hf/train_dist_adacpsp.py"

# Platform-provided distributed settings.
PLATFORM_NPROC_PER_NODE="${ARNOLD_WORKER_GPU:-${NPROC_PER_NODE:-8}}"
PLATFORM_NNODES="${ARNOLD_WORKER_NUM:-${NNODES:-1}}"
PLATFORM_NODE_RANK="${ARNOLD_ID:-${NODE_RANK:-0}}"
PLATFORM_MASTER_ADDR="${METIS_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}"
PLATFORM_MASTER_PORT="${METIS_WORKER_0_PORT:-${MASTER_PORT:-29500}}"
ALLOCATED_GPUS=$((PLATFORM_NPROC_PER_NODE * PLATFORM_NNODES))

# Sweep dimensions.
MODELS="${MODELS:-qwen2.5-7b qwen2.5-14b qwen2.5-32b}"
SEQ_LENGTHS_K="${SEQ_LENGTHS_K:-128 256 384 512}"
GBS_LIST="${GBS_LIST:-auto}"
GPU_CONFIGS="${GPU_CONFIGS:-${ALLOCATED_GPUS}}"
STRATEGIES="${STRATEGIES:-adacpsp flexsp}"

# Runtime knobs.
NUM_ITERS="${NUM_ITERS:-20}"
WARMUP_ITERS="${WARMUP_ITERS:-5}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-90}"
DEFAULT_DP_TYPE="${DEFAULT_DP_TYPE:-zero3}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DATASET="${DATASET:-wikipedia}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
# Whether to mirror rank-0 experiment logs to terminal in real time.
# 1 = print to terminal + file, 0 = file only.
LIVE_LOG_TO_STDOUT="${LIVE_LOG_TO_STDOUT:-1}"

# Paths.
LOCAL_RESULT_ROOT="${LOCAL_RESULT_ROOT:-${SCRIPT_DIR}/results}"
RESULT_ROOT="${RESULT_ROOT:-${LOCAL_RESULT_ROOT}}"
HDFS_RESULT_ROOT="${HDFS_RESULT_ROOT:-hdfs://harunawl/home/byte_data_seed_wl/user/liuqingshuo}"
HDFS_EXPERIMENT_DIR="${HDFS_EXPERIMENT_DIR:-byted_experiments}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_ROOT}/${TIMESTAMP}"
HDFS_RESULT_DIR="${HDFS_RESULT_ROOT%/}/${HDFS_EXPERIMENT_DIR}/${TIMESTAMP}"
DATASET_MOUNT_DIR="${DATASET_MOUNT_DIR:-}"
ANALYSIS_TXT="${RESULT_DIR}/analysis.txt"
HDFS_CMD=""

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

detect_hdfs_cmd() {
    if command -v hdfs >/dev/null 2>&1; then
        HDFS_CMD="hdfs"
    elif command -v /opt/tiger/yarn_deploy/hadoop/bin/hdfs >/dev/null 2>&1; then
        HDFS_CMD="/opt/tiger/yarn_deploy/hadoop/bin/hdfs"
    else
        HDFS_CMD=""
    fi
}

run_hdfs() {
    if [ -z "${HDFS_CMD}" ]; then
        return 1
    fi
    "${HDFS_CMD}" dfs "$@"
}

sync_results_to_hdfs() {
    if [ "${PLATFORM_NODE_RANK}" -ne 0 ]; then
        return 0
    fi

    if [ -z "${HDFS_CMD}" ]; then
        log "WARNING: HDFS client not found. Skip syncing results to ${HDFS_RESULT_DIR}"
        return 0
    fi

    log "Syncing results to HDFS: ${HDFS_RESULT_DIR}"
    if ! run_hdfs -mkdir -p "${HDFS_RESULT_DIR}"; then
        log "WARNING: failed to create HDFS directory ${HDFS_RESULT_DIR}"
        return 0
    fi

    if ! run_hdfs -put -f "${RESULT_DIR}"/* "${HDFS_RESULT_DIR}/"; then
        log "WARNING: failed to upload one or more result files to ${HDFS_RESULT_DIR}"
        return 0
    fi
}

run_analysis() {
    if [ "${PLATFORM_NODE_RANK}" -ne 0 ]; then
        return 0
    fi

    log "Running result analysis..."
    python "${SCRIPT_DIR}/analyze_results.py" "${RESULT_DIR}" --detailed > "${ANALYSIS_TXT}" 2>&1 || {
        log "WARNING: analysis command failed. See ${ANALYSIS_TXT}"
        return 0
    }
    log "Analysis written to ${ANALYSIS_TXT}"
}

calc_profile_end_iter() {
    local warmup=$1
    local measured=$2
    echo $((warmup + measured))
}

setup_cuda_runtime_env() {
    local extra_lib_dirs
    extra_lib_dirs=$(python - <<'PY'
import glob
import os
import site

dirs = []
seen = set()
for base in site.getsitepackages():
    for lib_dir in sorted(glob.glob(os.path.join(base, "nvidia", "*", "lib"))):
        if os.path.isdir(lib_dir) and lib_dir not in seen:
            seen.add(lib_dir)
            dirs.append(lib_dir)
print(":".join(dirs))
PY
)

    if [ -n "${extra_lib_dirs}" ]; then
        export LD_LIBRARY_PATH="${extra_lib_dirs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        log "Appended CUDA runtime libraries: ${extra_lib_dirs}"
    fi
}

setup_dataset_mount() {
    if [ -z "${DATASET_MOUNT_DIR}" ]; then
        return 0
    fi

    if [ ! -d "${DATASET_MOUNT_DIR}" ]; then
        log "ERROR: DATASET_MOUNT_DIR does not exist: ${DATASET_MOUNT_DIR}"
        return 1
    fi

    if [ -L "${PROJECT_DIR}/varlen_datasets" ] || [ ! -e "${PROJECT_DIR}/varlen_datasets" ]; then
        ln -sfn "${DATASET_MOUNT_DIR}" "${PROJECT_DIR}/varlen_datasets"
        log "Linked dataset mount: ${PROJECT_DIR}/varlen_datasets -> ${DATASET_MOUNT_DIR}"
        return 0
    fi

    log "Using existing dataset directory: ${PROJECT_DIR}/varlen_datasets"
}

auto_gbs() {
    # Temporary fixed sweep: always test the same two GBS values
    # regardless of model, GPU count, or sequence length.
    echo "128 256"
}

strategy_to_args() {
    local strategy=$1
    case "${strategy}" in
        adacpsp)
            echo "--use-adaCPSP --use-packing --adaCPSP-attn-types ulysses ring usp"
            ;;
        flexsp)
            echo "--use-adaCPSP --use-packing --adaCPSP-attn-types ulysses"
            ;;
        ring_only)
            echo "--use-adaCPSP --use-packing --adaCPSP-attn-types ring"
            ;;
        ulysses_ring)
            echo "--use-adaCPSP --use-packing --adaCPSP-attn-types ulysses ring"
            ;;
        *)
            log "ERROR: Unknown strategy: ${strategy}"
            return 1
            ;;
    esac
}

validate_model_size() {
    local model=$1
    local meta_config="${PROJECT_DIR}/galvatron/models/varlen_llama_hf/meta_configs/${model}.json"
    [ -f "${meta_config}" ]
}

model_meta_args() {
    local model=$1
    local seqlen=$2
    local meta_config="${PROJECT_DIR}/galvatron/models/varlen_llama_hf/meta_configs/${model}.json"

    python - <<PY
import json

with open("${meta_config}") as f:
    cfg = json.load(f)

parts = [
    f"--vocab_size {cfg['vocab_size']}",
    f"--hidden_size {cfg['dim']}",
    f"--num_hidden_layers {cfg['n_layers']}",
    f"--num_attention_heads {cfg['n_heads']}",
    f"--ffn_hidden_size {int(cfg.get('ffn_dim', cfg['dim'] * 4))}",
    f"--max-position-embeddings ${seqlen}",
]
print(" ".join(parts))
PY
}

resolve_torchrun_shape() {
    local target_gpus=$1
    local active_nproc
    local active_nnodes

    if [ "${target_gpus}" -eq "${ALLOCATED_GPUS}" ]; then
        echo "${PLATFORM_NPROC_PER_NODE} ${PLATFORM_NNODES}"
        return 0
    fi

    if [ "${PLATFORM_NNODES}" -eq 1 ] && [ "${target_gpus}" -le "${PLATFORM_NPROC_PER_NODE}" ]; then
        echo "${target_gpus} 1"
        return 0
    fi

    log "SKIP: target GPU count ${target_gpus} is incompatible with current allocation ${ALLOCATED_GPUS} (${PLATFORM_NNODES} nodes x ${PLATFORM_NPROC_PER_NODE} gpus/node)."
    log "      Submit a separate Merlin Seed job for each multi-node GPU configuration."
    return 1
}

mkdir -p "${RESULT_DIR}"

GPU_CONFIG_COUNT=$(printf "%s\n" "${GPU_CONFIGS}" | awk '{print NF}')
if [ "${GPU_CONFIG_COUNT}" -ne 1 ]; then
    log "ERROR: byted_experiments does not support multiple GPU counts in one run."
    log "       Please submit one Merlin Seed job per GPU configuration."
    log "       Current GPU_CONFIGS='${GPU_CONFIGS}'"
    exit 1
fi

TARGET_GPUS="${GPU_CONFIGS}"

cat > "${RESULT_DIR}/experiment_config.txt" <<EOF
=== AdaCPSP ByteDance Experiment Configuration ===
Timestamp:           ${TIMESTAMP}
Local Result Root:   ${RESULT_ROOT}
Local Result Dir:    ${RESULT_DIR}
HDFS Result Root:    ${HDFS_RESULT_ROOT}
HDFS Experiment Dir: ${HDFS_EXPERIMENT_DIR}
HDFS Result Dir:     ${HDFS_RESULT_DIR}
Models:              ${MODELS}
Seq Lengths (K):     ${SEQ_LENGTHS_K}
GBS:                 ${GBS_LIST}
GPU Configs:         ${GPU_CONFIGS}
Strategies:          ${STRATEGIES}
Learning Rate:       ${LR}
Default DP Type:     ${DEFAULT_DP_TYPE}
Num Workers:         ${NUM_WORKERS}
Num Iters:           ${NUM_ITERS}
Warmup Iters:        ${WARMUP_ITERS}
Timeout (s):         ${TIMEOUT_SECONDS}
Memory Limit:        ${MEMORY_LIMIT_GB} GB
Dataset:             ${DATASET}
Dataset Mount Dir:   ${DATASET_MOUNT_DIR:-<unset>}
Allocated Nodes:     ${PLATFORM_NNODES}
Allocated GPUs/node: ${PLATFORM_NPROC_PER_NODE}
Allocated GPUs:      ${ALLOCATED_GPUS}
Node Rank:           ${PLATFORM_NODE_RANK}
Master:              ${PLATFORM_MASTER_ADDR}:${PLATFORM_MASTER_PORT}
Train Script:        ${TRAIN_SCRIPT}
Extra Train Args:    ${EXTRA_TRAIN_ARGS}
EOF

log "=========================================="
log "AdaCPSP ByteDance experiment runner"
log "=========================================="
log "Result dir: ${RESULT_DIR}"
log "HDFS result dir: ${HDFS_RESULT_DIR}"
log "Allocated nodes: ${PLATFORM_NNODES}"
log "Allocated gpus/node: ${PLATFORM_NPROC_PER_NODE}"
log "Allocated gpus total: ${ALLOCATED_GPUS}"
log "Node rank: ${PLATFORM_NODE_RANK}"
log "Master: ${PLATFORM_MASTER_ADDR}:${PLATFORM_MASTER_PORT}"
log "Target gpus: ${TARGET_GPUS}"
log "=========================================="

setup_cuda_runtime_env
setup_dataset_mount || exit 1
detect_hdfs_cmd

TOTAL=0
PASSED=0
FAILED=0
TIMEOUT_COUNT=0
SKIPPED=0

SUMMARY_CSV="${RESULT_DIR}/summary.csv"
echo "model,ngpus,seqlen_k,gbs,strategy,status,wall_time_s,avg_iter_time_ms,throughput_tokens_per_s,peak_activation_mb,log_file" > "${SUMMARY_CSV}"

SHAPE=$(resolve_torchrun_shape "${TARGET_GPUS}")
if [ $? -ne 0 ]; then
    exit 1
fi

ACTIVE_NPROC_PER_NODE=$(echo "${SHAPE}" | awk '{print $1}')
ACTIVE_NNODES=$(echo "${SHAPE}" | awk '{print $2}')

if [ "${PLATFORM_NODE_RANK}" -ge "${ACTIVE_NNODES}" ]; then
    log "Node rank ${PLATFORM_NODE_RANK} is outside active nodes for ${TARGET_GPUS} GPUs, exiting."
    exit 0
fi

for model in ${MODELS}; do
    if ! validate_model_size "${model}" > /dev/null 2>&1; then
        log "SKIP: model ${model} not found in meta configs"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    for seqlen_k in ${SEQ_LENGTHS_K}; do
        seqlen=$((seqlen_k * 1024))

        if [ "${GBS_LIST}" = "auto" ]; then
            gbs_values=$(auto_gbs "${model}" "${TARGET_GPUS}" "${seqlen_k}")
        else
            gbs_values="${GBS_LIST}"
        fi

        for gbs in ${gbs_values}; do
            for strategy in ${STRATEGIES}; do
                TOTAL=$((TOTAL + 1))

                EXP_NAME="${model}_gpu${TARGET_GPUS}_seq${seqlen_k}k_gbs${gbs}_${strategy}"
                EXP_LOG="${RESULT_DIR}/${EXP_NAME}.log"
                if [ "${PLATFORM_NODE_RANK}" -eq 0 ]; then
                    NODE_LOG="${EXP_LOG}"
                else
                    NODE_LOG="${EXP_LOG}.node${PLATFORM_NODE_RANK}"
                fi

                if [ "${PLATFORM_NODE_RANK}" -eq 0 ]; then
                    log "----------------------------------------"
                    log "[${TOTAL}] Starting: ${EXP_NAME}"
                    log "  model=${model} gpus=${TARGET_GPUS} seq=${seqlen_k}K gbs=${gbs} strategy=${strategy}"
                fi

                STRATEGY_ARGS=$(strategy_to_args "${strategy}")
                if [ $? -ne 0 ]; then
                    SKIPPED=$((SKIPPED + 1))
                    if [ "${PLATFORM_NODE_RANK}" -eq 0 ]; then
                        echo "${model},${TARGET_GPUS},${seqlen_k},${gbs},${strategy},SKIPPED,0,0,0,0,${EXP_LOG}" >> "${SUMMARY_CSV}"
                    fi
                    continue
                fi

                PROFILE_END_ITER=$(calc_profile_end_iter "${WARMUP_ITERS}" "${NUM_ITERS}")
                MODEL_META_ARGS=$(model_meta_args "${model}" "${seqlen}")

                CMD="torchrun \
                    --nproc_per_node=${ACTIVE_NPROC_PER_NODE} \
                    --nnodes=${ACTIVE_NNODES} \
                    --node_rank=${PLATFORM_NODE_RANK} \
                    --master_addr=${PLATFORM_MASTER_ADDR} \
                    --master_port=${PLATFORM_MASTER_PORT} \
                    ${TRAIN_SCRIPT} \
                    --model_size ${model} \
                    ${MODEL_META_ARGS} \
                    --set_seqlen_manually 1 \
                    -s ${seqlen} \
                    --global_train_batch_size ${gbs} \
                    --train-iters ${PROFILE_END_ITER} \
                    --epochs ${EPOCHS} \
                    --lr ${LR} \
                    --num-workers ${NUM_WORKERS} \
                    --pp_deg 1 \
                    --global_tp_deg 1 \
                    --global_cp_deg 1 \
                    --default_dp_type ${DEFAULT_DP_TYPE} \
                    --mixed_precision bf16 \
                    --use-flash-attn \
                    --dataset ${DATASET} \
                    --initialize_on_meta 1 \
                    --memory-limit-gb ${MEMORY_LIMIT_GB} \
                    --profile 1 \
                    --profile_start_iter ${WARMUP_ITERS} \
                    --profile_end_iter ${PROFILE_END_ITER} \
                    --exit_after_profiling 1 \
                    ${STRATEGY_ARGS} \
                    ${EXTRA_TRAIN_ARGS}"

                if [ "${PLATFORM_NODE_RANK}" -eq 0 ]; then
                    {
                        echo "=== Command ==="
                        echo "${CMD}"
                        echo "=== Start: $(date) ==="
                        echo
                    } > "${NODE_LOG}"
                fi

                START_TIME=$(date +%s)
                if [ "${PLATFORM_NODE_RANK}" -eq 0 ] && [ "${LIVE_LOG_TO_STDOUT}" = "1" ]; then
                    timeout "${TIMEOUT_SECONDS}" bash -c "${CMD}" 2>&1 | tee -a "${NODE_LOG}"
                    EXIT_CODE=${PIPESTATUS[0]}
                else
                    timeout "${TIMEOUT_SECONDS}" bash -c "${CMD}" >> "${NODE_LOG}" 2>&1
                    EXIT_CODE=$?
                fi
                END_TIME=$(date +%s)
                WALL_TIME=$((END_TIME - START_TIME))

                if [ "${PLATFORM_NODE_RANK}" -eq 0 ]; then
                    {
                        echo
                        echo "=== End: $(date) ==="
                        echo "=== Exit Code: ${EXIT_CODE} ==="
                        echo "=== Wall Time: ${WALL_TIME}s ==="
                    } >> "${NODE_LOG}"

                    if [ ${EXIT_CODE} -eq 0 ]; then
                        STATUS="PASS"
                        PASSED=$((PASSED + 1))
                        log "  PASS (${WALL_TIME}s)"
                    elif [ ${EXIT_CODE} -eq 124 ]; then
                        STATUS="TIMEOUT"
                        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
                        log "  TIMEOUT after ${TIMEOUT_SECONDS}s"
                    else
                        STATUS="FAIL(${EXIT_CODE})"
                        FAILED=$((FAILED + 1))
                        if grep -q "OutOfMemoryError\|CUDA out of memory\|OOM" "${EXP_LOG}" 2>/dev/null; then
                            STATUS="OOM"
                            log "  OOM (${WALL_TIME}s)"
                        else
                            log "  FAIL exit=${EXIT_CODE} (${WALL_TIME}s)"
                        fi
                    fi

                    AVG_ITER_S=$(grep -oP 'Average iteration time is:\s*\K[\d.]+' "${EXP_LOG}" 2>/dev/null | tail -1 || echo "0")
                    AVG_ITER_MS=$(awk -v s="${AVG_ITER_S}" 'BEGIN { printf "%.3f", s * 1000 }')
                    THROUGHPUT=$(grep -oP '[Tt]hroughput.*?:\s*\K[\d.]+' "${EXP_LOG}" 2>/dev/null | tail -1 || echo "0")
                    PEAK_MEM=$(grep -oP 'peak_activation:\s*\K[\d.]+' "${EXP_LOG}" 2>/dev/null | tail -1 || echo "0")

                    echo "${model},${TARGET_GPUS},${seqlen_k},${gbs},${strategy},${STATUS},${WALL_TIME},${AVG_ITER_MS},${THROUGHPUT},${PEAK_MEM},${EXP_LOG}" >> "${SUMMARY_CSV}"
                fi

                sleep 5
            done
        done
    done
done

if [ "${PLATFORM_NODE_RANK}" -eq 0 ]; then
    run_analysis
    sync_results_to_hdfs
    log ""
    log "=========================================="
    log "Experiment sweep finished"
    log "=========================================="
    log "Total: ${TOTAL}"
    log "Passed: ${PASSED}"
    log "Failed: ${FAILED}"
    log "Timeout: ${TIMEOUT_COUNT}"
    log "Skipped: ${SKIPPED}"
    log "Result dir: ${RESULT_DIR}"
    log "Summary CSV: ${SUMMARY_CSV}"
    log "Analysis TXT: ${ANALYSIS_TXT}"
    log "Analyze command: python byted_experiments/analyze_results.py ${RESULT_DIR} --detailed"
    log "HDFS result dir: ${HDFS_RESULT_DIR}"
    log "=========================================="
    column -t -s',' "${SUMMARY_CSV}" 2>/dev/null || cat "${SUMMARY_CSV}"
fi
