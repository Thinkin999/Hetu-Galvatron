#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/00_common.sh"
write_run_metadata
load_model_meta >/dev/null

ATTN_ALIGN_DIR="${SUMMARY_DIR}/attention_wrapper"
ATTN_ALIGN_LOG_BASE="${LOG_DIR}/06_attention_wrapper_${MODEL_NAME}"
if [ "${NODE_RANK}" = "0" ]; then
    ATTN_ALIGN_LOG="${ATTN_ALIGN_LOG_BASE}.log"
else
    ATTN_ALIGN_LOG="${ATTN_ALIGN_LOG_BASE}.node${NODE_RANK}.log"
fi

SEQ_LENGTHS="${ATTN_ALIGN_SEQ_LENGTHS:-8192 16384 32768}"
NUM_SEQS="${ATTN_ALIGN_NUM_SEQS:-${GLOBAL_BATCH_SIZE}}"
BENCH_CASES="${ATTN_ALIGN_CASES:-local:1 ulysses:16 ring:16 usp:2x8 usp:4x4 usp:8x2}"
GROUP_TOPOLOGY="${ATTN_ALIGN_GROUP_TOPOLOGY:-consecutive}"
PLACEMENT="${ATTN_ALIGN_PLACEMENT:-context_first}"
BENCH_N_HEADS="${ATTN_ALIGN_N_HEADS:-${N_HEADS}}"
BENCH_N_KV_HEADS="${ATTN_ALIGN_N_KV_HEADS:-${N_KV_HEADS}}"
ULYSSES_N_HEADS="${ATTN_ALIGN_ULYSSES_N_HEADS:-}"
ULYSSES_N_KV_HEADS="${ATTN_ALIGN_ULYSSES_N_KV_HEADS:-}"

mkdir -p "${ATTN_ALIGN_DIR}"

if [ "${NODE_RANK}" = "0" ]; then
    {
        echo "=== Attention wrapper alignment ==="
        echo "MODEL_NAME=${MODEL_NAME}"
        echo "WORLD_SIZE=${WORLD_SIZE}"
        echo "SEQ_LENGTHS=${SEQ_LENGTHS}"
        echo "NUM_SEQS=${NUM_SEQS}"
        echo "BENCH_CASES=${BENCH_CASES}"
        echo "GROUP_TOPOLOGY=${GROUP_TOPOLOGY}"
        echo "PLACEMENT=${PLACEMENT}"
        echo "BENCH_N_HEADS=${BENCH_N_HEADS}"
        echo "BENCH_N_KV_HEADS=${BENCH_N_KV_HEADS}"
        echo "ULYSSES_N_HEADS=${ULYSSES_N_HEADS:-<default>}"
        echo "ULYSSES_N_KV_HEADS=${ULYSSES_N_KV_HEADS:-<default>}"
        echo "=== Start: $(date) ==="
        echo
    } > "${ATTN_ALIGN_LOG}"
fi

parse_case() {
    local case_spec=$1
    local kind=${case_spec%%:*}
    local size=${case_spec#*:}
    local attn_type parallel sp cp case_name

    if [ "${kind}" = "local" ]; then
        attn_type="local"
        parallel=1
        sp=1
        cp=1
        case_name="local_p1"
    elif [ "${kind}" = "ulysses" ]; then
        attn_type="ulysses"
        parallel="${size}"
        sp="${size}"
        cp=1
        case_name="ulysses_p${parallel}"
    elif [ "${kind}" = "ring" ]; then
        attn_type="ring"
        parallel="${size}"
        sp=1
        cp="${size}"
        case_name="ring_p${parallel}"
    elif [ "${kind}" = "usp" ]; then
        attn_type="usp"
        sp="${size%x*}"
        cp="${size#*x}"
        parallel=$((sp * cp))
        case_name="usp_s${sp}_c${cp}_${PLACEMENT}"
    else
        echo "ERROR: unknown case spec ${case_spec}" >&2
        return 1
    fi

    echo "${case_name}|${attn_type}|${parallel}|${sp}|${cp}"
}

for case_spec in ${BENCH_CASES}; do
    IFS='|' read -r case_name attn_type parallel_size sp_size cp_size < <(parse_case "${case_spec}")
    if [ "${parallel_size}" -gt "${WORLD_SIZE}" ]; then
        log "Skipping ${case_spec}: parallel_size=${parallel_size} > WORLD_SIZE=${WORLD_SIZE}"
        continue
    fi

    for seq_len in ${SEQ_LENGTHS}; do
        case_n_heads="${BENCH_N_HEADS}"
        case_n_kv_heads="${BENCH_N_KV_HEADS}"
        case_name_effective="${case_name}"
        if [ "${attn_type}" = "ulysses" ] && [ -n "${ULYSSES_N_HEADS}" ]; then
            case_n_heads="${ULYSSES_N_HEADS}"
            case_name_effective="${case_name_effective}_h${case_n_heads}"
        fi
        if [ "${attn_type}" = "ulysses" ] && [ -n "${ULYSSES_N_KV_HEADS}" ]; then
            case_n_kv_heads="${ULYSSES_N_KV_HEADS}"
            case_name_effective="${case_name_effective}_kv${case_n_kv_heads}"
        fi
        output_json="${ATTN_ALIGN_DIR}/${case_name}_seq${seq_len}.json"
        if [ "${case_name_effective}" != "${case_name}" ]; then
            output_json="${ATTN_ALIGN_DIR}/${case_name_effective}_seq${seq_len}.json"
        fi
        cmd="$(torchrun_prefix) \"${ATTN_BENCHMARK_SCRIPT}\" \
          --output-json \"${output_json}\" \
          --case-name \"${case_name_effective}_seq${seq_len}\" \
          --attn-type \"${attn_type}\" \
          --parallel-size ${parallel_size} \
          --sp-size ${sp_size} \
          --cp-size ${cp_size} \
          --group-topology \"${GROUP_TOPOLOGY}\" \
          --placement \"${PLACEMENT}\" \
          --seq-len ${seq_len} \
          --num-seqs ${NUM_SEQS} \
          --n-heads ${case_n_heads} \
          --n-kv-heads ${case_n_kv_heads} \
          --head-dim ${HEAD_DIM} \
          --warmup ${WARMUP_ITERS} \
          --iters ${MEASURE_ITERS}"

        if [ "${NODE_RANK}" = "0" ]; then
            {
                echo "=== Command (${case_name}, seq=${seq_len}) ==="
                echo "${cmd}"
                echo
            } >> "${ATTN_ALIGN_LOG}"
        fi

        bash -c "${cmd}" 2>&1 | tee -a "${ATTN_ALIGN_LOG}"
    done
done

if [ "${NODE_RANK}" = "0" ]; then
    python3 "${MODEL_DIR}/analyze_attention_wrapper_alignment.py" \
      --benchmark-json "${ATTN_ALIGN_DIR}" \
      --configs-dir "${CONFIGS_DIR}" \
      --output-dir "${ATTN_ALIGN_DIR}/alignment" \
      --gpus-per-node "${NPROC_PER_NODE}" \
      2>&1 | tee -a "${ATTN_ALIGN_LOG}"
fi
