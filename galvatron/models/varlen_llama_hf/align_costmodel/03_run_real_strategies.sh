#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/00_common.sh"
write_run_metadata
load_model_meta >/dev/null

run_one_case() {
    local case_name=$1
    local seq_len=$2
    local attn_type=$3
    local parallel_size=$4
    local sp_size=$5
    local cp_size=$6
    local group_topology=$7
    local placement=$8

    local log_base="${LOG_DIR}/03_bench_${case_name}_s${seq_len}_gbs${GLOBAL_BATCH_SIZE}"
    local node_log="${log_base}.node${NODE_RANK}.log"
    local output_json="${SUMMARY_DIR}/bench_${case_name}_s${seq_len}.json"
    local cmd

    cmd="$(torchrun_prefix) \"${ATTN_BENCHMARK_SCRIPT}\" \
      --output-json \"${output_json}\" \
      --case-name \"${case_name}\" \
      --attn-type \"${attn_type}\" \
      --parallel-size ${parallel_size} \
      --sp-size ${sp_size} \
      --cp-size ${cp_size} \
      --group-topology ${group_topology} \
      --placement ${placement} \
      --seq-len ${seq_len} \
      --num-seqs ${GLOBAL_BATCH_SIZE} \
      --n-heads ${N_HEADS} \
      --n-kv-heads ${N_KV_HEADS} \
      --head-dim ${HEAD_DIM} \
      --warmup ${WARMUP_ITERS} \
      --iters ${MEASURE_ITERS}"

    if [ "${NODE_RANK}" = "0" ]; then
        {
            echo "=== Command ==="
            echo "${cmd}"
            echo "=== Start: $(date) ==="
            echo
        } > "${node_log}"
    fi

    local exit_code=0
    if [ "${TIMEOUT_SECONDS}" -gt 0 ]; then
        timeout "${TIMEOUT_SECONDS}" bash -c "${cmd}" 2>&1 | tee -a "${node_log}"
        exit_code=${PIPESTATUS[0]}
    else
        bash -c "${cmd}" 2>&1 | tee -a "${node_log}"
        exit_code=${PIPESTATUS[0]}
    fi

    if [ "${NODE_RANK}" = "0" ]; then
        if [ "${exit_code}" -ne 0 ]; then
            log "Benchmark failed for ${case_name} seq=${seq_len} exit=${exit_code}"
        else
            log "Benchmark completed for ${case_name} seq=${seq_len} -> ${output_json}"
        fi
    fi
}

while IFS='|' read -r case_name attn_type parallel_size sp_size cp_size group_topology placement; do
    [ -z "${case_name}" ] && continue
    for seq_len in ${ALIGN_SEQ_LENGTHS}; do
        run_one_case "${case_name}" "${seq_len}" "${attn_type}" "${parallel_size}" "${sp_size}" "${cp_size}" "${group_topology}" "${placement}"
    done
done < <(benchmark_cases_for_world_size)
