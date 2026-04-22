#!/bin/bash
set -euo pipefail

ALIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${ALIGN_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MODEL_DIR}/../../.." && pwd)"
SITE_PACKAGE_DIR="${REPO_ROOT}/galvatron/site_package"

ROOT="${ROOT:-/mnt/bn/wyj-data0-hl/lqs}"
ADACPSP_ENV="${ADACPSP_ENV:-${ROOT}/envs/galvatron-adacpsp-py39-torch21-cu121}"
CONDA_SH="${CONDA_SH:-${ROOT}/tools/miniforge3/etc/profile.d/conda.sh}"

if [ -f "${CONDA_SH}" ] && [ -d "${ADACPSP_ENV}" ]; then
    # Auto-activate the expected benchmark env unless we are already inside it.
    if [ "${CONDA_PREFIX:-}" != "${ADACPSP_ENV}" ]; then
        # shellcheck disable=SC1090
        source "${CONDA_SH}"
        conda activate "${ADACPSP_ENV}"
    fi
fi

export PYTHONPATH="${SITE_PACKAGE_DIR}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

require_python_modules() {
    python - <<'PY'
required = ["scipy", "torch", "flash_attn"]
missing = []
for mod in required:
    try:
        __import__(mod)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{mod}: {exc!r}")
if missing:
    raise SystemExit("Missing Python modules for align_costmodel:\n" + "\n".join(missing))
PY
}

require_python_modules

ATTN_PROFILE_SCRIPT="${MODEL_DIR}/profile_and_validate.py"
COMM_PROFILE_SCRIPT="${MODEL_DIR}/profile_comm.py"
ATTN_BENCHMARK_SCRIPT="${ALIGN_DIR}/03_benchmark_attention.py"
CONFIGS_DIR="${MODEL_DIR}/configs"
META_CONFIG_DIR="${MODEL_DIR}/meta_configs"

MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"
HEAD_DIM="${HEAD_DIM:-128}"

ALIGN_SEQ_LENGTHS="${ALIGN_SEQ_LENGTHS:-2048 4096 8192 16384}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
WARMUP_ITERS="${WARMUP_ITERS:-5}"
MEASURE_ITERS="${MEASURE_ITERS:-20}"
ATTN_STEP="${ATTN_STEP:-128}"
ATTN_MAX="${ATTN_MAX:-16384}"
ACROSS_GROUP_AGG="${ACROSS_GROUP_AGG:-p90}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
BENCH_GROUP_SIZES="${BENCH_GROUP_SIZES:-auto}"

NPROC_PER_NODE="${ARNOLD_WORKER_GPU:-${NPROC_PER_NODE:-8}}"
NNODES="${ARNOLD_WORKER_NUM:-${NNODES:-2}}"
NODE_RANK="${ARNOLD_ID:-${NODE_RANK:-0}}"
MASTER_ADDR="${METIS_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${METIS_WORKER_0_PORT:-${MASTER_PORT:-29500}}"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

ALIGN_RUN_ID="${ALIGN_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${ALIGN_DIR}/results/${ALIGN_RUN_ID}"
LOG_DIR="${RESULT_ROOT}/logs"
SUMMARY_DIR="${RESULT_ROOT}/summary"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}" "${SUMMARY_DIR}" "${CONFIGS_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

meta_config_path() {
    echo "${META_CONFIG_DIR}/${MODEL_NAME}.json"
}

load_model_meta() {
    local meta_file
    meta_file="$(meta_config_path)"
    if [ ! -f "${meta_file}" ]; then
        echo "ERROR: meta config not found: ${meta_file}" >&2
        return 1
    fi

    eval "$(
        python3 - <<PY
import json
with open("${meta_file}") as f:
    cfg = json.load(f)
print(f'export HIDDEN={cfg["dim"]}')
print(f'export N_HEADS={cfg["n_heads"]}')
print(f'export N_KV_HEADS={cfg["n_kv_heads"]}')
print(f'export N_LAYERS={cfg["n_layers"]}')
print(f'export FFN_HIDDEN_SIZE={int(cfg.get("ffn_dim", cfg["dim"] * 4))}')
print(f'export VOCAB_SIZE={cfg["vocab_size"]}')
print(f'export MAX_POSITIONS={cfg.get("n_positions", 131072)}')
PY
    )"
}

model_runtime_args() {
    load_model_meta >/dev/null
    echo "--model_name ${MODEL_NAME} --hidden_size ${HIDDEN} --n-heads ${N_HEADS} --n-kv-heads ${N_KV_HEADS} --num-layers ${N_LAYERS} --head-dim ${HEAD_DIM}"
}

torchrun_prefix() {
    echo "torchrun --nnodes ${NNODES} --nproc_per_node ${NPROC_PER_NODE} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank ${NODE_RANK}"
}

latest_attention_profile() {
    python3 - <<PY
import glob, json, os
paths = sorted(glob.glob(os.path.join("${CONFIGS_DIR}", "profile_validate_*.json")), reverse=True)
for path in paths:
    try:
        with open(path) as f:
            data = json.load(f)
        if "attention" in data and "segments" in data.get("attention", {}):
            print(path)
            break
    except Exception:
        pass
PY
}

latest_comm_profile() {
    python3 - <<PY
import glob, json, os
paths = sorted(glob.glob(os.path.join("${CONFIGS_DIR}", "comm_profile_*.json")), reverse=True)
for path in paths:
    try:
        with open(path) as f:
            data = json.load(f)
        if "alltoall" in data and "p2p_ring" in data:
            print(path)
            break
    except Exception:
        pass
PY
}

_resolved_group_sizes() {
    python3 - <<PY
world = ${WORLD_SIZE}
raw = "${BENCH_GROUP_SIZES}".strip()
if raw and raw != "auto":
    print(raw)
else:
    vals = []
    p = 1
    while p <= world:
        vals.append(str(p))
        p *= 2
    print(" ".join(vals))
PY
}

benchmark_cases_for_world_size() {
    python3 - <<PY
world = ${WORLD_SIZE}
sizes = [int(x) for x in """$(_resolved_group_sizes)""".split() if int(x) <= world]
cases = []
for p in sizes:
    topologies = ["consecutive"] if p == 1 or p == world else ["consecutive", "strided"]
    for topo in topologies:
        if p == 1:
            cases.append((f"local_p1_{topo}", "local", 1, 1, 1, topo, "context_first"))
            continue
        cases.append((f"ulysses_p{p}_{topo}", "ulysses", p, p, 1, topo, "context_first"))
        cases.append((f"ring_p{p}_{topo}", "ring", p, 1, p, topo, "context_first"))
        sp = 2
        while sp < p:
            if p % sp == 0:
                cp = p // sp
                if cp >= 2:
                    cases.append((f"usp_p{p}_s{sp}_c{cp}_{topo}_cf", "usp", p, sp, cp, topo, "context_first"))
                    cases.append((f"usp_p{p}_s{sp}_c{cp}_{topo}_hf", "usp", p, sp, cp, topo, "head_first"))
            sp *= 2

for item in cases:
    print("|".join(str(x) for x in item))
PY
}

write_run_metadata() {
    load_model_meta >/dev/null
    cat > "${SUMMARY_DIR}/run_metadata.txt" <<EOF
ALIGN_RUN_ID=${ALIGN_RUN_ID}
MODEL_NAME=${MODEL_NAME}
HIDDEN=${HIDDEN}
N_HEADS=${N_HEADS}
N_KV_HEADS=${N_KV_HEADS}
N_LAYERS=${N_LAYERS}
HEAD_DIM=${HEAD_DIM}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
ALIGN_SEQ_LENGTHS=${ALIGN_SEQ_LENGTHS}
BENCH_GROUP_SIZES=$(_resolved_group_sizes)
NNODES=${NNODES}
NPROC_PER_NODE=${NPROC_PER_NODE}
WORLD_SIZE=${WORLD_SIZE}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
NODE_RANK=${NODE_RANK}
WARMUP_ITERS=${WARMUP_ITERS}
MEASURE_ITERS=${MEASURE_ITERS}
ACROSS_GROUP_AGG=${ACROSS_GROUP_AGG}
EOF
}
