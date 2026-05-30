#!/usr/bin/env bash
# Dispatch a *b-decomposition* benchmark sweep.
#
# Goal: separate the existing per-group constant `b(sp)` (fit at chunks=1) into
#   b(sp)  =  b_microbatch(sp)  +  b_step
# by varying the number of sequential microbatches per step while keeping each
# microbatch identical (`fix_length` dataset, fixed sp, fixed seq_length).
#
# For ulysses:K with world_size W the number of forced groups per microbatch is
# W/K. We pick `global_train_batch_size = (W/K) * chunks` so that each forced
# group sees exactly one sequence per microbatch, keeping tokens/GPU constant
# across the sweep. End-to-end profile JSONL is captured per cell and analysed
# by 25_fit_b_decomp.py.
#
# Mirrors 22_bench_residual_dispatch.sh in setsid/nohup/gpu_busy lifecycle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- run identity ----------
export BDEC_RUN_ID="${BDEC_RUN_ID:-bdec_$(date +%Y%m%d_%H%M%S)}"
# Each cell is "sp:seq:chunks". seq is the fix_length sequence length;
# chunks = number of sequential microbatches per step.
# sp=1 ulysses:1 (16 forced groups), sp=8 ulysses:8 (2 forced groups).
# Two sps let us check that b_step is sp-independent and b_microbatch(sp) decomposes cleanly.
export BDEC_CELLS="${BDEC_CELLS:-\
1:8192:1 1:8192:2 1:8192:4 1:8192:8 \
8:8192:1 8:8192:2 8:8192:4 8:8192:8\
}"
export BDEC_TRAIN_ITERS="${BDEC_TRAIN_ITERS:-22}"
export BDEC_PROFILE_START_ITER="${BDEC_PROFILE_START_ITER:-5}"
export BDEC_PROFILE_END_ITER="${BDEC_PROFILE_END_ITER:-21}"
# Which ranks write end2end JSONL.
export BDEC_PROFILE_RANKS="${BDEC_PROFILE_RANKS:-0 8 15}"

# ---------- cluster identity ----------
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-10.122.244.241}"
export MASTER_PORT="${MASTER_PORT:-40063}"
export MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"

WORKER1_HOST="${WORKER1_HOST:-worker-1}"
DISPATCH_TIMEOUT="${DISPATCH_TIMEOUT:-7200}"

RUN_DIR="${SCRIPT_DIR}/results/${BDEC_RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_DIR="${RUN_DIR}/end2end"
mkdir -p "${LOG_DIR}" "${PROFILE_DIR}"
W0_LOG="${LOG_DIR}/bdec_node0.log"
W1_LOG="${LOG_DIR}/bdec_node1.log"
W0_PIDFILE="${LOG_DIR}/w0.pid"

# ---------- gpu_busy lifecycle helpers (same as 13_/20_/22_) ----------
GPU_BUSY_PY="/mnt/bn/wyj-data0-hl/lqs/gpu_busy.py"
GPU_BUSY_PYBIN="/mnt/bn/wyj-data0-hl/lqs/envs/galvatron-adacpsp-py39-torch21-cu121/bin/python"

stop_gpu_busy() {
    echo "[dispatch] stopping gpu_busy on both workers..."
    set +e
    tmux kill-session -t gpu_job 2>/dev/null
    ssh "${WORKER1_HOST}" 'tmux kill-session -t gpu_job 2>/dev/null; true'
    pkill -9 -f 'gpu_busy.py' 2>/dev/null
    ssh "${WORKER1_HOST}" 'pkill -9 -f "gpu_busy.py" 2>/dev/null; true'
    set -e
    return 0
}
start_gpu_busy() {
    echo "[dispatch] (re)starting gpu_busy on both workers..."
    set +e
    tmux new-session -d -s gpu_job \
        "${GPU_BUSY_PYBIN} ${GPU_BUSY_PY} 2>&1 | tee /tmp/gpu_busy_w0.log"
    ssh "${WORKER1_HOST}" \
        "tmux new-session -d -s gpu_job '${GPU_BUSY_PYBIN} ${GPU_BUSY_PY} 2>&1 | tee /tmp/gpu_busy_w1.log'"
    set -e
    return 0
}

cleanup_zombies() {
    set +e
    pkill -9 -f 'train_dist_adacpsp.py' 2>/dev/null
    pkill -9 -f "torchrun.*${MASTER_PORT}" 2>/dev/null
    ssh "${WORKER1_HOST}" "pkill -9 -f 'train_dist_adacpsp.py' 2>/dev/null; \
                          pkill -9 -f 'torchrun.*${MASTER_PORT}' 2>/dev/null; true"
    set -e
    return 0
}

on_exit() {
    local rc=$?
    echo "[dispatch] exit handler, rc=${rc}"
    start_gpu_busy
    exit "${rc}"
}
trap on_exit EXIT

# ---------- env forwarded ----------
ENV_EXPORTS=$(cat <<EOF
export BDEC_RUN_ID='${BDEC_RUN_ID}'
export BDEC_CELLS='${BDEC_CELLS}'
export BDEC_TRAIN_ITERS='${BDEC_TRAIN_ITERS}'
export BDEC_PROFILE_START_ITER='${BDEC_PROFILE_START_ITER}'
export BDEC_PROFILE_END_ITER='${BDEC_PROFILE_END_ITER}'
export BDEC_PROFILE_RANKS='${BDEC_PROFILE_RANKS}'
export NNODES='${NNODES}'
export NPROC_PER_NODE='${NPROC_PER_NODE}'
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT='${MASTER_PORT}'
export MODEL_NAME='${MODEL_NAME}'
export PROFILE_DIR='${PROFILE_DIR}'
EOF
)

echo "==== Dispatching b-decomposition benchmark ${BDEC_RUN_ID} ===="
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}  NNODES=${NNODES}  GPUS/NODE=${NPROC_PER_NODE}"
echo "MODEL=${MODEL_NAME}"
echo "cells (sp:seq:chunks) ="
for c in ${BDEC_CELLS}; do echo "    ${c}"; done
echo "train_iters=${BDEC_TRAIN_ITERS}  profile=[${BDEC_PROFILE_START_ITER},${BDEC_PROFILE_END_ITER})"
echo "log_dir=${LOG_DIR}"
echo "profile_dir=${PROFILE_DIR}"

stop_gpu_busy
cleanup_zombies

# ---------- launch worker-1 detached over ssh ----------
ssh -o ConnectTimeout=10 "${WORKER1_HOST}" "
${ENV_EXPORTS}
export NODE_RANK=1
mkdir -p ${LOG_DIR} ${PROFILE_DIR}
nohup setsid bash '${SCRIPT_DIR}/24_bench_b_decomp_worker.sh' \
    > '${W1_LOG}' 2>&1 < /dev/null &
echo \$! > '${LOG_DIR}/w1.pid'
"
echo "[dispatch] worker-1 launched (detached)"

# ---------- launch worker-0 detached locally ----------
eval "${ENV_EXPORTS}"
export NODE_RANK=0
nohup setsid bash "${SCRIPT_DIR}/24_bench_b_decomp_worker.sh" \
    > "${W0_LOG}" 2>&1 < /dev/null &
W0_PID=$!
echo "${W0_PID}" > "${W0_PIDFILE}"
echo "[dispatch] worker-0 launched pid=${W0_PID} (detached)"

# ---------- poll worker-0 with progress ticker ----------
SECONDS=0
LAST_LOG_SIZE=0
while kill -0 "${W0_PID}" 2>/dev/null; do
    if (( SECONDS > DISPATCH_TIMEOUT )); then
        echo "[dispatch] WARN: worker-0 still running after ${DISPATCH_TIMEOUT}s"
        echo "[dispatch] not killing it - tail its log at ${W0_LOG}"
        exit 2
    fi
    sleep 15
    cur_size=$(stat -c %s "${W0_LOG}" 2>/dev/null || echo 0)
    if [ "${cur_size}" != "${LAST_LOG_SIZE}" ]; then
        echo "[dispatch] t=${SECONDS}s w0.log size=${cur_size}B"
        LAST_LOG_SIZE="${cur_size}"
    fi
done

echo "[dispatch] worker-0 finished after ${SECONDS}s"
echo "----- w0.log tail -----"
tail -40 "${W0_LOG}"
echo "----- w1.log tail -----"
ssh "${WORKER1_HOST}" "tail -40 ${W1_LOG}" || true

echo ""
echo "==== JSONL files produced ===="
find "${PROFILE_DIR}" -name "rank*.jsonl" 2>/dev/null | sort | head -20
