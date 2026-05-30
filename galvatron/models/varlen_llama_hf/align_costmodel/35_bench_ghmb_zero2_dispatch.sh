#!/usr/bin/env bash
# Dispatch ghmb benchmark with ZeRO-2 (NOT ZeRO-3 as 26_*).
#
# Identical to 26_bench_github_multimb_dispatch.sh except invokes
# 35_bench_ghmb_zero2_worker.sh which sets --sdp 0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GHMB_RUN_ID="${GHMB_RUN_ID:-ghmb_zero2_$(date +%Y%m%d_%H%M%S)}"
# Default cells: same as 26_* (full sweep)
export GHMB_CELLS="${GHMB_CELLS:-\
ulysses8:1 ulysses8:8 \
ring8:1 ring8:8 \
usp2x4:1 usp2x4:8 \
adacpsp:auto\
}"
export GHMB_GBS="${GHMB_GBS:-16}"
export GHMB_SEQ_LENGTH="${GHMB_SEQ_LENGTH:-65536}"
export GHMB_TRAIN_ITERS="${GHMB_TRAIN_ITERS:-22}"
export GHMB_PROFILE_START_ITER="${GHMB_PROFILE_START_ITER:-5}"
export GHMB_PROFILE_END_ITER="${GHMB_PROFILE_END_ITER:-21}"
export GHMB_PROFILE_RANKS="${GHMB_PROFILE_RANKS:-0 8 15}"

export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-10.122.244.241}"
export MASTER_PORT="${MASTER_PORT:-40067}"
export MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"

WORKER1_HOST="${WORKER1_HOST:-worker-1}"
DISPATCH_TIMEOUT="${DISPATCH_TIMEOUT:-7200}"

RUN_DIR="${SCRIPT_DIR}/results/${GHMB_RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_DIR="${RUN_DIR}/end2end"
mkdir -p "${LOG_DIR}" "${PROFILE_DIR}"
W0_LOG="${LOG_DIR}/ghmb_node0.log"
W1_LOG="${LOG_DIR}/ghmb_node1.log"

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

ENV_EXPORTS=$(cat <<EOF
export GHMB_RUN_ID='${GHMB_RUN_ID}'
export GHMB_CELLS='${GHMB_CELLS}'
export GHMB_GBS='${GHMB_GBS}'
export GHMB_SEQ_LENGTH='${GHMB_SEQ_LENGTH}'
export GHMB_TRAIN_ITERS='${GHMB_TRAIN_ITERS}'
export GHMB_PROFILE_START_ITER='${GHMB_PROFILE_START_ITER}'
export GHMB_PROFILE_END_ITER='${GHMB_PROFILE_END_ITER}'
export GHMB_PROFILE_RANKS='${GHMB_PROFILE_RANKS}'
export NNODES='${NNODES}'
export NPROC_PER_NODE='${NPROC_PER_NODE}'
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT='${MASTER_PORT}'
export MODEL_NAME='${MODEL_NAME}'
export PROFILE_DIR='${PROFILE_DIR}'
EOF
)

echo "==== Dispatching ghmb ZeRO-2 benchmark ${GHMB_RUN_ID} ===="
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}  NNODES=${NNODES}  GPUS/NODE=${NPROC_PER_NODE}"
echo "MODEL=${MODEL_NAME}  GBS=${GHMB_GBS}  seq_length=${GHMB_SEQ_LENGTH}"
echo "DP type: ZeRO-2 (--sdp 0 --default_dp_type zero2)"
echo "cells (<config>:<chunks>) ="
for c in ${GHMB_CELLS}; do echo "    ${c}"; done
echo "train_iters=${GHMB_TRAIN_ITERS}  profile=[${GHMB_PROFILE_START_ITER},${GHMB_PROFILE_END_ITER})"
echo "log_dir=${LOG_DIR}"
echo "profile_dir=${PROFILE_DIR}"

stop_gpu_busy
cleanup_zombies

ssh -o ConnectTimeout=10 "${WORKER1_HOST}" "
${ENV_EXPORTS}
export NODE_RANK=1
mkdir -p ${LOG_DIR} ${PROFILE_DIR}
nohup setsid bash '${SCRIPT_DIR}/35_bench_ghmb_zero2_worker.sh' \
    > '${W1_LOG}' 2>&1 < /dev/null &
echo \$! > '${LOG_DIR}/w1.pid'
"
echo "[dispatch] worker-1 launched (detached)"

eval "${ENV_EXPORTS}"
export NODE_RANK=0
nohup setsid bash "${SCRIPT_DIR}/35_bench_ghmb_zero2_worker.sh" \
    > "${W0_LOG}" 2>&1 < /dev/null &
W0_PID=$!
echo "${W0_PID}" > "${LOG_DIR}/w0.pid"
echo "[dispatch] worker-0 launched pid=${W0_PID} (detached)"

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
tail -60 "${W0_LOG}"
echo "----- w1.log tail -----"
ssh "${WORKER1_HOST}" "tail -60 ${W1_LOG}" || true

echo ""
echo "==== JSONL files produced ===="
find "${PROFILE_DIR}" -name "rank*.jsonl" 2>/dev/null | sort | head -20
