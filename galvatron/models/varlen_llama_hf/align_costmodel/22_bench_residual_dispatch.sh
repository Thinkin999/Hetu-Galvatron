#!/usr/bin/env bash
# Dispatch a residual benchmark sweep: for each (sp, seq_length, ckpt) cell,
# launch a 2-node 16-GPU `train_dist_adacpsp` run with `--adaCPSP-end2end-profile`
# so we capture per-step `forward_backward_ms` (and other stages) as JSONL.
#
# The benchmark assumes the residual cost model
#   T_per_group(seqlens, strat) = T_attention(seqlens, strat) + a * tokens/GPU + b(sp)
# and is meant to be analysed by 23_fit_residual.py.
#
# Mirrors 20_trace_fsdp_step_dispatch.sh in setsid/nohup/gpu_busy lifecycle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- run identity ----------
export RESID_RUN_ID="${RESID_RUN_ID:-resid_$(date +%Y%m%d_%H%M%S)}"
# Each cell is "sp:seq:ckpt", space separated. Defaults cover the realistic
# operating range for solver decisions (1k–32k tokens/GPU).
export RESID_CELLS="${RESID_CELLS:-\
1:1024:0 1:2048:0 1:4096:0 1:6144:0 1:8192:0 \
2:1024:0 2:2048:0 2:4096:0 2:8192:0 2:12288:0 \
4:1024:0 4:2048:0 4:4096:0 4:8192:0 4:16384:0 4:24576:0 \
8:1024:0 8:2048:0 8:4096:0 8:8192:0 8:16384:0 8:32768:0\
}"
export RESID_GBS="${RESID_GBS:-16}"
export RESID_TRAIN_ITERS="${RESID_TRAIN_ITERS:-20}"
export RESID_PROFILE_START_ITER="${RESID_PROFILE_START_ITER:-5}"
export RESID_PROFILE_END_ITER="${RESID_PROFILE_END_ITER:-19}"
# Which ranks write end2end JSONL. rank 0 + rank 8 covers both groups for sp<=8.
export RESID_PROFILE_RANKS="${RESID_PROFILE_RANKS:-0 8 15}"

# ---------- cluster identity ----------
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-10.122.244.241}"
export MASTER_PORT="${MASTER_PORT:-40061}"
export MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"

WORKER1_HOST="${WORKER1_HOST:-worker-1}"
DISPATCH_TIMEOUT="${DISPATCH_TIMEOUT:-7200}"

RUN_DIR="${SCRIPT_DIR}/results/${RESID_RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_DIR="${RUN_DIR}/end2end"
mkdir -p "${LOG_DIR}" "${PROFILE_DIR}"
W0_LOG="${LOG_DIR}/resid_node0.log"
W1_LOG="${LOG_DIR}/resid_node1.log"
W0_PIDFILE="${LOG_DIR}/w0.pid"

# ---------- gpu_busy lifecycle helpers (same as 13_/20_) ----------
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
export RESID_RUN_ID='${RESID_RUN_ID}'
export RESID_CELLS='${RESID_CELLS}'
export RESID_GBS='${RESID_GBS}'
export RESID_TRAIN_ITERS='${RESID_TRAIN_ITERS}'
export RESID_PROFILE_START_ITER='${RESID_PROFILE_START_ITER}'
export RESID_PROFILE_END_ITER='${RESID_PROFILE_END_ITER}'
export RESID_PROFILE_RANKS='${RESID_PROFILE_RANKS}'
export NNODES='${NNODES}'
export NPROC_PER_NODE='${NPROC_PER_NODE}'
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT='${MASTER_PORT}'
export MODEL_NAME='${MODEL_NAME}'
export PROFILE_DIR='${PROFILE_DIR}'
EOF
)

echo "==== Dispatching residual benchmark ${RESID_RUN_ID} ===="
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}  NNODES=${NNODES}  GPUS/NODE=${NPROC_PER_NODE}"
echo "MODEL=${MODEL_NAME}"
echo "cells (sp:seq:ckpt) ="
for c in ${RESID_CELLS}; do echo "    ${c}"; done
echo "GBS=${RESID_GBS}  train_iters=${RESID_TRAIN_ITERS}  profile=[${RESID_PROFILE_START_ITER},${RESID_PROFILE_END_ITER})"
echo "log_dir=${LOG_DIR}"
echo "profile_dir=${PROFILE_DIR}"

stop_gpu_busy
cleanup_zombies

# ---------- launch worker-1 detached over ssh ----------
ssh -o ConnectTimeout=10 "${WORKER1_HOST}" "
${ENV_EXPORTS}
export NODE_RANK=1
mkdir -p ${LOG_DIR} ${PROFILE_DIR}
nohup setsid bash '${SCRIPT_DIR}/22_bench_residual_worker.sh' \
    > '${W1_LOG}' 2>&1 < /dev/null &
echo \$! > '${LOG_DIR}/w1.pid'
"
echo "[dispatch] worker-1 launched (detached)"

# ---------- launch worker-0 detached locally ----------
eval "${ENV_EXPORTS}"
export NODE_RANK=0
nohup setsid bash "${SCRIPT_DIR}/22_bench_residual_worker.sh" \
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
find "${PROFILE_DIR}" -name "rank*.jsonl" 2>/dev/null | sort | head -10
