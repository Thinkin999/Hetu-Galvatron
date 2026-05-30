#!/usr/bin/env bash
# Dispatch a 2-node Ulysses-attention trace sweep.
# Mirrors 08_trace_ring_dispatch.sh: setsid/nohup detached workers + gpu_busy
# lifecycle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRACE_RUN_ID="${TRACE_RUN_ID:-uly_trace_$(date +%Y%m%d_%H%M%S)}"
# Default: cover sp ∈ {2, 8, 16} at one short and one long seq each to
# illustrate the CPU-overhead-hidden-by-GPU pattern.
export TRACE_CASES="${TRACE_CASES:-4096:16 8192:16 32768:16 4096:8 8192:8 32768:8 4096:2 32768:2}"
export TRACE_NUM_SEQS="${TRACE_NUM_SEQS:-16}"
export TRACE_PROFILE_RANKS="${TRACE_PROFILE_RANKS:-0 8}"
export TRACE_WARMUP="${TRACE_WARMUP:-3}"
export TRACE_ACTIVE="${TRACE_ACTIVE:-3}"

export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-10.122.244.241}"
export MASTER_PORT="${MASTER_PORT:-40041}"
export MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"

WORKER1_HOST="${WORKER1_HOST:-worker-1}"
DISPATCH_TIMEOUT="${DISPATCH_TIMEOUT:-1800}"

RUN_DIR="${SCRIPT_DIR}/results/${TRACE_RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
W0_LOG="${LOG_DIR}/trace_node0.log"
W1_LOG="${LOG_DIR}/trace_node1.log"
W0_PIDFILE="${LOG_DIR}/w0.pid"

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
    pkill -9 -f '15_trace_ulysses.py' 2>/dev/null
    pkill -9 -f "torchrun.*${MASTER_PORT}" 2>/dev/null
    ssh "${WORKER1_HOST}" "pkill -9 -f '15_trace_ulysses.py' 2>/dev/null; \
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
export TRACE_RUN_ID='${TRACE_RUN_ID}'
export TRACE_CASES='${TRACE_CASES}'
export TRACE_NUM_SEQS='${TRACE_NUM_SEQS}'
export TRACE_PROFILE_RANKS='${TRACE_PROFILE_RANKS}'
export TRACE_WARMUP='${TRACE_WARMUP}'
export TRACE_ACTIVE='${TRACE_ACTIVE}'
export NNODES='${NNODES}'
export NPROC_PER_NODE='${NPROC_PER_NODE}'
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT='${MASTER_PORT}'
export MODEL_NAME='${MODEL_NAME}'
EOF
)

echo "==== Dispatching ${TRACE_RUN_ID} ===="
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}  NNODES=${NNODES}  GPUS/NODE=${NPROC_PER_NODE}"
echo "TRACE_CASES=${TRACE_CASES}"
echo "log_dir=${LOG_DIR}"

stop_gpu_busy
cleanup_zombies

ssh -o ConnectTimeout=10 "${WORKER1_HOST}" "
${ENV_EXPORTS}
export NODE_RANK=1
mkdir -p ${LOG_DIR}
nohup setsid bash '${SCRIPT_DIR}/15_trace_ulysses_worker.sh' \
    > '${W1_LOG}' 2>&1 < /dev/null &
echo \$! > '${LOG_DIR}/w1.pid'
"
echo "[dispatch] worker-1 launched (detached)"

eval "${ENV_EXPORTS}"
export NODE_RANK=0
nohup setsid bash "${SCRIPT_DIR}/15_trace_ulysses_worker.sh" \
    > "${W0_LOG}" 2>&1 < /dev/null &
W0_PID=$!
echo "${W0_PID}" > "${W0_PIDFILE}"
echo "[dispatch] worker-0 launched pid=${W0_PID} (detached)"

SECONDS=0
LAST_LOG_SIZE=0
while kill -0 "${W0_PID}" 2>/dev/null; do
    if (( SECONDS > DISPATCH_TIMEOUT )); then
        echo "[dispatch] WARN: worker-0 still running after ${DISPATCH_TIMEOUT}s"
        exit 2
    fi
    sleep 10
    cur_size=$(stat -c %s "${W0_LOG}" 2>/dev/null || echo 0)
    if [ "${cur_size}" != "${LAST_LOG_SIZE}" ]; then
        echo "[dispatch] t=${SECONDS}s w0.log size=${cur_size}B"
        LAST_LOG_SIZE="${cur_size}"
    fi
done

echo "[dispatch] worker-0 finished after ${SECONDS}s"
echo "----- w0.log tail -----"
tail -40 "${W0_LOG}"

TRACE_ROOT="${RUN_DIR}/traces"
echo ""
echo "==== Trace files produced ===="
if [ -d "${TRACE_ROOT}" ]; then
    find "${TRACE_ROOT}" -name '*.pt.trace.json' | sort
    echo "count=$(find "${TRACE_ROOT}" -name '*.pt.trace.json' | wc -l)"
else
    echo "(no trace dir)"
fi
