#!/usr/bin/env bash
# Dispatch a 2-node Ring-attention trace sweep.
#
# Both workers are launched detached (setsid + nohup) so that Ctrl-C of this
# dispatcher does NOT kill the remote training jobs - we just stop watching.
# We re-attach by polling the local pid + tailing logs.
#
# This script also takes care of the mnist gpu_job lifecycle:
#   - kills it on both nodes before the trace
#   - restarts it on both nodes after the trace (even on Ctrl-C of dispatcher),
#     so the GPUs do not stay idle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- run identity ----------
export TRACE_RUN_ID="${TRACE_RUN_ID:-ring_trace_$(date +%Y%m%d_%H%M%S)}"
export TRACE_CASES="${TRACE_CASES:-8192:16}"
export TRACE_NUM_SEQS="${TRACE_NUM_SEQS:-1}"
export TRACE_TOPOLOGY="${TRACE_TOPOLOGY:-consecutive}"
export TRACE_PROFILE_RANKS="${TRACE_PROFILE_RANKS:-0 8}"
export TRACE_WARMUP="${TRACE_WARMUP:-2}"
export TRACE_ACTIVE="${TRACE_ACTIVE:-1}"

# ---------- cluster identity ----------
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-10.122.244.241}"
export MASTER_PORT="${MASTER_PORT:-40011}"
export MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"

WORKER1_HOST="${WORKER1_HOST:-worker-1}"
DISPATCH_TIMEOUT="${DISPATCH_TIMEOUT:-600}"   # max seconds we wait for worker-0

RUN_DIR="${SCRIPT_DIR}/results/${TRACE_RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
W0_LOG="${LOG_DIR}/trace_node0.log"
W1_LOG="${LOG_DIR}/trace_node1.log"
W0_PIDFILE="${LOG_DIR}/w0.pid"

# ---------- gpu_busy lifecycle helpers ----------
GPU_BUSY_PY="/mnt/bn/wyj-data0-hl/lqs/gpu_busy.py"
GPU_BUSY_PYBIN="/mnt/bn/wyj-data0-hl/lqs/envs/galvatron-adacpsp-py39-torch21-cu121/bin/python"

stop_gpu_busy() {
    echo "[dispatch] stopping mnist/gpu_busy on both workers..."
    set +e
    tmux kill-session -t gpu_job 2>/dev/null
    ssh "${WORKER1_HOST}" 'tmux kill-session -t gpu_job 2>/dev/null; true'
    pkill -9 -f 'gpu_busy.py' 2>/dev/null
    ssh "${WORKER1_HOST}" 'pkill -9 -f "gpu_busy.py" 2>/dev/null; true'
    set -e
    return 0
}
start_gpu_busy() {
    echo "[dispatch] (re)starting mnist/gpu_busy on both workers..."
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
    pkill -9 -f '08_trace_ring_attention.py' 2>/dev/null
    pkill -9 -f "torchrun.*${MASTER_PORT}" 2>/dev/null
    ssh "${WORKER1_HOST}" "pkill -9 -f '08_trace_ring_attention.py' 2>/dev/null; \
                          pkill -9 -f 'torchrun.*${MASTER_PORT}' 2>/dev/null; true"
    set -e
    return 0
}

# trap: on normal exit OR Ctrl-C, always restart mnist
on_exit() {
    local rc=$?
    echo "[dispatch] exit handler, rc=${rc}"
    # NB: we do NOT kill the trace workers on Ctrl-C - they keep running detached
    # and write trace files even if dispatcher dies. They will print their own
    # PASS/FAIL into the log files. Caller can re-attach with `tail -f`.
    start_gpu_busy
    exit "${rc}"
}
trap on_exit EXIT

# ---------- environment to forward to workers ----------
ENV_EXPORTS=$(cat <<EOF
export TRACE_RUN_ID='${TRACE_RUN_ID}'
export TRACE_CASES='${TRACE_CASES}'
export TRACE_NUM_SEQS='${TRACE_NUM_SEQS}'
export TRACE_TOPOLOGY='${TRACE_TOPOLOGY}'
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
echo "warmup=${TRACE_WARMUP} active=${TRACE_ACTIVE}"
echo "log_dir=${LOG_DIR}"

stop_gpu_busy
cleanup_zombies

# ---------- launch worker-1 detached over ssh ----------
ssh -o ConnectTimeout=10 "${WORKER1_HOST}" "
${ENV_EXPORTS}
export NODE_RANK=1
mkdir -p ${LOG_DIR}
nohup setsid bash '${SCRIPT_DIR}/08_trace_ring_attention.sh' \
    > '${W1_LOG}' 2>&1 < /dev/null &
echo \$! > '${LOG_DIR}/w1.pid'
"
echo "[dispatch] worker-1 launched (detached)"

# ---------- launch worker-0 detached locally ----------
eval "${ENV_EXPORTS}"
export NODE_RANK=0
nohup setsid bash "${SCRIPT_DIR}/08_trace_ring_attention.sh" \
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
    sleep 10
    cur_size=$(stat -c %s "${W0_LOG}" 2>/dev/null || echo 0)
    if [ "${cur_size}" != "${LAST_LOG_SIZE}" ]; then
        echo "[dispatch] t=${SECONDS}s w0.log size=${cur_size}B"
        LAST_LOG_SIZE="${cur_size}"
    fi
done

echo "[dispatch] worker-0 finished after ${SECONDS}s"
echo "----- w0.log tail -----"
tail -30 "${W0_LOG}"
echo "----- w1.log tail -----"
ssh "${WORKER1_HOST}" "tail -30 ${W1_LOG}" || true

# ---------- summary of produced traces ----------
TRACE_ROOT="${RUN_DIR}/traces"
echo ""
echo "==== Trace files produced ===="
if [ -d "${TRACE_ROOT}" ]; then
    find "${TRACE_ROOT}" -name '*.pt.trace.json' | sort
    echo "count=$(find "${TRACE_ROOT}" -name '*.pt.trace.json' | wc -l)"
else
    echo "(no trace dir)"
fi
