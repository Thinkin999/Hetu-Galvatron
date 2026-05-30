#!/usr/bin/env bash
# Dispatch a 2-node FSDP training-step trace using the existing
# profile_adacpsp_end2end_timeline.sh harness, captured by torch.profiler.
#
# Purpose (Stage 0 of non-attention residual modeling):
#   Validate the "compute hides ZeRO3 all-gather / reduce-scatter" saturation
#   assumption by looking at how much of the FSDP comm kernels (ncclAllGather,
#   ncclReduceScatter) overlap with non-attention compute kernels (matmul,
#   FlashAttention) on the GPU stream.
#
# Mirrors 13_profile_comm_dispatch.sh:
#   - Both workers launched detached (setsid + nohup)
#   - Stops gpu_busy before launch, restarts on exit (always)
#   - Local poll on worker-0 with progress ticker
#
# Defaults are intentionally small (1 strategy, ~6 iters per case, 2 iters
# traced) so the run finishes in a few minutes and yields actionable traces.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- run identity ----------
export FSDP_RUN_ID="${FSDP_RUN_ID:-fsdp_step_$(date +%Y%m%d_%H%M%S)}"
export FSDP_SEQ_LENGTHS="${FSDP_SEQ_LENGTHS:-4096 16384}"
export FSDP_STRATEGY="${FSDP_STRATEGY:-ulysses:8}"
export FSDP_GBS="${FSDP_GBS:-16}"
export FSDP_TRAIN_ITERS="${FSDP_TRAIN_ITERS:-6}"
export FSDP_TIMELINE_START_ITER="${FSDP_TIMELINE_START_ITER:-3}"
export FSDP_TIMELINE_END_ITER="${FSDP_TIMELINE_END_ITER:-5}"
# default: trace rank 0 (worker-0 local0) + rank 8 (worker-1 local0)
export FSDP_TIMELINE_RANKS="${FSDP_TIMELINE_RANKS:-0 8}"
export FSDP_SDP="${FSDP_SDP:-1}"          # 1 = encoder layer ZeRO3
export FSDP_DEFAULT_DP="${FSDP_DEFAULT_DP:-zero2}"
export FSDP_USE_CKPT="${FSDP_USE_CKPT:-0}"

# ---------- cluster identity ----------
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-10.122.244.241}"
export MASTER_PORT="${MASTER_PORT:-40041}"
export MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"

WORKER1_HOST="${WORKER1_HOST:-worker-1}"
DISPATCH_TIMEOUT="${DISPATCH_TIMEOUT:-1800}"

RUN_DIR="${SCRIPT_DIR}/results/${FSDP_RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
TRACE_ROOT="${RUN_DIR}/traces"
mkdir -p "${LOG_DIR}" "${TRACE_ROOT}"
W0_LOG="${LOG_DIR}/fsdp_node0.log"
W1_LOG="${LOG_DIR}/fsdp_node1.log"
W0_PIDFILE="${LOG_DIR}/w0.pid"

# ---------- gpu_busy lifecycle helpers ----------
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

# ---------- environment forwarded to workers ----------
ENV_EXPORTS=$(cat <<EOF
export FSDP_RUN_ID='${FSDP_RUN_ID}'
export FSDP_SEQ_LENGTHS='${FSDP_SEQ_LENGTHS}'
export FSDP_STRATEGY='${FSDP_STRATEGY}'
export FSDP_GBS='${FSDP_GBS}'
export FSDP_TRAIN_ITERS='${FSDP_TRAIN_ITERS}'
export FSDP_TIMELINE_START_ITER='${FSDP_TIMELINE_START_ITER}'
export FSDP_TIMELINE_END_ITER='${FSDP_TIMELINE_END_ITER}'
export FSDP_TIMELINE_RANKS='${FSDP_TIMELINE_RANKS}'
export FSDP_SDP='${FSDP_SDP}'
export FSDP_DEFAULT_DP='${FSDP_DEFAULT_DP}'
export FSDP_USE_CKPT='${FSDP_USE_CKPT}'
export NNODES='${NNODES}'
export NPROC_PER_NODE='${NPROC_PER_NODE}'
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT='${MASTER_PORT}'
export MODEL_NAME='${MODEL_NAME}'
export TRACE_ROOT='${TRACE_ROOT}'
EOF
)

echo "==== Dispatching FSDP step trace ${FSDP_RUN_ID} ===="
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}  NNODES=${NNODES}  GPUS/NODE=${NPROC_PER_NODE}"
echo "MODEL=${MODEL_NAME} STRATEGY=${FSDP_STRATEGY}"
echo "SEQ_LENGTHS=${FSDP_SEQ_LENGTHS} GBS=${FSDP_GBS}"
echo "sdp=${FSDP_SDP} default_dp=${FSDP_DEFAULT_DP} ckpt=${FSDP_USE_CKPT}"
echo "train_iters=${FSDP_TRAIN_ITERS} trace iters=[${FSDP_TIMELINE_START_ITER},${FSDP_TIMELINE_END_ITER})"
echo "trace_ranks=${FSDP_TIMELINE_RANKS}"
echo "log_dir=${LOG_DIR}"
echo "trace_root=${TRACE_ROOT}"

stop_gpu_busy
cleanup_zombies

# ---------- launch worker-1 detached over ssh ----------
ssh -o ConnectTimeout=10 "${WORKER1_HOST}" "
${ENV_EXPORTS}
export NODE_RANK=1
mkdir -p ${LOG_DIR} ${TRACE_ROOT}
nohup setsid bash '${SCRIPT_DIR}/20_trace_fsdp_step_worker.sh' \
    > '${W1_LOG}' 2>&1 < /dev/null &
echo \$! > '${LOG_DIR}/w1.pid'
"
echo "[dispatch] worker-1 launched (detached)"

# ---------- launch worker-0 detached locally ----------
eval "${ENV_EXPORTS}"
export NODE_RANK=0
nohup setsid bash "${SCRIPT_DIR}/20_trace_fsdp_step_worker.sh" \
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
tail -40 "${W0_LOG}"
echo "----- w1.log tail -----"
ssh "${WORKER1_HOST}" "tail -40 ${W1_LOG}" || true

echo ""
echo "==== Traces produced ===="
find "${TRACE_ROOT}" -maxdepth 3 -name "*.json*" 2>/dev/null | sort | head -20
