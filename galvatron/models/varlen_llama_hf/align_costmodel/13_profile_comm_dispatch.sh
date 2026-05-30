#!/usr/bin/env bash
# Dispatch a 2-node comm primitive profile run (alltoall_single + p2p_sendrecv
# + p2p_kv_pair) using profile_comm_v2.py.
#
# Mirrors 08_trace_ring_dispatch.sh:
#   - Both workers launched detached (setsid + nohup)
#   - Auto-manages gpu_busy lifecycle (stop before, restart after)
#   - Local poll on worker-0; worker-1 tail at end via ssh.
#
# Typical use: regenerate comm_profile_v2 *.json after adding the p2p_kv_pair
# primitive that matches RingComm.send_recv_kv (4-op batched send/recv).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- run identity ----------
export COMM_RUN_ID="${COMM_RUN_ID:-comm_v2_$(date +%Y%m%d_%H%M%S)}"
# default: profile all three primitives so we keep one source of truth.
export COMM_V2_PRIMITIVES="${COMM_V2_PRIMITIVES:-alltoall_single p2p_sendrecv p2p_kv_pair}"
export COMM_V2_MESSAGE_SIZES_MB="${COMM_V2_MESSAGE_SIZES_MB:-0.25 0.5 1 2 4 8 16 32 64 128 256}"
export COMM_V2_TOPOLOGY="${COMM_V2_TOPOLOGY:-both}"
export ACROSS_GROUP_AGG="${ACROSS_GROUP_AGG:-p90}"
export WARMUP_ITERS="${WARMUP_ITERS:-10}"
export MEASURE_ITERS="${MEASURE_ITERS:-50}"

# ---------- cluster identity ----------
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-10.122.244.241}"
export MASTER_PORT="${MASTER_PORT:-40021}"
export MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"

WORKER1_HOST="${WORKER1_HOST:-worker-1}"
DISPATCH_TIMEOUT="${DISPATCH_TIMEOUT:-1800}"

# We override 00_common.sh's ALIGN_RUN_ID indirectly via COMM_RUN_ID-based
# log dir; the profile_comm_v2 output filename uses its own timestamp so we
# don't need to set ALIGN_RUN_ID for collision avoidance.
RUN_DIR="${SCRIPT_DIR}/results/${COMM_RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
W0_LOG="${LOG_DIR}/comm_node0.log"
W1_LOG="${LOG_DIR}/comm_node1.log"
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
    pkill -9 -f 'profile_comm_v2.py' 2>/dev/null
    pkill -9 -f "torchrun.*${MASTER_PORT}" 2>/dev/null
    ssh "${WORKER1_HOST}" "pkill -9 -f 'profile_comm_v2.py' 2>/dev/null; \
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
export COMM_RUN_ID='${COMM_RUN_ID}'
export COMM_V2_PRIMITIVES='${COMM_V2_PRIMITIVES}'
export COMM_V2_MESSAGE_SIZES_MB='${COMM_V2_MESSAGE_SIZES_MB}'
export COMM_V2_TOPOLOGY='${COMM_V2_TOPOLOGY}'
export ACROSS_GROUP_AGG='${ACROSS_GROUP_AGG}'
export WARMUP_ITERS='${WARMUP_ITERS}'
export MEASURE_ITERS='${MEASURE_ITERS}'
export NNODES='${NNODES}'
export NPROC_PER_NODE='${NPROC_PER_NODE}'
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT='${MASTER_PORT}'
export MODEL_NAME='${MODEL_NAME}'
EOF
)

echo "==== Dispatching comm profile ${COMM_RUN_ID} ===="
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}  NNODES=${NNODES}  GPUS/NODE=${NPROC_PER_NODE}"
echo "primitives=${COMM_V2_PRIMITIVES}"
echo "message_sizes_MB=${COMM_V2_MESSAGE_SIZES_MB}"
echo "warmup=${WARMUP_ITERS} measure=${MEASURE_ITERS}"
echo "log_dir=${LOG_DIR}"

stop_gpu_busy
cleanup_zombies

# ---------- launch worker-1 detached over ssh ----------
ssh -o ConnectTimeout=10 "${WORKER1_HOST}" "
${ENV_EXPORTS}
export NODE_RANK=1
mkdir -p ${LOG_DIR}
nohup setsid bash '${SCRIPT_DIR}/13_profile_comm_worker.sh' \
    > '${W1_LOG}' 2>&1 < /dev/null &
echo \$! > '${LOG_DIR}/w1.pid'
"
echo "[dispatch] worker-1 launched (detached)"

# ---------- launch worker-0 detached locally ----------
eval "${ENV_EXPORTS}"
export NODE_RANK=0
nohup setsid bash "${SCRIPT_DIR}/13_profile_comm_worker.sh" \
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

# ---------- summary of produced profile json ----------
echo ""
echo "==== Latest comm_profile_v2 files ===="
ls -lt /mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron/galvatron/models/varlen_llama_hf/configs/comm_profile_v2_*.json 2>/dev/null | head -5 || true
