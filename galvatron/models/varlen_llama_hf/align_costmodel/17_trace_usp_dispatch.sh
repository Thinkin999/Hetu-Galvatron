#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ID="${RUN_ID:-usp_trace_$(date +%Y%m%d_%H%M%S)}"
# default cases: 4096:4:2 (big gap), 4096:2:4 (small gap), 4096:8:2 (worst gap cross-node)
CASES="${CASES:-4096:4:2 4096:2:4 4096:8:2}"
NUM_SEQS="${NUM_SEQS:-16}"
ACTIVE="${ACTIVE:-2}"
WARMUP="${WARMUP:-3}"

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-10.122.244.241}"
MASTER_PORT="${MASTER_PORT:-40071}"
WORKER1_HOST="${WORKER1_HOST:-worker-1}"

RUN_DIR="${SCRIPT_DIR}/traces/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
W0_LOG="${LOG_DIR}/w0.log"
W1_LOG="${LOG_DIR}/w1.log"

TORCHRUN="/mnt/bn/wyj-data0-hl/lqs/envs/galvatron-adacpsp-py39-torch21-cu121/bin/torchrun"

NCCL_EXPORTS=$(cat <<'EOF'
export NCCL_IB_HCA=mlx5
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_SOCKET_IFNAME==eth0
export NCCL_NET_PLUGIN=none
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_DEBUG=WARN
export GALVATRON_ULYSSES_TRACE=1
EOF
)

stop_gpu_busy() {
    set +e
    tmux kill-session -t gpu_job 2>/dev/null
    ssh "${WORKER1_HOST}" 'tmux kill-session -t gpu_job 2>/dev/null; true'
    pkill -9 -f 'gpu_busy.py' 2>/dev/null
    ssh "${WORKER1_HOST}" 'pkill -9 -f "gpu_busy.py" 2>/dev/null; true'
    set -e
}
start_gpu_busy() {
    set +e
    local pybin="/mnt/bn/wyj-data0-hl/lqs/envs/galvatron-adacpsp-py39-torch21-cu121/bin/python"
    tmux new-session -d -s gpu_job "${pybin} /mnt/bn/wyj-data0-hl/lqs/gpu_busy.py 2>&1 | tee /tmp/gpu_busy_w0.log"
    ssh "${WORKER1_HOST}" "tmux new-session -d -s gpu_job '${pybin} /mnt/bn/wyj-data0-hl/lqs/gpu_busy.py 2>&1 | tee /tmp/gpu_busy_w1.log'"
    set -e
}
on_exit() { local rc=$?; start_gpu_busy; exit "${rc}"; }
trap on_exit EXIT

stop_gpu_busy
set +e
pkill -9 -f '17_trace_usp.py' 2>/dev/null
ssh "${WORKER1_HOST}" "pkill -9 -f '17_trace_usp.py' 2>/dev/null; true"
set -e

# Cases passed as a comma-separated list to avoid shell word splitting.
CASES_CSV=$(echo "${CASES}" | tr ' ' ',')
CMD_BASE="${TORCHRUN} --nnodes ${NNODES} --nproc_per_node ${NPROC_PER_NODE} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"
CMD_W0="${CMD_BASE} --node_rank 0 ${SCRIPT_DIR}/17_trace_usp.py --trace-dir ${RUN_DIR} --cases ${CASES_CSV} --num-seqs ${NUM_SEQS} --warmup ${WARMUP} --active ${ACTIVE}"
CMD_W1="${CMD_BASE} --node_rank 1 ${SCRIPT_DIR}/17_trace_usp.py --trace-dir ${RUN_DIR} --cases ${CASES_CSV} --num-seqs ${NUM_SEQS} --warmup ${WARMUP} --active ${ACTIVE}"

ssh -o ConnectTimeout=10 "${WORKER1_HOST}" "
${NCCL_EXPORTS}
nohup setsid ${CMD_W1} > '${W1_LOG}' 2>&1 < /dev/null &
echo \$! > '${LOG_DIR}/w1.pid'
"
echo "[dispatch] worker-1 launched"

eval "${NCCL_EXPORTS}"
nohup setsid ${CMD_W0} > "${W0_LOG}" 2>&1 < /dev/null &
W0_PID=$!
echo "${W0_PID}" > "${LOG_DIR}/w0.pid"
echo "[dispatch] worker-0 pid=${W0_PID}"

SECONDS=0
while kill -0 "${W0_PID}" 2>/dev/null; do
    if (( SECONDS > 900 )); then echo "[dispatch] timeout"; exit 2; fi
    sleep 8
done
echo "[dispatch] done in ${SECONDS}s"
tail -40 "${W0_LOG}"
echo "--- traces ---"
ls -la "${RUN_DIR}"
