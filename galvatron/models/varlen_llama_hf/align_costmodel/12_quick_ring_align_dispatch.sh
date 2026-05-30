#!/usr/bin/env bash
# 2-node fresh Ring p16 attention layer benchmark for 3 seq lengths.
# Uses the same setsid/nohup + mnist lifecycle as 08_trace_ring_dispatch.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QUICK_RUN_ID="${QUICK_RUN_ID:-quick_ring_$(date +%H%M%S)}"
SEQS="${SEQS:-8192,16384,32768}"
NUM_SEQS="${NUM_SEQS:-16}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-3}"

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-10.122.244.241}"
MASTER_PORT="${MASTER_PORT:-40021}"
WORKER1_HOST="${WORKER1_HOST:-worker-1}"
DISPATCH_TIMEOUT="${DISPATCH_TIMEOUT:-600}"

RUN_DIR="${SCRIPT_DIR}/results/${QUICK_RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
W0_LOG="${LOG_DIR}/w0.log"
W1_LOG="${LOG_DIR}/w1.log"
OUTPUT_JSON="${RUN_DIR}/measured.json"

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
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
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
    ssh "${WORKER1_HOST}" \
        "tmux new-session -d -s gpu_job '${pybin} /mnt/bn/wyj-data0-hl/lqs/gpu_busy.py 2>&1 | tee /tmp/gpu_busy_w1.log'"
    set -e
}
on_exit() {
    local rc=$?
    start_gpu_busy
    exit "${rc}"
}
trap on_exit EXIT

echo "[dispatch] ${QUICK_RUN_ID}  seqs=${SEQS}  num_seqs=${NUM_SEQS}"
stop_gpu_busy

make_cmd() {
    local node_rank=$1
    echo "${TORCHRUN} \
      --nnodes ${NNODES} --nproc_per_node ${NPROC_PER_NODE} \
      --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
      --node_rank ${node_rank} \
      ${SCRIPT_DIR}/12_quick_ring_align.py \
      --seqs ${SEQS} --num-seqs ${NUM_SEQS} \
      --warmup ${WARMUP} --iters ${ITERS} \
      --output-json ${OUTPUT_JSON}"
}
CMD_W0=$(make_cmd 0)
CMD_W1=$(make_cmd 1)

# launch worker-1 detached
ssh -o ConnectTimeout=10 "${WORKER1_HOST}" "
${NCCL_EXPORTS}
nohup setsid ${CMD_W1} \
    > '${W1_LOG}' 2>&1 < /dev/null &
echo \$! > '${LOG_DIR}/w1.pid'
"
echo "[dispatch] worker-1 launched"

# launch worker-0 detached locally
eval "${NCCL_EXPORTS}"
nohup setsid ${CMD_W0} > "${W0_LOG}" 2>&1 < /dev/null &
W0_PID=$!
echo "${W0_PID}" > "${LOG_DIR}/w0.pid"
echo "[dispatch] worker-0 launched pid=${W0_PID}"

# poll worker-0
SECONDS=0
LAST=0
while kill -0 "${W0_PID}" 2>/dev/null; do
    if (( SECONDS > DISPATCH_TIMEOUT )); then
        echo "[dispatch] WARN: timeout, check ${W0_LOG}"
        exit 2
    fi
    sleep 10
    cur=$(stat -c %s "${W0_LOG}" 2>/dev/null || echo 0)
    if [ "${cur}" != "${LAST}" ]; then
        echo "[dispatch] t=${SECONDS}s w0.log=${cur}B"
        LAST="${cur}"
    fi
done

echo "[dispatch] worker-0 finished after ${SECONDS}s"
echo "----- w0.log tail -----"
tail -25 "${W0_LOG}"
echo "----- measured.json -----"
cat "${OUTPUT_JSON}" 2>/dev/null || echo "(no output)"
