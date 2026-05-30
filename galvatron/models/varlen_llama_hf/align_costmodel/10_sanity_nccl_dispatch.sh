#!/usr/bin/env bash
#
# 2-node cross-host NCCL sanity check. Should complete in ~30 seconds.
# Useful before any real cross-node experiment to confirm IB / GID config.
#
# This dispatcher uses setsid + nohup on both sides so the workers survive
# an accidental Ctrl-C of the dispatcher itself.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MASTER_ADDR="${MASTER_ADDR:-10.122.244.241}"
MASTER_PORT="${MASTER_PORT:-40007}"
WORKER1_HOST="${WORKER1_HOST:-worker-1}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/results/nccl_sanity_$(date +%H%M%S)}"
mkdir -p "${LOG_DIR}"

CONDA_PY="/mnt/bn/wyj-data0-hl/lqs/envs/galvatron-adacpsp-py39-torch21-cu121/bin/python"
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

echo "[dispatch] log_dir=${LOG_DIR}"
echo "[dispatch] master=${MASTER_ADDR}:${MASTER_PORT}"

# --- launch worker-1 in background via ssh, detached so it survives if we die.
LAUNCH_W1="\
${NCCL_EXPORTS}
cd ${SCRIPT_DIR}
nohup setsid ${TORCHRUN} \
  --nnodes 2 --nproc_per_node 8 \
  --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
  --node_rank 1 \
  ${SCRIPT_DIR}/10_sanity_nccl.py \
  > ${LOG_DIR}/w1.log 2>&1 < /dev/null &
"
ssh -o ConnectTimeout=10 "${WORKER1_HOST}" "${LAUNCH_W1}"
echo "[dispatch] worker-1 launched (detached)"

# --- launch worker-0 locally, also detached so dispatcher Ctrl-C does not kill it.
eval "${NCCL_EXPORTS}"
cd "${SCRIPT_DIR}"
nohup setsid "${TORCHRUN}" \
  --nnodes 2 --nproc_per_node 8 \
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" \
  --node_rank 0 \
  "${SCRIPT_DIR}/10_sanity_nccl.py" \
  > "${LOG_DIR}/w0.log" 2>&1 < /dev/null &
W0_PID=$!
echo "[dispatch] worker-0 launched pid=${W0_PID}"
echo "${W0_PID}" > "${LOG_DIR}/w0.pid"

# --- wait up to 300 s for worker-0 to finish; print its log tail when done.
SECONDS=0
TIMEOUT=300
while kill -0 "${W0_PID}" 2>/dev/null; do
    if (( SECONDS > TIMEOUT )); then
        echo "[dispatch] WARN: worker-0 still running after ${TIMEOUT}s, leaving it; check ${LOG_DIR}/w0.log"
        exit 2
    fi
    sleep 2
done
echo "[dispatch] worker-0 exited."
echo "----- w0.log tail -----"
tail -30 "${LOG_DIR}/w0.log"
echo "----- w1.log tail -----"
ssh "${WORKER1_HOST}" "tail -30 ${LOG_DIR}/w1.log"
