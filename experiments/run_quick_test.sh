#!/bin/bash
###############################################################################
# 快速验证脚本: 在单节点 8 卡上跑一个小实验，验证流程是否通畅
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export MASTER_ADDR=localhost
if [ -n "${QUICK_TEST_MASTER_PORT:-}" ]; then
    export MASTER_PORT="${QUICK_TEST_MASTER_PORT}"
else
    export MASTER_PORT="$(python - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"
fi
export NPROC_PER_NODE=8

# 只跑一个小模型 + 短序列 + 两种策略
export MODELS="qwen2.5-7b"
export SEQ_LENGTHS_K="128"
export GPU_CONFIGS="8"
export GBS_LIST="8"
export STRATEGIES="adacpsp flexsp"
export NUM_ITERS="10"
export WARMUP_ITERS="3"
export TIMEOUT_SECONDS="300"
export MEMORY_LIMIT_GB="40"
export EPOCHS="1"
export DEFAULT_DP_TYPE="${DEFAULT_DP_TYPE:-zero3}"
export NUM_WORKERS="${NUM_WORKERS:-0}"

bash "${SCRIPT_DIR}/run_all_experiments.sh"

