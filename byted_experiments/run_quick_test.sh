#!/bin/bash
###############################################################################
# Small sanity run for ByteDance Merlin Seed.
#
# - On Merlin Seed, it uses the platform-injected env vars.
# - Outside Merlin Seed, it falls back to a single-node local run.
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${ARNOLD_WORKER_NUM:-}" ]; then
    export NNODES=1
    export NODE_RANK=0
    export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    if [ -z "${MASTER_PORT:-}" ]; then
        export MASTER_PORT="$(python - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"
    fi
fi

export MODELS="${MODELS:-qwen2.5-7b}"
export SEQ_LENGTHS_K="${SEQ_LENGTHS_K:-128}"
export STRATEGIES="${STRATEGIES:-adacpsp flexsp}"
export NUM_ITERS="${NUM_ITERS:-10}"
export WARMUP_ITERS="${WARMUP_ITERS:-3}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:--1}"
export MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-80}"
export EPOCHS="${EPOCHS:-1}"
export DEFAULT_DP_TYPE="${DEFAULT_DP_TYPE:-zero3}"
export NUM_WORKERS="${NUM_WORKERS:-0}"

if [ -n "${ARNOLD_WORKER_NUM:-}" ]; then
    TOTAL_GPUS=$((ARNOLD_WORKER_NUM * ARNOLD_WORKER_GPU))
else
    TOTAL_GPUS="${NPROC_PER_NODE}"
fi

export GPU_CONFIGS="${GPU_CONFIGS:-${TOTAL_GPUS}}"
export GBS_LIST="${GBS_LIST:-${GPU_CONFIGS}}"

bash "${SCRIPT_DIR}/run_all_experiments.sh"
