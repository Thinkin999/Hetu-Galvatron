#!/usr/bin/env bash
# Dispatch the saturation-test benchmark.
#
# Hypothesis under test:
#   "At production-scale seq_length (>= ~65k) with ckpt=0 + ZeRO-3, FSDP
#    AllGather is fully hidden by per-layer compute, so the per-step constant
#    `b_step_fb` calibrated at short-seq (8k) is irrelevant in production."
#
# Validation design:
#   - 3 forced strategies × 4 chunks × 2 seq_length cells = 24 cells
#     strategies: ulysses8, ring8, usp2x4
#     chunks:     1, 2, 4, 8         (4 points → robust linear regression)
#     seq:        65536, 131072      (boundary → safe-saturation comparison)
#   - At each (strategy, seq), fit fb_clean(N) = slope·N + intercept
#       slope     = effective per-mb cost  (saturation-true if linear)
#       intercept = un-hidden FSDP overhead in production
#       saturation_holds  ⇔  |intercept| << slope    AND    slope ≈ fb_clean(1)
#
# Memory: ZeRO-3 + ckpt=0 + Qwen 7B at seq=131k, sp=8 → ~33 GB/card. Safe on A800-80GB.
#
# Once we know if saturation holds, we'll decide whether to:
#   (a) zero out b_step_fb in the cost model (saturation production)
#   (b) keep a small sp-dependent residual fitted from these runs
#
# Encoding: cells = "<config>:<chunks>:<seq_label>". seq_label ∈ {65k, 131k}.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- run identity ----------
export SAT_RUN_ID="${SAT_RUN_ID:-sat_$(date +%Y%m%d_%H%M%S)}"
# 24 cells. To reduce wall-clock, we can override via env to a subset.
# NOTE on memory constraints:
#   - All 3 strategies (ulysses8 sp=8 / ring8 cp=8 / usp2x4 sp=2·cp=4) yield
#     the same DP group = world / (sp·cp) = 16 / 8 = 2. ZeRO-3 + Qwen 7B +
#     ckpt=0 leaves static ~28 GB/rank and grows during activation backward.
#   - At chunks=1 + GBS=16, one microbatch packs ALL 16 sequences → per-rank
#     tokens ~ 60k → activations ~50+ GB → OOM with the ~28 GB static.
#   - chunks ≥ 2 splits the global batch into smaller microbatches and fits
#     comfortably. chunks=16 (1 seq per mb) is the most memory-friendly.
# So we sweep chunks ∈ {2, 4, 8, 16} (still 4 points for linear regression).
#
# Saturation hypothesis test at seq=65k:
#   FFN compute per layer ≈ 5 ms (fwd) × 3 (incl bwd) = 15 ms
#   FSDP AllGather per layer ≈ 6 ms (DP=2 over NVLink)
#   compute/comm ratio ≈ 2.5×  → with forward_prefetch, overlap should be full
# We test by fitting fb_clean(N) = slope·N + intercept; intercept ≈ 0 ⇒ saturated.
export SAT_CELLS="${SAT_CELLS:-\
ulysses8:2:65k  ulysses8:4:65k  ulysses8:8:65k  ulysses8:16:65k \
ring8:2:65k     ring8:4:65k     ring8:8:65k     ring8:16:65k \
usp2x4:2:65k    usp2x4:4:65k    usp2x4:8:65k    usp2x4:16:65k\
}"
# GBS must be a multiple of (num_groups × chunks). num_groups=2, max chunks=8 → 16.
export SAT_GBS="${SAT_GBS:-16}"
export SAT_TRAIN_ITERS="${SAT_TRAIN_ITERS:-24}"
export SAT_PROFILE_START_ITER="${SAT_PROFILE_START_ITER:-5}"
export SAT_PROFILE_END_ITER="${SAT_PROFILE_END_ITER:-23}"
export SAT_PROFILE_RANKS="${SAT_PROFILE_RANKS:-0 8 15}"

# ---------- cluster identity ----------
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-10.122.244.241}"
export MASTER_PORT="${MASTER_PORT:-40084}"
export MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"

WORKER1_HOST="${WORKER1_HOST:-worker-1}"
DISPATCH_TIMEOUT="${DISPATCH_TIMEOUT:-14400}"   # 4h ceiling for the whole sweep

RUN_DIR="${SCRIPT_DIR}/results/${SAT_RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_DIR="${RUN_DIR}/end2end"
mkdir -p "${LOG_DIR}" "${PROFILE_DIR}"
W0_LOG="${LOG_DIR}/sat_node0.log"
W1_LOG="${LOG_DIR}/sat_node1.log"

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

ENV_EXPORTS=$(cat <<EOF
export SAT_RUN_ID='${SAT_RUN_ID}'
export SAT_CELLS='${SAT_CELLS}'
export SAT_GBS='${SAT_GBS}'
export SAT_TRAIN_ITERS='${SAT_TRAIN_ITERS}'
export SAT_PROFILE_START_ITER='${SAT_PROFILE_START_ITER}'
export SAT_PROFILE_END_ITER='${SAT_PROFILE_END_ITER}'
export SAT_PROFILE_RANKS='${SAT_PROFILE_RANKS}'
export NNODES='${NNODES}'
export NPROC_PER_NODE='${NPROC_PER_NODE}'
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT='${MASTER_PORT}'
export MODEL_NAME='${MODEL_NAME}'
export PROFILE_DIR='${PROFILE_DIR}'
EOF
)

echo "==== Dispatching SATURATION benchmark ${SAT_RUN_ID} ===="
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}  NNODES=${NNODES}  GPUS/NODE=${NPROC_PER_NODE}"
echo "MODEL=${MODEL_NAME}  GBS=${SAT_GBS}"
echo "Test:   ZeRO-3 + ckpt=0 + seq ∈ {65k, 131k} + chunks ∈ {1,2,4,8}"
echo "cells (<config>:<chunks>:<seq>) ="
for c in ${SAT_CELLS}; do echo "    ${c}"; done
echo "train_iters=${SAT_TRAIN_ITERS}  profile=[${SAT_PROFILE_START_ITER},${SAT_PROFILE_END_ITER})"
echo "log_dir=${LOG_DIR}"
echo "profile_dir=${PROFILE_DIR}"

stop_gpu_busy
cleanup_zombies

ssh -o ConnectTimeout=10 "${WORKER1_HOST}" "
${ENV_EXPORTS}
export NODE_RANK=1
mkdir -p ${LOG_DIR} ${PROFILE_DIR}
nohup setsid bash '${SCRIPT_DIR}/28_bench_saturation_worker.sh' \
    > '${W1_LOG}' 2>&1 < /dev/null &
echo \$! > '${LOG_DIR}/w1.pid'
"
echo "[dispatch] worker-1 launched (detached)"

eval "${ENV_EXPORTS}"
export NODE_RANK=0
nohup setsid bash "${SCRIPT_DIR}/28_bench_saturation_worker.sh" \
    > "${W0_LOG}" 2>&1 < /dev/null &
W0_PID=$!
echo "${W0_PID}" > "${LOG_DIR}/w0.pid"
echo "[dispatch] worker-0 launched pid=${W0_PID} (detached)"

SECONDS=0
LAST_LOG_SIZE=0
while kill -0 "${W0_PID}" 2>/dev/null; do
    if (( SECONDS > DISPATCH_TIMEOUT )); then
        echo "[dispatch] WARN: worker-0 still running after ${DISPATCH_TIMEOUT}s"
        echo "[dispatch] not killing it - tail its log at ${W0_LOG}"
        exit 2
    fi
    sleep 30
    cur_size=$(stat -c %s "${W0_LOG}" 2>/dev/null || echo 0)
    if [ "${cur_size}" != "${LAST_LOG_SIZE}" ]; then
        echo "[dispatch] t=${SECONDS}s w0.log size=${cur_size}B"
        LAST_LOG_SIZE="${cur_size}"
    fi
done

echo "[dispatch] worker-0 finished after ${SECONDS}s"
echo "----- w0.log tail -----"
tail -50 "${W0_LOG}"
echo "----- w1.log tail -----"
ssh "${WORKER1_HOST}" "tail -50 ${W1_LOG}" || true

echo ""
echo "==== JSONL files produced ===="
find "${PROFILE_DIR}" -name "rank*.jsonl" 2>/dev/null | sort | head -50
