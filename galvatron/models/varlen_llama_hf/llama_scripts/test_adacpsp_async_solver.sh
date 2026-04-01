#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Validation: AdaCPSP async solver (double-buffering)
#
# Runs 3 tests:
#   1) Async mode (default) — solver overlaps with training
#   2) Sync mode (--adaCPSP-sync-solver) — baseline for comparison
#   3) Correctness: compare loss sequences from both modes
#
# Expected outcome:
#   - Both modes produce identical loss sequences (same data, same order)
#   - Async mode shows "Async solver enabled (double-buffering)" in logs
#   - Async mode shows "Async solver completed in X.XXXs" per iter
#   - Async mode wall time <= sync mode wall time
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

NUM_NODES=1
NUM_GPUS_PER_NODE=${NUM_GPUS:-8}
MASTER_ADDR=localhost
NODE_RANK=0

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TRAINER="train_dist_adacpsp.py"
LOGDIR="/tmp/adacpsp_async_test_$$"
mkdir -p "${LOGDIR}"

MODEL_ARGS="
    --model_size llama-7b
    --set_model_config_manually 0
    --set_layernum_manually 1
    --set_seqlen_manually 1
    --vocab_size 32000
    --hidden_size 4096
    --num_hidden_layers 2
    --num_attention_heads 32
    --seq_length 32768"

TRAIN_ARGS="
    --global_train_batch_size 16
    --train-iters 8
    --lr 1e-4
    --adam_weight_decay 0.01
    --dropout_prob 0.0
    --check_loss 1
    --profile 1
    --save_profiled_memory 0
    --dataset random"

PARALLEL_ARGS="
    --pp_deg 1
    --global_tp_deg 1
    --global_tp_consec 1
    --sdp 0
    --global_checkpoint 0
    --vocab_tp 1
    --chunks 1
    --global_cp_deg 1
    --pipeline_type pipedream_flush
    --default_dp_type zero2
    --mixed_precision bf16
    --use-flash-attn
    --initialize_on_meta 1
    --use-packing
    --use-adaCPSP"

run_test() {
    local name="$1"
    local extra_args="$2"
    local port="$3"
    local logfile="${LOGDIR}/${name}.log"

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  Test: ${name}"
    echo "═══════════════════════════════════════════════════"

    local start_ts
    start_ts=$(date +%s%N)

    torchrun \
        --nnodes=${NUM_NODES} \
        --nproc_per_node=${NUM_GPUS_PER_NODE} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${port} \
        --node_rank=${NODE_RANK} \
        ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${extra_args} \
        2>&1 | tee "${logfile}"

    local end_ts
    end_ts=$(date +%s%N)
    local elapsed_ms=$(( (end_ts - start_ts) / 1000000 ))
    echo ""
    echo "[${name}] Wall time: ${elapsed_ms} ms"
    echo "${elapsed_ms}" > "${LOGDIR}/${name}.walltime"
}

# ── Test 1: Async solver (default) ──
run_test "async" "" "29510"

# ── Test 2: Sync solver (baseline) ──
run_test "sync" "--adaCPSP-sync-solver" "29511"

# ── Test 3: Compare results ──
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Comparison"
echo "═══════════════════════════════════════════════════"

ASYNC_TIME=$(cat "${LOGDIR}/async.walltime")
SYNC_TIME=$(cat "${LOGDIR}/sync.walltime")

echo "Async wall time: ${ASYNC_TIME} ms"
echo "Sync  wall time: ${SYNC_TIME} ms"

if [ "${ASYNC_TIME}" -lt "${SYNC_TIME}" ]; then
    SAVED=$(( SYNC_TIME - ASYNC_TIME ))
    echo "Async saved ${SAVED} ms ($(( SAVED * 100 / SYNC_TIME ))%)"
else
    EXTRA=$(( ASYNC_TIME - SYNC_TIME ))
    echo "WARNING: Async was ${EXTRA} ms slower (expected to be faster)"
fi

# Check async-specific log markers
echo ""
echo "── Async mode log markers ──"
if grep -q "Async solver enabled" "${LOGDIR}/async.log"; then
    echo "[PASS] Async solver was enabled"
else
    echo "[FAIL] 'Async solver enabled' not found in async log"
fi

if grep -q "Warmup iter: solver launched" "${LOGDIR}/async.log"; then
    echo "[PASS] Warmup iteration executed"
else
    echo "[FAIL] Warmup iteration not found in async log"
fi

ASYNC_SOLVER_COUNT=$(grep -c "Async solver completed" "${LOGDIR}/async.log" || true)
echo "[INFO] Async solver completed ${ASYNC_SOLVER_COUNT} time(s)"

if grep -q "Sync solver mode" "${LOGDIR}/sync.log"; then
    echo "[PASS] Sync mode confirmed in baseline"
else
    echo "[FAIL] 'Sync solver mode' not found in sync log"
fi

# Compare loss values
echo ""
echo "── Loss comparison ──"
grep -oP 'Loss = [0-9.]+' "${LOGDIR}/async.log" | head -5 > "${LOGDIR}/async_loss.txt"
grep -oP 'Loss = [0-9.]+' "${LOGDIR}/sync.log"  | head -5 > "${LOGDIR}/sync_loss.txt"

echo "Async losses (first 5):"
cat "${LOGDIR}/async_loss.txt"
echo "Sync  losses (first 5):"
cat "${LOGDIR}/sync_loss.txt"

echo ""
echo "Logs saved to: ${LOGDIR}/"
echo "Done."
