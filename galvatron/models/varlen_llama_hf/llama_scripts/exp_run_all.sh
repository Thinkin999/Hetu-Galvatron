#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Master Script: Run ALL FlexSP vs AdaCPSP Experiments
# ═══════════════════════════════════════════════════════════════
cd "$(dirname "$0")" || exit 1

TOTAL_START=$(date +%s)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FlexSP vs AdaCPSP — Full Experiment Suite                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment: E5_fast_val — 2-layer quick validation"
echo "  max_seq=32768, GBS=32, layers=2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "  → Running FLEXSP..."
bash exp_E5_fast_val_flexsp.sh
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "  ✗ exp_E5_fast_val_flexsp.sh failed with exit code $EXIT_CODE"
    echo "  Continuing with next experiment..."
fi
echo "  Cooling down (10s)..."
sleep 10

echo "  → Running ADACPSP..."
bash exp_E5_fast_val_adacpsp.sh
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "  ✗ exp_E5_fast_val_adacpsp.sh failed with exit code $EXIT_CODE"
    echo "  Continuing with next experiment..."
fi
echo "  Cooling down (10s)..."
sleep 10

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment: E1_short — Short sequences (<=8k), high GBS baseline"
echo "  max_seq=8192, GBS=64, layers=32"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "  → Running FLEXSP..."
bash exp_E1_short_flexsp.sh
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "  ✗ exp_E1_short_flexsp.sh failed with exit code $EXIT_CODE"
    echo "  Continuing with next experiment..."
fi
echo "  Cooling down (10s)..."
sleep 10

echo "  → Running ADACPSP..."
bash exp_E1_short_adacpsp.sh
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "  ✗ exp_E1_short_adacpsp.sh failed with exit code $EXIT_CODE"
    echo "  Continuing with next experiment..."
fi
echo "  Cooling down (10s)..."
sleep 10

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment: E2_mixed — Mixed lengths (<=32k), moderate GBS"
echo "  max_seq=32768, GBS=32, layers=32"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "  → Running FLEXSP..."
bash exp_E2_mixed_flexsp.sh
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "  ✗ exp_E2_mixed_flexsp.sh failed with exit code $EXIT_CODE"
    echo "  Continuing with next experiment..."
fi
echo "  Cooling down (10s)..."
sleep 10

echo "  → Running ADACPSP..."
bash exp_E2_mixed_adacpsp.sh
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "  ✗ exp_E2_mixed_adacpsp.sh failed with exit code $EXIT_CODE"
    echo "  Continuing with next experiment..."
fi
echo "  Cooling down (10s)..."
sleep 10

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment: E3_long — Long-focused (<=32k), small GBS"
echo "  max_seq=32768, GBS=16, layers=32"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "  → Running FLEXSP..."
bash exp_E3_long_flexsp.sh
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "  ✗ exp_E3_long_flexsp.sh failed with exit code $EXIT_CODE"
    echo "  Continuing with next experiment..."
fi
echo "  Cooling down (10s)..."
sleep 10

echo "  → Running ADACPSP..."
bash exp_E3_long_adacpsp.sh
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "  ✗ exp_E3_long_adacpsp.sh failed with exit code $EXIT_CODE"
    echo "  Continuing with next experiment..."
fi
echo "  Cooling down (10s)..."
sleep 10

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment: E4_extreme — Extreme long (<=64k), minimal GBS"
echo "  max_seq=65536, GBS=8, layers=32"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "  → Running FLEXSP..."
bash exp_E4_extreme_flexsp.sh
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "  ✗ exp_E4_extreme_flexsp.sh failed with exit code $EXIT_CODE"
    echo "  Continuing with next experiment..."
fi
echo "  Cooling down (10s)..."
sleep 10

echo "  → Running ADACPSP..."
bash exp_E4_extreme_adacpsp.sh
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "  ✗ exp_E4_extreme_adacpsp.sh failed with exit code $EXIT_CODE"
    echo "  Continuing with next experiment..."
fi
echo "  Cooling down (10s)..."
sleep 10

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "All experiments completed in ${TOTAL_ELAPSED}s"
echo "Logs saved to ../logs/"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Next: Run analysis script:"
echo "  cd .. && python analyze_experiment_logs.py --log_dir logs/"
