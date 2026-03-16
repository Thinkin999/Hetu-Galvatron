#!/bin/bash
# ============================================================
# Master Runner: 运行全部 72 个实验
# 
# 容错机制:
#   - 每个实验独立运行, 失败不影响后续
#   - OOM / NCCL timeout / hang 自动捕获
#   - 所有结果记录到 summary.log
#
# 用法:
#   bash 02_run_all.sh 2>&1 | tee run_all.log
# ============================================================
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_SCRIPTS_DIR="$SCRIPT_DIR/exp_scripts"
LOG_DIR="$SCRIPT_DIR/logs"
SUMMARY="$LOG_DIR/summary.log"
mkdir -p "$LOG_DIR"

TOTAL=72
PASSED=0
FAILED=0
SKIPPED=0
OOM=0
TIMEOUT=0

echo "============================================" | tee "$SUMMARY"
echo "  AdaCPSP vs FlexSP Experiment Suite" | tee -a "$SUMMARY"
echo "  Total experiments: $TOTAL" | tee -a "$SUMMARY"
echo "  Start: $(date)" | tee -a "$SUMMARY"
echo "============================================" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
printf "%-60s %-12s %-10s\n" "EXPERIMENT" "STATUS" "TIME(s)" | tee -a "$SUMMARY"
printf "%-60s %-12s %-10s\n" "$(printf '%.0s-' {1..60})" "$(printf '%.0s-' {1..12})" "$(printf '%.0s-' {1..10})" | tee -a "$SUMMARY"


# ---- [1/72] qwen2.5-7b_common_crawl_128k_flexsp ----
echo ""
echo ">>> [1/$TOTAL] Running: qwen2.5-7b_common_crawl_128k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_128k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_128k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_128k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_128k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [2/72] qwen2.5-7b_github_128k_flexsp ----
echo ""
echo ">>> [2/$TOTAL] Running: qwen2.5-7b_github_128k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_128k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_128k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_128k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_128k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [3/72] qwen2.5-7b_common_crawl_128k_adacpsp_ur ----
echo ""
echo ">>> [3/$TOTAL] Running: qwen2.5-7b_common_crawl_128k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_128k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_128k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_128k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_128k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [4/72] qwen2.5-7b_github_128k_adacpsp_ur ----
echo ""
echo ">>> [4/$TOTAL] Running: qwen2.5-7b_github_128k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_128k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_128k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_128k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_128k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [5/72] qwen2.5-7b_common_crawl_128k_adacpsp_full ----
echo ""
echo ">>> [5/$TOTAL] Running: qwen2.5-7b_common_crawl_128k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_128k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_128k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_128k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_128k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [6/72] qwen2.5-7b_github_128k_adacpsp_full ----
echo ""
echo ">>> [6/$TOTAL] Running: qwen2.5-7b_github_128k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_128k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_128k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_128k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_128k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [7/72] qwen2.5-7b_common_crawl_256k_flexsp ----
echo ""
echo ">>> [7/$TOTAL] Running: qwen2.5-7b_common_crawl_256k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_256k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_256k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_256k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_256k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [8/72] qwen2.5-7b_github_256k_flexsp ----
echo ""
echo ">>> [8/$TOTAL] Running: qwen2.5-7b_github_256k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_256k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_256k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_256k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_256k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [9/72] qwen2.5-7b_common_crawl_256k_adacpsp_ur ----
echo ""
echo ">>> [9/$TOTAL] Running: qwen2.5-7b_common_crawl_256k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_256k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_256k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_256k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_256k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [10/72] qwen2.5-7b_github_256k_adacpsp_ur ----
echo ""
echo ">>> [10/$TOTAL] Running: qwen2.5-7b_github_256k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_256k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_256k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_256k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_256k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [11/72] qwen2.5-7b_common_crawl_256k_adacpsp_full ----
echo ""
echo ">>> [11/$TOTAL] Running: qwen2.5-7b_common_crawl_256k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_256k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_256k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_256k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_256k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [12/72] qwen2.5-7b_github_256k_adacpsp_full ----
echo ""
echo ">>> [12/$TOTAL] Running: qwen2.5-7b_github_256k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_256k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_256k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_256k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_256k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [13/72] qwen2.5-7b_common_crawl_384k_flexsp ----
echo ""
echo ">>> [13/$TOTAL] Running: qwen2.5-7b_common_crawl_384k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_384k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_384k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_384k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_384k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [14/72] qwen2.5-7b_github_384k_flexsp ----
echo ""
echo ">>> [14/$TOTAL] Running: qwen2.5-7b_github_384k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_384k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_384k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_384k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_384k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [15/72] qwen2.5-7b_common_crawl_384k_adacpsp_ur ----
echo ""
echo ">>> [15/$TOTAL] Running: qwen2.5-7b_common_crawl_384k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_384k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_384k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_384k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_384k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [16/72] qwen2.5-7b_github_384k_adacpsp_ur ----
echo ""
echo ">>> [16/$TOTAL] Running: qwen2.5-7b_github_384k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_384k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_384k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_384k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_384k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [17/72] qwen2.5-7b_common_crawl_384k_adacpsp_full ----
echo ""
echo ">>> [17/$TOTAL] Running: qwen2.5-7b_common_crawl_384k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_384k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_384k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_384k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_384k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [18/72] qwen2.5-7b_github_384k_adacpsp_full ----
echo ""
echo ">>> [18/$TOTAL] Running: qwen2.5-7b_github_384k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_384k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_384k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_384k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_384k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [19/72] qwen2.5-7b_common_crawl_512k_flexsp ----
echo ""
echo ">>> [19/$TOTAL] Running: qwen2.5-7b_common_crawl_512k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_512k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_512k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_512k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_512k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [20/72] qwen2.5-7b_github_512k_flexsp ----
echo ""
echo ">>> [20/$TOTAL] Running: qwen2.5-7b_github_512k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_512k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_512k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_512k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_512k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [21/72] qwen2.5-7b_common_crawl_512k_adacpsp_ur ----
echo ""
echo ">>> [21/$TOTAL] Running: qwen2.5-7b_common_crawl_512k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_512k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_512k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_512k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_512k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [22/72] qwen2.5-7b_github_512k_adacpsp_ur ----
echo ""
echo ">>> [22/$TOTAL] Running: qwen2.5-7b_github_512k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_512k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_512k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_512k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_512k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [23/72] qwen2.5-7b_common_crawl_512k_adacpsp_full ----
echo ""
echo ">>> [23/$TOTAL] Running: qwen2.5-7b_common_crawl_512k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_common_crawl_512k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_common_crawl_512k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_common_crawl_512k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_common_crawl_512k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [24/72] qwen2.5-7b_github_512k_adacpsp_full ----
echo ""
echo ">>> [24/$TOTAL] Running: qwen2.5-7b_github_512k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-7b_github_512k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-7b_github_512k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-7b_github_512k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-7b_github_512k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [25/72] qwen2.5-14b_common_crawl_128k_flexsp ----
echo ""
echo ">>> [25/$TOTAL] Running: qwen2.5-14b_common_crawl_128k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_128k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_128k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_128k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_128k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [26/72] qwen2.5-14b_github_128k_flexsp ----
echo ""
echo ">>> [26/$TOTAL] Running: qwen2.5-14b_github_128k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_128k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_128k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_128k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_128k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [27/72] qwen2.5-14b_common_crawl_128k_adacpsp_ur ----
echo ""
echo ">>> [27/$TOTAL] Running: qwen2.5-14b_common_crawl_128k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_128k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_128k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_128k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_128k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [28/72] qwen2.5-14b_github_128k_adacpsp_ur ----
echo ""
echo ">>> [28/$TOTAL] Running: qwen2.5-14b_github_128k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_128k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_128k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_128k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_128k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [29/72] qwen2.5-14b_common_crawl_128k_adacpsp_full ----
echo ""
echo ">>> [29/$TOTAL] Running: qwen2.5-14b_common_crawl_128k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_128k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_128k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_128k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_128k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [30/72] qwen2.5-14b_github_128k_adacpsp_full ----
echo ""
echo ">>> [30/$TOTAL] Running: qwen2.5-14b_github_128k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_128k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_128k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_128k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_128k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [31/72] qwen2.5-14b_common_crawl_256k_flexsp ----
echo ""
echo ">>> [31/$TOTAL] Running: qwen2.5-14b_common_crawl_256k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_256k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_256k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_256k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_256k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [32/72] qwen2.5-14b_github_256k_flexsp ----
echo ""
echo ">>> [32/$TOTAL] Running: qwen2.5-14b_github_256k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_256k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_256k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_256k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_256k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [33/72] qwen2.5-14b_common_crawl_256k_adacpsp_ur ----
echo ""
echo ">>> [33/$TOTAL] Running: qwen2.5-14b_common_crawl_256k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_256k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_256k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_256k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_256k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [34/72] qwen2.5-14b_github_256k_adacpsp_ur ----
echo ""
echo ">>> [34/$TOTAL] Running: qwen2.5-14b_github_256k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_256k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_256k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_256k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_256k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [35/72] qwen2.5-14b_common_crawl_256k_adacpsp_full ----
echo ""
echo ">>> [35/$TOTAL] Running: qwen2.5-14b_common_crawl_256k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_256k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_256k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_256k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_256k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [36/72] qwen2.5-14b_github_256k_adacpsp_full ----
echo ""
echo ">>> [36/$TOTAL] Running: qwen2.5-14b_github_256k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_256k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_256k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_256k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_256k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [37/72] qwen2.5-14b_common_crawl_384k_flexsp ----
echo ""
echo ">>> [37/$TOTAL] Running: qwen2.5-14b_common_crawl_384k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_384k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_384k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_384k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_384k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [38/72] qwen2.5-14b_github_384k_flexsp ----
echo ""
echo ">>> [38/$TOTAL] Running: qwen2.5-14b_github_384k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_384k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_384k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_384k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_384k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [39/72] qwen2.5-14b_common_crawl_384k_adacpsp_ur ----
echo ""
echo ">>> [39/$TOTAL] Running: qwen2.5-14b_common_crawl_384k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_384k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_384k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_384k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_384k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [40/72] qwen2.5-14b_github_384k_adacpsp_ur ----
echo ""
echo ">>> [40/$TOTAL] Running: qwen2.5-14b_github_384k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_384k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_384k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_384k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_384k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [41/72] qwen2.5-14b_common_crawl_384k_adacpsp_full ----
echo ""
echo ">>> [41/$TOTAL] Running: qwen2.5-14b_common_crawl_384k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_384k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_384k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_384k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_384k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [42/72] qwen2.5-14b_github_384k_adacpsp_full ----
echo ""
echo ">>> [42/$TOTAL] Running: qwen2.5-14b_github_384k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_384k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_384k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_384k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_384k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [43/72] qwen2.5-14b_common_crawl_512k_flexsp ----
echo ""
echo ">>> [43/$TOTAL] Running: qwen2.5-14b_common_crawl_512k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_512k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_512k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_512k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_512k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [44/72] qwen2.5-14b_github_512k_flexsp ----
echo ""
echo ">>> [44/$TOTAL] Running: qwen2.5-14b_github_512k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_512k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_512k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_512k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_512k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [45/72] qwen2.5-14b_common_crawl_512k_adacpsp_ur ----
echo ""
echo ">>> [45/$TOTAL] Running: qwen2.5-14b_common_crawl_512k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_512k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_512k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_512k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_512k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [46/72] qwen2.5-14b_github_512k_adacpsp_ur ----
echo ""
echo ">>> [46/$TOTAL] Running: qwen2.5-14b_github_512k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_512k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_512k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_512k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_512k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [47/72] qwen2.5-14b_common_crawl_512k_adacpsp_full ----
echo ""
echo ">>> [47/$TOTAL] Running: qwen2.5-14b_common_crawl_512k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_common_crawl_512k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_common_crawl_512k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_common_crawl_512k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_common_crawl_512k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [48/72] qwen2.5-14b_github_512k_adacpsp_full ----
echo ""
echo ">>> [48/$TOTAL] Running: qwen2.5-14b_github_512k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-14b_github_512k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-14b_github_512k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-14b_github_512k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-14b_github_512k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [49/72] qwen2.5-32b_common_crawl_128k_flexsp ----
echo ""
echo ">>> [49/$TOTAL] Running: qwen2.5-32b_common_crawl_128k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_128k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_128k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_128k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_128k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [50/72] qwen2.5-32b_github_128k_flexsp ----
echo ""
echo ">>> [50/$TOTAL] Running: qwen2.5-32b_github_128k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_128k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_128k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_128k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_128k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [51/72] qwen2.5-32b_common_crawl_128k_adacpsp_ur ----
echo ""
echo ">>> [51/$TOTAL] Running: qwen2.5-32b_common_crawl_128k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_128k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_128k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_128k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_128k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [52/72] qwen2.5-32b_github_128k_adacpsp_ur ----
echo ""
echo ">>> [52/$TOTAL] Running: qwen2.5-32b_github_128k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_128k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_128k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_128k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_128k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [53/72] qwen2.5-32b_common_crawl_128k_adacpsp_full ----
echo ""
echo ">>> [53/$TOTAL] Running: qwen2.5-32b_common_crawl_128k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_128k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_128k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_128k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_128k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [54/72] qwen2.5-32b_github_128k_adacpsp_full ----
echo ""
echo ">>> [54/$TOTAL] Running: qwen2.5-32b_github_128k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_128k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_128k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_128k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_128k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [55/72] qwen2.5-32b_common_crawl_256k_flexsp ----
echo ""
echo ">>> [55/$TOTAL] Running: qwen2.5-32b_common_crawl_256k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_256k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_256k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_256k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_256k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [56/72] qwen2.5-32b_github_256k_flexsp ----
echo ""
echo ">>> [56/$TOTAL] Running: qwen2.5-32b_github_256k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_256k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_256k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_256k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_256k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [57/72] qwen2.5-32b_common_crawl_256k_adacpsp_ur ----
echo ""
echo ">>> [57/$TOTAL] Running: qwen2.5-32b_common_crawl_256k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_256k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_256k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_256k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_256k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [58/72] qwen2.5-32b_github_256k_adacpsp_ur ----
echo ""
echo ">>> [58/$TOTAL] Running: qwen2.5-32b_github_256k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_256k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_256k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_256k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_256k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [59/72] qwen2.5-32b_common_crawl_256k_adacpsp_full ----
echo ""
echo ">>> [59/$TOTAL] Running: qwen2.5-32b_common_crawl_256k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_256k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_256k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_256k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_256k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [60/72] qwen2.5-32b_github_256k_adacpsp_full ----
echo ""
echo ">>> [60/$TOTAL] Running: qwen2.5-32b_github_256k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_256k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_256k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_256k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_256k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [61/72] qwen2.5-32b_common_crawl_384k_flexsp ----
echo ""
echo ">>> [61/$TOTAL] Running: qwen2.5-32b_common_crawl_384k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_384k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_384k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_384k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_384k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [62/72] qwen2.5-32b_github_384k_flexsp ----
echo ""
echo ">>> [62/$TOTAL] Running: qwen2.5-32b_github_384k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_384k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_384k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_384k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_384k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [63/72] qwen2.5-32b_common_crawl_384k_adacpsp_ur ----
echo ""
echo ">>> [63/$TOTAL] Running: qwen2.5-32b_common_crawl_384k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_384k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_384k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_384k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_384k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [64/72] qwen2.5-32b_github_384k_adacpsp_ur ----
echo ""
echo ">>> [64/$TOTAL] Running: qwen2.5-32b_github_384k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_384k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_384k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_384k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_384k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [65/72] qwen2.5-32b_common_crawl_384k_adacpsp_full ----
echo ""
echo ">>> [65/$TOTAL] Running: qwen2.5-32b_common_crawl_384k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_384k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_384k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_384k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_384k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [66/72] qwen2.5-32b_github_384k_adacpsp_full ----
echo ""
echo ">>> [66/$TOTAL] Running: qwen2.5-32b_github_384k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_384k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_384k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_384k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_384k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [67/72] qwen2.5-32b_common_crawl_512k_flexsp ----
echo ""
echo ">>> [67/$TOTAL] Running: qwen2.5-32b_common_crawl_512k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_512k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_512k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_512k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_512k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [68/72] qwen2.5-32b_github_512k_flexsp ----
echo ""
echo ">>> [68/$TOTAL] Running: qwen2.5-32b_github_512k_flexsp"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_512k_flexsp.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_512k_flexsp.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_512k_flexsp.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_512k_flexsp" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [69/72] qwen2.5-32b_common_crawl_512k_adacpsp_ur ----
echo ""
echo ">>> [69/$TOTAL] Running: qwen2.5-32b_common_crawl_512k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_512k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_512k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_512k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_512k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [70/72] qwen2.5-32b_github_512k_adacpsp_ur ----
echo ""
echo ">>> [70/$TOTAL] Running: qwen2.5-32b_github_512k_adacpsp_ur"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_512k_adacpsp_ur.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_512k_adacpsp_ur.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_512k_adacpsp_ur.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_512k_adacpsp_ur" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [71/72] qwen2.5-32b_common_crawl_512k_adacpsp_full ----
echo ""
echo ">>> [71/$TOTAL] Running: qwen2.5-32b_common_crawl_512k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_common_crawl_512k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_common_crawl_512k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_common_crawl_512k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_common_crawl_512k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

# ---- [72/72] qwen2.5-32b_github_512k_adacpsp_full ----
echo ""
echo ">>> [72/$TOTAL] Running: qwen2.5-32b_github_512k_adacpsp_full"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/qwen2.5-32b_github_512k_adacpsp_full.sh"
EXP_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXP_EXIT -eq 0 ]; then
    STATUS="SUCCESS"
    PASSED=$((PASSED + 1))
elif [ $EXP_EXIT -eq 124 ]; then
    STATUS="TIMEOUT"
    TIMEOUT=$((TIMEOUT + 1))
    FAILED=$((FAILED + 1))
else
    # 检查 OOM
    if [ -f "$LOG_DIR/qwen2.5-32b_github_512k_adacpsp_full.log" ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_DIR/qwen2.5-32b_github_512k_adacpsp_full.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\n" "qwen2.5-32b_github_512k_adacpsp_full" "$STATUS" "${ELAPSED}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5

echo "" | tee -a "$SUMMARY"
echo "============================================" | tee -a "$SUMMARY"
echo "  Experiment Suite Complete!" | tee -a "$SUMMARY"
echo "  End: $(date)" | tee -a "$SUMMARY"
echo "  Total: $TOTAL" | tee -a "$SUMMARY"
echo "  Passed: $PASSED" | tee -a "$SUMMARY"
echo "  Failed: $FAILED (OOM: $OOM, Timeout: $TIMEOUT)" | tee -a "$SUMMARY"
echo "============================================" | tee -a "$SUMMARY"
echo ""
echo "Log files in: $LOG_DIR/"
echo "Summary in: $SUMMARY"
echo "Run analysis: python analyze_results.py --log-dir $LOG_DIR"
