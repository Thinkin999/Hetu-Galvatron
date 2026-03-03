#!/bin/bash
#
# Benchmark: FlexSP with Ring Attention vs Ulysses-only
# 针对 Qwen2.5 系列模型，在 32 GPU 集群上进行测试
#
# SP Size 可选范围：1, 2, 4, 8, 16, 32
#
# Usage:
#   ./benchmark_ring_vs_ulysses.sh
#

set -e

# Change to script directory
cd "$(dirname "$0")/.."

# ============== 集群配置 ==============
CLUSTER_SIZE=32                    # 固定 32 GPU
MEMORY_LIMIT_GB=40                 # 每 GPU 显存限制

# ============== 模型选择 ==============
# Qwen2.5 系列模型
MODELS="qwen2.5-1.5b qwen2.5-3b qwen2.5-7b qwen2.5-72b"
# 也可以只测试部分模型：
# MODELS="qwen2.5-7b"

# ============== 数据集 ==============
DATASETS="github common_crawl wikipedia"

# ============== 序列长度限制 (K) ==============
# 参考 solver_all.sh 的设置
SEQ_LIMITS="32 64 128 192"

# ============== Global Batch Size ==============
GBS="256 512 1024"

# ============== 方法选择 ==============
# flexSP: FlexSP 自动求解
# adaptive: Homo-SP 自适应选择最优 SP
# static: 固定 SP = cluster_size
METHODS="flexSP"

# ============== Benchmark 设置 ==============
ITER_NUM=30                        # 每个配置的迭代次数
START_ITER=3                       # 起始迭代
BUCKET_NUM=16                      # FlexSP 桶数量
TIME_LIMIT=10                      # SCIP 求解时间限制

# ============== Ring Attention 设置 ==============
RING_OVERLAP_EFFICIENCY=0.15       # Ring 通信与计算重叠效率

# ============== 带宽配置 ==============
BANDWIDTH="ib_4x"                  # 可选: ib_4x, nvlink

# ============== 输出设置 ==============
OUTPUT_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="${OUTPUT_DIR}/benchmark_qwen25_${TIMESTAMP}.csv"
OUTPUT_LOG="${OUTPUT_DIR}/benchmark_qwen25_${TIMESTAMP}.log"

# ============== 创建输出目录 ==============
mkdir -p ${OUTPUT_DIR}

# ============== 打印配置 ==============
echo "=================================================================="
echo "FlexSP Benchmark: Ring Attention vs Ulysses"
echo "=================================================================="
echo "Cluster Size:     ${CLUSTER_SIZE} GPUs"
echo "Memory Limit:     ${MEMORY_LIMIT_GB} GB/GPU"
echo "Models:           ${MODELS}"
echo "Datasets:         ${DATASETS}"
echo "Seq Limits:       ${SEQ_LIMITS}K"
echo "Global Batch:     ${GBS}"
echo "Methods:          ${METHODS}"
echo "Ring Overlap:     ${RING_OVERLAP_EFFICIENCY}"
echo "Output CSV:       ${OUTPUT_CSV}"
echo "Output Log:       ${OUTPUT_LOG}"
echo "=================================================================="
echo ""

# ============== 运行 Benchmark ==============
python benchmark_ring_vs_ulysses.py \
    --output "${OUTPUT_CSV}" \
    --cluster_size ${CLUSTER_SIZE} \
    --memory_limit_gb ${MEMORY_LIMIT_GB} \
    --models ${MODELS} \
    --datasets ${DATASETS} \
    --seq_limit_k ${SEQ_LIMITS} \
    --global_batch_size ${GBS} \
    --methods ${METHODS} \
    --iter_num ${ITER_NUM} \
    --start_iter ${START_ITER} \
    --bucket_num ${BUCKET_NUM} \
    --ring_overlap_efficiency ${RING_OVERLAP_EFFICIENCY} \
    --bandwidth ${BANDWIDTH} \
    2>&1 | tee "${OUTPUT_LOG}"

# ============== 生成汇总报告 ==============
echo ""
echo "=================================================================="
echo "Generating Summary Report..."
echo "=================================================================="

SUMMARY_FILE="${OUTPUT_DIR}/summary_qwen25_${TIMESTAMP}.txt"

cat > "${SUMMARY_FILE}" << EOF
FlexSP Benchmark Summary Report
===============================
Generated: $(date)

Configuration:
  Cluster Size:     ${CLUSTER_SIZE} GPUs
  Memory Limit:     ${MEMORY_LIMIT_GB} GB/GPU
  Models:           ${MODELS}
  Datasets:         ${DATASETS}
  Seq Limits:       ${SEQ_LIMITS}K
  Global Batch:     ${GBS}
  Methods:          ${METHODS}
  Ring Overlap:     ${RING_OVERLAP_EFFICIENCY}
  Bandwidth:        ${BANDWIDTH}

Results:
EOF

# 使用 awk 从 CSV 提取统计数据
if [ -f "${OUTPUT_CSV}" ]; then
    echo "" >> "${SUMMARY_FILE}"
    echo "Per-Model Statistics:" >> "${SUMMARY_FILE}"
    echo "---------------------" >> "${SUMMARY_FILE}"
    
    awk -F',' 'NR>1 && $12>0 {
        model=$1
        speedup=$12
        improvement=$13
        sum_speedup[model] += speedup
        sum_improvement[model] += improvement
        count[model]++
        if (speedup > max_speedup[model] || max_speedup[model] == "") max_speedup[model] = speedup
        if (speedup < min_speedup[model] || min_speedup[model] == "") min_speedup[model] = speedup
    }
    END {
        for (m in sum_speedup) {
            avg_speedup = sum_speedup[m] / count[m]
            avg_improvement = sum_improvement[m] / count[m]
            printf "\n%s:\n", m
            printf "  Samples:          %d\n", count[m]
            printf "  Avg Speedup:      %.3fx\n", avg_speedup
            printf "  Max Speedup:      %.3fx\n", max_speedup[m]
            printf "  Min Speedup:      %.3fx\n", min_speedup[m]
            printf "  Avg Improvement:  %.2f%%\n", avg_improvement
        }
    }' "${OUTPUT_CSV}" >> "${SUMMARY_FILE}"
    
    echo "" >> "${SUMMARY_FILE}"
    echo "Overall Statistics:" >> "${SUMMARY_FILE}"
    echo "-------------------" >> "${SUMMARY_FILE}"
    
    awk -F',' 'NR>1 && $12>0 {
        speedup=$12
        total += speedup
        count++
        if (speedup > max || max == "") max = speedup
        if (speedup < min || min == "") min = speedup
    }
    END {
        if (count > 0) {
            printf "  Total Samples:    %d\n", count
            printf "  Avg Speedup:      %.3fx\n", total/count
            printf "  Max Speedup:      %.3fx\n", max
            printf "  Min Speedup:      %.3fx\n", min
        }
    }' "${OUTPUT_CSV}" >> "${SUMMARY_FILE}"
fi

echo ""
echo "Summary saved to: ${SUMMARY_FILE}"
cat "${SUMMARY_FILE}"

echo ""
echo "=================================================================="
echo "Benchmark Complete!"
echo "  CSV Results: ${OUTPUT_CSV}"
echo "  Log File:    ${OUTPUT_LOG}"
echo "  Summary:     ${SUMMARY_FILE}"
echo "=================================================================="
