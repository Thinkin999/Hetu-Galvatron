#!/bin/bash
###############################################################################
# AdaCPSP vs FlexSP 全量实验统一启动脚本
# ============================================================================
#
# 功能:
#   1. 自动遍历所有 (模型, 序列长度, GBS, 策略) 组合
#   2. 支持 64/32/16 卡自动切换 (通过 NNODES × NPROC_PER_NODE)
#   3. 单个实验失败 (OOM / hang / 其他错误) 不影响其他实验继续执行
#   4. 每个实验独立的 log 文件，方便后续分析
#   5. 自动超时保护 (防止 hang 住)
#
# 用法:
#   # 在主节点运行:
#   bash run_all_experiments.sh
#
#   # 或只跑某个模型:
#   MODELS="qwen2.5-7b" bash run_all_experiments.sh
#
#   # 或只跑某个卡数:
#   GPU_CONFIGS="64" bash run_all_experiments.sh
#
#   # 自定义 master 地址:
#   MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash run_all_experiments.sh
###############################################################################

set -o pipefail  # 不用 set -e，让脚本遇到失败继续

# ======================== 可覆盖的全局变量 ========================
# 集群配置
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"          # 每节点 GPU 数

# 实验维度 (空格分隔，可通过环境变量覆盖)
MODELS="${MODELS:-qwen2.5-7b qwen2.5-14b qwen2.5-32b}"
SEQ_LENGTHS_K="${SEQ_LENGTHS_K:-128 256 384 512}"
GBS_LIST="${GBS_LIST:-auto}"                    # "auto" = 按模型自动选，或指定如 "64 128 256"
GPU_CONFIGS="${GPU_CONFIGS:-64 32 16}"          # 总卡数列表

# 策略维度
# adacpsp        = 完整 AdaCPSP (ulysses + ring + usp)
# flexsp         = FlexSP (只用 ulysses 的 adacpsp)
# ring_only      = 只用 ring attention
# ulysses_only   = 只用 ulysses (无 solver，固定并行度)
STRATEGIES="${STRATEGIES:-adacpsp flexsp}"

# Benchmark 参数
NUM_ITERS="${NUM_ITERS:-20}"                    # 真正计时的 iteration 数
WARMUP_ITERS="${WARMUP_ITERS:-5}"               # 前几个 iter 做 warmup
EPOCHS="${EPOCHS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"       # 每个实验最大超时时间 (秒)
MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-90}"        # 显存限制 (H20 = 96GB, 留 6GB 余量)

# 数据集
DATASET="${DATASET:-wikipedia}"

# 输出
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$(dirname "$0")/results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${EXPERIMENT_DIR}/${TIMESTAMP}"

# 代码路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_DIR}/galvatron/models/varlen_llama_hf/train_dist_adacpsp.py"

###############################################################################
# 辅助函数
###############################################################################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

calc_profile_end_iter() {
    local warmup=$1
    local measured=$2
    echo $((warmup + measured))
}

# 根据模型和卡数自动决定 GBS
auto_gbs() {
    local model=$1
    local ngpus=$2
    local seqlen_k=$3
    # 经验公式: GBS 要能被 ngpus 整除，且不能太大导致 OOM
    # 基础 GBS = ngpus (每卡至少 1 个 micro-batch)
    case "${model}" in
        qwen2.5-7b)
            if [ "$seqlen_k" -le 128 ]; then echo $((ngpus * 2))
            elif [ "$seqlen_k" -le 256 ]; then echo $((ngpus))
            else echo $((ngpus))
            fi
            ;;
        qwen2.5-14b)
            if [ "$seqlen_k" -le 128 ]; then echo $((ngpus))
            else echo $((ngpus))
            fi
            ;;
        qwen2.5-32b)
            echo $((ngpus))
            ;;
        *)
            echo $((ngpus))
            ;;
    esac
}

# 根据总卡数计算节点数
calc_nnodes() {
    local ngpus=$1
    echo $(( (ngpus + NPROC_PER_NODE - 1) / NPROC_PER_NODE ))
}

# 策略 → 命令行参数
strategy_to_args() {
    local strategy=$1
    case "${strategy}" in
        adacpsp)
            # 完整 AdaCPSP: ulysses + ring + usp
            echo "--use-adaCPSP --use-packing --adaCPSP-attn-types ulysses ring usp"
            ;;
        flexsp)
            # FlexSP: 用只有 ulysses 的 adaCPSP 来代替
            echo "--use-adaCPSP --use-packing --adaCPSP-attn-types ulysses"
            ;;
        ring_only)
            # 只用 ring attention
            echo "--use-adaCPSP --use-packing --adaCPSP-attn-types ring"
            ;;
        ulysses_ring)
            # ulysses + ring (不含 usp)
            echo "--use-adaCPSP --use-packing --adaCPSP-attn-types ulysses ring"
            ;;
        *)
            log "ERROR: Unknown strategy: ${strategy}"
            return 1
            ;;
    esac
}

# 检查 model_size 是否在 arguments.py 的 choices 里
# 如果不在，需要添加（qwen2.5-14b/32b 可能没有）
validate_model_size() {
    local model=$1
    # config_utils.py 的 path_dict 有这些模型就能用
    python -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}')
from galvatron.models.varlen_llama_hf.meta_configs.config_utils import path_dict
assert '${model}' in path_dict, f'${model} not in path_dict: {list(path_dict.keys())}'
print('OK: ${model}')
" 2>&1
}

###############################################################################
# 主流程
###############################################################################

mkdir -p "${RESULT_DIR}"

# 写入实验配置
cat > "${RESULT_DIR}/experiment_config.txt" << EOF
=== AdaCPSP Experiment Configuration ===
Timestamp:        ${TIMESTAMP}
Models:           ${MODELS}
Seq Lengths (K):  ${SEQ_LENGTHS_K}
GBS:              ${GBS_LIST}
GPU Configs:      ${GPU_CONFIGS}
Strategies:       ${STRATEGIES}
Num Iters:        ${NUM_ITERS}
Warmup Iters:     ${WARMUP_ITERS}
Timeout (s):      ${TIMEOUT_SECONDS}
Memory Limit:     ${MEMORY_LIMIT_GB} GB
Dataset:          ${DATASET}
Master:           ${MASTER_ADDR}:${MASTER_PORT}
Nproc/Node:       ${NPROC_PER_NODE}
Train Script:     ${TRAIN_SCRIPT}
EOF

log "=========================================="
log "AdaCPSP 全量实验启动"
log "=========================================="
log "结果目录: ${RESULT_DIR}"
log "模型: ${MODELS}"
log "序列长度(K): ${SEQ_LENGTHS_K}"
log "总卡数: ${GPU_CONFIGS}"
log "策略: ${STRATEGIES}"
log "=========================================="

# 统计
TOTAL=0
PASSED=0
FAILED=0
TIMEOUT_COUNT=0
SKIPPED=0

# 结果汇总 CSV
SUMMARY_CSV="${RESULT_DIR}/summary.csv"
echo "model,ngpus,seqlen_k,gbs,strategy,status,wall_time_s,avg_iter_time_ms,throughput_tokens_per_s,peak_memory_gb,log_file" > "${SUMMARY_CSV}"

# ======================== 遍历所有组合 ========================
for ngpus in ${GPU_CONFIGS}; do
    NNODES=$(calc_nnodes ${ngpus})
    
    for model in ${MODELS}; do
        # 验证模型
        if ! validate_model_size "${model}" > /dev/null 2>&1; then
            log "SKIP: model ${model} not found in config"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        
        for seqlen_k in ${SEQ_LENGTHS_K}; do
            seqlen=$((seqlen_k * 1024))
            
            # 确定 GBS
            if [ "${GBS_LIST}" = "auto" ]; then
                gbs_values=$(auto_gbs "${model}" "${ngpus}" "${seqlen_k}")
            else
                gbs_values="${GBS_LIST}"
            fi
            
            for gbs in ${gbs_values}; do
                for strategy in ${STRATEGIES}; do
                    TOTAL=$((TOTAL + 1))
                    
                    # 实验标识
                    EXP_NAME="${model}_gpu${ngpus}_seq${seqlen_k}k_gbs${gbs}_${strategy}"
                    EXP_LOG="${RESULT_DIR}/${EXP_NAME}.log"
                    
                    log "────────────────────────────────────────"
                    log "[${TOTAL}] 启动: ${EXP_NAME}"
                    log "  模型=${model} 卡数=${ngpus} 序列=${seqlen_k}K GBS=${gbs} 策略=${strategy}"
                    
                    # 生成策略参数
                    STRATEGY_ARGS=$(strategy_to_args "${strategy}")
                    if [ $? -ne 0 ]; then
                        log "  ERROR: invalid strategy, skipping"
                        SKIPPED=$((SKIPPED + 1))
                        echo "${model},${ngpus},${seqlen_k},${gbs},${strategy},SKIPPED,0,0,0,0,${EXP_LOG}" >> "${SUMMARY_CSV}"
                        continue
                    fi
                    
                    PROFILE_END_ITER=$(calc_profile_end_iter "${WARMUP_ITERS}" "${NUM_ITERS}")

                    # 构建 torchrun 命令
                    CMD="torchrun \
                        --nproc_per_node=${NPROC_PER_NODE} \
                        --nnodes=${NNODES} \
                        --node_rank=\${NODE_RANK:-0} \
                        --master_addr=${MASTER_ADDR} \
                        --master_port=${MASTER_PORT} \
                        ${TRAIN_SCRIPT} \
                        --model_size ${model} \
                        --set_seqlen_manually 1 \
                        -s ${seqlen} \
                        --global_train_batch_size ${gbs} \
                        --epochs ${EPOCHS} \
                        --pp_deg 1 \
                        --global_tp_deg 1 \
                        --global_cp_deg 1 \
                        --default_dp_type zero3 \
                        --mixed_precision bf16 \
                        --use-flash-attn \
                        --dataset ${DATASET} \
                        --memory-limit-gb ${MEMORY_LIMIT_GB} \
                        --profile 1 \
                        --profile_start_iter ${WARMUP_ITERS} \
                        --profile_end_iter ${PROFILE_END_ITER} \
                        --exit_after_profiling 1 \
                        ${STRATEGY_ARGS}"
                    
                    # 写入命令到 log
                    echo "=== Command ===" > "${EXP_LOG}"
                    echo "${CMD}" >> "${EXP_LOG}"
                    echo "=== Start: $(date) ===" >> "${EXP_LOG}"
                    echo "" >> "${EXP_LOG}"
                    
                    # 执行 (带超时保护)
                    START_TIME=$(date +%s)
                    timeout ${TIMEOUT_SECONDS} bash -c "${CMD}" >> "${EXP_LOG}" 2>&1
                    EXIT_CODE=$?
                    END_TIME=$(date +%s)
                    WALL_TIME=$((END_TIME - START_TIME))
                    
                    # 写入结束信息
                    echo "" >> "${EXP_LOG}"
                    echo "=== End: $(date) ===" >> "${EXP_LOG}"
                    echo "=== Exit Code: ${EXIT_CODE} ===" >> "${EXP_LOG}"
                    echo "=== Wall Time: ${WALL_TIME}s ===" >> "${EXP_LOG}"
                    
                    # 判断结果
                    if [ ${EXIT_CODE} -eq 0 ]; then
                        STATUS="PASS"
                        PASSED=$((PASSED + 1))
                        log "  ✓ PASS (${WALL_TIME}s)"
                    elif [ ${EXIT_CODE} -eq 124 ]; then
                        STATUS="TIMEOUT"
                        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
                        log "  ✗ TIMEOUT after ${TIMEOUT_SECONDS}s"
                        # 清理可能残留的进程
                        pkill -f "${EXP_NAME}" 2>/dev/null || true
                    else
                        STATUS="FAIL(${EXIT_CODE})"
                        FAILED=$((FAILED + 1))
                        # 检测 OOM
                        if grep -q "OutOfMemoryError\|CUDA out of memory\|OOM" "${EXP_LOG}" 2>/dev/null; then
                            STATUS="OOM"
                            log "  ✗ OOM (${WALL_TIME}s)"
                        else
                            log "  ✗ FAIL exit=${EXIT_CODE} (${WALL_TIME}s)"
                        fi
                    fi
                    
                    # 从 log 中提取关键指标
                    AVG_ITER_S=$(grep -oP 'Average iteration time is:\s*\K[\d.]+' "${EXP_LOG}" 2>/dev/null | tail -1 || echo "0")
                    AVG_ITER_MS=$(awk -v s="${AVG_ITER_S}" 'BEGIN { printf "%.3f", s * 1000 }')
                    THROUGHPUT=$(grep -oP 'Throughput.*?:\s*\K[\d.]+' "${EXP_LOG}" 2>/dev/null | tail -1 || echo "0")
                    PEAK_MEM=$(grep -oP 'Peak memory.*?:\s*\K[\d.]+' "${EXP_LOG}" 2>/dev/null | tail -1 || echo "0")
                    
                    # 写入汇总
                    echo "${model},${ngpus},${seqlen_k},${gbs},${strategy},${STATUS},${WALL_TIME},${AVG_ITER_MS},${THROUGHPUT},${PEAK_MEM},${EXP_LOG}" >> "${SUMMARY_CSV}"
                    
                    # 实验间隔 (让 GPU 冷却、NCCL 状态清理)
                    sleep 5
                    
                done  # strategy
            done  # gbs
        done  # seqlen_k
    done  # model
done  # ngpus

# ======================== 最终报告 ========================
log ""
log "=========================================="
log "实验全部完成!"
log "=========================================="
log "总计: ${TOTAL}"
log "通过: ${PASSED}"
log "失败: ${FAILED}"
log "超时: ${TIMEOUT_COUNT}"
log "跳过: ${SKIPPED}"
log ""
log "结果目录: ${RESULT_DIR}"
log "汇总 CSV: ${SUMMARY_CSV}"
log "=========================================="

# 打印 CSV 汇总表
log ""
log "=== 结果汇总 ==="
column -t -s',' "${SUMMARY_CSV}" 2>/dev/null || cat "${SUMMARY_CSV}"

