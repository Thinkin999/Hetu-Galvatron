#!/bin/bash
###############################################################################
# 多节点分发器: 在所有节点上启动 run_all_experiments.sh
# ============================================================================
#
# 用法:
#   # 编辑 HOSTFILE 指定所有节点 (每行一个 hostname/IP)
#   # 然后在主节点运行:
#   bash launch_multinode.sh
#
#   # 或者直接传参:
#   HOSTFILE=/path/to/hostfile bash launch_multinode.sh
#
# 前提:
#   1. 所有节点间 SSH 免密互通
#   2. 所有节点上代码路径相同
#   3. 所有节点上 conda 环境相同
###############################################################################

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ======================== 配置 ========================
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hostfile.txt}"
CONDA_ENV="${CONDA_ENV:-lyx-py39torch210}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GPU_CONFIGS_LIST="${GPU_CONFIGS:-}"

# 所有要传递给 run_all_experiments.sh 的环境变量
EXPERIMENT_ENVS="${EXPERIMENT_ENVS:-}"  # 额外的环境变量，如 "MODELS=qwen2.5-7b"

# ======================== 读取节点列表 ========================
if [ ! -f "${HOSTFILE}" ]; then
    echo "ERROR: hostfile not found: ${HOSTFILE}"
    echo "请创建 hostfile，每行一个节点 hostname/IP:"
    echo "  node01"
    echo "  node02"
    echo "  ..."
    exit 1
fi

# 读取节点
HOSTS=()
while IFS= read -r line; do
    line=$(echo "$line" | xargs)  # trim
    [[ -z "$line" || "$line" == \#* ]] && continue
    HOSTS+=("$line")
done < "${HOSTFILE}"

NNODES=${#HOSTS[@]}
MASTER_ADDR="${HOSTS[0]}"
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))

if [ -z "${GPU_CONFIGS_LIST}" ]; then
    GPU_CONFIGS_LIST="${TOTAL_GPUS}"
fi

calc_nodes_for_gpus() {
    local ngpus=$1
    echo $(( (ngpus + NPROC_PER_NODE - 1) / NPROC_PER_NODE ))
}

echo "=========================================="
echo "多节点实验启动器"
echo "=========================================="
echo "节点数:       ${NNODES}"
echo "每节点 GPU:   ${NPROC_PER_NODE}"
echo "总 GPU 数:    ${TOTAL_GPUS}"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "节点列表:     ${HOSTS[*]}"
echo "=========================================="

# ======================== 按总卡数分波次启动 ========================
ALL_SUCCESS=true
for TARGET_GPUS in ${GPU_CONFIGS_LIST}; do
    NEED_NNODES=$(calc_nodes_for_gpus "${TARGET_GPUS}")
    if [ "${NEED_NNODES}" -gt "${NNODES}" ]; then
        echo "WARNING: 目标卡数 ${TARGET_GPUS} 需要 ${NEED_NNODES} 个节点，但 hostfile 只有 ${NNODES} 个节点，跳过"
        ALL_SUCCESS=false
        continue
    fi

    ACTIVE_MASTER_ADDR="${HOSTS[0]}"
    echo ""
    echo "=========================================="
    echo "启动 ${TARGET_GPUS} 卡实验，使用前 ${NEED_NNODES} 个节点"
    echo "=========================================="

    PIDS=()
    for ((i=0; i<NEED_NNODES; i++)); do
        NODE="${HOSTS[$i]}"
        NODE_RANK=$i

        echo "[$(date '+%H:%M:%S')] 在节点 ${NODE} (rank=${NODE_RANK}) 上启动 ${TARGET_GPUS} 卡实验..."

        REMOTE_CMD="
            source ~/.bashrc
            conda activate ${CONDA_ENV} 2>/dev/null || source activate ${CONDA_ENV}
            cd ${SCRIPT_DIR}
            export MASTER_ADDR=${ACTIVE_MASTER_ADDR}
            export MASTER_PORT=${MASTER_PORT}
            export NPROC_PER_NODE=${NPROC_PER_NODE}
            export NODE_RANK=${NODE_RANK}
            export GPU_CONFIGS=${TARGET_GPUS}
            ${EXPERIMENT_ENVS}
            bash run_all_experiments.sh
        "

        if [ "${NODE}" = "$(hostname)" ] || [ "${NODE}" = "localhost" ] || { [ "${NODE}" = "${ACTIVE_MASTER_ADDR}" ] && [ $i -eq 0 ]; }; then
            bash -c "${REMOTE_CMD}" &
            PIDS+=($!)
        else
            ssh -o StrictHostKeyChecking=no "${NODE}" "${REMOTE_CMD}" &
            PIDS+=($!)
        fi
    done

    echo "等待 ${TARGET_GPUS} 卡实验完成..."
    for ((i=0; i<NEED_NNODES; i++)); do
        wait ${PIDS[$i]}
        EXIT_CODE=$?
        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "WARNING: 节点 ${HOSTS[$i]} (rank=${i}) 在 ${TARGET_GPUS} 卡实验中退出码 ${EXIT_CODE}"
            ALL_SUCCESS=false
        else
            echo "OK: 节点 ${HOSTS[$i]} (rank=${i}) 完成 ${TARGET_GPUS} 卡实验"
        fi
    done
done

echo ""
if $ALL_SUCCESS; then
    echo "✓ 所有波次实验完成!"
else
    echo "⚠ 实验已跑完，但部分波次/节点有错误，请检查 log"
fi

