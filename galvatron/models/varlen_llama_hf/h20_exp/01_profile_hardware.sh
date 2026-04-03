#!/bin/bash
# ============================================================
# Phase 0.2: H20 硬件 Profiling
# 分三步: (1)注意力计算, (2)All-to-All, (3)P2P Ring
#
# 注意力 profiling 只需单卡; 通信 profiling 需要全部 64 卡
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/profile"
CONFIGS_DIR="$BASE_DIR/configs"
mkdir -p "$LOG_DIR" "$CONFIGS_DIR"

# ---- 集群参数 (根据实际环境修改) ----
NUM_NODES=${NUM_NODES:-8}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "  Hardware Profiling for AdaCPSP"
echo "  Nodes: $NUM_NODES × $NUM_GPUS_PER_NODE GPUs = $TOTAL_GPUS total"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  Node Rank: $NODE_RANK"
echo "  Timestamp: $TIMESTAMP"
echo "============================================"

# ================================================================
# Step 1: Attention Profiling (单卡, 只在 master 上运行)
# ================================================================
if [ "$NODE_RANK" -eq 0 ]; then
    echo ""
    echo ">>> [Step 1/4] Attention Profiling (single GPU)"
    echo "    Profiling for all 3 model configs..."
    
    for MODEL_CFG in "qwen2.5-7b:28:4:128" "qwen2.5-14b:40:8:128" "qwen2.5-32b:40:8:128"; do
        IFS=':' read -r MODEL_NAME N_HEADS N_KV_HEADS HEAD_DIM <<< "$MODEL_CFG"
        echo ""
        echo "    --- $MODEL_NAME (heads=$N_HEADS, kv_heads=$N_KV_HEADS, head_dim=$HEAD_DIM) ---"
        
        python3 "$BASE_DIR/profile_and_validate.py" \
            --mode attention \
            --n_heads "$N_HEADS" \
            --n_kv_heads "$N_KV_HEADS" \
            --head_dim "$HEAD_DIM" \
            --model_name "$MODEL_NAME" \
            --save_dir "$CONFIGS_DIR" \
            --attn_max 32768 \
            2>&1 | tee "$LOG_DIR/attn_${MODEL_NAME}_${TIMESTAMP}.log"
        
        echo "    ✓ $MODEL_NAME attention profile done"
    done
fi

# ================================================================
# Step 2: 综合 Profiling (使用 profile_and_validate.py, 全部 64 卡)
# 包括: attention自动分段 + All-to-All + P2P Ring + 验证
# ================================================================
echo ""
echo ">>> [Step 2/4] Communication Profiling ($TOTAL_GPUS GPUs)"
echo "    All-to-All + P2P Ring across all group sizes..."

# 对每个模型配置分别 profile
for MODEL_CFG in "qwen2.5-7b:3584:28:4:128:28" "qwen2.5-14b:5120:40:8:128:48" "qwen2.5-32b:5120:40:8:128:64"; do
    IFS=':' read -r MODEL_NAME HIDDEN N_HEADS N_KV_HEADS HEAD_DIM N_LAYERS <<< "$MODEL_CFG"
    
    echo ""
    echo "    --- $MODEL_NAME comm profiling ---"
    
    torchrun \
        --nnodes "$NUM_NODES" \
        --nproc_per_node "$NUM_GPUS_PER_NODE" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        --node_rank "$NODE_RANK" \
        "$BASE_DIR/profile_alltoall.py" \
            --hidden_size "$HIDDEN" \
            --num_attention_heads "$N_HEADS" \
            --num_layers "$N_LAYERS" \
            --save_dir "$CONFIGS_DIR" \
            --model_name "$MODEL_NAME" \
            --topology both \
            --gpus_per_node "$NUM_GPUS_PER_NODE" \
        2>&1 | tee "$LOG_DIR/a2a_${MODEL_NAME}_${TIMESTAMP}.log"
    
    echo "    ✓ $MODEL_NAME All-to-All profile done (consecutive + strided)"
    
    # P2P Ring profiling (consecutive + strided topologies)
    torchrun \
        --nnodes "$NUM_NODES" \
        --nproc_per_node "$NUM_GPUS_PER_NODE" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$((MASTER_PORT + 1))" \
        --node_rank "$NODE_RANK" \
        "$BASE_DIR/profile_p2p_ring.py" \
            --hidden_size "$HIDDEN" \
            --num_attention_heads "$N_HEADS" \
            --num_kv_heads "$N_KV_HEADS" \
            --num_layers "$N_LAYERS" \
            --save_dir "$CONFIGS_DIR" \
            --model_name "$MODEL_NAME" \
            --topology both \
            --gpus_per_node "$NUM_GPUS_PER_NODE" \
        2>&1 | tee "$LOG_DIR/p2p_${MODEL_NAME}_${TIMESTAMP}.log"
    
    echo "    ✓ $MODEL_NAME P2P Ring profile done"
done

# ================================================================
# Step 3: Cost Model Validation (可选, 验证建模准确性)
# ================================================================
echo ""
echo ">>> [Step 3/4] Cost Model Validation (optional)"
echo "    Running on 8 GPUs of this node..."

for MODEL_CFG in "qwen2.5-7b:3584:28:4:128:28" "qwen2.5-14b:5120:40:8:128:48" "qwen2.5-32b:5120:40:8:128:64"; do
    IFS=':' read -r MODEL_NAME HIDDEN N_HEADS N_KV_HEADS HEAD_DIM N_LAYERS <<< "$MODEL_CFG"
    echo "    --- Validating $MODEL_NAME cost model ---"
    
    torchrun \
        --nnodes 1 \
        --nproc_per_node "$NUM_GPUS_PER_NODE" \
        --master_addr "localhost" \
        --master_port "$((MASTER_PORT + 2))" \
        "$BASE_DIR/profile_and_validate.py" \
            --mode validate_cost_model \
            --n_heads "$N_HEADS" \
            --n_kv_heads "$N_KV_HEADS" \
            --head_dim "$HEAD_DIM" \
            --hidden_size "$HIDDEN" \
            --num_layers "$N_LAYERS" \
            --save_dir "$CONFIGS_DIR" \
        2>&1 | tee "$LOG_DIR/validate_${MODEL_NAME}_${TIMESTAMP}.log" || {
        echo "    ⚠ $MODEL_NAME validation failed (non-critical)"
    }
done

# ================================================================
# Step 4: 汇总结果
# ================================================================
echo ""
echo ">>> [Step 4/4] Profile Summary"
echo "============================================"
echo "Profile data saved to: $CONFIGS_DIR"
echo "Logs saved to: $LOG_DIR"
echo ""
echo "Generated files:"
ls -la "$CONFIGS_DIR"/*.json 2>/dev/null || echo "  (no JSON files yet)"
echo ""
echo "============================================"
echo "  Profiling complete!"
echo "  Next: python generate_experiments.py"
echo "============================================"

