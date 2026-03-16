#!/bin/bash
# ============================================================
# Experiment: qwen2.5-14b_common_crawl_256k_flexsp
# Model: qwen2.5-14b | Seq: 256k | Strategy: flexsp | Dataset: common_crawl
# GBS: 512 | GPUs: 64 | Iters: 30
# ============================================================
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_DIR="$(cd "$EXP_DIR/.." && pwd)"
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"

# ---- 集群参数 ----
NUM_NODES=${NUM_NODES:-8}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=600  # 10 min NCCL timeout

EXP_NAME="qwen2.5-14b_common_crawl_256k_flexsp"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

echo "============================================" | tee "$LOG_FILE"
echo "  Experiment: $EXP_NAME" | tee -a "$LOG_FILE"
echo "  Start: $(date)" | tee -a "$LOG_FILE"
echo "  Node: $(hostname), Rank: $NODE_RANK" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# ---- 运行实验 ----
timeout 15m torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $NUM_GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK \
    "$BASE_DIR/train_dist_adacpsp.py" \
    --model_size qwen2.5-14b \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --vocab_size 152064 \
    --hidden_size 5120 \
    --num_hidden_layers 48 \
    --num_attention_heads 40 \
    --seq_length 262144 \
    --global_train_batch_size 512 \
    --epochs 1 \
    --lr 0.0001 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 0 \
    --profile 1 \
    --save_profiled_memory 0 \
    --pp_deg 1 \
    --global_tp_deg 1 \
    --global_tp_consec 1 \
    --sdp 1 \
    --global_checkpoint 0 \
    --selective_checkpoint 1 \
    --vocab_tp 1 \
    --chunks 1 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --use-flash-attn \
    --use-packing \
    --use-adaCPSP \
    --adaCPSP-attn-types ulysses \
    --dataset common_crawl \
    --initialize_on_meta 1 \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "  End: $(date)" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 124 ]; then
    echo "  STATUS: TIMEOUT (exceeded 15 min)" | tee -a "$LOG_FILE"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "  STATUS: FAILED (exit code $EXIT_CODE)" | tee -a "$LOG_FILE"
    # 检查是否是 OOM
    if grep -q "CUDA out of memory\|OutOfMemoryError\|torch.cuda.OutOfMemoryError" "$LOG_FILE"; then
        echo "  REASON: OOM (CUDA out of memory)" | tee -a "$LOG_FILE"
    elif grep -q "NCCL\|nccl" "$LOG_FILE"; then
        echo "  REASON: Possible NCCL error" | tee -a "$LOG_FILE"
    fi
else
    echo "  STATUS: SUCCESS" | tee -a "$LOG_FILE"
fi
echo "============================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
