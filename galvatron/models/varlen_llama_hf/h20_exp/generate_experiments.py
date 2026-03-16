#!/usr/bin/env python3
"""
实验脚本生成器
=============
根据实验矩阵自动生成所有 shell 脚本 + master 运行脚本

模型: Qwen2.5-7B, 14B, 32B
序列长度: 128K, 256K, 384K, 512K
策略: FlexSP(ulysses-only), AdaCPSP-UR(ulysses+ring), AdaCPSP-Full(ulysses+ring+usp)
数据集: common_crawl, github
"""

import os
import stat
import json

# ═══════════════════════════════════════════════════════════════
# 实验配置
# ═══════════════════════════════════════════════════════════════

# 集群配置
NUM_NODES = 8
NUM_GPUS_PER_NODE = 8
TOTAL_GPUS = NUM_NODES * NUM_GPUS_PER_NODE  # 64

# 模型配置: (model_name, model_size_arg, hidden_size, num_layers, num_heads, num_kv_heads, head_dim, vocab_size)
MODELS = [
    ("qwen2.5-7b",  "qwen2.5-7b",  3584, 28, 28, 4,  128, 152064),
    ("qwen2.5-14b", "qwen2.5-14b", 5120, 48, 40, 8,  128, 152064),
    ("qwen2.5-32b", "qwen2.5-32b", 5120, 64, 40, 8,  128, 152064),
]

# 序列长度配置: (label, seq_length)
SEQ_LENGTHS = [
    ("128k", 131072),
    ("256k", 262144),
    ("384k", 393216),
    ("512k", 524288),
]

# 策略配置: (strategy_label, attn_types_arg, description)
STRATEGIES = [
    ("flexsp",       "ulysses",           "FlexSP (Ulysses only)"),
    ("adacpsp_ur",   "ulysses ring",      "AdaCPSP (Ulysses + Ring)"),
    ("adacpsp_full", "ulysses ring usp",  "AdaCPSP (Ulysses + Ring + USP)"),
]

# 数据集
DATASETS = ["common_crawl", "github"]

# 固定参数
GBS = 512           # Global Batch Size, 匹配 FlexSP
ITERS = 30          # Benchmark iterations
EPOCHS = 1          # 只跑 1 epoch (靠 iter 数控制)
LR = 1e-4
TIMEOUT_MINUTES = 15  # 每个实验超时时间

# 模型特定的降级 GBS (如果 512 OOM)
FALLBACK_GBS = {
    "qwen2.5-32b": 256,  # 32B 模型可能需要更小的 GBS
}


def generate_single_experiment(
    model_name, model_size, hidden_size, num_layers, num_heads, num_kv_heads,
    head_dim, vocab_size, seq_label, seq_length, strategy_label, attn_types,
    dataset, gbs, output_dir
):
    """生成单个实验的 shell 脚本"""
    
    exp_name = f"{model_name}_{dataset}_{seq_label}_{strategy_label}"
    log_file = f"logs/{exp_name}.log"
    
    script = f"""#!/bin/bash
# ============================================================
# Experiment: {exp_name}
# Model: {model_name} | Seq: {seq_label} | Strategy: {strategy_label} | Dataset: {dataset}
# GBS: {gbs} | GPUs: {TOTAL_GPUS} | Iters: {ITERS}
# ============================================================
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_DIR="$(cd "$EXP_DIR/.." && pwd)"
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"

# ---- 集群参数 ----
NUM_NODES=${{NUM_NODES:-{NUM_NODES}}}
NUM_GPUS_PER_NODE=${{NUM_GPUS_PER_NODE:-{NUM_GPUS_PER_NODE}}}
MASTER_ADDR=${{MASTER_ADDR:-$(hostname)}}
MASTER_PORT=${{MASTER_PORT:-29500}}
NODE_RANK=${{NODE_RANK:-${{RANK:-0}}}}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=600  # 10 min NCCL timeout

EXP_NAME="{exp_name}"
LOG_FILE="$LOG_DIR/${{EXP_NAME}}.log"

echo "============================================" | tee "$LOG_FILE"
echo "  Experiment: $EXP_NAME" | tee -a "$LOG_FILE"
echo "  Start: $(date)" | tee -a "$LOG_FILE"
echo "  Node: $(hostname), Rank: $NODE_RANK" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# ---- 运行实验 ----
timeout {TIMEOUT_MINUTES}m torchrun \\
    --nnodes $NUM_NODES \\
    --nproc_per_node $NUM_GPUS_PER_NODE \\
    --master_addr $MASTER_ADDR \\
    --master_port $MASTER_PORT \\
    --node_rank $NODE_RANK \\
    "$BASE_DIR/train_dist_adacpsp.py" \\
    --model_size {model_size} \\
    --set_model_config_manually 0 \\
    --set_layernum_manually 0 \\
    --vocab_size {vocab_size} \\
    --hidden_size {hidden_size} \\
    --num_hidden_layers {num_layers} \\
    --num_attention_heads {num_heads} \\
    --seq_length {seq_length} \\
    --global_train_batch_size {gbs} \\
    --epochs {EPOCHS} \\
    --lr {LR} \\
    --adam_weight_decay 0.01 \\
    --dropout_prob 0.1 \\
    --check_loss 0 \\
    --profile 1 \\
    --save_profiled_memory 0 \\
    --pp_deg 1 \\
    --global_tp_deg 1 \\
    --global_tp_consec 1 \\
    --sdp 1 \\
    --global_checkpoint 0 \\
    --selective_checkpoint 1 \\
    --vocab_tp 1 \\
    --chunks 1 \\
    --pipeline_type pipedream_flush \\
    --default_dp_type zero2 \\
    --mixed_precision bf16 \\
    --use-flash-attn \\
    --use-packing \\
    --use-adaCPSP \\
    --adaCPSP-attn-types {attn_types} \\
    --dataset {dataset} \\
    --initialize_on_meta 1 \\
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "  End: $(date)" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 124 ]; then
    echo "  STATUS: TIMEOUT (exceeded {TIMEOUT_MINUTES} min)" | tee -a "$LOG_FILE"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "  STATUS: FAILED (exit code $EXIT_CODE)" | tee -a "$LOG_FILE"
    # 检查是否是 OOM
    if grep -q "CUDA out of memory\\|OutOfMemoryError\\|torch.cuda.OutOfMemoryError" "$LOG_FILE"; then
        echo "  REASON: OOM (CUDA out of memory)" | tee -a "$LOG_FILE"
    elif grep -q "NCCL\\|nccl" "$LOG_FILE"; then
        echo "  REASON: Possible NCCL error" | tee -a "$LOG_FILE"
    fi
else
    echo "  STATUS: SUCCESS" | tee -a "$LOG_FILE"
fi
echo "============================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
"""
    
    script_path = os.path.join(output_dir, f"{exp_name}.sh")
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)
    return exp_name, script_path, log_file


def generate_master_runner(experiments, output_dir, script_dir_name="exp_scripts"):
    """生成 master 运行脚本 02_run_all.sh"""
    
    header = f"""#!/bin/bash
# ============================================================
# Master Runner: 运行全部 {len(experiments)} 个实验
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
EXP_SCRIPTS_DIR="$SCRIPT_DIR/{script_dir_name}"
LOG_DIR="$SCRIPT_DIR/logs"
SUMMARY="$LOG_DIR/summary.log"
mkdir -p "$LOG_DIR"

TOTAL={len(experiments)}
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
printf "%-60s %-12s %-10s\\n" "EXPERIMENT" "STATUS" "TIME(s)" | tee -a "$SUMMARY"
printf "%-60s %-12s %-10s\\n" "$(printf '%.0s-' {{1..60}})" "$(printf '%.0s-' {{1..12}})" "$(printf '%.0s-' {{1..10}})" | tee -a "$SUMMARY"

"""
    
    body = ""
    for i, (exp_name, script_path, log_file) in enumerate(experiments):
        script_basename = os.path.basename(script_path)
        body += f"""
# ---- [{i+1}/{len(experiments)}] {exp_name} ----
echo ""
echo ">>> [{i+1}/$TOTAL] Running: {exp_name}"
START_TIME=$(date +%s)

bash "$EXP_SCRIPTS_DIR/{script_basename}"
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
    if [ -f "$LOG_DIR/{exp_name}.log" ] && grep -q "CUDA out of memory\\|OutOfMemoryError" "$LOG_DIR/{exp_name}.log"; then
        STATUS="OOM"
        OOM=$((OOM + 1))
    else
        STATUS="FAILED($EXP_EXIT)"
    fi
    FAILED=$((FAILED + 1))
fi

printf "%-60s %-12s %-10s\\n" "{exp_name}" "$STATUS" "${{ELAPSED}}s" | tee -a "$SUMMARY"

# 清理 GPU 缓存, 等待进程结束
sleep 5
"""
    
    footer = """
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
"""
    
    runner_path = os.path.join(output_dir, "02_run_all.sh")
    with open(runner_path, "w") as f:
        f.write(header + body + footer)
    os.chmod(runner_path, os.stat(runner_path).st_mode | stat.S_IEXEC)
    return runner_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_scripts_dir = os.path.join(script_dir, "exp_scripts")
    os.makedirs(exp_scripts_dir, exist_ok=True)
    
    experiments = []
    
    print("=" * 70)
    print("  Generating Experiment Scripts")
    print("=" * 70)
    
    # 按特定顺序生成: 先短序列(快速验证), 再长序列
    for model_info in MODELS:
        model_name, model_size, hidden, layers, heads, kv_heads, head_dim, vocab = model_info
        
        for seq_label, seq_length in SEQ_LENGTHS:
            for strategy_label, attn_types, desc in STRATEGIES:
                for dataset in DATASETS:
                    gbs = GBS
                    # 对大模型+长序列, 可能需要降低 GBS
                    if model_name in FALLBACK_GBS and seq_length >= 393216:
                        gbs = FALLBACK_GBS[model_name]
                    
                    exp_name, script_path, log_file = generate_single_experiment(
                        model_name=model_name,
                        model_size=model_size,
                        hidden_size=hidden,
                        num_layers=layers,
                        num_heads=heads,
                        num_kv_heads=kv_heads,
                        head_dim=head_dim,
                        vocab_size=vocab,
                        seq_label=seq_label,
                        seq_length=seq_length,
                        strategy_label=strategy_label,
                        attn_types=attn_types,
                        dataset=dataset,
                        gbs=gbs,
                        output_dir=exp_scripts_dir,
                    )
                    experiments.append((exp_name, script_path, log_file))
                    print(f"  Generated: {exp_name} (GBS={gbs})")
    
    print(f"\n  Total experiments: {len(experiments)}")
    print(f"  Scripts in: {exp_scripts_dir}/")
    
    # 生成 master runner
    runner_path = generate_master_runner(experiments, script_dir)
    print(f"  Master runner: {runner_path}")
    
    # 生成实验索引 JSON (供 analyze_results.py 使用)
    index = {
        "total": len(experiments),
        "models": [m[0] for m in MODELS],
        "seq_lengths": [s[0] for s in SEQ_LENGTHS],
        "strategies": [s[0] for s in STRATEGIES],
        "datasets": DATASETS,
        "gbs": GBS,
        "experiments": [
            {
                "name": name,
                "log": log,
                "model": name.split("_")[0] + "_" + name.split("_")[1],  # e.g. qwen2.5-7b
                "dataset": "_".join(name.split("_")[2:-2]),
                "seq_len": name.split("_")[-2],
                "strategy": name.split("_")[-1],
            }
            for name, _, log in experiments
        ],
    }
    
    # 用更可靠的方式解析实验名
    parsed_experiments = []
    for name, _, log in experiments:
        # 名字格式: {model}_{dataset}_{seqlen}_{strategy}
        # model: qwen2.5-7b, qwen2.5-14b, qwen2.5-32b
        # dataset: common_crawl, github
        # seqlen: 128k, 256k, 384k, 512k
        # strategy: flexsp, adacpsp_ur, adacpsp_full
        for m in [m[0] for m in MODELS]:
            if name.startswith(m + "_"):
                rest = name[len(m) + 1:]
                for d in DATASETS:
                    if rest.startswith(d + "_"):
                        rest2 = rest[len(d) + 1:]
                        for s in [s[0] for s in SEQ_LENGTHS]:
                            if rest2.startswith(s + "_"):
                                strat = rest2[len(s) + 1:]
                                parsed_experiments.append({
                                    "name": name,
                                    "log": log,
                                    "model": m,
                                    "dataset": d,
                                    "seq_len": s,
                                    "strategy": strat,
                                })
                                break
                        break
                break
    
    index["experiments"] = parsed_experiments
    
    index_path = os.path.join(script_dir, "experiment_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Index: {index_path}")
    
    print("\n" + "=" * 70)
    print("  Done! Next steps:")
    print("  1. Run profiling: bash 01_profile_hardware.sh")
    print("  2. Run experiments: bash 02_run_all.sh")
    print("  3. Analyze results: python analyze_results.py --log-dir logs/")
    print("=" * 70)


if __name__ == "__main__":
    main()

