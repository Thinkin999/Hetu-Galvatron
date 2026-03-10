#!/usr/bin/env python3
"""
Generate experiment shell scripts for FlexSP vs AdaCPSP comparison.
Run once to create all scripts under llama_scripts/

Usage:
  python generate_exp_scripts.py
"""

import os
import stat

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "llama_scripts")
LOG_DIR = "logs"   # relative to varlen_llama_hf/

# ═══════════════════════════════════════════════════════════
# Hardware Constants
# ═══════════════════════════════════════════════════════════
NUM_NODES = 1
NUM_GPUS = 8
GPU_MEM_GB = 40
MEM_LIMIT_GB = 36  # 90% of 40GB

# ═══════════════════════════════════════════════════════════
# Model Constants (LLaMA-7B)
# ═══════════════════════════════════════════════════════════
HIDDEN_SIZE = 4096
NUM_HEADS = 32
VOCAB_SIZE = 32000

# ═══════════════════════════════════════════════════════════
# Experiment Configurations
# ═══════════════════════════════════════════════════════════
EXPERIMENTS = [
    {
        "id": "E5_fast_val",
        "desc": "2-layer quick validation",
        "max_seq": 32768,
        "gbs": 32,
        "layers": 2,
        "iters": 10,
        "dataset": "wikipedia",
        "warmup_iters": 3,
    },
    {
        "id": "E1_short",
        "desc": "Short sequences (<=8k), high GBS baseline",
        "max_seq": 8192,
        "gbs": 64,
        "layers": 32,
        "iters": 20,
        "dataset": "wikipedia",
        "warmup_iters": 5,
    },
    {
        "id": "E2_mixed",
        "desc": "Mixed lengths (<=32k), moderate GBS",
        "max_seq": 32768,
        "gbs": 32,
        "layers": 32,
        "iters": 20,
        "dataset": "wikipedia",
        "warmup_iters": 5,
    },
    {
        "id": "E3_long",
        "desc": "Long-focused (<=32k), small GBS",
        "max_seq": 32768,
        "gbs": 16,
        "layers": 32,
        "iters": 20,
        "dataset": "wikipedia",
        "warmup_iters": 5,
    },
    {
        "id": "E4_extreme",
        "desc": "Extreme long (<=64k), minimal GBS",
        "max_seq": 65536,
        "gbs": 8,
        "layers": 32,
        "iters": 20,
        "dataset": "wikipedia",
        "warmup_iters": 5,
    },
]

# Two modes: FlexSP (ulysses only) and AdaCPSP (ulysses + ring + usp)
MODES = {
    "flexsp":  {"attn_types": "ulysses",          "label": "FlexSP (Ulysses-only)"},
    "adacpsp": {"attn_types": "ulysses ring usp",  "label": "AdaCPSP (Ulysses+Ring+USP)"},
}


def generate_experiment_script(exp, mode_name, mode_cfg):
    """Generate a single experiment shell script."""
    exp_id = exp["id"]
    log_file = f"{LOG_DIR}/exp_{exp_id}_{mode_name}.log"
    # Use different master ports for different experiments to avoid conflict
    port_offset = hash(exp_id) % 100
    master_port = 29510 + port_offset

    script = f"""#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Experiment: {exp_id} — {mode_cfg['label']}
# Description: {exp['desc']}
# Config: max_seq={exp['max_seq']}, GBS={exp['gbs']}, layers={exp['layers']}
# ═══════════════════════════════════════════════════════════════════════
cd "$(dirname "$0")/.." || exit 1

export NUM_NODES={NUM_NODES}
export NUM_GPUS_PER_NODE={NUM_GPUS}
export MASTER_ADDR=localhost
export MASTER_PORT={master_port}
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_HCA=mlx5_2,mlx5_5

mkdir -p {LOG_DIR}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║ Experiment: {exp_id:12s} — {mode_name.upper():8s}                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ max_seq={exp['max_seq']:6d} | GBS={exp['gbs']:3d} | layers={exp['layers']:2d} | iters={exp['iters']:2d}       ║"
echo "║ attn_types: {mode_cfg['attn_types']:46s}║"
echo "║ dataset: {exp['dataset']:49s}║"
echo "║ memory_limit: {MEM_LIMIT_GB} GB                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

START_TIME=$(date +%s)

torchrun \\
    --nnodes ${{NUM_NODES}} \\
    --nproc_per_node ${{NUM_GPUS_PER_NODE}} \\
    --master_addr ${{MASTER_ADDR}} \\
    --master_port ${{MASTER_PORT}} \\
    --node_rank ${{NODE_RANK}} \\
    train_dist_adacpsp.py \\
    --model_size llama-7b \\
    --set_model_config_manually 0 \\
    --set_layernum_manually 1 \\
    --set_seqlen_manually 1 \\
    --vocab_size {VOCAB_SIZE} \\
    --hidden_size {HIDDEN_SIZE} \\
    --num_hidden_layers {exp['layers']} \\
    --num_attention_heads {NUM_HEADS} \\
    --seq_length {exp['max_seq']} \\
    --global_train_batch_size {exp['gbs']} \\
    --train-iters {exp['iters']} \\
    --lr 1e-4 \\
    --adam_weight_decay 0.01 \\
    --dropout_prob 0.0 \\
    --check_loss 0 \\
    --profile 1 \\
    --save_profiled_memory 0 \\
    --dataset {exp['dataset']} \\
    --pp_deg 1 \\
    --global_tp_deg 1 \\
    --global_tp_consec 1 \\
    --sdp 0 \\
    --global_checkpoint 0 \\
    --vocab_tp 1 \\
    --chunks 1 \\
    --global_cp_deg 1 \\
    --pipeline_type pipedream_flush \\
    --default_dp_type zero2 \\
    --mixed_precision bf16 \\
    --use-flash-attn \\
    --initialize_on_meta 1 \\
    --use-packing \\
    --use-adaCPSP \\
    --adaCPSP-attn-types {mode_cfg['attn_types']} \\
    --memory-limit-gb {MEM_LIMIT_GB} \\
    2>&1 | tee {log_file}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Experiment {exp_id} ({mode_name}) completed in ${{ELAPSED}}s"
echo "Log: {log_file}"
echo "═══════════════════════════════════════════════════════════════"
"""
    return script


def generate_run_all_script(experiments, modes):
    """Generate the master script that runs all experiments."""
    lines = [
        "#!/bin/bash",
        "# ═══════════════════════════════════════════════════════════════",
        "# Master Script: Run ALL FlexSP vs AdaCPSP Experiments",
        "# ═══════════════════════════════════════════════════════════════",
        'cd "$(dirname "$0")" || exit 1',
        "",
        "TOTAL_START=$(date +%s)",
        "",
        'echo "╔══════════════════════════════════════════════════════════════╗"',
        'echo "║  FlexSP vs AdaCPSP — Full Experiment Suite                 ║"',
        'echo "╚══════════════════════════════════════════════════════════════╝"',
        'echo ""',
        "",
    ]

    for exp in experiments:
        exp_id = exp["id"]
        lines.append(f'echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"')
        lines.append(f'echo "Experiment: {exp_id} — {exp["desc"]}"')
        lines.append(f'echo "  max_seq={exp["max_seq"]}, GBS={exp["gbs"]}, layers={exp["layers"]}"')
        lines.append(f'echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"')
        lines.append("")

        for mode_name in modes:
            script_name = f"exp_{exp_id}_{mode_name}.sh"
            lines.append(f'echo "  → Running {mode_name.upper()}..."')
            lines.append(f'bash {script_name}')
            lines.append(f'EXIT_CODE=$?')
            lines.append(f'if [ $EXIT_CODE -ne 0 ]; then')
            lines.append(f'    echo "  ✗ {script_name} failed with exit code $EXIT_CODE"')
            lines.append(f'    echo "  Continuing with next experiment..."')
            lines.append(f'fi')
            lines.append(f'echo "  Cooling down (10s)..."')
            lines.append(f'sleep 10')
            lines.append("")

    lines.append("TOTAL_END=$(date +%s)")
    lines.append("TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))")
    lines.append("")
    lines.append('echo ""')
    lines.append('echo "═══════════════════════════════════════════════════════════════"')
    lines.append('echo "All experiments completed in ${TOTAL_ELAPSED}s"')
    lines.append('echo "Logs saved to ../logs/"')
    lines.append('echo "═══════════════════════════════════════════════════════════════"')
    lines.append('echo ""')
    lines.append('echo "Next: Run analysis script:"')
    lines.append('echo "  cd .. && python analyze_experiment_logs.py --log_dir logs/"')

    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, LOG_DIR), exist_ok=True)

    generated = []

    # Generate per-experiment scripts
    for exp in EXPERIMENTS:
        for mode_name, mode_cfg in MODES.items():
            script_name = f"exp_{exp['id']}_{mode_name}.sh"
            script_path = os.path.join(OUTPUT_DIR, script_name)
            content = generate_experiment_script(exp, mode_name, mode_cfg)
            with open(script_path, 'w') as f:
                f.write(content)
            os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)
            generated.append(script_name)
            print(f"  ✓ {script_name}")

    # Generate master script
    master_name = "exp_run_all.sh"
    master_path = os.path.join(OUTPUT_DIR, master_name)
    content = generate_run_all_script(EXPERIMENTS, MODES)
    with open(master_path, 'w') as f:
        f.write(content)
    os.chmod(master_path, os.stat(master_path).st_mode | stat.S_IEXEC)
    generated.append(master_name)
    print(f"  ✓ {master_name} (master)")

    print(f"\nGenerated {len(generated)} scripts in {OUTPUT_DIR}/")
    print(f"\nTo run all experiments:")
    print(f"  cd {SCRIPT_DIR}/llama_scripts")
    print(f"  bash exp_run_all.sh")
    print(f"\nTo run a single experiment:")
    print(f"  bash exp_E5_fast_val_flexsp.sh   # quick validation")


if __name__ == "__main__":
    main()

