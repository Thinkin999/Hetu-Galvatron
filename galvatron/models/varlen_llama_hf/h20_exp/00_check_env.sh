#!/bin/bash
# ============================================================
# Phase 0.1: 环境检查脚本
# 在 H20 集群上运行，验证所有依赖是否就绪
# ============================================================
set -e
echo "============================================"
echo "  AdaCPSP Environment Check"
echo "  Date: $(date)"
echo "  Hostname: $(hostname)"
echo "============================================"

# ---- GPU ----
echo ""
echo "[1/8] GPU Information"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPU count on this node: $GPU_COUNT"

# ---- CUDA ----
echo ""
echo "[2/8] CUDA Version"
nvcc --version 2>/dev/null || echo "nvcc not found in PATH"

# ---- Python & PyTorch ----
echo ""
echo "[3/8] Python & PyTorch"
python3 -c "
import sys; print(f'Python: {sys.version}')
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# ---- Flash Attention ----
echo ""
echo "[4/8] Flash Attention"
python3 -c "
try:
    import flash_attn; print(f'flash_attn version: {flash_attn.__version__}')
except ImportError:
    print('ERROR: flash_attn not installed')
"

# ---- Galvatron ----
echo ""
echo "[5/8] Galvatron"
python3 -c "
try:
    import galvatron; print('galvatron: OK')
except ImportError:
    print('ERROR: galvatron not importable')
try:
    from galvatron.models.varlen_llama_hf.adacpsp_solver import AdaCPSPCostModel
    print('AdaCPSPCostModel: OK')
except ImportError as e:
    print(f'ERROR: {e}')
"

# ---- Datasets ----
echo ""
echo "[6/8] Datasets"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
for ds in common_crawl github; do
    found=0
    for dir in "$BASE_DIR/../varlen_datasets" "$BASE_DIR/../../flexsp/Hetu-Galvatron/galvatron/datasets"; do
        if [ -f "$dir/$ds.txt" ]; then
            lines=$(wc -l < "$dir/$ds.txt")
            echo "  $ds.txt: FOUND at $dir ($lines lines)"
            found=1
            break
        fi
    done
    if [ $found -eq 0 ]; then
        echo "  $ds.txt: NOT FOUND"
    fi
done

# ---- Meta Configs ----
echo ""
echo "[7/8] Model Meta Configs"
for model in qwen2.5-7b qwen2.5-14b qwen2.5-32b; do
    cfg="$SCRIPT_DIR/../meta_configs/$model.json"
    if [ -f "$cfg" ]; then
        echo "  $model.json: OK"
        cat "$cfg"
    else
        echo "  $model.json: NOT FOUND at $cfg"
    fi
done

# ---- NCCL / Network ----
echo ""
echo "[8/8] NCCL & Network"
python3 -c "
import torch.distributed as dist
import os
print(f'NCCL available: {dist.is_nccl_available()}')
print(f'MASTER_ADDR: {os.environ.get(\"MASTER_ADDR\", \"NOT SET\")}')
print(f'MASTER_PORT: {os.environ.get(\"MASTER_PORT\", \"NOT SET\")}')
print(f'RANK: {os.environ.get(\"RANK\", \"NOT SET\")}')
print(f'WORLD_SIZE: {os.environ.get(\"WORLD_SIZE\", \"NOT SET\")}')
"

# Check InfiniBand
echo ""
echo "InfiniBand devices:"
ibstat 2>/dev/null | grep -E "^CA|State|Rate" || echo "  ibstat not available or no IB devices"

# Check NVLink
echo ""
echo "NVLink topology:"
nvidia-smi topo -m 2>/dev/null || echo "  nvidia-smi topo not available"

echo ""
echo "============================================"
echo "  Environment check complete!"
echo "============================================"


