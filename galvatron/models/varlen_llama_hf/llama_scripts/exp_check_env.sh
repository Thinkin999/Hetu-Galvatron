#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Environment Check for AdaCPSP vs FlexSP Experiments
# ═══════════════════════════════════════════════════════════════
set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  AdaCPSP Experiment Environment Check                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. GPU Check ──
echo "═══ GPU Information ═══"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Total GPUs: ${NUM_GPUS}"
echo ""

# ── 2. Python / PyTorch ──
echo "═══ Python & PyTorch ═══"
python3 -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA (torch): {torch.version.cuda}')
print(f'NCCL: {torch.cuda.nccl.version()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)')
"
echo ""

# ── 3. Key Packages ──
echo "═══ Key Packages ═══"
python3 -c "
packages = ['flash_attn', 'transformers', 'numpy', 'scipy', 'h5py']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        print(f'  {pkg}: {ver} ✓')
    except ImportError:
        print(f'  {pkg}: NOT FOUND ✗')

# ring_flash_attn (optional but needed for Ring Attention)
try:
    import ring_flash_attn
    print(f'  ring_flash_attn: available ✓')
except ImportError:
    print(f'  ring_flash_attn: NOT FOUND ⚠ (Ring Attention will not work)')

# pyscipopt (optional, for ILP solver)
try:
    from pyscipopt import Model
    print(f'  pyscipopt: available ✓')
except ImportError:
    print(f'  pyscipopt: NOT FOUND (ILP solver disabled, heuristic only)')
"
echo ""

# ── 4. Galvatron Import ──
echo "═══ Galvatron / AdaCPSP ═══"
python3 -c "
try:
    from galvatron.models.varlen_llama_hf.adacpsp_solver import AdaCPSPCostModel, AdaCPSPOptimizer
    print('  AdaCPSPCostModel: importable ✓')
    print('  AdaCPSPOptimizer: importable ✓')
except Exception as e:
    print(f'  Import error: {e} ✗')

try:
    from galvatron.models.varlen_llama_hf.adacpsp_group_manager import convert_microbatch_res
    print('  convert_microbatch_res: importable ✓')
except Exception as e:
    print(f'  Import error: {e} ✗')
"
echo ""

# ── 5. Dataset ──
echo "═══ Dataset ═══"
DATASET_PATH="/home/pkuhetu/lqs/flexsp/Hetu-Galvatron/galvatron/datasets/wikipedia.txt"
if [ -f "${DATASET_PATH}" ]; then
    LINES=$(wc -l < "${DATASET_PATH}")
    echo "  Wikipedia dataset: ${DATASET_PATH}"
    echo "  Total samples: ${LINES}"
    echo "  ✓ Dataset available"
else
    echo "  ✗ Dataset NOT FOUND at ${DATASET_PATH}"
    echo "    Please set correct path in training scripts"
fi
echo ""

# ── 6. Profiling Data ──
echo "═══ Profiling Data ═══"
CONFIGS_DIR="$(dirname "$0")/../configs"
PROFILE_COUNT=$(ls ${CONFIGS_DIR}/profile_validate_*.json 2>/dev/null | wc -l)
A2A_COUNT=$(ls ${CONFIGS_DIR}/alltoall_profile_*.json 2>/dev/null | wc -l)
P2P_COUNT=$(ls ${CONFIGS_DIR}/p2p_ring_profile_*.json 2>/dev/null | wc -l)
echo "  Profile validate JSONs: ${PROFILE_COUNT}"
echo "  AlltoAll profile JSONs: ${A2A_COUNT}"
echo "  P2P Ring profile JSONs: ${P2P_COUNT}"
if [ "${PROFILE_COUNT}" -gt 0 ]; then
    echo "  ✓ Profiling data available"
else
    echo "  ⚠ No profiling data found (cost model will use defaults)"
fi
echo ""

# ── 7. NCCL Connectivity ──
echo "═══ NCCL Quick Test (2 GPUs) ═══"
python3 -c "
import torch
import torch.distributed as dist
import os, sys
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29599'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
dist.init_process_group('nccl', rank=0, world_size=1)
t = torch.ones(1024, device='cuda:0')
print(f'  Single-GPU NCCL init: ✓')
dist.destroy_process_group()
" 2>/dev/null && echo "  NCCL basic check: PASSED ✓" || echo "  NCCL basic check: FAILED ✗"
echo ""

echo "═══ Environment Check Complete ═══"
echo "If all checks pass, proceed with experiments."

