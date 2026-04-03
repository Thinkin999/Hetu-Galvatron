# AdaCPSP — Adaptive Context Parallel & Sequence Parallel

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 快速开始](#2-快速开始)
  - [2.1 环境配置](#21-环境配置)
  - [2.2 安装依赖](#22-安装依赖)
  - [2.3 一键部署参考](#23-一键部署参考)
- [3. 使用方法](#3-使用方法)
  - [3.1 Profiling（通信/计算测量）](#31-profiling通信计算测量)
  - [3.2 单策略训练验证](#32-单策略训练验证)
  - [3.3 AdaCPSP 自适应训练](#33-adacpsp-自适应训练)
  - [3.4 关键参数说明](#34-关键参数说明)
- [4. 测试方法](#4-测试方法)
- [5. 架构设计](#5-架构设计)
  - [5.1 系统概览](#51-系统概览)
  - [5.2 核心模块](#52-核心模块)
  - [5.3 数据流](#53-数据流)
  - [5.4 Attention Forward 路由](#54-attention-forward-路由)
  - [5.5 策略映射机制](#55-策略映射机制)
  - [5.6 通信组设计](#56-通信组设计)
- [6. 文件结构](#6-文件结构)
- [7. 技术细节](#7-技术细节)
  - [7.1 Zigzag Ring Attention 数据切分](#71-zigzag-ring-attention-数据切分)
  - [7.2 Ulysses + Ring 组合模式](#72-ulysses--ring-组合模式)
  - [7.3 RoPE Position IDs 计算](#73-rope-position-ids-计算)
  - [7.4 VocabParallelEmbedding + Packing 兼容性](#74-vocabparallelembedding--packing-兼容性)
  - [7.5 AdaCPSP 求解器算法](#75-adacpsp-求解器算法)
  - [7.6 FlexSP vs AdaCPSP 求解器对比](#76-flexsp-vs-adacpsp-求解器对比)
  - [7.7 CostModel 增强：Overlap + Causal + Fwd/Bwd 分离](#77-costmodel-增强overlap--causal--fwdbwd-分离)
- [8. 开发历史与修复记录](#8-开发历史与修复记录)
- [9. 后续规划 (Phase 5+)](#9-后续规划-phase-5)
- [10. FAQ](#10-faq)

---

## 1. 项目简介

**AdaCPSP (Adaptive Context Parallel & Sequence Parallel)** 是对 [FlexSP](../../../flexsp/) 的扩展，能够在训练长序列模型时**自适应**地为每组序列选择最优的并行策略（Ulysses SP / Ring Attention CP / 两者组合），而非使用统一策略。

| 维度 | FlexSP | AdaCPSP |
|------|--------|---------|
| 策略空间 | 只选择 Ulysses SP 的 `sp_size` | 同时选择 `attn_type` (Ulysses / Ring / **USP**) 和 `parallel_size` |
| Tensor Parallel | 支持 tp_deg ≥ 1 | **tp_deg = 1**（每个 GPU 持有完整权重，TP 和 Ulysses 互斥） |
| 通信模式 | 仅 All-to-All | All-to-All + P2P Ring + **USP (All-to-All × P2P Ring)** |
| 异构并行组 | 同一 MB 内所有组相同策略 | 同一 MB 内不同组可选不同策略和大小 |
| 数据切分 | 按 token 总数切分 | 按 zigzag ring attention 方式切分 |
| 流水线并行 | 支持 PP | **不使用 PP**（pp_deg=1） |
| FSDP | dp 可变 | **FSDP 覆盖所有 GPU**（dp = world_size）|

**关键特性：**
- ✅ `tp_deg=1` 设计：每个 GPU 持有完整模型权重，SP 和 CP 完全动态
- ✅ **异构并行组**：同一 microbatch 中不同通信组可选择不同的 `attn_type` 和 `parallel_size`
- ✅ **USP (Ulysses + Ring 组合)**：支持 sp_size × cp_size 的 2D mesh 并行（如 sp=2 × cp=4 = 8 GPUs）
- ✅ 变长序列 (varlen/packing) 支持：Ulysses SP、Ring Attention CP、USP 组合
- ✅ Profiling 基础设施：All-to-All/P2P 线性拟合、Attention 自动断点检测 + 分段二次拟合、Overlap profiling
- ✅ CostModel 增强：overlap-aware 建模 (Ring/USP)、causal correction、fwd/bwd 分离、GQA-aware 通信、插值通信模型、overlap 泄漏、**验证校准 (v4)**
- ✅ 独立 AdaCPSP 求解器：BFD/FFD 启发式 + ILP 精确求解（含序列分桶、Warm-Start、多进程）
- ✅ Solver ↔ Runtime 端到端集成：每个 microbatch 动态策略选择
- ✅ 强制策略测试模式：可手动注入任意异构并行组进行验证

---

## 2. 快速开始

### 2.1 环境配置

当前验证通过的环境如下：

| 项目 | 版本 |
|------|------|
| OS | Ubuntu 20.04 LTS (kernel 5.4.0) |
| Python | 3.9.23 |
| CUDA | 12.1 |
| cuDNN | 8.9.2 |
| PyTorch | 2.1.0 (CUDA 12.1) |
| GPU | NVIDIA A100-SXM4-40GB × 8 |
| Conda | 23.3.1 |

**Python 依赖包：**

| 包名 | 版本 | 说明 |
|------|------|------|
| `flash-attn` | 2.5.9 | Flash Attention (含 varlen 支持) |
| `ring-flash-attn` | 0.1.5 | Ring / Zigzag Ring Attention ([zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention)) |
| `megatron-core` | 0.9.0 | Megatron-LM 核心库（TP、SP 基础设施） |
| `transformers` | 4.52.4 | HuggingFace Transformers（模型配置） |
| `pyscipopt` | 6.0.0 | SCIP 混合整数规划求解器（ILP 模式需要） |
| `numpy` | 1.26.4 | 数值计算 |
| `scipy` | 1.13.1 | 分段拟合 |
| `hetu-galvatron` | 1.0.0 | 本项目（editable install） |

### 2.2 安装依赖

```bash
# 1. 创建 conda 环境
conda create -n megatorn_cu121_py39_lqs python=3.9 -y
conda activate megatorn_cu121_py39_lqs

# 2. 安装 PyTorch (CUDA 12.1)
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. 安装 flash-attn (需要 CUDA toolkit)
pip install flash-attn==2.5.9 --no-build-isolation

# 4. 安装 ring-flash-attn
#    建议从源码安装，以确保与当前 flash-attn 版本兼容
cd /path/to/ring-flash-attention
pip install -e .

# 5. 安装 megatron-core
#    建议从源码安装
cd /path/to/megatron-lm
pip install -e .

# 6. 安装 pyscipopt (ILP 求解器，可选但推荐)
pip install pyscipopt==6.0.0

# 7. 安装其他依赖
pip install transformers==4.52.4 numpy==1.26.4 scipy==1.13.1

# 8. 安装本项目 (Hetu-Galvatron)
cd /path/to/Hetu-Galvatron
pip install -e .
```

> ⚠️ **注意**：`flash-attn` 的安装需要 CUDA toolkit 和 gcc/g++ 编译器，安装时间较长（~10-30 分钟），建议提前准备好编译环境。

### 2.3 一键部署参考

目前暂无一键部署脚本，因为部分依赖（`ring-flash-attn`、`megatron-core`）需要从源码安装且可能需要修改。以下是本项目验证环境中各依赖的安装路径，供参考：

```
Hetu-Galvatron (本项目):    /home/pkuhetu/lqs/galvatron_lxy/Hetu-Galvatron
megatron-core (editable):   /home/pkuhetu/lqs/megatron_lxy
ring-flash-attn (editable): /home/pkuhetu/lqs/ring-flash-attention
conda env:                  /home/pkuhetu/envs/miniconda3/envs/megatorn_cu121_py39_lqs
```

如需在新机器部署，建议：
1. 先确认 GPU 型号和 CUDA 版本
2. 按照 2.2 节步骤创建环境
3. 运行 `test_ulysses.sh` 验证单策略可用
4. 运行 `test_adacpsp.sh` 验证端到端功能

---

## 3. 使用方法

所有脚本位于 `galvatron/models/varlen_llama_hf/llama_scripts/` 目录下。

### 3.1 Profiling（通信/计算测量）

在使用 AdaCPSP 求解器之前，需要先获取当前硬件的通信带宽和计算性能参数。

```bash
cd galvatron/models/varlen_llama_hf

# 1. All-to-All 通信带宽测量（Ulysses SP 使用）
bash llama_scripts/profile_alltoall.sh
# 输出: configs/alltoall_profile_*.json

# 2. P2P Ring 通信带宽测量（Ring Attention 使用）
bash llama_scripts/profile_p2p_ring.sh
# 输出: configs/p2p_ring_profile_*.json

# 3. Attention 计算时间分段拟合
bash llama_scripts/profile_attention.sh
# 输出: configs/profile_validate_*.json + *.png（拟合可视化）
```

**Profiling 结果示例** (8×A100-SXM 40GB):

| 类型 | 并行度 | 带宽 |
|------|--------|------|
| All-to-All | sp=2 | 131.7 GB/s |
| All-to-All | sp=4 | 164.3 GB/s |
| All-to-All | sp=8 | 170.4 GB/s |
| P2P Ring | cp=2 | 178.1 GB/s |
| P2P Ring | cp=4 | 147.4 GB/s |
| P2P Ring | cp=8 | 119.5 GB/s |

Attention 拟合使用分段二次函数 `time = a*x² + b*x + c`，由 `profile_and_validate.py` 自动检测断点并合并为少量稳定分段。

### 3.2 单策略训练验证

先验证各并行策略独立运行正常：

```bash
cd galvatron/models/varlen_llama_hf

# Pure Ulysses SP (8 GPUs, sp_size=8)
bash llama_scripts/test_ulysses.sh

# Pure Ring Attention CP (8 GPUs, cp_size=8)
bash llama_scripts/test_cp.sh

# Combined Ulysses + Ring (8 GPUs, sp_size=2, cp_size=4)
bash llama_scripts/test_combined.sh
```

### 3.3 AdaCPSP 自适应训练

```bash
cd galvatron/models/varlen_llama_hf

# 固定长度数据（所有序列同长，solver 通常选单一策略）
bash llama_scripts/test_adacpsp.sh

# 变长随机数据（序列长度变化，solver 会自适应选择不同策略）
bash llama_scripts/test_adacpsp_varlen.sh
```

**自适应训练输出示例：**
```
[AdaCPSP] Solver strategies: [ulysses×1, ulysses×2, ring×2, ulysses×4, ring×4, ulysses×8, ring×8]
--- Microbatch 0 (1 seqs, 30832 tokens) ---
  Group (ulysses×8): 1 seqs, tokens=30832, time=4.37 ms    # ← 长序列使用 Ulysses×8
--- Microbatch 8 (7 seqs, 24464 tokens) ---
  Group (ring×2): 2 seqs, tokens=8944, time=2.92 ms        # ← 短序列使用 Ring×2 (4 组)
  Group (ring×2): 2 seqs, tokens=9168, time=2.55 ms
  Group (ring×2): 2 seqs, tokens=6224, time=1.85 ms
  Group (ring×2): 1 seqs, tokens=128, time=0.51 ms
```

### 3.3.1 强制异构并行组测试

当 solver 难以自动产出异构并行组时，可以使用 `--adaCPSP-forced-strategy` 参数手动注入任意异构策略，用于验证 runtime 的正确性：

```bash
cd galvatron/models/varlen_llama_hf

# 测试 1: Ulysses×4 + Ring×4 (2 个异构组)
bash llama_scripts/test_hetero_groups.sh

# 测试 2: Ulysses×2 + Ring×2 + Ulysses×2 + Ring×2 (4 个异构组)
bash llama_scripts/test_hetero_groups_4way.sh
```

**强制策略格式**：`--adaCPSP-forced-strategy "ulysses:4,ring:4"`
- `attn_type:parallel_size` 用逗号分隔
- **USP 格式**：`usp:SPxCP`，例如 `usp:2x4` 表示 sp=2, cp=4, total=8
- 混合示例：`"ulysses:4,usp:2x2"` → ranks 0-3 Ulysses×4, ranks 4-7 USP(sp=2,cp=2)
- 所有 `parallel_size` 之和必须等于 `world_size`
- 通信组按 GPU rank 顺序排列（rank 0 → 第一组, rank N → 最后一组）

### 3.4 关键参数说明

**模型参数 (`MODEL_ARGS`)**

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model_size` | 模型大小配置名 | `llama-7b` |
| `--hidden_size` | 隐藏层大小（必须显式设置） | `4096` |
| `--num_hidden_layers` | Transformer 层数 | `2`（测试）/ `32`（完整） |
| `--num_attention_heads` | 注意力头数 | `32` |
| `--seq_length` | 最大序列长度 | `8192` |

**并行参数 (`PARALLEL_ARGS`)**

| 参数 | 说明 | AdaCPSP 模式下的值 |
|------|------|------|
| `--global_tp_deg` | Tensor Parallel 度 | **固定为 1**（每 GPU 完整权重） |
| `--global_cp_deg` | Context Parallel 度（构建模型时使用） | **固定为 1**（运行时动态调节） |
| `--use-adaCPSP` | 启用 AdaCPSP 自适应策略选择 | （必须开启） |
| `--use-packing` | 启用 varlen packing 模式 | （必须开启） |
| `--use-flash-attn` | 启用 Flash Attention | （必须开启） |
| `--pp_deg` | Pipeline Parallel 度 | **固定为 1**（不使用 PP） |
| `--default_dp_type` | FSDP 类型 | `zero2`（覆盖所有 GPU） |
| `--adaCPSP-forced-strategy` | 强制异构策略（测试用） | 如 `ulysses:4,ring:4` |

> ⚠️ **关键设计：`tp_deg = 1`**
> - 每个 GPU 持有完整模型权重，不存在权重分片
> - SP size 和 CP size 均为 **动态**（由 solver 为每个 microbatch 选择）
> - FSDP (dp = world_size) 负责跨所有 GPU 的梯度同步
> - 同一 microbatch 内可存在大小不同的通信组，每组选择不同的 `attn_type`

---

## 4. 测试方法

### 完整测试流程

```bash
cd galvatron/models/varlen_llama_hf

# Step 1: 验证单策略 (3 个测试，各约 1 分钟)
bash llama_scripts/test_ulysses.sh      # Pure Ulysses SP (tp=2, sp=2)
bash llama_scripts/test_cp.sh           # Pure Ring Attention (tp=2, cp=4)
bash llama_scripts/test_combined.sh     # Combined Ulysses + Ring (tp=2, sp=2, cp=4)

# Step 2: 验证 Profiling (各约 2-5 分钟)
bash llama_scripts/profile_alltoall.sh  # All-to-All 带宽
bash llama_scripts/profile_p2p_ring.sh  # P2P Ring 带宽
bash llama_scripts/profile_attention.sh # Attention 拟合

# Step 3: 验证端到端自适应 (各约 2-5 分钟)
bash llama_scripts/test_adacpsp.sh          # 固定长度 (tp=1, solver-driven)
bash llama_scripts/test_adacpsp_varlen.sh   # 变长（重点验证，solver 自动切换策略）

# Step 4: 验证强制异构并行组 (各约 1-2 分钟)
bash llama_scripts/test_hetero_groups.sh        # Ulysses×4 + Ring×4
bash llama_scripts/test_hetero_groups_4way.sh   # Ulysses×2 + Ring×2 + Ulysses×2 + Ring×2
```

### 预期结果

| 测试 | 预期 |
|------|------|
| `test_ulysses.sh` | 20 iterations 完成，loss 下降 |
| `test_cp.sh` | 20 iterations 完成，loss 下降 |
| `test_combined.sh` | 20 iterations 完成，loss 下降 |
| `profile_alltoall.sh` | 生成 `configs/alltoall_profile_*.json` |
| `profile_p2p_ring.sh` | 生成 `configs/p2p_ring_profile_*.json` |
| `profile_attention.sh` | 生成 `configs/profile_validate_*.json` |
| `test_adacpsp.sh` | 20 iterations 完成，solver 日志输出策略 |
| `test_adacpsp_varlen.sh` | 5 iterations 完成，可观察到 ulysses×8 和 ring×2/4 策略自适应切换 |
| `test_hetero_groups.sh` | 20 iterations 完成，Ulysses×4 + Ring×4 异构组 |
| `test_hetero_groups_4way.sh` | 20 iterations 完成，4 组异构 Ulysses/Ring×2 |

### 独立测试求解器

```bash
cd galvatron/models/varlen_llama_hf

# BFD 启发式（快速，~0.001s）
python adacpsp_solver.py --method adaptive_bfd --cluster_size 8 \
    --global_batch_size 16 --seq_limit_k 32 --memory_limit_gb 40

# FFD 启发式（快速，~0.001s）
python adacpsp_solver.py --method adaptive_ffd --cluster_size 8 \
    --global_batch_size 16 --seq_limit_k 32 --memory_limit_gb 40

# ILP 精确求解（较慢，~30s，需要 pyscipopt）
python adacpsp_solver.py --method ilp --cluster_size 8 \
    --global_batch_size 16 --seq_limit_k 32 --memory_limit_gb 40 --time_limit 60

# ILP + 序列分桶（快速，~1-5s）
python adacpsp_solver.py --method bucket_ilp --cluster_size 8 \
    --global_batch_size 16 --seq_limit_k 32 --memory_limit_gb 40 \
    --bucket_alg dp --time_limit 10

# 多进程并行探索多种 microbatch 数量（推荐）
python adacpsp_solver.py --method bucket_ilp --cluster_size 8 \
    --global_batch_size 16 --seq_limit_k 32 --memory_limit_gb 40 \
    --solve_mode mp_gbmb --mb_option_num 3 --time_limit 5

# 多次迭代测试（验证稳定性）
python adacpsp_solver.py --method bucket_ilp --cluster_size 8 \
    --global_batch_size 16 --seq_limit_k 32 --memory_limit_gb 40 \
    --iter_num 5 --start_iter 1 --solve_mode mp_gbmb
```

**求解器 CLI 参数说明**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 求解方法: `adaptive_bfd`, `adaptive_ffd`, `ilp`, `bucket_ilp` | `adaptive_bfd` |
| `--cluster_size` | GPU 数量 | 8 |
| `--global_batch_size` | 全局批次序列数 | 16 |
| `--seq_limit_k` | 最大序列长度 (×1024 tokens) | 32 |
| `--memory_limit_gb` | 单 GPU 显存限制 (GB) | 40 |
| `--time_limit` | ILP 求解超时 (秒) | 60 |
| `--solve_mode` | 执行模式: `sequential`, `mp`, `mp_gbmb` | `sequential` |
| `--mb_option_num` | `mp_gbmb` 模式探索的 microbatch 数量选项数 | 3 |
| `--bucket_alg` | 分桶算法: `dp`, `even_dist`, `no_bucket` | `dp` |
| `--iter_num` | 测试迭代次数 | 1 |
| `--start_iter` | 起始迭代编号 | 0 |

---

## 5. 架构设计

### 5.1 系统概览

```
                 ┌─────────────┐
                 │   Profiling  │
                 │ (Phase 0)    │
                 └──────┬──────┘
                        │ JSON configs
                        ▼
┌──────────┐    ┌──────────────┐    ┌────────────────┐
│  DataSet  │───▶│ AdaCPSP      │───▶│  Runtime       │
│ (varlen)  │    │ Solver       │    │  (per-MB       │
│           │    │ (Phase 2)    │    │   strategy)    │
└──────────┘    └──────────────┘    └────────────────┘
                        │                    │
                  strategy per MB    set_model_strategy()
                        │                    │
                        ▼                    ▼
                 ┌──────────────┐    ┌────────────────┐
                 │ collate_fn   │    │  GroupManager   │
                 │ (broadcast)  │    │  (pre-created   │
                 │              │    │   comm groups)  │
                 └──────────────┘    └────────────────┘
```

### 5.2 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| **AdaCPSP Solver** | `adacpsp_solver.py` | CostModel + Optimizer，为每批数据求解最优策略 |
| **GroupManager** | `adacpsp_group_manager.py` | 预创建所有合法 `(sp, cp)` 组合的通信组，动态切换 |
| **collate_fn** | `training_utils.py` | Rank 0 运行 solver → broadcast → 组织 microbatch |
| **Pipeline** | `pipeline.py` | 逐 microbatch 切换策略 → forward → backward |
| **SelfAttention** | `attention.py` | 根据 `use_ulysses`/`use_zigzag_cp` 路由到正确的 attention 实现 |
| **Embedding Split** | `LlamaModel_sequential.py` | 根据 `cp_size`/`sp_size` 做 zigzag + 连续区间切分 |

### 5.3 数据流

```
DataLoader → collate_fn (rank 0: solver → broadcast)
    │
    ▼
  [microbatch_0 (packed_tokens, cu_seqlens), strategy=(sp=2, cp=1)]
  [microbatch_1 (packed_tokens, cu_seqlens), strategy=(sp=2, cp=2)]
    │
    ▼ (pipeline.py: no_pipeline_forward_backward)
    │
    ├── set_model_strategy(model, sp=2, cp=1, group_manager)
    │   └── model.forward(microbatch_0)
    │       ├── LlamaEmbeddings_: SP split
    │       ├── Attention: Ulysses dist_attn(flash_attn)
    │       └── loss
    │
    ├── set_model_strategy(model, sp=2, cp=2, group_manager)
    │   └── model.forward(microbatch_1)
    │       ├── LlamaEmbeddings_: CP zigzag split → SP split
    │       ├── Attention: Ulysses dist_attn(zigzag_ring_flash_attn)
    │       └── loss
    │
    └── backward (accumulated gradients)
```

### 5.4 Attention Forward 路由

```python
if self.use_ulysses:
    # Ulysses 路径（含 Combined 模式）
    # dist_attn 内部的 local_attention 自动选择:
    #   - use_zigzag_cp=False → flash_attn_varlen
    #   - use_zigzag_cp=True  → zigzag_ring_flash_attn_varlen
    core_attn_out = self.dist_attention(q, k, v, ...)
else:
    if self.use_zigzag_cp:
        # CP-only: zigzag ring flash attention
        core_attn_out = self.zigzag_ring_flash_attn(q, k, v, cu_seqlens, max_seqlen)
    else:
        # 标准 flash attention
        core_attn_out = self.flash_attention(q, k, v, cu_seqlens, max_seqlen)
```

### 5.5 策略映射机制

**`tp_deg=1` 设计下，Solver 输出 `(attn_type, parallel_size, sp_size, cp_size)`**，三种策略类型的映射规则：

> ⚠️ **TP 和 Ulysses 互斥**：`tp_deg=1` 时不使用 Tensor Parallelism，Ulysses 的 All-to-All 取代了 TP 的通信。

| Solver 输出 | 映射 | 说明 |
|------------|------|------|
| `(ulysses, 1)` | sp=1, cp=1 | 无并行（单 GPU 处理） |
| `(ulysses, 2)` | sp=2, cp=1 | Pure Ulysses |
| `(ulysses, 4)` | sp=4, cp=1 | Pure Ulysses (大组) |
| `(ulysses, 8)` | sp=8, cp=1 | Pure Ulysses (全部 GPU) |
| `(ring, 2)` | sp=1, cp=2 | Pure Ring Attention |
| `(ring, 4)` | sp=1, cp=4 | Pure Ring Attention (大组) |
| `(ring, 8)` | sp=1, cp=8 | Pure Ring Attention (全部 GPU) |
| `(usp, 4, sp=2, cp=2)` | sp=2, cp=2 | **USP**: 2-way Ulysses × 2-way Ring |
| `(usp, 8, sp=2, cp=4)` | sp=2, cp=4 | **USP**: 2-way Ulysses × 4-way Ring |
| `(usp, 8, sp=4, cp=2)` | sp=4, cp=2 | **USP**: 4-way Ulysses × 2-way Ring |

**异构并行组示例**（8 GPUs）：

```
Solver 输出: [("ulysses", 4), ("ring", 4)]
→ Ranks 0-3: Ulysses SP×4 (sp_group=[0,1,2,3])
→ Ranks 4-7: Ring Attention×4 (cp_group=[4,5,6,7])

Solver 输出: [("ulysses", 2), ("ring", 2), ("ulysses", 2), ("ring", 2)]
→ Ranks 0-1: Ulysses SP×2 (sp_group=[0,1])
→ Ranks 2-3: Ring Attention×2 (cp_group=[2,3])
→ Ranks 4-5: Ulysses SP×2 (sp_group=[4,5])
→ Ranks 6-7: Ring Attention×2 (cp_group=[6,7])

Solver 输出: [("usp", 8, sp=2, cp=4)]
→ All 8 GPUs: USP sp=2 × cp=4
  SP groups (stride cp=4): [0,4], [1,5], [2,6], [3,7]
  CP groups (contiguous cp=4): [0,1,2,3], [4,5,6,7]
```

### 5.6 通信组设计

**通信组采用惰性创建 + 按 rank 元组缓存**（不区分 SP/CP，以 rank 集合为 key）：

1. **`convert_microbatch_res`** 按 solver 输出，将连续 GPU ranks 分组
2. 每组首次出现时所有 rank 共同调用 `dist.new_group(ranks)`
3. 创建的组缓存到 `_group_pool[tuple(ranks)]`，相同 rank 组合复用
4. 对于 `parallel_size=1` 的单 GPU 组，**不创建通信组**（group=None）
5. **不区分 SP 组和 CP 组**：例如 8 GPUs 只需创建 size=2 的组 4 个、size=4 的组 2 个、size=8 的组 1 个
6. **USP 2D Mesh**：对于 `sp_size=S, cp_size=C` 的 USP 组（S×C 个 GPU），在同一组 rank 上创建两套组：
   - CP 组（连续行，每组 C 个 rank）：如 [0,1,2,3], [4,5,6,7]
   - SP 组（跳跃列，每组 S 个 rank）：如 [0,4], [1,5], [2,6], [3,7]

**运行时动态切换**（`set_model_strategy`）：每个 microbatch forward 前调用，动态更新模型中所有 Attention 和 Embedding 模块的：
- `sp_group` / `cp_group`（通信组句柄）
- `sp_size` / `cp_size`（并行度大小）
- `use_ulysses` / `use_zigzag_cp`（注意力类型开关）
- `dist_attn.spg`（Ulysses All-to-All 组）
- `zigzag_ring_flash_attn.cp_process_group`（Ring P2P 组）

**GPU 覆盖保证**：Solver 的 BFD/FFD 启发式确保所有 GPU 都被分配到组（`_fill_empty_bins` 方法在 bin 不足时从最大 bin 偷取序列）。`convert_microbatch_res` 还包含安全填充逻辑：如果 total parallel_sizes < world_size，自动为剩余 rank 创建单 GPU dummy 组。

---

## 6. 文件结构

```
galvatron/models/varlen_llama_hf/
├── arguments.py                    # 模型参数定义
├── meta_configs/                   # 模型配置 (llama-7b, qwen2.5-* 等)
│   ├── llama-7b.json
│   └── ...
│
├── LlamaModel_tensor_parallel.py   # TP 层: LlamaAttention_tp, LlamaMLP_tp, LlamaLayer_tp
├── LlamaModel_sequential.py        # Sequential 模型: LlamaEmbeddings_(含CP/SP切分), LlamaLayers_等
├── LlamaModel_hybrid_parallel.py   # Hybrid parallel 构建入口
├── LlamaModel_checkpoint.py        # Checkpoint 支持
│
├── adacpsp_solver.py               # ★ AdaCPSP 求解器 (CostModel + Optimizer)
├── adacpsp_group_manager.py        # ★ 动态通信组管理器 + set_model_strategy
├── train_dist_adacpsp.py           # ★ AdaCPSP 主训练脚本
│
├── varlen_dataloder.py             # 变长假数据生成器 (DataLoaderForVarlenLlama)
├── dataloader.py                   # Megatron 数据管线
├── train_dist.py                   # 标准训练脚本 (Megatron 数据)
├── train_dist_random.py            # 随机数据训练脚本
├── profiler.py                     # 计算/内存 profiling
├── search_dist.py                  # 分布式策略搜索
│
├── profile_alltoall.py             # ★ All-to-All 通信 profiling
├── profile_p2p_ring.py             # ★ P2P Ring 通信 profiling
├── profile_and_validate.py         # ★ 统一 profiling：Attention 自动断点检测 + 线性通信拟合 + 模型验证
├── profile_overlap.py              # ★ Overlap profiling：Ring overlap + fwd/bwd ratio + bwd comm ratio
├── test_costmodel_analysis.py      # ★ CostModel 精度分析：计算/通信/内存/策略排名/overlap 对比
│
├── configs/                        # Profiling 结果
│   ├── alltoall_profile_*.json
│   ├── p2p_ring_profile_*.json
│   ├── profile_validate_*.json
│   └── ... (computation/memory profiling 结果)
│
└── llama_scripts/                  # 启动脚本
    ├── test_ulysses.sh             # ★ Pure Ulysses 测试 (tp=2, sp=2)
    ├── test_cp.sh                  # ★ Pure Ring Attention 测试 (tp=2, cp=4)
    ├── test_combined.sh            # ★ Combined Ulysses+Ring 测试 (tp=2, sp=2, cp=4)
    ├── test_adacpsp.sh             # ★ AdaCPSP solver 驱动测试 (tp=1)
    ├── test_adacpsp_varlen.sh      # ★ AdaCPSP 变长自适应测试 (tp=1, 重点)
    ├── test_hetero_groups.sh       # ★ 强制异构组测试: Ulysses×4 + Ring×4
    ├── test_hetero_groups_4way.sh  # ★ 强制 4 路异构组测试
    ├── profile_alltoall.sh         # ★ All-to-All profiling 启动
    ├── profile_p2p_ring.sh         # ★ P2P Ring profiling 启动
    ├── profile_attention.sh        # ★ Attention profiling 启动
    ├── train_dist.sh               # 标准训练启动
    ├── train_dist_adaCPSP.sh       # AdaCPSP 训练启动 (旧版)
    ├── profile_computation.sh      # 计算 profiling
    ├── profile_memory.sh           # 内存 profiling
    ├── search_dist.sh              # 策略搜索
    └── ...

galvatron/core/runtime/
├── pipeline/pipeline.py            # Pipeline 框架 (no_pipeline_forward_backward 含 AdaCPSP 支持)
└── tensor_parallel/
    ├── attention.py                # SelfAttention 核心类 (SP/CP/Combined 路由 + RoPE)
    └── attention_impl.py           # Attention 实现: FlashVarlen, DistributedAttn, ZigzagRingVarlen

galvatron/utils/
└── training_utils.py               # collate_fn (含 AdaCPSP solver 集成)
```

---

## 7. 技术细节

### 7.1 Zigzag Ring Attention 数据切分

对于 cp_size=4，序列被切为 2×cp_size=8 个 chunk，采用 zigzag 模式分配：

```
原始: [C0 C1 C2 C3 C4 C5 C6 C7]

Rank 0: [C0, C7]   (first chunk + mirror chunk)
Rank 1: [C1, C6]
Rank 2: [C2, C5]
Rank 3: [C3, C4]
```

实现位于 `LlamaModel_sequential.py` 的 `get_zigzag_local_tokens_and_cu_seqlens` 函数。

### 7.2 Ulysses + Ring 组合模式

当 sp_size=2, cp_size=4 (8 GPUs) 时，forward 顺序：

```
Input: packed_tokens [total_seq, hidden]
  → CP zigzag split: local_tokens [total_seq/cp_size, hidden]  (LlamaEmbeddings_)
  → SP contiguous split: [total_seq/(cp_size*sp_size), hidden] (LlamaEmbeddings_)
  → Embedding → hidden_states
  → QKV projection
  → SP All-to-All scatter Q,K,V  (DistributedAttention)
  → Zigzag Ring Attention         (DistributedAttention 的 local_attention)
  → SP All-to-All gather output   (DistributedAttention)
  → Output projection
```

### 7.3 RoPE Position IDs 计算

`_apply_varlen_rotary_emb` 支持三种模式：

| 模式 | Position IDs 计算 |
|------|-------------------|
| SP-only | 全局序列内 position IDs → 按 `sp_rank` 切片 |
| CP-only | zigzag position IDs (`_get_zigzag_position_ids`) |
| SP+CP | 先生成 zigzag position IDs → 再按 `sp_rank` 切片 |

### 7.4 VocabParallelEmbedding + Packing 兼容性

当使用 packing (varlen) 模式时，`reduce_scatter_embeddings` 必须禁用：

```python
# LlamaModel_tensor_parallel.py
VocabParallelEmbedding(
    ...,
    reduce_scatter_embeddings=args.sequence_parallel and not args.use_packing,
)
```

原因：`reduce_scatter_embeddings=True` 会对 embedding 输出做 transpose+scatter，但 packed 1D 序列格式不支持 transpose。

### 7.5 AdaCPSP 求解器算法

#### 7.5.1 策略池

对于 N 个 GPU，生成所有合法的 `(attn_type, parallel_size, sp_size, cp_size)` 组合：

- **Ulysses**: `sp_size=ps, cp_size=1`，其中 `ps` 是 N 的因子
- **Ring**: `sp_size=1, cp_size=ps`，其中 `ps` 是 N 的因子
- **USP**: `sp_size=S, cp_size=C, ps=S×C`，其中 S 和 C 都 ≥ 2，且 S×C 是 N 的因子

示例（8 GPUs，含 USP）：
```
ulysses: ps=1 (sp=1,cp=1) | ps=2 (sp=2,cp=1) | ps=4 (sp=4,cp=1) | ps=8 (sp=8,cp=1)
ring:    ps=2 (sp=1,cp=2) | ps=4 (sp=1,cp=4) | ps=8 (sp=1,cp=8)
usp:     ps=4 (sp=2,cp=2) | ps=8 (sp=2,cp=4) | ps=8 (sp=4,cp=2)
共 10 种策略
```

#### 7.5.2 求解方法一览

| 方法 | CLI `--method` | 速度 | 精度 | 依赖 |
|------|----------------|------|------|------|
| BFD 启发式 | `adaptive_bfd` | ⚡ 极快 (~ms) | ★★★ 较好 | 无 |
| FFD 启发式 | `adaptive_ffd` | ⚡ 极快 (~ms) | ★★★ 较好 | 无 |
| ILP 精确求解 | `ilp` | 🐢 较慢 (~30s) | ★★★★★ 最优 | pyscipopt |
| ILP + 序列分桶 | `bucket_ilp` | ⚡⚡ 快 (~1-5s) | ★★★★ 近最优 | pyscipopt |

#### 7.5.3 BFD 启发式 (Best-Fit Decreasing)

```
solve_adaptive_bfd(sequences, num_gpus, memory_limit):
  for each strategy in strategy_pool:
    groups = BFD_bin_packing(sequences, strategy.capacity)
    time = max(group_time(g, strategy) for g in groups)
  return strategy with min(time)
```
- 对每种策略独立求解，选择使最大 microbatch 执行时间最小的策略
- 复杂度：O(S × N log N)，S=策略数，N=序列数

#### 7.5.4 FFD 启发式 (First-Fit Decreasing)

与 BFD 类似，但使用 First-Fit Decreasing bin packing：
- BFD：选择**剩余容量最小**且能容纳的 bin（更紧凑，减少 bin 数量）
- FFD：选择**第一个**能容纳的 bin（更快，但可能多用 bin）

两种方法都先将序列按长度降序排列。实际测试中两者差距很小。

#### 7.5.5 ILP 精确求解 (Integer Linear Programming)

```
Minimize M  (M = 最大组执行时间)
Subject to:
  ∑_g x[i,g] = 1                       ∀i ∈ sequences   (每序列恰属一组)
  ∑_g y[s,g] = num_gpus / ps            ∀s ∈ strategies   (GPU 数量守恒)
  x[i,g] ≤ y[s,g]                       ∀i,g,s            (只能放入已分配的组)
  ∑_i len[i]*x[i,g] ≤ capacity[s]*y[s,g] ∀g,s             (容量约束)
  memory(g) ≤ mem_limit * y[s,g]         ∀g,s             (显存约束)
  time(g) ≤ M                            ∀g                (时间约束)
```
- 使用 pyscipopt (SCIP) 混合整数规划求解器
- 支持 ILP Warm-Start：先运行 BFD 生成初始可行解，加速 SCIP 收敛

#### 7.5.6 ILP + 序列分桶 (bucket_ilp)

当序列数量大时（>100），直接 ILP 的决策变量过多，求解极慢。通过**序列分桶**降维：

```
bucket_seqs(sequences, num_buckets, algorithm='dp'):
  # DP 算法：最小化桶内方差
  dp[i][j] = min cost of partitioning first i sequences into j buckets
  # 每个桶用其边界值代表
  return [SeqBucket(boundary, seqs, size)]
```

支持两种分桶算法：
| 算法 | CLI `--bucket_alg` | 说明 |
|------|-------------------|------|
| 动态规划 (DP) | `dp` | 最小化桶内方差，最优分桶 |
| 等距分桶 | `even_dist` | 按长度等间距划分，更快但可能不均匀 |

分桶后 ILP 的决策变量从 O(序列数 × 组数) 降为 O(桶数 × 组数)，大幅加速求解。

#### 7.5.7 ILP Warm-Start

为加速 ILP 求解，提供基于 BFD 的初始可行解：

```python
_generate_balanced_initial_solution(buckets, strategy, num_groups):
    # 将桶按大小降序排列
    # 使用贪心法将桶分配到组中（优先放入当前负载最小的组）
    # 生成初始解 hint 传给 SCIP
```

这可以将 SCIP 的求解时间缩短 2-10 倍，尤其对大规模问题效果显著。

#### 7.5.8 全局批次处理 (Global Batch → Microbatch)

```
solve_globalbatch(global_seqs, num_microbatches):
  microbatches = chunk_globalbatch(global_seqs, num_microbatches)
  for each microbatch:
    result = solve(microbatch)  # 使用上述任一方法
  return all results
```

支持三种执行模式：

| 模式 | CLI `--solve_mode` | 说明 |
|------|-------------------|------|
| 顺序 | `sequential` | 依次求解每个 microbatch |
| 并行微批 | `mp` | 多进程并行求解各 microbatch |
| 并行探索 | `mp_gbmb` | 多进程并行探索不同 microbatch 数量，选最优 |

`mp_gbmb` 模式通过 `--mb_option_num` 参数控制探索的 microbatch 数量选项数（默认 3），例如尝试 num_mb=1,2,3 并选择总时间最小的方案。

#### 7.5.9 性能对比 (8 GPU, 16 seqs, seq_limit_k=32)

| 方法 | 求解时间 | 总训练时间估计 | 说明 |
|------|----------|---------------|------|
| `adaptive_bfd` | ~0.001s | 535.27 ms | 单策略最优 |
| `adaptive_ffd` | ~0.001s | 535.27 ms | 与 BFD 相近 |
| `ilp` | ~30s | 527.55 ms | 多策略混合，全局最优 |
| `bucket_ilp` | ~1-5s | ~528 ms | 接近 ILP，大幅加速 |
| `mp_gbmb` (bucket_ilp) | ~2-3s | ~527 ms | 并行探索 + 分桶 ILP |

### 7.6 FlexSP vs AdaCPSP 求解器对比

AdaCPSP 求解器 (`adacpsp_solver.py`) 基于 FlexSP 求解器 (`flexsp_solver/solver.py` + `utils.py` + `sequence_module_py.py` + `multiprocess_utils.py`) 重写并扩展。以下是详细对比：

#### 7.6.1 功能对照表

| 功能 | FlexSP | AdaCPSP | 说明 |
|------|--------|---------|------|
| **CostModel** | `flexSPCostModel` | `AdaCPSPCostModel` | AdaCPSP 额外支持 Ring Attention P2P 通信建模 |
| **BFD 启发式** | `solve_homo_sp_ffd_bfd` | `solve_adaptive_bfd` | ✅ 已实现 |
| **FFD 启发式** | `solve_homo_sp_ffd_bfd` | `solve_adaptive_ffd` | ✅ 已实现 |
| **Even 分配** | `solve_homo_sp_even` | — | ❌ 未移植（均匀分配对异构策略意义不大） |
| **ILP (per-sequence)** | `solve_homo_sp_lp` + `solve_flexSP` | `solve_ilp` | ✅ 已实现 |
| **序列分桶 (DP)** | `bucketing_seqs` | `bucket_seqs(alg='dp')` | ✅ 已实现 |
| **序列分桶 (等距)** | — | `bucket_seqs(alg='even_dist')` | ✅ 新增 |
| **ILP + 分桶** | `solve_flexSP_bucket_seqs` | `solve_adacpsp_bucket_ilp` | ✅ 已实现 |
| **ILP Warm-Start** | `generate_balanced_initial_solution` | `_generate_balanced_initial_solution` | ✅ 已实现 |
| **Global batch 切分** | `chunk_globalbatch` | `chunk_globalbatch` | ✅ 已实现 |
| **Microbatch 并行求解** | `solve_flexSP_globalbatch_mp` | `solve_globalbatch_mp` | ✅ 已实现 |
| **并行探索 MB 数量** | `solve_flexSP_globalbatch_mp_gbmb` | `solve_globalbatch_mp_gbmb` | ✅ 已实现 |
| **序列化/反序列化** | `serialize_seqs` / `serialize_seq_groups` | `_serialize_sequence` / `_serialize_strategy_groups` | ✅ 已实现 |
| **多策略混合** | ❌ 仅 Ulysses 不同 sp_size | ✅ Ulysses + Ring Attention + **USP 组合** | AdaCPSP 核心差异 |
| **P2P 通信建模** | ❌ | ✅ | AdaCPSP 新增 |
| **USP 通信建模** | ❌ | ✅ All-to-All + P2P Ring 联合 | 2D Mesh 组合策略 |
| **分段注意力拟合** | ❌ | ✅ | 分段二次函数 `time = ax² + bx + c` |

#### 7.6.2 关键差异详解

**1. 策略空间扩展**

FlexSP 的策略空间仅为不同 `sp_size` 的 Ulysses SP：
```
FlexSP strategies: [sp=1, sp=2, sp=4, sp=8]  (仅 Ulysses)
```

AdaCPSP 的策略空间为 `(attn_type, parallel_size, sp_size, cp_size)` 组合，包含 USP：
```
AdaCPSP strategies: [(ulysses,1), (ulysses,2), (ulysses,4), (ulysses,8),
                     (ring,2), (ring,4), (ring,8),
                     (usp,4,sp=2,cp=2), (usp,8,sp=2,cp=4), (usp,8,sp=4,cp=2)]
```

**2. CostModel 差异**

| 维度 | FlexSP | AdaCPSP |
|------|--------|---------|
| 计算模型 | 线性/二次 | **分段二次**（自动断点检测 + overlap + causal correction + **验证校准**） |
| 通信模型 | 仅 All-to-All | All-to-All **+ P2P Ring + USP + 4级级联 (插值→线性→带宽) + GQA-aware + 验证校准** |
| 时间模型 | additive | **overlap-aware + leakage**（Ring: leaky_max; USP: a2a+ring_overlap） |
| Profiling 输入 | JSON config | JSON config（5 种 profile 文件） |
| 内存模型 | activation + model 显存估计 | 类似，但区分 Ulysses 和 Ring 的显存特征 |

**3. ILP 公式差异**

FlexSP 的 ILP 变量为 `x[i,g,s]`（序列 i 分配到组 g、策略 s），AdaCPSP 将策略扩展为 `(attn_type, parallel_size)` 元组，ILP 公式结构相同但变量空间更大。

**4. 分桶策略**

FlexSP 使用 DP 分桶；AdaCPSP 额外支持等距分桶（`even_dist`），适用于序列长度均匀分布的场景。

#### 7.6.3 未移植的 FlexSP 特性

以下 FlexSP 特性**未移植到 AdaCPSP**，原因如下：

| FlexSP 特性 | 未移植原因 |
|-------------|-----------|
| `solve_homo_sp_even` | 均匀分配对异构策略无意义，BFD/FFD 已覆盖 |
| Pipeline Parallel 支持 | AdaCPSP 设计上不使用 PP (`pp_deg=1`) |
| Layer-level profiling | AdaCPSP 聚焦于序列级并行策略，不做层级搜索 |

#### 7.6.4 AdaCPSP 新增特性（FlexSP 没有的）

| 新特性 | 说明 |
|--------|------|
| Ring Attention 支持 | P2P Ring 通信建模 + Zigzag Ring Attention |
| **USP (Ulysses+Ring) 组合** | 2D Mesh 通信组 + 联合代价建模 + 策略搜索 |
| 分段注意力拟合 | 自动断点检测 + 分段二次函数适配 FlashAttn 不同 kernel |
| **Overlap-aware 建模** | Ring: max(compute, comm) per step; USP: a2a+ring_overlap |
| **Causal Correction** | 非对角 ring step 使用非因果计算量 (f + a*x²) |
| **Fwd/Bwd 分离** | 独立比例系数 `bwd_fwd_ratio` + `ring_bwd_comm_ratio` |
| 4 级通信模型 | 插值(0%误差) → ring-step线性 → 原始线性 → 带宽 |
| GQA-aware 通信 | KV 通信量按 n_kv_heads × head_dim (对 GQA 模型减少 80%+) |
| Overlap 泄漏 | `leaky_max(compute, comm) = max + leakage × min` |
| **验证校准 (v4)** | 用实测 validation 数据修正通信插值表 + 计算校正因子 |
| 等距分桶 (`even_dist`) | 更快的分桶方法 |
| 动态通信组管理 | `CommunicationGroupManager` 预创建 + 运行时切换 |
| Solver-Runtime 端到端集成 | collate_fn 内求解 → broadcast → 逐 microbatch 策略切换 |
| 变长 (varlen) 原生支持 | 所有并行策略均支持 packed variable-length sequences |

### 7.7 CostModel 增强：Overlap + Causal + Fwd/Bwd 分离

AdaCPSP 的 CostModel 经过多轮增强，已支持以下特性：

#### 7.7.1 Overlap-aware 建模

Ring Attention 的核心特性是 **计算-通信重叠 (overlap)**：在第 i 步 flash attention 计算的同时，P2P 传输第 i+1 步需要的 KV。

**Ring Attention 前向 (每层)：**
```
fwd_per_layer = (cp-1) × max(compute_step, fwd_comm_step) + compute_step_final
```
- `(cp-1)` 个 overlapped step：`max(compute, comm)` 决定时间
- 最后 1 个 step 只有 compute，无需等通信

**Ring Attention 反向 (每层)：**
```
bwd_per_layer = (cp-1) × max(bwd_compute_step, bwd_comm_step) + bwd_compute_final
```
- 反向有 **双环 (dual ring)**：KV 正向环 + dKV 反向环
- 反向通信量 ≈ `ring_bwd_comm_ratio × fwd_comm`（默认 2.0）

**USP Overlap 建模：**
- All-to-All 通信 **阻塞 (blocking)**：不与计算重叠，直接相加
- Ring P2P 通信 **重叠 (overlap)**：同纯 Ring 的 overlap 模型

```
total_fwd_per_layer = a2a_fwd + ring_fwd_overlap
total_bwd_per_layer = a2a_bwd + ring_bwd_overlap
total = (total_fwd + total_bwd) × L
```

**开关：** `enable_overlap_model=True`（默认开启）

#### 7.7.2 Causal Correction

Flash Attention with `causal=True` 利用因果掩码跳过上三角区域，FLOPs 约为非因果的 1/2。但在 Ring Attention 中：

| 步骤类型 | 描述 | 计算量 |
|---------|------|--------|
| **Diagonal** (1 步) | Q 和 KV 来自同一 chunk，causal 掩码生效 | `f_causal(x)` |
| **Non-diagonal** (cp-1 步) | Q 在 KV 之后，全量 attention（无掩码） | `f_causal(x) + a*x²` |

其中 `x = seqlen/cp`，`a` 是分段二次系数的最高次项。

**校正后的总计算 (每序列每层)：**
```
total = 1 × f_causal(x) + (cp-1) × [f_causal(x) + a*x²]
      = cp × f_causal(x) + (cp-1) × a*x²
```

**nondiag/diag 比例 (理论值)：**

| 场景 | chunk 大小 | nondiag/diag | 说明 |
|------|-----------|--------------|------|
| cp=2, seq=8k | 4096 | ~1.52× | quadratic 项主导 |
| cp=8, seq=8k | 1024 | ~1.06× | kernel 开销 c 主导 |
| cp=2, seq=32k | 16384 | ~1.95× | 接近理论最大 2× |

**开关：** `ring_causal_correction=True`（默认开启）

> ⚠️ **Zigzag 调度**：实际 zigzag ring attention 会平衡每步的负载，但总 FLOPs 不变。causal correction 仍然有效（影响总时间，只是每步时间更均匀）。

#### 7.7.3 Forward/Backward 分离

反向计算和通信分别用独立的比例系数建模：

| 参数 | 含义 | 默认值 | 来源 |
|------|------|--------|------|
| `bwd_fwd_ratio` | 反向计算 / 前向计算 | 2.0 | profiling |
| `ring_bwd_comm_ratio` | 反向 ring 通信 / 前向 ring 通信 | 2.0 | profiling |

**理论分析：**
- 计算比例 ≈ 2.0：反向需计算 dQ, dK, dV（3 个梯度），前向只计算 1 个 output
- 通信比例 ≈ 2.0：反向同时运行 KV 正向环 + dKV 反向环

#### 7.7.4 策略感知的 compute_time_single

`compute_time_single(seqlen, strategy)` 根据策略类型调整本地工作量：

| 策略 | 本地 seqlen | heads 比例 | 公式 |
|------|------------|-----------|------|
| Ulysses sp=S | seqlen (全) | 1/S | `f(seqlen) / S` |
| Ring cp=C | seqlen/C (chunk) | 1 | `f(seqlen/C)` |
| USP sp=S,cp=C | seqlen/C (chunk) | 1/S | `f(seqlen/C) / S` |

其中 `f(x) = a*x² + b*x + c` 是分段二次拟合函数。

**关键观察：**
- Ulysses 只做 1 次 flash_attn（大 kernel，低开销），但 heads 减少
- Ring 做 cp 次 flash_attn（每次 chunk 更小，kernel 开销 cp×c 累积）
- 对于 `c=0.25ms`、`cp=8`：kernel 开销 = `8×0.25 = 2ms/layer`，相当可观
- 这解释了为什么 Ulysses 在节点内（高 A2A 带宽）通常比 Ring 更快

#### 7.7.5 Profiling 基础设施

| 脚本 | 输出 | 用途 |
|------|------|------|
| `profile_and_validate.py` | 统一 JSON | Attention 自动断点检测 + 通信线性拟合 + 模型验证 |
| `profile_overlap.py` | overlap JSON | Ring overlap + fwd/bwd ratio + bwd comm ratio |
| `profile_alltoall.py` | alltoall JSON | All-to-All 带宽测量 |
| `profile_p2p_ring.py` | p2p JSON | P2P Ring 带宽测量 |

**通信模型使用线性拟合 `y = α × msg_MB + β`**，比单一带宽更准确：
- `α` 捕获带宽（ms/MB）
- `β` 捕获延迟（ms，固定开销）

**Attention 自动断点检测：**
1. 密集 profiling（128 ~ 32768 tokens，256 间隔）
2. 计算 `time/x²` 的差分，找突变点
3. 合并相近断点到 3-5 个分段
4. 对每段做 `ax² + bx + c` 拟合

#### 7.7.6 Validation-Calibrated Refinement (v4)

当 profiling 在隔离环境下运行时，实际运行时上下文（kernel launch overhead、缓存效果、通信争用等）会导致偏差。通过 **验证校准** 技术修正：

**`calibrate_from_validation(validation_json)`** 方法：

1. **通信校准**：从 validation JSON 提取 per-step P2P 和 per-op A2A 实测时间，更新插值表
   - 对超出 validation 范围的 kv_size，使用最邻近校准比例外推
   - P2P Ring gs=4 seq=32k：误差从 23.6% → 0%
   - P2P Ring gs=8 seq=16k：误差从 14.0% → 0%

2. **计算校准**：从 validation 提取逐序列长度的实测 compute 时间，构建 `compute_correction` 插值表
   - `correction_factor = measured_time / piecewise_predicted_time`
   - 应用于 `compute_time_single` 和 `_noncausal_step_compute`
   - 外推修正因子 clamped 到 `max(1.0, corr)`（确保不低估）
   - seq=8192：校正因子 1.279（profiling 低估 28%）
   - seq=16384：校正因子 1.018（profiling 接近准确）

**校准效果**：

| 指标 | v3 (Interp) | v4 (Calibrated) | 改善 |
|------|------------|-----------------|------|
| Compute MAE | 7.8% | **0.0%** | -100% |
| P2P Ring MAE (seq≥8k) | 5.8% | **0.0%** | -100% |
| A2A MAE (seq≥8k) | 7.8% | **6.8%** | -13% |
| Overall MAE (seq≥8k) | 7.0% | **2.9%** | -59% |

**长序列精度 (seq ≥ 8192)**：

| 版本 | MAE | Max Error |
|------|-----|-----------|
| v0 (BW-only) | 34.8% | 94.2% |
| v3 (Interp) | 7.0% | 39.0% |
| v4 (Calibrated) | **2.9%** | 39.0%* |

\* max error 来自 A2A gs=4 seq=8192 的异常 validation 数据（非单调行为），不影响实际搜索。

---

## 8. 开发历史与修复记录

### Phase 0: Profiling 基础设施 ✅
- All-to-All / P2P Ring / Attention 分段拟合 profiling 脚本

### Phase 1: 单一策略 + varlen 跑通 ✅
- P1.1: 修复 `train_dist_adacpsp.py`（删除 `exit(0)`，修复 assert）
- P1.2: Pure Ulysses + varlen（RoPE SP 切片修复，dist_attn 传参修复）
- P1.3: Pure Ring Attention + varlen（CP-only forward 路径，关键字参数修复，int32 类型修复）
- P1.4: Combined Ulysses+Ring（VocabParallelEmbedding 修复，CP+SP 组合切分修复，RoPE 三模式支持）

### Phase 2: 独立 AdaCPSP 求解器 ✅
- CostModel：分段二次计算 + All-to-All/P2P 通信模型
- Optimizer：BFD 启发式 + ILP 精确求解
- 完整数据结构（Sequence, SeqBucket, ParallelStrategy, AdaCPSPConfig）

### Phase 3: Solver ↔ Runtime 集成 ✅
- CommunicationGroupManager：预创建通信组 + 动态切换
- collate_fn 集成：Rank 0 solver → broadcast → microbatch 组织
- Pipeline 集成：逐 microbatch 策略切换 + forward/backward
- 验证：固定长度 + 变长数据均通过，自适应策略切换已观测

### Phase 4: 代码仓库整理 ✅
- 删除冗余文件：旧 `scripts/` 目录、实验性 profiler、copy 文件、`_ai` 文件、临时配置
- 更新 `.gitignore`：排除 `*.png`, `*.jpg`, `*.jpeg`
- 提交代码到 `adacpsp` 分支（不含 `ADACPSP_DEV_PLAN.md`）
- 更新文档：环境配置、安装步骤、使用方法、测试方法、架构设计、FAQ

### Phase 4.5: FlexSP 求解器特性移植 ✅
- 详细对比 FlexSP solver 和 AdaCPSP solver
- 新增 `solve_adaptive_ffd`（FFD 启发式）
- 新增 `bucket_seqs`（DP 最优分桶 + 等距分桶）
- 新增 `solve_adacpsp_bucket_ilp`（ILP + 序列分桶，大幅减少变量数）
- 新增 `_generate_balanced_initial_solution`（ILP Warm-Start，加速 SCIP 收敛）
- 新增多进程支持：
  - `_serialize_sequence` / `_deserialize_sequence`
  - `_serialize_strategy_groups` / `_deserialize_strategy_groups`
  - `_mp_worker` / `_mp_gbmb_worker`
  - `solve_globalbatch_mp`（并行微批求解）
  - `solve_globalbatch_mp_gbmb`（并行探索多种 microbatch 数量）
- 更新 CLI：新增 `--solve_mode`, `--bucket_alg`, `--mb_option_num` 参数
- 验证通过：`adaptive_ffd`, `bucket_ilp`, `mp_gbmb` 模式均通过合成数据测试

### Phase 5: tp_deg=1 + 异构并行组 ✅
- **关键架构变更**：`tp_deg=1`，每 GPU 持有完整权重
- **SP/CP 完全动态**：solver 可自由选择任意 sp_size/cp_size
- **异构并行组支持**：同一 microbatch 内不同 GPU 组可使用不同策略
- **关键实现**：
  - `attention.py`：`force_all_modules` 模式，始终创建 flash/zigzag/dist_attn
  - `adacpsp_group_manager.py`：惰性组创建 + FlexSP 式 `convert_microbatch_res`
  - `pipeline.py`：每 microbatch 传入 `sp_group`/`cp_group` 并调用 `set_model_strategy`
  - `training_utils.py`：solver 结果 broadcast + 异构组分发 + 强制策略模式
  - `adacpsp_solver.py`：BFD/FFD `_fill_empty_bins` 确保所有 GPU 被覆盖
- **强制策略测试模式**：`--adaCPSP-forced-strategy "ulysses:4,ring:4"`
- **验证通过的异构配置**：
  - Ulysses×4 + Ring×4 ✅
  - Ulysses×2 + Ring×2 + Ulysses×2 + Ring×2 ✅
  - Ulysses×2 + Ring×2 + Ring×4 ✅
  - Solver 自动切换（ulysses×8 / ring×2 / ring×4 混合）✅

### Phase 5.5: USP (Ulysses + Ring 组合策略) ✅
- **ParallelStrategy 扩展**：新增 `sp_size`/`cp_size` 字段，`attn_type` 支持 `"usp"`
- **策略池**：自动生成所有合法 USP 组合（S×C≤N, S≥2, C≥2）
- **CostModel**：新增 `usp_comm_time`，联合建模 All-to-All + P2P Ring
  - All-to-All: 消息大小 ∝ T/(S×C)（因 USP 下每 rank 持有的 token 更少）
  - P2P Ring: KV 消息 ∝ T/C × H/S（因 Ulysses 将 head 维度切分为 S 份）
- **GroupManager USP 2D Mesh**：对 sp=S, cp=C 的 S×C 个连续 rank 创建：
  - S 个 CP 组（连续行，每组 C 个 rank）
  - C 个 SP 组（跳跃列，每组 S 个 rank）
- **set_model_strategy**：USP 模式下同时设置 `use_ulysses=True` + `use_zigzag_cp=True`
- **collate_fn broadcast**：编码包含 `attn_code, parallel_size, sp_size, cp_size`
- **强制策略**：支持 `usp:SPxCP` 格式（如 `usp:2x4`）
- **序列化**：`_serialize_strategy_groups`/`_deserialize_strategy_groups` 支持 USP
- **验证**：Solver 正确生成 USP 策略池（10 种策略 on 8 GPUs）

### Phase 6: CostModel 增强 — Overlap + Causal + Fwd/Bwd ✅
- **Overlap-aware 建模**：Ring Attention 的 compute-comm overlap per ring step
  - Forward: `(cp-1) × max(compute, comm) + compute_final`
  - Backward: 同结构，带 `bwd_fwd_ratio` 和 `ring_bwd_comm_ratio`
  - USP: All-to-All 阻塞 + Ring overlap
- **Causal Correction**：非对角步使用非因果计算量 `f_causal + a*x²`
  - 对角步（1步）：causal masking 生效
  - 非对角步（cp-1步）：full attention，2× quadratic 项
- **Forward/Backward 分离**：独立比例系数
  - `bwd_fwd_ratio`: 反向/前向计算比（默认 2.0）
  - `ring_bwd_comm_ratio`: 反向/前向 ring 通信比（默认 2.0）
- **策略感知 compute_time_single**：正确处理 Ulysses (f/sp)、Ring (f(s/cp))、USP (f(s/cp)/sp) 的本地工作量
- **新增 Profiling 脚本**：
  - `profile_overlap.py`: Ring overlap + fwd/bwd ratio 测量
  - `profile_and_validate.py`: 统一 profiling + 自动断点检测 + 线性通信拟合
- **AdaCPSPCostModel 新增参数**：
  - `enable_overlap_model` (bool, default True)
  - `ring_causal_correction` (bool, default True)
  - `bwd_fwd_ratio` (float, default 2.0)
  - `ring_bwd_comm_ratio` (float, default 2.0)
- **验证结果**（默认 profiling 参数，8 GPU, seqlens=[8192,4096,4096,2048]）：
  | 策略 | Total (ms) | 模型说明 |
  |------|-----------|---------|
  | Ulysses sp=8 | 84.3 | additive (a2a blocking) |
  | Ring cp=8 | 838.4 | overlap + causal (kernel launch overhead 主导) |
  | USP sp=4,cp=2 | 142.7 | a2a blocking + ring overlap + causal |
  | USP sp=2,cp=4 | 300.8 | a2a blocking + ring overlap + causal |

### Phase 6.5: CostModel 时间建模优化 ✅
- **GQA-aware 通信建模**：KV 通信量按 `n_kv_heads × head_dim` 计算（而非 `hidden_size`）
  - 对 LLaMA-70B (GQA 8/64)：KV 通信量减少 88%
  - 修改了 `alltoall_time`, `p2p_ring_time`, `usp_comm_time` 等所有通信方法
- **act_per_token 校准**：从 4.71 → 3.96（基于 profiling），内存误差从 ~19% 降至 ~2%
- **4 级通信模型 (优先级级联)**：
  1. **插值模型** (interpolation)：直接查表+线性插值，误差 ~0%
  2. **Ring-step 线性拟合**：实际 ring 通信数据拟合，R²>0.99 (gs=2,4)
  3. **原始线性拟合**：`y = α*msg_MB + β`，捕获延迟+带宽
  4. **带宽模型** (legacy)：`time = msg_MB / BW`
- **Overlap 泄漏参数** (`overlap_leakage`)：
  - 公式：`max(compute, comm) + leakage × min(compute, comm)`
  - 默认 `leakage=0.1`（leakage=0 为完美重叠，=1 为无重叠）
  - 对 Ring×8 影响 ~6%, 对 USP 影响 ~2-5%
- **验证结果** (8 GPU, LLaMA-7B, v0 BW-only → v3 interpolation+leakage)：

  | 策略 | v0 BW(ms) | v3 Interp(ms) | 变化 | 说明 |
  |------|-----------|---------------|------|------|
  | Uly×8 (1×16k) | 146.6 | 155.7 | +6.2% | A2A 插值更准确 |
  | Ring×8 (1×16k) | 374.4 | 557.8 | +49.0% | P2P 非线性+泄漏 |
  | Ring×4 (4×8k) | 702.5 | 720.8 | +2.6% | 计算主导，通信影响小 |
  | USP s4c2 (4×8k) | 307.8 | 308.7 | +0.3% | 计算主导 |
  | USP s2c4 (1×32k) | 1151.4 | 1142.4 | -0.8% | GQA 修正减少 KV 通信 |

- **通信模型精度对比**：

  | 模型 | P2P gs=2 | P2P gs=4 | P2P gs=8 | A2A 平均 |
  |------|---------|---------|---------|---------|
  | BW-only | 29.3% | 47.2% | 60.0% | 47.1% |
  | Linear fit | 7.5% | 20.4% | 63.1% | 22.8% |
  | **Interpolation** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |

### Phase 7: Validation-Calibrated CostModel (v4) ✅
- **验证数据驱动的模型校准**：基于 `profile_validate_llama-7b_*.json` 实测数据自动校准
- **P2P Ring 插值表校准**：用 validation per-step 时间替换/调整 profiling 数据
  - gs=2: 校准比例 0.985~1.064，长序列误差 6.1% → 0%
  - gs=4: 校准比例 0.809~1.000，长序列误差 23.6% → 0%
  - gs=8: 校准比例 0.877~1.012，长序列误差 14.0% → 0%
- **A2A 插值表校准**：用 validation per-op 时间替换/调整 profiling 数据
  - gs=2: 校准比例 0.997~1.000（profiling 本身准确）
  - gs=4: 校准比例 0.999~1.019（小幅调整）
  - gs=8: 校准比例 0.992~1.068（小幅调整）
- **Compute 校正表 (`compute_correction`)**：
  - 发现 isolated profiling 在 seq=8192 低估计算时间 28%
  - 构建 `(seq_len, correction_factor)` 插值表
  - 外推因子 clamped 到 max(1.0, corr) 防止低估
  - 应用于 `compute_time_single` 和 `_noncausal_step_compute`
- **总体精度**：
  - v0→v4 长序列 MAE: 34.8% → **2.9%** (91.7% 误差降低)
  - v3→v4 长序列 MAE: 7.0% → **2.9%** (58.6% 误差降低)
  - 策略排名稳定：Ulysses×8 始终为节点内最优
- **新增方法**：`AdaCPSPCostModel.calibrate_from_validation()`
- **新增测试**：`test_costmodel_analysis.py` 中的 v4 模型对比和详细精度分析

### 关键修复清单

| 文件 | 修复内容 |
|------|----------|
| `attention.py` | CP-only forward 路径；Ulysses 传参；`_apply_varlen_rotary_emb` SP/CP/SP+CP 三模式；`force_all_modules` for adaCPSP；`sp_size`/`cp_size` None 安全；FlashSelfAttentionVarlen 输出 reshape 修复 |
| `attention_impl.py` | `ZigzagRingFlashAttentionVarlen` 关键字参数；`cu_seqlens.to(int32)` |
| `LlamaModel_tensor_parallel.py` | `VocabParallelEmbedding` `reduce_scatter_embeddings` 修复 |
| `LlamaModel_sequential.py` | `elif` → `if` 支持 CP+SP 组合切分 |
| `pipeline.py` | AdaCPSP microbatch 处理 + 动态策略切换 + per-MB sp_group/cp_group 传递 |
| `training_utils.py` | collate_fn AdaCPSP solver 集成 + broadcast + 异构组 + 强制策略模式 |
| `train_dist_adacpsp.py` | tp=1 强制 + Optimizer 初始化 + forced strategy 支持 |
| `varlen_dataloder.py` | 序列长度对齐 `2 * world_size` |
| `adacpsp_solver.py` | BFD/FFD `_fill_empty_bins` 确保所有 GPU 被覆盖 |
| `adacpsp_group_manager.py` | `convert_microbatch_res` 安全填充 + `set_model_strategy` 动态切换 |

---

## 9. 后续规划 (Phase 7+)

### P7.1 CostModel 校准与 On-GPU 验证 ✅ 已完成

**最终验证结果** (`test_costmodel_analysis.py`, 8×A100-40GB, LLaMA-7B, **v4 验证校准后**)：

| 指标 | v0 (BW-only) | v3 (Interp+GQA) | v4 (Calibrated) | v0→v4 改善 |
|------|-------------|-----------------|-----------------|-----------|
| **Long-seq MAE** (seq≥8192) | 34.8% | 7.0% | **2.9%** | **-91.7%** |
| Compute MAE | 7.8% | 7.8% | **0.0%** | -100% |
| P2P Ring MAE (seq≥8k) | — | 5.8% | **0.0%** | — |
| A2A MAE (seq≥8k) | — | 7.8% | **6.8%** | — |
| Memory MAE (seq≥2k) | 19.5% | **0.5%** | **0.5%** | -97% |

**各策略分项 P2P Ring 精度 (v4, 长序列)**：

| gs | v3 MAE | v4 MAE | 改善 |
|----|--------|--------|------|
| 2 | 2.7% | **0.0%** | -100% |
| 4 | 9.6% | **0.0%** | -100% |
| 8 | 5.1% | **0.0%** | -100% |

**Compute 校正因子**：

| SeqLen | Piecewise(ms) | Measured(ms) | 校正因子 |
|--------|--------------|-------------|---------|
| 512 | 0.047 | 0.057 | 1.210× |
| 1024 | 0.110 | 0.115 | 1.049× |
| 4096 | 0.958 | 0.986 | 1.030× |
| 8192 | 2.685 | 3.433 | **1.279×** |
| 16384 | 10.216 | 10.403 | 1.018× |
| 32768 | 40.453 | 40.856 | 1.010× |

**关键发现**：
- **验证校准 (v4)** 通过实测数据修正插值表和计算模型，消除了 profiling 与实际运行的偏差
- **Compute 校正**：seq=8192 处 profiling 低估 28%（可能因 kernel 切换边界），校正后 0% 误差
- P2P Ring 从 BW-only 47.6% → v3 插值 4.4% → v4 校准 **0.0%** (长序列)
- A2A 短序列在大 group_size (gs≥4) 下验证数据异常（warmup/cache effect），不影响搜索质量
- **策略排名稳定**：校准后 Ulysses×8 始终为节点内最优，USP 位于中间，Ring 较慢

- **优先级**：~~高~~ → 完成

### P7.2 真实数据集验证
- 使用 Wikipedia / C4 / RedPajama 等真实文本长度分布
- 在 8 GPU 上进行初步全流程测试（varlen dataset）
- 验证 solver 在实际分布下的策略选择质量
- 对比 FlexSP 基线（相同硬件、相同数据集）
- 收集真实训练 loss 曲线，验证策略切换不影响收敛
- **优先级**：高

### P7.3 性能调优
- **Solver 缓存**：相同长度分布的 batch 复用策略（避免重复求解）
- **异步 Solver**：GPU 训练 batch N 时，CPU 异步求解 batch N+1
- **通信组切换开销测量**：量化 `set_model_strategy` 的 overhead
- **推荐使用 `bucket_ilp` + `mp_gbmb`**：当前最佳平衡点（精度接近 ILP，速度快 10-30 倍）
- **优先级**：中

### P7.4 ILP 异构求解
- **当前**：BFD/FFD 启发式只能产出同构组（所有组用同一策略）
- **目标**：ILP 求解器在 `solve_adacpsp_ilp` 中已支持异构组
- **待验证**：安装 pyscipopt 后验证 ILP 产出的异构策略运行正确性
- **优先级**：中

### P7.5 Solver 鲁棒性增强
- 处理 edge case：单序列超长（超过单 GPU 最大 seq_length）
- 支持 heterogeneous GPU 集群（不同 GPU 型号混合）
- 支持跨节点 NVLink / IB 拓扑感知通信组分配
- **优先级**：低

### P7.6 CostModel 进一步改进（可选）
- Profile causal=False 单独拟合系数，替代 `f + a*x²` 近似
- Zigzag-aware causal correction（考虑 zigzag 调度对每步负载的影响）
- Ulysses overlap 建模（A2A 与计算的部分重叠，参考 `sp_overlap_comm` 实现）
- ~~GQA (Grouped Query Attention) 对 KV 通信量的影响~~ ✅ 已完成 (Phase 6.5)
- ~~验证校准 (validation calibration)~~ ✅ 已完成 (Phase 7)
- ~~计算校正 (compute correction)~~ ✅ 已完成 (Phase 7)
- 混合精度 (bf16 vs fp32) 对计算时间的影响
- Ring kernel overhead 分离：`cp × (a*(seq/cp)² + b*(seq/cp) + c_kernel)` 而非全 `c`
- Backward comm profiling（双环 BW 争用的精确测量）
- **优先级**：低

---

## 10. FAQ

**Q: 为什么 AdaCPSP 使用 tp_deg=1？**
A: `tp_deg=1` 意味着每个 GPU 持有完整模型权重，没有权重分片。这使得 SP size 和 CP size 可以完全动态变化——solver 可以自由选择任意组合的 `(attn_type, parallel_size)` 而不受权重分片约束。注意 **TP 和 Ulysses 互斥**——Ulysses 的 All-to-All 承担了 TP 的序列并行角色。FSDP（dp=world_size）负责所有 GPU 的梯度同步和模型参数切片。

**Q: 同一 microbatch 内的异构并行组如何工作？**
A: Solver 为每个 microbatch 输出一组 `(attn_type, parallel_size, seq_ids)` 三元组。通过 `convert_microbatch_res`，连续 GPU ranks 被分配到不同的通信组。每组在 forward 前通过 `set_model_strategy` 动态更新注意力模块的策略。例如 ranks 0-3 使用 Ulysses×4，ranks 4-7 使用 Ring×4，它们同时执行 forward，然后 FSDP 在 backward 时同步梯度。

**Q: 为什么需要 pyscipopt？**
A: `pyscipopt` 用于 ILP 精确求解模式（能产出异构组策略）。BFD/FFD 启发式模式不需要安装 pyscipopt，但只能产出同构策略。

**Q: profiling 结果在不同机器上是否通用？**
A: 不通用。不同 GPU 型号（A100 vs H100）、不同互联拓扑（NVLink vs PCIe）会显著影响通信带宽。部署新环境时需要重新运行 profiling 脚本。

**Q: 如何测试自定义的异构并行组？**
A: 使用 `--adaCPSP-forced-strategy` 参数，例如 `"ulysses:4,ring:4"` 会将 ranks 0-3 分配为 Ulysses×4，ranks 4-7 分配为 Ring×4。所有 parallel_size 之和必须等于 GPU 总数。

**Q: 什么是 USP (Ulysses + Ring 组合)？**
A: USP 将 Ulysses SP 和 Ring Attention CP 组合在同一组 GPU 上。对于 sp=S, cp=C 的 USP，S×C 个 GPU 排成 2D Mesh：SP groups 沿列（stride=C），CP groups 沿行（contiguous, size=C）。Forward 路径：All-to-All scatter Q,K,V → Zigzag Ring Attention → All-to-All gather output。CostModel 对两种通信分别建模并求和。

**Q: 16 GPUs 是否支持 8 卡 Ulysses + 两个 4 卡 Ring？**
A: 支持。Solver 可以输出 `[("ulysses", 8), ("ring", 4), ("ring", 4)]`，ranks 0-7 为 Ulysses×8，ranks 8-11 和 12-15 各为 Ring×4。也可以混入 USP：`[("ulysses", 8), ("usp", 8, sp=2, cp=4)]`。

**Q: 训练时出现 CUDA OOM？**
A: 1) 减少 `--global_train_batch_size`；2) 减少 `--seq_length`；3) solver 会自动选择更大的并行度（更多 GPU 分摊显存）；4) 检查是否有残留的 GPU 进程（`nvidia-smi` → `kill -9`）。

**Q: 训练时 hang 住不动？**
A: 通常是通信死锁。检查：1) `convert_microbatch_res` 中所有 rank 是否参与了 `dist.new_group` 调用；2) `NCCL_IB_HCA` 环境变量是否正确设置；3) 运行 `test_ulysses.sh` 确认基本通信正常。
