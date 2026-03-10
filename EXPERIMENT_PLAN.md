# AdaCPSP vs FlexSP 实验计划

## 一、环境配置

### 1.1 硬件环境
| 项目 | 规格 |
|------|------|
| GPU | 8× NVIDIA A100-SXM4-40GB (NVLink) |
| GPU 显存 | 40 GB × 8 |
| 机内互联 | NVLink 600 GB/s |
| 驱动版本 | 470.129.06 |
| CUDA Toolkit | 12.1 (nvcc) |

### 1.2 软件环境（conda 环境 `lqs`）

已验证可用的 `lqs` 环境：

| 包 | 版本 |
|----|------|
| Python | 3.x (conda env `lqs`) |
| PyTorch | 2.4.1+cu118 |
| CUDA (torch) | 11.8 |
| Flash Attention | 2.7.2.post1 |
| Transformers | 4.47.0 |
| NumPy | 2.0.1 |
| SciPy | 1.14.1 |
| NCCL | 2.20.5 |

### 1.3 环境安装步骤（在新镜像中从零搭建）

```bash
# ═══════════════════════════════════════════════════════════
# Step 1: 创建 conda 环境并选择 Python 版本
# ═══════════════════════════════════════════════════════════
conda create -n adacpsp python=3.10 -y
conda activate adacpsp

# ═══════════════════════════════════════════════════════════
# Step 2: 安装 PyTorch (CUDA 11.8 版本，与 A100 兼容)
# ═══════════════════════════════════════════════════════════
# 方式 A: pip (推荐)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# 方式 B: conda
# conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

# ═══════════════════════════════════════════════════════════
# Step 3: 安装 Flash Attention (必须与 PyTorch/CUDA 版本匹配)
# ═══════════════════════════════════════════════════════════
pip install flash-attn==2.7.2.post1 --no-build-isolation

# ═══════════════════════════════════════════════════════════
# Step 4: 安装 ring-flash-attention (P2P Ring Attention 实现)
# ═══════════════════════════════════════════════════════════
cd /path/to/lqs/ring-flash-attention
pip install -e .

# ═══════════════════════════════════════════════════════════
# Step 5: 安装其他依赖
# ═══════════════════════════════════════════════════════════
pip install transformers==4.49.0 numpy"<2.0.0" scipy h5py attrs yacs six sentencepiece pybind11

# ═══════════════════════════════════════════════════════════
# Step 6: 安装 Galvatron (开发模式)
# ═══════════════════════════════════════════════════════════
cd /path/to/galvatron_lxy/Hetu-Galvatron
pip install -e .

# ═══════════════════════════════════════════════════════════
# Step 7: 验证安装
# ═══════════════════════════════════════════════════════════
python -c "
import torch, flash_attn, transformers
print(f'torch={torch.__version__} cuda={torch.version.cuda}')
print(f'flash_attn={flash_attn.__version__}')
print(f'transformers={transformers.__version__}')
print(f'nccl={torch.cuda.nccl.version()}')
print(f'gpus={torch.cuda.device_count()}')
from galvatron.models.varlen_llama_hf.adacpsp_solver import AdaCPSPCostModel
print('AdaCPSPCostModel importable ✓')
"
```

### 1.4 已有环境快速验证

如果使用已有的 `lqs` 环境：

```bash
conda activate lqs
# 只需确保 ring-flash-attention 已安装
cd /home/pkuhetu/lqs/ring-flash-attention && pip install -e .
# 确保 galvatron_lxy 版本已安装
cd /home/pkuhetu/lqs/galvatron_lxy/Hetu-Galvatron && pip install -e .
```

---

## 二、实验设计

### 2.1 对比对象

| 方法 | 说明 | 实现方式 |
|------|------|----------|
| **FlexSP** | 只使用 Ulysses SP | `--adaCPSP-attn-types ulysses` |
| **AdaCPSP** | Ulysses + Ring Attention + USP | `--adaCPSP-attn-types ulysses ring usp` |

> **关键设计**：FlexSP 用只选择 ulysses 的 AdaCPSP 代替，确保使用完全相同的代码路径（solver、dataloader、训练循环），唯一区别是 solver 可选的策略空间。

### 2.2 模型配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | LLaMA-7B | `--model_size llama-7b` |
| Hidden Size | 4096 | |
| Num Attention Heads | 32 | |
| Num KV Heads | 32 | LLaMA-7B 无 GQA |
| Head Dim | 128 | |
| Vocab Size | 32000 | |
| **Num Layers** | **32 (full) / 2 (fast)** | full 用于真实性能测试，2层用于快速迭代验证 |
| 精度 | bf16 | `--mixed_precision bf16` |
| 优化器 | ZeRO-2 | `--default_dp_type zero2` |
| TP | 1 | AdaCPSP 核心设计 |
| PP | 1 | 单机不需要流水线 |

### 2.3 实验配置矩阵

#### 8 卡实验（单机 NVLink）

| 实验ID | max_seq | GBS | Layers | 数据集 | 目的 |
|--------|---------|-----|--------|--------|------|
| **E1_short** | 8192 | 64 | 32 | wikipedia | 短序列基线，FlexSP 应最优 |
| **E2_mixed** | 32768 | 32 | 32 | wikipedia | 混合长度，AdaCPSP 可能受益于 Ring |
| **E3_long** | 32768 | 16 | 32 | wikipedia | 长序列为主，Ring/USP 潜在优势 |
| **E4_extreme** | 65536 | 8 | 32 | wikipedia | 极长序列，内存受限场景 |
| **E5_fast_val** | 32768 | 32 | 2 | wikipedia | 2 层快速验证，用于调试确认流程正确 |

> **注意**：
> - Wikipedia 数据集中位数约 347 tokens，大多数序列很短（62.6% < 512）
> - 长序列比例：≥8192 仅 1.6%，≥16384 仅 0.5%，≥32768 仅 0.1%
> - 在 max_seq=32768 时，GBS=32 意味着约 1-2 条 ≥16k 的长序列 + 大量短序列

#### 变量说明

| 变量 | 含义 | 影响 |
|------|------|------|
| `max_seq` (seq_length) | 序列最大长度上限 | 决定了长序列的上界，影响 Ring/USP 的价值 |
| `GBS` (global_train_batch_size) | 全局 batch 大小（序列条数） | 决定每个 iter 处理的总 token 数和 microbatch 划分 |
| `Layers` | 模型层数 | 影响显存和计算量；2 层用于快速调试 |
| `--adaCPSP-attn-types` | 允许的注意力类型 | FlexSP=ulysses; AdaCPSP=ulysses ring usp |
| `--memory-limit-gb` | 每卡显存上限 | 建议 36 GB (A100-40G 的 90%) |
| `--train-iters` | 训练迭代次数 | 性能测试用 20 iter，前 5 做 warmup |

### 2.4 显存容量估算

对于 LLaMA-7B (32 层) + ZeRO-2 + bf16：
- 模型状态 (ZeRO-2)：~26 GB
- 可用于 activation：~10 GB (36 GB limit - 26 GB model)
- act_per_token ≈ 3.96 MB/token
- Token 容量 ≈ 10240 / 3.96 ≈ 2585 tokens/GPU

> 对于 8 卡 Ulysses×8：cluster_capacity ≈ 2585 × 8 = 20680 tokens
> 对于 2 卡 Ulysses×2：每组 capacity ≈ 2585 × 2 = 5170 tokens

---

## 三、实验执行流程

### 3.1 Phase 0: 环境验证（5 分钟）

```bash
# 运行环境检查脚本
bash llama_scripts/exp_check_env.sh
```

### 3.2 Phase 1: 快速验证（~10 分钟）

用 2 层模型确认流程正确：

```bash
# FlexSP (Ulysses only)
bash llama_scripts/exp_E5_fast_val_flexsp.sh

# AdaCPSP (Ulysses + Ring + USP)
bash llama_scripts/exp_E5_fast_val_adacpsp.sh
```

**检查项**：
- [ ] 两个脚本都能正常运行完毕
- [ ] Loss 在下降（不需要收敛，但不应该 NaN）
- [ ] Solver 输出策略分配信息
- [ ] 无 NCCL 错误

### 3.3 Phase 2: 正式实验（~2-4 小时）

按实验 ID 依次执行：

```bash
# 运行全部实验（自动按顺序执行）
bash llama_scripts/exp_run_all.sh

# 或单独运行某个实验
bash llama_scripts/exp_E1_short_flexsp.sh
bash llama_scripts/exp_E1_short_adacpsp.sh
```

### 3.4 Phase 3: 结果收集与分析

```bash
# 运行分析脚本，自动解析所有日志
python analyze_experiment_logs.py --log_dir logs/
```

---

## 四、观察和对比指标

### 4.1 核心指标

| 指标 | 来源 | 说明 |
|------|------|------|
| **Average iteration time (ms)** | `RuntimeProfiler._process_time_results` | 去掉 warmup 后的平均每 iter 时间 |
| **Elapsed time per iteration (ms)** | `RuntimeProfiler._log_iteration_stats` | 每个 iter 实时打印 |
| **MFU (%)** | 后处理计算 | `tokens × 6 × param_B / (iter_time × peak_TFLOPS)` |
| **Throughput (tokens/s)** | 后处理计算 | `total_tokens_per_iter / iter_time` |
| **Loss** | 训练循环 | 确认训练正确性 |

### 4.2 日志解析关键字

```bash
# 平均迭代时间（去 warmup）
grep "Average iteration time" logs/exp_*.log

# 每 iteration 时间
grep "Elapsed time per iteration" logs/exp_*.log

# AdaCPSP Solver 策略选择
grep "\[AdaCPSP\] MB" logs/exp_*.log

# Loss 趋势
grep "Loss = " logs/exp_*.log

# 显存使用
grep "After Backward" logs/exp_*.log   # RuntimeProfiler memory snapshot
```

### 4.3 对比分析维度

#### (A) End-to-End 时间对比

最核心的指标。对每个实验 ID 对比：

```
实验ID  |  FlexSP avg_iter(ms)  |  AdaCPSP avg_iter(ms)  |  Speedup
E1      |  xxx                  |  xxx                   |  x.xx×
E2      |  xxx                  |  xxx                   |  x.xx×
...
```

#### (B) MFU 对比

```
MFU = total_tokens_in_batch × FLOPs_per_token / (iter_time_s × peak_TFLOPS × N_GPUs)

其中:
- FLOPs_per_token = 6 × param_B × 1e9 (forward + backward approx)
- peak_TFLOPS = 312 TF/s (A100 bf16)
- N_GPUs = 8
```

#### (C) 策略分布对比

AdaCPSP 是否选择了不同于纯 Ulysses 的策略？

```
统计每个 iter 中 solver 选择的策略分布:
- ulysses×1: xx%
- ulysses×2: xx%
- ulysses×4: xx%
- ulysses×8: xx%
- ring×4: xx%
- ring×8: xx%
- usp(2×4): xx%
```

#### (D) Microbatch 划分对比

FlexSP 和 AdaCPSP 是否产生了不同数量的 microbatch？更多 microbatch 意味着更多 pipeline bubble。

#### (E) 显存利用率对比

观察是否有 OOM，以及显存峰值是否接近上限。

#### (F) Loss 一致性

两种方法的 Loss 轨迹应该不同（因为 batch 划分不同），但量级应相近。

### 4.4 预期结论

基于 solver 级别分析的预测：

1. **E1_short (max=8k, GBS=64)**：FlexSP ≈ AdaCPSP（短序列时 Ulysses 最优）
2. **E2_mixed (max=32k, GBS=32)**：差距很小（NVLink 下 Ulysses 带宽优势大）
3. **E3_long (max=32k, GBS=16)**：可能出现微小差异
4. **E4_extreme (max=64k, GBS=8)**：如果有极长序列，Ring/USP 可能有优势

> **注意**：在 8 卡 NVLink 环境下，solver 分析已表明 Ulysses 在所有长度都优于 Ring。
> AdaCPSP 的真正优势预期出现在：
> - 跨机场景（AlltoAll 带宽降低）
> - GQA 模型（Ring 的 KV 通信更小）
> - 更大集群（16+卡，多节点）

---

## 五、其他注意事项

### 5.1 Warmup 处理
- 前 5 个 iteration 作为 warmup，不计入性能统计
- `RuntimeProfiler` 默认从 `start_iter` 开始统计
- 建议 `--train-iters 20`，取后 15 个 iter 的平均

### 5.2 可复现性
- 设置固定随机种子（代码中 `set_seed()` 使用 seed=123）
- 每个实验运行 3 次取中位数（如时间允许）

### 5.3 GPU 冷却
- 每个实验之间等待 10 秒（脚本中已加入 `sleep 10`）
- 如果 GPU 温度过高，可延长等待时间

### 5.4 数据集路径
- Wikipedia 数据集：`/home/pkuhetu/lqs/flexsp/Hetu-Galvatron/galvatron/datasets/wikipedia.txt`
- 100万条序列长度，无需实际文本内容

### 5.5 可能的问题和解决
| 问题 | 解决 |
|------|------|
| OOM | 降低 GBS 或 max_seq，或增加 --memory-limit-gb |
| NCCL timeout | 增加 `NCCL_TIMEOUT` 环境变量 |
| ring_flash_attn 未安装 | `cd ring-flash-attention && pip install -e .` |
| Solver 找不到可行解 | 降低 GBS 或增加 --memory-limit-gb |
| Loss = NaN | 降低学习率到 1e-5 |

### 5.6 扩展到 32/64 卡

当前实验在 8 卡上进行。扩展建议：

- **32 卡 (4 节点)**：跨机 AlltoAll 带宽会显著降低，AdaCPSP 的 Ring Attention 路径可能真正受益
- **64 卡**：更大的策略空间，USP 的组合优势更明显
- 需要修改 `NUM_NODES` 和 `MASTER_ADDR`，并配置跨节点网络

---

## 六、文件清单

| 文件 | 用途 |
|------|------|
| `EXPERIMENT_PLAN.md` | 本文档 |
| `llama_scripts/exp_check_env.sh` | 环境检查 |
| `llama_scripts/exp_E*_flexsp.sh` | 各实验 FlexSP 脚本 |
| `llama_scripts/exp_E*_adacpsp.sh` | 各实验 AdaCPSP 脚本 |
| `llama_scripts/exp_run_all.sh` | 全部实验主控脚本 |
| `analyze_experiment_logs.py` | 日志分析脚本 |

