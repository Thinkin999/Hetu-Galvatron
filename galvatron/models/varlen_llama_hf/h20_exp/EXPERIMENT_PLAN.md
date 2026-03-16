# AdaCPSP vs FlexSP 实验计划（H20 64卡）

## 1. 实验目标

对比 **FlexSP**（仅 Ulysses）与 **AdaCPSP**（Ulysses + Ring Attention + USP）在 GQA 模型上的性能差异。

**核心假设**：GQA 模型的 KV heads 远少于 Q heads，Ring Attention 的 P2P 通信量大幅减少（仅传输 KV），
而 Ulysses 的 All-to-All 需要传输全量 Q+K+V+O。在长序列 + 跨机场景下，AdaCPSP 应优于 FlexSP。

## 2. 集群环境

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA H20 × 64 (96GB HBM3) |
| 节点 | 8 节点 × 8 GPU/节点 |
| 节点内互联 | NVLink (需 profile 确认带宽) |
| 节点间互联 | InfiniBand (需 profile 确认带宽) |
| 精度 | bf16 |
| 分布式策略 | ZeRO-2, tp=1, dp=world_size |

## 3. 模型配置

| 模型 | hidden | ffn | layers | heads | kv_heads | GQA ratio | head_dim | ~param |
|------|--------|-----|--------|-------|----------|-----------|----------|--------|
| Qwen2.5-7B | 3584 | 18944 | 28 | 28 | 4 | **1:7** | 128 | ~7B |
| Qwen2.5-14B | 5120 | 13824 | 48 | 40 | 8 | **1:5** | 128 | ~14B |
| Qwen2.5-32B | 5120 | 27648 | 64 | 40 | 8 | **1:5** | 128 | ~32B |

**GQA 的意义**：
- Qwen2.5-7B: Ring P2P 只传 4/28 = **14%** 的 KV vs All-to-All 传完整 Q+K+V+O
- Qwen2.5-14B/32B: Ring P2P 只传 8/40 = **20%** 的 KV

## 4. 实验变量

### 4.1 模型 (3种)
- `qwen2.5-7b`, `qwen2.5-14b`, `qwen2.5-32b`

### 4.2 最大序列长度 (4种)
- `128K`, `256K`, `384K`, `512K`
- 对应 `--seq_length`: 131072, 262144, 393216, 524288

### 4.3 并行策略 (3种)
- **FlexSP** (仅 Ulysses): `--adaCPSP-attn-types ulysses`
- **AdaCPSP-UR** (Ulysses + Ring): `--adaCPSP-attn-types ulysses ring`
- **AdaCPSP-Full** (Ulysses + Ring + USP): `--adaCPSP-attn-types ulysses ring usp`

### 4.4 数据集 (2种)
- `common_crawl`: 长序列比例较高 (20.4% ≥ 4K, 2.8% ≥ 16K, 0.4% ≥ 64K)
- `github`: 代码数据 (16.8% ≥ 4K, 3.9% ≥ 16K, 0.6% ≥ 64K)

### 4.5 固定参数
| 参数 | 值 | 说明 |
|------|-----|------|
| GBS | **512** | 匹配 FlexSP 原始实验 |
| iterations | **30** | benchmark 模式 |
| world_size | **64** | 8×8 |
| ZeRO stage | **2** | FSDP 分片 |
| precision | **bf16** | 混合精度 |
| selective_checkpoint | **1** | 减少激活显存 |
| packing | **true** | 变长序列打包 |

## 5. 实验矩阵

总计: 3 模型 × 4 seq_len × 3 策略 × 2 数据集 = **72 组实验**

对于显存不足的情况，脚本会自动跳过并记录 OOM。

## 6. 执行流程

### Phase 0: 环境检查与 Profiling
```bash
# Step 0.1: 检查环境
bash 00_check_env.sh

# Step 0.2: Profile 硬件 (在 64 卡上执行)
bash 01_profile_hardware.sh
```
这一步生成 attention 计算时间、All-to-All 带宽、P2P Ring 带宽的 profile 数据，
保存到 `configs/` 目录，供 cost model 使用。

### Phase 1: 运行实验
```bash
# Step 1.1: 生成所有实验脚本
python generate_experiments.py

# Step 1.2: 运行所有实验 (容错模式)
bash 02_run_all.sh 2>&1 | tee run_all.log
```

### Phase 2: 分析结果
```bash
# 收集所有 log 后执行
python analyze_results.py --log-dir logs/
```

## 7. 日志与指标

### 7.1 每个实验的日志文件命名
```
logs/{model}_{dataset}_{seqlen}k_{strategy}.log
```
例: `logs/qwen2.5-7b_common_crawl_256k_flexsp.log`

### 7.2 关键指标 (从日志中提取)
| 指标 | 说明 | 日志关键词 |
|------|------|-----------|
| Avg Iteration Time (ms) | 去掉前 5 个 warmup iter 后的平均 | `Iteration time:` |
| Throughput (tokens/sec) | 总 token 数 / 总时间 | `Throughput:` |
| MFU (%) | Model FLOPs Utilization | 从 throughput 计算 |
| Peak Memory (GB) | 最大显存占用 | `Max memory:` |
| Strategy Distribution | 各策略被选择的比例 | `[AdaCPSP]` |
| OOM / Error | 是否发生 OOM 或 hang | `CUDA out of memory` / timeout |

### 7.3 对比维度
1. **相同模型+数据集+seq_len**: FlexSP vs AdaCPSP-UR vs AdaCPSP-Full
   → 展示 Ring/USP 的加速效果
2. **相同策略，不同seq_len**: 展示策略在不同序列长度下的变化趋势
   → Ring 在长序列下应更有优势
3. **相同策略，不同模型**: 展示 GQA 比例对 Ring 优势的影响
   → 7B (1:7 GQA) vs 14B/32B (1:5 GQA)

## 8. 容错机制

- 每个实验有独立 timeout (默认 10 分钟)
- OOM 自动捕获，记录到日志，跳过继续
- NCCL timeout / hang: 通过 `timeout` 命令强制终止
- 实验脚本失败不影响后续实验的执行

## 9. GBS 选择理由

**固定 GBS=512**，理由：
1. **公平性**：同一 (model, seq_len, dataset) 下，所有策略处理相同的序列集合，
   只是分组/并行方式不同，确保 speedup 有意义
2. **与 FlexSP 一致**：FlexSP 原始实验也用 GBS=512 on 64 GPUs
3. **足够大**：512 个序列提供足够的搜索空间，让 solver 做有意义的决策
4. **H20 96GB 充裕**：比 A100-40GB 多 2.4× 显存，GBS=512 应可行

如果 32B+512K 确实 OOM，脚本会自动尝试 GBS=256 作为 fallback。


