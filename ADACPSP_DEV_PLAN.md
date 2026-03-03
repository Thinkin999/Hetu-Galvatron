# AdaCPSP 开发计划

## 1. 项目目标

**AdaCPSP (Adaptive Context Parallel & Sequence Parallel)** 是对 FlexSP 的扩展，核心区别在于：

| 维度 | FlexSP | AdaCPSP |
|------|--------|---------|
| 策略空间 | 只选择 Ulysses SP 的 `sp_size` | 同时选择 `attn_type` (Ulysses / Ring / Combined) 和 `(sp_size, cp_size)` |
| 通信模式 | 仅 All-to-All | All-to-All + P2P Ring |
| 组合模式 | 单一策略 | 支持 Ulysses × Ring 混合（如 sp=4, cp=2 → 8 GPUs）|
| 数据切分 | 按 token 总数切分 | 需按 zigzag ring attention 方式切分 |

## 2. 开发分阶段规划

### Phase 0: 基础设施 (Profiling & Cost Model)
- [ ] **P0.1** All-to-All 通信 profiling 脚本 (`profile_alltoall.py`)
  - 测量不同 SP size 下的 All-to-All 带宽
  - 输出 `alltoall_bandwidth_dict_gbs = {2: xxx, 4: xxx, 8: xxx, ...}`
- [ ] **P0.2** P2P Ring 通信 profiling 脚本 (`profile_p2p_ring.py`)
  - 测量不同 CP size 下的 P2P Ring 带宽和延迟
  - 区分 intra-node 和 inter-node 场景
  - 输出 `p2p_bandwidth_gbs` 和 `p2p_latency_ms`
- [ ] **P0.3** Attention 计算 profiling + 分段拟合脚本
  - 已有基础：`llama_hf/profile_flash_attention.py` + `predict_attention_time.py`
  - 需要扩展：在 AdaCPSP 仓库中独立一份，支持分段函数 `time = a*x² + b*x + c`
  - 注意：Flash Attention 对不同 seqlen 使用不同 kernel，必须拟合分段函数
  - 输出：`{segment_name: {a, b, c, seq_range}, ...}` 的 JSON 配置

### Phase 1: 单一策略 + varlen 场景跑通
> 目标：让 Ulysses SP 和 Ring Attention 各自在 varlen (packing) 场景下独立跑通

- [ ] **P1.1** 基于 `llama_hf` 创建 AdaCPSP 的模型文件
  - 参考 `llama_hf/LlamaModel_tensor_parallel.py` → 创建支持 varlen 的 `LlamaModel_tensor_parallel.py`
  - 参考 `llama_hf/LlamaModel_sequential.py` → 创建支持 varlen 的 `LlamaModel_sequential.py`（已有部分基础在 `varlen_llama_hf/LlamaModel_sequential.py`）
  - 参考 `llama_hf/LlamaModel_hybrid_parallel.py` → 创建对应文件
  - 创建缺失的 `arguments.py`
  - 关键改动：Attention 层的 forward 要支持 `cu_seqlens`（flash_attn_varlen_func 接口）
- [ ] **P1.2** 创建 varlen dataloader + fake data
  - 创建 `DataLoaderForVarlenLlama`：生成变长序列假数据
  - `collate_fn`：packing 多条序列 → `(packed_tokens, cu_seqlens)`
  - 先不做策略搜索，使用固定配置 (pure Ulysses / pure Ring / no parallel)
- [ ] **P1.3** Pure Ulysses SP + varlen 跑通
  - Embedding 层：按 `VocabUtility.vocab_range_from_global_vocab_size` 切分 packed tokens
  - Attention 层：使用 `DistributedAttention` (All-to-All scatter/gather Q,K,V)
  - Loss 层：对应的 gather + cross_entropy
  - 验证：loss 收敛，梯度正确
- [ ] **P1.4** Pure Ring Attention (Zigzag CP) + varlen 跑通
  - Embedding 层：使用 `get_zigzag_local_tokens_and_cu_seqlens` 切分数据
  - Attention 层：使用 `ZigzagRingFlashAttention`
  - Loss 层：对应处理
  - 验证：loss 收敛，梯度正确

### Phase 2: 混合策略 (Ulysses + Ring) 在 varlen 场景跑通
> 目标：让 Ulysses SP 和 Ring Attention 可以组合使用 (如 sp=4, cp=2)

- [ ] **P2.1** 数据切分设计
  - 组合模式下的切分顺序：先 CP (zigzag split) → 再 SP (ulysses split within each CP position)
  - 或者：先 SP (ulysses) → 再 CP (zigzag) —— 需要验证哪种更合理
  - `cu_seqlens` 的正确变换
- [ ] **P2.2** 通信组管理
  - 给定 `total_gpus = sp_size * cp_size`，创建：
    - SP 组：同一 CP position 内的 ranks（用于 All-to-All）
    - CP 组：同一 SP position 跨 CP 的 ranks（用于 Ring P2P）
  - 参考 `CommunicationGroupManager`，但需要重新实现，使其与 galvatron 的 group 管理兼容
- [ ] **P2.3** 模型层适配
  - Attention 层需要同时持有 `sp_group` 和 `cp_group`
  - Forward 顺序：
    1. All-to-All scatter Q,K,V (across SP group)
    2. Zigzag Ring Attention (across CP group)
    3. All-to-All gather output (across SP group)
  - Rotary Embedding 需要正确处理 offset
- [ ] **P2.4** 验证混合策略
  - 使用假数据，固定 `(sp_size, cp_size)` 跑通训练
  - 对比纯 Ulysses / 纯 Ring / 混合 的 loss 和性能

### Phase 3: 自适应策略搜索 (AdaCPSP Solver)
> 目标：复用 FlexSP 的 ILP 求解框架，扩展为支持 `(sp_size, cp_size, attn_type)` 搜索

- [ ] **P3.1** 独立 AdaCPSP CostModel
  - 基于 `flexSPCostModel` 扩展
  - 新增：`total_time_ulysses(seqlen, sp_size)`, `total_time_cp(seqlen, cp_size)`, `total_time_combined(seqlen, sp_size, cp_size)`
  - 使用 Phase 0 的 profiling 数据作为参数
  - 支持分段二次函数的 compute time 估算
- [ ] **P3.2** 独立 AdaCPSP Optimizer
  - 扩展 FlexSP 的 ILP 框架：
    - `sp_options` 扩展为 `strategy_options = [(sp, cp) for sp, cp in valid_combinations]`
    - 约束条件新增：`sp_size * cp_size * group_count = N`
    - 目标函数不变：`min max(group_time)`
  - 支持三种模式：
    - `fix_sp_bfd` → 固定策略 BFD
    - `adaptive_bfd` → 自适应选择最优固定策略
    - `adaCPSP` → ILP 异构搜索
  - 支持 sequence bucketing（复用 FlexSP 的 `bucketing_seqs`）
- [ ] **P3.3** Dataloader 集成
  - 异步求解：`collate_fn` 中使用 `multiprocessing` 提前为下一个 batch 求解
  - Broadcast 求解结果到所有 ranks
  - 动态创建/复用通信组
  - 数据切分：根据求解结果，按 zigzag 方式切分并分配数据
- [ ] **P3.4** 端到端验证
  - 使用真实长度分布数据测试
  - 对比 static/adaptive/adaCPSP 三种模式的吞吐量

## 3. 文件结构规划

```
galvatron_lxy/Hetu-Galvatron/galvatron/models/varlen_llama_hf/
├── arguments.py                    # [新建] 模型参数定义
├── meta_configs/                   # [已有] 模型配置
├── LlamaModel_tensor_parallel.py   # [新建] 基于 llama_hf 版本，添加 varlen 支持
├── LlamaModel_sequential.py        # [已有/重构] 顺序模型包装，需完善
├── LlamaModel_hybrid_parallel.py   # [新建] 混合并行构建入口
├── dataloader.py                   # [新建] varlen dataloader + collate_fn
├── train_dist.py                   # [新建/重构] 主训练脚本
├── llama_scripts/
│   └── train_dist.sh               # [新建] 启动脚本
├── profile_flash_attention.py      # [新建] attention 计算 profiling（从 llama_hf 移植）
├── profile_alltoall.py             # [新建] All-to-All 通信 profiling
├── profile_p2p_ring.py             # [新建] P2P Ring 通信 profiling
└── predict_attention_time.py       # [新建] 预测 attention 时间（从 llama_hf 移植）

galvatron_lxy/Hetu-Galvatron/galvatron/adacpsp_solver/  # [新建] 独立求解器模块
├── __init__.py
├── solver.py                       # AdaCPSPCostModel + AdaCPSPOptimizer
├── utils.py                        # BFD/FFD bin packing, bucketing 等工具
└── multiprocess_utils.py           # 多进程求解工具
```

## 4. 需要删除/废弃的文件

以下文件标记为有问题或冗余，建议删除：

| 文件 | 原因 |
|------|------|
| `varlen_llama_hf/LlamaModel_tensor_parallel_ai.py` | 用户标注为有问题的代码 |
| `varlen_llama_hf/LlamaModel_sequential_ai.py` | 用户标注为有问题的代码 |
| `varlen_llama_hf/adacpsp_dataloader.py` | 需要重新设计，当前逻辑不可用 |
| `varlen_llama_hf/adacpsp_solver.py` | 将移至独立 `adacpsp_solver/` 模块重写 |
| `varlen_llama_hf/train_dist_adacpsp.py` | 基于错误文件的训练脚本，需重写 |
| `varlen_llama_hf/train_dist_varlen_llama.py` | flexsp 的 GPT 训练脚本复制，不适用 |

> ⚠️ 删除前请确认

## 5. 关键技术细节

### 5.1 varlen Flash Attention 接口

```python
# Flash Attention varlen 接口
from flash_attn import flash_attn_varlen_func

# 输入格式：packed tokens, shape = [total_tokens, num_heads, head_dim]
# cu_seqlens: [0, len1, len1+len2, ..., total_tokens], dtype=int32
# max_seqlen: max(seq_lengths)
output = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
```

### 5.2 Zigzag Ring Attention 的数据切分

对于 cp_size=4，序列被切为 8 个 chunk (2*cp_size)：
```
原始: [C0 C1 C2 C3 C4 C5 C6 C7]

Rank 0: [C0, C7]   (first chunk + mirror chunk)
Rank 1: [C1, C6]
Rank 2: [C2, C5]
Rank 3: [C3, C4]
```

### 5.3 Ulysses + Ring 组合的通信组

对于 8 GPUs, sp_size=4, cp_size=2：
```
SP Groups (All-to-All):     CP Groups (Ring P2P):
  [0,1,2,3]                   [0,4]
  [4,5,6,7]                   [1,5]
                               [2,6]
                               [3,7]
```

### 5.4 Attention 层 Forward 流程 (Combined Mode)

```
Input: packed_tokens [total_seq, hidden]
  → cp split (zigzag): local_tokens [total_seq/cp_size, hidden]
  → embedding: hidden_states [total_seq/cp_size, hidden]
  → QKV projection: q,k,v [total_seq/cp_size, num_heads, head_dim]
  → sp split (ulysses all2all): q,k,v [total_seq/cp_size/sp_size, num_heads*sp_size, head_dim]
  → zigzag ring attention (cp ring): output [total_seq/cp_size/sp_size, num_heads*sp_size, head_dim]
  → sp gather (ulysses all2all): output [total_seq/cp_size, num_heads, head_dim]
  → output projection
```

## 6. 即时工作项优先级

1. ✅ 理解现有代码结构
2. 🔄 确认要删除的文件（等用户 review）
3. 🔜 Phase 1: 实现 varlen 场景下的单一策略
   - 先让 Pure Ulysses 在 varlen 下跑通（参考 llama_hf 的实现 + varlen_llama_hf/LlamaModel_sequential.py 的改动）
   - 再让 Pure Ring 在 varlen 下跑通
4. 后续按 Phase 2 → Phase 3 推进

## 7. 开发日志

### 2026-03-02
- 完成代码浏览和理解
- 确认 `_ai` 文件不可用，需基于 `llama_hf` 重新开发
- 确认 `varlen_llama_hf` 缺失的关键文件：`arguments.py`, `LlamaModel_hybrid_parallel.py`, `LlamaModel_tensor_parallel.py`, `dataloader.py`
- 确认 `varlen_llama_hf/LlamaModel_sequential.py` 有部分可用代码（zigzag 切分逻辑）
- 创建开发计划文档

