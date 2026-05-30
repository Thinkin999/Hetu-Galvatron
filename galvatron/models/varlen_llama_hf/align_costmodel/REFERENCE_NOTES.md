# 参考项目知识点回顾（FlexAR + FlexSP + adacpsp runtime）

> 三个 subagent 调研结果汇总，作为我们做 per-group / 异构组建模与运行时优化的基础。
> 时间：2026-05-27

---

## A. FlexAR（/mnt/bn/wyj-data0-hl/ziyi/FlexAR）

### 核心文件
- 成本模型：`flexar/src/flexar/core/solver/cost_model.py`（`CostModel`, `CostModelConfig`）
- LP 求解：`flexar/src/flexar/core/solver/rematerialize_solver.py`
- 算子多项式：`flexar/src/flexar/core/solver/operators.py`
- 设计文档：`flexar/src/flexar/core/solver/COST_MODEL.md`

### Step 时间分解（最重要）
```
total_iter = Σ_microbatch makespan_mb
makespan_mb = max_p { (fwd + bwd + recompute)_p × L_layers }   # P 个 DP 组取 max
fwd_layer_p = Σ_k (a·s_k² + b·s_k)/sp + Σ_ops c_op + A2A(seq)
```

- 单层 fwd 用 **二次多项式** `a·s² + b·s + c`（big-op 拟合系数）
- 常数项 `cpt_c`（big-op bias）**每组每层加一次**，但 makespan 是 max，**不随组数线性增长**
- 通信用 `bytes / bandwidth_gbs[sp]`，**无 latency floor**
- 异构维度：每组 `d_r` (recompute ratio) 和 `d_o` (offload) 可不同，但 **sp 在 microbatch 内统一**

### 没做的事
- 没有 `c_setup × n_groups` 项
- 没有 latency floor
- 不区分 intra/inter-node
- 没建 FSDP AllGather（仅 D2H overlap guard `flexar_fsdp_ag_overlap_guard_ms`）

### 验证机制
- Plan vs Observed 的 `gap_ms`（每组、每 microbatch、整 iter）
- 整 iter 实测高估 ~+41.8%，主因：microbatch 切换、FSDP AG 争用、offload 时序

---

## B. FlexSP（/mnt/bn/wyj-data0-hl/lqs/src/flexsp/Hetu-Galvatron）

### 核心文件
- `galvatron/flexsp_solver/solver.py`（`flexSPCostModel`, `flexSPOptimizer`）

### Step 时间分解
```
total_iter = Σ_microbatch M_mb
M_mb = max_p { Σ_k total_time_single(seq_k, sp_p) + cpt_beta1 }   # LP min-max
total_time_single = (α₁L² + α₂L)/sp + alltoall_time
```

- `cpt_beta1`（7B ≈ **629 ms**）= **每组每 microbatch 加一次**的固定 bias（compute_bias）
- 是 FlexSP 唯一接近 "per-group setup" 的项
- 但被 **max-over-groups** 吸收 → 不会因为组数变多而线性放大 step time

### 通信
- AllToAll only（DeepSpeed-Ulysses 风格）
- 带宽字典 `alltoall_bandwidth_dict_gbs = {1: ∞, 2: 119.5, ..., 16: 10.3, 32: 5.9}`
- 跨 node 自然 cliff（隐式 intra/inter-node）
- **无 latency floor**

### Compute
- 二次多项式 `α₁·L² + α₂·L`（每模型规模硬编码系数）
- 无 LUT，全靠拟合
- 长 seq 直接外推，无 safeguard

### 异构组
- `sp_options = [1, 1, ..., 2, 2, ..., 8, 8, ...]`（不同 sp 槽位可混在一个 microbatch）
- LP 约束：`Σ_p m[p]·sp_options[p] == N`（设备数守恒）
- 序列→组分配矩阵 `A[k, p]`
- Microbatch 级取 max，跨 microbatch 求和

---

## C. AdaCPSP Runtime 实际 group 创建路径（最关键）

### 核心文件
- `train_dist_adacpsp.py`（训练入口）
- `adacpsp_group_manager.py`（group 创建 + strategy 切换）
- `adacpsp_solver.py`（cost model + solver）
- `core/runtime/pipeline/pipeline.py`（`no_pipeline_forward_backward`）

### Group 创建：**懒创建 + 跨 iter 缓存**
位置：`adacpsp_group_manager.py::_get_or_create_group`（L48-75）

```python
if key not in _created_group_keys:
    new_group = dist.new_group(ranks)   # 集体调用，cold start
    _created_group_keys.add(key)
```

- 模块级全局缓存 `_group_pool` / `_created_group_keys`
- **首次** 遇到某 rank-tuple → 全 world 集体 `new_group`（贵）
- **之后** 同 tuple → 只查缓存（廉）
- ⇒ NCCL communicator **不是** 每 iter 重建

### 每 iter 实际开销点

| Phase | 内容 | 频率 | 是否被建模 |
|-------|------|------|------------|
| solve | rank0 ILP/BFD + broadcast | 1× per iter | 部分（async 模式下 overlap） |
| convert_microbatch_res | 遍历 groups + 集体 `new_group` 查找 | n_microbatches × n_groups | **未建模** |
| set_model_strategy | 全 module 扫描设属性 | n_microbatches | **未建模** |
| token 切片 + cu_seqlens 重建 | torch.cat + new tensor | n_microbatches | **未建模** |
| FSDP all_gather (per-layer) | unshard params | **n_microbatches × n_layers** | b_step_fb（粗糙） |
| forward + backward | 实际计算 + comm | per group, parallel within mb | 已建模（attention + comm） |
| dummy rank 空跑 | 未分配序列的 rank 仍跑全部 layer | per such rank | 未建模 |
| barrier | 每 iter 1 次 | 1× per iter | 未建模 |

### 异构组运行语义（重要）
- 同一 microbatch 内 N 个异构组：**并行**，墙钟 = max(group_time)
- 多个 microbatch：**串行**（无 pipeline overlap）
- 每 rank 同一 microbatch 内只属于一个 group（mutually exclusive）
- **隐式 straggler**：fast group 的 rank 在下次 FSDP all_gather（world 级）或 iter 末 barrier 处等慢 group ⇒ **空等开销不在任何组的 group_time 里**

### Cost model 当前如何聚合（已实现）
```python
mb_time = max_g predict_group(g)              # group 内取 max ✓
step_time = Σ_mb mb_time + b_step_fb         # microbatch 求和 + 全局 bias ✓
# 未加：c_setup × n_groups（待定，34 脚本验证）
# 未加：straggler 项（fast group 等 slow group 的空等）
# 未加：FSDP AG per-microbatch 项（被 b_step_fb 部分覆盖）
```

---

## D. 三个项目的对比

| 维度 | FlexAR | FlexSP | AdaCPSP (我们) |
|------|--------|--------|----------------|
| Attention 模型 | big-op 多项式 `a·s² + b·s + c` | 二次多项式 `α₁s² + α₂s` | LUT + 插值（profile_validate） |
| Per-group bias | `cpt_c` 各 op 求和 | `cpt_beta1` ≈629ms | 隐含在 LUT |
| Comm 模型 | `bytes / bw[sp]` | `bytes / bw[sp]` | v2 profile + **latency floor**（Fix 1）|
| Group 内异构 sp | ❌ | ❌ | ✅（这是论文创新点）|
| Group 内异构 attn_type | ❌（无 Ring/USP）| ❌ | ✅ |
| 异构组聚合 | max | max | max ✓ |
| Microbatch 聚合 | sum | sum | sum ✓ |
| FSDP AG 建模 | overlap guard | 无 | b_step_fb（per sp）|
| Latency floor | ❌ | ❌ | ✅ |
| 长 seq 外推 | 多项式直接外推 | 同左 | LUT 插值（短于最小点用 floor，长于最大点外推）|
| Per-group setup ms | ❌ | ❌（用 cpt_beta1）| **待加** |
| Straggler 模型 | ❌ | ❌ | **待加** |

---

## E. 我们项目的独特挑战 + 借鉴

### 论文创新点（不能丢的差异化）
1. **同一 microbatch 内多种 attention 实现混合**（Ring/Ulysses/USP）
2. **同一 microbatch 内异构 sp/cp**
3. Solver 在异构空间中自动选最优

### FlexSP/FlexAR 都没有的项 → 我们必须加
1. **Per-group setup overhead**（c_setup_ms × n_groups）
   - 来源：dispatch 逻辑、new_group 冷启动、set_model_strategy 扫描、tensor alloc
   - 但要分辨：**冷启动**（仅首 iter）vs **稳态**（每 iter 都有）
2. **Per-strategy launch overhead**（c_launch_ms × n_unique_strategies）
   - Ring 每 step 有 send/recv 固定 latency
   - Ulysses 每次 A2A 有 4 次 NCCL launch
   - USP 套两层 → 双倍
3. **Straggler 空等项**（fast group 等 slow group 的差额）

### 可直接照搬的设计
1. **聚合公式**：FlexAR/FlexSP 都用 `max(group) + sum(microbatch)`，我们已对齐 ✓
2. **二次多项式 compute 拟合**：长序列时比 LUT 更稳定（可考虑混合：LUT + 超出范围用多项式外推）
3. **`compute_bias` 思路**：FlexSP 的 `cpt_beta1` 加在每组，与 "per-group setup" 概念正交：beta1 是**计算端的 fixed overhead per group**，我们的 c_setup 是**dispatch/runtime 端的 fixed overhead per group**

### Runtime 端可以改源码的优化（不是建模，是真去做掉）
1. **预创建 communicator pool**：启动时 init 全部可能的 rank tuple
2. **set_model_strategy 缓存**：避免 per-mb 全 module 扫描
3. **token buffer 预分配**：避免 per-mb torch.cat/new tensor
4. **跨 microbatch 共享 FSDP unshard**：unshard 一次跑完所有 microbatch
5. **dummy rank fast path**：跳过空 group 的 compute（需 FSDP 兼容）
6. **启用 Ulysses sp_stream overlap**：当前 dead code 未启用

---

## F. 当前 cost model 的剩余误差画像（offline 验证）

| cell | error% | 主因猜测 | 与上面 E 的对应 |
|---|---|---|---|
| `ulysses8_c1` | 26.7% | 长 seq compute 低估 | F.compute（LUT 外推不准）|
| `usp2x4_c8` | 33.9% | sp=2 + microbatch×8，FSDP AG 累计 | b_step_fb 仍偏低 |
| `adacpsp_cauto` | 44.6% | solver 多小组方案 | **per-group setup + straggler 未建模** |

---

## G. 建议的下一步优先级（待用户确认）

### 优先级 1（高 ROI，必做）
- **同时**做建模 + 源码优化两条线：
  - 建模线：加 `c_setup_ms × n_groups`（短期降低 adacpsp_cauto 误差）
  - 源码线：实现 communicator 预创建、token buffer 预分配（实测降低真实开销）

### 优先级 2
- 重新 profile attention：补长 seq 数据点（>64k）让 LUT 不靠外推
- 添加 straggler 项：`max(group) + α × (max - min)`（不平衡惩罚）

### 优先级 3
- 探索跨 microbatch FSDP unshard 共享（最大潜在收益但改动大）

---

## H. 参考代码片段（可直接借鉴的写法）

### FlexSP 的 per-group bias 加法（最清晰）
```python
# /mnt/bn/wyj-data0-hl/lqs/src/flexsp/Hetu-Galvatron/galvatron/flexsp_solver/solver.py
def compute_time(self, seqlen, sp_size=1):
    cpt_times = [self.compute_time_single(seq, sp_size) for seq in seqlen]
    return sum(cpt_times) + self.cpt_beta1   # ← 每组一次

# Microbatch level (LP):
for p in range(P):
    group_time = sum(total_time_single(seq_k, sp_p) for k) + compute_bias()
    M = max(M, group_time)
```

### FlexAR 的 big-op 拟合（每个 op 一个常数项）
```python
# /mnt/bn/wyj-data0-hl/ziyi/FlexAR/flexar/src/flexar/core/solver/operators.py
popt, _ = curve_fit(quadratic_func, seqlens, fwd_times, sigma=fwd_times)
operators.append(Operator(cpt_a=popt[0], cpt_b=popt[1], cpt_c=popt[2]))
# layer 聚合：cpt_fwd_coeff = [Σa, Σb, Σc]
```

### AdaCPSP runtime group 创建（已 cached）
```python
# adacpsp_group_manager.py:48-75
def _get_or_create_group(ranks):
    key = tuple(sorted(ranks))
    if key not in _created_group_keys:
        new_group = dist.new_group(ranks)
        _created_group_keys.add(key)
        if rank in ranks:
            _group_pool[key] = new_group
    return _group_pool.get(key)
```
