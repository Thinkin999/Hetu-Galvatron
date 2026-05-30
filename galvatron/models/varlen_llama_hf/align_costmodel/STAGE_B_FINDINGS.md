# Stage B + C 诊断结果（ZeRO-2 数据上）

## TL;DR

发现并修复了一个**严重 bug**：cost model 一直以 L=32 层（LLaMA-7B 默认）跑预测，
但实际模型是 Qwen2.5-7B（L=28 层）。修复后 `ulysses8_chunks1` 从 -15% 改善到 -3.3%。

但仍有 30-50% 系统性偏差，分两类：
- **chunks=8 cells（ring, ulysses8, adacpsp_cauto）欠预测 30-50%**：缺 per-mb overhead
- **usp2x4 cells 过预测 10-30%**：USP 模型里的常数项（layer_extra, ring_step_overhead）在小 mb 上膨胀

---

## 1. 已修复：L=32 vs L=28 bug

### Bug 位置
- `profile_and_validate.py` 第 1673 行：保存的 `attention.config` 缺 `num_layers` 字段
- `adacpsp_solver.py` 第 1597/1797/1896 行：`attn_data.get("num_layers", 32)` 从顶层找，找不到 fallback 到 32

### 修复
1. `profile_and_validate.py`：在 `config` 字典里加 `"num_layers": args.num_layers`
2. `adacpsp_solver.py`：三处构造方法都改成：先看 `config["num_layers"]`，再看顶层，最后 fallback 32
3. **就地修补**了已有 profile `profile_validate_qwen2.5-7b_20260406_225144.json`：`config.num_layers = 28`

### 影响（ZeRO-2 数据，offline reprediction）

| Cell | L=32 (bug) | L=28 (fixed) |
|------|-----------|--------------|
| ulysses8_chunks1 | -15.0% (over) | **-3.3% (close to perfect)** |
| usp2x4_chunks1 | -18.3% (over) | -9.5% (better) |
| usp2x4_chunks8 | -41.0% (over) | -32.4% (better) |
| ring8_chunks8 | +23.6% (under) | +30.5% (worse) |
| ulysses8_chunks8 | +42.5% (under) | +47.5% (worse) |
| adacpsp_cauto | +47.4% (under) | +49.8% (worse) |

L 修复让 chunks=1 cells 大幅改善（因为之前 L=32 过预测 attention，恰好抵消了一些其他过预测），
但 chunks=8 cells 的欠预测**暴露出来**了。

---

## 2. 各 cell 预测分解（修复后）

### Per-cell components（attn / comm / residual / b_step_fb 占 pred 比例）

```
cell                  meas  pred   err   |attn  comm  resid bstep
adacpsp_cauto         7314  3669  +50%  | 29%    8%   62%    0%
ring8_chunks1         2838  2448  +14%  | 53%   10%   38%    0%
ring8_chunks8         5967  4144  +30%  | 20%   22%   27%    1%
ulysses8_chunks1      2728  2819   -3%  | 77%    4%   19%    0%
ulysses8_chunks8      4166  2189  +47%  | 46%   23%   30%    1%
usp2x4_chunks1        2586  2833  -10%  | 42%   19%   41%    0%
usp2x4_chunks8        5063  6703  -32%  |  9%   41%   51%    0%
```

### 关键观察
- **chunks=8 都有大问题**（除了 usp2x4_c8 是过预测，其他都是欠预测）
- **ring8 / ulysses8 / cauto** 同样欠预测 30-50%，差距是 1500-3500ms 每 step
- **usp2x4 c8** 过预测：因为 comm 项（含 a2a_cpu_overhead, ring_step_overhead, layer_extra）
  按 8 个 mb 累计 → 2939ms（41% of pred）。每 mb 即使 seq 很小也会有 ~80ms 固定开销

---

## 3. USP comm 细分

测试用例：usp2x4 (sp=2, cp=4), L=28, mb=11k tokens

```
qo_a2a:        60.7 ms   ( 15%)   — 真实数据 A2A
kv_a2a:        15.0 ms   (  4%)   — 真实数据 A2A
a2a_cpu_oh:    102.4 ms   ( 26%)  — 224 ops × 0.4ms（ulysses+usp overhead constant）
ring_fwd:      12.5 ms   (  3%)   — 真实 ring 数据
ring_bwd:      25.1 ms   (  6%)   — × ring_bwd_comm_ratio=2
ring_step_oh:  96.0 ms   ( 24%)   — 0.5ms × 3 × 2 × 28（ring_step_overhead constant）
layer_extra:   83.2 ms   ( 21%)   — (1.0 + 0.8×2) × 28（usp_layer_overhead constant）
TOTAL:        394.9 ms              ← 71% 是常数项！
```

当 mb=11k tokens 时，**常数项占 71%**！这些常数项在 chunks=8 下会被乘 8 倍，导致严重过预测。

**改进方向**：
- `usp_layer_overhead_base/per_sp` 应该是 per-step 而不是 per-mb
- `ring_step_overhead_ms` 同理
- `ulysses/usp_a2a_overhead_ms` 是 per-op 开销，但在小 msg 下 0.4ms 还是偏大

---

## 4. Long-seq LUT 外推情况

新 profile（C）测得（截至 200/256 点）：
- seq=25k:  21.4 ms/layer
- seq=51k:  86.8 ms/layer
- seq=76k:  199.1 ms/layer
- seq=102k: 358.9 ms/layer

LUT 末段外推到 seq=102k 给出 **341 ms/layer**，**实际只差 5%**！

→ **结论：LUT 外推没我想的那么差**。ulysses 长 seq 过预测的真正原因可能是 `bwd_fwd_ratio=3.2` 在长 seq 下偏高，
不是 LUT 的问题。需要单独验证。

C profile 跑完后能给出更平滑的二次拟合，但收益有限。

---

## 5. ✅ 已实施的修复：per-sp b_microbatch_ms 拟合

### 方法
通过 `chunks=1` vs `chunks=8` 的差额拟合 per-mb 常数：

```
b_sp = ((meas - pred)_c8 - (meas - pred)_c1) / (8 - 1)
```

### 拟合结果

| sp | 旧值（b_decomp_estimated） | 拟合值 | 物理含义 |
|----|---------|-------|--------|
| 1 (ring/cauto) | 0 (zero out) | **201 ms** | 每 mb 200ms 调度+grad-accum 开销 |
| 2 (usp2x4) | 313 (ZeRO-3 残留) | **114 ms** | ZeRO-3 数据高估，ZeRO-2 实际 113 |
| 8 (ulysses8) | 0 (zero out) | **303 ms** | 比 ring 高 100ms，可能因 A2A 多 |

文件：`configs/b_decomp_profile_20260529_zero2_fitted.json`

### 最终效果（ZeRO-2 数据 offline reprediction）

| 指标 | 原始（live bug） | L=28 fix only | + per-mb fit |
|------|------------------|---------------|--------------|
| MAPE | 35.8% | 32.3% | **17.0%** |
| Ranking | 16/21 (76%) | 13/21 (62%) | **16/21 (76%)** |

per-cell（最终）：
```
cell                 meas  pred   err
ring8_c1             2838  2636   +7.1%
ring8_c8             5967  5766   +3.4%   ✓
ulysses8_c1          2728  3185  -16.7%
ulysses8_c8          4166  4622  -11.0%
usp2x4_c1            2586  2638   -2.0%   ✓
usp2x4_c8            5063  5115   -1.0%   ✓
adacpsp_chunksauto   7314  3858  +47.2%   ✗ (混合策略，尚未对齐)
```

### 关键 pairwise speedup 改善

| 对比 | 旧 | 新 |
|------|---|---|
| ring_c1 vs usp2x4_c8 | 76.7% | **8.7%** |
| ulysses_c1 vs usp2x4_c8 | 114% | **13.5%** |
| ulysses_c8 vs usp2x4_c8 | 134% | **9.0%** |
| usp2x4_c1 vs usp2x4_c8 | 46.8% | **0.9%** |

### 物理直觉
- `b_microbatch_ms` 是 **每个微批次的固定开销**：python dispatch、grad-accum buffer 操作、
  backward graph 构造、checkpoint 重计算调度等
- ZeRO-3 的旧 residual 值偏高（含 all-gather），ZeRO-2 下应该减半
- sp=1 vs sp=8 的 ~100ms 差可能因为 ulysses 多 A2A 触发额外 launch

---

## 6. 剩余问题

### adacpsp_cauto +47% 欠预测
solver 自选异构策略（mix of ring/ulysses/usp），per-mb 拟合按 sp 单独做未覆盖混合场景。
需要单独诊断 cauto 选了什么策略，是不是策略组合本身有 bug。

### ulysses8 c1 vs c8 余下 ~15% 偏差（c1 略过、c8 略过）
"per-mb 线性"模型不完美，可能需要更复杂的拟合（含 seq 维度或 chunks 阶跃）。
但这个偏差比之前小一个量级，可暂时容忍。

### 长 seq bwd_fwd_ratio 待验证
当前 3.2 是 4K-16K 测的。等长 seq profile（131k）跑完后，可以重测扩展。
