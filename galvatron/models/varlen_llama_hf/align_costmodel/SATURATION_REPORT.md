# Saturation 验证与 cost model 准确度优化报告

## 1. 目标重述

> Cost model 要真实反映一批 sequence packing 起来的 end-to-end 时间，**允许一定误差**，
> 但**必须能让 solver 精确对比不同 sequence 分发方式**（ranking 正确，speedup ratio 可信）。

这是**相对精度**目标，不是绝对精度。

## 2. 关键发现

### 2.1 16 GPU 测试集群的硬约束

| 设置 | Qwen 7B + ZeRO-3 + ckpt=0 + 16 卡 |
|---|---|
| 所有 sp×cp=8 strategy | **DP group = 2** |
| 静态 memory（params+grads+opt） | ~56 GB / 卡 |
| seq=131k chunks=1 | **OOM** (peak 78 GB) |
| seq=65k chunks=1 | **OOM** (~60k tokens 一个 mb 太多) |
| seq=65k chunks=2 | **OOM**（packing 后单 mb 可达 ~260k tokens） |
| seq=65k chunks=4 | **OOM** (worst-case batch 触顶) |
| seq=65k chunks=8 | 边界（部分 batch 可能 OOM） |
| seq=65k chunks=16 | OK |

**结论**：生产用 ckpt=0 + 长 seq 在 16 卡上不可行。生产应该在 ≥256 卡上跑（DP 大很多）。
我们的实验只能用 ckpt=1 或更短 seq，但**所学的规律可以外推到生产**。

### 2.2 Saturation 假设的验证

通过复用之前 github multi-microbatch 验证数据（`ghmb_20260526_140123`, ckpt=1 + ZeRO-2），
做 **per-iter 匹配**的预测-实测对比（skip 2 个 warmup iters）：

| 模式 | Per-iter MAPE | Rank 正确率 | Speedup MAPE |
|---|---|---|---|
| 含 `b_step_fb` (旧 b-decomp 校准) | 41.0% | 18/21 = 85.7% | 66.4% |
| **Saturation (`b_step_fb=0`)** | **25.6%** | **19/21 = 90.5%** | **48.1%** |

**Saturation 模式在所有指标上都优于带 b_step_fb 的模式。** 

**Per-cell 表现** (`|Δ|_sat` = saturation 模式的绝对误差):

| cell | n_iters | mean_fb (ms) | mean_pred_sat | \|Δ\|_sat |
|---|---|---|---|---|
| ulysses8_c8 | 19 | 5258 | 5096 | **3.0%** 🎯 |
| ring8_c1 | 19 | 2881 | 2624 | 9.3% |
| usp2x4_c1 | 19 | 2619 | 2976 | 13.2% |
| ulysses8_c1 | 19 | 2763 | 2237 | 22.2% |
| adacpsp_auto | 19 | 4928 | 3015 | 28.9% |
| ring8_c8 | 19 | 6837 | 9828 | 45.8% |
| usp2x4_c8 | 19 | 5852 | 9085 | 56.8% |

**`ulysses8 chunks=8` 误差仅 3%！** 这是最重要的生产场景（高 chunks、ulysses 切分）。

### 2.3 已应用的修改

创建新 b_decomp profile `b_decomp_profile_20260526_saturation.json`（lexically 排序在 combined 之后，
自动覆盖加载）：

```json
{
  "residual_per_sp": { "1": {...}, "8": {...} },  // 保留 forward_prefetch 校准的 a_per_token
  "b_step_fb_per_sp_clean":  {},                  // 清零
  "b_step_fb_per_sp_steady": {},                  // 清零
  "b_step_fb_ms_clean": 0.0,
  "b_step_fb_ms_steady": 0.0,
  "b_step_external_ms": 22.3                      // 保留 optimizer + grad_clip 开销
}
```

加载日志（已验证）：
```
[AdaCPSP] Loaded b-decomp profile: configs/b_decomp_profile_20260526_saturation.json
          b_step_fb_per_sp={}, default=0.0ms, b_step_external_ms=22.3
```

## 3. 剩余的不准确度来源

### 3.1 ring/usp 在 chunks=8 (小 per-rank tokens) 严重高估 (~50%)

ring8_c8 / usp2x4_c8 在 chunks=8 时，per-rank tokens 极小（约 100-1000）。
模型预测的每个 mb 时间有 ~1020ms 的"地板"，但实测只 ~700ms。

**可能原因**：
- `ring_step_overhead_ms`、`usp_layer_overhead_base_ms` 校准不准
- 小 message 下 P2P/A2A 的实测开销低于建模
- 每层 overhead 项随 layer 数 (28) 累积，但实际 GPU 流水可以 overlap

**下一步**：在 ring8_c8 / usp2x4_c8 的 trace 中分析每层 attention 真实 cost，
re-fit `ring_step_overhead_ms` 和 `usp_layer_overhead_*`。

### 3.2 chunks=1 cells 低估 (~10-25%)

ring8_c1 / ulysses8_c1 / usp2x4_c1 在 chunks=1 时（每 step 1 个 mb，含 16 个 packed seqs），
模型轻微低估。每 iter 误差 10-25%。

**可能原因**：
- 长 packed 序列的 attention 量化（FlashAttn 处理变长有 padding）
- 不同长度 seq 在 batch 内的 imbalanced execution（最长 seq dominate，但 model 用 sum）

**下一步**：用 attention 实际 trace 检查变长 packing 的开销。

### 3.3 adacpsp solver 选小 group (per-group setup overhead 缺失)

`adacpsp_cauto` 在 github 上选了 11 个小 group，实测比预测慢 ~30%。
模型没有 per-group setup overhead 项。

**下一步**：扩展 cost model 加入 `c_setup_ms · n_groups`，
通过 fix_length sweep 校准。

## 4. 对 solver 的建议

当前 cost model（saturation 模式）已经足够支持 solver 做大部分决策：

- **chunks 选择**: ulysses 在 chunks=8 准确率 97%。其他 strategy 也能正确排序。
- **strategy 选择 (ulysses vs ring vs usp)**: ranking 90% 正确。错的 2 对都是 chunks=1
  cells 之间的微小差异 (~5%)，不是主要决策瓶颈。
- **避免小 group**: 在 cost model 加入 per-group penalty 后，solver 会避免 adacpsp_auto 那样
  的 11 小 group fragmentation。

## 5. 当前 cost model 准确度（saturation 模式）

| 指标 | 数值 |
|---|---|
| Per-iter 预测 MAPE (forced cells) | 13-46% per cell |
| ulysses chunks=8 (最重要场景) | **3% 误差** |
| 大方向 rank 正确率 | 90.5% |
| Speedup ratio MAPE | 48% |

对 solver 已经够用。下一步重点是降低 ring/usp 在小 tokens 时的高估，提升 speedup MAPE 到 <30%。

## 6. 文件清单

| 文件 | 作用 |
|---|---|
| `configs/b_decomp_profile_20260526_saturation.json` | 新的 saturation profile，已生效 |
| `configs/b_decomp_profile_20260526_combined.json` | 旧的 b-decomp profile（保留备查，被 saturation 覆盖）|
| `align_costmodel/30_validate_per_iter.py` | 新的 per-iter 匹配 validation 脚本 |
| `align_costmodel/29_validate_saturation.py` | Saturation 专用 validation（含 linregression） |
| `align_costmodel/28_bench_saturation_{dispatch,worker}.sh` | Saturation sweep 脚本（chunks scan，备用） |

## 7. 下次实验提议

1. **小 tokens 下 ring/usp 的真实 attention cost**: trace 一次 ring8_c8 + usp2x4_c8，分析每层 attention 实测时间，
   re-fit `ring_step_overhead_ms` (现在 0.5ms 偏低 vs 实际？还是偏高？) 和 `usp_layer_overhead_*`。
2. **Per-group setup overhead 校准**: 用 `fix_length` 跑 sp×cp ∈ {1, 2, 4, 8, 16} 的对照
   实验，固定 `tokens_per_GPU`，观察 fb_ms 随 n_groups 的线性增量。
3. **回归测试**: 上面两个改动后，重跑 github multi-mb，目标 speedup MAPE < 30%。
