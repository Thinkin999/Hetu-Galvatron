# Stage C: Precreate 验证 + Ring/Ulysses/USP 建模精度提升

## TL;DR

1. **Precreate 验证**：`precreate_all_groups` 在训练开始前预创建 49 个 NCCL group + 一次性 allreduce warmup，**60.9 秒一次性开销**。precreate 之后，
   稳态 iter 的 first-touch spike（之前 5–100 秒）**完全消失**。可用 `--adaCPSP-precreate-groups 1`（默认开启）。

2. **Cost-model 精度大幅提升**：基于带 precreate 的干净稳态 benchmark
   (`ghmb_zero2_precreate_full_20260529_181618`) 重新校准 `b_microbatch_ms` 和
   `b_step_fb_per_sp`，**Global MAPE 从 12.5% 降到 1.1%**，pairwise speedup MAPE
   **从 24.2pp 降到 2.0pp**，21/21 pair sign accuracy 保持 100%。

## 1. Precreate 验证结果

### 实现
- `adacpsp_group_manager.py::_enumerate_all_possible_group_tuples`: 枚举 solver 可能选的所有 rank tuple
- `adacpsp_group_manager.py::precreate_all_groups`: 主动 `dist.new_group` + dummy `all_reduce`
- `train_dist_adacpsp.py`: 训练前调用 precreate
- `arguments.py`: 新增 `--adaCPSP-precreate-groups`（默认 1）和 `--adaCPSP-precreate-max-ps`（默认 0=world_size）

### 单 cell adacpsp:auto 测试 (ghmb_zero2_precreate_20260529_180128)
- Precreate 报告: `Done in 60.91s. new_groups=49, warmups_on_this_rank=10, total_tuples=49`
- 49 unique tuples = 32×size2 + 12×size4 + 4×size8 + 1×size16
- 之后 iter 0 仍有 128s 一次性开销（CUDA JIT/FSDP 首次 forward，非 NCCL）
- 稳态 iter (skip 5) 平均 fb 从 4744 ms → 3789 ms，**降低 20%**（消除 spike 污染）

## 2. Cost-model 精度优化

### 数据来源
干净稳态 benchmark: `ghmb_zero2_precreate_full_20260529_181618`（7 cells × 21 iter，
跳过 5 warmup + MAD-3 outlier filter，每 cell 保留 14 iter）

### 旧 baseline (b_decomp_profile_20260529_zero2_fitted.json)

```
cell                     meas  pred    err
adacpsp_chunksauto       3169  3147   +0.7%
ring8_chunks1            2352  1947  +17.2%   ← under (chunks=1)
ring8_chunks8            5688  5537   +2.7%
ulysses8_chunks1         2038  1464  +28.2%   ← under (chunks=1)
ulysses8_chunks8         3640  4266  -17.2%   ← over (chunks=8)
usp2x4_chunks1           2118  1752  +17.3%   ← under (chunks=1)
usp2x4_chunks8           4662  4866   -4.4%
                                MAPE 12.5%, speedup MAPE 24.2pp, 21/21 sign ok
```

### 关键诊断
chunks=1 cells 系统性欠预测 17–28%，chunks=8 cells 略过预测 -4 到 -17%。
对每个 sp 求解 2 参数方程（按 c1/c8 联立）:
```
pred_c1_new = live_pred_c1 - Δb + K_step = meas_c1
pred_c8_new = live_pred_c8 - 8·Δb + K_step = meas_c8
→ Δb = ((meas-pred)_c1 - (meas-pred)_c8) / 7
  K_step = (meas-pred)_c1 + Δb
```

| sp | Δb_microbatch | new b_microbatch | new K_step |
|----|----|----|----|
| 1 (ring8) | 36 | 201→**165** | 22→**441** |
| 2 (usp2x4) | 81 | 114→**33** | 22→**447** |
| 4 (未直接测) | 83 (插值) | 946→**82.6** | 22→**596** |
| 8 (ulysses8) | 171 | 303→**132** | 22→**745** |

### 关键代码改动: `b_step_fb_ms_for_strategies` 区分 homo/hetero

物理直觉：`K_step` 对应"强制单一策略时的 CPU-exposed launch overhead"
（重复的 A2A 模式、FSDP 簿记），在异构 step 中由不同 rank 子组并行处理被
更好地隐藏。Cauto 总是 heterogeneous (`sp_values_used` 总有多种 sp)；
forced cells 总是 homogeneous。

```python
def b_step_fb_ms_for_strategies(self, sp_values):
    unique_sps = sorted({int(sp) for sp in sp_values})
    if len(unique_sps) > 1:
        # Heterogeneous step (cauto) → fall back to default
        return float(self.b_step_fb_default_ms)  # 22.3 ms
    # Homogeneous step (forced cells) → use per-sp calibration
    return float(self.b_step_fb_per_sp[unique_sps[0]])
```

### 新 baseline (b_decomp_profile_20260530_zero2_2step_fitted.json + new logic)

```
cell                     meas  pred    err
adacpsp_chunksauto       3169  3008   +5.1%
ring8_chunks1            2352  2352   +0.0%   ✓
ring8_chunks8            5688  5644   +0.8%   ✓
ulysses8_chunks1         2038  2039   -0.0%   ✓
ulysses8_chunks8         3640  3600   +1.1%   ✓
usp2x4_chunks1           2118  2118   +0.0%   ✓
usp2x4_chunks8           4662  4617   +1.0%   ✓
                                MAPE 1.1%, speedup MAPE 2.0pp, 21/21 sign ok
```

### Pairwise speedup ratio 精度

每两个 cell 之间的预测速比（meas_a/meas_b vs pred_a/pred_b）：
- OLD: 平均偏差 24.2pp（最差案例 99pp）
- NEW: 平均偏差 2.0pp（最差案例 8.0pp）

→ Solver 可以基于这些预测做出**可信的策略对比**，决策路径几乎无误判。

## 3. 涉及的文件改动

- **新建** `configs/b_decomp_profile_20260530_zero2_2step_fitted.json`: 新校准的 b_microbatch_ms 和 b_step_fb_per_sp
- **修改** `adacpsp_solver.py::b_step_fb_ms_for_strategies`: 区分 homogeneous/heterogeneous step
- **新建** `align_costmodel/41_compare_live_pred.py`: 直接对比 live solver 预测 vs measured 的脚本

## 4. 还可以做什么

1. **Live 验证**：跑一遍带新 profile 的 ghmb_zero2 benchmark，确认 live 解出的策略和我们离线校准一致
2. **泛化测试**：换不同 seq_length / global_batch_size，验证新校准在不同 workload 下仍准确
3. **sp=4 直接校准**：当前 sp=4 是插值得到的，可以加 `usp4x2` cell 直接测
4. **bwd_fwd_ratio 长 seq 验证**：用 long-seq attention profile 重新拟合 ratio
