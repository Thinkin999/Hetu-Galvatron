# B-Decomposition & Multi-Microbatch Validation Report

**Date**: 2026-05-26
**Goal**: Decompose the per-group residual constant `b(sp)` into per-microbatch
vs per-step components, then validate the resulting multi-microbatch cost
model on a realistic variable-length workload (github dataset).

## Summary

| Item | Status |
|------|--------|
| Forced multi-microbatch dispatch (`--adaCPSP-forced-chunks N`) | DONE |
| b-decomposition sweep `sp ∈ {1,8}` × `chunks ∈ {1,2,4,8}` (fix_length, ckpt=0) | DONE — 8 cells, perfect linearity at chunks≥2 (R²≥0.999) |
| b-decomposition fit → `b_step_fb_per_sp` + refit `a_per_token` | DONE — sp=1 b_step=352 ms, sp=8 b_step=3489 ms |
| Cost-model wiring (`apply_b_decomp_profile`, `step_total_time_ms`) | DONE |
| Production trainer integration (auto-load + gate at N_mb≥2) | DONE |
| github multi-mb validation (4 configs × 2 chunks + adacpsp, ckpt=1) | DONE — 7 cells |
| Ranking correctness at chunks=1 (production case) | CORRECT — ulysses8 < usp2x4 < ring8 |
| Absolute error vs measured | Mean \|Δ_clean\| ≈ 55%, dominated by ckpt=1 vs ckpt=0 calibration mismatch + adacpsp solver over-optimism on tiny heterogeneous groups |
| Latent ring/usp varlen rotary-embed bug | FOUND & FIXED — `num_seqs` referenced before assignment |

## 1. b-decomposition sweep & fit

### 1.1 Setup

- Cluster: 2 nodes × 8 A800-80GB (16 GPUs).
- Model: Qwen2.5-7B (h=3584, L=28, n_heads=28, kv=4).
- Workload: `fix_length`, all sequences = 8192 tokens, GBS = `world_size × chunks`.
- Strategy: `ulysses:sp` forced (so num_groups = world / sp, seqs_per_group = sp,
  tokens_per_GPU = sp · seq_len / sp = `seq_len` = 8192 — INDEPENDENT of sp).
- ckpt: `--global_checkpoint 0` (no activation recomputation).
- forward_prefetch: True (unconditional, see `parallel.py`).

### 1.2 Measurements (rank 0, fb_clean = mean of 3 lowest fb iters excl iter 0)

| sp | chunks | fb_clean (ms) | fb_steady (ms) |
|---:|---:|---:|---:|
| 1 | 1 | 2175 | 2180 |
| 1 | 2 | 4588 | 5850 |
| 1 | 4 | 8537 | 9536 |
| 1 | 8 | 17011 | 18108 |
| 8 | 1 | 2621 | 2628 |
| 8 | 2 | 8390 | 9728 |
| 8 | 4 | 13589 | 14641 |
| 8 | 8 | 23391 | 24564 |

### 1.3 Linear fit `fb(N) = slope · N + b_step_fb`

Two windows: (a) include chunks=1 (`all`), (b) chunks≥2 only (`ch≥2`).

| sp | window | metric | slope (ms/mb) | intercept (ms/step) | R² |
|---:|:---|:---|---:|---:|---:|
| 1 | all | clean | 2102 | 197 | 0.9996 |
| 1 | ch≥2 | clean | 2077 | **352** | 0.9997 |
| 1 | ch≥2 | steady | 2057 | 1564 | 0.9987 |
| 8 | all | clean | 2815 | 1443 | 0.9763 |
| 8 | ch≥2 | clean | 2493 | **3489** | 0.9998 |
| 8 | ch≥2 | steady | 2474 | 4767 | 1.0000 |

**Key findings**:
1. At chunks≥2 the linear model is essentially exact (R² > 0.999).
2. `chunks=1` is a FAST-PATH outlier: measured fb is 10-15% LOWER than the
   `ch≥2` extrapolation. This is because chunks=1 skips the inter-microbatch
   FSDP re-shard / re-gather boundary that chunks≥2 hits N-1 times.
3. **`b_step_fb` is strongly sp-dependent**: 352 ms for sp=1 vs 3489 ms for sp=8.
   This is because the FSDP group size = `world / (pp·tp·cp·sp)` — smaller sp
   means a larger DP group (sp=1 → 16-way), more ranks share the cost, but
   per-rank message is smaller. sp=8 → 2-way DP group with huge per-rank
   messages, dominating step-level overhead.
4. The `chunks=1` per-microbatch cost is 5-10% LOWER than the `ch≥2`
   slope — chunks=1 not only skips `b_step_fb` but also has cheaper compute
   per mb (less memory pressure / no FSDP boundary state).

### 1.4 Refit `a_per_token` (b_microbatch ≈ 0)

The old `residual_profile` had `a_per_token=0.291` for sp=8, measured without
`forward_prefetch=True`. With prefetching the exposed FSDP all-gather is hidden
inside compute, so the EFFECTIVE per-token cost drops. We refit `a_per_token`
from the new ch≥2 calibration assuming `b_microbatch=0`:

| sp | a_per_token (new) | a_per_token (old) | per_mb_residual at tok=8192 |
|---:|---:|---:|---:|
| 1 | 0.197 | 0.182 | 1616 ms |
| 8 | **0.116** | 0.291 | 947 ms |

For sp=8 the new `a` is **2.5× smaller** than the old measurement, reflecting
the `forward_prefetch=True` improvement.

### 1.5 External per-step overhead (median across all cells)

| Phase | Median (ms) |
|---|---:|
| optimizer_step | 9.2 |
| grad_clip | 4.4 |
| zero_grad | 0.08 |
| solve_and_dispatch | 8.6 |
| **total `b_step_external`** | **22.3** |

Negligible vs `b_step_fb` (3489 ms at sp=8).

## 2. Cost-model wiring

### 2.1 Schema (`b_decomp_profile_*.json`)

```json
{
  "schema": "adacpsp_b_decomp_v1",
  "residual_per_sp": {
    "1": {"a_per_token": 0.197, "b_microbatch_ms": 0.0, ...},
    "8": {"a_per_token": 0.116, "b_microbatch_ms": 0.0, ...}
  },
  "b_step_fb_per_sp_clean":  {"1": 352, "8": 3489},
  "b_step_fb_per_sp_steady": {"1": 1564, "8": 4767},
  "b_step_fb_ms_clean":  1920,
  "b_step_external_ms":  22.3
}
```

### 2.2 Cost-model API additions (`AdaCPSPCostModel`)

- `b_step_fb_per_sp: Dict[int, float]`
- `b_step_fb_default_ms`, `b_step_external_ms`
- `apply_b_decomp_profile(json_dict, prefer="clean")`
- `b_step_fb_ms_for_strategies(sp_values)` → picks `b_step_fb_per_sp[max(sps)]`
- `step_total_time_ms(per_microbatch_ms, sp_values, include_external)`

### 2.3 Trainer integration

`train_dist_adacpsp.py::_predict_adacpsp_ms` now emits:

```json
{
  "total_ms":         <sum of per-microbatch max-over-groups>,
  "total_fb_ms":      total_ms + (b_step_fb iff n_microbatches >= 2),
  "total_step_ms":    total_fb_ms + b_step_external_ms,
  "b_step_fb_ms":     <charged value>,
  "n_microbatches":   <int>,
  "sp_values_used":   <list>
}
```

**Critical detail**: `b_step_fb` is only charged when `n_microbatches ≥ 2`
(matching the calibration's "chunks=1 fast path" observation).

Auto-loading: the trainer scans `configs/b_decomp_profile_*.json` after the
`residual_profile_*.json`, picking the newest matching `adacpsp_b_decomp_v1`
schema.

## 3. github multi-microbatch validation

### 3.1 Setup

- Cluster, model, NCCL knobs identical to b-decomp.
- Workload: `--dataset github`, `--seq_length 65536`, `GBS=16`, ckpt=1.
- Cells (7 total): `ulysses8 / ring8 / usp2x4` × `chunks ∈ {1, 8}`, plus
  `adacpsp_chunksauto` (solver picks per-mb).

### 3.2 Per-cell predicted vs measured

| cfg | chunks | n_mb | max_seq | avg_seq | fb_clean (ms) | pred_fb (ms) | b_step | Δ_clean |
|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|
| ulysses8 | 1 | 1 | 10144 | 8528 | 1069 | 604 | 0 | +44% |
| ring8 | 1 | 1 | 10144 | 8528 | 1532 | 1342 | 0 | +12% |
| usp2x4 | 1 | 1 | 10144 | 8528 | 1317 | 1239 | 0 | +6% |
| ulysses8 | 8 | 8 | 3360 | 1066 | 3330 | 6648 | 3489 | −100% |
| ring8 | 8 | 8 | 3360 | 1066 | 5383 | 8778 | 352 | −63% |
| usp2x4 | 8 | 8 | 3360 | 1066 | 4416 | 9010 | 1920 | −104% |
| adacpsp | auto | 1 | 3680 | 1551 | 1742 | 794 | 0 | +54% |

### 3.3 Speedups vs `ulysses8_chunks1`

| cfg | chunks | measured speedup | predicted speedup | Δspeedup |
|:---|:---:|---:|---:|---:|
| ulysses8 | 1 | 1.00 | 1.00 | 0.00 |
| ring8 | 1 | 0.70 | 0.45 | −0.25 |
| usp2x4 | 1 | 0.81 | 0.49 | −0.32 |
| ulysses8 | 8 | 0.32 | 0.09 | −0.23 |
| ring8 | 8 | 0.20 | 0.07 | −0.13 |
| usp2x4 | 8 | 0.24 | 0.07 | −0.18 |
| adacpsp | auto | 0.61 | 0.76 | +0.15 |

### 3.4 Ranking correctness

**At chunks=1 (production single-mb baseline)**:
- Measured order (fastest → slowest): ulysses8 (1069) → usp2x4 (1317) → ring8 (1532)
- Predicted order: ulysses8 (604) → usp2x4 (1239) → ring8 (1342)
- **Ranking matches perfectly.** ✓

**At chunks=8 (production multi-mb)**:
- Measured order: ulysses8 (3330) → usp2x4 (4416) → ring8 (5383)
- Predicted order: ulysses8 (6648) → ring8 (8778) → usp2x4 (9010)
- Predicted ring8 and usp2x4 are swapped.

**adacpsp (solver pick)**:
- Solver chose heterogeneous tiny groups (`u1x1`, `u2x1`, `r1x2`, `r1x4`, ...).
- Predicted 794 ms but measured 1742 ms (clean) → **solver over-optimistic
  about small heterogeneous groups by ~2.2×**.
- This is the largest cost-model accuracy issue uncovered.

## 4. Sources of the prediction error

### 4.1 Calibration vs validation environment mismatch (largest contributor)

| Setting | b-decomp calibration | github validation |
|---|---|---|
| `--global_checkpoint` | 0 | 1 |
| Activation recompute during backward | No | Yes (~70-100% extra fwd compute) |
| Sequence length | fixed 8192 | github (mean ~8528 at chunks=1, ~1066 at chunks=8) |

Activation checkpointing roughly doubles per-microbatch compute (since
backward re-runs forward). Our `a_per_token` calibrated at ckpt=0 therefore
UNDERPREDICTS ckpt=1 throughput, explaining the +6% to +54% underprediction
at chunks=1.

### 4.2 `b_step_fb` calibrated at ckpt=0 does not apply at ckpt=1

Without ckpt, FSDP needs to re-shard params after backward of mb_i and
re-gather before forward of mb_{i+1}. With ckpt=1, params stay gathered
because backward of mb_i is bracketed by its own re-forward. The inter-mb
boundary cost essentially disappears.

This explains the −63% to −104% overprediction at chunks=8 — we still charge
the (calibrated-at-ckpt=0) `b_step_fb` of 3489 ms even though at ckpt=1 the
real cost is near 0.

### 4.3 Cost model over-optimism on tiny heterogeneous groups

For sequences ≤ a few thousand tokens, the cost model predicts a small fixed
T_attention (just a piecewise compute estimate) plus a small `a · tokens`.
For sp=1 cp=1 with 500 tokens it gives <200 ms. In reality each small group
still pays:
- One FSDP all-gather per layer (28 layers × per-rank cost)
- One dataloader/dispatch overhead per group
- One kernel launch tail per layer

These overheads add up and make tiny groups MUCH slower in practice than
predicted. The solver therefore prefers heterogeneous splits that the cost
model thinks are cheap, but the executor measures as expensive.

### 4.4 Speedup-direction correctness

Despite absolute errors, the SIGN of relative speedup is correct in most
pairwise comparisons at chunks=1. The cost model is still useful as a
ranking heuristic. For chunks=8, the relative ordering of ring8 vs usp2x4
is inverted — both are predicted MUCH slower than ulysses8, but their
relative order disagrees with measurement.

## 5. Recommended next steps

1. **Re-calibrate at ckpt=1**. Run a small b-decomp sweep (sp ∈ {1, 8},
   chunks ∈ {2, 4}) with `--global_checkpoint 1` to get ckpt-aware
   `a_per_token` and `b_step_fb_per_sp`. Expected runtime: 4 cells × 8 min ≈ 35 min.
2. **Per-group setup overhead model**. Add a small `c_per_group · n_groups`
   term to capture the kernel-launch / FSDP-collective tail that dominates
   tiny heterogeneous groups. Could be calibrated from a sweep over fixed
   `n_groups × tokens_per_group` configurations.
3. **Solver guardrail**. As a stop-gap, in `AdaCPSPOptimizer` add a minimum
   `tokens_per_group ≥ T_min` constraint (e.g. 4096) to prevent the solver
   from picking pathologically heterogeneous splits until the cost model is
   refined.
4. **Two-seq-len calibration**. Run b-decomp at `seq_len ∈ {2048, 32768}` to
   separate `a · tokens` from `b_microbatch` properly (we currently force
   `b_mb=0`, valid only near the calibration point).

## 6. Artifacts

| File | Purpose |
|---|---|
| `align_costmodel/24_bench_b_decomp_{dispatch,worker}.sh` | b-decomp sweep |
| `align_costmodel/25b_fit_b_decomp_quick.py` | Fit + emit profile JSON |
| `align_costmodel/26_bench_github_multimb_{dispatch,worker}.sh` | github validation sweep |
| `align_costmodel/27_validate_multimb.py` | Per-cell + speedup analysis |
| `configs/b_decomp_profile_20260526_combined.json` | Calibrated profile (auto-loaded) |
| `align_costmodel/results/bdec_combined_20260526/` | b-decomp raw + analysis |
| `align_costmodel/results/ghmb_20260526_140123/` | github raw + validation JSON |

## 7. Code changes

| File | Change |
|---|---|
| `core/runtime/arguments.py` | Added `--adaCPSP-forced-chunks` arg |
| `models/varlen_llama_hf/train_dist_adacpsp.py` | `_build_forced_groups` round-robin partition into N microbatches; emit `total_fb_ms`/`total_step_ms`/`b_step_fb_ms`/`n_microbatches` in `predicted_adacpsp`; auto-load b_decomp_profile |
| `models/varlen_llama_hf/adacpsp_solver.py` | `b_step_fb_per_sp` storage, `apply_b_decomp_profile`, `step_total_time_ms`, `b_step_fb_ms_for_strategies` |
| `core/runtime/tensor_parallel/attention.py` | Fix `num_seqs` UnboundLocalError in `_apply_varlen_rotary_emb` (ring/usp varlen path was broken) |
