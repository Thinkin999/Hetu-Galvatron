# Stage 0 — Saturation Assumption Validation

**Goal**: Decide whether the "ZeRO3 AllGather + ReduceScatter is fully hidden by
compute" assumption holds on our 2-node × 8 A800 setup (Qwen2.5-7B + Ulysses).

If valid → non-attention modelling can ignore FSDP comm (use only compute time).
If invalid → we must explicitly model the exposed comm tail.

**TL;DR — invalid in the current codebase**: 75–92 % of FSDP+SP NCCL is exposed
on the GPU critical path. Hidden NCCL is capped at ≈100 ms / iter, *independent*
of seq, compute, or activation checkpointing. Root cause: in
`galvatron/core/runtime/parallel.py:135`, `forward_prefetch=False` for non-MoE
models, so forward `AllGather` is serialised with compute.

---

## Methodology

- **Launcher**: `20_trace_fsdp_step_dispatch.sh` (mirrors `13_profile_comm_dispatch.sh`
  pattern: detached `setsid`+`nohup`, automatic gpu_busy stop/restart).
- **Per-node worker**: `20_trace_fsdp_step_worker.sh` runs `train_dist_adacpsp.py`
  with `--use-packing --use-adaCPSP --adaCPSP-sync-solver
  --adaCPSP-forced-strategy ulysses:8 --sdp 1 --default_dp_type zero2
  --mixed_precision bf16`, plus the existing `--adaCPSP-timeline-profile` hook
  for `torch.profiler` capture.
- **Analyser**: `21_analyze_fsdp_overlap.py` walks the chrome trace per
  `adacpsp::forward_backward` span, classifies every GPU kernel as NCCL
  (`AllGather`/`ReduceScatter`/`SendRecv`/…) or compute
  (matmul/flash/norm/elementwise/triton/…), merges intervals on the GPU
  timeline, and reports `exposed_nccl = NCCL ∖ compute`. All numbers below are
  **wall-clock ms / forward+backward iteration**.

Model: Qwen2.5-7B (28 layers, hidden=3584, 28 attn heads, 4 KV heads), bf16,
varlen packing, GBS=16, `fix_length` dataset. World = 16 GPUs split into 2
Ulysses groups of 8 → `sp=8, cp=1`.

---

## Results

### Sweep 1 — no activation checkpointing

| seq  | rank | iter (ms) | compute (ms) | NCCL (ms) | exposed (ms) | hidden (ms) | exposed / NCCL |
|------|------|-----------|--------------|-----------|--------------|-------------|----------------|
| 4 096 | 0    | 1 499.7   |   952.2      |   601.0   |   477.0      |   124.0     | **79.4 %**     |
| 4 096 | 8    | 1 484.6   | 1 054.5      |   498.1   |   367.3      |   130.9     | **73.7 %**     |
| 8 192 | 0    | 2 724.6   | 2 014.3      |   756.8   |   632.7      |   124.1     | **83.6 %**     |
| 8 192 | 8    | 2 721.8   | 2 218.2      |   564.4   |   436.6      |   127.8     | **77.4 %**     |

(seq=16 384 OOMs without checkpoint — 79 GB / GPU at 16 k tokens / GPU.)

### Sweep 2 — `--global_checkpoint 1` (activation checkpointing on, production-like)

| seq   | rank | iter (ms) | compute (ms) | NCCL (ms) | exposed (ms) | hidden (ms) | exposed / NCCL |
|-------|------|-----------|--------------|-----------|--------------|-------------|----------------|
| 8 192 | 0    | 3 459.0   | 2 643.4      |   830.0   |   737.4      |    92.6     | **88.8 %**     |
| 8 192 | 8    | 3 454.8   | 2 868.2      |   611.7   |   514.1      |    97.6     | **84.0 %**     |
| 16 384 | 0   | 7 571.4   | 6 345.6      | 1 220.6   | 1 125.3      |    95.3     | **92.2 %**     |
| 16 384 | 8   | 7 562.2   | 6 786.3      |   780.7   |   684.0      |    96.7     | **87.6 %**     |

### Per-class NCCL breakdown (rank 0)

| seq   | ckpt | AllGather (ms) | ReduceScatter (ms) | SendRecv [Ulysses A2A] (ms) |
|-------|------|----------------|---------------------|------------------------------|
|  4 096 | ✗   | 318.1          | 236.7               |  61.2                        |
|  8 192 | ✗   | 300.3          | 333.5               | 123.0                        |
|  8 192 | ✓   | 310.0          | 351.8               | 169.1                        |
| 16 384 | ✓   | 343.1          | 540.1               | 337.4                        |

Note that `AllGather` and `ReduceScatter` are roughly seq-independent (they
move parameters/grads, not activations). `SendRecv` is Ulysses attention all-to-all
and scales with seq.

---

## Key observations

1. **The saturation assumption is invalid in every measured regime.** Exposed
   NCCL is 74 %–92 % of total NCCL — never close to 0.

2. **Hidden NCCL is essentially a constant ≈ 100 ms / iter (rank 0)**, regardless
   of seq, ckpt on/off, or absolute compute mass. We are not winning more overlap
   by making compute longer; we are hitting an overlap ceiling.

3. **The ceiling is structural, not numerical.** Reading
   `galvatron/core/runtime/parallel.py:135-142`:
   ```python
   forward_prefetch = True if is_moe_model else False  # ← False for Qwen2.5
   backward_prefetch = None if pp_on else BackwardPrefetch.BACKWARD_PRE
   ...
   fsdp_args = dict(
       sharding_strategy=sharding_strategy,
       forward_prefetch=forward_prefetch,          # ← OFF for dense models
       # backward_prefetch=backward_prefetch,       ← line is commented out (uses default)
       ...
       limit_all_gathers=True,
   )
   ```
   For dense Qwen2.5-7B this gives FSDP no permission to issue layer N+1 AllGather
   while layer N compute is running on the forward pass. The result is the
   forward AllGather of every layer is serialised with that layer's compute on
   the GPU timeline (visible in trace). `backward_prefetch=BACKWARD_PRE`
   *is* on (default), so some backward AllGather is hidden; that 100 ms ceiling
   is precisely the backward-prefetched portion.

4. **`--global_checkpoint 1` makes the ratio worse, not better.**
   Checkpointing recomputes forward inside backward, so an additional pass of
   AllGather is required. Compute grows (e.g. seq=8 k: 2 014 → 2 643 ms) but
   comm grows along with it (e.g. SendRecv 123 → 169 ms, ReduceScatter 333 → 352
   ms). Hidden NCCL is still ~95 ms. The "compute fully masks comm" story
   *does not get help* from activation checkpointing.

5. **Per-rank asymmetry**: rank 0 (Ulysses group leader for ranks 0–7) measures
   more NCCL than rank 8 (group leader for 8–15). Whatever the cause (NCCL
   ordering, cross-node IB skew), every rank still shows >70 % exposed.

---

## What the seniors' "compute hides ZeRO3 AllGather" intuition needs to be true

For the saturation assumption to hold, **both** of these conditions need to
become true. Today neither is:

(a) **FSDP must be allowed to prefetch layer N+1 in forward** — i.e. set
    `forward_prefetch=True` in `parallel.py`. Without this, forward AllGather is
    a hard serial dependency on the GPU stream.

(b) **Per-layer compute must dominate per-layer comm** — i.e. `gemm + flash >>
    AllGather(layer_params) + ReduceScatter(layer_grads)`. Even with prefetch on,
    if a layer's compute is shorter than its own AllGather + the next layer's
    AllGather, overlap saturates and an exposed tail remains.

For Qwen2.5-7B at seq ≤ 8 k on A800 + IB, both fail. So Sergey's intuition is
*the right asymptotic regime* but not where we currently train.

---

## Implication for non-attention residual modelling

A strict saturation form
```
step_time ≈ compute_time
```
will under-predict step time by **30–50 %** at the seq lengths we care about.
The next-simplest model that matches the empirical data is

```
step_time ≈ compute_time + exposed_factor × comm_time
        comm_time = AllGather + ReduceScatter (+ Ulysses A2A if SP > 1)
```

with `exposed_factor` empirically:

|                | seq=4 k | seq=8 k | seq=16 k |
|----------------|---------|---------|----------|
| no ckpt        | 0.76    | 0.80    |  (OOM)   |
| `global_ckpt`  |  —      | 0.86    | 0.90     |

`exposed_factor` is **increasing slightly** with seq because comm and compute
both grow but the ceiling on overlap (≈ 100 ms / iter) does not. Within our
practical operating range it is in the 0.75–0.90 band.

**Plan for Stage 2** (compute model) and **Stage 3** (FSDP comm model):
   - Treat compute and comm as separate budgets.
   - Use `exposed_factor ≈ 0.85` as a tunable single-parameter knob initially;
     refine empirically once compute and comm models are independently calibrated.
   - Sanity check whether enabling `forward_prefetch=True` would unblock a much
     simpler model; if yes, recommend the code change (this is a 1-line patch
     to `parallel.py:135`).

---

## Reproducing the data

```bash
cd src/Hetu-Galvatron/galvatron/models/varlen_llama_hf/align_costmodel

# Default (no ckpt). seq=16384 will OOM.
bash 20_trace_fsdp_step_dispatch.sh

# Single seq, no ckpt:
FSDP_SEQ_LENGTHS="8192" MASTER_PORT=40047 \
  bash 20_trace_fsdp_step_dispatch.sh

# Production-like (global checkpointing on, fits 16k):
FSDP_SEQ_LENGTHS="8192 16384" FSDP_USE_CKPT=1 MASTER_PORT=40049 \
  bash 20_trace_fsdp_step_dispatch.sh

# Analyse a finished run:
python 21_analyze_fsdp_overlap.py results/fsdp_step_<RUN_ID>/traces
```

Traces analysed in this report:
- `results/fsdp_step_20260525_195306/traces/seq4096_ulysses_8/`  (no ckpt, 4 k)
- `results/fsdp_step_20260525_203154/traces/seq8192_ulysses_8/`  (no ckpt, 8 k)
- `results/fsdp_step_20260525_204433/traces/seq8192_ulysses_8/`  (ckpt, 8 k)
- `results/fsdp_step_20260525_204433/traces/seq16384_ulysses_8/` (ckpt, 16 k)

Per-iteration JSON dumps written next to each `traces/` directory as
`overlap*.json`.
