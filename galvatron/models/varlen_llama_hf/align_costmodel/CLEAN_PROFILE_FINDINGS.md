# Clean single-GPU layer-diff profiling — validated cost-model coefficients

Date: 2026-05-31. Model: Qwen2.5-7B (hidden=3584, heads=28, kv=4, ffn=18944, vocab=152064, L=28).

## TL;DR — corrected mental model

1. **The compute model is accurate.** galvatron single-GPU fwd+bwd == bare-HF ==
   multi-GPU (1/2/4/8 GPU) ≈ 380 ms for L=4/seq=8192/sp=1. ZeRO-2 comm is **well
   hidden** (overlapped) up to 8 GPU on NVLink — there is NO mysterious exposed
   communication intra-node.
2. **The earlier "8-GPU = 2700ms (7×)" was a measurement artifact** — local mnist
   GPU-filler job was still dying / contending when that run started. After mnist
   fully exited, 8-GPU == 380ms. LESSON: always confirm GPU fully free before timing.
3. **The linear residual `a` is sp-INDEPENDENT** (MLP/proj/embed/LM-head are GEMMs on
   tokens_per_gpu, independent of sp). The old per-sp `a` table
   {1:0.197,2:0.184,4:0.250,8:0.116} was noise/overfit absorbing other errors.
4. **The old cauto 2.5× under-prediction** is explained by config mismatches, NOT a
   mysterious factor: cauto ran with `--global_checkpoint 1` (FULL recompute, ~1.3×
   per-layer) + cross-NODE comm (16-GPU 2 nodes, RoCE/IB ≠ intra-node NVLink) +
   ulysses sp=16 head-padding (4×), while the cost model `a` was no-recompute-ish.

## Method

`51_galv_layerdiff_sweep.sh`: single-GPU galvatron `train_dist_adacpsp.py`, forced
`ulysses:1` (sp=1 → no attention comm, no straggler), `--dataset fix_length` (clean
fixed-length sequences), `--set_layernum_manually 1` to control L. Layer-diff over
L∈{1,2,4} isolates per-layer; intercept = embed+LM-head+loss. fwd+bwd timed by a
minimal CUDA-event probe `[FBPROF]` added around `model.forward_backward`. Flash-attn
fwd+bwd measured standalone (`50_profile_layer_standalone.py`) for subtraction.

## TIME coefficients (fwd+bwd, ms)

per-layer (regression fb = per_layer·L + intercept):

| seq | no-recompute per-layer | recompute per-layer | flash-attn | no-rc per-layer LINEAR | lin/token |
|-----|------|------|------|------|------|
| 4096 | 31.4 | — | 2.83 | 28.6 | 6.97 us |
| 8192 | 60.6 | 79.2 | 9.54 | 51.1 | 6.24 us |
| 16384| 132.5| 172.8| 35.3 | 97.2 | 5.93 us |

intercept (embed+LM-head+loss): ~15–18 us/token (linear in tokens, decreasing w/ seq).

**Recompute multiplier (per-layer): ~1.30–1.41×** (mean ≈ 1.35).

### Residual `a` for full model (L=28), ms/token
- **no-recompute: a ≈ 0.195** (0.213@4k, 0.190@8k, 0.181@16k — mildly ↓ with seq)
- **recompute:    a ≈ 0.252** (0.254@8k, 0.250@16k)

(matches old `a=0.197` for the no-recompute case → the linear model was ~right; the
cauto gap was recompute + cross-node comm + head-padding, not `a`.)

## MEMORY coefficients (activation = after_fwd − before)

per-layer activation ≈ **163 KB/token/layer** (no-recompute), constant across seq.

### act_per_token for full model (L=28), MB/token
- **no-recompute: ≈ 5.3 MB/token** (per-layer 0.16 MB/tok ×28 + embed/head/logits intercept)
- **recompute:    ≈ 0.87 MB/token** (recompute frees layer activations → per-layer only 7 KB/tok)

(old cost model used a single `act_per_token=3.96`, between the two — should be
recompute-aware: 5.3 vs 0.87.)

model_states (before-forward, single GPU ZeRO-2): grows ~890 MB/layer (params+grad+
lazily-allocated optimizer); embed+head baseline ≈ 4–5 GB.

## Recommended cost-model changes

1. **Replace per-sp residual `a` with a single sp-independent constant**, recompute-aware:
   `a = 0.195` (no-recompute) or `0.252` (recompute) ms/token.
2. **Fold embed/LM-head into the linear term** (they ARE linear in tokens) — already
   captured by the L=28 `a` above; the fixed `b` (per-layer launch) is small (~tens ms).
3. **Memory**: `act_per_token = 5.3` (no-recompute) / `0.87` (recompute) MB/token;
   `/ parallel_size`; pass correct `zero_stage`.
4. **Recompute factor 1.35×** can derive recompute-`a` from no-recompute-`a` if only
   one is profiled.

## Raw data
`/tmp/galv_layerdiff/summary_ckpt0.tsv`, `summary_ckpt1.tsv`;
`results/layerdiff_norecompute_20260531_015139.json` (bare-HF cross-check).
