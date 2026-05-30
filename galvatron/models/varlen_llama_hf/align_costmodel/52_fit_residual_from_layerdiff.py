"""
Fit cost-model coefficients from the single-GPU galvatron layer-diff sweeps
(51_galv_layerdiff_sweep.sh outputs in /tmp/galv_layerdiff/summary_ckpt{0,1}.tsv).

Produces, per recompute mode:
  * per-layer fwd+bwd time (regression fb = per_layer*L + intercept over L)
  * per-layer LINEAR time (per_layer - flash_attn), i.e. the NON-attention
    residual `a` contribution per layer, per token
  * intercept per token (embed + LM-head + loss)
  * full-model (L=28) residual a per token = per_layer_linear*28 + intercept
  * activation memory per token (layer-diff of after_fwd - before)
  * recompute multiplier (recompute per-layer / no-recompute per-layer)

These feed the AdaCPSP cost model:
  total(group) = attn_LUT*(1+bwd_fwd) [already modeled]
               + a(recompute) * tokens_per_gpu        [residual linear]
               + b                                     [embed/head intercept + fixed]
"""
import os

# flash-attn fwd+bwd (ms) measured standalone (script 50, bare-HF, no recompute)
FLASH_ATTN_FB = {2048: 0.97, 4096: 2.83, 8192: 9.54, 16384: 35.3, 32768: 137.5}
N_LAYERS_REAL = 28
HIDDEN = 3584
VOCAB = 152064


def read_tsv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < len(header):
                continue
            d = dict(zip(header, parts))
            try:
                rows.append({
                    "layers": int(d["layers"]), "seq": int(d["seq"]),
                    "ckpt": int(d["ckpt"]),
                    "fb": float(d["fb_ms_median"]) if d["fb_ms_median"] != "NA" else None,
                    "before": float(d["before_mb"]) if d["before_mb"] != "NA" else None,
                    "after_fwd": float(d["after_fwd_mb"]) if d["after_fwd_mb"] != "NA" else None,
                    "after_bwd": float(d["after_bwd_mb"]) if d["after_bwd_mb"] != "NA" else None,
                })
            except Exception:
                pass
    return rows


def linregress(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    return slope, intercept


def analyze(rows, label):
    print(f"\n{'='*78}\n  {label}\n{'='*78}")
    seqs = sorted(set(r["seq"] for r in rows))
    print(f"{'seq':>7} {'per_layer_ms':>13} {'intercept_ms':>13} {'attn_ms':>9} "
          f"{'lin/tok_us':>11} {'icpt/tok_us':>12} {'a28/tok_us':>11}")
    per_layer_lin_by_seq = {}
    icpt_by_seq = {}
    for seq in seqs:
        pts = [(r["layers"], r["fb"]) for r in rows if r["seq"] == seq and r["fb"]]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        per_layer, intercept = linregress(xs, ys)
        attn = FLASH_ATTN_FB.get(seq, 0.0)
        per_layer_lin = per_layer - attn
        lin_per_tok = per_layer_lin / seq * 1000
        icpt_per_tok = intercept / seq * 1000
        a28 = (per_layer_lin * N_LAYERS_REAL + intercept) / seq * 1000
        per_layer_lin_by_seq[seq] = per_layer_lin
        icpt_by_seq[seq] = intercept
        print(f"{seq:>7} {per_layer:>13.2f} {intercept:>13.2f} {attn:>9.2f} "
              f"{lin_per_tok:>11.3f} {icpt_per_tok:>12.3f} {a28:>11.3f}")
    # memory: activation per token (after_fwd - before), layer-diff to per-layer
    print(f"\n  --- activation memory (MB) ---")
    print(f"{'seq':>7} {'per_layer_act':>14} {'act/tok_per_layer_KB':>22} {'a28_act/tok_KB':>16}")
    for seq in seqs:
        pts = [(r["layers"], r["after_fwd"] - r["before"]) for r in rows
               if r["seq"] == seq and r["after_fwd"] and r["before"]]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        per_layer_act, icpt_act = linregress(xs, ys)
        per_tok_kb = per_layer_act / seq * 1024
        a28_act_kb = (per_layer_act * N_LAYERS_REAL + icpt_act) / seq * 1024
        print(f"{seq:>7} {per_layer_act:>14.1f} {per_tok_kb:>22.3f} {a28_act_kb:>16.3f}")
    return per_layer_lin_by_seq, icpt_by_seq


def main():
    r0 = read_tsv("/tmp/galv_layerdiff/summary_ckpt0.tsv")
    r1 = read_tsv("/tmp/galv_layerdiff/summary_ckpt1.tsv")
    pl0, ic0 = analyze(r0, "NO RECOMPUTE (ckpt=0)")
    pl1, ic1 = analyze(r1, "FULL RECOMPUTE (ckpt=1)")

    # recompute multiplier on per-layer linear
    print(f"\n{'='*78}\n  RECOMPUTE MULTIPLIER (per-layer, recompute/no-recompute)\n{'='*78}")
    common = sorted(set(pl0) & set(pl1))
    for seq in common:
        # use full per-layer (incl attn) ratio too
        mult = pl1[seq] / pl0[seq] if pl0[seq] else float("nan")
        print(f"  seq={seq:>6}: no_recomp_lin={pl0[seq]:.1f}ms  recomp_lin={pl1[seq]:.1f}ms  mult={mult:.3f}")

    # Recommended a (ms/token) for L=28, averaged over mid seqs
    print(f"\n{'='*78}\n  RECOMMENDED COST-MODEL COEFFICIENTS (Qwen2.5-7B, L=28)\n{'='*78}")
    for label, pl, ic in [("no_recompute", pl0, ic0), ("recompute", pl1, ic1)]:
        seqs = sorted(pl)
        if not seqs:
            continue
        # use the largest measured seq (closest to production regime) + mid
        a_vals = [(pl[s] * N_LAYERS_REAL + ic[s]) / s * 1000 for s in seqs]  # us/token
        a_ms = [v / 1000 for v in a_vals]
        print(f"  {label:>14}: a(L=28) per seq (ms/token) = "
              + ", ".join(f"{s}:{v:.3f}" for s, v in zip(seqs, a_ms))
              + f"  | mean={sum(a_ms)/len(a_ms):.3f}")


if __name__ == "__main__":
    main()
