"""Quick (no-torch) b-decomposition fit.

Reads `predicted_adacpsp.total_ms` from the JSONL (the attention prediction
recorded by the trainer at the time of the run) instead of re-importing
AdaCPSPCostModel. This skips a ~2-minute torch import on heavily-loaded
machines, but uses the *stored* attention prediction (which was computed by
the cost model in effect during the sweep -- typically the residual profile
was already loaded, so the stored value includes residual; we strip residual
back out using `--strip-residual <profile.json>`).

Outputs the same b-decomposition JSON the cost model can load, plus a
per-cell table for inspection.
"""

from __future__ import annotations
import argparse, collections, glob, json, math, os, re, sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

CELL_RE = re.compile(r"sp(\d+)_seq(\d+)_chunks(\d+)")


def parse_cell(d: str):
    m = CELL_RE.match(os.path.basename(d.rstrip("/")))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def pct(xs, p):
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs2[int(k)]
    return xs2[f] + (xs2[c] - xs2[f]) * (k - f)


def linreg(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0, float("nan")
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n, float("nan")
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    ym = sy / n
    ss_tot = sum((y - ym) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


@dataclass
class Cell:
    sp: int
    seq: int
    chunks: int
    rank: int
    n_iters: int
    fb_clean: float        # mean of 3 smallest fb_ms excluding iter 0
    fb_steady: float       # mean of [p10, median]
    fb_median: float
    fb_p10: float
    opt_med: float
    grad_med: float
    zero_med: float
    solve_med: float
    pred_attn_stored_ms: float
    pred_attn_pure_ms: float  # = stored - residual(sp)*N if residual was loaded
    tokens_per_gpu: float
    group_tokens: int
    sp_size_seen: int = 0
    attn_type: str = "ulysses"


def measure(p, strip_residual=None):
    recs = load_jsonl(p)
    tr = [r for r in recs if r.get("phase") == "train_step"]
    if not tr:
        return None
    fb_all = [r["timings_ms"]["forward_backward"] for r in tr
              if "forward_backward" in r.get("timings_ms", {})]
    if not fb_all:
        return None
    body = fb_all[1:] if len(fb_all) > 1 else fb_all
    s = sorted(body)
    fb_clean = sum(s[:3]) / 3 if len(s) >= 3 else sum(s) / max(1, len(s))
    p10 = pct(body, 0.1); med = pct(body, 0.5)
    lh = [v for v in body if p10 <= v <= med]
    fb_steady = (sum(lh) / len(lh)) if lh else med

    last = tr[-1]
    pred = last.get("predicted_adacpsp") or {}
    stored_total = float(pred.get("total_ms", 0.0))
    mbs = pred.get("microbatches", []) or []
    n_mbs = max(1, len(mbs))
    g0 = (mbs[0]["groups"][0] if mbs and mbs[0].get("groups") else {})
    sp_size = int(g0.get("sp_size", 0))
    group_tokens = int(g0.get("tokens", 0))
    tokens_per_gpu = (group_tokens / sp_size) if sp_size > 0 else 0.0
    attn_type = str(g0.get("attn_type", "ulysses"))

    pure_attn = stored_total
    if strip_residual is not None:
        per_sp = strip_residual.get("residual_per_sp", {})
        ent = per_sp.get(str(sp_size)) or {}
        a_per_token = float(ent.get("a_per_token",
                                    strip_residual.get("residual_a_default_per_token", 0.0)))
        b_ms = float(ent.get("b_ms",
                             strip_residual.get("residual_b_default_ms", 0.0)))
        residual_per_group = a_per_token * tokens_per_gpu + b_ms
        n_groups_per_mb = 1  # forced-strategy: one group per microbatch (per node-pair)
        residual_per_mb = residual_per_group * n_groups_per_mb
        pure_attn = stored_total - residual_per_mb * n_mbs

    return Cell(
        sp=0, seq=0, chunks=0,
        rank=int(last.get("rank", -1)),
        n_iters=len(fb_all),
        fb_clean=fb_clean,
        fb_steady=fb_steady,
        fb_median=med,
        fb_p10=p10,
        opt_med=pct([r["timings_ms"].get("optimizer_step", 0) for r in tr], 0.5),
        grad_med=pct([r["timings_ms"].get("grad_clip", 0) for r in tr], 0.5),
        zero_med=pct([r["timings_ms"].get("zero_grad", 0) for r in tr], 0.5),
        solve_med=pct([r["timings_ms"].get("solve_and_dispatch", 0) for r in tr], 0.5),
        pred_attn_stored_ms=stored_total,
        pred_attn_pure_ms=pure_attn,
        tokens_per_gpu=tokens_per_gpu,
        group_tokens=group_tokens,
        sp_size_seen=sp_size,
        attn_type=attn_type,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--ranks", default="0")
    ap.add_argument("--residual-profile", default=None,
                    help="Existing residual_profile JSON to strip from stored predictions.")
    ap.add_argument("--b-decomp-out", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--skip-ch1-fit", action="store_true",
                    help="Also report a fit that drops chunks=1 (often a fast-path outlier).")
    args = ap.parse_args()

    allowed = set(int(x) for x in re.split(r"[,\s]+", args.ranks.strip()) if x)

    e2e = args.results_dir
    if os.path.basename(e2e.rstrip("/")) != "end2end":
        cand = os.path.join(e2e, "end2end")
        if os.path.isdir(cand):
            e2e = cand
    if not os.path.isdir(e2e):
        print(f"missing dir: {e2e}", file=sys.stderr); return 1

    strip = None
    if args.residual_profile:
        with open(args.residual_profile) as f:
            strip = json.load(f)
        print(f"[residual strip] {os.path.basename(args.residual_profile)}")
    else:
        # default to newest residual_profile_live_*.json
        configs = os.path.join(os.path.dirname(__file__), "..", "configs")
        cands = sorted(glob.glob(os.path.join(configs, "residual_profile_live_*.json")))
        if cands:
            with open(cands[-1]) as f:
                strip = json.load(f)
            print(f"[residual strip] {os.path.basename(cands[-1])} (auto)")

    cells = []
    by_cfg = collections.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(e2e, "sp*_seq*_chunks*"))):
        parsed = parse_cell(d)
        if parsed is None:
            continue
        sp, seq, ch = parsed
        for rank in sorted(allowed):
            p = os.path.join(d, f"rank{rank}.jsonl")
            if not os.path.isfile(p):
                continue
            m = measure(p, strip_residual=strip)
            if m is None:
                continue
            m.sp = sp; m.seq = seq; m.chunks = ch; m.rank = rank
            cells.append(m)
            by_cfg[(sp, seq)].append(m)

    if not cells:
        print("no cells", file=sys.stderr); return 1

    print(f"\n=== Per-cell measurements ===")
    print(f"  fb_clean  = mean of 3 lowest fb excl iter 0")
    print(f"  fb_steady = mean of [p10, median] window")
    hdr = (f"{'sp':>3} {'seq':>5} {'N':>2} {'rk':>3} {'iter':>4} "
           f"{'fb_clean':>9} {'fb_steady':>10} {'fb_med':>8} {'fb_p10':>8} "
           f"{'attn_st':>8} {'attn_pure':>9} {'opt':>5} {'gclip':>5} {'zgrad':>5} {'solve':>5}")
    print(hdr); print("-" * len(hdr))
    for c in sorted(cells, key=lambda x: (x.sp, x.seq, x.chunks)):
        print(f"{c.sp:>3} {c.seq:>5} {c.chunks:>2} {c.rank:>3} {c.n_iters:>4} "
              f"{c.fb_clean:>9.1f} {c.fb_steady:>10.1f} {c.fb_median:>8.1f} {c.fb_p10:>8.1f} "
              f"{c.pred_attn_stored_ms:>8.1f} {c.pred_attn_pure_ms:>9.1f} "
              f"{c.opt_med:>5.1f} {c.grad_med:>5.1f} {c.zero_med:>5.2f} {c.solve_med:>5.2f}")

    # Fits per (sp, seq), with two windows: all chunks and chunks>=2.
    print(f"\n=== Linear fits fb(N) = slope·N + intercept ===")
    print(f"{'sp':>3} {'seq':>5} {'metric':>7} {'cells':>11} {'n':>2} {'slope':>9} "
          f"{'intercept':>10} {'R²':>6} {'attn_pure(N=1)':>14} "
          f"{'per_mb_resid':>13}")

    fits = {}
    for (sp, seq), cs in sorted(by_cfg.items()):
        cr0 = sorted([c for c in cs if c.rank == 0], key=lambda x: x.chunks)
        if len(cr0) < 2:
            print(f"{sp:>3} {seq:>5}  insufficient cells={len(cr0)}"); continue
        attn_n1 = cr0[0].pred_attn_pure_ms
        for windows in [
            ("all", cr0),
            ("ch>=2", [c for c in cr0 if c.chunks >= 2]),
        ]:
            label, cs2 = windows
            if len(cs2) < 2:
                continue
            xs = [float(c.chunks) for c in cs2]
            for metric, ys_get in [("clean", lambda c: c.fb_clean),
                                    ("steady", lambda c: c.fb_steady)]:
                ys = [ys_get(c) for c in cs2]
                slope, intercept, r2 = linreg(xs, ys)
                per_mb_resid = slope - attn_n1
                print(f"{sp:>3} {seq:>5} {metric:>7} {label:>11} {len(cs2):>2} {slope:>9.1f} "
                      f"{intercept:>10.1f} {r2:>6.4f} {attn_n1:>14.1f} {per_mb_resid:>13.1f}")
                fits[(sp, seq, label, metric)] = {
                    "slope_per_mb_ms": slope,
                    "intercept_fb_ms": intercept,
                    "r2": r2,
                    "n_points": len(cs2),
                    "per_mb_residual_ms": per_mb_resid,
                    "attn_pure_n1_ms": attn_n1,
                    "tokens_per_gpu": cr0[0].tokens_per_gpu,
                }

    # External overhead
    print(f"\n=== Per-step external (non-FB) overhead medians ===")
    rank0 = [c for c in cells if c.rank == 0]
    def mm(k):
        return pct([getattr(c, k) for c in rank0], 0.5) if rank0 else 0.0
    opt_m = mm("opt_med"); gr_m = mm("grad_med"); zg_m = mm("zero_med"); sv_m = mm("solve_med")
    ext_tot = opt_m + gr_m + zg_m + sv_m
    print(f"  optimizer_step    median = {opt_m:8.2f} ms")
    print(f"  grad_clip         median = {gr_m:8.2f} ms")
    print(f"  zero_grad         median = {zg_m:8.2f} ms")
    print(f"  solve_and_dispatch median = {sv_m:8.2f} ms")
    print(f"  ── total per-step external ── = {ext_tot:8.2f} ms")

    if args.b_decomp_out:
        # Default: use "ch>=2 + clean" fit for production multi-mb scenarios.
        # Each (sp, seq) gives one (slope=per_mb_total, intercept=b_step_fb_sp) pair.
        per_sp_b_mb = {}
        per_sp_b_step_clean = {}
        per_sp_b_step_steady = {}
        for (sp, seq, win, metric), f in fits.items():
            if win != "ch>=2":
                continue
            if metric == "clean":
                per_sp_b_step_clean[sp] = f["intercept_fb_ms"]
            else:
                per_sp_b_step_steady[sp] = f["intercept_fb_ms"]

        # Refit `a_per_token` assuming b_microbatch ≈ 0 at our single calibration
        # point (tokens_per_gpu = seq_len). The old residual profile's `a` was
        # measured without forward_prefetch=True, which exposed FSDP all-gather
        # at higher sp; re-using it would yield negative b_microbatch. Re-fitting
        # at fixed tokens means we lose the b_mb breakdown but reproduce the
        # measured per-mb cost exactly at our calibration point. Note that the
        # per-mb cost still scales linearly with tokens — extrapolation to
        # higher seq lengths assumes b_mb = 0.
        for (sp, seq, win, metric), f in fits.items():
            if win != "ch>=2" or metric != "clean":
                continue
            tok = f["tokens_per_gpu"]
            per_mb_resid = f["per_mb_residual_ms"]
            a_refit = per_mb_resid / max(1.0, tok)
            per_sp_b_mb[str(sp)] = {
                "a_per_token": a_refit,
                "b_microbatch_ms": 0.0,
                "calibrated_at_seq": seq,
                "calibrated_at_tokens_per_gpu": tok,
                "per_mb_residual_total_ms": per_mb_resid,
            }

        # Default fallbacks: median across sps.
        b_step_fb_clean_default = pct(list(per_sp_b_step_clean.values()), 0.5) \
            if per_sp_b_step_clean else 0.0
        b_step_fb_steady_default = pct(list(per_sp_b_step_steady.values()), 0.5) \
            if per_sp_b_step_steady else 0.0
        cfg = {
            "schema": "adacpsp_b_decomp_v1",
            "source_run": os.path.basename(args.results_dir.rstrip("/")),
            "stripped_residual": os.path.basename(args.residual_profile)
                                  if args.residual_profile else "auto",
            "fit_window": "chunks>=2 (chunks=1 outlier with FSDP fast-path)",
            "residual_per_sp": per_sp_b_mb,
            "b_step_fb_per_sp_clean": {str(k): v for k, v in per_sp_b_step_clean.items()},
            "b_step_fb_per_sp_steady": {str(k): v for k, v in per_sp_b_step_steady.items()},
            "b_step_fb_ms_clean": b_step_fb_clean_default,
            "b_step_fb_ms_steady": b_step_fb_steady_default,
            "b_step_external_ms": ext_tot,
        }
        with open(args.b_decomp_out, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"\nWrote b-decomp config -> {args.b_decomp_out}")

    if args.json_out:
        out = {
            "schema": "adacpsp_b_decomp_quick_v1",
            "source_run": os.path.basename(args.results_dir.rstrip("/")),
            "cells": [vars(c) for c in cells],
            "fits": {f"sp{k[0]}_seq{k[1]}_{k[2]}_{k[3]}": v for k, v in fits.items()},
            "per_step_external_ms": {
                "optimizer_step_median": opt_m,
                "grad_clip_median": gr_m,
                "zero_grad_median": zg_m,
                "solve_and_dispatch_median": sv_m,
                "total_median": ext_tot,
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote analysis JSON -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
