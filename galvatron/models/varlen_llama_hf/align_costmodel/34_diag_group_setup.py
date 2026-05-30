"""Diagnose per-group setup overhead.

Hypothesis: solver-chosen configs (adacpsp_cauto) create many small groups per
microbatch. Each group needs NCCL subgroup init + buffer alloc + per-iter
group sync, which adds a fixed overhead per group not captured by current
cost model. If true, residual (measured - predicted) should scale linearly
with sum_over_mbs(n_groups_in_mb).

For each iter:
  - Re-predict end-to-end ms with the fixed cost model (v2 + latency-floor +
    b_step_fb interpolation).
  - Read measured forward_backward.
  - Count n_groups_total = sum over microbatches of len(groups).
  - residual = measured - predicted.

Then fit residual = c_setup_ms · n_groups_total + intercept across iters of
all cells. If c_setup_ms > 0 with significant R^2, the hypothesis is
supported; the fitted slope becomes the recommended overhead constant.

Usage:
  python 34_diag_group_setup.py <results_dir> [--skip-warmup 2]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from typing import List, Tuple

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "galvatron/site_package"))

from galvatron.models.varlen_llama_hf.adacpsp_solver import (
    AdaCPSPCostModel, ParallelStrategy,
)


def latest(pattern, d):
    paths = sorted(glob.glob(os.path.join(d, pattern)), reverse=True)
    for p in paths:
        try:
            with open(p) as f:
                return p, json.load(f)
        except Exception:
            pass
    return None, None


def build_cm():
    cfg = os.path.join(REPO, "galvatron/models/varlen_llama_hf/configs")
    attn_p, _ = latest("profile_validate_*.json", cfg)
    comm_p, _ = latest("comm_profile_*.json", cfg)
    _, resid = latest("residual_profile_*.json", cfg)
    _, bdec = latest("b_decomp_profile_*.json", cfg)
    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn_p, comm_profile_json=comm_p,
        cluster_size=16, validation_json=attn_p, gpus_per_node=8,
    )
    if resid:
        cm.apply_residual_profile(resid)
    if bdec:
        cm.apply_b_decomp_profile(bdec)
    print(f"# loaded comm={os.path.basename(comm_p)} bdec={'y' if bdec else 'n'}",
          file=sys.stderr)
    return cm


def predict_group(cm, g):
    s = int(g.get("tokens", 0))
    if s <= 0:
        return 0.0
    strat = ParallelStrategy(
        attn_type=g["attn_type"],
        parallel_size=int(g.get("parallel_size",
                                g["sp_size"] * g["cp_size"])),
        sp_size=int(g["sp_size"]),
        cp_size=int(g["cp_size"]),
        placement=g.get("placement", "context_first"),
    )
    return cm.total_time([s], strat)


def linfit(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Return (slope, intercept, r_squared) of OLS y = a x + b."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, my, 0.0
    slope = sxy / sxx
    intercept = my - slope * mx
    syy = sum((y - my) ** 2 for y in ys)
    sse = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - sse / syy if syy > 0 else 0.0
    return slope, intercept, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--skip-warmup", type=int, default=2)
    args = ap.parse_args()

    cm = build_cm()
    e2e = args.results_dir
    if os.path.basename(e2e.rstrip("/")) != "end2end":
        cand = os.path.join(e2e, "end2end")
        if os.path.isdir(cand):
            e2e = cand

    print()
    hdr = (f"{'cell':<22} {'iter':>4} {'tokens':>7} {'n_mb':>4} "
           f"{'n_grp':>5} {'mean_g_per_mb':>14} "
           f"{'meas':>7} {'pred':>7} {'resid':>7} {'resid/grp':>10}")
    print(hdr); print("-" * len(hdr))

    rows_all_groups = []  # for the regression: (n_groups_total, residual)
    rows_extra_groups = []  # excluding 2-group "minimum" baseline

    per_cell_summary = {}

    for d in sorted(glob.glob(os.path.join(e2e, "*_chunks*"))):
        label = os.path.basename(d.rstrip("/"))
        jp = os.path.join(d, "rank0.jsonl")
        if not os.path.isfile(jp):
            continue
        with open(jp) as f:
            recs = [json.loads(l) for l in f if l.strip()]
        train_recs = [r for r in recs if r.get("phase") == "train_step"]
        cell_xs, cell_ys = [], []
        for i, r in enumerate(train_recs):
            if i < args.skip_warmup:
                continue
            fb = float(r.get("timings_ms", {}).get("forward_backward", 0.0))
            gb = r.get("global_batch") or {}
            mbs = gb.get("microbatches") or []
            n_mb = len(mbs)
            n_grp = sum(len(mb.get("groups", [])) for mb in mbs)
            tokens = int(gb.get("global_tokens", 0))
            # Re-predict offline (matches live train-time _predict_adacpsp_ms).
            new_total = 0.0
            sp_values = []
            for mb in mbs:
                mb_max = 0.0
                for g in mb.get("groups", []):
                    sp = int(g.get("sp_size", 1))
                    if sp not in sp_values:
                        sp_values.append(sp)
                    mb_max = max(mb_max, predict_group(cm, g))
                new_total += mb_max
            if len(mbs) >= 2 and hasattr(cm, "b_step_fb_ms_for_strategies"):
                new_total += float(cm.b_step_fb_ms_for_strategies(sp_values))
            resid = fb - new_total
            mean_grp = n_grp / max(1, n_mb)
            per_grp = resid / max(1, n_grp)
            cell_xs.append(n_grp); cell_ys.append(resid)
            rows_all_groups.append((n_grp, resid))
            # "Extra" groups above the 2-per-mb baseline (forced strategies)
            extra = n_grp - 2 * n_mb
            if extra >= 0:
                rows_extra_groups.append((extra, resid))
            print(f"{label:<22} {i:>4} {tokens:>7} {n_mb:>4} {n_grp:>5} "
                  f"{mean_grp:>14.2f} "
                  f"{fb:>7.0f} {new_total:>7.0f} {resid:>+7.0f} "
                  f"{per_grp:>+10.1f}")
        # per-cell fit
        if cell_xs:
            sl, ic, r2 = linfit(cell_xs, cell_ys)
            per_cell_summary[label] = (len(cell_xs), sum(cell_xs)/len(cell_xs),
                                       sum(cell_ys)/len(cell_ys),
                                       sl, ic, r2)
        print()

    print("=" * 88)
    print("Per-cell aggregate")
    h = (f"{'cell':<22} {'n':>3} {'mean_n_grp':>10} "
         f"{'mean_resid':>10} {'fit_slope':>10} {'fit_int':>9} {'R^2':>6}")
    print(h); print("-" * len(h))
    for label, (n, mng, mr, sl, ic, r2) in per_cell_summary.items():
        print(f"{label:<22} {n:>3} {mng:>10.2f} {mr:>+10.0f} {sl:>+10.1f} "
              f"{ic:>+9.0f} {r2:>6.3f}")

    # Global fits
    print()
    print("=" * 88)
    print("Global fits across all iters:")
    if rows_all_groups:
        xs = [x for x, _ in rows_all_groups]
        ys = [y for _, y in rows_all_groups]
        sl, ic, r2 = linfit(xs, ys)
        print(f"  residual = {sl:+.1f} · n_groups + {ic:+.0f}   "
              f"R^2={r2:.3f}   n_iters={len(xs)}")
    if rows_extra_groups:
        xs = [x for x, _ in rows_extra_groups]
        ys = [y for _, y in rows_extra_groups]
        sl, ic, r2 = linfit(xs, ys)
        print(f"  residual = {sl:+.1f} · extra_groups + {ic:+.0f}   "
              f"R^2={r2:.3f}   n_iters={len(xs)}")
        print(f"    (extra_groups = n_groups − 2·n_microbatches, i.e. the")
        print(f"     groups beyond the minimum 2-parallel-groups baseline)")


if __name__ == "__main__":
    sys.exit(main() or 0)
