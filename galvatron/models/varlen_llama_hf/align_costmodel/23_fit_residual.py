"""Fit the non-attention residual cost model from 22_bench_residual outputs.

Model:
    T_per_group_total(seqlens, strat) = T_attention(seqlens, strat)
                                       + a * tokens_per_GPU + b(sp)

where T_attention is the existing AdaCPSPCostModel.total_time prediction. The
residual we fit here is the non-attention component: MLP, layernorm,
projections, embedding, LM head, plus any FSDP comm not hidden by compute.

The default mode (`--attn-source live`) recomputes T_attention using the
CURRENT cost-model code, instead of trusting the value stored in the JSONL
(which was recorded at run-time and may drift if the cost-model code is
edited between profiling and fitting). This guarantees
    residual_a*x + residual_b + attn_pred_now ≈ measured
holds at production time, even when adacpsp_solver.py has been touched since
the profile sweep ran.

Workflow:
  1. Walk results/<RUN_ID>/end2end/sp<sp>_seq<seq>_ckpt<ckpt>/rank<R>.jsonl
  2. For each cell, parse records with phase=="train_step", extract steady-
     state forward_backward_ms, group layout (seqlens, sp, parallel_size).
  3. Compute `predicted_attention_ms` either from JSONL (`stored`) or by
     calling the current AdaCPSPCostModel.total_time (`live`, default).
  4. residual_ms = measured_fb_ms - predicted_attention_ms
  5. Group by (sp, ckpt); fit `residual = a*x + b` via least squares.
  6. Report per-sp (a, b, R^2, max abs error) + cross-sp consistency check.

Usage:
  python 23_fit_residual.py <results_dir> [--attn-source live|stored]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


# --------------------------- IO --------------------------- #

CELL_RE = re.compile(r"sp(\d+)_seq(\d+)_ckpt(\d+)")


def parse_cell(cell_dir: str) -> Tuple[int, int, int] | None:
    name = os.path.basename(cell_dir.rstrip("/"))
    m = CELL_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_jsonl(path: str) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# --------------------------- per-cell parsing --------------------------- #

@dataclass
class CellMeasurement:
    sp: int
    seq_len: int
    ckpt: int
    rank: int
    n_iters: int
    fb_ms_median: float
    fb_ms_p10: float
    fb_ms_p25: float
    fb_ms_p90: float
    # Steady-state estimator: mean of iters in [p10, median]. Robust to slow-
    # iter outliers (warm-up cache, occasional NCCL stalls) while not biasing
    # toward unrealistically-fast minimum.
    fb_ms_steady: float
    predicted_attention_ms: float       # from JSONL (run-time value)
    predicted_attention_ms_live: float  # recomputed with current cost-model
    measured_opt_ms_median: float
    group_tokens: int        # group_tokens (= sp * seq_len for our packing)
    tokens_per_gpu: float    # group_tokens / sp_size
    group_seqlens: list      # first group's seqlens, for live recomputation
    attn_type: str           # first group's attn_type
    placement: str           # first group's placement
    strategy_label: str


def percentile(xs: list, p: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs2[int(k)]
    return xs2[f] + (xs2[c] - xs2[f]) * (k - f)


def measure_cell(jsonl_path: str) -> CellMeasurement | None:
    records = load_jsonl(jsonl_path)
    # Only consider real train_step records (skip warmups / non-trained iters
    # where forward_backward was not actually run).
    train_recs = [r for r in records if r.get("phase") == "train_step"]
    if not train_recs:
        return None

    fb = [r["timings_ms"]["forward_backward"] for r in train_recs
          if "timings_ms" in r and "forward_backward" in r["timings_ms"]]
    if not fb:
        return None
    opt = [r["timings_ms"].get("optimizer_step", 0.0) for r in train_recs]
    fb_p10 = percentile(fb, 0.1)
    fb_p25 = percentile(fb, 0.25)
    fb_med = percentile(fb, 0.5)
    fb_p90 = percentile(fb, 0.9)
    # Steady-state = mean of iters between p10 and median (the stable lower
    # half), which excludes slow-iter outliers but stays representative.
    lower_half = [v for v in fb if fb_p10 <= v <= fb_med]
    fb_steady = (sum(lower_half) / len(lower_half)) if lower_half else fb_med

    # All records in a cell share the same strategy / sp.
    last = train_recs[-1]
    pred = last.get("predicted_adacpsp") or {}
    pred_attn_ms = float(pred.get("total_ms", 0.0))

    # Figure out group_tokens + sp_size from the first microbatch's first group.
    mbs = pred.get("microbatches", [])
    g0_list = (mbs[0] or {}).get("groups", []) if mbs else []
    if g0_list:
        g0 = g0_list[0]
        sp_size = int(g0["sp_size"])
        group_tokens = int(g0["tokens"])
        attn_type = str(g0.get("attn_type", "ulysses"))
        placement = str(g0.get("placement", "head_first"))
    else:
        sp_size = 0
        group_tokens = 0
        g0 = {}
        attn_type = "ulysses"
        placement = "head_first"
    # Reconstruct the group's seqlens: in uniform-packing benchmarks each group
    # has `sp_size` sequences of length `group_tokens/sp_size`. The exact list
    # rebuilt here is needed for live recomputation of the attention cost.
    if sp_size > 0 and group_tokens > 0:
        n_seqs = max(1, int(g0.get("num_sequences", sp_size)))
        per_seq = group_tokens // max(1, n_seqs)
        group_seqlens = [per_seq] * n_seqs
        # Reconcile: ensure total tokens match (handle rounding from non-uniform packing).
        deficit = group_tokens - sum(group_seqlens)
        if deficit and group_seqlens:
            group_seqlens[-1] += deficit
    else:
        group_seqlens = []

    rank = int(last.get("rank", -1))
    strategy_label = str(last.get("strategy_label") or "")

    return CellMeasurement(
        sp=sp_size,
        seq_len=0,  # filled in by caller from dir name
        ckpt=0,
        rank=rank,
        n_iters=len(fb),
        fb_ms_median=fb_med,
        fb_ms_p10=fb_p10,
        fb_ms_p25=fb_p25,
        fb_ms_p90=fb_p90,
        fb_ms_steady=fb_steady,
        predicted_attention_ms=pred_attn_ms,
        predicted_attention_ms_live=pred_attn_ms,  # overwritten if --attn-source=live
        measured_opt_ms_median=percentile(opt, 0.5),
        group_tokens=group_tokens,
        tokens_per_gpu=(group_tokens / sp_size) if sp_size > 0 else 0.0,
        group_seqlens=group_seqlens,
        attn_type=attn_type,
        placement=placement,
        strategy_label=strategy_label,
    )


# --------------------------- linear fit --------------------------- #

def linreg(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Return (a, b, R^2) for y = a*x + b."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0, float("nan")
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n, float("nan")
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    ymean = sy / n
    ss_tot = sum((y - ymean) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


# --------------------------- main --------------------------- #

def fmt_ms(v: float) -> str:
    return f"{v:7.2f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", help="Either run-dir containing end2end/, or end2end/ itself.")
    p.add_argument("--ranks", default="0",
                   help="Space- or comma-separated list of ranks to include (default: 0)")
    p.add_argument("--attn-source", choices=["live", "stored"], default="live",
                   help="`live` (default) recomputes T_attention using the current "
                        "AdaCPSPCostModel; `stored` trusts the predicted_adacpsp.total_ms "
                        "value the worker wrote into the JSONL.")
    p.add_argument("--cluster-size", type=int, default=16)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--json-out", default=None)
    p.add_argument("--residual-config-out", default=None,
                   help="Path to write a compact loader-friendly residual JSON "
                        "(consumable by train_dist_adacpsp.py).")
    p.add_argument("--median-p10-ratio-max", type=float, default=1.05,
                   help="Drop cells where fb_median/fb_p10 exceeds this ratio "
                        "(default 1.05) — these are memory-pressure / unstable cells.")
    p.add_argument("--metric", choices=["steady", "median", "p25", "p10"], default="steady",
                   help="Per-cell time estimator (default: steady = mean of [p10, median]).")
    args = p.parse_args()

    allowed_ranks = set(int(x) for x in re.split(r"[,\s]+", args.ranks.strip()) if x)

    end2end_dir = args.results_dir
    if not os.path.basename(end2end_dir.rstrip("/")) == "end2end":
        cand = os.path.join(end2end_dir, "end2end")
        if os.path.isdir(cand):
            end2end_dir = cand
    if not os.path.isdir(end2end_dir):
        print(f"No such directory: {end2end_dir}", file=sys.stderr)
        return 1

    cell_dirs = sorted(d for d in glob.glob(os.path.join(end2end_dir, "sp*_seq*_ckpt*"))
                       if os.path.isdir(d))
    if not cell_dirs:
        print(f"No sp*_seq*_ckpt* cells under {end2end_dir}", file=sys.stderr)
        return 1

    # Collect measurements.
    by_cfg = collections.defaultdict(list)   # (sp, ckpt) -> [CellMeasurement, ...]
    all_measurements = []
    for cell_dir in cell_dirs:
        parsed = parse_cell(cell_dir)
        if parsed is None:
            continue
        sp, seq_len, ckpt = parsed
        for rank in sorted(allowed_ranks):
            jsonl = os.path.join(cell_dir, f"rank{rank}.jsonl")
            if not os.path.isfile(jsonl):
                continue
            m = measure_cell(jsonl)
            if m is None:
                continue
            # Patch in the metadata we know from the dir name.
            m.sp = sp
            m.seq_len = seq_len
            m.ckpt = ckpt
            m.rank = rank
            all_measurements.append(m)
            by_cfg[(sp, ckpt)].append(m)

    if not all_measurements:
        print("No measurements collected.", file=sys.stderr)
        return 1

    # Optionally overwrite the stored predicted_attention_ms with a fresh
    # computation under the current cost-model code. This keeps the fit
    # consistent with whatever attention model production loads.
    if args.attn_source == "live":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "galvatron", "site_package"))
        try:
            from galvatron.models.varlen_llama_hf.adacpsp_solver import (
                AdaCPSPCostModel, ParallelStrategy,
            )
        except ImportError as exc:
            print(f"Failed to import AdaCPSPCostModel for --attn-source=live: {exc}",
                  file=sys.stderr)
            return 1
        configs_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
        attn_json = sorted(glob.glob(os.path.join(configs_dir, "profile_validate_qwen2.5-7b_*.json")))[-1]
        comm_json = sorted(glob.glob(os.path.join(configs_dir, "comm_profile_v2_qwen2.5-7b_*.json")))[-1]
        print(f"\n[live] cost model from:")
        print(f"       attn={os.path.basename(attn_json)}")
        print(f"       comm={os.path.basename(comm_json)}")
        cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
            attention_json=attn_json, comm_profile_json=comm_json,
            cluster_size=args.cluster_size, gpus_per_node=args.gpus_per_node)
        # Reset residual to zero so it doesn't pollute the attention term we're
        # subtracting (we are calibrating it from scratch).
        cm.residual_a_per_sp = {}
        cm.residual_b_per_sp = {}
        cm.residual_a_default_per_token = 0.0
        cm.residual_b_default_ms = 0.0
        for m in all_measurements:
            if not m.group_seqlens or m.sp <= 0:
                continue
            strat = ParallelStrategy(
                attn_type=m.attn_type, parallel_size=m.sp, sp_size=m.sp,
                cp_size=1, placement=m.placement)
            m.predicted_attention_ms_live = float(cm.total_time(m.group_seqlens, strat))

    def metric_of(m: CellMeasurement) -> float:
        return {"steady": m.fb_ms_steady, "median": m.fb_ms_median,
                "p25": m.fb_ms_p25, "p10": m.fb_ms_p10}[args.metric]

    def attn_of(m: CellMeasurement) -> float:
        return (m.predicted_attention_ms_live
                if args.attn_source == "live" else m.predicted_attention_ms)

    print(f"\n=== Per-cell measurements (estimator = {args.metric}, attn = {args.attn_source}) ===")
    print(f"{'sp':>3s} {'seq':>6s} {'ckpt':>4s} {'rank':>4s} {'n':>3s} "
          f"{'fb_steady':>10s} {'fb_p10':>9s} {'fb_med':>9s} {'fb_p90':>9s} "
          f"{'attn_pred':>10s} {'residual':>10s} {'opt_med':>9s} {'tokens/gpu':>11s}")
    for m in sorted(all_measurements, key=lambda x: (x.sp, x.seq_len, x.ckpt, x.rank)):
        attn = attn_of(m)
        residual = metric_of(m) - attn
        print(f"{m.sp:>3d} {m.seq_len:>6d} {m.ckpt:>4d} {m.rank:>4d} {m.n_iters:>3d} "
              f"{fmt_ms(m.fb_ms_steady):>10s} {fmt_ms(m.fb_ms_p10):>9s} {fmt_ms(m.fb_ms_median):>9s} "
              f"{fmt_ms(m.fb_ms_p90):>9s} "
              f"{fmt_ms(attn):>10s} {fmt_ms(residual):>10s} "
              f"{fmt_ms(m.measured_opt_ms_median):>9s} {m.tokens_per_gpu:>11.0f}")

    dropped = []
    print("\n=== Linear residual fit (residual ≈ a·tokens_per_GPU + b) ===")
    print(f"  (dropping cells with median/p10 > {args.median_p10_ratio_max} "
          f"— real lower-bound is unstable)")
    print(f"{'sp':>3s} {'ckpt':>4s} {'n':>3s} {'a (ms/Ktoken)':>15s} {'b (ms)':>9s} "
          f"{'R²':>6s} {'max|err|':>9s}")
    fits = {}
    for (sp, ckpt), ms in sorted(by_cfg.items()):
        ms_r0 = [m for m in ms if m.rank == 0]
        clean = []
        for m in ms_r0:
            ratio = (m.fb_ms_median / m.fb_ms_p10) if m.fb_ms_p10 > 0 else float("inf")
            if ratio > args.median_p10_ratio_max:
                dropped.append((m.sp, m.seq_len, ratio, m.fb_ms_median, m.fb_ms_p10))
                continue
            clean.append(m)
        xs = [m.tokens_per_gpu for m in clean]
        ys = [metric_of(m) - attn_of(m) for m in clean]
        if len(xs) < 2:
            print(f"{sp:>3d} {ckpt:>4d} {len(xs):>3d}  (need ≥2 points to fit)")
            continue
        a, b, r2 = linreg(xs, ys)
        max_err = max(abs(y - (a * x + b)) for x, y in zip(xs, ys))
        print(f"{sp:>3d} {ckpt:>4d} {len(xs):>3d} {a*1000:>15.4f} {b:>9.2f} "
              f"{r2:>6.4f} {fmt_ms(max_err):>9s}")
        fits[(sp, ckpt)] = {"a": a, "b": b, "r2": r2, "max_err_ms": max_err,
                            "n_points": len(xs),
                            "points": [(x, y) for x, y in zip(xs, ys)]}
    if dropped:
        print("\n  Dropped as outliers (memory pressure / unstable):")
        for sp, seq, ratio, fb_med, fb_p10 in dropped:
            print(f"    sp={sp}  seq={seq:>6d}  median/p10={ratio:.2f}  "
                  f"med={fb_med:.0f}ms  p10={fb_p10:.0f}ms")

    # Cross-sp consistency: with uniform packing tokens/GPU = seq, a should be
    # roughly constant across sp (within a ckpt slice).
    print("\n=== Cross-sp `a` consistency (per ckpt) ===")
    by_ckpt = collections.defaultdict(list)
    for (sp, ckpt), fit in fits.items():
        by_ckpt[ckpt].append((sp, fit["a"], fit["b"]))
    for ckpt, lst in sorted(by_ckpt.items()):
        lst.sort()
        if len(lst) < 2:
            continue
        aa = [a for _, a, _ in lst]
        a_mean = sum(aa) / len(aa)
        a_std = (sum((x - a_mean) ** 2 for x in aa) / len(aa)) ** 0.5
        print(f"ckpt={ckpt}: a_per_token mean={a_mean*1000:.4f} ms/Ktoken  std={a_std*1000:.4f}  "
              f"std/mean={(a_std/a_mean*100 if a_mean else 0):.1f}%")
        for sp, a, b in lst:
            print(f"   sp={sp}: a={a*1000:.4f} ms/Ktoken  b={b:.2f} ms")

    if args.json_out:
        out = {
            "attn_source": args.attn_source,
            "metric": args.metric,
            "cells": [{k: v for k, v in m.__dict__.items()} for m in all_measurements],
            "fits": {f"sp{sp}_ckpt{ckpt}": fit for (sp, ckpt), fit in fits.items()},
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote fit JSON to {args.json_out}")

    # Also emit a compact loader-friendly file the cost model can ingest.
    if args.residual_config_out:
        # Use ckpt=0 fits (the production case). For each sp, store (a, b).
        residual_by_sp = {}
        for (sp, ckpt), fit in fits.items():
            if ckpt != 0:
                continue
            residual_by_sp[str(sp)] = {"a_per_token": fit["a"], "b_ms": fit["b"]}
        # `a_default` = median of per-sp a's (sp invariance heuristic for sp's we
        # haven't measured). `b_default` similar.
        if residual_by_sp:
            a_med = sorted(v["a_per_token"] for v in residual_by_sp.values())[len(residual_by_sp) // 2]
            b_med = sorted(v["b_ms"] for v in residual_by_sp.values())[len(residual_by_sp) // 2]
        else:
            a_med = 0.0
            b_med = 0.0
        cfg = {
            "schema": "adacpsp_residual_v1",
            "source_run": os.path.basename(args.results_dir.rstrip("/")),
            "estimator": args.metric,
            "residual_a_default_per_token": a_med,
            "residual_b_default_ms": b_med,
            "residual_per_sp": residual_by_sp,
        }
        with open(args.residual_config_out, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Wrote residual config to {args.residual_config_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
