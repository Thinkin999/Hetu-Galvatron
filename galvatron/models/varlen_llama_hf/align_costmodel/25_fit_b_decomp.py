"""Fit the b-decomposition cost model from 24_bench_b_decomp outputs.

Background
----------
The per-microbatch residual model fit by 23_fit_residual.py is
    T_per_group(seqlens, strat)
        = T_attention(seqlens, strat) + a · tokens_per_GPU + b(sp)
and was calibrated at chunks=1 (single forward_backward per step). When the
training loop uses N>1 microbatches per step, the predicted per-step
forward_backward should be N · T_per_group, but the measured forward_backward
might have an additional per-step constant (e.g. dispatcher sync, dataloader
overhead) that is NOT N-fold. Likewise the wall-clock step time includes
optimizer.step + grad_clip + zero_grad which is strictly per-step.

This script analyses the sweep produced by 24_bench_b_decomp_dispatch.sh
(varies `chunks` while holding sp, seq_length, fix_length dataset constant)
and decomposes:
    measured_fb(N)         = N · slope_per_mb + intercept_fb_per_step
    measured_wall(N)       = N · slope_per_mb + intercept_wall_per_step
                           = measured_fb(N) + per-step opt/grad_clip/zero_grad

If `intercept_fb_per_step` is small relative to `slope_per_mb`, the existing
b(sp) is already a clean per-microbatch quantity and the cost model needs no
change. Otherwise we report a per-step `b_step_fb` term that should be added
to the predicted step time (independent of N).

Output
------
- Per-cell summary table (fb_steady, opt, grad_clip, zero_grad, predicted_attn).
- Per-(sp, seq) linear fit `forward_backward = slope·N + intercept`.
- Implied b_microbatch(sp) = slope - a_existing·tokens - T_attention.
- Comparison with existing b(sp) from configs/residual_profile_*.json.
- Per-step external overhead (mean opt+grad_clip+zero_grad).

Usage:
  python 25_fit_b_decomp.py <results_dir> [--ranks 0]
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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# --------------------------- IO --------------------------- #

CELL_RE = re.compile(r"sp(\d+)_seq(\d+)_chunks(\d+)")


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


def percentile(xs: list, p: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs2[int(k)]
    return xs2[f] + (xs2[c] - xs2[f]) * (k - f)


@dataclass
class CellMeasurement:
    sp: int
    seq_len: int
    chunks: int
    rank: int
    n_iters: int
    n_predicted_mbs: int
    fb_ms_median: float
    fb_ms_p10: float
    fb_ms_steady: float  # mean of iters in [p10, median]
    # Mean of the 3 cleanest (smallest) iters EXCLUDING iter 0. This isolates
    # the ideal per-microbatch cost before memory-pressure / fragmentation
    # slowdown kicks in (observed in chunks≥2 sweeps where iters 1-2 cleanly
    # match N·T_mb but later iters drift higher).
    fb_ms_clean: float
    opt_ms_median: float
    grad_clip_ms_median: float
    zero_grad_ms_median: float
    solve_dispatch_ms_median: float
    wall_step_ms_steady: float
    predicted_attention_ms_stored: float
    predicted_attention_ms_live: float = 0.0
    tokens_per_gpu: float = 0.0
    group_seqlens: list = field(default_factory=list)
    attn_type: str = "ulysses"
    placement: str = "head_first"
    strategy_label: str = ""


def measure_cell(jsonl_path: str) -> CellMeasurement | None:
    records = load_jsonl(jsonl_path)
    train_recs = [r for r in records if r.get("phase") == "train_step"]
    if not train_recs:
        return None

    def collect(timing_key, agg):
        vals = [r["timings_ms"].get(timing_key, 0.0) for r in train_recs
                if "timings_ms" in r]
        if not vals:
            return 0.0
        if agg == "median":
            return percentile(vals, 0.5)
        if agg == "steady":
            p10 = percentile(vals, 0.1)
            med = percentile(vals, 0.5)
            lh = [v for v in vals if p10 <= v <= med]
            return (sum(lh) / len(lh)) if lh else med
        if agg == "p10":
            return percentile(vals, 0.1)
        raise ValueError(agg)

    fb = [r["timings_ms"]["forward_backward"] for r in train_recs
          if "timings_ms" in r and "forward_backward" in r["timings_ms"]]
    if not fb:
        return None
    # "Cleanest" = mean of the 3 smallest fb values excluding the very first
    # iteration (which is dominated by torch.compile / cache warm-up).
    fb_skip_first = fb[1:] if len(fb) > 1 else fb
    cleanest = sorted(fb_skip_first)[:3] if fb_skip_first else []
    fb_clean = (sum(cleanest) / len(cleanest)) if cleanest else 0.0

    last = train_recs[-1]
    pred = last.get("predicted_adacpsp") or {}
    pred_attn_ms_stored = float(pred.get("total_ms", 0.0))
    pred_mbs = pred.get("microbatches", []) or []
    n_predicted_mbs = len(pred_mbs)

    # First microbatch, first group → strategy + group_tokens (uniform across mbs in our sweep)
    g0 = (pred_mbs[0]["groups"][0] if pred_mbs and pred_mbs[0].get("groups") else {})
    sp_size = int(g0.get("sp_size", 0))
    group_tokens = int(g0.get("tokens", 0))
    attn_type = str(g0.get("attn_type", "ulysses"))
    placement = str(g0.get("placement", "head_first"))
    tokens_per_gpu = (group_tokens / sp_size) if sp_size > 0 else 0.0

    if sp_size > 0 and group_tokens > 0:
        # In fix_length all sequences = seq_len; each forced group sees 1 seq.
        group_seqlens = [group_tokens]  # ulysses group has 1 contiguous packed sample
    else:
        group_seqlens = []

    rank = int(last.get("rank", -1))
    strategy_label = str(last.get("strategy_label") or "")

    return CellMeasurement(
        sp=sp_size,
        seq_len=0,
        chunks=0,
        rank=rank,
        n_iters=len(fb),
        n_predicted_mbs=n_predicted_mbs,
        fb_ms_median=collect("forward_backward", "median"),
        fb_ms_p10=collect("forward_backward", "p10"),
        fb_ms_steady=collect("forward_backward", "steady"),
        fb_ms_clean=fb_clean,
        opt_ms_median=collect("optimizer_step", "median"),
        grad_clip_ms_median=collect("grad_clip", "median"),
        zero_grad_ms_median=collect("zero_grad", "median"),
        solve_dispatch_ms_median=collect("solve_and_dispatch", "median"),
        wall_step_ms_steady=collect("wall_step_total", "steady"),
        predicted_attention_ms_stored=pred_attn_ms_stored,
        predicted_attention_ms_live=pred_attn_ms_stored,
        tokens_per_gpu=tokens_per_gpu,
        group_seqlens=group_seqlens,
        attn_type=attn_type,
        placement=placement,
        strategy_label=strategy_label,
    )


# --------------------------- linear fit --------------------------- #

def linreg(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
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
    return f"{v:8.2f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", help="run dir containing end2end/ or end2end/ itself.")
    p.add_argument("--ranks", default="0",
                   help="Space/comma separated ranks to consider (default: 0).")
    p.add_argument("--cluster-size", type=int, default=16)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--residual-profile", default=None,
                   help="Existing residual_profile JSON (defaults to newest in configs/).")
    p.add_argument("--json-out", default=None)
    p.add_argument("--b-decomp-out", default=None,
                   help="Path to write the b-decomposition JSON consumable by the cost model.")
    args = p.parse_args()

    allowed_ranks = set(int(x) for x in re.split(r"[,\s]+", args.ranks.strip()) if x)

    end2end_dir = args.results_dir
    if os.path.basename(end2end_dir.rstrip("/")) != "end2end":
        cand = os.path.join(end2end_dir, "end2end")
        if os.path.isdir(cand):
            end2end_dir = cand
    if not os.path.isdir(end2end_dir):
        print(f"No such directory: {end2end_dir}", file=sys.stderr)
        return 1

    cell_dirs = sorted(d for d in glob.glob(os.path.join(end2end_dir, "sp*_seq*_chunks*"))
                       if os.path.isdir(d))
    if not cell_dirs:
        print(f"No sp*_seq*_chunks* cells under {end2end_dir}", file=sys.stderr)
        return 1

    # Collect measurements.
    by_cfg = collections.defaultdict(list)   # (sp, seq) -> [CellMeasurement, ...]
    all_measurements: List[CellMeasurement] = []
    for cell_dir in cell_dirs:
        parsed = parse_cell(cell_dir)
        if parsed is None:
            continue
        sp, seq_len, chunks = parsed
        for rank in sorted(allowed_ranks):
            jsonl = os.path.join(cell_dir, f"rank{rank}.jsonl")
            if not os.path.isfile(jsonl):
                continue
            m = measure_cell(jsonl)
            if m is None:
                continue
            m.sp = sp
            m.seq_len = seq_len
            m.chunks = chunks
            m.rank = rank
            all_measurements.append(m)
            by_cfg[(sp, seq_len)].append(m)

    if not all_measurements:
        print("No measurements collected.", file=sys.stderr)
        return 1

    # Recompute live T_attention so the residual stays consistent with current
    # cost-model code.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "galvatron", "site_package"))
    try:
        from galvatron.models.varlen_llama_hf.adacpsp_solver import (
            AdaCPSPCostModel, ParallelStrategy,
        )
    except ImportError as exc:
        print(f"Failed to import AdaCPSPCostModel: {exc}", file=sys.stderr)
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
    # Reset residual to zero — we want the pure attention term for subtraction.
    cm.residual_a_per_sp = {}
    cm.residual_b_per_sp = {}
    cm.residual_a_default_per_token = 0.0
    cm.residual_b_default_ms = 0.0

    for m in all_measurements:
        if m.sp <= 0 or not m.group_seqlens:
            continue
        strat = ParallelStrategy(attn_type=m.attn_type, parallel_size=m.sp,
                                 sp_size=m.sp, cp_size=1, placement=m.placement)
        m.predicted_attention_ms_live = float(cm.total_time(m.group_seqlens, strat))

    # Load existing residual profile (a_per_sp, b_per_sp) for comparison.
    existing_residual = None
    if args.residual_profile and os.path.isfile(args.residual_profile):
        existing_path = args.residual_profile
    else:
        cand = sorted(glob.glob(os.path.join(configs_dir, "residual_profile_*.json")))
        existing_path = cand[-1] if cand else None
    if existing_path:
        with open(existing_path) as f:
            existing_residual = json.load(f)
        print(f"[existing residual] {os.path.basename(existing_path)}")
    else:
        print("[existing residual] NONE found — will only show new fit.")

    # --------------------------- per-cell summary table --------------------------- #
    print(f"\n=== Per-cell measurements ===")
    print(f"  fb_clean  = mean of 3 lowest fb values excluding iter 0 (ideal pre-pressure)")
    print(f"  fb_steady = mean of iters in [p10, median] (typical production)")
    hdr = f"{'sp':>3s} {'seq':>6s} {'N':>3s} {'rk':>3s} {'iters':>5s} {'mbs':>4s} " \
          f"{'fb_clean':>9s} {'fb_steady':>10s} {'fb_p10':>9s} {'opt':>6s} " \
          f"{'gclip':>6s} {'zgrad':>6s} {'solve':>6s} " \
          f"{'attn':>9s} {'res_clean':>10s} {'res_steady':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for m in sorted(all_measurements, key=lambda x: (x.sp, x.seq_len, x.chunks, x.rank)):
        residual_clean = m.fb_ms_clean - m.predicted_attention_ms_live
        residual_steady = m.fb_ms_steady - m.predicted_attention_ms_live
        print(f"{m.sp:>3d} {m.seq_len:>6d} {m.chunks:>3d} {m.rank:>3d} {m.n_iters:>5d} "
              f"{m.n_predicted_mbs:>4d} "
              f"{m.fb_ms_clean:>9.1f} {m.fb_ms_steady:>10.1f} {m.fb_ms_p10:>9.1f} "
              f"{m.opt_ms_median:>6.1f} {m.grad_clip_ms_median:>6.1f} "
              f"{m.zero_grad_ms_median:>6.2f} {m.solve_dispatch_ms_median:>6.2f} "
              f"{m.predicted_attention_ms_live:>9.1f} {residual_clean:>10.1f} {residual_steady:>11.1f}")

    # --------------------------- linear fit forward_backward vs chunks --------------------------- #
    print("\n=== Linear fit: forward_backward(N) ≈ slope_per_mb · N + intercept_fb ===")
    print("    Fit on `fb_clean` (ideal pre-pressure) AND `fb_steady` (typical).")
    print(f"{'sp':>3s} {'seq':>6s} {'metric':>7s} {'n':>3s} {'slope_per_mb':>13s} "
          f"{'intercept':>10s} {'R²':>6s} {'attn(N=1)':>10s} {'a·tok+b_mb':>12s}")

    decomp_fits: Dict[Tuple[int, int], dict] = {}
    for (sp, seq), cells in sorted(by_cfg.items()):
        cells_r0 = [c for c in cells if c.rank == 0]
        cells_r0.sort(key=lambda c: c.chunks)
        if len(cells_r0) < 2:
            print(f"{sp:>3d} {seq:>6d} {len(cells_r0):>3d}  (need ≥2 chunks points)")
            continue
        xs = [float(c.chunks) for c in cells_r0]
        attn_n1 = cells_r0[0].predicted_attention_ms_live

        ys_clean = [c.fb_ms_clean for c in cells_r0]
        slope_c, intercept_c, r2_c = linreg(xs, ys_clean)
        per_mb_residual_c = slope_c - attn_n1
        print(f"{sp:>3d} {seq:>6d} {'clean':>7s} {len(cells_r0):>3d} {slope_c:>13.2f} "
              f"{intercept_c:>10.2f} {r2_c:>6.4f} {attn_n1:>10.2f} {per_mb_residual_c:>12.2f}")

        ys_steady = [c.fb_ms_steady for c in cells_r0]
        slope_s, intercept_s, r2_s = linreg(xs, ys_steady)
        per_mb_residual_s = slope_s - attn_n1
        print(f"{sp:>3d} {seq:>6d} {'steady':>7s} {len(cells_r0):>3d} {slope_s:>13.2f} "
              f"{intercept_s:>10.2f} {r2_s:>6.4f} {attn_n1:>10.2f} {per_mb_residual_s:>12.2f}")

        decomp_fits[(sp, seq)] = {
            "clean": {
                "slope_per_mb_ms": slope_c, "intercept_fb_ms": intercept_c,
                "r2": r2_c, "per_mb_residual_ms": per_mb_residual_c,
            },
            "steady": {
                "slope_per_mb_ms": slope_s, "intercept_fb_ms": intercept_s,
                "r2": r2_s, "per_mb_residual_ms": per_mb_residual_s,
            },
            "attn_n1_ms": attn_n1,
            "tokens_per_gpu": cells_r0[0].tokens_per_gpu,
            "points": [(c.chunks, c.fb_ms_clean, c.fb_ms_steady,
                        c.predicted_attention_ms_live) for c in cells_r0],
        }

    # --------------------------- per-step external overhead --------------------------- #
    print("\n=== Per-step external (non-FB) overhead, median over all cells ===")
    all_opt = sorted(m.opt_ms_median for m in all_measurements if m.rank == 0)
    all_grad = sorted(m.grad_clip_ms_median for m in all_measurements if m.rank == 0)
    all_zero = sorted(m.zero_grad_ms_median for m in all_measurements if m.rank == 0)
    all_solve = sorted(m.solve_dispatch_ms_median for m in all_measurements if m.rank == 0)
    def med(v):
        return percentile(v, 0.5) if v else 0.0
    opt_m, grad_m, zero_m, solve_m = med(all_opt), med(all_grad), med(all_zero), med(all_solve)
    ext_total = opt_m + grad_m + zero_m + solve_m
    print(f"  optimizer_step    median = {opt_m:8.2f} ms")
    print(f"  grad_clip         median = {grad_m:8.2f} ms")
    print(f"  zero_grad         median = {zero_m:8.2f} ms")
    print(f"  solve_and_dispatch median = {solve_m:8.2f} ms")
    print(f"  ── total per-step external ── = {ext_total:8.2f} ms")

    # --------------------------- comparison with existing residual --------------------------- #
    if existing_residual:
        print("\n=== Comparison vs existing residual_profile (chunks=1 baseline) ===")
        print(f"{'sp':>3s} {'seq':>6s} {'metric':>7s} {'tok/gpu':>8s} {'exist_b':>9s} "
              f"{'per_mb_res':>11s} {'a·tok':>9s} {'implied_b_mb':>14s} {'fb_intercept':>13s}")
        per_sp = existing_residual.get("residual_per_sp", {})
        a_def = float(existing_residual.get("residual_a_default_per_token", 0.0))
        b_def = float(existing_residual.get("residual_b_default_ms", 0.0))
        for (sp, seq), fit in sorted(decomp_fits.items()):
            ent = per_sp.get(str(sp)) or {}
            a_sp = float(ent.get("a_per_token", a_def))
            b_sp = float(ent.get("b_ms", b_def))
            tok = fit["tokens_per_gpu"]
            a_tok = a_sp * tok
            for label in ("clean", "steady"):
                sub = fit[label]
                implied_b_mb = sub["per_mb_residual_ms"] - a_tok
                print(f"{sp:>3d} {seq:>6d} {label:>7s} {tok:>8.0f} {b_sp:>9.2f} "
                      f"{sub['per_mb_residual_ms']:>11.2f} {a_tok:>9.2f} "
                      f"{implied_b_mb:>14.2f} {sub['intercept_fb_ms']:>13.2f}")

    # --------------------------- write outputs --------------------------- #
    out = {
        "schema": "adacpsp_b_decomp_v1",
        "source_run": os.path.basename(args.results_dir.rstrip("/")),
        "cluster_size": args.cluster_size,
        "cells": [vars(m) for m in all_measurements],
        "fits": {f"sp{sp}_seq{seq}": fit for (sp, seq), fit in decomp_fits.items()},
        "per_step_external_ms": {
            "optimizer_step_median": opt_m,
            "grad_clip_median": grad_m,
            "zero_grad_median": zero_m,
            "solve_and_dispatch_median": solve_m,
            "total_median": ext_total,
        },
    }
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote analysis JSON to {args.json_out}")

    if args.b_decomp_out:
        # Compact loader-friendly JSON: per-sp b_microbatch + a (reuse existing
        # a from residual_profile), plus a global b_step (= intercept_fb median).
        # Use the `clean` fit — it isolates the ideal per-microbatch cost
        # before memory-pressure / fragmentation slowdown kicks in.
        per_sp_b_mb = {}
        if existing_residual:
            per_sp = existing_residual.get("residual_per_sp", {})
            a_def = float(existing_residual.get("residual_a_default_per_token", 0.0))
            for (sp, seq), fit in decomp_fits.items():
                ent = per_sp.get(str(sp)) or {}
                a_sp = float(ent.get("a_per_token", a_def))
                tok = fit["tokens_per_gpu"]
                implied_b_mb = fit["clean"]["per_mb_residual_ms"] - a_sp * tok
                per_sp_b_mb[str(sp)] = {
                    "a_per_token": a_sp,
                    "b_microbatch_ms": implied_b_mb,
                }
        intercepts_clean = [fit["clean"]["intercept_fb_ms"] for fit in decomp_fits.values()]
        intercepts_steady = [fit["steady"]["intercept_fb_ms"] for fit in decomp_fits.values()]
        b_step_fb_clean = percentile(intercepts_clean, 0.5) if intercepts_clean else 0.0
        b_step_fb_steady = percentile(intercepts_steady, 0.5) if intercepts_steady else 0.0
        cfg = {
            "schema": "adacpsp_b_decomp_v1",
            "source_run": os.path.basename(args.results_dir.rstrip("/")),
            "residual_per_sp": per_sp_b_mb,
            "b_step_fb_ms_clean": b_step_fb_clean,
            "b_step_fb_ms_steady": b_step_fb_steady,
            "b_step_external_ms": ext_total,
        }
        with open(args.b_decomp_out, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Wrote b-decomp config to {args.b_decomp_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
