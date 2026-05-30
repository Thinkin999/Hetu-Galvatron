"""Validate the multi-microbatch cost model against measured github runs.

Reads `26_bench_github_multimb_dispatch.sh` outputs (end2end JSONL per cell)
and the recorded `predicted_adacpsp` fields, then compares:

  measured_fb_clean   vs   predicted_total_fb_ms (= sum per-mb + b_step_fb_ms)
  measured_fb_steady  vs   predicted_total_fb_ms

It also computes pairwise speedups (relative to a baseline cell, default
`ulysses8_chunks1`) and reports predicted-vs-measured speedup deltas.

Note: github sequences are variable length, so each cell will see different
sequence-length distributions; the predicted_adacpsp values are recorded at
training step 0 (single dataloader batch) for these runs, which is what we
compare against the steady-state measured fb. Because dataloader iteration
re-samples, predicted may shift across iters; we use the last logged record's
predicted as the representative point estimate per cell.

Usage:
  python 27_validate_multimb.py <results_dir> [--baseline ulysses8_chunks1]
"""

from __future__ import annotations
import argparse, collections, glob, json, math, os, re, sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


CELL_RE = re.compile(r"(?P<cfg>[a-z0-9]+)_chunks(?P<ch>[a-zA-Z0-9]+)")


def parse_cell(d: str):
    m = CELL_RE.match(os.path.basename(d.rstrip("/")))
    if not m:
        return None
    return m.group("cfg"), m.group("ch")


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


@dataclass
class CellResult:
    cfg: str
    chunks: str
    rank: int
    n_iters: int
    n_predicted_mbs: int

    fb_clean_ms: float           # mean of 3 lowest forward_backward iters (excl iter 0)
    fb_steady_ms: float          # mean of [p10, median] window
    fb_median_ms: float
    fb_p10_ms: float
    wall_steady_ms: float        # measured wall-clock step time

    predicted_per_mb_total_ms: float   # sum of per-mb max-over-groups predictions
    predicted_total_fb_ms: float       # predicted total fb (incl b_step_fb)
    predicted_total_step_ms: float     # predicted wall step (incl b_step_ext)
    predicted_b_step_fb_ms: float
    predicted_b_step_external_ms: float

    measured_external_ms: float        # median(opt+grad_clip+zero_grad+solve)
    avg_seq_len: float
    max_seq_len: int


def measure(p):
    recs = load_jsonl(p)
    tr = [r for r in recs if r.get("phase") == "train_step"]
    if not tr:
        return None
    fb = [r["timings_ms"].get("forward_backward", 0.0) for r in tr]
    fb_body = fb[1:] if len(fb) > 1 else fb
    fb_clean = (sum(sorted(fb_body)[:3]) / min(3, len(fb_body))) if fb_body else 0.0
    p10 = pct(fb_body, 0.1)
    med = pct(fb_body, 0.5)
    lh = [v for v in fb_body if p10 <= v <= med]
    fb_steady = (sum(lh) / len(lh)) if lh else med

    wall = [r["timings_ms"].get("wall_step_total", 0.0) for r in tr]
    wall_body = wall[1:] if len(wall) > 1 else wall
    p10w = pct(wall_body, 0.1); medw = pct(wall_body, 0.5)
    lhw = [v for v in wall_body if p10w <= v <= medw]
    wall_steady = (sum(lhw) / len(lhw)) if lhw else medw

    ext_iter = []
    for r in tr:
        t = r.get("timings_ms", {})
        ext_iter.append(t.get("optimizer_step", 0.0) + t.get("grad_clip", 0.0)
                        + t.get("zero_grad", 0.0) + t.get("solve_and_dispatch", 0.0))
    ext_med = pct(ext_iter, 0.5) if ext_iter else 0.0

    last = tr[-1]
    pred = last.get("predicted_adacpsp") or {}
    mbs = pred.get("microbatches", []) or []
    n_mbs = len(mbs)
    pred_per_mb_total = float(pred.get("total_ms", 0.0))
    pred_total_fb = float(pred.get("total_fb_ms", pred_per_mb_total))
    pred_total_step = float(pred.get("total_step_ms", pred_total_fb))
    b_step_fb = float(pred.get("b_step_fb_ms", 0.0))
    b_step_ext = float(pred.get("b_step_external_ms", 0.0))

    # gather sequence-length stats from predicted microbatches
    all_seqlens = []
    for mb in mbs:
        for g in mb.get("groups", []) or []:
            tokens = g.get("tokens", 0)
            nseqs = g.get("num_sequences", 1) or 1
            avg = tokens / max(1, nseqs)
            all_seqlens.append(avg)
    avg_seq = (sum(all_seqlens) / len(all_seqlens)) if all_seqlens else 0.0
    max_seq = max(all_seqlens) if all_seqlens else 0

    return CellResult(
        cfg="", chunks="", rank=int(last.get("rank", -1)),
        n_iters=len(fb), n_predicted_mbs=n_mbs,
        fb_clean_ms=fb_clean, fb_steady_ms=fb_steady,
        fb_median_ms=med, fb_p10_ms=p10,
        wall_steady_ms=wall_steady,
        predicted_per_mb_total_ms=pred_per_mb_total,
        predicted_total_fb_ms=pred_total_fb,
        predicted_total_step_ms=pred_total_step,
        predicted_b_step_fb_ms=b_step_fb,
        predicted_b_step_external_ms=b_step_ext,
        measured_external_ms=ext_med,
        avg_seq_len=avg_seq,
        max_seq_len=max_seq,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--ranks", default="0")
    ap.add_argument("--baseline", default="ulysses8_chunks1",
                    help="Cell to use as speedup denominator.")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    allowed = set(int(x) for x in re.split(r"[,\s]+", args.ranks.strip()) if x)

    e2e = args.results_dir
    if os.path.basename(e2e.rstrip("/")) != "end2end":
        cand = os.path.join(e2e, "end2end")
        if os.path.isdir(cand):
            e2e = cand
    if not os.path.isdir(e2e):
        print(f"missing: {e2e}", file=sys.stderr); return 1

    cells = []
    for d in sorted(glob.glob(os.path.join(e2e, "*_chunks*"))):
        p = parse_cell(d)
        if p is None:
            continue
        cfg, ch = p
        for rank in sorted(allowed):
            jp = os.path.join(d, f"rank{rank}.jsonl")
            if not os.path.isfile(jp):
                continue
            m = measure(jp)
            if m is None:
                continue
            m.cfg = cfg; m.chunks = ch; m.rank = rank
            cells.append(m)

    if not cells:
        print("no cells", file=sys.stderr); return 1

    # ------------------------------------------------------------------ #
    # Per-cell predicted vs measured                                     #
    # ------------------------------------------------------------------ #
    print("\n=== Per-cell measurements vs predictions (rank 0) ===")
    print("  fb_clean : mean of 3 lowest fb iters excl iter 0 (ideal pre-pressure)")
    print("  fb_steady: mean of [p10, median] window")
    print("  Δ_clean  : (measured_clean  - predicted_fb) / measured_clean  × 100")
    print("  Δ_steady : (measured_steady - predicted_fb) / measured_steady × 100")
    hdr = (f"{'cfg':>10} {'ch':>4} {'mb':>3} {'iters':>5} {'maxseq':>6} {'avgseq':>6} "
           f"{'fb_clean':>9} {'fb_steady':>10} {'pred_fb':>8} {'b_step':>7} "
           f"{'Δ_clean':>8} {'Δ_steady':>9} "
           f"{'wall':>7} {'pred_step':>10} {'ext_meas':>9} {'ext_pred':>9}")
    print(hdr); print("-" * len(hdr))

    rows = []
    for c in sorted(cells, key=lambda x: (x.cfg, x.chunks)):
        dpc = (c.fb_clean_ms - c.predicted_total_fb_ms) / max(1.0, c.fb_clean_ms) * 100.0
        dps = (c.fb_steady_ms - c.predicted_total_fb_ms) / max(1.0, c.fb_steady_ms) * 100.0
        print(f"{c.cfg:>10} {c.chunks:>4} {c.n_predicted_mbs:>3} {c.n_iters:>5} "
              f"{c.max_seq_len:>6} {c.avg_seq_len:>6.0f} "
              f"{c.fb_clean_ms:>9.1f} {c.fb_steady_ms:>10.1f} "
              f"{c.predicted_total_fb_ms:>8.1f} {c.predicted_b_step_fb_ms:>7.1f} "
              f"{dpc:>+7.1f}% {dps:>+8.1f}% "
              f"{c.wall_steady_ms:>7.1f} {c.predicted_total_step_ms:>10.1f} "
              f"{c.measured_external_ms:>9.1f} {c.predicted_b_step_external_ms:>9.1f}")
        rows.append({
            "cfg": c.cfg, "chunks": c.chunks,
            "n_microbatches": c.n_predicted_mbs,
            "fb_clean_ms": c.fb_clean_ms,
            "fb_steady_ms": c.fb_steady_ms,
            "predicted_total_fb_ms": c.predicted_total_fb_ms,
            "delta_clean_pct": dpc,
            "delta_steady_pct": dps,
            "wall_steady_ms": c.wall_steady_ms,
            "predicted_total_step_ms": c.predicted_total_step_ms,
            "external_measured_ms": c.measured_external_ms,
            "external_predicted_ms": c.predicted_b_step_external_ms,
            "max_seq_len": c.max_seq_len,
            "avg_seq_len": c.avg_seq_len,
        })

    # ------------------------------------------------------------------ #
    # Speedup table relative to baseline cell                            #
    # ------------------------------------------------------------------ #
    cells_by_label = {f"{c.cfg}_chunks{c.chunks}": c for c in cells if c.rank == 0}
    if args.baseline in cells_by_label:
        base = cells_by_label[args.baseline]
        print(f"\n=== Speedup vs {args.baseline} (fb_clean as primary measured signal) ===")
        print("  speedup = T_baseline / T_strategy")
        print("  Δsu = predicted - measured (positive = model over-predicts gains)")
        h2 = (f"{'cfg':>10} {'ch':>4} "
              f"{'meas_cl':>8} {'meas_st':>8} {'pred_fb':>8} "
              f"{'su_cl':>6} {'su_st':>6} {'su_pred':>7} "
              f"{'Δsu_cl':>7} {'Δsu_st':>7}")
        print(h2); print("-" * len(h2))
        speedup_rows = []
        for c in sorted(cells, key=lambda x: (x.cfg, x.chunks)):
            if c.rank != 0:
                continue
            su_clean = base.fb_clean_ms / max(1.0, c.fb_clean_ms)
            su_steady = base.fb_steady_ms / max(1.0, c.fb_steady_ms)
            su_pred = base.predicted_total_fb_ms / max(1.0, c.predicted_total_fb_ms)
            dsu_clean = su_pred - su_clean
            dsu_steady = su_pred - su_steady
            print(f"{c.cfg:>10} {c.chunks:>4} "
                  f"{c.fb_clean_ms:>8.0f} {c.fb_steady_ms:>8.0f} {c.predicted_total_fb_ms:>8.0f} "
                  f"{su_clean:>6.3f} {su_steady:>6.3f} {su_pred:>7.3f} "
                  f"{dsu_clean:>+6.3f} {dsu_steady:>+6.3f}")
            speedup_rows.append({
                "cfg": c.cfg, "chunks": c.chunks,
                "measured_speedup_clean": su_clean,
                "measured_speedup_steady": su_steady,
                "predicted_speedup_fb": su_pred,
                "delta_speedup_clean": dsu_clean,
                "delta_speedup_steady": dsu_steady,
            })
    else:
        print(f"\n(baseline {args.baseline} not found; available: {list(cells_by_label.keys())})")
        speedup_rows = []

    # ------------------------------------------------------------------ #
    # Aggregate fit-quality stats                                        #
    # ------------------------------------------------------------------ #
    def mean_abs(xs):
        return (sum(abs(x) for x in xs) / len(xs)) if xs else 0.0
    deltas_clean = [r["delta_clean_pct"] for r in rows]
    deltas_steady = [r["delta_steady_pct"] for r in rows]
    print("\n=== Aggregate fit quality (Δ = measured - predicted) ===")
    print(f"  mean |Δ_clean|  = {mean_abs(deltas_clean):6.2f}%   "
          f"max |Δ_clean|  = {max(abs(d) for d in deltas_clean) if deltas_clean else 0:6.2f}%")
    print(f"  mean |Δ_steady| = {mean_abs(deltas_steady):6.2f}%   "
          f"max |Δ_steady| = {max(abs(d) for d in deltas_steady) if deltas_steady else 0:6.2f}%")

    if args.json_out:
        out = {
            "schema": "adacpsp_multimb_validation_v1",
            "source_run": os.path.basename(args.results_dir.rstrip("/")),
            "baseline_cell": args.baseline,
            "per_cell": rows,
            "speedups_vs_baseline": speedup_rows,
            "summary": {
                "mean_abs_delta_clean_pct": mean_abs(deltas_clean),
                "max_abs_delta_clean_pct": max((abs(d) for d in deltas_clean), default=0.0),
                "mean_abs_delta_steady_pct": mean_abs(deltas_steady),
                "max_abs_delta_steady_pct": max((abs(d) for d in deltas_steady), default=0.0),
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote validation JSON -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
