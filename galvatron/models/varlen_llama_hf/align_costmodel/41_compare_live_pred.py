"""
Compare live solver predictions vs measured fb_ms, per cell, with MAD outlier
filtering. Live predictions are recorded by train_dist_adacpsp.py as
`predicted_adacpsp.total_fb_ms` — exactly what the cost model said for the
specific seq pack the runtime saw, so it's the cleanest ground truth for
diagnosing model accuracy.

Outputs per-cell:
  meas_clean = MAD-filtered measured forward_backward (ms)
  live_pred  = recorded predicted_adacpsp.total_fb_ms
  err_signed = (meas - pred) / meas  (positive = under-pred, negative = over-pred)
  ratio      = meas / pred

Also computes pairwise speedup MAPE to gauge solver decision quality.

Usage:
  python 41_compare_live_pred.py <results_dir> [--skip 5]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
from typing import Dict, List, Tuple


def stats_cell(jsonl: Path, skip: int):
    recs = [json.loads(l) for l in open(jsonl) if l.strip()]
    train = [r for r in recs if r.get("phase") == "train_step"]
    if len(train) <= skip:
        return None
    train = train[skip:]
    fbs = [float(r["timings_ms"]["forward_backward"]) for r in train]
    preds = []
    for r in train:
        p = r.get("predicted_adacpsp")
        if isinstance(p, dict):
            preds.append(float(p.get("total_fb_ms", 0)))
        else:
            preds.append(None)

    # MAD outlier filter on fbs
    sorted_fbs = sorted(fbs)
    med = sorted_fbs[len(sorted_fbs) // 2]
    mad = sorted([abs(x - med) for x in fbs])[len(fbs) // 2]
    thr = med + 3 * (mad if mad > 0 else med * 0.3)
    pairs = [(fb, p) for fb, p in zip(fbs, preds) if fb <= thr and p is not None]
    if not pairs:
        return None
    fbs_clean = [x for x, _ in pairs]
    preds_clean = [p for _, p in pairs]
    return dict(
        n_total=len(fbs),
        n_clean=len(pairs),
        meas_avg=sum(fbs_clean) / len(fbs_clean),
        meas_med=sorted(fbs_clean)[len(fbs_clean) // 2],
        pred_avg=sum(preds_clean) / len(preds_clean),
        ratio=(sum(fbs_clean) / len(fbs_clean)) / max(1, sum(preds_clean) / len(preds_clean)),
        err_signed=100 * (sum(fbs_clean) / len(fbs_clean)
                          - sum(preds_clean) / len(preds_clean))
                       / max(1, sum(fbs_clean) / len(fbs_clean)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--skip", type=int, default=5)
    args = ap.parse_args()
    e2e = Path(args.results_dir)
    if e2e.name != "end2end":
        cand = e2e / "end2end"
        if cand.is_dir():
            e2e = cand

    cell_stats = {}
    for cell_dir in sorted(e2e.glob("*_chunks*")):
        jp = cell_dir / "rank0.jsonl"
        if not jp.exists():
            continue
        s = stats_cell(jp, args.skip)
        if s is not None:
            cell_stats[cell_dir.name] = s

    if not cell_stats:
        print(f"No usable cells in {e2e}", file=sys.stderr)
        return 1

    print(f"# Source: {e2e}")
    print(f"# skip first {args.skip} iters, MAD-filter outliers above 3*MAD")
    print()
    print(f"{'cell':<22} {'n':>8} {'meas_clean':>10} {'live_pred':>10} "
          f"{'ratio':>6} {'err':>7}")
    print("-" * 70)
    sum_abs = 0; n_total = 0
    for name, s in cell_stats.items():
        print(f"{name:<22} {s['n_clean']:>3}/{s['n_total']:<4} "
              f"{s['meas_avg']:>10.0f} {s['pred_avg']:>10.0f} "
              f"{s['ratio']:>6.2f} {s['err_signed']:>+6.1f}%")
        sum_abs += abs(s['err_signed']) * s['n_clean']
        n_total += s['n_clean']
    print("-" * 70)
    print(f"Weighted MAPE: {sum_abs / max(1, n_total):.1f}%")
    print()

    # Speedup ranking accuracy if we have multiple cells
    names = list(cell_stats.keys())
    if len(names) >= 2:
        print("=== Pairwise speedup accuracy (meas vs pred) ===")
        print(f"{'pair':<48} {'meas_speedup':>13} {'pred_speedup':>13} "
              f"{'err_pp':>7} {'sign_ok':>7}")
        print("-" * 95)
        pairs_total = 0
        pairs_correct_sign = 0
        speedup_abs_errs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                sa, sb = cell_stats[a], cell_stats[b]
                meas_sp = sa["meas_avg"] / sb["meas_avg"]
                pred_sp = sa["pred_avg"] / sb["pred_avg"]
                err_pp = (meas_sp - pred_sp) * 100  # percentage points
                # Sign correct if both > 1 or both < 1
                sign_ok = (meas_sp >= 1) == (pred_sp >= 1)
                pairs_total += 1
                if sign_ok:
                    pairs_correct_sign += 1
                speedup_abs_errs.append(abs(err_pp))
                print(f"{a + ' vs ' + b:<48} {meas_sp:>13.3f} {pred_sp:>13.3f} "
                      f"{err_pp:>+6.1f}pp {('Y' if sign_ok else 'N'):>7}")
        print("-" * 95)
        print(f"Speedup MAPE (avg of |err_pp|): {sum(speedup_abs_errs)/len(speedup_abs_errs):.1f}pp")
        print(f"Sign accuracy: {pairs_correct_sign}/{pairs_total} = "
              f"{100*pairs_correct_sign/pairs_total:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
