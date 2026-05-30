"""Per-iter matched prediction-vs-measurement validation.

For each iteration of each cell, we have:
  - measured fb_ms = timings_ms.forward_backward (one number per iter)
  - predicted_total_ms = predicted_adacpsp.total_ms (per-mb sum, no b_step_fb)
  - predicted_total_fb_ms = predicted_adacpsp.total_fb_ms (= total_ms + b_step_fb)
Both prediction columns reflect THAT iter's batch, so they can be compared
iter-by-iter against measured fb_ms — eliminating the batch-mismatch error
that occurs when comparing last-iter prediction against fb_clean (averaged
over a different subset of iters).

Output:
  1) per-cell per-iter MAPE table (excluding warmup iter 0)
  2) per-cell aggregate Δ (measured - predicted) under both prediction modes
  3) pairwise ranking quality over CELL means (each cell's mean over its
     stable iters, both for predictions and measurements)
  4) optional JSON dump

Usage:
  python 30_validate_per_iter.py <results_dir> [--ranks 0]
"""

from __future__ import annotations
import argparse, glob, itertools, json, os, re, sys
from typing import Dict, List, Tuple


CELL_RE_FULL = re.compile(r"(?P<cfg>[a-z0-9]+)_chunks(?P<ch>[a-zA-Z0-9]+)(?:_seq(?P<seq>[a-zA-Z0-9]+))?")


def parse_cell(d: str):
    m = CELL_RE_FULL.match(os.path.basename(d.rstrip("/")))
    if m is None:
        return None
    return (m.group("cfg"), m.group("ch"), m.group("seq") or "-")


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def mean(xs):
    return (sum(xs) / len(xs)) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--ranks", default="0")
    ap.add_argument("--skip-warmup", type=int, default=1,
                    help="Number of leading iters to drop (default 1 for compile)")
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

    print("\n=== Per-iter predicted vs measured (rank 0) ===")
    print("  After skipping warmup iter(s), each row shows measured fb,")
    print("  predicted_per_mb (saturation), predicted_total_fb (with b_step_fb).")
    print("  Δ_sat  = (measured - pred_sat) / measured × 100   (positive ⇒ pred too low)")
    print("  Δ_fb   = (measured - pred_fb)  / measured × 100\n")

    hdr = (f"{'cell':<28} {'iter':>4} {'tokens':>8} "
           f"{'measured':>9} {'pred_sat':>9} {'pred_fb':>8} "
           f"{'Δ_sat':>8} {'Δ_fb':>8}")
    print(hdr); print("-" * len(hdr))

    cell_iters: Dict[str, List[Tuple[int, float, float, float, int]]] = {}
    # cell_iters[cell_label] = [(iter, measured, pred_sat, pred_fb, total_tokens)]

    for d in sorted(glob.glob(os.path.join(e2e, "*_chunks*"))):
        p = parse_cell(d)
        if p is None:
            continue
        cfg, ch, seq = p
        label = f"{cfg}_c{ch}" + (f"_{seq}" if seq != "-" else "")
        for rank in sorted(allowed):
            jp = os.path.join(d, f"rank{rank}.jsonl")
            if not os.path.isfile(jp):
                continue
            recs = load_jsonl(jp)
            tr = [r for r in recs if r.get("phase") == "train_step"]
            if not tr:
                continue
            rows = []
            for i, r in enumerate(tr):
                if i < args.skip_warmup:
                    continue
                fb = float(r.get("timings_ms", {}).get("forward_backward", 0.0))
                pred = r.get("predicted_adacpsp") or {}
                pred_sat = float(pred.get("total_ms", 0.0))
                pred_fb = float(pred.get("total_fb_ms", pred_sat))
                tokens = sum(
                    g.get("tokens", 0)
                    for mb in (pred.get("microbatches") or [])
                    for g in (mb.get("groups") or [])
                )
                rows.append((i, fb, pred_sat, pred_fb, tokens))
            cell_iters[label] = rows
            for i, fb, pred_sat, pred_fb, tokens in rows:
                d_sat = (fb - pred_sat) / max(1.0, fb) * 100
                d_fb = (fb - pred_fb) / max(1.0, fb) * 100
                print(f"{label:<28} {i:>4} {tokens:>8} "
                      f"{fb:>9.0f} {pred_sat:>9.0f} {pred_fb:>8.0f} "
                      f"{d_sat:>+7.1f}% {d_fb:>+7.1f}%")
            print()

    # ------------------------------------------------------------------ #
    # Aggregate: per-cell mean Δ                                          #
    # ------------------------------------------------------------------ #
    print("=== Per-cell aggregate (mean over stable iters) ===")
    print(f"{'cell':<28} {'n_iters':>7} {'mean_fb':>9} {'mean_psat':>10} {'mean_pfb':>9} "
          f"{'|Δ|_sat':>8} {'|Δ|_fb':>8}")
    print("-" * 86)
    cell_means: Dict[str, Tuple[float, float, float]] = {}
    deltas_sat_all: List[float] = []
    deltas_fb_all: List[float] = []
    for label, rows in cell_iters.items():
        if not rows:
            continue
        fbs = [r[1] for r in rows]
        psats = [r[2] for r in rows]
        pfbs = [r[3] for r in rows]
        mfb = mean(fbs); mps = mean(psats); mpf = mean(pfbs)
        cell_means[label] = (mfb, mps, mpf)
        # MAPE per iter
        d_sat_per = [(f - p) / max(1.0, f) * 100 for f, p in zip(fbs, psats)]
        d_fb_per = [(f - p) / max(1.0, f) * 100 for f, p in zip(fbs, pfbs)]
        deltas_sat_all.extend(d_sat_per)
        deltas_fb_all.extend(d_fb_per)
        abs_dsat = mean([abs(x) for x in d_sat_per])
        abs_dfb = mean([abs(x) for x in d_fb_per])
        print(f"{label:<28} {len(rows):>7} {mfb:>9.0f} {mps:>10.0f} {mpf:>9.0f} "
              f"{abs_dsat:>7.1f}% {abs_dfb:>7.1f}%")

    print()
    print(f"  Pooled |Δ|_sat across all iters/cells:   "
          f"mean={mean([abs(x) for x in deltas_sat_all]):.1f}%  "
          f"max={max([abs(x) for x in deltas_sat_all], default=0):.1f}%")
    print(f"  Pooled |Δ|_fb  across all iters/cells:   "
          f"mean={mean([abs(x) for x in deltas_fb_all]):.1f}%  "
          f"max={max([abs(x) for x in deltas_fb_all], default=0):.1f}%")

    # ------------------------------------------------------------------ #
    # Pairwise ranking quality on cell means                              #
    # ------------------------------------------------------------------ #
    print("\n=== Pairwise ranking & speedup quality (cell means) ===")
    cells = list(cell_means.keys())
    total = ok_sat = ok_fb = 0
    err_sat: List[float] = []
    err_fb: List[float] = []
    h3 = (f"{'A':<28} {'B':<28} {'meas_su':>8} "
          f"{'sat_su':>7} {'fb_su':>7} {'err_sat':>8} {'err_fb':>8} {'rank':>8}")
    print(h3); print("-" * len(h3))
    for a, b in itertools.combinations(cells, 2):
        ma, sa, fa = cell_means[a]
        mb, sb, fb = cell_means[b]
        if ma <= 0 or mb <= 0 or sa <= 0 or sb <= 0 or fa <= 0 or fb <= 0:
            continue
        meas_su = max(ma, mb) / min(ma, mb)
        sat_su = max(sa, sb) / min(sa, sb)
        fb_su = max(fa, fb) / min(fa, fb)
        # rank: which is faster in measured vs predicted?
        meas_a_faster = ma < mb
        sat_a_faster = sa < sb
        fb_a_faster = fa < fb
        total += 1
        ok_sat += (meas_a_faster == sat_a_faster)
        ok_fb += (meas_a_faster == fb_a_faster)
        err_sat.append(abs(sat_su - meas_su) / meas_su * 100)
        err_fb.append(abs(fb_su - meas_su) / meas_su * 100)
        flag = "ok" if meas_a_faster == sat_a_faster else "WRONG"
        print(f"{a:<28} {b:<28} {meas_su:>8.3f} "
              f"{sat_su:>7.3f} {fb_su:>7.3f} {err_sat[-1]:>+7.1f}% {err_fb[-1]:>+7.1f}% {flag:>8}")

    if total > 0:
        print()
        print(f"  Pairs:                 {total}")
        print(f"  Rank correct (sat):    {ok_sat}/{total} = {100*ok_sat/total:.1f}%")
        print(f"  Rank correct (fb):     {ok_fb}/{total} = {100*ok_fb/total:.1f}%")
        print(f"  Speedup MAPE (sat):    {mean(err_sat):.1f}%  max={max(err_sat):.1f}%")
        print(f"  Speedup MAPE (fb):     {mean(err_fb):.1f}%  max={max(err_fb):.1f}%")

    if args.json_out:
        out = {
            "schema": "adacpsp_per_iter_validation_v1",
            "source_run": os.path.basename(args.results_dir.rstrip("/")),
            "cell_means": {
                k: {"measured": v[0], "pred_sat": v[1], "pred_fb": v[2]}
                for k, v in cell_means.items()
            },
            "summary": {
                "iter_mape_sat_pct": mean([abs(x) for x in deltas_sat_all]),
                "iter_mape_fb_pct": mean([abs(x) for x in deltas_fb_all]),
                "rank_correctness_sat": (ok_sat / total) if total else 0.0,
                "rank_correctness_fb": (ok_fb / total) if total else 0.0,
                "speedup_mape_sat": mean(err_sat),
                "speedup_mape_fb": mean(err_fb),
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote per-iter validation JSON -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
