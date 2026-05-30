"""Validate the saturation hypothesis and compute differential-quality metrics
for the cost model.

Reads `28_bench_saturation_*.sh` outputs and produces:

  1. Per-cell predicted-vs-measured table, with TWO predictions side-by-side:
       pred_fb       = predicted_total_fb_ms  (= Σ per-mb + b_step_fb)
       pred_per_mb   = predicted_per_mb_total_ms (= Σ per-mb only, saturation mode)

  2. Linearity test (saturation): for each (cfg, seq), fit
         fb_clean(N) = slope·N + intercept    with N = chunks
       Saturation holds  ⇔  intercept / fb_clean(1) << 1   (e.g., < 10%)

  3. Pairwise ranking quality: enumerate all (cell_A, cell_B) pairs at the same
     seq + chunks. For each, compute
         measured_speedup_AB = fb_clean(B) / fb_clean(A)
         predicted_speedup_AB (under both prediction modes)
     Report rank-agreement % and speedup MAPE per mode.

Usage:
  python 29_validate_saturation.py <results_dir>
"""

from __future__ import annotations
import argparse, glob, json, math, os, re, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


# cell dir like "ulysses8_chunks2_seq131k"
CELL_RE = re.compile(
    r"(?P<cfg>[a-z0-9]+)_chunks(?P<ch>[a-zA-Z0-9]+)_seq(?P<seq>[a-zA-Z0-9]+)"
)


def parse_cell(d: str):
    m = CELL_RE.match(os.path.basename(d.rstrip("/")))
    return None if m is None else (m.group("cfg"), m.group("ch"), m.group("seq"))


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
    chunks: int          # numeric (1, 2, 4, 8)
    seq: str             # "65k", "131k"
    rank: int
    n_iters: int
    n_predicted_mbs: int

    fb_clean_ms: float
    fb_steady_ms: float
    fb_median_ms: float
    wall_steady_ms: float

    predicted_per_mb_total_ms: float    # Σ per-mb (saturation mode)
    predicted_total_fb_ms: float        # Σ per-mb + b_step_fb (current model)
    predicted_b_step_fb_ms: float

    measured_external_ms: float
    avg_seq_len: float
    max_seq_len: int


def measure(p, cfg, chunks, seq, rank) -> "CellResult | None":
    recs = load_jsonl(p)
    tr = [r for r in recs if r.get("phase") == "train_step"]
    if not tr:
        return None
    fb = [r["timings_ms"].get("forward_backward", 0.0) for r in tr]
    fb_body = fb[1:] if len(fb) > 1 else fb
    if not fb_body:
        return None
    fb_clean = sum(sorted(fb_body)[:3]) / min(3, len(fb_body))
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
        ext_iter.append(
            t.get("optimizer_step", 0.0) + t.get("grad_clip", 0.0)
            + t.get("zero_grad", 0.0) + t.get("solve_and_dispatch", 0.0)
        )
    ext_med = pct(ext_iter, 0.5) if ext_iter else 0.0

    last = tr[-1]
    pred = last.get("predicted_adacpsp") or {}
    mbs = pred.get("microbatches", []) or []
    n_mbs = len(mbs)
    pred_per_mb = float(pred.get("total_ms", 0.0))
    pred_total_fb = float(pred.get("total_fb_ms", pred_per_mb))
    b_step_fb = float(pred.get("b_step_fb_ms", 0.0))

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
        cfg=cfg, chunks=int(chunks), seq=seq, rank=int(rank),
        n_iters=len(fb), n_predicted_mbs=n_mbs,
        fb_clean_ms=fb_clean, fb_steady_ms=fb_steady,
        fb_median_ms=med, wall_steady_ms=wall_steady,
        predicted_per_mb_total_ms=pred_per_mb,
        predicted_total_fb_ms=pred_total_fb,
        predicted_b_step_fb_ms=b_step_fb,
        measured_external_ms=ext_med,
        avg_seq_len=avg_seq, max_seq_len=int(max_seq),
    )


def linreg(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """OLS slope, intercept, R². Returns (slope, intercept, r2)."""
    if len(xs) < 2:
        return 0.0, ys[0] if ys else 0.0, 0.0
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    yhat = [slope * x + intercept for x in xs]
    ss_res = sum((y - h) ** 2 for y, h in zip(ys, yhat))
    ss_tot = sum((y - sy / n) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--ranks", default="0")
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

    cells: List[CellResult] = []
    for d in sorted(glob.glob(os.path.join(e2e, "*_chunks*_seq*"))):
        p = parse_cell(d)
        if p is None:
            continue
        cfg, ch, seq = p
        for rank in sorted(allowed):
            jp = os.path.join(d, f"rank{rank}.jsonl")
            if not os.path.isfile(jp):
                continue
            m = measure(jp, cfg, ch, seq, rank)
            if m is None:
                continue
            cells.append(m)

    if not cells:
        print("no cells", file=sys.stderr); return 1

    # ------------------------------------------------------------------ #
    # Section 1: Per-cell predicted-vs-measured (both prediction modes)   #
    # ------------------------------------------------------------------ #
    print("\n=== Per-cell predictions (rank 0) ===")
    print("  fb_clean       : mean of 3 lowest fb iters excl iter 0")
    print("  pred_fb        : Σ per-mb + b_step_fb  (current cost model)")
    print("  pred_per_mb    : Σ per-mb only          (saturation assumption)")
    print("  Δ_fb           : (measured - pred_fb)        / measured × 100")
    print("  Δ_sat          : (measured - pred_per_mb)    / measured × 100")
    hdr = (f"{'cfg':>10} {'seq':>5} {'ch':>3} {'mb':>3} {'maxseq':>6} {'avgseq':>7} "
           f"{'fb_clean':>9} {'fb_steady':>10} "
           f"{'pred_fb':>9} {'pred_pmb':>9} {'b_step':>7} "
           f"{'Δ_fb':>7} {'Δ_sat':>7}")
    print(hdr); print("-" * len(hdr))

    cells_r0 = [c for c in cells if c.rank == 0]
    cells_r0.sort(key=lambda x: (x.seq, x.cfg, x.chunks))
    deltas_fb, deltas_sat = [], []
    for c in cells_r0:
        df = (c.fb_clean_ms - c.predicted_total_fb_ms) / max(1.0, c.fb_clean_ms) * 100.0
        ds = (c.fb_clean_ms - c.predicted_per_mb_total_ms) / max(1.0, c.fb_clean_ms) * 100.0
        deltas_fb.append(df); deltas_sat.append(ds)
        print(f"{c.cfg:>10} {c.seq:>5} {c.chunks:>3} {c.n_predicted_mbs:>3} "
              f"{c.max_seq_len:>6} {c.avg_seq_len:>7.0f} "
              f"{c.fb_clean_ms:>9.1f} {c.fb_steady_ms:>10.1f} "
              f"{c.predicted_total_fb_ms:>9.1f} {c.predicted_per_mb_total_ms:>9.1f} "
              f"{c.predicted_b_step_fb_ms:>7.1f} "
              f"{df:>+6.1f}% {ds:>+6.1f}%")

    def mean_abs(xs): return (sum(abs(x) for x in xs) / len(xs)) if xs else 0.0
    print()
    print(f"  mean |Δ_fb |  = {mean_abs(deltas_fb):5.1f}%   max = {max((abs(d) for d in deltas_fb), default=0):5.1f}%")
    print(f"  mean |Δ_sat|  = {mean_abs(deltas_sat):5.1f}%   max = {max((abs(d) for d in deltas_sat), default=0):5.1f}%")
    if mean_abs(deltas_sat) < mean_abs(deltas_fb):
        print("  → saturation prediction is MORE accurate on average.")
    else:
        print("  → b_step_fb-included prediction is more accurate.")

    # ------------------------------------------------------------------ #
    # Section 2: Linearity test  (fb_clean vs chunks)                     #
    # ------------------------------------------------------------------ #
    print("\n=== Linearity / saturation test ===")
    print("  Fit fb_clean(N) = slope·N + intercept for each (cfg, seq)")
    print("  saturation_strict   ⇔  intercept / slope     < 0.10")
    print("  saturation_relaxed  ⇔  intercept / fb(N=1)   < 0.20")
    h2 = (f"{'cfg':>10} {'seq':>5} {'n_pts':>5} {'fb(lo)':>8} {'fb(hi)':>8} "
          f"{'ratio':>6} {'slope':>8} {'intercept':>10} {'R2':>5} {'sat?':>5}")
    print(h2); print("-" * len(h2))
    linreg_rows = []
    by_group: Dict[Tuple[str, str], List[CellResult]] = {}
    for c in cells_r0:
        by_group.setdefault((c.cfg, c.seq), []).append(c)
    for (cfg, seq), group in sorted(by_group.items()):
        group_sorted = sorted(group, key=lambda x: x.chunks)
        xs = [c.chunks for c in group_sorted]
        ys = [c.fb_clean_ms for c in group_sorted]
        if len(xs) < 2:
            continue
        slope, intercept, r2 = linreg(xs, ys)
        # fb at smallest and largest chunks we measured (typically 2 and 16)
        fb_lo = ys[0]; ch_lo = xs[0]
        fb_hi = ys[-1]; ch_hi = xs[-1]
        ratio = fb_hi / max(1.0, fb_lo)
        ratio_ideal = ch_hi / ch_lo            # ideal if perfectly linear with 0 intercept
        sat_strict = abs(intercept) / max(1.0, slope) < 0.10
        sat_relax = abs(intercept) / max(1.0, fb_lo) < 0.20
        flag = ("yes" if sat_strict else ("loose" if sat_relax else "no"))
        print(f"{cfg:>10} {seq:>5} {len(xs):>5} {fb_lo:>8.0f} {fb_hi:>8.0f} "
              f"{ratio:>6.2f} {slope:>8.1f} {intercept:>+9.1f} {r2:>5.3f} {flag:>5}")
        linreg_rows.append({
            "cfg": cfg, "seq": seq,
            "ch_lo": ch_lo, "ch_hi": ch_hi,
            "fb_lo": fb_lo, "fb_hi": fb_hi, "ratio": ratio, "ratio_ideal": ratio_ideal,
            "slope_ms": slope, "intercept_ms": intercept, "r2": r2,
            "saturation_strict": sat_strict,
            "saturation_relaxed": sat_relax,
        })

    # ------------------------------------------------------------------ #
    # Section 3: Pairwise speedup & ranking quality                       #
    # ------------------------------------------------------------------ #
    print("\n=== Pairwise speedup / ranking quality (within same seq+chunks) ===")
    print("  For each pair (A, B) at same (seq, chunks):")
    print("    measured_speedup = fb_clean(B) / fb_clean(A)")
    print("    predicted_speedup = pred_*(B) / pred_*(A)")
    print("  pred_fb mode  → uses predicted_total_fb_ms")
    print("  pred_pmb mode → uses predicted_per_mb_total_ms (saturation)")

    by_grouping: Dict[Tuple[str, int], List[CellResult]] = {}
    for c in cells_r0:
        by_grouping.setdefault((c.seq, c.chunks), []).append(c)

    rank_correct_fb = rank_correct_sat = total_pairs = 0
    speedup_err_fb, speedup_err_sat = [], []

    h3 = (f"{'seq':>5} {'ch':>3} {'A':>10} {'B':>10} "
          f"{'meas_su':>8} {'su_fb':>7} {'su_pmb':>7} "
          f"{'err_fb':>8} {'err_pmb':>8} {'rank?':>10}")
    print(h3); print("-" * len(h3))
    pairs_dump = []
    for (seq, ch), group in sorted(by_grouping.items()):
        # All ordered pairs (A, B) where A != B.
        for i, A in enumerate(group):
            for j, B in enumerate(group):
                if i == j:
                    continue
                meas_su = B.fb_clean_ms / max(1.0, A.fb_clean_ms)
                pred_fb_su = B.predicted_total_fb_ms / max(1.0, A.predicted_total_fb_ms)
                pred_pmb_su = B.predicted_per_mb_total_ms / max(1.0, A.predicted_per_mb_total_ms)
                err_fb = (pred_fb_su - meas_su) / max(0.001, meas_su) * 100.0
                err_pmb = (pred_pmb_su - meas_su) / max(0.001, meas_su) * 100.0
                # rank correctness: sign of (meas-1) matches sign of (pred-1)
                same_fb = ((meas_su > 1.0) == (pred_fb_su > 1.0))
                same_pmb = ((meas_su > 1.0) == (pred_pmb_su > 1.0))
                rank_correct_fb += int(same_fb)
                rank_correct_sat += int(same_pmb)
                speedup_err_fb.append(err_fb)
                speedup_err_sat.append(err_pmb)
                total_pairs += 1
                flag = ("ok" if same_fb and same_pmb else
                        ("only_fb" if same_fb else
                         ("only_sat" if same_pmb else "both_wrong")))
                print(f"{seq:>5} {ch:>3} {A.cfg:>10} {B.cfg:>10} "
                      f"{meas_su:>8.3f} {pred_fb_su:>7.3f} {pred_pmb_su:>7.3f} "
                      f"{err_fb:>+7.1f}% {err_pmb:>+7.1f}% {flag:>10}")
                pairs_dump.append({
                    "seq": seq, "chunks": ch,
                    "A": A.cfg, "B": B.cfg,
                    "measured_speedup": meas_su,
                    "pred_fb_speedup": pred_fb_su,
                    "pred_pmb_speedup": pred_pmb_su,
                    "err_fb_pct": err_fb, "err_pmb_pct": err_pmb,
                    "rank_correct_fb": same_fb,
                    "rank_correct_pmb": same_pmb,
                })

    if total_pairs > 0:
        print()
        print(f"  Total pairs           : {total_pairs}")
        print(f"  Rank correctness (fb) : {rank_correct_fb}/{total_pairs} = "
              f"{100.0 * rank_correct_fb / total_pairs:.1f}%")
        print(f"  Rank correctness (sat): {rank_correct_sat}/{total_pairs} = "
              f"{100.0 * rank_correct_sat / total_pairs:.1f}%")
        print(f"  Speedup MAPE (fb)     : {mean_abs(speedup_err_fb):.1f}%   "
              f"max = {max(abs(e) for e in speedup_err_fb):.1f}%")
        print(f"  Speedup MAPE (sat)    : {mean_abs(speedup_err_sat):.1f}%   "
              f"max = {max(abs(e) for e in speedup_err_sat):.1f}%")

    if args.json_out:
        out = {
            "schema": "adacpsp_saturation_validation_v1",
            "source_run": os.path.basename(args.results_dir.rstrip("/")),
            "per_cell": [
                {
                    "cfg": c.cfg, "chunks": c.chunks, "seq": c.seq,
                    "fb_clean_ms": c.fb_clean_ms, "fb_steady_ms": c.fb_steady_ms,
                    "pred_fb_ms": c.predicted_total_fb_ms,
                    "pred_per_mb_ms": c.predicted_per_mb_total_ms,
                    "b_step_fb_ms": c.predicted_b_step_fb_ms,
                    "n_microbatches": c.n_predicted_mbs,
                    "max_seq_len": c.max_seq_len,
                    "avg_seq_len": c.avg_seq_len,
                }
                for c in cells_r0
            ],
            "linregs": linreg_rows,
            "pairs": pairs_dump,
            "summary": {
                "mean_abs_delta_fb_pct": mean_abs(deltas_fb),
                "mean_abs_delta_sat_pct": mean_abs(deltas_sat),
                "rank_correctness_fb": (rank_correct_fb / total_pairs) if total_pairs else 0.0,
                "rank_correctness_sat": (rank_correct_sat / total_pairs) if total_pairs else 0.0,
                "speedup_mape_fb": mean_abs(speedup_err_fb),
                "speedup_mape_sat": mean_abs(speedup_err_sat),
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote saturation-validation JSON -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
