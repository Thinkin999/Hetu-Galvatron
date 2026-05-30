"""
Global calibration: jointly tune (compute_correction_factor, bwd_fwd_ratio_long_seq,
a2a_constant_per_op, ring_step_overhead, usp_layer_overhead_*) against ALL the
ZeRO-2 + precreate steady-state benchmark cells.

Approach:
  1. Read the end2end JSONL for every cell × chunk combo in a benchmark dir.
  2. Apply MAD-based outlier filtering on each cell's fb_ms list.
  3. Reload the cost model from current configs, BUT expose its key constants
     as tunable hyperparameters.
  4. Use scipy.optimize.minimize or a simple grid search to find the best
     constants that minimize MAPE across all cells, weighted by N_iter.
  5. Bonus objective: pairwise speedup ranking accuracy (ring_c8 / ulysses_c8
     etc).

Usage:
  python 40_global_calibration.py <results_dir> [--out new_constants.json]
"""
from __future__ import annotations
import argparse, glob, json, os, sys, copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
sys.path.insert(0, REPO)

from galvatron.models.varlen_llama_hf.adacpsp_solver import (
    AdaCPSPCostModel, ParallelStrategy,
)


def latest_pat(pat, d):
    cs = sorted(glob.glob(os.path.join(d, pat)), reverse=True)
    return cs[0] if cs else None


def build_base_cm() -> AdaCPSPCostModel:
    cfg = os.path.join(REPO, "galvatron/models/varlen_llama_hf/configs")
    attn_p = latest_pat("profile_validate_*.json", cfg)
    comm_p = latest_pat("comm_profile_*.json", cfg)
    resid_p = latest_pat("residual_profile_*.json", cfg)
    bdec_p = latest_pat("b_decomp_profile_*.json", cfg)
    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn_p, comm_profile_json=comm_p,
        cluster_size=16, validation_json=attn_p, gpus_per_node=8,
    )
    if resid_p:
        with open(resid_p) as f:
            cm.apply_residual_profile(json.load(f))
    if bdec_p:
        with open(bdec_p) as f:
            cm.apply_b_decomp_profile(json.load(f))
    print(f"# base profiles:")
    print(f"  attn={os.path.basename(attn_p) if attn_p else 'None'}")
    print(f"  comm={os.path.basename(comm_p) if comm_p else 'None'}")
    print(f"  resid={os.path.basename(resid_p) if resid_p else 'None'}")
    print(f"  bdec={os.path.basename(bdec_p) if bdec_p else 'None'}")
    return cm


def load_clean_meas(results_dir: str, skip: int = 5) -> Dict[str, Dict]:
    """For every cell, return {'fbs_clean': [...], 'records_clean': [...]}."""
    e2e = Path(results_dir)
    if e2e.name != "end2end":
        e2e = e2e / "end2end"
    out = {}
    for cell_dir in sorted(e2e.glob("*_chunks*")):
        jp = cell_dir / "rank0.jsonl"
        if not jp.exists():
            continue
        recs = [json.loads(l) for l in open(jp) if l.strip()]
        train = [r for r in recs if r.get("phase") == "train_step"]
        if len(train) <= skip:
            continue
        train = train[skip:]
        fbs = [float(r["timings_ms"]["forward_backward"]) for r in train]
        sorted_fbs = sorted(fbs)
        med = sorted_fbs[len(sorted_fbs) // 2]
        mad = sorted([abs(x - med) for x in fbs])[len(fbs) // 2]
        thr = med + 3 * (mad if mad > 0 else med * 0.3)
        clean_pairs = [(r, f) for r, f in zip(train, fbs) if f <= thr]
        out[cell_dir.name] = dict(
            records=[r for r, _ in clean_pairs],
            fbs=[f for _, f in clean_pairs],
            n_total=len(fbs),
            n_clean=len(clean_pairs),
        )
    return out


def predict_with_overrides(cm: AdaCPSPCostModel, mbs: List[dict],
                           overrides: Optional[Dict] = None) -> float:
    """Re-predict total_fb_ms for one train_step's microbatch list, with
    arbitrary cost-model constant overrides.

    overrides keys (any subset of these):
      bwd_fwd_ratio, ulysses_a2a_overhead_ms, usp_a2a_overhead_extra_ms,
      ring_step_overhead_ms, usp_layer_overhead_base_ms,
      usp_layer_overhead_per_sp_ms, compute_correction_factor.

    `compute_correction_factor` is applied as a multiplicative scalar on the
    raw piecewise eval -- a separate calibration knob from the existing
    `compute_correction` table lookup. We piggy-back on `compute_correction`
    being a list of (x, corr) points; if not present we set a uniform corr.
    """
    overrides = overrides or {}

    # Save and override
    saved = {}
    for k, v in overrides.items():
        if k == "compute_correction_factor":
            saved[k] = cm.compute_correction
            cm.compute_correction = [(0.0, v), (1e9, v)]
        else:
            if hasattr(cm, k):
                saved[k] = getattr(cm, k)
                setattr(cm, k, v)

    # Predict
    total_fb = 0.0
    sp_values_set = []
    try:
        for mb in mbs:
            mb_max = 0.0
            for g in mb.get("groups", []):
                ps = int(g.get("parallel_size",
                               int(g["sp_size"]) * int(g["cp_size"])))
                sp = int(g["sp_size"])
                cp = int(g["cp_size"])
                # We don't have individual seq lengths in the recorded
                # microbatch summary, but `tokens` and `num_sequences` are.
                # Approximate equal-length packing: N seqs of T/N tokens each.
                tokens = int(g["tokens"])
                nseq = max(1, int(g.get("num_sequences", 1)))
                # Equal-length seqs of size tokens/nseq each
                per_seq = max(1, tokens // nseq)
                seqlens = [per_seq] * nseq
                # If tokens not perfectly divisible, last seq has remainder
                remainder = tokens - per_seq * nseq
                if remainder > 0:
                    seqlens[-1] += remainder
                strat = ParallelStrategy(
                    attn_type=g["attn_type"],
                    parallel_size=ps, sp_size=sp, cp_size=cp,
                    placement=g.get("placement", "context_first"),
                )
                t = cm.total_time(seqlens, strat)
                if t > mb_max:
                    mb_max = t
                if sp not in sp_values_set:
                    sp_values_set.append(sp)
            total_fb += mb_max
        # Add b_step_fb if multi-mb
        if len(mbs) >= 2 and hasattr(cm, "b_step_fb_ms_for_strategies"):
            b_step = float(cm.b_step_fb_ms_for_strategies(sp_values_set))
            total_fb += b_step
    finally:
        # Restore
        for k, v in saved.items():
            if k == "compute_correction_factor":
                cm.compute_correction = v
            else:
                setattr(cm, k, v)

    return total_fb


def evaluate_overrides(cm: AdaCPSPCostModel, cells: Dict[str, dict],
                       overrides: Optional[Dict] = None) -> Dict:
    """Compute per-cell MAPE and global summary under given overrides."""
    per_cell = {}
    abs_errs = []
    signed_errs = []
    for name, data in cells.items():
        cell_errs = []
        for r, fb in zip(data["records"], data["fbs"]):
            mbs = r.get("global_batch", {}).get("microbatches", [])
            pred = predict_with_overrides(cm, mbs, overrides=overrides)
            err = (fb - pred) / max(1.0, fb)
            cell_errs.append(err)
            abs_errs.append(abs(err))
            signed_errs.append(err)
        ape = 100 * sum(abs(e) for e in cell_errs) / len(cell_errs)
        ame = 100 * sum(cell_errs) / len(cell_errs)
        per_cell[name] = dict(
            mape=ape, mean_signed=ame, n=len(cell_errs),
            meas_avg=sum(data["fbs"]) / len(data["fbs"]),
        )
    overall_mape = 100 * sum(abs_errs) / len(abs_errs)
    overall_mse = 100 * sum(signed_errs) / len(signed_errs)
    return dict(per_cell=per_cell, mape=overall_mape, mean_signed=overall_mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cm = build_base_cm()
    print()
    cells = load_clean_meas(args.results_dir, skip=args.skip)
    if not cells:
        print(f"No cells found in {args.results_dir}", file=sys.stderr)
        return 1
    print(f"# Cells loaded: {list(cells.keys())}")
    for name, d in cells.items():
        print(f"  {name}: n_clean={d['n_clean']}/{d['n_total']}, "
              f"meas_avg={sum(d['fbs'])/len(d['fbs']):.0f}ms")
    print()

    # Baseline
    print("=== BASELINE (current cost model defaults) ===")
    base = evaluate_overrides(cm, cells, overrides=None)
    print(f"Global MAPE: {base['mape']:.1f}%  mean_signed: {base['mean_signed']:+.1f}%")
    for name, p in base["per_cell"].items():
        print(f"  {name:<22}  meas={p['meas_avg']:>6.0f}  MAPE={p['mape']:>5.1f}%  "
              f"signed={p['mean_signed']:>+6.1f}%")
    print()

    # Sweep on key constants
    print("=== SWEEPS ===")
    knobs = [
        ("bwd_fwd_ratio", [2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]),
        ("ulysses_a2a_overhead_ms", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
        ("usp_a2a_overhead_extra_ms", [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]),
        ("ring_step_overhead_ms", [0.20, 0.35, 0.50, 0.65, 0.80, 1.0]),
        ("usp_layer_overhead_base_ms", [0.0, 0.5, 1.0, 1.5, 2.0]),
        ("usp_layer_overhead_per_sp_ms", [0.0, 0.40, 0.80, 1.20, 1.60]),
        ("compute_correction_factor", [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20]),
        ("overlap_slowdown", [1.0, 1.05, 1.10, 1.15, 1.20]),
    ]
    print(f"{'knob':<28} {'value':>8}  {'MAPE':>6}  {'signed':>8}  {'per-cell signed errs'}")
    for knob_name, vals in knobs:
        for v in vals:
            r = evaluate_overrides(cm, cells, overrides={knob_name: v})
            per_cell_str = " ".join(
                f"{name.split('_')[0][:3]}{name.split('_')[1][-1]}:{p['mean_signed']:+.0f}"
                for name, p in r['per_cell'].items()
            )
            print(f"{knob_name:<28} {v:>8.3f}  {r['mape']:>5.1f}%  "
                  f"{r['mean_signed']:>+7.1f}%  {per_cell_str}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
