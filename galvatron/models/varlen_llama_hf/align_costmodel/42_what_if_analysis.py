"""
What-if analysis for solver optimality.

For each historical iter (collected from a benchmark run), evaluate the cost
model's prediction for EVERY candidate strategy applied to the SAME seqs.
Then identify which strategy the cost model believes is fastest, and compare
to the strategy the solver actually picked.

This helps surface cases where solver decisions diverge from min-cost
strategies due to:
  - Per-group cost model bias (predicts wrong relative ordering)
  - Solver constraints (memory, layout) that exclude optimal strategies
  - ILP suboptimality

Outputs:
  Per cauto iter, the predicted top-3 strategies (and cauto's actual pick)
  Aggregated: how often does cauto pick the cost-model-optimal strategy?
  Cross-benchmark: how do the FORCED-cell measured times compare to the
                   cost model's optimal strategy choice for those iters?

Usage:
  python 42_what_if_analysis.py <results_dir>
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
# Import adacpsp_solver directly without triggering the varlen_llama_hf package
# __init__.py (which imports megatron and breaks in this env).
import importlib.util
_solver_path = os.path.join(REPO, "galvatron/models/varlen_llama_hf/adacpsp_solver.py")
spec = importlib.util.spec_from_file_location("adacpsp_solver", _solver_path)
_solver = importlib.util.module_from_spec(spec)
sys.modules["adacpsp_solver"] = _solver  # required for dataclass introspection
spec.loader.exec_module(_solver)
AdaCPSPCostModel = _solver.AdaCPSPCostModel
ParallelStrategy = _solver.ParallelStrategy


def build_cm() -> AdaCPSPCostModel:
    cfg = os.path.join(REPO, "galvatron/models/varlen_llama_hf/configs")
    attn_p = sorted(glob.glob(os.path.join(cfg, "profile_validate_*.json")), reverse=True)[0]
    comm_p = sorted(glob.glob(os.path.join(cfg, "comm_profile_*.json")), reverse=True)[0]
    resid_p = sorted(glob.glob(os.path.join(cfg, "residual_profile_*.json")), reverse=True)
    bdec_p = sorted(glob.glob(os.path.join(cfg, "b_decomp_profile_*.json")), reverse=True)
    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn_p, comm_profile_json=comm_p,
        cluster_size=16, validation_json=attn_p, gpus_per_node=8,
    )
    if resid_p:
        with open(resid_p[0]) as f:
            cm.apply_residual_profile(json.load(f))
    if bdec_p:
        with open(bdec_p[0]) as f:
            cm.apply_b_decomp_profile(json.load(f))
    print(f"# cm loaded: bdec={os.path.basename(bdec_p[0])}")
    return cm


# Strategy candidates for world_size=16
def enumerate_candidates(world_size: int = 16) -> List[Tuple[str, int, int, str]]:
    """Return [(attn_type, sp, cp, placement)] tuples we want to evaluate."""
    out = []
    # Ulysses sp ∈ {1, 2, 4, 8, 16}
    for sp in [1, 2, 4, 8, 16]:
        out.append(("ulysses", sp, 1, "context_first"))
    # Ring cp ∈ {1, 2, 4, 8, 16}
    for cp in [1, 2, 4, 8, 16]:
        out.append(("ring", 1, cp, "context_first"))
    # USP sp×cp combos (sp*cp ≤ world_size)
    for sp in [2, 4]:
        for cp in [2, 4, 8]:
            if sp * cp > world_size:
                continue
            for placement in ("context_first", "head_first"):
                out.append(("usp", sp, cp, placement))
    return out


def predict_all_in_one_mb(cm: AdaCPSPCostModel, seqs: List[int],
                          parallel_size: int = 16) -> List[Dict]:
    """For all seqs packed in 1 mb across 1 group of parallel_size GPUs,
    predict each candidate strategy's total step time."""
    cands = enumerate_candidates(parallel_size)
    rows = []
    for attn, sp, cp, place in cands:
        ps = sp * cp
        if ps != parallel_size:
            continue
        strat = ParallelStrategy(
            attn_type=attn, parallel_size=ps,
            sp_size=sp, cp_size=cp, placement=place,
        )
        try:
            per_group = cm.total_time(seqs, strat)
            # One mb, one group → step_fb = per_group + b_step_fb
            step_fb = per_group + cm.b_step_fb_ms_for_strategies([sp if attn in ("ulysses","usp") else 1])
        except Exception as exc:
            continue
        rows.append(dict(
            attn=attn, sp=sp, cp=cp, place=place,
            per_group_ms=per_group, step_fb_ms=step_fb,
        ))
    return sorted(rows, key=lambda r: r["step_fb_ms"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--top", type=int, default=5,
                    help="Show top-N candidate strategies per iter")
    args = ap.parse_args()

    cm = build_cm()

    e2e = Path(args.results_dir)
    if e2e.name != "end2end":
        cand = e2e / "end2end"
        if cand.is_dir():
            e2e = cand

    # Look for adacpsp_chunksauto cell
    cauto_dir = e2e / "adacpsp_chunksauto"
    if not cauto_dir.exists():
        print(f"adacpsp_chunksauto not found in {e2e}", file=sys.stderr)
        return 1
    recs = [json.loads(l) for l in open(cauto_dir / "rank0.jsonl") if l.strip()]
    train = [r for r in recs if r.get("phase") == "train_step"]

    # For each iter, we need the individual seq lengths in the global batch.
    # Unfortunately the JSONL summarizes per-group as (tokens, max_sequence,
    # num_sequences), not full seq list. We must approximate by reading from
    # the verbose dispatch log lines.
    # Fallback: use equal-length packing approximation.
    print(f"\n# Note: using equal-length packing approximation (tokens // num_seq)")
    print(f"# Cells found in {e2e}:")
    for d in sorted(e2e.glob("*_chunks*")):
        print(f"   {d.name}")
    print()

    # Get measured wallclock for each FORCED cell (for cross-reference)
    forced_meas = {}
    for cell_dir in e2e.glob("*_chunks*"):
        if cell_dir.name == "adacpsp_chunksauto":
            continue
        recs2 = [json.loads(l) for l in open(cell_dir / "rank0.jsonl") if l.strip()]
        tr = [r for r in recs2 if r.get("phase") == "train_step"]
        if not tr: continue
        fbs = [float(r["timings_ms"]["forward_backward"]) for r in tr[args.skip:]]
        sorted_fbs = sorted(fbs); med = sorted_fbs[len(sorted_fbs)//2]
        mad = sorted([abs(x-med) for x in fbs])[len(fbs)//2]
        thr = med + 3*(mad if mad>0 else med*0.3)
        clean = [x for x in fbs if x <= thr]
        forced_meas[cell_dir.name] = sum(clean) / len(clean) if clean else None

    print(f"=== Forced-cell measurements ===")
    for k, v in sorted(forced_meas.items()):
        print(f"  {k:<22}  meas_clean={v:>7.0f}ms")
    print()

    # For each cauto iter, reconstruct seq lengths approximately
    cauto_correctness = []  # list of (cauto_pick_pred, model_optimal_pred)
    print(f"=== Per-iter what-if analysis ({len(train)} iters, skip first {args.skip}) ===")
    for i, r in enumerate(train[args.skip:], start=args.skip):
        fb = float(r["timings_ms"]["forward_backward"])
        op = r.get("predicted_adacpsp", {})
        if not isinstance(op, dict): continue

        # Reconstruct global seqs (approximate: per group → equal-length seqs)
        all_seqs = []
        for mb in op.get("microbatches", []):
            for g in mb.get("groups", []):
                tok = int(g["tokens"])
                n = max(1, int(g.get("num_sequences", 1)))
                per = max(1, tok // n)
                seqs = [per] * n
                rem = tok - per*n
                if rem > 0: seqs[-1] += rem
                all_seqs.extend(seqs)
        if not all_seqs: continue

        # Sort descending (Best Fit Decreasing style) for readability
        all_seqs_sorted = sorted(all_seqs, reverse=True)

        # Candidate predictions (all seqs in 1 mb, 1 group of size 16)
        cands = predict_all_in_one_mb(cm, all_seqs_sorted, parallel_size=16)

        # cauto's actual pick: total_fb from solver
        cauto_pred = float(op.get("total_fb_ms", 0))

        # Print: cauto fb, model's top-N strategies for "1mb-all" partition
        top = cands[:args.top]
        opt_pred = top[0]["step_fb_ms"] if top else 0
        cauto_correctness.append((cauto_pred, opt_pred, fb, all_seqs_sorted))

        print(f"\niter {i}: meas={fb:.0f}ms, cauto_pred={cauto_pred:.0f}ms, "
              f"n_seqs={len(all_seqs)}, max_seq={max(all_seqs)}")
        print(f"  Top-{args.top} candidates (all seqs in 1 mb of 16 GPUs):")
        for c in top:
            print(f"    {c['attn']:7s} sp={c['sp']:2d} cp={c['cp']:2d} "
                  f"{c['place']:<14s}  per_group={c['per_group_ms']:>7.0f} "
                  f"step={c['step_fb_ms']:>7.0f}ms")

    # Aggregate: how often does cauto match the model's optimal?
    if cauto_correctness:
        print(f"\n\n=== Summary across {len(cauto_correctness)} iters ===")
        cauto_avg = sum(c[0] for c in cauto_correctness)/len(cauto_correctness)
        opt_avg = sum(c[1] for c in cauto_correctness)/len(cauto_correctness)
        meas_avg = sum(c[2] for c in cauto_correctness)/len(cauto_correctness)
        print(f"  Avg measured fb: {meas_avg:.0f}ms")
        print(f"  Avg cauto pred:  {cauto_avg:.0f}ms  (gap vs meas: {(meas_avg-cauto_avg)/meas_avg*100:+.1f}%)")
        print(f"  Avg model-opt pred (1mb-all): {opt_avg:.0f}ms")
        print(f"  Cauto pred / model-opt pred: {cauto_avg/opt_avg*100:.0f}%")
        print(f"  -> If model-opt is correct, cauto over-spends by {(cauto_avg/opt_avg-1)*100:.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
