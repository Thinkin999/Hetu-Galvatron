"""Fit per-mb overhead (residual_b per sp) from chunks=1 vs chunks=8 data.

The current b_decomp profile sets b_microbatch_ms = 0 for sp=1 and sp=8,
causing 30-50% under-prediction for chunks=8 ring/ulysses cells.

Method: assume cost = predicted(without b) + n_mb × b
  → b_sp = (measured - predicted_without_b) / n_mb

But the comparison is muddier because chunks=1 vs chunks=8 also differ in
attention compute (sum of f(seqs) is much smaller for many small mbs).

Cleaner approach: solve for b_sp such that for both chunks=1 and chunks=8,
prediction matches measurement on average:
  pred(c1) + 1 × b_sp = meas(c1)
  pred(c8) + 8 × b_sp = meas(c8)
  → b_sp = ((meas-pred)_c8 - (meas-pred)_c1) / (8 - 1)
"""
from __future__ import annotations
import argparse, glob, json, os, sys, itertools, copy
from typing import Dict, List, Tuple

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "galvatron/site_package"))

from galvatron.models.varlen_llama_hf.adacpsp_solver import (
    AdaCPSPCostModel, ParallelStrategy,
)


def latest(pattern, d):
    for p in sorted(glob.glob(os.path.join(d, pattern)), reverse=True):
        try:
            with open(p) as f:
                return p, json.load(f)
        except Exception:
            pass
    return None, None


def build_cm():
    cfg = os.path.join(REPO, "galvatron/models/varlen_llama_hf/configs")
    attn_p, _ = latest("profile_validate_qwen2.5-7b_*.json", cfg)
    comm_p, _ = latest("comm_profile_v2_*.json", cfg)
    _, resid = latest("residual_profile_*.json", cfg)
    _, bdec = latest("b_decomp_profile_*.json", cfg)
    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn_p, comm_profile_json=comm_p, cluster_size=16,
        validation_json=attn_p, gpus_per_node=8,
    )
    if resid:
        cm.apply_residual_profile(resid)
    if bdec:
        cm.apply_b_decomp_profile(bdec)
    return cm


def repredict_all(cm, e2e_dir, skip_warmup=2):
    cells = {}
    for d in sorted(glob.glob(os.path.join(e2e_dir, "*_chunks*"))):
        label = os.path.basename(d)
        jp = os.path.join(d, "rank0.jsonl")
        if not os.path.isfile(jp):
            continue
        recs = [json.loads(l) for l in open(jp) if l.strip()]
        train = [r for r in recs if r.get("phase") == "train_step"]
        fbs, preds, n_mbs = [], [], []
        for i, r in enumerate(train):
            if i < skip_warmup:
                continue
            fb = float(r.get("timings_ms", {}).get("forward_backward", 0))
            mbs = (r.get("global_batch") or {}).get("microbatches") or []
            new_total = 0.0
            sp_vals = []
            for mb in mbs:
                mb_max = 0.0
                for g in mb.get("groups", []):
                    s = int(g["tokens"])
                    if s <= 0:
                        continue
                    strat = ParallelStrategy(
                        attn_type=g["attn_type"],
                        parallel_size=int(g.get("parallel_size", g["sp_size"] * g["cp_size"])),
                        sp_size=int(g["sp_size"]),
                        cp_size=int(g["cp_size"]),
                        placement=g.get("placement", "context_first"),
                    )
                    mb_max = max(mb_max, cm.total_time([s], strat))
                    sp = int(g["sp_size"])
                    if sp not in sp_vals:
                        sp_vals.append(sp)
                new_total += mb_max
            if len(mbs) >= 2:
                new_total += float(cm.b_step_fb_ms_for_strategies(sp_vals))
            fbs.append(fb)
            preds.append(new_total)
            n_mbs.append(len(mbs))
        if fbs:
            cells[label] = dict(
                fb=sum(fbs) / len(fbs),
                pred=sum(preds) / len(preds),
                n_mb=sum(n_mbs) / len(n_mbs),
            )
    return cells


def fit_per_mb_overhead(cells):
    """For ring (sp=1) and ulysses8 (sp=8): solve b_sp."""
    fits = {}
    pairs = [
        ("sp=1 (ring/cauto)", "ring8_chunks1", "ring8_chunks8"),
        ("sp=8 (ulysses8)",   "ulysses8_chunks1", "ulysses8_chunks8"),
        ("sp=2 (usp2x4)",     "usp2x4_chunks1", "usp2x4_chunks8"),
    ]
    for label, k_c1, k_c8 in pairs:
        if k_c1 not in cells or k_c8 not in cells:
            continue
        e1 = cells[k_c1]
        e8 = cells[k_c8]
        # (meas-pred) gives the missing cost. Diff across c1/c8 gives extra mbs.
        diff = (e8["fb"] - e8["pred"]) - (e1["fb"] - e1["pred"])
        n_diff = e8["n_mb"] - e1["n_mb"]
        if n_diff > 0:
            b = diff / n_diff
            fits[label] = b
            print(f"  {label}: missing_c1={e1['fb']-e1['pred']:+.0f}, "
                  f"missing_c8={e8['fb']-e8['pred']:+.0f}, "
                  f"delta_mb={n_diff:.0f}  → b_per_mb = {b:+.0f} ms")
    return fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--apply-overrides", action="store_true",
                    help="Apply fitted b_microbatch_ms and re-predict")
    args = ap.parse_args()

    e2e = args.results_dir
    if os.path.basename(e2e.rstrip("/")) != "end2end":
        e2e = os.path.join(e2e, "end2end")

    cm = build_cm()

    print("=== Step 1: baseline reprediction ===")
    cells = repredict_all(cm, e2e)
    for label, e in sorted(cells.items()):
        err = (e["fb"] - e["pred"]) / e["fb"] * 100
        print(f"  {label:<22} meas={e['fb']:>5.0f} pred={e['pred']:>5.0f} "
              f"err={err:+5.1f}%   n_mb_avg={e['n_mb']:.1f}")

    print()
    print("=== Step 2: solve b_microbatch_ms (per-mb constant) ===")
    fits = fit_per_mb_overhead(cells)

    if not args.apply_overrides:
        print()
        print("(Re-run with --apply-overrides to test these values.)")
        return

    # Apply overrides
    print()
    print("=== Step 3: apply overrides & re-predict ===")
    # Override mappings — extract sp values
    sp_overrides = {}
    if "sp=1 (ring/cauto)" in fits:
        sp_overrides[1] = fits["sp=1 (ring/cauto)"]
    if "sp=8 (ulysses8)" in fits:
        sp_overrides[8] = fits["sp=8 (ulysses8)"]
    if "sp=2 (usp2x4)" in fits:
        # b_per_mb is a delta (correction): current b_sp + delta
        current_b_sp2 = cm.residual_b_per_sp.get(2, 0)
        sp_overrides[2] = current_b_sp2 + fits["sp=2 (usp2x4)"]
    print(f"Overrides: {sp_overrides}")

    for sp, b in sp_overrides.items():
        cm.residual_b_per_sp[sp] = float(b)

    cells2 = repredict_all(cm, e2e)
    n = 0; ok = 0; sum_e = 0
    print()
    print(f"{'cell':<22} {'meas':>5} {'pred_old':>9} {'pred_new':>9} {'err_new':>8}")
    print("-" * 60)
    for label in sorted(cells2.keys()):
        e_old = cells[label]
        e_new = cells2[label]
        err_new = (e_new["fb"] - e_new["pred"]) / e_new["fb"] * 100
        print(f"{label:<22} {e_old['fb']:>5.0f} {e_old['pred']:>9.0f} "
              f"{e_new['pred']:>9.0f} {err_new:+7.1f}%")

    # Pairwise
    print()
    print("=== Pairwise speedup MAPE ===")
    labels = list(cells2.keys())
    for a, b in itertools.combinations(labels, 2):
        if min(cells2[a]['fb'], cells2[b]['fb'], cells2[a]['pred'], cells2[b]['pred']) <= 0:
            continue
        meas_su = max(cells2[a]['fb'], cells2[b]['fb']) / min(cells2[a]['fb'], cells2[b]['fb'])
        pred_su = max(cells2[a]['pred'], cells2[b]['pred']) / min(cells2[a]['pred'], cells2[b]['pred'])
        e = abs(pred_su - meas_su) / meas_su * 100
        sum_e += e; n += 1
        ok += ((cells2[a]['fb'] < cells2[b]['fb']) == (cells2[a]['pred'] < cells2[b]['pred']))
    print(f"  MAPE: {sum_e/n:.1f}%   rank: {ok}/{n}  ({ok/n*100:.0f}%)")


if __name__ == "__main__":
    sys.exit(main() or 0)
