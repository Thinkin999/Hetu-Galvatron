"""Offline re-prediction with FIXED cost model on saved benchmark data.

Reads each iteration's saved per-group breakdown (attn_type, sp/cp, tokens)
from rank0.jsonl in an end2end results directory, then re-runs the (now fixed)
cost model to produce a new predicted total_ms. Compares against measured
forward_backward time.

Designed for the github multi-mb benchmark where every group has exactly one
packed sequence (num_sequences=1, max_sequence == tokens), making single-seq
prediction exact.

Usage:
  python 32_offline_repredict.py /path/to/end2end --skip-warmup 2
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from typing import Dict, List, Tuple

# Make solver module importable
REPO_ROOT = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "galvatron/site_package"))

from galvatron.models.varlen_llama_hf.adacpsp_solver import (
    AdaCPSPCostModel, ParallelStrategy,
)


def latest_profile(pattern: str, configs_dir: str):
    paths = sorted(glob.glob(os.path.join(configs_dir, pattern)), reverse=True)
    for p in paths:
        try:
            with open(p) as f:
                return p, json.load(f)
        except Exception:
            pass
    return None, None


def build_costmodel(world_size: int = 16):
    configs_dir = os.path.join(REPO_ROOT,
                               "galvatron/models/varlen_llama_hf/configs")
    attn_path, _ = latest_profile("profile_validate_*.json", configs_dir)
    comm_path, _ = latest_profile("comm_profile_*.json", configs_dir)
    _, resid = latest_profile("residual_profile_*.json", configs_dir)
    _, bdec = latest_profile("b_decomp_profile_*.json", configs_dir)

    # Also pass validation_json so compute_correction is loaded — without it,
    # the cost model's per-seq compute predictions can differ by 2-5x from
    # what live training actually used.
    validation_json = attn_path  # validation lives in the same file
    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn_path,
        comm_profile_json=comm_path,
        cluster_size=world_size,
        validation_json=validation_json,
        gpus_per_node=8,
    )
    if resid is not None:
        cm.apply_residual_profile(resid)
    if bdec is not None:
        cm.apply_b_decomp_profile(bdec)
    print(f"loaded attn={os.path.basename(attn_path)} comm={os.path.basename(comm_path)} "
          f"b_decomp={'yes' if bdec else 'no'}", file=sys.stderr)
    return cm


def predict_group_ms(cm: AdaCPSPCostModel, g: dict) -> float:
    """Re-predict ms for one (attn_type, sp_size, cp_size, tokens) group."""
    seqlen = int(g["tokens"])
    if seqlen <= 0:
        return 0.0
    strat = ParallelStrategy(
        attn_type=g["attn_type"],
        parallel_size=int(g.get("parallel_size", g["sp_size"] * g["cp_size"])),
        sp_size=int(g["sp_size"]),
        cp_size=int(g["cp_size"]),
        placement=g.get("placement", "context_first"),
    )
    seqlens = [seqlen]
    # Standard end-to-end cost: compute + comm + residual.
    return cm.total_time(seqlens, strat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--skip-warmup", type=int, default=2,
                    help="iters to drop at the start (warmup/compile)")
    args = ap.parse_args()

    cm = build_costmodel()

    e2e = args.results_dir
    if os.path.basename(e2e.rstrip("/")) != "end2end":
        cand = os.path.join(e2e, "end2end")
        if os.path.isdir(cand):
            e2e = cand
    if not os.path.isdir(e2e):
        print(f"missing: {e2e}", file=sys.stderr); return 1

    print()
    hdr = (f"{'cell':<22} {'iter':>4} {'tokens':>8} "
           f"{'measured':>9} {'pred_old':>9} {'pred_new':>9} "
           f"{'old_err':>8} {'new_err':>8}")
    print(hdr); print("-" * len(hdr))

    per_cell: Dict[str, List[Tuple]] = {}
    for d in sorted(glob.glob(os.path.join(e2e, "*_chunks*"))):
        label = os.path.basename(d.rstrip("/"))
        jp = os.path.join(d, "rank0.jsonl")
        if not os.path.isfile(jp):
            continue

        rows = []
        with open(jp) as f:
            recs = [json.loads(l) for l in f if l.strip()]
        train_recs = [r for r in recs if r.get("phase") == "train_step"]

        for i, r in enumerate(train_recs):
            if i < args.skip_warmup:
                continue
            fb = float(r.get("timings_ms", {}).get("forward_backward", 0.0))
            pred_old = float(r.get("predicted_adacpsp", {}).get("total_ms", 0.0))

            # Build new prediction by re-running cost model on every group
            gb = r.get("global_batch") or {}
            mbs = gb.get("microbatches") or []
            tokens_total = 0
            new_total = 0.0
            for mb in mbs:
                # Per-mb time = max over groups (groups run in parallel
                # across distinct GPU slices), same convention as solver.
                mb_max = 0.0
                for g in mb.get("groups", []):
                    tokens_total += int(g.get("tokens", 0))
                    pred_g = predict_group_ms(cm, g)
                    mb_max = max(mb_max, pred_g)
                new_total += mb_max
            # Charge b_step_fb if chunks>=2 (matches training-time logic
            # in _predict_adacpsp_ms).
            if len(mbs) >= 2 and hasattr(cm, "b_step_fb_ms_for_strategies"):
                sp_values = []
                for mb in mbs:
                    for g in mb.get("groups", []):
                        sp = int(g.get("sp_size", 1))
                        if sp not in sp_values:
                            sp_values.append(sp)
                new_total += float(cm.b_step_fb_ms_for_strategies(sp_values))

            old_err = (fb - pred_old) / max(1.0, fb) * 100
            new_err = (fb - new_total) / max(1.0, fb) * 100
            rows.append((i, tokens_total, fb, pred_old, new_total,
                         old_err, new_err))

        for row in rows:
            i, tk, fb, p_old, p_new, e_old, e_new = row
            print(f"{label:<22} {i:>4} {tk:>8} "
                  f"{fb:>9.0f} {p_old:>9.0f} {p_new:>9.0f} "
                  f"{e_old:>+7.1f}% {e_new:>+7.1f}%")
        per_cell[label] = rows
        if rows:
            print()

    # Aggregate
    print("=" * 90)
    print("Aggregate (mean over stable iters)")
    h = f"{'cell':<22} {'n':>3} {'mean_meas':>10} {'mean_old':>9} {'mean_new':>9} {'|err|_old':>10} {'|err|_new':>10}"
    print(h); print("-" * len(h))
    for label, rows in per_cell.items():
        if not rows: continue
        mfb = sum(r[2] for r in rows) / len(rows)
        mold = sum(r[3] for r in rows) / len(rows)
        mnew = sum(r[4] for r in rows) / len(rows)
        eold = sum(abs(r[5]) for r in rows) / len(rows)
        enew = sum(abs(r[6]) for r in rows) / len(rows)
        print(f"{label:<22} {len(rows):>3} {mfb:>10.0f} {mold:>9.0f} {mnew:>9.0f} "
              f"{eold:>9.1f}% {enew:>9.1f}%")

    # Pairwise speedup ranking
    print()
    print("=" * 90)
    print("Pairwise speedup (cell means)")
    cell_means = {l: (sum(r[2] for r in rs)/len(rs),
                       sum(r[3] for r in rs)/len(rs),
                       sum(r[4] for r in rs)/len(rs))
                  for l, rs in per_cell.items() if rs}
    cells = list(cell_means.keys())
    h2 = f"{'A':<22} {'B':<22} {'meas':>8} {'old':>7} {'new':>7} {'errO':>7} {'errN':>7} {'rankO':>6} {'rankN':>6}"
    print(h2); print("-" * len(h2))
    import itertools
    n_pairs = ok_old = ok_new = 0
    sum_errO = sum_errN = 0.0
    for a, b in itertools.combinations(cells, 2):
        ma, oa, na = cell_means[a]; mb, ob, nb = cell_means[b]
        if min(ma, mb, oa, ob, na, nb) <= 0: continue
        meas_su = max(ma, mb) / min(ma, mb)
        old_su = max(oa, ob) / min(oa, ob)
        new_su = max(na, nb) / min(na, nb)
        # Speedup error
        eO = abs(old_su - meas_su) / meas_su * 100
        eN = abs(new_su - meas_su) / meas_su * 100
        sum_errO += eO; sum_errN += eN; n_pairs += 1
        # Ranking
        rO = "ok" if (ma < mb) == (oa < ob) else "FLIP"
        rN = "ok" if (ma < mb) == (na < nb) else "FLIP"
        ok_old += (rO == "ok"); ok_new += (rN == "ok")
        print(f"{a:<22} {b:<22} {meas_su:>8.2f} {old_su:>7.2f} {new_su:>7.2f} "
              f"{eO:>6.1f}% {eN:>6.1f}% {rO:>6} {rN:>6}")
    if n_pairs > 0:
        print()
        print(f"Mean speedup error:  old={sum_errO/n_pairs:.1f}%   new={sum_errN/n_pairs:.1f}%")
        print(f"Ranking accuracy:    old={ok_old}/{n_pairs} ({ok_old/n_pairs*100:.0f}%)   "
              f"new={ok_new}/{n_pairs} ({ok_new/n_pairs*100:.0f}%)")


if __name__ == "__main__":
    sys.exit(main() or 0)
