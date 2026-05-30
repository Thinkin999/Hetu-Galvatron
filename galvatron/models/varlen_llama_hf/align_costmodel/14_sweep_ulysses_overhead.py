"""Counterfactual sweep of ulysses_a2a_overhead_ms over a candidate range.
For each value, recompute predictions vs the latest 14 measured.json and
report mean/max abs error. Helps pick the best overhead without re-running.
"""
import glob
import json
import os
import statistics
import sys
from typing import Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
for p in (REPO_ROOT, MODEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from adacpsp_solver import AdaCPSPCostModel, ParallelStrategy  # noqa: E402


def latest_matching(pattern, predicate=None):
    for p in sorted(glob.glob(pattern), reverse=True):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if predicate is None or predicate(d):
            return p
    return ""


def build_cm(overhead: float) -> AdaCPSPCostModel:
    configs = os.path.join(MODEL_DIR, "configs")
    attn = latest_matching(
        os.path.join(configs, "profile_validate_*.json"),
        predicate=lambda d: "attention" in d and "segments" in d.get("attention", {}),
    )
    comm = latest_matching(
        os.path.join(configs, "comm_profile_v2_*.json"),
        predicate=lambda d: d.get("type") == "comm_profile_v2",
    )
    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn,
        comm_profile_json=comm,
        cluster_size=16,
        gpus_per_node=8,
    )
    cm.ulysses_a2a_overhead_ms = overhead
    return cm


def predict(cm, sp, seq, num_seqs=16):
    strat = ParallelStrategy(
        attn_type="ulysses", parallel_size=sp, sp_size=sp, cp_size=1,
        placement="context_first",
    )
    return cm.total_time([seq] * num_seqs, strat) / cm.l


def main():
    if len(sys.argv) >= 2:
        measured_path = sys.argv[1]
    else:
        cands = sorted(glob.glob(os.path.join(
            SCRIPT_DIR, "results", "quick_uly_*", "measured.json")), reverse=True)
        measured_path = cands[0]
    data = json.load(open(measured_path))
    measured = data["measured_per_layer_ms"]
    num_seqs = data.get("num_seqs", 16)
    cases = []
    for sp_s, m in measured.items():
        for seq_s, ms in m.items():
            if ms is None:
                continue
            cases.append((int(sp_s), int(seq_s), ms))

    candidates = [0.00, 0.05, 0.10, 0.13, 0.15, 0.17, 0.20, 0.22, 0.25, 0.30]
    print(f"  measured = {measured_path}")
    print(f"  {len(cases)} cases")
    print()
    print("| overhead_ms | mean_abs | max_abs | worst_case | mean_signed |")
    print("|------------:|---------:|--------:|:-----------|------------:|")
    for oh in candidates:
        cm = build_cm(oh)
        errs = []
        for sp, seq, m in cases:
            try:
                pred = predict(cm, sp, seq, num_seqs)
                e = (pred - m) / m * 100
                errs.append((sp, seq, e))
            except Exception:
                pass
        abs_errs = [abs(e) for _, _, e in errs]
        signed = [e for _, _, e in errs]
        mean_abs = sum(abs_errs) / len(abs_errs)
        max_idx = abs_errs.index(max(abs_errs))
        wsp, wseq, werr = errs[max_idx]
        mean_signed = sum(signed) / len(signed)
        print(f"|       {oh:.2f} | {mean_abs:6.2f}% | {max(abs_errs):6.2f}% | "
              f"sp={wsp} seq={wseq} ({werr:+.2f}%) | {mean_signed:+6.2f}% |")


if __name__ == "__main__":
    main()
