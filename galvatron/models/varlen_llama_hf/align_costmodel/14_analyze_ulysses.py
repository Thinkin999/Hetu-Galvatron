"""Compare 14_quick_ulysses_align measured times against AdaCPSPCostModel
predictions. Also break down predicted time into (compute + comm) parts so we
can see whether errors are compute-driven, comm-driven, or both.

Output: a Markdown table per sp_size with seq x measured/pred/err columns.
"""

import glob
import json
import os
import sys
from typing import Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
for p in (REPO_ROOT, MODEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from adacpsp_solver import AdaCPSPCostModel, ParallelStrategy  # noqa: E402


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def latest_matching(pattern, predicate=None):
    for p in sorted(glob.glob(pattern), reverse=True):
        try:
            d = load_json(p)
        except Exception:
            continue
        if predicate is None or predicate(d):
            return p
    return ""


def build_cm() -> AdaCPSPCostModel:
    configs = os.path.join(MODEL_DIR, "configs")
    attn = latest_matching(
        os.path.join(configs, "profile_validate_*.json"),
        predicate=lambda d: "attention" in d and "segments" in d.get("attention", {}),
    )
    comm = latest_matching(
        os.path.join(configs, "comm_profile_v2_*.json"),
        predicate=lambda d: d.get("type") == "comm_profile_v2",
    )
    print(f"  attn  = {os.path.basename(attn)}")
    print(f"  comm  = {os.path.basename(comm)}")
    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn,
        comm_profile_json=comm,
        cluster_size=16,
        gpus_per_node=8,
    )
    return cm


def predict_uly(cm: AdaCPSPCostModel, sp_size: int, seq_len: int,
                num_seqs: int = 16):
    """Return (total_ms_per_layer, comm_ms_per_layer, compute_ms_per_layer)."""
    strategy = ParallelStrategy(
        attn_type="ulysses",
        parallel_size=sp_size,
        sp_size=sp_size,
        cp_size=1,
        placement="context_first",
    )
    seqlens = [seq_len] * num_seqs
    total_ms = cm.total_time(seqlens, strategy) / cm.l
    comm_ms = cm.comm_time(seqlens, strategy) / cm.l
    # compute = total - comm (fwd) + bwd-equivalents. The cost model uses
    # bwd_fwd_ratio * compute on the compute side. Let's just compute the
    # fwd-only compute as a reference.
    fwd_compute = cm.compute_time(seqlens, strategy) / cm.l
    return total_ms, comm_ms, fwd_compute


def main():
    if len(sys.argv) >= 2:
        measured_path = sys.argv[1]
    else:
        # find the latest quick_uly run
        cands = sorted(glob.glob(os.path.join(
            SCRIPT_DIR, "results", "quick_uly_*", "measured.json")), reverse=True)
        if not cands:
            raise SystemExit("no quick_uly run found")
        measured_path = cands[0]

    print(f"  measured = {measured_path}")
    data = load_json(measured_path)
    measured = data["measured_per_layer_ms"]
    num_seqs = data.get("num_seqs", 16)

    cm = build_cm()
    print()

    sps = sorted(int(k) for k in measured.keys())
    seqs = sorted(int(k) for k in next(iter(measured.values())).keys())

    print("Ulysses alignment per layer (ms):")
    print()
    header = (f"| sp | seq | measured | predicted | err | "
              f"pred_comm | pred_fwd_compute |")
    print(header)
    print("|----|-----|---------:|----------:|----:|---------:|-----------------:|")
    for sp in sps:
        for seq in seqs:
            m = measured[str(sp)].get(str(seq))
            if m is None:
                continue
            try:
                pred, comm, fwd_comp = predict_uly(cm, sp, seq, num_seqs)
            except Exception as e:
                print(f"| {sp} | {seq} | {m:.2f} | ERR: {e} | | | |")
                continue
            err = (pred - m) / m * 100 if m > 0 else 0
            print(f"| {sp} | {seq} | {m:.2f} | {pred:.2f} | {err:+.2f}% | "
                  f"{comm:.2f} | {fwd_comp:.2f} |")
        # separator row for visual grouping
    print()
    # Summary
    errs = []
    for sp in sps:
        for seq in seqs:
            m = measured[str(sp)].get(str(seq))
            if m is None or m <= 0:
                continue
            pred, _, _ = predict_uly(cm, sp, seq, num_seqs)
            errs.append((sp, seq, abs((pred - m) / m * 100), (pred - m) / m * 100))
    if errs:
        absmean = sum(e[2] for e in errs) / len(errs)
        absmax = max(e[2] for e in errs)
        # find max
        worst = max(errs, key=lambda x: x[2])
        print(f"mean abs error = {absmean:.2f}%, max = {absmax:.2f}% "
              f"(worst sp={worst[0]} seq={worst[1]} err={worst[3]:+.2f}%)")


if __name__ == "__main__":
    main()
