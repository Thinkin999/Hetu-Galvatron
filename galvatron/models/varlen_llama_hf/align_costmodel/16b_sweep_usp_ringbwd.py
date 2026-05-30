"""Test whether USP-specific ring_bwd_comm_ratio uplift closes the (8,2)
and (4,4) long-seq under-prediction.

Trace shows USP bwd ring p2p is ~2x what the cost model expects (cost model
uses ring_bwd_comm_ratio=2.0; trace suggests effective ratio ~3.0-3.5 for
USP nested context).
"""

import glob
import json
import os
import sys
from itertools import product

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
for p in (REPO_ROOT, MODEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from adacpsp_solver import AdaCPSPCostModel, ParallelStrategy  # noqa: E402


def latest(pattern, predicate=None):
    for p in sorted(glob.glob(pattern), reverse=True):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if predicate is None or predicate(d):
            return p
    return ""


def main():
    if len(sys.argv) >= 2:
        measured_path = sys.argv[1]
    else:
        cands = sorted(glob.glob(os.path.join(
            SCRIPT_DIR, "results", "quick_usp_*", "measured.json")), reverse=True)
        measured_path = cands[0]
    data = json.load(open(measured_path))
    measured = data["measured_per_layer_ms"]
    num_seqs = data.get("num_seqs", 16)

    configs = os.path.join(MODEL_DIR, "configs")
    attn = latest(os.path.join(configs, "profile_validate_*.json"),
                   lambda d: "attention" in d and "segments" in d.get("attention", {}))
    comm = latest(os.path.join(configs, "comm_profile_v2_*.json"),
                   lambda d: d.get("type") == "comm_profile_v2")

    cases = []
    for sp_s, by_cp in measured.items():
        for cp_s, by_seq in by_cp.items():
            for seq_s, m in by_seq.items():
                if m is None:
                    continue
                cases.append((int(sp_s), int(cp_s), int(seq_s), float(m)))

    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn, comm_profile_json=comm,
        cluster_size=16, gpus_per_node=8,
    )

    orig_ratio = cm.ring_bwd_comm_ratio

    # sp-scaled extras: const = const_base + per_sp_extra * sp_size
    # This captures the empirical observation that sp=8 USP cases are heavily
    # under-predicted while sp=2 cases are over-predicted at short seq.
    a2a_extras = [0.0, 0.20, 0.30]
    const_bases = [0.0, 1.0, 2.0, 3.0]
    per_sp_extras = [0.0, 0.2, 0.4, 0.6, 0.8]

    best = None
    print(f"{'a2a':>5s} {'cbase':>6s} {'persp':>6s} {'mean_abs':>9s} "
          f"{'max_abs':>8s} {'mean_signed':>12s} {'short_max':>10s} {'long_max':>9s}")
    for a2a_e, const_base, per_sp in product(a2a_extras, const_bases, per_sp_extras):
        cm.ulysses_a2a_overhead_ms = 0.20 + a2a_e
        cm.ring_step_overhead_ms = 0.5
        cm.usp_a2a_overhead_extra_ms = 0.0
        cm.usp_layer_overhead_ms = 0.0  # we'll add scaled overhead manually
        cm.ring_bwd_comm_ratio = 2.0
        errs = []
        short_errs = []
        long_errs = []
        for sp, cp, seq, m in cases:
            strategy = ParallelStrategy(
                attn_type="usp", parallel_size=sp*cp, sp_size=sp, cp_size=cp,
                placement="head_first",
            )
            extra = const_base + per_sp * sp
            pred = cm.total_time([seq] * num_seqs, strategy) / cm.l + extra
            err = (pred - m) / m * 100
            errs.append(err)
            if seq <= 8192:
                short_errs.append(err)
            else:
                long_errs.append(err)
        abs_errs = [abs(e) for e in errs]
        mean_abs = sum(abs_errs) / len(abs_errs)
        max_abs = max(abs_errs)
        mean_signed = sum(errs) / len(errs)
        short_max = max((abs(e) for e in short_errs), default=0)
        long_max = max((abs(e) for e in long_errs), default=0)
        score = max_abs + mean_abs
        marker = ""
        if best is None or score < best[0]:
            best = (score, a2a_e, const_base, per_sp, mean_abs, max_abs,
                     mean_signed, short_max, long_max)
            marker = " <-"
        if max_abs < 15:
            print(f"{a2a_e:>5.2f} {const_base:>5.2f} {per_sp:>5.2f} "
                  f"{mean_abs:>8.2f}% {max_abs:>7.2f}% {mean_signed:>+11.2f}% "
                  f"{short_max:>9.2f}% {long_max:>8.2f}%{marker}")

    cm.ring_bwd_comm_ratio = orig_ratio

    print(f"\nbest: a2a={best[1]:.2f} const_base={best[2]:.2f} per_sp={best[3]:.2f}")
    print(f"  max_abs={best[5]:.2f}%  mean_abs={best[4]:.2f}%  mean_signed={best[6]:+.2f}%")
    print(f"  short_max={best[7]:.2f}%  long_max={best[8]:.2f}%")
    print(f"\nNote: bwd_ratio affects pure Ring too. If best needs bwd_r != 2.0, "
          f"\ncheck that Ring alignment doesn't break.")


if __name__ == "__main__":
    main()
