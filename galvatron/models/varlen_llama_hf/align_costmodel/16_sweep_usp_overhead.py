"""Counterfactually sweep USP-specific overhead parameters against measured
data, picking the (per_a2a_extra, per_step_extra) combo that minimizes max
abs error while keeping mean abs error low.
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
    print(f"  measured = {measured_path}")
    print(f"  attn = {os.path.basename(attn)}")
    print(f"  comm = {os.path.basename(comm)}")

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

    # Save originals
    orig_uly_overhead = cm.ulysses_a2a_overhead_ms
    orig_ring_overhead = cm.ring_step_overhead_ms

    # Zero out USP-specific extras so sweep starts from clean baseline. The
    # actual sweep values below are the *only* USP-extra contribution we test,
    # plus the existing per-step ring/uly_a2a overheads which sweep as deltas.
    cm.usp_a2a_overhead_extra_ms = 0.0
    cm.usp_layer_overhead_ms = 0.0

    print(f"\nbaseline params: uly_a2a={orig_uly_overhead} ring_step={orig_ring_overhead}")
    print(f"USP extras zeroed; sweeping all USP overhead from scratch.\n")
    print(f"Sweeping USP-specific extra per-a2a (on top of Uly) and per-ring-step extras\n")

    # Sweep 4 params: per-a2a extra, per-ring-step extra, per-layer constant,
    # per-token-per-rank scaling. The last captures the empirical
    # "other_gpu" reshape work seen in trace that scales ~linearly with seq.
    a2a_extras = [0.0, 0.20, 0.30, 0.40]
    step_extras = [0.0, 0.5, 1.0]
    const_extras = [0.0, 1.0, 2.0, 3.0]
    # us/token/rank scaling; effective_tokens_per_rank = total_tokens/parallel
    tok_scales_us = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00]

    best = None
    print(f"{'a2a':>5s} {'step':>5s} {'const':>5s} {'tok_us':>7s} {'mean_abs':>9s} "
          f"{'max_abs':>8s} {'mean_signed':>12s} {'short_max':>10s} {'long_max':>9s}")
    for a2a_e, step_e, const_e, tok_us in product(
            a2a_extras, step_extras, const_extras, tok_scales_us):
        cm.ulysses_a2a_overhead_ms = orig_uly_overhead + a2a_e
        cm.ring_step_overhead_ms = orig_ring_overhead + step_e
        errs = []
        short_errs = []
        long_errs = []
        for sp, cp, seq, m in cases:
            strategy = ParallelStrategy(
                attn_type="usp", parallel_size=sp*cp, sp_size=sp, cp_size=cp,
                placement="head_first",
            )
            tokens_per_rank = num_seqs * seq / (sp * cp)
            extra = const_e + tok_us * 1e-3 * tokens_per_rank
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
        # Score: balance max with mean
        score = max_abs + mean_abs
        marker = ""
        if best is None or score < best[0]:
            best = (score, a2a_e, step_e, const_e, tok_us, mean_abs, max_abs,
                     mean_signed, short_max, long_max)
            marker = " <-"
        if max_abs < 19:
            print(f"{a2a_e:>5.2f} {step_e:>5.2f} {const_e:>5.2f} {tok_us:>6.2f}us "
                  f"{mean_abs:>8.2f}% {max_abs:>7.2f}% {mean_signed:>+11.2f}% "
                  f"{short_max:>9.2f}% {long_max:>8.2f}%{marker}")

    cm.ulysses_a2a_overhead_ms = orig_uly_overhead
    cm.ring_step_overhead_ms = orig_ring_overhead

    print(f"\nbest: a2a={best[1]:.2f} step={best[2]:.2f} const={best[3]:.2f} tok={best[4]:.2f}us")
    print(f"  max_abs={best[6]:.2f}%  mean_abs={best[5]:.2f}%  mean_signed={best[7]:+.2f}%")
    print(f"  short_max={best[8]:.2f}%  long_max={best[9]:.2f}%")


if __name__ == "__main__":
    main()
