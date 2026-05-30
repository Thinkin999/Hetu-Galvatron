"""Compare 16_quick_usp_align measured times against AdaCPSPCostModel
predictions. Break down prediction into (a2a_comm + ring_comm + fwd_compute).
"""

import glob
import json
import os
import sys

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
    print(f"  attn = {os.path.basename(attn)}")
    print(f"  comm = {os.path.basename(comm)}")
    return AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn,
        comm_profile_json=comm,
        cluster_size=16,
        gpus_per_node=8,
    )


def predict_usp(cm: AdaCPSPCostModel, sp: int, cp: int, seq: int,
                 num_seqs: int = 16, placement: str = "head_first"):
    strategy = ParallelStrategy(
        attn_type="usp", parallel_size=sp * cp, sp_size=sp, cp_size=cp,
        placement=placement,
    )
    seqlens = [seq] * num_seqs
    total = cm.total_time(seqlens, strategy) / cm.l
    comm = cm.comm_time(seqlens, strategy) / cm.l
    fwd_comp = cm.compute_time(seqlens, strategy)  # compute_time is already per layer (not multiplied by l for cp>1)
    # Actually compute_time returns total compute per layer × num_layers? Let me check.
    # In adacpsp_solver.compute_time: total = sum(...) * self.l. So divide by l.
    fwd_comp /= cm.l
    return total, comm, fwd_comp


def main():
    if len(sys.argv) >= 2:
        measured_path = sys.argv[1]
    else:
        cands = sorted(glob.glob(os.path.join(
            SCRIPT_DIR, "results", "quick_usp_*", "measured.json")), reverse=True)
        if not cands:
            raise SystemExit("no quick_usp run found")
        measured_path = cands[0]
    print(f"  measured = {measured_path}")
    data = json.load(open(measured_path))
    measured = data["measured_per_layer_ms"]
    num_seqs = data.get("num_seqs", 16)

    cm = build_cm()
    print()

    print("USP alignment per layer (ms):")
    print()
    print("| sp | cp | seq | measured | predicted | err | pred_comm | pred_fwd_comp |")
    print("|----|----|-----|---------:|----------:|----:|----------:|--------------:|")
    errs = []
    for sp_s in sorted(measured.keys(), key=int):
        sp = int(sp_s)
        for cp_s in sorted(measured[sp_s].keys(), key=int):
            cp = int(cp_s)
            for seq_s in sorted(measured[sp_s][cp_s].keys(), key=int):
                seq = int(seq_s)
                m = measured[sp_s][cp_s][seq_s]
                if m is None:
                    continue
                try:
                    pred, comm, fwd_comp = predict_usp(cm, sp, cp, seq, num_seqs)
                except Exception as e:
                    print(f"| {sp} | {cp} | {seq} | {m:.2f} | ERR | | | | |")
                    continue
                err = (pred - m) / m * 100 if m > 0 else 0
                print(f"| {sp} | {cp} | {seq} | {m:.2f} | {pred:.2f} | {err:+.2f}% | "
                      f"{comm:.2f} | {fwd_comp:.2f} |")
                errs.append((sp, cp, seq, err))
    print()
    if errs:
        abs_errs = [abs(e[3]) for e in errs]
        mean_abs = sum(abs_errs) / len(abs_errs)
        max_idx = abs_errs.index(max(abs_errs))
        wsp, wcp, wseq, werr = errs[max_idx]
        mean_signed = sum(e[3] for e in errs) / len(errs)
        print(f"mean abs error = {mean_abs:.2f}%, max = {max(abs_errs):.2f}% "
              f"(worst sp={wsp} cp={wcp} seq={wseq} err={werr:+.2f}%)")
        print(f"mean signed error = {mean_signed:+.2f}%")


if __name__ == "__main__":
    main()
