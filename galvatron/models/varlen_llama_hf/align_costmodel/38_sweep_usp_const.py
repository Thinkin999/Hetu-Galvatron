"""Sweep USP overhead constants and measure offline reprediction MAPE.

Tries different reductions of (ulysses_a2a_overhead_ms, usp_a2a_overhead_extra_ms,
ring_step_overhead_ms, usp_layer_overhead_base/per_sp_ms) to see which
combination best improves usp2x4 over-prediction without breaking other cells.

Per-mb cost decomposition (for usp2x4 mb=11k):
  qo_a2a:      60 ms  ( 15%)  — real data, leave alone
  kv_a2a:      15 ms  (  4%)  — real data
  a2a_cpu_oh: 102 ms  ( 26%)  ← controlled by ulysses_a2a_overhead_ms + usp_a2a_overhead_extra_ms
  ring_fwd:    12 ms  (  3%)
  ring_bwd:    25 ms  (  6%)
  ring_step_oh:96 ms  ( 24%)  ← controlled by ring_step_overhead_ms
  layer_extra: 83 ms  ( 21%)  ← controlled by usp_layer_overhead_base + per_sp
  TOTAL:      395 ms

Constants account for 281/395 = 71% of usp2x4 mb=11k comm prediction.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, itertools
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


def build_cm(uly_a2a_oh=0.2, usp_a2a_extra=0.2, ring_step_oh=0.5,
             usp_layer_base=1.0, usp_layer_per_sp=0.8):
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
    cm.ulysses_a2a_overhead_ms = uly_a2a_oh
    cm.usp_a2a_overhead_extra_ms = usp_a2a_extra
    cm.ring_step_overhead_ms = ring_step_oh
    cm.usp_layer_overhead_base_ms = usp_layer_base
    cm.usp_layer_overhead_per_sp_ms = usp_layer_per_sp
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
        rows = []
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
            rows.append((fb, new_total))
        if rows:
            mfb = sum(r[0] for r in rows) / len(rows)
            mpred = sum(r[1] for r in rows) / len(rows)
            cells[label] = (mfb, mpred)
    return cells


def pairwise_metrics(cells):
    labels = list(cells.keys())
    ok = n = 0
    sum_e = 0
    for a, b in itertools.combinations(labels, 2):
        ma, pa = cells[a]
        mb, pb = cells[b]
        if min(ma, mb, pa, pb) <= 0:
            continue
        meas_su = max(ma, mb) / min(ma, mb)
        pred_su = max(pa, pb) / min(pa, pb)
        e = abs(pred_su - meas_su) / meas_su * 100
        sum_e += e
        ok += ((ma < mb) == (pa < pb))
        n += 1
    return sum_e / n, ok, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    args = ap.parse_args()

    e2e = args.results_dir
    if os.path.basename(e2e.rstrip("/")) != "end2end":
        e2e = os.path.join(e2e, "end2end")

    # Configurations to try
    # (label, uly_a2a, usp_a2a_extra, ring_step, layer_base, layer_per_sp)
    configs = [
        ("baseline (current)",         0.2, 0.2, 0.5, 1.0, 0.8),
        ("half all constants",         0.1, 0.1, 0.25, 0.5, 0.4),
        ("zero a2a_cpu_oh",            0.0, 0.0, 0.5, 1.0, 0.8),
        ("zero ring_step_oh",          0.2, 0.2, 0.0, 1.0, 0.8),
        ("zero layer_extra",           0.2, 0.2, 0.5, 0.0, 0.0),
        ("zero all 3 const groups",    0.0, 0.0, 0.0, 0.0, 0.0),
        ("a2a_cpu_oh * 0.5",           0.1, 0.1, 0.5, 1.0, 0.8),
        ("ring_step_oh * 0.5",         0.2, 0.2, 0.25, 1.0, 0.8),
        ("layer_extra * 0.5",          0.2, 0.2, 0.5, 0.5, 0.4),
        ("aggressive: /4",             0.05, 0.05, 0.125, 0.25, 0.2),
    ]

    print(f"{'config':<30} {'MAPE':>7} {'rank':>8} | "
          f"{'usp2x4_c1':>9} {'usp2x4_c8':>9} {'ulysses_c8':>10} {'ring_c8':>8}")
    print("-" * 110)
    for label, *params in configs:
        cm = build_cm(*params)
        cells = repredict_all(cm, e2e)
        mape, ok, n = pairwise_metrics(cells)
        u2x4_c1 = cells.get("usp2x4_chunks1", (0, 0))
        u2x4_c8 = cells.get("usp2x4_chunks8", (0, 0))
        uly_c8 = cells.get("ulysses8_chunks8", (0, 0))
        ring_c8 = cells.get("ring8_chunks8", (0, 0))

        def err(t):
            m, p = t
            return (m - p) / m * 100 if m > 0 else 0

        print(f"{label:<30} {mape:>6.1f}% {ok:>3}/{n:>3} | "
              f"{err(u2x4_c1):>+8.1f}% {err(u2x4_c8):>+8.1f}% "
              f"{err(uly_c8):>+9.1f}% {err(ring_c8):>+7.1f}%")


if __name__ == "__main__":
    sys.exit(main() or 0)
