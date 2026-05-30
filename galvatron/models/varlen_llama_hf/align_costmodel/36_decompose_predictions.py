"""Decompose each prediction into (attn_compute, comm, residual, b_step_fb)
and compare against measured. Used to locate which component is most wrong
per strategy/cell.

For ulysses8_chunks1 we expect attention compute to be over-estimated (LUT
extrapolation beyond 16k). For ring8 we expect ring compute / comm to be
mis-modeled. This script gives the data to confirm.

Usage:
  python 36_decompose_predictions.py <results_dir> [--skip-warmup 2]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
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
    attn_p, _ = latest("profile_validate_*.json", cfg)
    comm_p, _ = latest("comm_profile_*.json", cfg)
    _, resid = latest("residual_profile_*.json", cfg)
    _, bdec = latest("b_decomp_profile_*.json", cfg)
    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn_p, comm_profile_json=comm_p,
        cluster_size=16, validation_json=attn_p, gpus_per_node=8,
    )
    if resid:
        cm.apply_residual_profile(resid)
    if bdec:
        cm.apply_b_decomp_profile(bdec)
    print(f"# loaded attn={os.path.basename(attn_p)}", file=sys.stderr)
    print(f"# loaded comm={os.path.basename(comm_p)}", file=sys.stderr)
    print(f"# bdec={'y' if bdec else 'n'}", file=sys.stderr)
    return cm


def decompose_group(cm: AdaCPSPCostModel, g: dict) -> Dict[str, float]:
    """Break one group prediction into its components.

    Returns ms for: attn_compute (fwd+bwd, all layers), comm (all layers),
    residual, total, plus the seqlen used for the LUT lookup.
    """
    s = int(g["tokens"])
    if s <= 0:
        return dict(attn=0, comm=0, resid=0, total=0, lut_seq=0,
                    extrapolation=False)
    strat = ParallelStrategy(
        attn_type=g["attn_type"],
        parallel_size=int(g.get("parallel_size",
                                g["sp_size"] * g["cp_size"])),
        sp_size=int(g["sp_size"]),
        cp_size=int(g["cp_size"]),
        placement=g.get("placement", "context_first"),
    )

    # Attention compute (fwd × (1 + bwd_fwd_ratio)). compute_time is fwd-only.
    fwd_compute = cm.compute_time([s], strat)  # ms (1 call/layer × layers)
    if strat.attn_type == "ring":
        # Ring: compute_time returns per-step compute × num_layers.
        # Total fwd compute = cp × per-step × num_layers.
        fwd_compute = strat.cp_size * fwd_compute
    elif strat.attn_type == "usp":
        fwd_compute = strat.cp_size * fwd_compute
    attn_total = fwd_compute * (1 + cm.bwd_fwd_ratio)

    # Comm
    comm = cm.comm_time([s], strat)
    if strat.attn_type == "ring":
        comm = comm * (1 + cm.ring_bwd_comm_ratio)
    elif strat.attn_type == "usp":
        comm = comm  # already includes fwd+bwd internally? check below.
    else:  # ulysses
        # ulysses comm_time returns one direction; add bwd direction
        comm = comm  # check: do we need ×2?

    # Residual
    tokens_per_gpu = s / strat.parallel_size
    resid_a = cm.residual_a_per_sp.get(
        int(strat.sp_size) if strat.attn_type in ("ulysses", "usp") else 1,
        cm.residual_a_default_per_token,
    )
    resid_b = cm.residual_b_per_sp.get(
        int(strat.sp_size) if strat.attn_type in ("ulysses", "usp") else 1,
        cm.residual_b_default_ms,
    )
    residual = resid_a * tokens_per_gpu + resid_b

    # LUT seq used
    if strat.attn_type == "ulysses":
        lut_seq = s
    elif strat.attn_type == "ring":
        lut_seq = s / strat.cp_size
    elif strat.attn_type == "usp":
        lut_seq = s / strat.cp_size
    else:
        lut_seq = s / strat.parallel_size

    extrapolation = False
    if cm.piecewise:
        last_hi = cm.piecewise[-1]["range"][1]
        extrapolation = lut_seq > last_hi

    total_via_cm = cm.total_time([s], strat)

    return dict(
        attn=attn_total,
        comm=comm,
        resid=residual,
        total_manual=attn_total + comm + residual,
        total_cm=total_via_cm,
        lut_seq=lut_seq,
        extrapolation=extrapolation,
        tokens=s,
        sp=strat.sp_size, cp=strat.cp_size,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--skip-warmup", type=int, default=2)
    ap.add_argument("--n-show", type=int, default=4,
                    help="iters per cell to show in detail")
    args = ap.parse_args()

    cm = build_cm()

    e2e = args.results_dir
    if os.path.basename(e2e.rstrip("/")) != "end2end":
        cand = os.path.join(e2e, "end2end")
        if os.path.isdir(cand):
            e2e = cand

    # LUT info
    lut_max = cm.piecewise[-1]["range"][1] if cm.piecewise else 0
    print()
    print(f"Attention LUT max seq: {lut_max}")
    print(f"bwd_fwd_ratio: {cm.bwd_fwd_ratio}")
    print()

    cells_summary = {}  # cell -> list of breakdowns (one per iter)

    for d in sorted(glob.glob(os.path.join(e2e, "*_chunks*"))):
        label = os.path.basename(d.rstrip("/"))
        jp = os.path.join(d, "rank0.jsonl")
        if not os.path.isfile(jp):
            continue
        with open(jp) as f:
            recs = [json.loads(l) for l in f if l.strip()]
        train_recs = [r for r in recs if r.get("phase") == "train_step"]
        breakdowns = []
        for i, r in enumerate(train_recs):
            if i < args.skip_warmup:
                continue
            fb = float(r.get("timings_ms", {}).get("forward_backward", 0.0))
            gb = r.get("global_batch") or {}
            mbs = gb.get("microbatches") or []
            # Get the slowest group per mb, sum across mbs (matches solver
            # convention).
            sum_mb_attn = 0.0
            sum_mb_comm = 0.0
            sum_mb_resid = 0.0
            sum_mb_total_cm = 0.0
            extrapolation_any = False
            lut_seq_max = 0
            n_groups = 0
            for mb in mbs:
                mb_max_attn = 0
                mb_max_comm = 0
                mb_max_resid = 0
                mb_max_total = 0
                for g in mb.get("groups", []):
                    n_groups += 1
                    parts = decompose_group(cm, g)
                    if parts["extrapolation"]:
                        extrapolation_any = True
                    lut_seq_max = max(lut_seq_max, parts["lut_seq"])
                    # Within a microbatch, groups run in parallel; we
                    # account for the slowest group's contribution to each
                    # component (this is an approximation: max-of-totals,
                    # not max-of-each-component, but matches solver math).
                    if parts["total_cm"] > mb_max_total:
                        mb_max_total = parts["total_cm"]
                        mb_max_attn = parts["attn"]
                        mb_max_comm = parts["comm"]
                        mb_max_resid = parts["resid"]
                sum_mb_attn += mb_max_attn
                sum_mb_comm += mb_max_comm
                sum_mb_resid += mb_max_resid
                sum_mb_total_cm += mb_max_total

            # b_step_fb
            b_step = 0.0
            if len(mbs) >= 2 and hasattr(cm, "b_step_fb_ms_for_strategies"):
                sp_values = []
                for mb in mbs:
                    for g in mb.get("groups", []):
                        sp = int(g.get("sp_size", 1))
                        if sp not in sp_values:
                            sp_values.append(sp)
                b_step = float(cm.b_step_fb_ms_for_strategies(sp_values))

            pred = sum_mb_total_cm + b_step
            err = (fb - pred) / max(1.0, fb) * 100
            breakdowns.append(dict(
                i=i,
                tokens=int(gb.get("global_tokens", 0)),
                n_mb=len(mbs), n_groups=n_groups,
                lut_seq_max=lut_seq_max,
                extrap=extrapolation_any,
                fb=fb,
                attn=sum_mb_attn, comm=sum_mb_comm,
                resid=sum_mb_resid, b_step=b_step,
                pred=pred, err=err,
            ))
        cells_summary[label] = breakdowns

    # Detailed per-iter table (first N iters per cell)
    print(f"Per-iter decomposition (showing first {args.n_show} stable iters per cell)")
    print(f"{'cell':<22} {'i':>2} {'tokens':>7} {'lut_max':>7} {'ext':>3} "
          f"{'meas':>6} {'attn':>5} {'comm':>5} {'resid':>5} {'bstep':>5} "
          f"{'pred':>6} {'err%':>6}")
    print("-" * 102)
    for label, rows in cells_summary.items():
        for row in rows[:args.n_show]:
            print(f"{label:<22} {row['i']:>2} {row['tokens']:>7} "
                  f"{row['lut_seq_max']:>7.0f} {'Y' if row['extrap'] else '-':>3} "
                  f"{row['fb']:>6.0f} {row['attn']:>5.0f} {row['comm']:>5.0f} "
                  f"{row['resid']:>5.0f} {row['b_step']:>5.0f} "
                  f"{row['pred']:>6.0f} {row['err']:>+5.1f}%")
        if rows:
            print()

    # Per-cell averages
    print("=" * 110)
    print("Per-cell averages and component contribution to total prediction")
    h = (f"{'cell':<22} {'n':>3} {'meas':>6} {'pred':>6} {'err':>7} | "
         f"{'attn':>5} ({'%':>3}) {'comm':>5} ({'%':>3}) "
         f"{'resid':>5} ({'%':>3}) {'bstep':>5} ({'%':>3})  ext")
    print(h); print("-" * len(h))
    for label, rows in cells_summary.items():
        if not rows: continue
        n = len(rows)
        m_fb = sum(r['fb'] for r in rows) / n
        m_pred = sum(r['pred'] for r in rows) / n
        m_attn = sum(r['attn'] for r in rows) / n
        m_comm = sum(r['comm'] for r in rows) / n
        m_resid = sum(r['resid'] for r in rows) / n
        m_bstep = sum(r['b_step'] for r in rows) / n
        err = (m_fb - m_pred) / max(1.0, m_fb) * 100
        # contributions to predicted (not including b_step? include it)
        denom = max(1.0, m_pred)
        any_ext = any(r['extrap'] for r in rows)
        print(f"{label:<22} {n:>3} {m_fb:>6.0f} {m_pred:>6.0f} {err:>+6.1f}% | "
              f"{m_attn:>5.0f} ({m_attn/denom*100:>3.0f}) "
              f"{m_comm:>5.0f} ({m_comm/denom*100:>3.0f}) "
              f"{m_resid:>5.0f} ({m_resid/denom*100:>3.0f}) "
              f"{m_bstep:>5.0f} ({m_bstep/denom*100:>3.0f})  {'Y' if any_ext else '-'}")

    # Diagnose attention component specifically
    print()
    print("=" * 110)
    print("Attention compute as fraction of measured forward_backward")
    print(f"{'cell':<22} {'meas':>6} {'attn_pred':>9} {'attn/meas':>10} {'lut_seq':>8} {'extrap':>7}")
    print("-" * 70)
    for label, rows in cells_summary.items():
        if not rows: continue
        n = len(rows)
        m_fb = sum(r['fb'] for r in rows) / n
        m_attn = sum(r['attn'] for r in rows) / n
        m_lut = max(r['lut_seq_max'] for r in rows)
        any_ext = any(r['extrap'] for r in rows)
        ratio = m_attn / max(1, m_fb)
        print(f"{label:<22} {m_fb:>6.0f} {m_attn:>9.0f} "
              f"{ratio*100:>9.0f}% {m_lut:>8.0f} "
              f"{'Y' if any_ext else '-':>7}")


if __name__ == "__main__":
    sys.exit(main() or 0)
