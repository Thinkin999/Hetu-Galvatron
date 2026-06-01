"""Break USP comm prediction into its sub-components for a few representative
microbatches.

Components (per microbatch):
  1. qo_a2a × 4L ops (QO data transfer)
  2. kv_a2a × 4L ops (KV data transfer)
  3. per_op_cpu × 8L (ulysses + usp per-a2a CPU overhead)
  4. ring fwd data × (cp-1) × L
  5. ring bwd data × ring_bwd_comm_ratio × (cp-1) × L
  6. ring_step_overhead × 2 × (cp-1) × L
  7. layer_extra (base + per_sp × sp) × L
"""
from __future__ import annotations
import argparse, glob, json, os, sys

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
    if resid: cm.apply_residual_profile(resid)
    if bdec: cm.apply_b_decomp_profile(bdec)
    return cm


def usp_breakdown(cm, tokens, sp, cp, placement="context_first"):
    """Mirrors total_time() USP branch but returns each comm component."""
    a2a_topo = cm._get_topo(placement, "alltoall", sp, cp)
    ring_topo = cm._get_topo(placement, "ring", sp, cp)
    parallel_size = sp * cp
    q_factor, kv_factor = cm.head_padding_overhead(sp)

    qo_msg_mb = cm.h * q_factor * tokens * 2 / 1024 / 1024 / parallel_size
    kv_msg_mb = cm.kv_hidden * kv_factor * tokens * 2 / 1024 / 1024 / parallel_size
    qo_a2a_per = cm._a2a_per_op_time(qo_msg_mb, sp, a2a_topo)
    kv_a2a_per = cm._a2a_per_op_time(kv_msg_mb, sp, a2a_topo)
    num_qo_ops = 2 * 2 * cm.l
    num_kv_ops = 2 * 2 * cm.l
    per_op_cpu = cm.ulysses_a2a_overhead_ms + cm.usp_a2a_overhead_extra_ms

    qo_total = qo_a2a_per * num_qo_ops
    kv_total = kv_a2a_per * num_kv_ops
    a2a_cpu = per_op_cpu * (num_qo_ops + num_kv_ops)

    # Ring component
    kv_h = cm.kv_hidden * kv_factor / sp
    fwd_comm = cm._p2p_fwd_comm_per_step(tokens, cp, kv_h, ring_topo)
    fwd_ring = fwd_comm * (cp - 1) * cm.l
    bwd_ring = fwd_ring * cm.ring_bwd_comm_ratio
    ring_overhead = cm.ring_step_overhead_ms * (cp - 1) * 2 * cm.l

    layer_extra = (cm.usp_layer_overhead_base_ms
                   + cm.usp_layer_overhead_per_sp_ms * sp) * cm.l

    total = qo_total + kv_total + a2a_cpu + fwd_ring + bwd_ring + ring_overhead + layer_extra
    return dict(qo=qo_total, kv=kv_total, a2a_cpu=a2a_cpu,
                ring_fwd=fwd_ring, ring_bwd=bwd_ring,
                ring_oh=ring_overhead, layer_extra=layer_extra,
                total=total,
                qo_msg_mb=qo_msg_mb, kv_msg_mb=kv_msg_mb,
                qo_per=qo_a2a_per, kv_per=kv_a2a_per,
                fwd_step=fwd_comm)


def ulysses_breakdown(cm, tokens, sp, placement="context_first"):
    a2a_topo = cm._get_topo(placement, "alltoall", sp, 1)
    q_factor, kv_factor = cm.head_padding_overhead(sp)
    qo_msg_mb = cm.h * q_factor * tokens * 2 / 1024 / 1024 / sp
    kv_msg_mb = cm.kv_hidden * kv_factor * tokens * 2 / 1024 / 1024 / sp
    qo_per = cm._a2a_per_op_time(qo_msg_mb, sp, a2a_topo)
    kv_per = cm._a2a_per_op_time(kv_msg_mb, sp, a2a_topo)
    num_qo_ops = 2 * 2 * cm.l
    num_kv_ops = 2 * 2 * cm.l
    qo_total = qo_per * num_qo_ops
    kv_total = kv_per * num_kv_ops
    cpu = cm.ulysses_a2a_overhead_ms * (num_qo_ops + num_kv_ops)
    total = qo_total + kv_total + cpu
    return dict(qo=qo_total, kv=kv_total, cpu=cpu, total=total,
                qo_msg_mb=qo_msg_mb, kv_msg_mb=kv_msg_mb,
                qo_per=qo_per, kv_per=kv_per)


def ring_breakdown(cm, tokens, cp, placement="context_first"):
    ring_topo = cm._get_topo(placement, "ring", 1, cp)
    fwd_step = cm._p2p_fwd_comm_per_step(tokens, cp, None, ring_topo)
    fwd_total = fwd_step * (cp - 1) * cm.l
    bwd_total = fwd_total * cm.ring_bwd_comm_ratio
    # Note: standalone ring (additive model) doesn't add ring_step_overhead
    # to comm_time; it's only in the overlap model.
    return dict(fwd=fwd_total, bwd=bwd_total,
                total=fwd_total + bwd_total, fwd_step=fwd_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=None,
                    help="If given, override cm.l for hypothetical analysis")
    args = ap.parse_args()

    cm = build_cm()
    if args.n_layers:
        cm.l = args.n_layers
    print(f"n_layers L={cm.l}")
    print(f"ulysses_a2a_overhead_ms={cm.ulysses_a2a_overhead_ms}")
    print(f"usp_a2a_overhead_extra_ms={cm.usp_a2a_overhead_extra_ms}")
    print(f"ring_step_overhead_ms={cm.ring_step_overhead_ms}")
    print(f"usp_layer_overhead_base_ms={cm.usp_layer_overhead_base_ms}")
    print(f"usp_layer_overhead_per_sp_ms={cm.usp_layer_overhead_per_sp_ms}")
    print(f"ring_bwd_comm_ratio={cm.ring_bwd_comm_ratio}")
    print()

    # Representative cells: short and long mb
    test_cases = [
        # (label, tokens, strategy fn args, attn_type)
        ("usp2x4 mb=11k (chunks=8 avg)", 11500, dict(sp=2, cp=4), "usp"),
        ("usp2x4 mb=92k (chunks=1)", 92256, dict(sp=2, cp=4), "usp"),
        ("ulysses8 mb=11k", 11500, dict(sp=8), "ulysses"),
        ("ulysses8 mb=92k", 92256, dict(sp=8), "ulysses"),
        ("ring8 mb=11k", 11500, dict(cp=8), "ring"),
        ("ring8 mb=92k", 92256, dict(cp=8), "ring"),
    ]

    for label, tokens, kw, attn in test_cases:
        print(f"=== {label} ({attn}) ===")
        if attn == "usp":
            b = usp_breakdown(cm, tokens, **kw)
            print(f"  msg_mb: qo={b['qo_msg_mb']:.3f}  kv={b['kv_msg_mb']:.3f}")
            print(f"  per-op: qo={b['qo_per']:.3f}ms  kv={b['kv_per']:.3f}ms"
                  f"  fwd_step={b['fwd_step']:.3f}ms")
            print(f"  components ({cm.l} layers, fwd+bwd):")
            print(f"    qo_a2a:        {b['qo']:>6.1f} ms   ({b['qo']/b['total']*100:>4.0f}%)")
            print(f"    kv_a2a:        {b['kv']:>6.1f} ms   ({b['kv']/b['total']*100:>4.0f}%)")
            print(f"    a2a_cpu_oh:    {b['a2a_cpu']:>6.1f} ms   ({b['a2a_cpu']/b['total']*100:>4.0f}%)")
            print(f"    ring_fwd:      {b['ring_fwd']:>6.1f} ms   ({b['ring_fwd']/b['total']*100:>4.0f}%)")
            print(f"    ring_bwd:      {b['ring_bwd']:>6.1f} ms   ({b['ring_bwd']/b['total']*100:>4.0f}%)")
            print(f"    ring_step_oh:  {b['ring_oh']:>6.1f} ms   ({b['ring_oh']/b['total']*100:>4.0f}%)")
            print(f"    layer_extra:   {b['layer_extra']:>6.1f} ms   ({b['layer_extra']/b['total']*100:>4.0f}%)")
            print(f"  TOTAL:          {b['total']:>6.1f} ms")
        elif attn == "ulysses":
            b = ulysses_breakdown(cm, tokens, **kw)
            print(f"  msg_mb: qo={b['qo_msg_mb']:.3f}  kv={b['kv_msg_mb']:.3f}")
            print(f"  per-op: qo={b['qo_per']:.3f}ms  kv={b['kv_per']:.3f}ms")
            print(f"  components:")
            print(f"    qo_a2a:        {b['qo']:>6.1f} ms   ({b['qo']/b['total']*100:>4.0f}%)")
            print(f"    kv_a2a:        {b['kv']:>6.1f} ms   ({b['kv']/b['total']*100:>4.0f}%)")
            print(f"    cpu_overhead:  {b['cpu']:>6.1f} ms   ({b['cpu']/b['total']*100:>4.0f}%)")
            print(f"  TOTAL:          {b['total']:>6.1f} ms")
        elif attn == "ring":
            b = ring_breakdown(cm, tokens, **kw)
            print(f"  fwd_step={b['fwd_step']:.3f}ms")
            print(f"  components:")
            print(f"    fwd:           {b['fwd']:>6.1f} ms   ({b['fwd']/b['total']*100:>4.0f}%)")
            print(f"    bwd:           {b['bwd']:>6.1f} ms   ({b['bwd']/b['total']*100:>4.0f}%)")
            print(f"  TOTAL:          {b['total']:>6.1f} ms")
        print()


if __name__ == "__main__":
    sys.exit(main() or 0)
