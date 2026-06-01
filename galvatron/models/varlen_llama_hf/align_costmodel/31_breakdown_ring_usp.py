"""Decompose ring/usp predicted cost at varying token counts.

For ring8 chunks=8 we see ~1020 ms floor per microbatch even at tiny tokens.
This script reproduces the cost model's prediction and prints WHICH components
contribute that floor — needed to diagnose why ring/usp are over-predicted
at small per-rank tokens (~45-56% in github multi-mb validation).

Usage:
  python 31_breakdown_ring_usp.py
"""

from __future__ import annotations
import os, sys, glob, json

# Make src/Hetu-Galvatron importable
REPO_ROOT = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "galvatron/site_package"))

from galvatron.models.varlen_llama_hf.adacpsp_solver import (
    AdaCPSPCostModel,
    ParallelStrategy,
)


def latest_profile(pattern: str, configs_dir: str):
    paths = sorted(glob.glob(os.path.join(configs_dir, pattern)), reverse=True)
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
            return p, data
        except Exception:
            pass
    return None, None


def build_costmodel() -> AdaCPSPCostModel:
    configs_dir = os.path.join(REPO_ROOT, "galvatron/models/varlen_llama_hf/configs")

    attn_path, attn = latest_profile("profile_validate_*.json", configs_dir)
    comm_path, comm = latest_profile("comm_profile_*.json", configs_dir)
    resid_path, resid = latest_profile("residual_profile_*.json", configs_dir)
    bdec_path, bdec = latest_profile("b_decomp_profile_*.json", configs_dir)

    print("Profiles loaded:")
    print(f"  attention : {os.path.basename(attn_path) if attn_path else None}")
    print(f"  comm      : {os.path.basename(comm_path) if comm_path else None}")
    print(f"  residual  : {os.path.basename(resid_path) if resid_path else None}")
    print(f"  b-decomp  : {os.path.basename(bdec_path) if bdec_path else None}")
    print()

    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn_path,
        comm_profile_json=comm_path,
        cluster_size=16,
        gpus_per_node=8,
    )
    if resid is not None:
        cm.apply_residual_profile(resid)
    if bdec is not None:
        cm.apply_b_decomp_profile(bdec)
    return cm


def breakdown_ring(cm, seqlens, cp_size: int):
    """Reproduce _total_time_ring_overlap math and print components."""
    strat = ParallelStrategy(
        attn_type="ring", parallel_size=cp_size, sp_size=1, cp_size=cp_size,
        placement="context_first",
    )
    total_tokens = sum(seqlens)
    step_compute = cm._ring_step_compute_per_layer(seqlens, strat)
    ring_topo = cm._get_topo(strat.placement, "ring", strat.sp_size, cp_size)
    fwd_comm = cm._p2p_fwd_comm_per_step(total_tokens, cp_size, topo=ring_topo)
    bwd_comm = cm._p2p_bwd_comm_per_step(total_tokens, cp_size, topo=ring_topo)
    overlap_fwd = cm._overlap_time(step_compute, fwd_comm)
    overlap_bwd = cm._overlap_time(step_compute * cm.bwd_fwd_ratio, bwd_comm)
    ring_overhead = cm.ring_step_overhead_ms

    # Per-layer breakdown
    fwd_per_layer = ((cp_size - 1) * (overlap_fwd + ring_overhead)
                     + step_compute)
    bwd_per_layer = ((cp_size - 1) * (overlap_bwd + ring_overhead)
                     + step_compute * cm.bwd_fwd_ratio)
    attn_total = (fwd_per_layer + bwd_per_layer) * cm.l
    residual_total = cm.residual_time(seqlens, strat)

    return {
        "seqlens": seqlens, "total_tokens": total_tokens,
        "step_compute_per_layer_ms": step_compute,
        "fwd_comm_per_step_ms": fwd_comm,
        "bwd_comm_per_step_ms": bwd_comm,
        "overlap_fwd_per_step_ms": overlap_fwd,
        "overlap_bwd_per_step_ms": overlap_bwd,
        "ring_step_overhead_ms": ring_overhead,
        "fwd_per_layer_ms": fwd_per_layer,
        "bwd_per_layer_ms": bwd_per_layer,
        "attn_total_ms": attn_total,                  # (fwd+bwd) * L
        "residual_ms": residual_total,
        "total_ms": attn_total + residual_total,
    }


def breakdown_usp(cm, seqlens, sp_size: int, cp_size: int):
    strat = ParallelStrategy(
        attn_type="usp", parallel_size=sp_size * cp_size,
        sp_size=sp_size, cp_size=cp_size, placement="context_first",
    )
    total_tokens = sum(seqlens)
    parallel_size = sp_size * cp_size
    a2a_topo = cm._get_topo(strat.placement, "alltoall", sp_size, cp_size)
    ring_topo = cm._get_topo(strat.placement, "ring", sp_size, cp_size)

    q_factor, kv_factor = cm.head_padding_overhead(sp_size)
    qo_msg_mb = cm.h * q_factor * total_tokens * 2 / 1024 / 1024 / parallel_size
    kv_msg_mb = cm.kv_hidden * kv_factor * total_tokens * 2 / 1024 / 1024 / parallel_size
    qo_a2a = cm._a2a_per_op_time(qo_msg_mb, sp_size, a2a_topo)
    kv_a2a = cm._a2a_per_op_time(kv_msg_mb, sp_size, a2a_topo)
    per_op_cpu = cm.ulysses_a2a_overhead_ms + cm.usp_a2a_overhead_extra_ms
    a2a_per_dir = 2 * qo_a2a + 2 * kv_a2a + 4 * per_op_cpu
    a2a_total = (a2a_per_dir + a2a_per_dir) * cm.l   # fwd + bwd

    step_compute = cm._ring_step_compute_per_layer(seqlens, strat)
    kv_h_after_uly = cm.kv_hidden * kv_factor / sp_size
    fwd_comm_step = cm._p2p_fwd_comm_per_step(total_tokens, cp_size, kv_h_after_uly, ring_topo)
    bwd_comm_step = cm._p2p_bwd_comm_per_step(total_tokens, cp_size, kv_h_after_uly, ring_topo)

    ring_fwd_per_layer = ((cp_size - 1) * (
        cm._overlap_time(step_compute, fwd_comm_step) + cm.ring_step_overhead_ms
    ) + step_compute)
    bwd_step = step_compute * cm.bwd_fwd_ratio
    ring_bwd_per_layer = ((cp_size - 1) * (
        cm._overlap_time(bwd_step, bwd_comm_step) + cm.ring_step_overhead_ms
    ) + bwd_step)
    ring_total = (ring_fwd_per_layer + ring_bwd_per_layer) * cm.l

    layer_extra = (cm.usp_layer_overhead_base_ms + cm.usp_layer_overhead_per_sp_ms * sp_size) * cm.l
    residual_total = cm.residual_time(seqlens, strat)

    return {
        "seqlens": seqlens, "total_tokens": total_tokens,
        "step_compute_per_layer_ms": step_compute,
        "qo_a2a_per_op_ms": qo_a2a, "kv_a2a_per_op_ms": kv_a2a,
        "per_op_cpu_ms": per_op_cpu,
        "a2a_total_ms": a2a_total,
        "ring_total_ms": ring_total,
        "layer_extra_ms": layer_extra,
        "residual_ms": residual_total,
        "total_ms": a2a_total + ring_total + layer_extra + residual_total,
    }


def main():
    cm = build_costmodel()
    print(f"Cost model parameters:")
    print(f"  ring_step_overhead_ms     = {cm.ring_step_overhead_ms}")
    print(f"  ulysses_a2a_overhead_ms   = {cm.ulysses_a2a_overhead_ms}")
    print(f"  usp_a2a_overhead_extra_ms = {cm.usp_a2a_overhead_extra_ms}")
    print(f"  usp_layer_overhead_base_ms = {cm.usp_layer_overhead_base_ms}")
    print(f"  usp_layer_overhead_per_sp_ms = {cm.usp_layer_overhead_per_sp_ms}")
    print(f"  bwd_fwd_ratio              = {cm.bwd_fwd_ratio}")
    print(f"  num_layers (L)             = {cm.l}")
    print(f"  residual_a_per_sp          = {dict(cm.residual_a_per_sp)}")
    print(f"  residual_b_per_sp          = {dict(cm.residual_b_per_sp)}")
    print()

    print("=" * 80)
    print("RING8 breakdown at varying group tokens (cp=8, sp=1)")
    print("=" * 80)
    test_cases_ring = [
        [160], [640], [1568], [5888], [10208], [29760],   # actual sizes from ring8_c8 iter 5
        [160] * 8, [640] * 8,                               # multi-seq packing
    ]
    for seqs in test_cases_ring:
        bd = breakdown_ring(cm, seqs, cp_size=8)
        n_seqs = len(seqs)
        T = bd['total_tokens']
        per_rank = T // 8
        print(f"\nseqlens={seqs[:3]}{'...' if n_seqs>3 else ''} (n={n_seqs}, T={T}, per-rank={per_rank})")
        print(f"  step_compute/layer  = {bd['step_compute_per_layer_ms']:8.4f} ms")
        print(f"  fwd_comm/step       = {bd['fwd_comm_per_step_ms']:8.4f} ms")
        print(f"  bwd_comm/step       = {bd['bwd_comm_per_step_ms']:8.4f} ms")
        print(f"  overlap_fwd/step    = {bd['overlap_fwd_per_step_ms']:8.4f} ms")
        print(f"  overlap_bwd/step    = {bd['overlap_bwd_per_step_ms']:8.4f} ms")
        print(f"  ring_step_overhead  = {bd['ring_step_overhead_ms']:8.4f} ms")
        print(f"  fwd_per_layer       = {bd['fwd_per_layer_ms']:8.2f} ms = (cp-1)·(overlap+oh) + step_compute")
        print(f"                      = 7·({bd['overlap_fwd_per_step_ms']:.2f}+{bd['ring_step_overhead_ms']:.2f}) + {bd['step_compute_per_layer_ms']:.2f}")
        print(f"  bwd_per_layer       = {bd['bwd_per_layer_ms']:8.2f} ms")
        print(f"  attn_total (L={cm.l}) = {bd['attn_total_ms']:8.2f} ms")
        print(f"  residual_ms         = {bd['residual_ms']:8.2f} ms")
        print(f"  TOTAL               = {bd['total_ms']:8.2f} ms")

    print("\n" + "=" * 80)
    print("USP 2x4 breakdown at varying group tokens (sp=2, cp=4)")
    print("=" * 80)
    test_cases_usp = [
        [160], [640], [1568], [5888], [29760],
    ]
    for seqs in test_cases_usp:
        bd = breakdown_usp(cm, seqs, sp_size=2, cp_size=4)
        T = bd['total_tokens']
        print(f"\nseqlens={seqs} (T={T}, per-rank={T // 8})")
        print(f"  step_compute/layer  = {bd['step_compute_per_layer_ms']:8.4f} ms")
        print(f"  qo_a2a/op           = {bd['qo_a2a_per_op_ms']:8.4f} ms")
        print(f"  kv_a2a/op           = {bd['kv_a2a_per_op_ms']:8.4f} ms")
        print(f"  per_op_cpu_overhead = {bd['per_op_cpu_ms']:8.4f} ms")
        print(f"  a2a_total (L={cm.l}) = {bd['a2a_total_ms']:8.2f} ms")
        print(f"  ring_total (L={cm.l})= {bd['ring_total_ms']:8.2f} ms")
        print(f"  layer_extra         = {bd['layer_extra_ms']:8.2f} ms")
        print(f"  residual_ms         = {bd['residual_ms']:8.2f} ms")
        print(f"  TOTAL               = {bd['total_ms']:8.2f} ms")


if __name__ == "__main__":
    main()
