#!/usr/bin/env python3
"""Regression tests for AdaCPSPCostModel head-padding and USP formula correctness.

Tests:
  1. USP additive ring comm uses padded kv_hidden.
  2. USP additive ring comm without padding matches kv_hidden/sp.
  3. USP additive A2A comm divides by parallel_size (sp*cp), not just sp.
  4. head_padding_extra_activation_mb uses parallel_size for USP.
  5. Ulysses total_time (additive) matches analytical formula.
  6. Ring overlap vs additive sanity: overlap <= additive.
  7. USP overlap A2A msg matches inline computation.

Usage:
  python tests/test_adacpsp_solver_padding.py
"""

import math
import os
import sys
import importlib.util

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOLVER_PATH = os.path.join(
    PROJECT_ROOT, "galvatron", "models", "varlen_llama_hf", "adacpsp_solver.py",
)


def _load_solver_module():
    spec = importlib.util.spec_from_file_location("adacpsp_solver", SOLVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_solver = _load_solver_module()
AdaCPSPCostModel = _solver.AdaCPSPCostModel
ParallelStrategy = _solver.ParallelStrategy


# ────────────────────────────────────────────────────────
# Test 1: USP additive ring comm uses padded kv_hidden
# ────────────────────────────────────────────────────────
def test_usp_additive_ring_comm_uses_padded_kv_hidden():
    """Ring side of USP additive must use kv_hidden * kv_factor / sp, not kv_hidden / sp."""
    model = AdaCPSPCostModel(
        cluster_size=16, hidden_size=28 * 128, layer_num=1, param_size_B=7.0,
        zero_stage=3, num_attention_heads=28, num_kv_heads=4, head_dim=128,
        cpt_alpha1=0.0, cpt_alpha2=0.0, cpt_beta1=0.0,
        alltoall_linear_fit={8: {"alpha": 0.0, "beta": 0.0}},
        p2p_linear_fit={2: {"alpha": 1.0, "beta": 0.0}},
        enable_overlap_model=False, bwd_fwd_ratio=0.0, ring_bwd_comm_ratio=2.0,
    )
    strategy = ParallelStrategy("usp", 16, sp_size=8, cp_size=2)
    seqlens = [8192]

    _, kv_factor = model.head_padding_overhead(8)
    assert kv_factor == 2.0

    T = sum(seqlens)
    kv_h = model.kv_hidden * kv_factor / 8
    single_kv_mb = (T / 2) * kv_h * 2 / 1024 / 1024
    kv_per_step_mb = 2 * single_kv_mb
    expected = kv_per_step_mb * 1 * 1 * (1 + 2.0)  # (cp-1) * L * (1 + bwd_ratio)

    got = model.total_time(seqlens, strategy)
    assert math.isclose(got, expected, rel_tol=1e-9), \
        f"Ring comm mismatch: {got} vs {expected}"


# ────────────────────────────────────────────────────────
# Test 2: USP additive ring comm without padding
# ────────────────────────────────────────────────────────
def test_usp_additive_ring_comm_matches_unpadded_case():
    model = AdaCPSPCostModel(
        cluster_size=16, hidden_size=40 * 128, layer_num=1, param_size_B=14.0,
        zero_stage=3, num_attention_heads=40, num_kv_heads=8, head_dim=128,
        cpt_alpha1=0.0, cpt_alpha2=0.0, cpt_beta1=0.0,
        alltoall_linear_fit={4: {"alpha": 0.0, "beta": 0.0}},
        p2p_linear_fit={2: {"alpha": 1.0, "beta": 0.0}},
        enable_overlap_model=False, bwd_fwd_ratio=0.0, ring_bwd_comm_ratio=2.0,
    )
    strategy = ParallelStrategy("usp", 8, sp_size=4, cp_size=2)
    seqlens = [4096]

    _, kv_factor = model.head_padding_overhead(4)
    assert kv_factor == 1.0

    T = sum(seqlens)
    kv_h = model.kv_hidden / 4
    single_kv_mb = (T / 2) * kv_h * 2 / 1024 / 1024
    kv_per_step_mb = 2 * single_kv_mb
    expected = kv_per_step_mb * 1 * 1 * (1 + 2.0)

    got = model.total_time(seqlens, strategy)
    assert math.isclose(got, expected, rel_tol=1e-9), \
        f"No-pad ring comm mismatch: {got} vs {expected}"


# ────────────────────────────────────────────────────────
# Test 3: USP additive A2A comm divides by sp*cp
# ────────────────────────────────────────────────────────
def test_usp_additive_a2a_msg_divides_by_parallel_size():
    """A2A msg in USP additive must be T*H*2 / (sp*cp), NOT T*H*2 / sp."""
    sp, cp = 4, 2
    model = AdaCPSPCostModel(
        cluster_size=sp * cp, hidden_size=40 * 128, layer_num=1, param_size_B=14.0,
        zero_stage=3, num_attention_heads=40, num_kv_heads=8, head_dim=128,
        cpt_alpha1=0.0, cpt_alpha2=0.0, cpt_beta1=0.0,
        # A2A: time = alpha * msg_MB (beta=0), so time is exactly proportional to msg size
        alltoall_linear_fit={sp: {"alpha": 1.0, "beta": 0.0}},
        # No ring comm (to isolate A2A term)
        p2p_linear_fit={cp: {"alpha": 0.0, "beta": 0.0}},
        enable_overlap_model=False, bwd_fwd_ratio=0.0, ring_bwd_comm_ratio=0.0,
    )
    strategy = ParallelStrategy("usp", sp * cp, sp_size=sp, cp_size=cp)
    seqlens = [16384]
    T = sum(seqlens)

    q_factor, kv_factor = model.head_padding_overhead(sp)
    assert q_factor == 1.0 and kv_factor == 1.0  # 8 kv heads divisible by 4

    # Expected: msg_mb must divide by parallel_size = sp * cp
    parallel_size = sp * cp
    qo_msg = model.h * q_factor * T * 2 / 1024 / 1024 / parallel_size
    kv_msg = model.kv_hidden * kv_factor * T * 2 / 1024 / 1024 / parallel_size

    # alpha=1, beta=0 → per_op_time = msg_mb. 4 QO + 4 KV ops per layer (fwd+bwd).
    expected_a2a = (2 * qo_msg + 2 * kv_msg) * 2 * model.l

    got = model.total_time(seqlens, strategy)
    assert math.isclose(got, expected_a2a, rel_tol=1e-9), \
        f"USP A2A msg mismatch: got={got:.6f}, expected={expected_a2a:.6f}"

    # Compare: the WRONG formula would divide by sp only → cp× larger
    wrong_qo_msg = model.h * q_factor * T * 2 / 1024 / 1024 / sp
    wrong_kv_msg = model.kv_hidden * kv_factor * T * 2 / 1024 / 1024 / sp
    wrong_a2a = (2 * wrong_qo_msg + 2 * wrong_kv_msg) * 2 * model.l
    assert math.isclose(wrong_a2a, expected_a2a * cp, rel_tol=1e-6), \
        f"Wrong formula should be exactly cp={cp}× larger: {wrong_a2a} vs {expected_a2a * cp}"


# ────────────────────────────────────────────────────────
# Test 4: head_padding_extra_activation_mb uses parallel_size
# ────────────────────────────────────────────────────────
def test_head_padding_extra_activation_uses_parallel_size():
    """For USP, extra activation should divide T by sp*cp, not sp."""
    model = AdaCPSPCostModel(
        cluster_size=16, hidden_size=28 * 128, layer_num=32, param_size_B=7.0,
        zero_stage=3, num_attention_heads=28, num_kv_heads=4, head_dim=128,
    )
    seqlens = [8192, 8192]
    sp, cp = 8, 2
    parallel_size = sp * cp

    extra = model.head_padding_extra_activation_mb(seqlens, sp, parallel_size)

    # Manual calculation
    q_factor, kv_factor = model.head_padding_overhead(sp)
    assert q_factor == 2.0 and kv_factor == 2.0

    T = sum(seqlens)
    tokens_per_device = T / parallel_size  # T/(8*2) = T/16
    q_extra_per_token = (q_factor - 1.0) * model.n_heads * model.head_dim * 2
    kv_extra_per_token = (kv_factor - 1.0) * model.n_kv_heads * model.head_dim * 2 * 2
    expected = tokens_per_device * (q_extra_per_token + kv_extra_per_token) / 1024 / 1024

    assert math.isclose(extra, expected, rel_tol=1e-9), \
        f"Extra activation mismatch: {extra} vs {expected}"

    # The old formula would have used T/sp = T/8 → 2× larger
    extra_old = model.head_padding_extra_activation_mb(seqlens, sp)  # parallel_size defaults to sp
    assert math.isclose(extra_old, extra * cp, rel_tol=1e-9), \
        f"Old formula should be cp={cp}× larger: {extra_old} vs {extra * cp}"


# ────────────────────────────────────────────────────────
# Test 5: Ulysses total_time additive matches formula
# ────────────────────────────────────────────────────────
def test_ulysses_total_time_additive():
    """Ulysses total = (1+r)*L*Σ f(s)*q_factor/P + alltoall_time."""
    model = AdaCPSPCostModel(
        cluster_size=8, hidden_size=28 * 128, layer_num=2, param_size_B=7.0,
        zero_stage=3, num_attention_heads=28, num_kv_heads=4, head_dim=128,
        cpt_alpha1=1e-7, cpt_alpha2=0.0, cpt_beta1=0.0,
        alltoall_linear_fit={4: {"alpha": 0.5, "beta": 0.1}},
        enable_overlap_model=False, bwd_fwd_ratio=2.0,
    )
    strategy = ParallelStrategy("ulysses", 4)
    seqlens = [4096, 8192]

    q_factor, kv_factor = model.head_padding_overhead(4)
    assert q_factor == 1.0  # 28 % 4 == 0

    # Compute
    fwd_compute = sum(model._eval_piecewise(s) * q_factor / 4 for s in seqlens) * model.l
    total_compute = fwd_compute * (1 + 2.0)

    # A2A comm
    T = sum(seqlens)
    qo_msg = model.h * q_factor * T * 2 / 1024 / 1024 / 4
    kv_msg = model.kv_hidden * kv_factor * T * 2 / 1024 / 1024 / 4
    qo_time = 0.5 * qo_msg + 0.1
    kv_time = 0.5 * kv_msg + 0.1
    a2a_comm = (qo_time * 4 + kv_time * 4) * model.l

    expected = total_compute + a2a_comm
    got = model.total_time(seqlens, strategy)
    assert math.isclose(got, expected, rel_tol=1e-6), \
        f"Ulysses total mismatch: {got} vs {expected}"


# ────────────────────────────────────────────────────────
# Test 6: Ring overlap ≤ Ring additive (sanity)
# ────────────────────────────────────────────────────────
def test_ring_overlap_leq_additive():
    """Ring overlap model should be ≤ additive model (overlap saves time)."""
    for cp in [2, 4, 8]:
        model = AdaCPSPCostModel(
            cluster_size=cp, hidden_size=40 * 128, layer_num=4, param_size_B=14.0,
            zero_stage=3, num_attention_heads=40, num_kv_heads=8, head_dim=128,
            cpt_alpha1=3e-8, cpt_alpha2=0.0, cpt_beta1=0.1,
            p2p_linear_fit={cp: {"alpha": 0.3, "beta": 0.05}},
            bwd_fwd_ratio=2.0, ring_bwd_comm_ratio=2.0,
            overlap_leakage=0.1,
        )
        strategy = ParallelStrategy("ring", cp)
        seqlens = [65536]

        model.enable_overlap_model = True
        t_overlap = model.total_time(seqlens, strategy)

        model.enable_overlap_model = False
        t_additive = model.total_time(seqlens, strategy)

        assert t_overlap <= t_additive * 1.01, \
            f"cp={cp}: overlap={t_overlap:.3f} > additive={t_additive:.3f}"


# ────────────────────────────────────────────────────────
# Test 7: USP overlap A2A msg consistency
# ────────────────────────────────────────────────────────
def test_usp_overlap_a2a_msg_consistent():
    """USP overlap and USP additive A2A comm should agree on message size."""
    model = AdaCPSPCostModel(
        cluster_size=16, hidden_size=28 * 128, layer_num=1, param_size_B=7.0,
        zero_stage=3, num_attention_heads=28, num_kv_heads=4, head_dim=128,
        cpt_alpha1=0.0, cpt_alpha2=0.0, cpt_beta1=0.0,
        alltoall_linear_fit={8: {"alpha": 1.0, "beta": 0.0}},
        p2p_linear_fit={2: {"alpha": 0.0, "beta": 0.0}},
        bwd_fwd_ratio=0.0, ring_bwd_comm_ratio=0.0,
    )
    strategy = ParallelStrategy("usp", 16, sp_size=8, cp_size=2)
    seqlens = [8192]

    # With overlap
    model.enable_overlap_model = True
    t_overlap = model.total_time(seqlens, strategy)

    # Without overlap
    model.enable_overlap_model = False
    t_additive = model.total_time(seqlens, strategy)

    # Both should give identical A2A comm (compute=0, ring=0, bwd_ratio=0)
    assert math.isclose(t_overlap, t_additive, rel_tol=1e-6), \
        f"USP A2A disagrees: overlap={t_overlap:.6f} vs additive={t_additive:.6f}"


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────
def main():
    tests = [
        test_usp_additive_ring_comm_uses_padded_kv_hidden,
        test_usp_additive_ring_comm_matches_unpadded_case,
        test_usp_additive_a2a_msg_divides_by_parallel_size,
        test_head_padding_extra_activation_uses_parallel_size,
        test_ulysses_total_time_additive,
        test_ring_overlap_leq_additive,
        test_usp_overlap_a2a_msg_consistent,
    ]
    for t in tests:
        t()
        print(f"✓ {t.__name__}")
    print(f"\nAll {len(tests)} tests PASSED ✓")


if __name__ == "__main__":
    main()
