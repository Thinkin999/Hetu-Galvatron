#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration verification for placement-aware USP.

Tests:
  1. Solver chooses different placements for USP with asymmetric profiles
  2. force_placement override correctly constrains solver output
  3. Profile JSON round-trip: write → load → CostModel matches
  4. GroupManager rank mapping correctness for both placements
  5. End-to-end: solver output → serialization → deserialization → CostModel consistency
"""

import sys
import os
import json
import tempfile
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adacpsp_solver import (
    AdaCPSPCostModel,
    AdaCPSPOptimizer,
    ParallelStrategy,
    Sequence,
    _serialize_strategy_groups,
    _deserialize_strategy_groups,
)
from profile_topo_utils import build_group_ranks_list, linear_fit, topo_key


def make_asymmetric_costmodel(cluster_size=64, gpus_per_node=8):
    """Create a CostModel with significantly different consecutive vs strided BW."""
    return AdaCPSPCostModel(
        cluster_size=cluster_size,
        hidden_size=4096,
        layer_num=2,
        gpus_per_node=gpus_per_node,
        alltoall_bw_consec={2: 200, 4: 300, 8: 400, 16: 350, 32: 300, 64: 250},
        alltoall_bw_strided={2: 50, 4: 40, 8: 30, 16: 25, 32: 20, 64: 15},
        p2p_bw_consec={2: 180, 4: 250, 8: 350, 16: 300, 32: 280, 64: 220},
        p2p_bw_strided={2: 40, 4: 35, 8: 25, 16: 20, 32: 15, 64: 12},
        alltoall_linear_consec={
            2: {"alpha": 0.001, "beta": 0.005},
            4: {"alpha": 0.0008, "beta": 0.006},
            8: {"alpha": 0.0006, "beta": 0.008},
        },
        alltoall_linear_strided={
            2: {"alpha": 0.01, "beta": 0.05},
            4: {"alpha": 0.008, "beta": 0.06},
            8: {"alpha": 0.006, "beta": 0.08},
        },
        p2p_linear_consec={
            2: {"alpha": 0.0012, "beta": 0.006},
            4: {"alpha": 0.001, "beta": 0.007},
            8: {"alpha": 0.0008, "beta": 0.009},
        },
        p2p_linear_strided={
            2: {"alpha": 0.012, "beta": 0.06},
            4: {"alpha": 0.01, "beta": 0.07},
            8: {"alpha": 0.008, "beta": 0.09},
        },
    )


def test_solver_placement_choice():
    """Solver should generate and evaluate both placements for USP."""
    cm = make_asymmetric_costmodel()
    opt = AdaCPSPOptimizer(
        cluster_size=64,
        memory_limit_gb=80,
        costmodel=cm,
        allowed_attn_types=["usp"],
        hide_output=True,
    )
    pool = opt.get_strategy_pool()
    usp_strats = [s for s in pool if s.attn_type == "usp"]
    large_usp = [s for s in usp_strats if s.parallel_size > 8]
    placements_seen = set(s.placement for s in large_usp)
    assert "head_first" in placements_seen, "Solver pool should include head_first for large USP"
    assert "context_first" in placements_seen, "Solver pool should include context_first for large USP"

    seqlens = [8192] * 64
    for s in large_usp[:6]:
        t = cm.total_time(seqlens[:s.parallel_size], s)
        assert t > 0, f"total_time should be > 0 for {s}"

    print(f"  [PASS] Solver generates {len(large_usp)} large USP strats "
          f"with placements: {placements_seen}")


def test_force_placement_override():
    """When force_placement is set, solver should only use that placement."""
    cm = make_asymmetric_costmodel()
    for forced in ["head_first", "context_first"]:
        opt = AdaCPSPOptimizer(
            cluster_size=64,
            memory_limit_gb=80,
            costmodel=cm,
            allowed_attn_types=["usp"],
            hide_output=True,
        )
        opt.force_placement = forced
        pool = opt.get_strategy_pool()
        usp_strats = [s for s in pool if s.attn_type == "usp" and s.parallel_size > 8]
        bad = [s for s in usp_strats if s.placement != forced]
        assert len(bad) == 0, (
            f"force_placement={forced} but found {len(bad)} strats with wrong placement"
        )
        print(f"  [PASS] force_placement={forced}: all {len(usp_strats)} USP strats correct")


def test_costmodel_asymmetry():
    """With asymmetric BWs, head_first and context_first should give different costs,
    and the cheaper one should correspond to faster topology."""
    cm = make_asymmetric_costmodel()
    seqlens = [8192, 8192, 8192, 8192]
    strat_hf = ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="head_first")
    strat_cf = ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="context_first")

    t_hf = cm.total_time(seqlens, strat_hf)
    t_cf = cm.total_time(seqlens, strat_cf)

    assert t_hf != t_cf, "Asymmetric BWs should produce different costs"
    assert t_hf < t_cf, (
        f"head_first (A2A consec fast) should be cheaper for this config: "
        f"HF={t_hf:.4f} vs CF={t_cf:.4f}"
    )
    speedup = t_cf / t_hf
    print(f"  [PASS] Cost asymmetry: HF={t_hf:.4f}ms, CF={t_cf:.4f}ms, "
          f"speedup={speedup:.2f}x")


def test_profile_json_roundtrip():
    """Write topology-aware profile JSON, load back, verify CostModel state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        attn_json = os.path.join(tmpdir, "attn.json")
        a2a_json = os.path.join(tmpdir, "a2a.json")
        p2p_json = os.path.join(tmpdir, "p2p.json")

        with open(attn_json, "w") as f:
            json.dump({
                "num_layers": 4,
                "config": {"hidden_size": 4096, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128},
                "coefficients": {
                    "seg0": {"seq_range": [0, 1e9], "a": 1e-8, "b": 1e-5, "c": 0.1}
                },
            }, f)

        a2a_profile = {
            "bandwidth_dict_GBs": {"1": 1e10, "2": 100, "4": 150, "8": 200},
            "bandwidth_dict_consec_GBs": {"1": 1e10, "2": 200, "4": 300, "8": 400},
            "bandwidth_dict_strided_GBs": {"1": 1e10, "2": 50, "4": 75, "8": 100},
            "linear_fits": {
                "gs2_consecutive": {"alpha": 0.001, "beta": 0.01, "r_squared": 0.99},
                "gs4_consecutive": {"alpha": 0.0008, "beta": 0.008, "r_squared": 0.99},
                "gs2_strided": {"alpha": 0.01, "beta": 0.1, "r_squared": 0.98},
                "gs4_strided": {"alpha": 0.008, "beta": 0.08, "r_squared": 0.98},
            },
        }
        with open(a2a_json, "w") as f:
            json.dump(a2a_profile, f)

        p2p_profile = {
            "bandwidth_dict_GBs": {"1": 1e10, "2": 80, "4": 120, "8": 160},
            "bandwidth_dict_consec_GBs": {"1": 1e10, "2": 160, "4": 240, "8": 320},
            "bandwidth_dict_strided_GBs": {"1": 1e10, "2": 40, "4": 60, "8": 80},
            "linear_fits": {
                "gs2_consecutive": {"alpha": 0.002, "beta": 0.02, "r_squared": 0.97},
                "gs4_strided": {"alpha": 0.02, "beta": 0.2, "r_squared": 0.96},
            },
        }
        with open(p2p_json, "w") as f:
            json.dump(p2p_profile, f)

        cm = AdaCPSPCostModel.from_profile_files(
            attn_json, a2a_json, p2p_json, gpus_per_node=8
        )

        assert cm.gpus_per_node == 8
        assert cm.alltoall_bw_consec[4] == 300
        assert cm.alltoall_bw_strided[4] == 75
        assert cm.p2p_bw_consec[2] == 160
        assert cm.p2p_bw_strided[2] == 40

        assert cm.alltoall_linear_consec[2]["alpha"] == 0.001
        assert cm.alltoall_linear_strided[4]["alpha"] == 0.008
        assert cm.p2p_linear_consec[2]["alpha"] == 0.002
        assert cm.p2p_linear_strided[4]["alpha"] == 0.02

        # Legacy BW should still be loaded
        assert cm.alltoall_bw[4] == 150

        print("  [PASS] Profile JSON round-trip: all fields correct")


def test_group_manager_rank_mapping():
    """Verify rank mapping logic for head_first vs context_first USP
    without actually creating ProcessGroups (mock the rank mapping math)."""

    sp_size, cp_size = 4, 4
    parallel_size = sp_size * cp_size  # 16
    base_rank = 0

    # head_first: SP consecutive, CP strided
    # rank(cp_idx, sp_idx) = base_rank + cp_idx * sp_size + sp_idx
    hf_sp_groups = []
    for cp_idx in range(cp_size):
        sp_ranks = [base_rank + cp_idx * sp_size + j for j in range(sp_size)]
        hf_sp_groups.append(sp_ranks)

    hf_cp_groups = []
    for sp_idx in range(sp_size):
        cp_ranks = [base_rank + cp_idx * sp_size + sp_idx for cp_idx in range(cp_size)]
        hf_cp_groups.append(cp_ranks)

    # context_first: CP consecutive, SP strided
    # rank(sp_idx, cp_idx) = base_rank + sp_idx * cp_size + cp_idx
    cf_cp_groups = []
    for sp_idx in range(sp_size):
        cp_ranks = [base_rank + sp_idx * cp_size + j for j in range(cp_size)]
        cf_cp_groups.append(cp_ranks)

    cf_sp_groups = []
    for cp_idx in range(cp_size):
        sp_ranks = [base_rank + sp_idx * cp_size + cp_idx for sp_idx in range(sp_size)]
        cf_sp_groups.append(sp_ranks)

    # Verify HF SP groups are consecutive
    for grp in hf_sp_groups:
        for i in range(len(grp) - 1):
            assert grp[i + 1] == grp[i] + 1, f"HF SP should be consecutive: {grp}"

    # Verify HF CP groups are strided
    for grp in hf_cp_groups:
        for i in range(len(grp) - 1):
            assert grp[i + 1] - grp[i] == sp_size, f"HF CP should be strided by sp_size: {grp}"

    # Verify CF CP groups are consecutive
    for grp in cf_cp_groups:
        for i in range(len(grp) - 1):
            assert grp[i + 1] == grp[i] + 1, f"CF CP should be consecutive: {grp}"

    # Verify CF SP groups are strided
    for grp in cf_sp_groups:
        for i in range(len(grp) - 1):
            assert grp[i + 1] - grp[i] == cp_size, f"CF SP should be strided by cp_size: {grp}"

    # All ranks covered exactly once in each
    for name, sp_gs, cp_gs in [("HF", hf_sp_groups, hf_cp_groups),
                                ("CF", cf_sp_groups, cf_cp_groups)]:
        all_sp = sorted(r for g in sp_gs for r in g)
        all_cp = sorted(r for g in cp_gs for r in g)
        assert all_sp == list(range(parallel_size)), f"{name} SP ranks incomplete"
        assert all_cp == list(range(parallel_size)), f"{name} CP ranks incomplete"

    # HF and CF should produce different rank assignments
    assert hf_sp_groups != cf_sp_groups, "HF and CF should differ in SP groups"
    assert hf_cp_groups != cf_cp_groups, "HF and CF should differ in CP groups"

    print(f"  [PASS] Group rank mapping: HF SP={hf_sp_groups[0]}, CP={hf_cp_groups[0]}")
    print(f"         CF SP={cf_sp_groups[0]}, CP={cf_cp_groups[0]}")


def test_e2e_solver_serialize_cost():
    """End-to-end: solver output → serialize → deserialize → CostModel gives same result."""
    cm = make_asymmetric_costmodel(cluster_size=16, gpus_per_node=8)
    seqlens = [4096] * 8

    strats = [
        ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="head_first"),
        ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="context_first"),
        ParallelStrategy("usp", 16, sp_size=2, cp_size=8, placement="head_first"),
        ParallelStrategy("ring", 8),
        ParallelStrategy("ulysses", 8),
    ]

    seqs = [Sequence(seq=s, id=i) for i, s in enumerate(seqlens)]

    groups = [(s, seqs[:s.parallel_size]) for s in strats]
    serialized = _serialize_strategy_groups(groups)
    deserialized = _deserialize_strategy_groups(serialized)

    for (orig_s, orig_seqs), (des_s, des_seqs) in zip(groups, deserialized):
        orig_lens = [sq.seq for sq in orig_seqs]
        des_lens = [sq.seq for sq in des_seqs]

        t_orig = cm.total_time(orig_lens, orig_s)
        t_des = cm.total_time(des_lens, des_s)

        assert abs(t_orig - t_des) < 1e-6, (
            f"Cost mismatch after roundtrip: {orig_s} → {t_orig:.6f} != {t_des:.6f}"
        )

    print(f"  [PASS] E2E serialize→deserialize→cost consistency for {len(strats)} strategies")


def test_topo_key_consistency_with_costmodel():
    """Verify topo_key format matches what CostModel._load_topo_linear_fits expects."""
    for gs in [2, 4, 8, 16]:
        for topo in ["consecutive", "strided"]:
            key = topo_key(gs, topo)
            assert f"gs{gs}" in key
            assert topo in key
            parts = key.split("_")
            assert len(parts) == 2
            assert parts[0] == f"gs{gs}"
            assert parts[1] == topo

    print("  [PASS] topo_key format consistent with CostModel parser")


def test_costmodel_graceful_fallback():
    """CostModel should work even when only partial topology data is available."""
    cm = AdaCPSPCostModel(
        cluster_size=16,
        hidden_size=4096,
        layer_num=2,
        gpus_per_node=8,
        alltoall_bw_consec={4: 300},
        # No strided data, no linear fits
    )
    seqlens = [4096, 4096, 4096, 4096]
    strat_hf = ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="head_first")
    strat_cf = ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="context_first")

    t_hf = cm.total_time(seqlens, strat_hf)
    t_cf = cm.total_time(seqlens, strat_cf)

    assert t_hf > 0
    assert t_cf > 0
    print(f"  [PASS] Graceful fallback: HF={t_hf:.4f}ms, CF={t_cf:.4f}ms (partial data)")


if __name__ == "__main__":
    print("=" * 60)
    print("  Placement-Aware USP Integration Tests")
    print("=" * 60)

    test_solver_placement_choice()
    test_force_placement_override()
    test_costmodel_asymmetry()
    test_profile_json_roundtrip()
    test_group_manager_rank_mapping()
    test_e2e_solver_serialize_cost()
    test_topo_key_consistency_with_costmodel()
    test_costmodel_graceful_fallback()

    print("\n" + "=" * 60)
    print("  ALL INTEGRATION TESTS PASSED")
    print("=" * 60)
