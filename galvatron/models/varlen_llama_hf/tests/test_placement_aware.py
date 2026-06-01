#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for placement-aware USP.

Covers:
  1. CostModel: head_first vs context_first produce different costs
  2. Solver: correct placement selection given asymmetric bandwidths
  3. Serialization: 6-tuple round-trip for strategy groups
  4. ParallelStrategy: placement field in hash/eq/repr
  5. profile_topo_utils: build_group_ranks_list correctness
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adacpsp_solver import (
    AdaCPSPCostModel,
    AdaCPSPOptimizer,
    ParallelStrategy,
    Sequence,
    _serialize_strategy_groups,
    _deserialize_strategy_groups,
    _serialize_seqs,
)
from profile_topo_utils import build_group_ranks_list, linear_fit, topo_key, parse_topo_key


def test_parallel_strategy_placement():
    """ParallelStrategy stores placement and uses it in hash/eq."""
    s1 = ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="head_first")
    s2 = ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="context_first")
    s3 = ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="head_first")

    assert s1 != s2, "Different placements should not be equal"
    assert s1 == s3, "Same placements should be equal"
    assert hash(s1) != hash(s2), "Different placements should have different hashes"
    assert hash(s1) == hash(s3)
    assert "hf" in repr(s1)
    assert "cf" in repr(s2)
    print("  [PASS] ParallelStrategy placement")


def test_ulysses_ring_placement_fixed():
    """Pure Ulysses uses head_first; pure Ring uses context_first."""
    s_u = ParallelStrategy("ulysses", 8)
    s_r = ParallelStrategy("ring", 8)
    assert s_u.placement == "head_first"
    assert s_r.placement == "context_first"
    print("  [PASS] Ulysses/Ring placement fixed")


def test_costmodel_topo_routing():
    """CostModel._get_topo returns correct topology for each placement."""
    # Pure ulysses (cp=1): always consecutive regardless of placement
    assert AdaCPSPCostModel._get_topo("context_first", "alltoall", sp_size=8, cp_size=1) == "consecutive"
    assert AdaCPSPCostModel._get_topo("head_first", "alltoall", sp_size=8, cp_size=1) == "consecutive"
    # Pure ring (sp=1): always consecutive regardless of placement
    assert AdaCPSPCostModel._get_topo("context_first", "ring", sp_size=1, cp_size=8) == "consecutive"
    assert AdaCPSPCostModel._get_topo("head_first", "ring", sp_size=1, cp_size=8) == "consecutive"
    # USP: placement determines which gets consecutive
    assert AdaCPSPCostModel._get_topo("head_first", "alltoall", sp_size=4, cp_size=4) == "consecutive"
    assert AdaCPSPCostModel._get_topo("head_first", "ring", sp_size=4, cp_size=4) == "strided"
    assert AdaCPSPCostModel._get_topo("context_first", "alltoall", sp_size=4, cp_size=4) == "strided"
    assert AdaCPSPCostModel._get_topo("context_first", "ring", sp_size=4, cp_size=4) == "consecutive"
    print("  [PASS] _get_topo routing")


def test_costmodel_different_placements():
    """CostModel produces different USP costs for different placements when
    consecutive and strided bandwidths differ significantly."""
    cm = AdaCPSPCostModel(
        cluster_size=64,
        hidden_size=4096,
        layer_num=2,
        gpus_per_node=8,
        alltoall_linear_consec={4: {"alpha": 0.001, "beta": 0.01}},
        alltoall_linear_strided={4: {"alpha": 0.01, "beta": 0.1}},
        p2p_linear_consec={4: {"alpha": 0.001, "beta": 0.01}},
        p2p_linear_strided={4: {"alpha": 0.01, "beta": 0.1}},
    )

    seqlens = [8192, 8192, 8192, 8192]

    strat_hf = ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="head_first")
    strat_cf = ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="context_first")

    time_hf = cm.total_time(seqlens, strat_hf)
    time_cf = cm.total_time(seqlens, strat_cf)

    assert time_hf != time_cf, (
        f"head_first ({time_hf:.4f}) and context_first ({time_cf:.4f}) "
        f"should produce different times with asymmetric bandwidths"
    )
    assert time_hf > 0 and time_cf > 0
    print(f"  [PASS] CostModel different placements: HF={time_hf:.4f}ms, CF={time_cf:.4f}ms")


def test_costmodel_backward_compat():
    """CostModel still works without topology-aware data (backward compat)."""
    cm = AdaCPSPCostModel(
        cluster_size=8,
        hidden_size=4096,
        layer_num=2,
    )
    seqlens = [4096, 4096]
    strat = ParallelStrategy("usp", 8, sp_size=2, cp_size=4, placement="context_first")
    t = cm.total_time(seqlens, strat)
    assert t > 0
    print(f"  [PASS] CostModel backward compat: {t:.4f}ms")


def test_serialization_roundtrip():
    """6-tuple serialization round-trips correctly."""
    strat1 = ParallelStrategy("usp", 16, sp_size=4, cp_size=4, placement="head_first")
    strat2 = ParallelStrategy("ring", 8)
    strat3 = ParallelStrategy("usp", 16, sp_size=2, cp_size=8, placement="context_first")

    seqs1 = [Sequence(seq=1024, id=0), Sequence(seq=2048, id=1)]
    seqs2 = [Sequence(seq=4096, id=2)]
    seqs3 = [Sequence(seq=512, id=3), Sequence(seq=512, id=4)]

    groups = [(strat1, seqs1), (strat2, seqs2), (strat3, seqs3)]

    serialized = _serialize_strategy_groups(groups)
    deserialized = _deserialize_strategy_groups(serialized)

    assert len(deserialized) == 3
    for (orig_strat, orig_seqs), (deser_strat, deser_seqs) in zip(groups, deserialized):
        assert orig_strat == deser_strat, f"{orig_strat} != {deser_strat}"
        assert len(orig_seqs) == len(deser_seqs)
        for os_, ds_ in zip(orig_seqs, deser_seqs):
            assert os_.seq == ds_.seq and os_.id == ds_.id

    print("  [PASS] 6-tuple serialization round-trip")


def test_serialization_backward_compat():
    """5-tuple (legacy) deserialization still works."""
    legacy_ser = [
        ("usp", 8, 2, 4, [(1024, 0), (2048, 1)]),
    ]
    deser = _deserialize_strategy_groups(legacy_ser)
    assert len(deser) == 1
    strat, seqs = deser[0]
    assert strat.placement == "context_first"
    assert strat.sp_size == 2
    assert strat.cp_size == 4
    print("  [PASS] 5-tuple backward compat deserialization")


def test_build_group_ranks_consecutive():
    """build_group_ranks_list consecutive produces correct groups."""
    groups = build_group_ranks_list(16, 4, "consecutive")
    assert len(groups) == 4
    assert groups[0] == [0, 1, 2, 3]
    assert groups[1] == [4, 5, 6, 7]
    assert groups[3] == [12, 13, 14, 15]
    print("  [PASS] build_group_ranks_list consecutive")


def test_build_group_ranks_strided():
    """build_group_ranks_list strided produces correct groups."""
    groups = build_group_ranks_list(16, 4, "strided")
    assert len(groups) == 4
    assert groups[0] == [0, 4, 8, 12]
    assert groups[1] == [1, 5, 9, 13]
    print("  [PASS] build_group_ranks_list strided")


def test_build_group_ranks_coverage():
    """All ranks appear exactly once in both topologies."""
    for ws in [8, 16, 32, 64]:
        for gs in [2, 4, 8]:
            if gs > ws:
                continue
            for topo in ["consecutive", "strided"]:
                groups = build_group_ranks_list(ws, gs, topo)
                all_ranks = sorted(r for g in groups for r in g)
                assert all_ranks == list(range(ws)), (
                    f"ws={ws}, gs={gs}, topo={topo}: ranks mismatch"
                )
    print("  [PASS] build_group_ranks_list coverage")


def test_linear_fit():
    """linear_fit produces correct slope and intercept."""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.1, 4.0, 6.1, 7.9, 10.1]  # ~2x + 0
    result = linear_fit(xs, ys)
    assert abs(result["alpha"] - 2.0) < 0.2
    assert abs(result["beta"] - 0.0) < 0.5
    assert result["r_squared"] > 0.99
    print(f"  [PASS] linear_fit: alpha={result['alpha']:.4f}, beta={result['beta']:.4f}")


def test_topo_key_roundtrip():
    """topo_key and parse_topo_key are inverses."""
    for gs in [2, 4, 8, 16, 32, 64]:
        for topo in ["consecutive", "strided"]:
            key = topo_key(gs, topo)
            gs2, topo2 = parse_topo_key(key)
            assert gs2 == gs and topo2 == topo, f"Roundtrip failed for gs={gs}, topo={topo}"
    print("  [PASS] topo_key roundtrip")


def test_strategy_pool_placement():
    """get_strategy_pool generates both placements for USP when parallel > gpus_per_node."""
    cm = AdaCPSPCostModel(cluster_size=16, gpus_per_node=8, hidden_size=4096, layer_num=2)
    opt = AdaCPSPOptimizer(cluster_size=16, memory_limit_gb=40, costmodel=cm,
                           allowed_attn_types=["ulysses", "ring", "usp"],
                           hide_output=True)
    pool = opt.get_strategy_pool()

    usp_strats = [s for s in pool if s.attn_type == "usp"]
    hf_strats = [s for s in usp_strats if s.placement == "head_first"]
    cf_strats = [s for s in usp_strats if s.placement == "context_first"]

    assert len(cf_strats) > 0, "Should have context_first USP strategies"

    large_usp = [s for s in usp_strats if s.parallel_size > 8]
    if large_usp:
        hf_large = [s for s in large_usp if s.placement == "head_first"]
        cf_large = [s for s in large_usp if s.placement == "context_first"]
        assert len(hf_large) > 0, "USP with parallel>gpn should have head_first"
        assert len(cf_large) > 0, "USP with parallel>gpn should have context_first"
        print(f"  [PASS] strategy_pool: {len(usp_strats)} USP strats "
              f"({len(hf_strats)} HF, {len(cf_strats)} CF)")
    else:
        print(f"  [PASS] strategy_pool: {len(usp_strats)} USP strats (all <= gpn, CF only)")


def test_from_profile_files_topo():
    """from_profile_files loads topology-aware data correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        attn_json = os.path.join(tmpdir, "attn.json")
        a2a_json = os.path.join(tmpdir, "a2a.json")
        p2p_json = os.path.join(tmpdir, "p2p.json")

        with open(attn_json, "w") as f:
            json.dump({
                "num_layers": 2,
                "config": {"hidden_size": 4096, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128},
                "coefficients": {"seg0": {"seq_range": [0, 1e9], "a": 1e-8, "b": 1e-5, "c": 0.1}},
            }, f)

        with open(a2a_json, "w") as f:
            json.dump({
                "bandwidth_dict_GBs": {"1": 1e10, "2": 100, "4": 150},
                "bandwidth_dict_consec_GBs": {"1": 1e10, "2": 200, "4": 300},
                "bandwidth_dict_strided_GBs": {"1": 1e10, "2": 50, "4": 75},
                "linear_fits": {
                    "gs2_consecutive": {"alpha": 0.001, "beta": 0.01, "r_squared": 0.99},
                    "gs2_strided": {"alpha": 0.01, "beta": 0.1, "r_squared": 0.98},
                },
            }, f)

        with open(p2p_json, "w") as f:
            json.dump({
                "bandwidth_dict_GBs": {"1": 1e10, "2": 80, "4": 120},
                "bandwidth_dict_consec_GBs": {"1": 1e10, "2": 160, "4": 240},
                "bandwidth_dict_strided_GBs": {"1": 1e10, "2": 40, "4": 60},
                "linear_fits": {
                    "gs2_consecutive": {"alpha": 0.002, "beta": 0.02, "r_squared": 0.97},
                    "gs4_strided": {"alpha": 0.02, "beta": 0.2, "r_squared": 0.96},
                },
            }, f)

        cm = AdaCPSPCostModel.from_profile_files(attn_json, a2a_json, p2p_json, gpus_per_node=8)

        assert cm.alltoall_bw_consec is not None
        assert cm.alltoall_bw_strided is not None
        assert cm.alltoall_bw_consec[2] == 200
        assert cm.alltoall_bw_strided[2] == 50
        assert cm.alltoall_linear_consec[2]["alpha"] == 0.001
        assert cm.alltoall_linear_strided[2]["alpha"] == 0.01
        assert cm.p2p_linear_consec[2]["alpha"] == 0.002
        assert cm.p2p_linear_strided is not None
        assert 4 in cm.p2p_linear_strided
        print("  [PASS] from_profile_files loads topology-aware data")


def test_from_attention_and_comm_profiles():
    """from_attention_and_comm_profiles loads unified comm profile correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        attn_json = os.path.join(tmpdir, "attn.json")
        comm_json = os.path.join(tmpdir, "comm.json")

        with open(attn_json, "w") as f:
            json.dump({
                "num_layers": 2,
                "attention": {
                    "segments": [{"range": [0, 1e9], "a": 1e-8, "b": 1e-5, "c": 0.1}],
                    "config": {"hidden_size": 4096, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128},
                },
            }, f)

        with open(comm_json, "w") as f:
            json.dump({
                "alltoall": {
                    "bandwidth_dict_GBs": {"1": 1e10, "2": 100, "4": 150},
                    "bandwidth_dict_consec_GBs": {"1": 1e10, "2": 200, "4": 300},
                    "bandwidth_dict_strided_GBs": {"1": 1e10, "2": 50, "4": 75},
                    "linear_fits": {
                        "gs2_consecutive": {"alpha": 0.001, "beta": 0.01},
                        "gs2_strided": {"alpha": 0.01, "beta": 0.1},
                    },
                    "interp_tables": {
                        "gs2_consecutive": [[4.0, 0.02], [8.0, 0.03]],
                        "gs2_strided": [[4.0, 0.12], [8.0, 0.18]],
                    },
                },
                "p2p_ring": {
                    "bandwidth_dict_GBs": {"1": 1e10, "2": 80, "4": 120},
                    "bandwidth_dict_consec_GBs": {"1": 1e10, "2": 160, "4": 240},
                    "bandwidth_dict_strided_GBs": {"1": 1e10, "2": 40, "4": 60},
                    "linear_fits": {
                        "gs2_consecutive": {"alpha": 0.002, "beta": 0.02},
                        "gs4_strided": {"alpha": 0.02, "beta": 0.2},
                    },
                    "ring_step_fits": {
                        "gs2_consecutive": {"alpha": 0.003, "beta": 0.03},
                        "gs4_strided": {"alpha": 0.03, "beta": 0.3},
                    },
                    "interp_tables": {
                        "gs2_consecutive": [[8.0, 0.04], [16.0, 0.07]],
                        "gs4_strided": [[8.0, 0.4], [16.0, 0.7]],
                    },
                },
            }, f)

        cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
            attn_json, comm_json, gpus_per_node=8
        )

        assert cm.alltoall_bw_consec[2] == 200
        assert cm.alltoall_bw_strided[2] == 50
        assert cm.p2p_bw_consec[2] == 160
        assert cm.p2p_bw_strided[2] == 40

        assert cm.alltoall_linear_consec[2]["alpha"] == 0.001
        assert cm.alltoall_linear_strided[2]["alpha"] == 0.01
        assert cm.p2p_linear_consec[2]["alpha"] == 0.002
        assert cm.p2p_linear_strided[4]["alpha"] == 0.02

        assert cm.p2p_ring_step_consec[2]["alpha"] == 0.003
        assert cm.p2p_ring_step_strided[4]["alpha"] == 0.03
        assert cm.a2a_interp_consec[2][0] == (4.0, 0.02)
        assert cm.a2a_interp_strided[2][1] == (8.0, 0.18)
        assert cm.p2p_ring_interp_consec[2][0] == (8.0, 0.04)
        assert cm.p2p_ring_interp_strided[4][1] == (16.0, 0.7)
        print("  [PASS] from_attention_and_comm_profiles loads unified comm profile")


if __name__ == "__main__":
    print("=" * 60)
    print("  Placement-Aware USP Unit Tests")
    print("=" * 60)

    test_parallel_strategy_placement()
    test_ulysses_ring_placement_fixed()
    test_costmodel_topo_routing()
    test_costmodel_different_placements()
    test_costmodel_backward_compat()
    test_serialization_roundtrip()
    test_serialization_backward_compat()
    test_build_group_ranks_consecutive()
    test_build_group_ranks_strided()
    test_build_group_ranks_coverage()
    test_linear_fit()
    test_topo_key_roundtrip()
    test_strategy_pool_placement()
    test_from_profile_files_topo()
    test_from_attention_and_comm_profiles()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
