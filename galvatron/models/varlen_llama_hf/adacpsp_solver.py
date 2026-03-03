#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaCPSP Solver — Adaptive Context Parallel & Sequence Parallel Strategy Optimizer

Extends the FlexSP approach to jointly select:
  1. Attention type  — Ulysses SP (All-to-All) or Ring Attention CP (P2P)
  2. Parallel size   — sp_size or cp_size (must be power of 2)

For each microbatch of variable-length sequences, the solver assigns every
sequence to a *parallel group*, where a group is characterised by
  (attn_type ∈ {"ulysses", "ring"}, parallel_size ∈ {1,2,4,8,...}).

The optimisation objective is identical to FlexSP:
  min  max_{group g}  time(g)
where
  time(g) = Σ_{seq in g} compute_time(seq, strategy) + comm_time(g, strategy)

Key differences from FlexSP:
  * Communication model uses both All-to-All **and** P2P bandwidth dicts.
  * Compute model uses **piecewise** quadratic fitting (per-segment coefficients).
  * Strategy pool includes (attn_type, parallel_size) pairs, not just sp_size.
  * The solver supports a combined mode where Ulysses and Ring can coexist, i.e.
    some groups use Ulysses while others use Ring within the same microbatch.

Usage (standalone):
    python adacpsp_solver.py --cluster_size 8 --memory_limit_gb 28 \
        --global_batch_size 64 --dataset github
"""

from __future__ import annotations

import os
import json
import random
import argparse
import time as time_module
from typing import List, Dict, Tuple, Optional, Union, Literal
from dataclasses import dataclass, field
from collections import Counter

import numpy as np

# ──────────────────────────────────────────────────────────
# Sequence & Bucket (self-contained, no C++ dependency)
# ──────────────────────────────────────────────────────────

@dataclass(order=True)
class Sequence:
    """A single sequence with length and id."""
    seq: int
    id: int = 0

    def __str__(self):
        return f"{self.id}-{self.seq}"


@dataclass
class SeqBucket:
    boundary: int
    seqs: List[Sequence] = field(default_factory=list)
    size: int = 0

    def add_seqs(self, seqs: List[Sequence]):
        self.size += len(seqs)
        self.seqs.extend(seqs)

    def random_pop_seqs(self, num: int = 1):
        if self.size == 0:
            return None
        if num > self.size:
            raise ValueError("num > bucket size")
        indices = random.sample(range(self.size), num)
        indices.sort(reverse=True)
        popped = []
        for idx in indices:
            popped.append(self.seqs.pop(idx))
        self.size -= num
        return popped

    def print(self):
        print(f"[Bucket] boundary={self.boundary}, size={self.size}, seqs=", end=" ")
        print_seqs(self.seqs)


def print_seqs(seqs: List[Sequence]):
    if not seqs:
        print("[]")
        return
    print("[" + ", ".join(str(s) for s in seqs) + "]")


def get_lens(seqs: List[Sequence]) -> List[int]:
    return [s.seq for s in seqs]


def bucketing_seqs(sequences: List[Sequence], B: int):
    """DP-based optimal sequence bucketing (minimise padding waste)."""
    n = len(sequences)
    sequences_sorted = sorted(sequences)
    inf = float("inf")
    dp = [[inf] * (B + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + sequences_sorted[i - 1].seq
    for b in range(1, B + 1):
        for i in range(1, n + 1):
            for j in range(i):
                cost = sequences_sorted[i - 1].seq * (i - j) - (prefix_sum[i] - prefix_sum[j])
                if dp[j][b - 1] + cost < dp[i][b]:
                    dp[i][b] = dp[j][b - 1] + cost
    avg_error = dp[n][B] / n
    buckets = []
    current = n
    for b in range(B, 0, -1):
        for i in range(current):
            cost = sequences_sorted[current - 1].seq * (current - i) - (prefix_sum[current] - prefix_sum[i])
            if dp[i][b - 1] + cost == dp[current][b]:
                bucket = SeqBucket(sequences_sorted[current - 1].seq)
                bucket.add_seqs(sequences_sorted[i:current])
                buckets.append(bucket)
                current = i
                break
    if current > 0:
        bucket = SeqBucket(float("inf"))
        bucket.add_seqs(sequences_sorted[:current])
        buckets.append(bucket)
    buckets.reverse()
    return buckets, avg_error


def chunk_globalbatch(seqs_gb: List[Sequence], mb_num: int, chunk_alg: str = "sort_consec") -> List[List[Sequence]]:
    """Split global batch into microbatches using DP."""
    seqs_gb_sorted = sorted(seqs_gb, key=lambda x: x.seq, reverse=True)
    n = len(seqs_gb_sorted)
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + seqs_gb_sorted[i - 1].seq
    dp = [[float("inf")] * (mb_num + 1) for _ in range(n + 1)]
    partition = [[0] * (mb_num + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for k in range(1, mb_num + 1):
        for i in range(1, n + 1):
            for j in range(k - 1, i):
                cost = prefix_sum[i] - prefix_sum[j]
                if max(dp[j][k - 1], cost) < dp[i][k]:
                    dp[i][k] = max(dp[j][k - 1], cost)
                    partition[i][k] = j
    mbs = []
    k, idx = mb_num, n
    while k > 0:
        start = partition[idx][k]
        mbs.append(seqs_gb_sorted[start:idx])
        idx = start
        k -= 1
    mbs.reverse()
    return mbs


# ──────────────────────────────────────────────────────────
# Strategy dataclass
# ──────────────────────────────────────────────────────────

@dataclass
class ParallelStrategy:
    """
    A (attn_type, parallel_size) pair representing a parallel group strategy.
    attn_type: "ulysses" (All-to-All / SP) or "ring" (Zigzag Ring Attention / CP)
    parallel_size: 1, 2, 4, 8, ...
    """
    attn_type: str   # "ulysses" or "ring"
    parallel_size: int
    
    def __repr__(self):
        return f"{self.attn_type}×{self.parallel_size}"

    def __hash__(self):
        return hash((self.attn_type, self.parallel_size))

    def __eq__(self, other):
        return self.attn_type == other.attn_type and self.parallel_size == other.parallel_size


# ──────────────────────────────────────────────────────────
# Cost Model
# ──────────────────────────────────────────────────────────

class AdaCPSPCostModel:
    """
    Estimates computation time, communication time, and memory for a given
    set of sequences under a given parallel strategy.

    Piecewise quadratic compute model:
      For each segment [lo, hi], time(s) = a*s² + b*s + c  (per sequence, per layer)
      Total compute = Σ_{seq} piecewise(seq/parallel_size) * num_layers  (approx.)

    Communication model:
      Ulysses: All-to-All.  comm = 4 * 2 * layers * hidden * total_tokens * 2 / sp_size / bandwidth
      Ring:    P2P ring.     comm = 2 * kv_bytes * (cp_size-1) / bandwidth (overlapped, but charged)

    Memory model:
      Same as FlexSP: model_states + activation(seqlens, parallel_size)
    """

    def __init__(
        self,
                 cluster_size: int = 8,
                 hidden_size: int = 4096,
                 layer_num: int = 32,
        param_size_B: float = 7.0,
                 zero_stage: int = 3,
                 mixed_precision: bool = True,
                 act_per_token: float = 4.71,
        # Piecewise compute coefficients:
        #   list of dicts: [{"range": [lo, hi], "a": ..., "b": ..., "c": ...}, ...]
        piecewise_compute_coeffs: Optional[List[Dict]] = None,
        # Fallback single-segment compute coefficients
        cpt_alpha1: float = 3.78e-8,
        cpt_alpha2: float = -1.06e-5,
        cpt_beta1: float = 0.25,  # per-layer bias
        # Communication bandwidths
        alltoall_bandwidth_dict_gbs: Optional[Dict[int, float]] = None,
        p2p_bandwidth_dict_gbs: Optional[Dict[int, float]] = None,
                 ):
        self.N = cluster_size
        self.h = hidden_size
        self.l = layer_num
        self.p = param_size_B
        self.zero_stage = zero_stage
        self.act_per_token = act_per_token
        
        # Model states memory
        zero_ratio = {
            0: 1,
            1: (6 / 8 * (1 / self.N) + 2 / 8) if mixed_precision else (2 / 4 * (1 / self.N) + 2 / 4),
            2: (7 / 8 * (1 / self.N) + 1 / 8) if mixed_precision else (3 / 4 * (1 / self.N) + 1 / 4),
            3: 1 / self.N,
        }[self.zero_stage]
        self.model_states_mb = param_size_B * 16 * zero_ratio * 1024

        # Piecewise compute
        if piecewise_compute_coeffs is not None:
            self.piecewise = sorted(piecewise_compute_coeffs, key=lambda x: x["range"][0])
        else:
            self.piecewise = [{"range": [0, 1e9], "a": cpt_alpha1, "b": cpt_alpha2, "c": cpt_beta1}]

        # Communication bandwidth dicts
        self.alltoall_bw = alltoall_bandwidth_dict_gbs or {1: 1e10, 2: 131.7, 4: 164.3, 8: 170.4}
        self.p2p_bw = p2p_bandwidth_dict_gbs or {1: 1e10, 2: 178.1, 4: 147.4, 8: 119.5}

    # ---- Compute ----

    def _get_coeffs(self, seqlen: int) -> Tuple[float, float, float]:
        """Get (a, b, c) for a given seqlen from piecewise segments."""
        for seg in self.piecewise:
            lo, hi = seg["range"]
            if lo <= seqlen <= hi:
                return seg["a"], seg["b"], seg["c"]
        # Fallback: use last segment
        seg = self.piecewise[-1]
        return seg["a"], seg["b"], seg["c"]

    def compute_time_single(self, seqlen: int, strategy: ParallelStrategy) -> float:
        """Compute time (ms) for a single sequence under a strategy (single layer)."""
        local_seq = seqlen / strategy.parallel_size
        a, b, c = self._get_coeffs(local_seq)
        return a * local_seq ** 2 + b * local_seq + c

    def compute_time(self, seqlens: List[int], strategy: ParallelStrategy) -> float:
        """Total compute time (ms) for a list of sequences (single layer * num_layers)."""
        total = sum(self.compute_time_single(s, strategy) for s in seqlens)
        return total * self.l

    # ---- Communication ----

    def alltoall_time(self, seqlens: List[int], sp_size: int) -> float:
        """All-to-All communication time (ms) for Ulysses SP."""
        if sp_size <= 1:
            return 0.0
        total_tokens = sum(seqlens)
        # Size of all-to-all tensor: 4 directions * 2 (fwd+bwd) * layers * hidden * total_tokens * bytes / sp_size
        tensor_size_mb = 4 * 2 * self.l * self.h * total_tokens * 2 / 1024 / 1024 / sp_size
        bw = self.alltoall_bw.get(sp_size, self.alltoall_bw.get(max(self.alltoall_bw.keys()), 100))
        return tensor_size_mb / bw

    def p2p_ring_time(self, seqlens: List[int], cp_size: int) -> float:
        """P2P ring communication time (ms) for Ring Attention."""
        if cp_size <= 1:
            return 0.0
        total_tokens = sum(seqlens)
        # Ring attention: (cp_size - 1) steps, each sending KV = 2 * (total_tokens/cp_size) * hidden * 2 bytes
        kv_per_step_mb = 2 * (total_tokens / cp_size) * self.h * 2 / 1024 / 1024
        bw = self.p2p_bw.get(cp_size, self.p2p_bw.get(max(self.p2p_bw.keys()), 100))
        # Total steps = (cp_size - 1), per-layer * num_layers
        total_time = kv_per_step_mb * (cp_size - 1) * self.l / bw
        return total_time

    def comm_time(self, seqlens: List[int], strategy: ParallelStrategy) -> float:
        """Communication time for a strategy."""
        if strategy.attn_type == "ulysses":
            return self.alltoall_time(seqlens, strategy.parallel_size)
        elif strategy.attn_type == "ring":
            return self.p2p_ring_time(seqlens, strategy.parallel_size)
        else:
            raise ValueError(f"Unknown attn_type: {strategy.attn_type}")

    # ---- Total time ----

    def total_time_single(self, seqlen: int, strategy: ParallelStrategy) -> float:
        """Total time for a single sequence (compute + comm, single forward pass)."""
        return self.compute_time_single(seqlen, strategy) * self.l + self.comm_time([seqlen], strategy)

    def total_time(self, seqlens: List[int], strategy: ParallelStrategy) -> float:
        """Total time for a set of sequences in one group."""
        return self.compute_time(seqlens, strategy) + self.comm_time(seqlens, strategy)

    # ---- Memory ----

    def activation_size(self, seqlens: Union[int, List[int]], parallel_size: int = 1) -> float:
        """Activation memory in MB."""
        if isinstance(seqlens, list):
            total = sum(seqlens)
        else:
            total = seqlens
        return self.act_per_token * total / parallel_size

    def total_memory(self, seqlens: Union[int, List[int]] = 0, parallel_size: int = 1) -> float:
        return self.model_states_mb + self.activation_size(seqlens, parallel_size)

    def token_capacity(self, memory_limit_gb: int) -> int:
        """Max tokens per device given memory budget."""
        return int((memory_limit_gb * 1024 - self.model_states_mb) / self.act_per_token)

    # ---- Check / debug ----

    def check(self, seqlens: List[int], strategy: ParallelStrategy):
        print(f"\n[seqlens={seqlens}, strategy={strategy}]")
        print(f"  Compute:  {self.compute_time(seqlens, strategy):.4f} ms")
        print(f"  Comm:     {self.comm_time(seqlens, strategy):.4f} ms")
        print(f"  Total:    {self.total_time(seqlens, strategy):.4f} ms")
        print(f"  Mem (MB): {self.total_memory(seqlens, strategy.parallel_size):.1f}")

    @classmethod
    def from_profile_files(
        cls,
        attention_json: str,
        alltoall_json: str,
        p2p_json: str,
        cluster_size: int = 8,
        param_size_B: float = 7.0,
        zero_stage: int = 3,
        act_per_token: float = 4.71,
    ) -> "AdaCPSPCostModel":
        """Construct a cost model from profiling output files."""
        # Attention coefficients
        with open(attention_json, "r") as f:
            attn_data = json.load(f)
        piecewise = []
        for seg_name, coeff in attn_data["coefficients"].items():
            if coeff is not None:
                piecewise.append({
                    "range": coeff["seq_range"],
                    "a": coeff["a"],
                    "b": coeff["b"],
                    "c": coeff["c"],
                })
        config = attn_data["config"]

        # All-to-All bandwidth
        with open(alltoall_json, "r") as f:
            a2a_data = json.load(f)
        alltoall_bw = {int(k): v for k, v in a2a_data["bandwidth_dict_GBs"].items()}

        # P2P bandwidth
        with open(p2p_json, "r") as f:
            p2p_data = json.load(f)
        p2p_bw = {int(k): v for k, v in p2p_data["bandwidth_dict_GBs"].items()}

        return cls(
            cluster_size=cluster_size,
            hidden_size=config["hidden_size"],
            layer_num=attn_data.get("num_layers", 32),
            param_size_B=param_size_B,
            zero_stage=zero_stage,
            act_per_token=act_per_token,
            piecewise_compute_coeffs=piecewise,
            alltoall_bandwidth_dict_gbs=alltoall_bw,
            p2p_bandwidth_dict_gbs=p2p_bw,
        )


# ──────────────────────────────────────────────────────────
# Bin-Packing Utilities
# ──────────────────────────────────────────────────────────

def BestFitDecreasing(seqs: List[Sequence], bin_capacity, bin_num=-1):
    K = len(seqs)
    P = bin_num if bin_num > 0 else K
    A = np.zeros((K, P), dtype=np.int32)
    seqs_sorted = sorted(seqs, reverse=True)
    bins = [bin_capacity] * P
    max_bin_id = -1
    for seq in seqs_sorted:
        best_idx = -1
        min_remain = bin_capacity + 1
        for p in range(P):
            if bins[p] >= seq.seq and (bins[p] - seq.seq) < min_remain:
                best_idx = p
                min_remain = bins[p] - seq.seq
        if best_idx != -1:
            bins[best_idx] -= seq.seq
            A[seq.id, best_idx] = 1
            max_bin_id = max(max_bin_id, best_idx)
        else:
            return None
    if max_bin_id + 1 < P:
        A = A[:, :max_bin_id + 1]
    return A


def FirstFitDecreasing(seqs: List[Sequence], bin_capacity, bin_num=-1):
    K = len(seqs)
    P = bin_num if bin_num > 0 else K
    A = np.zeros((K, P), dtype=np.int32)
    seqs_sorted = sorted(seqs, reverse=True)
    bins = [bin_capacity] * P
    max_bin_id = -1
    for seq in seqs_sorted:
        placed = False
        for p in range(P):
            if bins[p] >= seq.seq:
                bins[p] -= seq.seq
                A[seq.id, p] = 1
                max_bin_id = max(max_bin_id, p)
                placed = True
                break
        if not placed:
            return None
    if max_bin_id + 1 < P:
        A = A[:, :max_bin_id + 1]
    return A


# ──────────────────────────────────────────────────────────
# AdaCPSP Optimizer
# ──────────────────────────────────────────────────────────

class AdaCPSPOptimizer:
    """
    The AdaCPSP strategy optimizer.

    Given a set of sequences (a microbatch), finds the assignment of sequences
    to parallel groups and the strategy (attn_type, parallel_size) for each
    group that minimises the maximum group execution time, subject to memory
    constraints and device count constraints.

    Supports:
      - Heuristic solvers: BFD / FFD per strategy, pick best
      - ILP solver via pyscipopt (when available)
    """

    def __init__(
        self,
                 cluster_size: int,
                 memory_limit_gb: int,
                 costmodel: AdaCPSPCostModel,
        hide_output: bool = False,
        scip_param_dict: Optional[dict] = None,
        allowed_attn_types: Optional[List[str]] = None,
        max_parallel_size: int = 0,  # 0 means up to cluster_size
        min_parallel_size: int = 1,  # minimum parallel_size (= tp_deg for constrained mode)
    ):
        self.N = cluster_size
        self.mem_limit_gb = memory_limit_gb
        self.costmodel = costmodel
        self.device_token_capacity = costmodel.token_capacity(memory_limit_gb)
        self.cluster_token_capacity = self.device_token_capacity * self.N
        self.hide_output = hide_output
        self.scip_param_dict = scip_param_dict or {"limits/time": 10}
        self.allowed_attn_types = allowed_attn_types or ["ulysses", "ring"]
        self.max_parallel_size = max_parallel_size if max_parallel_size > 0 else cluster_size
        self.min_parallel_size = min_parallel_size

    def _log(self, msg):
        if not self.hide_output:
            print(msg)

    # ---- Strategy pool generation ----

    def get_strategy_pool(self, seqs: Optional[List[Sequence]] = None) -> List[ParallelStrategy]:
        """Generate all valid (attn_type, parallel_size) strategies.
        
        When min_parallel_size > 1 (constrained mode, e.g. tp_deg=2),
        only strategies with parallel_size >= min_parallel_size are included.
        This ensures compatibility with the fixed weight partitioning (TP degree).
        """
        strategies = []
        
        # Start from min_parallel_size (= tp_deg in constrained mode)
        ps = self.min_parallel_size
        if ps <= 1:
            # Include no-parallelism baseline only when unconstrained
            strategies.append(ParallelStrategy("ulysses", 1))
            ps = 2
        
        while ps <= self.max_parallel_size:
            for at in self.allowed_attn_types:
                strategies.append(ParallelStrategy(at, ps))
            ps *= 2
        
        if not strategies:
            # Fallback: at least include the minimum strategy
            strategies.append(ParallelStrategy("ulysses", self.min_parallel_size))
        
        return strategies

    def get_strategy_options(self, seqs: Optional[List[Sequence]] = None) -> List[ParallelStrategy]:
        """
        Generate the pool of strategy options for ILP.
        Each option represents one potential parallel group.
        Multiple groups can share the same strategy.
        """
        base_strategies = self.get_strategy_pool(seqs)
        # For each strategy, allow up to N/parallel_size groups
        options = []
        for strat in base_strategies:
            max_groups = self.N // strat.parallel_size
            for _ in range(max_groups):
                options.append(strat)
        return options

    def _min_parallel_size(self, seqlen: int) -> int:
        """Minimum parallel_size needed for a sequence to fit in memory."""
        min_ps = max(1, int(np.ceil(seqlen / self.device_token_capacity)))
        # Round up to next power of 2
        log2 = np.log(min_ps) / np.log(2)
        return int(2 ** int(np.ceil(log2)))

    # ---- Heuristic solver: BFD per strategy ----

    def solve_homo_strategy_bfd(
        self,
        seqs: List[Sequence],
        strategy: ParallelStrategy,
        group_num: int,
    ) -> Optional[Dict]:
        """
        Solve with a fixed strategy using Best-Fit Decreasing.
        All groups use the same strategy.
        """
        bin_capacity = self.device_token_capacity * strategy.parallel_size
        A = BestFitDecreasing(seqs, bin_capacity, group_num)
        if A is None:
            return None

        K, P = len(seqs), A.shape[1]
        M = -1
        for p in range(P):
            group_tokens = sum(seqs[k].seq * A[k, p] for k in range(K)) / strategy.parallel_size
            if group_tokens > self.device_token_capacity:
                return None
            group_time = sum(
                self.costmodel.total_time_single(seqs[k].seq, strategy) * A[k, p]
                for k in range(K)
            )
            M = max(group_time, M)

        return {
            "seqs": seqs,
            "strategies": [strategy] * P,
            "A": A,
            "M": M,
        }

    def solve_adaptive_bfd(self, seqs: List[Sequence]) -> Optional[Dict]:
        """
        Try all homogeneous strategies with BFD, pick the one with minimum time.
        This is a fast heuristic — each group uses the same strategy.
        """
        best_result = None
        strategies = self.get_strategy_pool(seqs)

        for strat in strategies:
            if strat.parallel_size > self.N:
                continue
            group_num = self.N // strat.parallel_size
            result = self.solve_homo_strategy_bfd(seqs, strat, group_num)
            if result is not None:
                if best_result is None or result["M"] < best_result["M"]:
                    best_result = result

        return best_result

    # ---- ILP solver ----

    def solve_adacpsp_ilp(
        self,
        seqs: List[Sequence],
        bucket_num: int = 16,
    ) -> Optional[Dict]:
        """
        Solve the AdaCPSP optimisation problem using ILP (pyscipopt).

        Decision variables:
          A[k, p] ∈ {0,1} — assign sequence k to group p
          m[p] ∈ {0,1}    — is group p active?
          M (continuous)   — minimax objective

        Groups are indexed by their strategy (from the strategy pool).
        """
        try:
            from pyscipopt import Model as SCIPModel, quicksum
        except ImportError:
            self._log("[AdaCPSP] pyscipopt not available, falling back to BFD heuristic")
            return self.solve_adaptive_bfd(seqs)

        strategy_options = self.get_strategy_options(seqs)
        seq_min_ps = [self._min_parallel_size(s.seq) for s in seqs]

        K = len(seqs)
        P = len(strategy_options)

        model = SCIPModel("AdaCPSP ILP")
        if self.hide_output:
            model.hideOutput()
        model.setParams(self.scip_param_dict)

        # Variables
        M = model.addVar(vtype="C", name="M", lb=0)
        model.setObjective(M, "minimize")

        A = {(k, p): model.addVar(vtype="B", name=f"A_{k}_{p}") for k in range(K) for p in range(P)}
        m = {p: model.addVar(vtype="B", name=f"m_{p}") for p in range(P)}

        # Constraints
        # 1. Each sequence assigned to exactly one group
        for k in range(K):
            model.addCons(quicksum(A[k, p] for p in range(P)) == 1)

            # Prune: can't assign to groups with parallel_size < min needed
            for p in range(P):
                if strategy_options[p].parallel_size < seq_min_ps[k]:
                    model.addCons(A[k, p] == 0)

        # 2. Group constraints
        for p in range(P):
            strat = strategy_options[p]
            ps = strat.parallel_size

            # Memory: total tokens in group / parallel_size <= capacity
            model.addCons(
                quicksum(seqs[k].seq * A[k, p] for k in range(K)) / ps
                <= self.device_token_capacity
            )

            # Time: group time <= M
            model.addCons(
                quicksum(
                    self.costmodel.total_time_single(seqs[k].seq, strat) * A[k, p]
                    for k in range(K)
                )
                <= M
            )

            # m[p] links
            model.addCons(quicksum(A[k, p] for k in range(K)) <= K * m[p])
            model.addCons(quicksum(A[k, p] for k in range(K)) >= m[p])

        # 3. Device constraint: sum of active groups * parallel_size == N
        model.addCons(quicksum(m[p] * strategy_options[p].parallel_size for p in range(P)) == self.N)

        # Solve
        model.optimize()
        status = model.getStatus()
        self._log(f"[AdaCPSP ILP] Status: {status}")

        if status not in ["optimal", "timelimit"]:
            return None

        # Extract solution
        result_A = np.zeros((K, P), dtype=np.int32)
        result_m = np.zeros(P, dtype=np.int32)
        for p in range(P):
            result_m[p] = round(model.getVal(m[p]))
            for k in range(K):
                result_A[k, p] = round(model.getVal(A[k, p]))

        return {
            "seqs": seqs,
            "strategies": strategy_options,
            "A": result_A,
            "m": result_m,
            "M": model.getVal(M),
        }

    # ---- Global batch solver ----

    def solve_globalbatch(
        self,
        seqs_gb: List[Sequence],
        chunk_alg: str = "sort_consec",
        method: str = "adaptive_bfd",
        _max_mb_retries: int = 20,
    ) -> Tuple[List, List]:
        """
        Solve for a global batch:
          1. Determine minimum microbatch count
          2. Split into microbatches
          3. Solve each microbatch
        """
        # Find min microbatch number
        total_tokens = sum(get_lens(seqs_gb))
        mb_num = max(1, int(np.ceil(total_tokens / self.cluster_token_capacity)))

        for _ in range(_max_mb_retries):
            seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)
            valid = True
            for seqs_mb in seqs_mb_all:
                if sum(get_lens(seqs_mb)) >= self.cluster_token_capacity:
                    valid = False
                    break
            if valid:
                break
            mb_num += 1
        else:
            self._log(f"[AdaCPSP] Cannot find valid microbatch split after {_max_mb_retries} retries")
            return [], []

        self._log(f"[AdaCPSP] Using {mb_num} microbatch(es)")

        all_groups = []
        all_results = []
        retry_count = 0

        for i, seqs_mb in enumerate(seqs_mb_all):
            # Save original IDs before re-indexing (needed for batch data retrieval)
            orig_ids = [s.id for s in seqs_mb]
            # Re-index sequences for this microbatch so BFD/ILP indices are valid
            seqs_mb_reindexed = [Sequence(seq=s.seq, id=j) for j, s in enumerate(seqs_mb)]

            self._log(f"\n--- Microbatch {i} ({len(seqs_mb_reindexed)} seqs, "
                      f"{sum(get_lens(seqs_mb_reindexed))} tokens) ---")

            if method == "ilp":
                result = self.solve_adacpsp_ilp(seqs_mb_reindexed)
            elif method == "adaptive_bfd":
                result = self.solve_adaptive_bfd(seqs_mb_reindexed)
            else:
                raise ValueError(f"Unknown method: {method}")

            if result is None:
                retry_count += 1
                if retry_count > _max_mb_retries:
                    self._log(f"  [ERROR] Too many retries, giving up")
                    return [], []
                self._log(f"  [WARN] Microbatch {i} infeasible, splitting into more microbatches")
                mb_num += 1
                return self.solve_globalbatch(seqs_gb, chunk_alg, method,
                                              _max_mb_retries=_max_mb_retries - retry_count)

            groups = self._extract_groups(result)
            # Restore original IDs so collate_fn can retrieve correct batch data
            for strat, group_seqs in groups:
                for seq in group_seqs:
                    seq.id = orig_ids[seq.id]
            all_groups.append(groups)
            all_results.append(result)

            if not self.hide_output:
                for strat, group_seqs in groups:
                    seqlens = get_lens(group_seqs)
                    t = self.costmodel.total_time(seqlens, strat)
                    print(f"  Group ({strat}): {len(group_seqs)} seqs, "
                          f"tokens={sum(seqlens)}, time={t:.2f} ms")

        return all_groups, all_results

    def _extract_groups(self, result: Dict) -> List[Tuple[ParallelStrategy, List[Sequence]]]:
        """Extract (strategy, [sequences]) list from solver result."""
        seqs = result["seqs"]
        strategies = result["strategies"]
        A = result["A"]
        K, P = A.shape

        groups = []
        for p in range(P):
            group_seqs = [seqs[k] for k in range(K) if A[k, p] > 0]
            if group_seqs:
                groups.append((strategies[p], group_seqs))
        return groups

    # ---- Reporting ----

    def print_solution(self, all_groups, all_results):
        """Pretty-print the solution."""
        total_time = 0
        for i, (groups, result) in enumerate(zip(all_groups, all_results)):
            mb_time = result["M"]
            total_time += mb_time
            print(f"\n=== Microbatch {i}: time={mb_time:.2f} ms ===")
            for strat, group_seqs in groups:
                seqlens = get_lens(group_seqs)
                t = self.costmodel.total_time(seqlens, strat)
                mem = self.costmodel.total_memory(seqlens, strat.parallel_size)
                print(f"  [{strat}] {len(group_seqs)} seqs, "
                      f"tokens={sum(seqlens)}, local_tokens={sum(seqlens)//strat.parallel_size}, "
                      f"time={t:.2f} ms, mem={mem:.1f} MB, seqlens={seqlens}")
        print(f"\nTotal time: {total_time:.2f} ms ({len(all_groups)} microbatches)")
        return total_time


# ──────────────────────────────────────────────────────────
# Config dataclass for integration with training script
# ──────────────────────────────────────────────────────────

@dataclass
class AdaCPSPConfig:
    """
    Configuration for AdaCPSP, produced by the solver and consumed by the runtime.
    Describes the assignment of sequences to groups and the strategy for each group.
    """
    # Per microbatch: list of (attn_type, parallel_size, sequence_ids)
    microbatch_plans: List[List[Tuple[str, int, List[int]]]] = field(default_factory=list)
    cluster_size: int = 8
    total_microbatches: int = 1

    def to_json(self, path: str):
        data = {
            "cluster_size": self.cluster_size,
            "total_microbatches": self.total_microbatches,
            "microbatch_plans": self.microbatch_plans,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "AdaCPSPConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_solver_output(
        cls,
        all_groups: List[List[Tuple[ParallelStrategy, List[Sequence]]]],
        cluster_size: int,
    ) -> "AdaCPSPConfig":
        """Convert solver output to config."""
        plans = []
        for groups in all_groups:
            mb_plan = []
            for strat, seqs in groups:
                seq_ids = [s.id for s in seqs]
                mb_plan.append((strat.attn_type, strat.parallel_size, seq_ids))
            plans.append(mb_plan)
        return cls(
            microbatch_plans=plans,
            cluster_size=cluster_size,
            total_microbatches=len(plans),
        )


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def read_dataset(name="github", seq_limit=65536, world_size=64):
    """Read dataset of sequence lengths (same format as flexsp)."""
    path = os.path.join(os.path.dirname(__file__), "..", "datasets", name + ".txt")
    if not os.path.exists(path):
        # Try alternate paths
        for alt in [f"../datasets/{name}.txt", f"datasets/{name}.txt"]:
            if os.path.exists(alt):
                path = alt
                break
    data = []
    try:
        with open(path, "r") as f:
            for line in f:
                d = int(line.strip())
                pad_len = ((d - 1) // (2 * world_size) + 1) * (2 * world_size)
                if pad_len <= seq_limit - 2 * world_size:
                    data.append(pad_len)
    except FileNotFoundError:
        print(f"[WARN] Dataset file not found: {path}, generating synthetic data")
        random.seed(42)
        for _ in range(10000):
            s = random.randint(128, seq_limit)
            pad = ((s - 1) // (2 * world_size) + 1) * (2 * world_size)
            data.append(pad)
    return data


def main():
    parser = argparse.ArgumentParser(description="AdaCPSP Solver")
    parser.add_argument("--cluster_size", type=int, default=8)
    parser.add_argument("--memory_limit_gb", type=int, default=28)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--layer_num", type=int, default=32)
    parser.add_argument("--param_size_B", type=float, default=7.0)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="github")
    parser.add_argument("--seq_limit_k", type=int, default=128)
    parser.add_argument("--iter_num", type=int, default=5)
    parser.add_argument("--start_iter", type=int, default=3)
    parser.add_argument("--method", type=str, default="adaptive_bfd",
                        choices=["adaptive_bfd", "ilp"])
    parser.add_argument("--attn_types", type=str, nargs="+",
                        default=["ulysses", "ring"],
                        choices=["ulysses", "ring"])
    parser.add_argument("--save_dir", type=str, default="./configs")
    # Profile file paths (optional)
    parser.add_argument("--attention_profile", type=str, default=None)
    parser.add_argument("--alltoall_profile", type=str, default=None)
    parser.add_argument("--p2p_profile", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print(" AdaCPSP Solver")
    print("=" * 70)
    print(f" Cluster: {args.cluster_size} GPUs, Memory: {args.memory_limit_gb} GB")
    print(f" Model: hidden={args.hidden_size}, layers={args.layer_num}, params={args.param_size_B}B")
    print(f" Method: {args.method}, Attn types: {args.attn_types}")
    print("=" * 70)

    # Build cost model
    if args.attention_profile and args.alltoall_profile and args.p2p_profile:
        costmodel = AdaCPSPCostModel.from_profile_files(
            args.attention_profile,
            args.alltoall_profile,
            args.p2p_profile,
            cluster_size=args.cluster_size,
            param_size_B=args.param_size_B,
        )
    else:
        costmodel = AdaCPSPCostModel(
            cluster_size=args.cluster_size,
            hidden_size=args.hidden_size,
            layer_num=args.layer_num,
            param_size_B=args.param_size_B,
        )

    optimizer = AdaCPSPOptimizer(
        cluster_size=args.cluster_size,
        memory_limit_gb=args.memory_limit_gb,
        costmodel=costmodel,
        hide_output=False,
        scip_param_dict={"limits/time": 10},
        allowed_attn_types=args.attn_types,
    )

    print(f"\nDevice token capacity: {optimizer.device_token_capacity}")
    print(f"Cluster token capacity: {optimizer.cluster_token_capacity}")

    # Load dataset
    seq_limit = args.seq_limit_k * 1000
    data = read_dataset(args.dataset, seq_limit=seq_limit, world_size=args.cluster_size)
    print(f"Dataset: {len(data)} sequences loaded")

    # Run solver across iterations
    total_time = 0
    strategy_counter = Counter()

    for it in range(args.start_iter, args.start_iter + args.iter_num):
        gb_start = it * args.global_batch_size
        gb_end = (it + 1) * args.global_batch_size
        if gb_end > len(data):
            break

        sequences = [Sequence(seq=data[i], id=i - gb_start) for i in range(gb_start, gb_end)]

        print(f"\n{'=' * 70}")
        print(f" Iteration {it}: {len(sequences)} seqs, "
              f"{sum(s.seq for s in sequences)} total tokens")
        print(f"{'=' * 70}")

        start = time_module.time()
        all_groups, all_results = optimizer.solve_globalbatch(
            sequences, method=args.method
        )
        elapsed = time_module.time() - start

        iter_time = sum(r["M"] for r in all_results)
        total_time += iter_time

        # Count strategies
        for groups in all_groups:
            for strat, _ in groups:
                strategy_counter[str(strat)] += 1

        print(f"\n  Iter[{it}]: solver_time={elapsed:.3f}s, "
              f"estimated_time={iter_time:.2f} ms, "
              f"microbatches={len(all_groups)}")

        optimizer.print_solution(all_groups, all_results)

    print(f"\n{'=' * 70}")
    print(f" Summary: {args.iter_num} iterations, total estimated time = {total_time:.2f} ms")
    print(f" Strategy usage: {dict(strategy_counter)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

