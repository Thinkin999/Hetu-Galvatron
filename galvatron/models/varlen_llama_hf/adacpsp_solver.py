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

Solver methods (from FlexSP, extended for AdaCPSP):
  * adaptive_bfd  — Try all homogeneous strategies with BFD, pick best
  * adaptive_ffd  — Try all homogeneous strategies with FFD, pick best
  * ilp           — ILP on individual sequences (small batches)
  * bucket_ilp    — ILP with sequence bucketing (large batches, much faster)

Global batch solving:
  * solve_globalbatch       — Sequential microbatch solving
  * solve_globalbatch_mp    — Parallel microbatch solving (multiprocessing)
  * solve_globalbatch_mp_gbmb — Parallel mb_num exploration (multiprocessing)

Usage (standalone):
    python adacpsp_solver.py --cluster_size 8 --memory_limit_gb 28 \
        --global_batch_size 64 --dataset github
"""

from __future__ import annotations

import os
import json
import random
import argparse
import heapq
import math
import time as time_module
import multiprocessing as mp
from typing import Any, List, Dict, Tuple, Optional, Union, Literal
from dataclasses import dataclass, field
from collections import Counter
from copy import deepcopy

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
    A (attn_type, parallel_size, placement) tuple representing a parallel group strategy.

    attn_type:
      - "ulysses" : All-to-All (Ulysses SP), sp_size = parallel_size, cp_size = 1
      - "ring"    : P2P ring  (Ring Attention CP), sp_size = 1, cp_size = parallel_size
      - "usp"     : Combined Ulysses + Ring, parallel_size = sp_size × cp_size

    parallel_size: total GPUs occupied by one such group (1, 2, 4, 8, …)
    sp_size / cp_size: explicit decomposition (auto-derived from attn_type if 0)
    placement: "head_first" or "context_first"
      - head_first:    SP groups use consecutive ranks (AlltoAll intra-node), CP groups strided
      - context_first: CP groups use consecutive ranks (Ring intra-node), SP groups strided
      Only meaningful for USP; ulysses/ring always use "context_first".
    """
    attn_type: str   # "ulysses", "ring", or "usp"
    parallel_size: int
    sp_size: int = 0
    cp_size: int = 0
    placement: str = "context_first"  # "head_first" | "context_first"

    def __post_init__(self):
        if self.attn_type == "ulysses":
            self.sp_size = self.parallel_size
            self.cp_size = 1
            self.placement = "context_first"
        elif self.attn_type == "ring":
            self.sp_size = 1
            self.cp_size = self.parallel_size
            self.placement = "context_first"
        elif self.attn_type == "usp":
            assert self.sp_size > 1 and self.cp_size > 1, \
                f"USP requires sp_size>1 and cp_size>1, got sp={self.sp_size}, cp={self.cp_size}"
            assert self.sp_size * self.cp_size == self.parallel_size, \
                f"sp_size*cp_size ({self.sp_size}*{self.cp_size}) != parallel_size ({self.parallel_size})"
            assert self.placement in ("head_first", "context_first"), \
                f"Unknown placement: {self.placement!r}"
        else:
            raise ValueError(f"Unknown attn_type: {self.attn_type}")
    
    def __repr__(self):
        if self.attn_type == "usp":
            pl = "hf" if self.placement == "head_first" else "cf"
            return f"usp(sp{self.sp_size}×cp{self.cp_size},{pl})"
        return f"{self.attn_type}×{self.parallel_size}"

    def __hash__(self):
        return hash((self.attn_type, self.parallel_size, self.sp_size, self.cp_size, self.placement))

    def __eq__(self, other):
        return (self.attn_type == other.attn_type
                and self.parallel_size == other.parallel_size
                and self.sp_size == other.sp_size
                and self.cp_size == other.cp_size
                and self.placement == other.placement)


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
                 act_per_token: float = 3.96,
        # ── GQA (Grouped-Query Attention) ──
        num_attention_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        head_dim: int = 128,
        # Piecewise compute coefficients
        piecewise_compute_coeffs: Optional[List[Dict]] = None,
        cpt_alpha1: float = 3.78e-8,
        cpt_alpha2: float = -1.06e-5,
        cpt_beta1: float = 0.25,
        # Communication bandwidths (legacy: simple BW model)
        alltoall_bandwidth_dict_gbs: Optional[Dict[int, float]] = None,
        p2p_bandwidth_dict_gbs: Optional[Dict[int, float]] = None,
        # Communication linear fit
        alltoall_linear_fit: Optional[Dict[int, Dict[str, float]]] = None,
        p2p_linear_fit: Optional[Dict[int, Dict[str, float]]] = None,
        p2p_ring_step_fit: Optional[Dict[int, Dict[str, float]]] = None,
        p2p_ring_interp: Optional[Dict[int, List[Tuple[float, float]]]] = None,
        a2a_interp: Optional[Dict[int, List[Tuple[float, float]]]] = None,
        # ── Overlap-aware modeling parameters ──
        bwd_fwd_ratio: float = 2.0,
        ring_bwd_comm_ratio: float = 2.0,
        enable_overlap_model: bool = True,
        ring_causal_correction: bool = False,
        overlap_leakage: float = 0.1,
        # ── Placement-aware topology bandwidth data ──
        gpus_per_node: int = 8,
        # Topology-split bandwidth dicts: {group_size: bandwidth_GBs}
        alltoall_bw_consec: Optional[Dict[int, float]] = None,
        alltoall_bw_strided: Optional[Dict[int, float]] = None,
        p2p_bw_consec: Optional[Dict[int, float]] = None,
        p2p_bw_strided: Optional[Dict[int, float]] = None,
        # Topology-split linear fits: {group_size: {"alpha": .., "beta": ..}}
        alltoall_linear_consec: Optional[Dict[int, Dict[str, float]]] = None,
        alltoall_linear_strided: Optional[Dict[int, Dict[str, float]]] = None,
        p2p_linear_consec: Optional[Dict[int, Dict[str, float]]] = None,
        p2p_linear_strided: Optional[Dict[int, Dict[str, float]]] = None,
                 ):
        self.N = cluster_size
        self.h = hidden_size
        self.l = layer_num
        self.p = param_size_B
        self.zero_stage = zero_stage
        self.act_per_token = act_per_token
        self.bwd_fwd_ratio = bwd_fwd_ratio
        self.ring_bwd_comm_ratio = ring_bwd_comm_ratio
        self.enable_overlap_model = enable_overlap_model
        self.ring_causal_correction = ring_causal_correction
        self.overlap_leakage = overlap_leakage

        # GQA: KV hidden dim for communication sizing
        self.head_dim = head_dim
        self.n_heads = num_attention_heads or (hidden_size // head_dim)
        self.n_kv_heads = num_kv_heads if num_kv_heads is not None else self.n_heads
        self.kv_hidden = self.n_kv_heads * self.head_dim  # KV tensor hidden dim
        
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

        # Communication: prefer linear fit if available, fallback to bandwidth
        self.alltoall_bw = alltoall_bandwidth_dict_gbs or {1: 1e10, 2: 131.7, 4: 164.3, 8: 170.4}
        self.p2p_bw = p2p_bandwidth_dict_gbs or {1: 1e10, 2: 178.1, 4: 147.4, 8: 119.5}
        self.alltoall_linear = alltoall_linear_fit
        self.p2p_linear = p2p_linear_fit
        self.p2p_ring_step = p2p_ring_step_fit
        self.p2p_ring_interp = p2p_ring_interp
        self.a2a_interp = a2a_interp

        # Placement-aware topology data
        self.gpus_per_node = gpus_per_node
        self.alltoall_bw_consec = alltoall_bw_consec
        self.alltoall_bw_strided = alltoall_bw_strided
        self.p2p_bw_consec = p2p_bw_consec
        self.p2p_bw_strided = p2p_bw_strided
        self.alltoall_linear_consec = alltoall_linear_consec
        self.alltoall_linear_strided = alltoall_linear_strided
        self.p2p_linear_consec = p2p_linear_consec
        self.p2p_linear_strided = p2p_linear_strided

        self.compute_correction: Optional[List[Tuple[int, float]]] = None

    # ---- Placement-aware topology routing ----

    @staticmethod
    def _get_topo(placement: str, comm_type: str) -> str:
        """Derive topology type from placement and communication primitive.

        head_first:    AlltoAll=consecutive, Ring=strided
        context_first: AlltoAll=strided,     Ring=consecutive
        """
        if placement == "head_first":
            return "consecutive" if comm_type == "alltoall" else "strided"
        else:  # context_first (default)
            return "strided" if comm_type == "alltoall" else "consecutive"

    def _select_a2a_bw(self, topo: str) -> Optional[Dict[int, float]]:
        """Pick alltoall BW dict for the given topology, with fallback."""
        if topo == "consecutive" and self.alltoall_bw_consec:
            return self.alltoall_bw_consec
        if topo == "strided" and self.alltoall_bw_strided:
            return self.alltoall_bw_strided
        return None  # caller uses self.alltoall_bw

    def _select_a2a_linear(self, topo: str) -> Optional[Dict[int, Dict[str, float]]]:
        if topo == "consecutive" and self.alltoall_linear_consec:
            return self.alltoall_linear_consec
        if topo == "strided" and self.alltoall_linear_strided:
            return self.alltoall_linear_strided
        return None  # caller uses self.alltoall_linear

    def _select_p2p_bw(self, topo: str) -> Optional[Dict[int, float]]:
        if topo == "consecutive" and self.p2p_bw_consec:
            return self.p2p_bw_consec
        if topo == "strided" and self.p2p_bw_strided:
            return self.p2p_bw_strided
        return None

    def _select_p2p_linear(self, topo: str) -> Optional[Dict[int, Dict[str, float]]]:
        if topo == "consecutive" and self.p2p_linear_consec:
            return self.p2p_linear_consec
        if topo == "strided" and self.p2p_linear_strided:
            return self.p2p_linear_strided
        return None

    # ---- GQA Head Padding Overhead ----

    def head_padding_overhead(self, sp_size: int) -> Tuple[float, float]:
        """Compute the overhead factor due to head padding for a given sp_size.
        
        When n_heads or n_kv_heads is not divisible by sp_size, Ulysses SP
        must replicate entire GQA groups to make the All-to-All work.
        
        Returns:
            (q_factor, kv_factor): multiplicative overhead factors (>= 1.0).
            q_factor applies to Q/O communication and compute.
            kv_factor applies to K/V communication.
            If no padding is needed, both are 1.0.
        """
        if sp_size <= 1:
            return 1.0, 1.0
        if self.n_kv_heads % sp_size == 0 and self.n_heads % sp_size == 0:
            return 1.0, 1.0
        
        g = self.n_heads // self.n_kv_heads  # GQA ratio
        padded_n_kv = math.ceil(self.n_kv_heads / sp_size) * sp_size
        padded_n_q = padded_n_kv * g
        
        q_factor = padded_n_q / self.n_heads
        kv_factor = padded_n_kv / self.n_kv_heads
        return q_factor, kv_factor

    def head_padding_extra_activation_mb(self, seqlens: List[int], sp_size: int,
                                         parallel_size: Optional[int] = None) -> float:
        """Extra activation memory (MB) due to head padding.
        
        Head padding only affects the attention layer (between QKV projection and
        output projection). The extra memory is for the padded Q, K, V tensors
        that exist during the All-to-All and attention computation.
        
        Extra per-token memory (fp16):
          Q: (padded_n_q - n_q) * head_dim * 2 bytes
          K: (padded_n_kv - n_kv) * head_dim * 2 bytes
          V: (padded_n_kv - n_kv) * head_dim * 2 bytes
        
        Args:
            seqlens: sequence lengths in the group.
            sp_size: Ulysses SP size (determines padding factors).
            parallel_size: total parallel size (sp * cp for USP).
                For pure Ulysses: parallel_size = sp_size.
                For USP: parallel_size = sp_size * cp_size.
                If None, defaults to sp_size (backward compatible).
        """
        if sp_size <= 1:
            return 0.0
        q_factor, kv_factor = self.head_padding_overhead(sp_size)
        if q_factor == 1.0 and kv_factor == 1.0:
            return 0.0
        
        if parallel_size is None:
            parallel_size = sp_size
        total_tokens = sum(seqlens) / parallel_size  # tokens per device
        q_extra = (q_factor - 1.0) * self.n_heads * self.head_dim * 2  # bytes per token
        kv_extra = (kv_factor - 1.0) * self.n_kv_heads * self.head_dim * 2 * 2  # K+V
        extra_bytes_per_token = q_extra + kv_extra
        return total_tokens * extra_bytes_per_token / 1024 / 1024

    # ---- Interpolation helpers ----

    @staticmethod
    def _interp_lookup(msg_mb: float,
                       pts: List[Tuple[float, float]]) -> float:
        """Linear interpolation/extrapolation on a sorted (x, y) table.
        
        - Within range: linear interpolation between adjacent points.
        - Below range: extrapolate from first two points.
        - Above range: extrapolate from last two points.
        """
        if len(pts) < 2:
            return pts[0][1] if pts else 0.0

        if msg_mb <= pts[0][0]:
            slope = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
            return max(0.0, pts[0][1] + slope * (msg_mb - pts[0][0]))

        if msg_mb >= pts[-1][0]:
            slope = (pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0])
            return pts[-1][1] + slope * (msg_mb - pts[-1][0])

        for i in range(len(pts) - 1):
            if pts[i][0] <= msg_mb <= pts[i + 1][0]:
                t = (msg_mb - pts[i][0]) / (pts[i + 1][0] - pts[i][0])
                return pts[i][1] + t * (pts[i + 1][1] - pts[i][1])

        return pts[-1][1]

    def _interp_ring_per_step(self, kv_per_step_mb: float, cp_size: int) -> Optional[float]:
        """Interpolate ring per-step time from profiled data."""
        if not self.p2p_ring_interp or cp_size not in self.p2p_ring_interp:
            return None
        return self._interp_lookup(kv_per_step_mb, self.p2p_ring_interp[cp_size])

    def _interp_a2a(self, msg_mb: float, sp_size: int) -> Optional[float]:
        """Interpolate A2A per-op time from profiled data."""
        if not self.a2a_interp or sp_size not in self.a2a_interp:
            return None
        return self._interp_lookup(msg_mb, self.a2a_interp[sp_size])

    # ---- Compute ----

    def _get_coeffs(self, seqlen: float) -> Tuple[float, float, float]:
        """Get (a, b, c) for a given seqlen from piecewise segments.
        
        For seqlen below all ranges: use first segment (extrapolate down).
        For seqlen above all ranges: use last segment (extrapolate up).
        """
        if not self.piecewise:
            return self.alpha1, self.alpha2, self.beta1

        # Check each segment
        for seg in self.piecewise:
            lo, hi = seg["range"]
            if lo <= seqlen <= hi:
                return seg["a"], seg["b"], seg["c"]

        # Below all segments: use first segment (safe extrapolation down)
        first_lo = self.piecewise[0]["range"][0]
        if seqlen < first_lo:
            seg = self.piecewise[0]
            return seg["a"], seg["b"], seg["c"]

        # Above all segments: use last segment (extrapolate up)
        seg = self.piecewise[-1]
        return seg["a"], seg["b"], seg["c"]

    def _eval_piecewise(self, x: float) -> float:
        """Evaluate piecewise quadratic at x, with optional calibration correction.
        
        The correction factor is clamped to [0.95, 1.50] to prevent runaway
        extrapolation for seq_lens far outside the validation range.
        Result is always clamped to >= 0 (compute time cannot be negative).
        """
        a, b, c = self._get_coeffs(x)
        raw = a * x ** 2 + b * x + c
        raw = max(raw, 0.0)  # compute time cannot be negative
        if self.compute_correction:
            corr = self._interp_lookup(x, self.compute_correction)
            corr = max(0.95, min(1.50, corr))
            raw *= corr
        return raw

    def compute_time_single(self, seqlen: int, strategy: ParallelStrategy) -> float:
        """Compute time (ms) for a single sequence under a strategy (single layer).
        
        Strategy-aware:
          - Ulysses: Each rank processes full seq_len but with h/sp heads.
            Flash attention time scales linearly with #heads, so:
            time = f(seqlen) / sp_size
            With head padding: time = f(seqlen) * q_factor / sp_size
            (q_factor reflects the padded/original head ratio, >= 1.0)
          
          - Ring: Per-step compute on local chunk (seqlen/cp tokens, all heads).
            Returns time for ONE ring step (total per layer = cp × this).
            time = f(seqlen / cp_size)
            Ring does not require head divisibility → no padding overhead.
          
          - USP: Per-step compute on seqlen/cp tokens with h/sp heads.
            time = f(seqlen / cp_size) * q_factor / sp_size
            (q_factor applies to the Ulysses SP component)
        
        Where f(x) = a*x² + b*x + c is the profiled piecewise quadratic,
        optionally corrected by calibration factors from validation data.
        """
        if strategy.attn_type == "ulysses":
            q_factor, _ = self.head_padding_overhead(strategy.sp_size)
            return self._eval_piecewise(seqlen) * q_factor / strategy.sp_size
        elif strategy.attn_type == "ring":
            local_seq = seqlen / strategy.cp_size
            return self._eval_piecewise(local_seq)
        elif strategy.attn_type == "usp":
            local_seq = seqlen / strategy.cp_size
            q_factor, _ = self.head_padding_overhead(strategy.sp_size)
            return self._eval_piecewise(local_seq) * q_factor / strategy.sp_size
        else:
            local_seq = seqlen / strategy.parallel_size
            return self._eval_piecewise(local_seq)

    def compute_time(self, seqlens: List[int], strategy: ParallelStrategy) -> float:
        """Forward compute time (ms) for a list of sequences (per-step × num_layers).
        
        Returns the time for ONE flash_attn call (per-step for Ring/USP,
        or the full per-layer call for Ulysses), summed across all sequences
        in the group, then multiplied by num_layers.
        
        For Ring/USP: total forward compute = cp_size × compute_time
        For Ulysses:  total forward compute = compute_time
        """
        total = sum(self.compute_time_single(s, strategy) for s in seqlens)
        return total * self.l

    def _ring_step_compute_per_layer(self, seqlens: List[int],
                                      strategy: ParallelStrategy) -> float:
        """Compute time per layer for one ring step (all sequences in group).
        
        In zigzag ring attention, EVERY step has the same FLOPs ≈ S_local²/2:
          - Diagonal step: flash_attn(Q[S_local], K[S_local], causal=True)
            → lower-triangle only = S_local²/2 pairs
          - Non-diagonal (step ≤ rank): flash_attn(Q[S_local], K[S_local/2], causal=False)
            → rectangular S_local × S_local/2 = S_local²/2 pairs
          - Non-diagonal (step > rank): flash_attn(Q[S_local/2], K[S_local], causal=False)
            → rectangular S_local/2 × S_local = S_local²/2 pairs
        
        Therefore all steps use the same compute estimate: f_causal(S/cp_size).
        The total Ring FLOPs = P × f_causal(S/P) = f(S)/P = Ulysses FLOPs. ✓
        """
        return sum(self.compute_time_single(s, strategy) for s in seqlens)

    # ---- Communication ----

    def _a2a_per_op_time(self, msg_mb: float, sp_size: int,
                          topo: str = "consecutive") -> float:
        """Get per-op A2A time (ms) with cascading fallback.
        
        Priority: topology-aware linear fit → generic interpolation →
                  generic linear fit → topology-aware BW → generic BW.
        """
        # Priority 1: Topology-aware linear fit
        topo_lin = self._select_a2a_linear(topo)
        if topo_lin and sp_size in topo_lin:
            fit = topo_lin[sp_size]
            return max(0.0, fit["alpha"] * msg_mb + fit["beta"])

        # Priority 2: Generic interpolation from actual A2A profiling
        interp_val = self._interp_a2a(msg_mb, sp_size)
        if interp_val is not None:
            return interp_val

        # Priority 3: Generic linear fit
        if self.alltoall_linear and sp_size in self.alltoall_linear:
            fit = self.alltoall_linear[sp_size]
            return max(0.0, fit["alpha"] * msg_mb + fit["beta"])

        # Priority 4: Topology-aware BW model
        topo_bw = self._select_a2a_bw(topo)
        if topo_bw and sp_size in topo_bw:
            return msg_mb / topo_bw[sp_size]

        # Priority 5: Generic BW model
        bw = self.alltoall_bw.get(sp_size, self.alltoall_bw.get(max(self.alltoall_bw.keys()), 100))
        return msg_mb / bw

    def alltoall_time(self, seqlens: List[int], sp_size: int,
                      topo: str = "consecutive") -> float:
        """All-to-All communication time (ms) for Ulysses SP."""
        if sp_size <= 1:
            return 0.0
        total_tokens = sum(seqlens)
        q_factor, kv_factor = self.head_padding_overhead(sp_size)
        qo_msg_mb = self.h * q_factor * total_tokens * 2 / 1024 / 1024 / sp_size
        kv_msg_mb = self.kv_hidden * kv_factor * total_tokens * 2 / 1024 / 1024 / sp_size
        num_qo_ops = 2 * 2 * self.l
        num_kv_ops = 2 * 2 * self.l

        qo_time = self._a2a_per_op_time(qo_msg_mb, sp_size, topo)
        kv_time = self._a2a_per_op_time(kv_msg_mb, sp_size, topo)
        return qo_time * num_qo_ops + kv_time * num_kv_ops

    def p2p_ring_time(self, seqlens: List[int], cp_size: int,
                      topo: str = "consecutive") -> float:
        """P2P ring communication time (ms) for Ring Attention."""
        if cp_size <= 1:
            return 0.0
        total_tokens = sum(seqlens)
        single_kv_mb = (total_tokens / cp_size) * self.kv_hidden * 2 / 1024 / 1024
        kv_per_step_mb = 2 * single_kv_mb

        per_step_time = self._ring_per_step_time(kv_per_step_mb, cp_size, topo)
        return per_step_time * (cp_size - 1) * self.l

    def _ring_per_step_time(self, kv_per_step_mb: float, cp_size: int,
                             topo: str = "consecutive") -> float:
        """Get per-step ring comm time (ms) with cascading fallback.
        
        Priority: topology-aware P2P linear fit → generic interpolation →
                  ring-step fit → generic P2P fit → topology-aware BW → generic BW.
        """
        # Priority 1: Topology-aware P2P linear fit
        topo_lin = self._select_p2p_linear(topo)
        if topo_lin and cp_size in topo_lin:
            fit = topo_lin[cp_size]
            return max(0.0, fit["alpha"] * kv_per_step_mb + fit["beta"])

        # Priority 2: Generic interpolation from actual ring profiling
        interp_val = self._interp_ring_per_step(kv_per_step_mb, cp_size)
        if interp_val is not None:
            return interp_val

        # Priority 3: Ring per-step linear fit
        if self.p2p_ring_step and cp_size in self.p2p_ring_step:
            fit = self.p2p_ring_step[cp_size]
            return max(0.0, fit["alpha"] * kv_per_step_mb + fit["beta"])

        # Priority 4: Generic raw P2P linear fit
        if self.p2p_linear and cp_size in self.p2p_linear:
            fit = self.p2p_linear[cp_size]
            single_kv_mb = kv_per_step_mb / 2
            per_kv_time = fit["alpha"] * single_kv_mb + fit["beta"]
            return max(0.0, 2 * per_kv_time)

        # Priority 5: Topology-aware BW model
        topo_bw = self._select_p2p_bw(topo)
        if topo_bw and cp_size in topo_bw:
            return kv_per_step_mb / topo_bw[cp_size]

        # Priority 6: Generic BW model
        bw = self.p2p_bw.get(cp_size, self.p2p_bw.get(max(self.p2p_bw.keys()), 100))
        return kv_per_step_mb / bw

    def usp_comm_time(self, seqlens: List[int], sp_size: int, cp_size: int,
                      placement: str = "context_first") -> float:
        """Communication time (ms) for USP (Ulysses + Ring combined).

        Placement determines which communication primitive gets the faster
        (consecutive/intra-node) topology and which gets the slower (strided).
        """
        a2a_topo = self._get_topo(placement, "alltoall")
        ring_topo = self._get_topo(placement, "ring")

        if sp_size <= 1:
            return self.p2p_ring_time(seqlens, cp_size, ring_topo)
        if cp_size <= 1:
            return self.alltoall_time(seqlens, sp_size, a2a_topo)

        total_tokens = sum(seqlens)
        parallel_size = sp_size * cp_size
        q_factor, kv_factor = self.head_padding_overhead(sp_size)

        qo_msg_mb = self.h * q_factor * total_tokens * 2 / 1024 / 1024 / parallel_size
        kv_msg_mb = self.kv_hidden * kv_factor * total_tokens * 2 / 1024 / 1024 / parallel_size
        num_qo_ops = 2 * 2 * self.l
        num_kv_ops = 2 * 2 * self.l

        qo_per_op = self._a2a_per_op_time(qo_msg_mb, sp_size, a2a_topo)
        kv_per_op = self._a2a_per_op_time(kv_msg_mb, sp_size, a2a_topo)
        a2a_time = qo_per_op * num_qo_ops + kv_per_op * num_kv_ops

        padded_kv_hidden = self.kv_hidden * kv_factor
        kv_hidden_after_uly = padded_kv_hidden / sp_size
        single_kv_mb = (total_tokens / cp_size) * kv_hidden_after_uly * 2 / 1024 / 1024
        kv_per_step_mb = 2 * single_kv_mb
        per_step_time = self._ring_per_step_time(kv_per_step_mb, cp_size, ring_topo)
        p2p_time = per_step_time * (cp_size - 1) * self.l

        return a2a_time + p2p_time

    def comm_time(self, seqlens: List[int], strategy: ParallelStrategy) -> float:
        """Communication time for a strategy."""
        placement = strategy.placement
        a2a_topo = self._get_topo(placement, "alltoall")
        ring_topo = self._get_topo(placement, "ring")
        if strategy.attn_type == "ulysses":
            return self.alltoall_time(seqlens, strategy.sp_size, a2a_topo)
        elif strategy.attn_type == "ring":
            return self.p2p_ring_time(seqlens, strategy.cp_size, ring_topo)
        elif strategy.attn_type == "usp":
            return self.usp_comm_time(seqlens, strategy.sp_size, strategy.cp_size, placement)
        else:
            raise ValueError(f"Unknown attn_type: {strategy.attn_type}")

    # ---- Ring P2P comm helpers (per-step, single direction) ----

    def _p2p_fwd_comm_per_step(self, total_tokens: int, cp_size: int,
                                kv_hidden: Optional[int] = None,
                                topo: str = "consecutive") -> float:
        """Forward ring: one step KV transfer time (ms)."""
        if kv_hidden is None:
            kv_hidden = self.kv_hidden
        single_kv_mb = (total_tokens / cp_size) * kv_hidden * 2 / 1024 / 1024
        kv_per_step_mb = 2 * single_kv_mb
        return self._ring_per_step_time(kv_per_step_mb, cp_size, topo)

    def _p2p_bwd_comm_per_step(self, total_tokens: int, cp_size: int,
                                kv_hidden: Optional[int] = None,
                                topo: str = "consecutive") -> float:
        """Backward ring: one step dual-ring transfer time (ms)."""
        fwd_step = self._p2p_fwd_comm_per_step(total_tokens, cp_size, kv_hidden, topo)
        return fwd_step * self.ring_bwd_comm_ratio

    # ---- Overlap-aware total time ----

    def _leaky_max(self, a: float, b: float) -> float:
        """Imperfect overlap: max(a,b) + leakage * min(a,b).
        
        leakage=0 → perfect overlap (pure max).
        leakage=1 → fully additive (a + b).
        """
        return max(a, b) + self.overlap_leakage * min(a, b)

    def _total_time_ring_overlap(self, seqlens: List[int],
                                  strategy: ParallelStrategy) -> float:
        """Overlap-aware total time for Ring Attention (fwd + bwd)."""
        cp_size = strategy.cp_size
        if cp_size <= 1:
            fwd_compute = self.compute_time(seqlens, strategy)
            return fwd_compute * (1 + self.bwd_fwd_ratio)

        total_tokens = sum(seqlens)
        ring_topo = self._get_topo(strategy.placement, "ring")

        step_compute_per_layer = self._ring_step_compute_per_layer(
            seqlens, strategy)

        fwd_comm_per_step = self._p2p_fwd_comm_per_step(total_tokens, cp_size, topo=ring_topo)
        bwd_comm_per_step = self._p2p_bwd_comm_per_step(total_tokens, cp_size, topo=ring_topo)

        fwd_per_layer = ((cp_size - 1) * self._leaky_max(step_compute_per_layer, fwd_comm_per_step)
                         + step_compute_per_layer)

        bwd_step = step_compute_per_layer * self.bwd_fwd_ratio
        bwd_per_layer = ((cp_size - 1) * self._leaky_max(bwd_step, bwd_comm_per_step)
                         + bwd_step)

        return (fwd_per_layer + bwd_per_layer) * self.l

    def _total_time_usp_overlap(self, seqlens: List[int],
                                 strategy: ParallelStrategy) -> float:
        """Overlap-aware total time for USP (Ulysses + Ring)."""
        sp_size = strategy.sp_size
        cp_size = strategy.cp_size
        placement = strategy.placement
        a2a_topo = self._get_topo(placement, "alltoall")
        ring_topo = self._get_topo(placement, "ring")

        if cp_size <= 1:
            fwd_compute = self.compute_time(seqlens, strategy)
            a2a_comm = self.alltoall_time(seqlens, sp_size, a2a_topo)
            return fwd_compute * (1 + self.bwd_fwd_ratio) + a2a_comm

        if sp_size <= 1:
            return self._total_time_ring_overlap(seqlens, strategy)

        total_tokens = sum(seqlens)
        parallel_size = sp_size * cp_size
        q_factor, kv_factor = self.head_padding_overhead(sp_size)

        qo_msg_mb = self.h * q_factor * total_tokens * 2 / 1024 / 1024 / parallel_size
        kv_msg_mb = self.kv_hidden * kv_factor * total_tokens * 2 / 1024 / 1024 / parallel_size
        qo_a2a_time = self._a2a_per_op_time(qo_msg_mb, sp_size, a2a_topo)
        kv_a2a_time = self._a2a_per_op_time(kv_msg_mb, sp_size, a2a_topo)
        a2a_fwd_per_layer = 2 * qo_a2a_time + 2 * kv_a2a_time
        a2a_bwd_per_layer = 2 * qo_a2a_time + 2 * kv_a2a_time

        step_compute = self._ring_step_compute_per_layer(seqlens, strategy)

        kv_hidden_after_uly = self.kv_hidden * kv_factor / sp_size
        fwd_comm_step = self._p2p_fwd_comm_per_step(total_tokens, cp_size, kv_hidden_after_uly, ring_topo)
        bwd_comm_step = self._p2p_bwd_comm_per_step(total_tokens, cp_size, kv_hidden_after_uly, ring_topo)

        ring_fwd_per_layer = ((cp_size - 1) * self._leaky_max(step_compute, fwd_comm_step)
                              + step_compute)
        bwd_step = step_compute * self.bwd_fwd_ratio
        ring_bwd_per_layer = ((cp_size - 1) * self._leaky_max(bwd_step, bwd_comm_step)
                              + bwd_step)

        fwd_per_layer = a2a_fwd_per_layer + ring_fwd_per_layer
        bwd_per_layer = a2a_bwd_per_layer + ring_bwd_per_layer

        return (fwd_per_layer + bwd_per_layer) * self.l

    # ---- Total time ----

    def total_time_single(self, seqlen: int, strategy: ParallelStrategy) -> float:
        """Total time for a single sequence (fwd + bwd, all layers)."""
        return self.total_time([seqlen], strategy)

    def total_time(self, seqlens: List[int], strategy: ParallelStrategy) -> float:
        """Total time for a set of sequences in one group (fwd + bwd, all layers).
        
        Dispatches to overlap-aware or additive model based on strategy type:
          - Ring Attention: overlap-aware (compute-comm overlap per ring step)
          - USP (Ulysses+Ring): overlap-aware (a2a blocking + ring overlap)
          - Ulysses: additive (a2a is blocking, no overlap opportunity)
        
        When enable_overlap_model=False, uses additive model for all strategies.
        Note: Ring additive model still correctly accounts for cp ring steps.
        """
        if self.enable_overlap_model and strategy.attn_type == "ring":
            return self._total_time_ring_overlap(seqlens, strategy)
        elif self.enable_overlap_model and strategy.attn_type == "usp":
            return self._total_time_usp_overlap(seqlens, strategy)
        elif strategy.attn_type == "ring":
            cp = strategy.cp_size
            ring_topo = self._get_topo(strategy.placement, "ring")
            step_compute = self._ring_step_compute_per_layer(seqlens, strategy)
            fwd_compute_per_layer = cp * step_compute
            total_compute = fwd_compute_per_layer * (1 + self.bwd_fwd_ratio) * self.l
            fwd_ring_comm = self.p2p_ring_time(seqlens, cp, ring_topo)
            total_comm = fwd_ring_comm * (1 + self.ring_bwd_comm_ratio)
            return total_compute + total_comm
        elif strategy.attn_type == "usp":
            sp, cp = strategy.sp_size, strategy.cp_size
            placement = strategy.placement
            a2a_topo = self._get_topo(placement, "alltoall")
            ring_topo = self._get_topo(placement, "ring")
            step_compute = self._ring_step_compute_per_layer(seqlens, strategy)
            fwd_compute_per_layer = cp * step_compute
            total_compute = fwd_compute_per_layer * (1 + self.bwd_fwd_ratio) * self.l

            total_tokens = sum(seqlens)
            parallel_size = sp * cp
            q_factor, kv_factor = self.head_padding_overhead(sp)
            qo_msg_mb = self.h * q_factor * total_tokens * 2 / 1024 / 1024 / parallel_size
            kv_msg_mb = self.kv_hidden * kv_factor * total_tokens * 2 / 1024 / 1024 / parallel_size
            qo_a2a = self._a2a_per_op_time(qo_msg_mb, sp, a2a_topo)
            kv_a2a = self._a2a_per_op_time(kv_msg_mb, sp, a2a_topo)
            a2a_comm = (2 * qo_a2a + 2 * kv_a2a) * 2 * self.l

            kv_h = self.kv_hidden * kv_factor / sp
            fwd_comm = self._p2p_fwd_comm_per_step(total_tokens, cp, kv_h, ring_topo)
            fwd_ring_comm = fwd_comm * (cp - 1) * self.l
            total_ring_comm = fwd_ring_comm * (1 + self.ring_bwd_comm_ratio)
            return total_compute + a2a_comm + total_ring_comm
        else:
            # Ulysses or legacy additive model
            # compute_time is fwd-only (1 call per layer); multiply by (1+bwd_fwd_ratio)
            fwd_compute = self.compute_time(seqlens, strategy)
            total_compute = fwd_compute * (1 + self.bwd_fwd_ratio)
            return total_compute + self.comm_time(seqlens, strategy)

    # ---- Memory ----

    def activation_size(self, seqlens: Union[int, List[int]], parallel_size: int = 1,
                         sp_size: int = 1) -> float:
        """Activation memory in MB.
        
        Args:
            seqlens: sequence lengths in the group.
            parallel_size: total parallel size (sp * cp or just sp or cp).
            sp_size: Ulysses SP size (for head padding overhead calculation).
                     Only relevant when sp_size is specified.
        """
        if isinstance(seqlens, list):
            total = sum(seqlens)
        else:
            total = seqlens
        base = self.act_per_token * total / parallel_size
        # Add extra activation memory for head padding (if applicable)
        if sp_size > 1:
            base += self.head_padding_extra_activation_mb(
                seqlens if isinstance(seqlens, list) else [seqlens],
                sp_size, parallel_size
            )
        return base

    def total_memory(self, seqlens: Union[int, List[int]] = 0, parallel_size: int = 1,
                     sp_size: int = 1) -> float:
        return self.model_states_mb + self.activation_size(seqlens, parallel_size, sp_size)

    def token_capacity(self, memory_limit_gb: int) -> int:
        """Max tokens per device given memory budget."""
        return int((memory_limit_gb * 1024 - self.model_states_mb) / self.act_per_token)

    # ---- Check / debug ----

    def check(self, seqlens: List[int], strategy: ParallelStrategy):
        fwd_compute = self.compute_time(seqlens, strategy)
        comm = self.comm_time(seqlens, strategy)
        total = self.total_time(seqlens, strategy)
        sp_for_mem = strategy.sp_size if strategy.attn_type in ("ulysses", "usp") else 1
        mem = self.total_memory(seqlens, strategy.parallel_size, sp_size=sp_for_mem)

        print(f"\n[seqlens={seqlens}, strategy={strategy}]")
        # Show head padding info
        if strategy.attn_type in ("ulysses", "usp") and strategy.sp_size > 1:
            q_fac, kv_fac = self.head_padding_overhead(strategy.sp_size)
            if q_fac > 1.0 or kv_fac > 1.0:
                print(f"  ⚠ Head padding: Q×{q_fac:.2f}, KV×{kv_fac:.2f} "
                      f"(n_heads={self.n_heads}, n_kv={self.n_kv_heads}, sp={strategy.sp_size})")
        cp = strategy.cp_size
        # For Ring/USP, show per-step breakdown
        if strategy.attn_type in ("ring", "usp") and cp > 1:
            step_compute = self._ring_step_compute_per_layer(seqlens, strategy)
            fwd_per_layer = cp * step_compute  # P equal steps
            fwd_total = fwd_per_layer * self.l
            print(f"  Fwd compute:   {fwd_total:.4f} ms "
                  f"({cp}×step={step_compute:.4f}, ×L={self.l})")
            print(f"  Bwd compute:   {fwd_total * self.bwd_fwd_ratio:.4f} ms "
                  f"(ratio={self.bwd_fwd_ratio:.2f})")
        else:
            fwd_total = fwd_compute
            print(f"  Fwd compute:   {fwd_compute:.4f} ms (×L={self.l})")
            print(f"  Bwd compute:   {fwd_compute * self.bwd_fwd_ratio:.4f} ms "
                  f"(ratio={self.bwd_fwd_ratio:.2f})")
        print(f"  Comm (fwd raw):{comm:.4f} ms")
        mode = 'overlap' if self.enable_overlap_model else 'additive'
        print(f"  Total:         {total:.4f} ms ({mode})")
        if self.enable_overlap_model and strategy.attn_type in ("ring", "usp"):
            # Show per-step breakdown for ring
            if cp > 1:
                total_tokens = sum(seqlens)
                kv_h = self.kv_hidden if strategy.attn_type == "ring" else self.kv_hidden // strategy.sp_size
                fwd_comm_step = self._p2p_fwd_comm_per_step(total_tokens, cp, kv_h)
                bwd_comm_step = self._p2p_bwd_comm_per_step(total_tokens, cp, kv_h)
                print(f"  Per-step (fwd): compute={step_compute:.4f}ms, "
                      f"comm={fwd_comm_step:.4f}ms → "
                      f"{'compute-bound' if step_compute > fwd_comm_step else 'comm-bound'}")
                print(f"  Per-step (bwd): compute={step_compute * self.bwd_fwd_ratio:.4f}ms, "
                      f"comm={bwd_comm_step:.4f}ms → "
                      f"{'compute-bound' if step_compute * self.bwd_fwd_ratio > bwd_comm_step else 'comm-bound'}")
        print(f"  Mem (MB):      {mem:.1f}")
    
    @classmethod
    def from_profile_files(
        cls,
        attention_json: str,
        alltoall_json: str,
        p2p_json: str,
        cluster_size: int = 8,
        param_size_B: float = 7.0,
        zero_stage: int = 3,
        act_per_token: float = 3.96,
        overlap_json: Optional[str] = None,
        gpus_per_node: int = 8,
    ) -> "AdaCPSPCostModel":
        """Construct a cost model from profiling output files."""
        with open(attention_json, "r") as f:
            attn_data = json.load(f)
        piecewise = []
        if "coefficients" in attn_data:
            for seg_name, coeff in attn_data["coefficients"].items():
                if coeff is not None:
                    piecewise.append({
                        "range": coeff["seq_range"],
                        "a": coeff["a"],
                        "b": coeff["b"],
                        "c": coeff["c"],
                    })
        elif "attention" in attn_data and "segments" in attn_data["attention"]:
            piecewise = attn_data["attention"]["segments"]
        config = attn_data.get("config", attn_data.get("attention", {}).get("config", {}))

        with open(alltoall_json, "r") as f:
            a2a_data = json.load(f)
        alltoall_bw = {int(k): v for k, v in a2a_data["bandwidth_dict_GBs"].items()}

        with open(p2p_json, "r") as f:
            p2p_data = json.load(f)
        p2p_bw = {int(k): v for k, v in p2p_data["bandwidth_dict_GBs"].items()}

        # Topology-aware bandwidth and linear fits (new format)
        alltoall_bw_consec = cls._load_topo_bw(a2a_data, "bandwidth_dict_consec_GBs")
        alltoall_bw_strided = cls._load_topo_bw(a2a_data, "bandwidth_dict_strided_GBs")
        p2p_bw_consec = cls._load_topo_bw(p2p_data, "bandwidth_dict_consec_GBs")
        p2p_bw_strided = cls._load_topo_bw(p2p_data, "bandwidth_dict_strided_GBs")
        alltoall_lin_c, alltoall_lin_s = cls._load_topo_linear_fits(a2a_data)
        p2p_lin_c, p2p_lin_s = cls._load_topo_linear_fits(p2p_data)

        bwd_fwd_ratio = 2.0
        ring_bwd_comm_ratio = 2.0
        if overlap_json is not None:
            with open(overlap_json, "r") as f:
                ovlp_data = json.load(f)
            if "fwd_bwd" in ovlp_data:
                bwd_fwd_ratio = ovlp_data["fwd_bwd"].get("avg_bwd_fwd_ratio", 2.0)
            if "ring_bwd_comm" in ovlp_data and "summary" in ovlp_data["ring_bwd_comm"]:
                ratios = [s["avg_bwd_fwd_comm_ratio"]
                          for s in ovlp_data["ring_bwd_comm"]["summary"].values()]
                if ratios:
                    ring_bwd_comm_ratio = sum(ratios) / len(ratios)

        return cls(
            cluster_size=cluster_size,
            hidden_size=config.get("hidden_size", 4096),
            layer_num=attn_data.get("num_layers", 32),
            param_size_B=param_size_B,
            zero_stage=zero_stage,
            act_per_token=act_per_token,
            num_attention_heads=config.get("n_heads", None),
            num_kv_heads=config.get("n_kv_heads", None),
            head_dim=config.get("head_dim", 128),
            piecewise_compute_coeffs=piecewise,
            alltoall_bandwidth_dict_gbs=alltoall_bw,
            p2p_bandwidth_dict_gbs=p2p_bw,
            bwd_fwd_ratio=bwd_fwd_ratio,
            ring_bwd_comm_ratio=ring_bwd_comm_ratio,
            gpus_per_node=gpus_per_node,
            alltoall_bw_consec=alltoall_bw_consec,
            alltoall_bw_strided=alltoall_bw_strided,
            p2p_bw_consec=p2p_bw_consec,
            p2p_bw_strided=p2p_bw_strided,
            alltoall_linear_consec=alltoall_lin_c,
            alltoall_linear_strided=alltoall_lin_s,
            p2p_linear_consec=p2p_lin_c,
            p2p_linear_strided=p2p_lin_s,
        )

    @staticmethod
    def _load_topo_bw(data: Dict, key: str) -> Optional[Dict[int, float]]:
        """Load topology-specific bandwidth dict from profile JSON."""
        if key not in data:
            return None
        return {int(k): v for k, v in data[key].items()}

    @staticmethod
    def _load_topo_linear_fits(data: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Extract consecutive/strided linear fits from profile JSON.

        The new JSON format stores linear_fits as {topo_key: {alpha, beta, r_squared}}.
        topo_key looks like "gs8_consecutive" or "gs16_strided".
        Returns (consec_dict, strided_dict) each mapping group_size -> {alpha, beta}.
        """
        if "linear_fits" not in data:
            return None, None
        consec, strided = {}, {}
        for tk, fit in data["linear_fits"].items():
            if "_consecutive" in tk:
                gs = int(tk.split("_")[0].replace("gs", ""))
                consec[gs] = {"alpha": fit["alpha"], "beta": fit["beta"]}
            elif "_strided" in tk:
                gs = int(tk.split("_")[0].replace("gs", ""))
                strided[gs] = {"alpha": fit["alpha"], "beta": fit["beta"]}
        return consec if consec else None, strided if strided else None

    @classmethod
    def from_unified_profile(
        cls,
        profile_json: str,
        cluster_size: int = 8,
        param_size_B: float = 7.0,
        zero_stage: int = 3,
        act_per_token: float = 3.96,
        hidden_size: int = 4096,
        layer_num: int = 32,
        overlap_json: Optional[str] = None,
        gpus_per_node: int = 8,
    ) -> "AdaCPSPCostModel":
        """Construct from unified profile_and_validate.py output (single JSON)."""
        with open(profile_json, "r") as f:
            data = json.load(f)

        piecewise = None
        if "attention" in data and "segments" in data["attention"]:
            piecewise = data["attention"]["segments"]
            cfg = data["attention"].get("config", {})
            hidden_size = cfg.get("hidden_size", hidden_size)

        alltoall_linear = {}
        p2p_linear = {}
        if "communication" in data and "linear_fits" in data["communication"]:
            for key, fit in data["communication"]["linear_fits"].items():
                gs = int(key.split("gs")[1])
                entry = {"alpha": fit["alpha_ms_per_MB"], "beta": fit["beta_ms"]}
                if key.startswith("alltoall"):
                    alltoall_linear[gs] = entry
                elif key.startswith("p2p"):
                    p2p_linear[gs] = entry

        bwd_fwd_ratio = 2.0
        ring_bwd_comm_ratio = 2.0
        if overlap_json is not None:
            with open(overlap_json, "r") as f:
                ovlp_data = json.load(f)
            if "fwd_bwd" in ovlp_data:
                bwd_fwd_ratio = ovlp_data["fwd_bwd"].get("avg_bwd_fwd_ratio", 2.0)
            if "ring_bwd_comm" in ovlp_data and "summary" in ovlp_data["ring_bwd_comm"]:
                ratios = [s["avg_bwd_fwd_comm_ratio"]
                          for s in ovlp_data["ring_bwd_comm"]["summary"].values()]
                if ratios:
                    ring_bwd_comm_ratio = sum(ratios) / len(ratios)

        return cls(
            cluster_size=cluster_size,
            hidden_size=hidden_size,
            layer_num=layer_num,
            param_size_B=param_size_B,
            zero_stage=zero_stage,
            act_per_token=act_per_token,
            piecewise_compute_coeffs=piecewise,
            alltoall_linear_fit=alltoall_linear if alltoall_linear else None,
            p2p_linear_fit=p2p_linear if p2p_linear else None,
            bwd_fwd_ratio=bwd_fwd_ratio,
            ring_bwd_comm_ratio=ring_bwd_comm_ratio,
            gpus_per_node=gpus_per_node,
        )

    @staticmethod
    def fit_linear_comm(profile_json: str, comm_type: str = "alltoall") -> Dict[int, Dict[str, float]]:
        """Fit linear model (time_ms = alpha * msg_MB + beta) from raw profile data.
        
        Args:
            profile_json: Path to alltoall_profile or p2p_ring_profile JSON.
            comm_type: "alltoall" or "p2p".
            
        Returns:
            Dict mapping group_size -> {"alpha": ms_per_MB, "beta": ms, "r_squared": float}
        """
        with open(profile_json, "r") as f:
            data = json.load(f)
        
        result = {}
        for gs_str, gs_data in data["results"].items():
            gs = int(gs_str)
            xs = []  # message sizes in MB
            ys = []  # times in ms
            for pt in gs_data["raw"]:
                xs.append(pt["msg_size_MB"])
                ys.append(pt["time_ms"])
            
            if len(xs) < 2:
                continue
            
            xs = np.array(xs, dtype=np.float64)
            ys = np.array(ys, dtype=np.float64)
            
            # Linear fit: y = alpha * x + beta
            n = len(xs)
            sx = np.sum(xs)
            sy = np.sum(ys)
            sxy = np.sum(xs * ys)
            sx2 = np.sum(xs ** 2)
            
            denom = n * sx2 - sx ** 2
            if abs(denom) < 1e-12:
                continue
            
            alpha = (n * sxy - sx * sy) / denom
            beta = (sy - alpha * sx) / n
            
            # R² calculation
            y_pred = alpha * xs + beta
            ss_res = np.sum((ys - y_pred) ** 2)
            ss_tot = np.sum((ys - np.mean(ys)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            result[gs] = {
                "alpha": float(alpha),
                "beta": float(beta),
                "r_squared": float(r_squared),
            }
        
        return result

    @staticmethod
    def fit_ring_per_step(profile_json: str,
                          max_kv_mb: float = 256.0,
                          ) -> Dict[int, Dict[str, float]]:
        """Fit ring per-step time from actual ring profiling (model data).
        
        Unlike fit_linear_comm which uses raw isolated P2P data, this uses the
        actual ring communication profile which captures:
          - Ring contention from multiple simultaneous send/recv
          - Bidirectional traffic effects
          - NCCL ring algorithm behavior
        
        Fits: per_step_time_ms = alpha * kv_per_step_MB + beta
        where kv_per_step_MB is the total K+V transfer per ring step.
        
        Args:
            profile_json: Path to p2p_ring_profile JSON.
            max_kv_mb: Maximum kv_per_step_MB to include in fit. Data points above
                       this are excluded because NCCL algorithm switching creates
                       bimodal behavior at very large message sizes.
            
        Returns:
            Dict mapping group_size -> {
                "alpha": ms per MB of KV transfer,
                "beta": ms latency per step,
                "r_squared": fit quality,
                "data_source": "ring_model"
            }
        """
        with open(profile_json, "r") as f:
            data = json.load(f)
        
        result = {}
        for gs_str, gs_data in data["results"].items():
            gs = int(gs_str)
            model_pts = gs_data.get("model", [])
            if not model_pts:
                continue
            
            xs = []  # kv_bytes_per_step_MB (total K+V per step)
            ys = []  # per_step_time_ms
            for pt in model_pts:
                kv_mb = pt["kv_bytes_per_step_MB"]
                t_ms = pt["per_step_time_ms"]
                if kv_mb <= max_kv_mb:
                    xs.append(kv_mb)
                    ys.append(t_ms)
            
            if len(xs) < 2:
                continue
            
            xs = np.array(xs, dtype=np.float64)
            ys = np.array(ys, dtype=np.float64)
            
            # Linear fit: per_step_time = alpha * kv_per_step_MB + beta
            n = len(xs)
            sx = np.sum(xs)
            sy = np.sum(ys)
            sxy = np.sum(xs * ys)
            sx2 = np.sum(xs ** 2)
            
            denom = n * sx2 - sx ** 2
            if abs(denom) < 1e-12:
                continue
            
            alpha = (n * sxy - sx * sy) / denom
            beta = (sy - alpha * sx) / n
            
            y_pred = alpha * xs + beta
            ss_res = np.sum((ys - y_pred) ** 2)
            ss_tot = np.sum((ys - np.mean(ys)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            result[gs] = {
                "alpha": float(alpha),
                "beta": float(beta),
                "r_squared": float(r_squared),
                "data_source": "ring_model",
            }
        
        return result

    @staticmethod
    def load_ring_interp(profile_json: str) -> Dict[int, List[Tuple[float, float]]]:
        """Load ring per-step interpolation table from P2P ring profile.
        
        Extracts (kv_per_step_MB, per_step_time_ms) pairs from the 'model' data
        in the profile JSON, which contains actual ring profiling results.
        
        Args:
            profile_json: Path to p2p_ring_profile JSON.
            
        Returns:
            Dict mapping group_size -> [(kv_per_step_MB, time_ms), ...] sorted by kv.
        """
        with open(profile_json, "r") as f:
            data = json.load(f)
        
        result = {}
        for gs_str, gs_data in data["results"].items():
            gs = int(gs_str)
            model_pts = gs_data.get("model", [])
            if not model_pts:
                continue
            
            pts = []
            for pt in model_pts:
                kv_mb = pt["kv_bytes_per_step_MB"]
                t_ms = pt["per_step_time_ms"]
                pts.append((float(kv_mb), float(t_ms)))
            
            pts.sort(key=lambda x: x[0])
            result[gs] = pts
        
        return result

    @staticmethod
    def load_a2a_interp(profile_json: str) -> Dict[int, List[Tuple[float, float]]]:
        """Load A2A per-op interpolation table from alltoall profile.
        
        Extracts (total_bytes_MB, time_ms) pairs from the 'model' data
        in the profile JSON, which measures actual A2A on attention tensors.
        
        Args:
            profile_json: Path to alltoall_profile JSON.
            
        Returns:
            Dict mapping group_size -> [(msg_MB, time_ms), ...] sorted by msg.
        """
        with open(profile_json, "r") as f:
            data = json.load(f)
        
        result = {}
        for gs_str, gs_data in data["results"].items():
            gs = int(gs_str)
            model_pts = gs_data.get("model", [])
            if not model_pts:
                continue
            
            pts = []
            for pt in model_pts:
                msg_mb = pt["total_bytes_MB"]
                t_ms = pt["time_ms"]
                pts.append((float(msg_mb), float(t_ms)))
            
            pts.sort(key=lambda x: x[0])
            result[gs] = pts
        
        return result

    def calibrate_from_validation(
        self,
        validation_json: str,
        min_seq_for_p2p: int = 8192,
        min_seq_for_a2a: int = 16384,
        min_seq_for_compute: int = 512,
        extrapolate_correction: bool = True,
    ) -> Dict[str, Any]:
        """Calibrate interpolation tables using real validation measurements.
        
        Profiling measures communication in isolation (tight loops), which can
        overestimate ring contention. Validation data measures communication
        in the context of actual layer-by-layer execution, giving more realistic
        timings. This method replaces profiling-based interpolation points with
        validation-derived values for clean (long-seq) data points.
        
        Also calibrates compute time: builds a seq_len → correction_factor
        table from validation compute data, correcting for the gap between
        isolated kernel profiling and actual execution context.
        
        Args:
            validation_json: Path to profile_validate_*.json file.
            min_seq_for_p2p: Minimum sequence length for P2P calibration points.
                Shorter sequences may have warmup artifacts.
            min_seq_for_a2a: Minimum sequence length for A2A calibration points.
                A2A short-seq data is often noisy/non-monotonic.
            min_seq_for_compute: Minimum sequence length for compute calibration.
            extrapolate_correction: If True, apply the last known correction ratio
                to profiling points beyond the validation range.
                
        Returns:
            Dict with calibration statistics (corrections applied, ratios, etc.)
        """
        import json as _json
        with open(validation_json, "r") as f:
            val_data = _json.load(f)
        
        stats = {"p2p_corrections": {}, "a2a_corrections": {}, 
                 "compute_corrections": [], "source": validation_json}
        
        # ── P2P Ring Calibration ──
        p2p_entries = [e for e in val_data.get("comm_validation", [])
                       if e["comm_type"] == "p2p" and e["seq_len"] >= min_seq_for_p2p]
        
        if self.p2p_ring_interp and p2p_entries:
            for gs in sorted(set(e["group_size"] for e in p2p_entries)):
                gs_entries = sorted(
                    [e for e in p2p_entries if e["group_size"] == gs],
                    key=lambda e: e["seq_len"]
                )
                if gs not in self.p2p_ring_interp:
                    continue
                
                old_pts = dict(self.p2p_ring_interp[gs])  # kv_mb -> time
                corrections = []
                
                for entry in gs_entries:
                    seq = entry["seq_len"]
                    measured_ms = entry["measured_ms"]
                    num_layers = entry.get("num_layers", self.l)
                    num_steps = gs - 1
                    
                    # Derive per-step time from validation
                    val_per_step = measured_ms / (num_layers * num_steps)
                    
                    # Compute kv_per_step_mb for this (seq, gs)
                    kv_per_step_mb = 2.0 * (seq / gs) * self.kv_hidden * 2 / 1024 / 1024
                    
                    # Find the closest profiling point
                    prof_per_step = old_pts.get(kv_per_step_mb)
                    if prof_per_step is None:
                        # Find nearest
                        closest_kv = min(old_pts.keys(), key=lambda k: abs(k - kv_per_step_mb))
                        if abs(closest_kv - kv_per_step_mb) / max(kv_per_step_mb, 1) < 0.01:
                            prof_per_step = old_pts[closest_kv]
                            kv_per_step_mb = closest_kv  # snap to profiled point
                    
                    ratio = val_per_step / prof_per_step if prof_per_step and prof_per_step > 0 else 1.0
                    corrections.append((kv_per_step_mb, val_per_step, prof_per_step, ratio))
                    
                    # Replace the profiling value with validation value
                    old_pts[kv_per_step_mb] = val_per_step
                
                stats["p2p_corrections"][gs] = corrections
                
                # Apply extrapolation correction to points beyond validation range
                if extrapolate_correction and corrections:
                    last_ratio = corrections[-1][3]
                    max_val_kv = corrections[-1][0]
                    for kv_mb in sorted(old_pts.keys()):
                        if kv_mb > max_val_kv:
                            old_pts[kv_mb] *= last_ratio
                
                # Rebuild sorted interpolation table
                self.p2p_ring_interp[gs] = sorted(old_pts.items(), key=lambda x: x[0])
        
        # ── A2A Calibration ──
        a2a_entries = [e for e in val_data.get("comm_validation", [])
                       if e["comm_type"] == "alltoall" and e["seq_len"] >= min_seq_for_a2a]
        
        if self.a2a_interp and a2a_entries:
            for gs in sorted(set(e["group_size"] for e in a2a_entries)):
                gs_entries = sorted(
                    [e for e in a2a_entries if e["group_size"] == gs],
                    key=lambda e: e["seq_len"]
                )
                if gs not in self.a2a_interp:
                    continue
                
                old_pts = dict(self.a2a_interp[gs])
                corrections = []
                
                for entry in gs_entries:
                    seq = entry["seq_len"]
                    measured_ms = entry["measured_ms"]
                    num_ops = entry["num_ops"]
                    
                    # Derive per-op time from validation
                    val_per_op = measured_ms / num_ops
                    
                    # Compute msg_mb for this (seq, gs) — using full hidden for Q/O tensor
                    msg_mb = seq * self.h * 2 / 1024 / 1024 / gs
                    
                    # Find the closest profiling point
                    prof_per_op = old_pts.get(msg_mb)
                    if prof_per_op is None:
                        closest_mb = min(old_pts.keys(), key=lambda k: abs(k - msg_mb))
                        if abs(closest_mb - msg_mb) / max(msg_mb, 1) < 0.01:
                            prof_per_op = old_pts[closest_mb]
                            msg_mb = closest_mb
                    
                    ratio = val_per_op / prof_per_op if prof_per_op and prof_per_op > 0 else 1.0
                    corrections.append((msg_mb, val_per_op, prof_per_op, ratio))
                    
                    old_pts[msg_mb] = val_per_op
                
                stats["a2a_corrections"][gs] = corrections
                
                if extrapolate_correction and corrections:
                    last_ratio = corrections[-1][3]
                    max_val_mb = corrections[-1][0]
                    for mb in sorted(old_pts.keys()):
                        if mb > max_val_mb:
                            old_pts[mb] *= last_ratio
                
                self.a2a_interp[gs] = sorted(old_pts.items(), key=lambda x: x[0])
        
        # ── Compute Calibration ──
        # Build a correction factor table from validation compute data.
        # Profiling in tight loops can differ from real execution, especially
        # for mid-range seq_lens where kernel caching/warmup differs.
        compute_entries = val_data.get("compute_validation", [])
        if compute_entries and isinstance(compute_entries, list):
            correction_pts = []
            for entry in sorted(compute_entries, key=lambda e: e.get("seq_len", 0)):
                seq = entry.get("seq_len", 0)
                if seq < min_seq_for_compute:
                    continue
                measured = entry.get("measured_per_layer_ms", 0)
                predicted = entry.get("predicted_per_layer_ms", 0)
                if predicted > 0 and measured > 0:
                    # Also compute from raw piecewise (in case predicted was from old model)
                    a, b, c = self._get_coeffs(seq)
                    raw_pred = a * seq ** 2 + b * seq + c
                    if raw_pred > 0:
                        ratio = measured / raw_pred
                        correction_pts.append((float(seq), ratio))
                        stats["compute_corrections"].append({
                            "seq_len": seq,
                            "measured_ms": measured,
                            "piecewise_ms": raw_pred,
                            "ratio": ratio,
                        })
            
            if correction_pts:
                self.compute_correction = sorted(correction_pts, key=lambda x: x[0])
        
        return stats


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

        # Placement override: "auto" (solver decides), "head_first", "context_first"
        self.force_placement: str = "auto"

        # Solver cache: maps a frozen set of sequence lengths to solver result
        # Avoids re-solving for batches with identical length distributions
        self._cache: Dict[tuple, Tuple[List, List]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._log_context: Optional[str] = None

    def _log(self, msg):
        if not self.hide_output:
            print(self._format_log_message(msg))

    def _format_log_message(self, msg: str) -> str:
        if not self._log_context:
            return msg

        lines = msg.split("\n")
        formatted = []
        for line in lines:
            if line:
                formatted.append(f"{self._log_context} {line}")
            else:
                formatted.append("")
        return "\n".join(formatted)

    # ---- Strategy pool generation ----

    def get_strategy_pool(self, seqs: Optional[List[Sequence]] = None,
                           max_head_padding_factor: float = 2.0) -> List[ParallelStrategy]:
        """Generate all valid strategies: ulysses, ring, AND usp combinations.

        GQA Head-Padding Aware:
          - Ring Attention has NO head divisibility constraint (no padding needed).
          - Ulysses SP requires n_heads % sp_size == 0 AND n_kv_heads % sp_size == 0.
            When not satisfied, head padding (GQA group replication) is used.
            The padding overhead factor = padded_n_q / n_q.
          - Strategies with padding overhead > max_head_padding_factor are excluded
            (they waste too much compute/communication on replicated heads).
          - USP can bypass this by using a smaller sp_size with a larger cp_size.

        For USP (combined Ulysses + Ring), we enumerate all (sp_size, cp_size)
        pairs where sp_size >= 2, cp_size >= 2, both powers of 2, and
        sp_size * cp_size <= max_parallel_size.

        Args:
            seqs: optional sequence list (unused, for API compatibility).
            max_head_padding_factor: maximum allowed padding overhead for Q heads.
                Default 2.0 means at most 2× compute overhead from head replication.
                Set to float('inf') to allow all strategies regardless of overhead.

        Example for Qwen2.5-7B (n_heads=28, n_kv=4), N=8:
          ulysses×1 (no pad), ulysses×2 (no pad), ulysses×4 (no pad),
          ulysses×8 → pad to kv=8, q=56 → factor=2.0 (included if max_factor>=2)
          ring×2, ring×4, ring×8 (always included, no head constraint)
          usp(sp2×cp2)=4, usp(sp2×cp4)=8, usp(sp4×cp2)=8 (sp≤4, no pad)
        """
        strategies = []
        n_kv = self.costmodel.n_kv_heads
        n_q = self.costmodel.n_heads

        def _q_pad_factor(sp_size: int) -> float:
            """Compute Q head padding factor for a given sp_size."""
            if sp_size <= 1:
                return 1.0
            if n_kv % sp_size == 0 and n_q % sp_size == 0:
                return 1.0
            g = n_q // n_kv
            padded_n_kv = math.ceil(n_kv / sp_size) * sp_size
            padded_n_q = padded_n_kv * g
            return padded_n_q / n_q

        # Start from min_parallel_size (= tp_deg in constrained mode)
        ps = self.min_parallel_size
        if ps <= 1:
            # Include no-parallelism baseline only when unconstrained
            strategies.append(ParallelStrategy("ulysses", 1))
            ps = 2

        # Pure Ulysses and pure Ring
        while ps <= self.max_parallel_size:
            for at in self.allowed_attn_types:
                if at == "ring":
                    # Ring Attention has no head divisibility constraint
                    strategies.append(ParallelStrategy(at, ps))
                elif at == "ulysses":
                    # Check head padding overhead
                    factor = _q_pad_factor(ps)
                    if factor <= max_head_padding_factor:
                        strategies.append(ParallelStrategy(at, ps))
                    else:
                        self._log(f"  [strategy_pool] Skipping ulysses×{ps}: "
                                  f"head padding factor {factor:.2f} > {max_head_padding_factor:.2f} "
                                  f"(n_heads={n_q}, n_kv={n_kv})")
            ps *= 2

        # USP combinations (if "usp" in allowed_attn_types)
        gpn = self.costmodel.gpus_per_node
        if "usp" in self.allowed_attn_types:
            sp = 2
            while sp <= self.max_parallel_size // 2:
                factor = _q_pad_factor(sp)
                if factor > max_head_padding_factor:
                    self._log(f"  [strategy_pool] Skipping usp(sp={sp},cp=*): "
                              f"head padding factor {factor:.2f} > {max_head_padding_factor:.2f}")
                    sp *= 2
                    continue
                cp = 2
                while sp * cp <= self.max_parallel_size:
                    total = sp * cp
                    if total >= self.min_parallel_size:
                        strategies.append(ParallelStrategy("usp", total, sp_size=sp, cp_size=cp,
                                                           placement="context_first"))
                        if total > gpn:
                            strategies.append(ParallelStrategy("usp", total, sp_size=sp, cp_size=cp,
                                                               placement="head_first"))
                    cp *= 2
                sp *= 2

        if not strategies:
            # Fallback: at least include the minimum strategy
            strategies.append(ParallelStrategy("ulysses", self.min_parallel_size))

        if self.force_placement != "auto":
            strategies = [
                s for s in strategies
                if s.attn_type != "usp" or s.placement == self.force_placement
            ]

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
        Ensures exactly group_num groups are produced (fills empty bins).
        """
        bin_capacity = self.device_token_capacity * strategy.parallel_size
        A = BestFitDecreasing(seqs, bin_capacity, group_num)
        if A is None:
            return None

        K = len(seqs)
        P_actual = A.shape[1]

        # Pad to ensure exactly group_num columns (fill empty bins)
        if P_actual < group_num:
            A = np.append(A, np.zeros((K, group_num - P_actual), dtype=np.int32), axis=1)
        P = group_num

        # Fill empty bins by stealing 1 sequence from the largest bin
        self._fill_empty_bins(A, seqs, K, P, bin_capacity)

        M = -1
        for p in range(P):
            group_tokens = sum(seqs[k].seq * A[k, p] for k in range(K)) / strategy.parallel_size
            if group_tokens > self.device_token_capacity:
                return None
            # Use full cost model (compute + comm + overlap) for accurate M
            group_seqlens = [seqs[k].seq for k in range(K) if A[k, p] > 0]
            if group_seqlens:
                group_time = self.costmodel.total_time(group_seqlens, strategy)
            else:
                group_time = 0.0
            M = max(group_time, M)

        return {
            "seqs": seqs,
            "strategies": [strategy] * P,
            "A": A,
            "M": M,
        }

    def solve_homo_strategy_ffd(
        self,
        seqs: List[Sequence],
        strategy: ParallelStrategy,
        group_num: int,
    ) -> Optional[Dict]:
        """
        Solve with a fixed strategy using First-Fit Decreasing.
        All groups use the same strategy.
        Ensures exactly group_num groups are produced (fills empty bins).
        """
        bin_capacity = self.device_token_capacity * strategy.parallel_size
        A = FirstFitDecreasing(seqs, bin_capacity, group_num)
        if A is None:
            return None

        K = len(seqs)
        P_actual = A.shape[1]

        # Pad to ensure exactly group_num columns (fill empty bins)
        if P_actual < group_num:
            A = np.append(A, np.zeros((K, group_num - P_actual), dtype=np.int32), axis=1)
        P = group_num

        # Fill empty bins by stealing 1 sequence from the largest bin
        self._fill_empty_bins(A, seqs, K, P, bin_capacity)

        M = -1
        for p in range(P):
            group_tokens = sum(seqs[k].seq * A[k, p] for k in range(K)) / strategy.parallel_size
            if group_tokens > self.device_token_capacity:
                return None
            # Use full cost model (compute + comm + overlap) for accurate M
            group_seqlens = [seqs[k].seq for k in range(K) if A[k, p] > 0]
            if group_seqlens:
                group_time = self.costmodel.total_time(group_seqlens, strategy)
            else:
                group_time = 0.0
            M = max(group_time, M)

        return {
            "seqs": seqs,
            "strategies": [strategy] * P,
            "A": A,
            "M": M,
        }

    @staticmethod
    def _fill_empty_bins(A, seqs, K, P, bin_capacity):
        """
        Fill empty bins by stealing 1 sequence from the largest non-empty bin.
        Ensures every bin has at least 1 sequence so all GPU groups are active.
        (Ported from FlexSP: fill_empty logic in solve_homo_sp_ffd_bfd_globalbatch)
        """
        for p in range(P):
            if np.sum(A[:, p]) == 0:
                # Find the largest non-empty bin (by sequence count)
                best_donor = -1
                best_count = 0
                for q in range(P):
                    cnt = int(np.sum(A[:, q]))
                    if cnt > best_count:
                        best_count = cnt
                        best_donor = q
                if best_donor >= 0 and best_count > 1:
                    # Steal the smallest sequence from the donor
                    donor_seqs_idx = [k for k in range(K) if A[k, best_donor] == 1]
                    donor_seqs_idx.sort(key=lambda k: seqs[k].seq)
                    k_steal = donor_seqs_idx[0]
                    A[k_steal, best_donor] = 0
                    A[k_steal, p] = 1

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

    def solve_adaptive_ffd(self, seqs: List[Sequence]) -> Optional[Dict]:
        """
        Try all homogeneous strategies with FFD, pick the one with minimum time.
        (Ported from FlexSP: homo_sp_baseline_ffd_bfd with type='ffd')
        """
        best_result = None
        strategies = self.get_strategy_pool(seqs)

        for strat in strategies:
            if strat.parallel_size > self.N:
                continue
            group_num = self.N // strat.parallel_size
            result = self.solve_homo_strategy_ffd(seqs, strat, group_num)
            if result is not None:
                if best_result is None or result["M"] < best_result["M"]:
                    best_result = result

        return best_result

    def solve_adaptive_heuristic(self, seqs: List[Sequence], heuristic: str = "bfd") -> Optional[Dict]:
        """
        Unified adaptive heuristic solver: try all strategies with BFD or FFD.
        (Ported from FlexSP: homo_sp_baseline_ffd_bfd)
        
        Args:
            seqs: list of sequences
            heuristic: "bfd" or "ffd"
        """
        if heuristic == "bfd":
            return self.solve_adaptive_bfd(seqs)
        elif heuristic == "ffd":
            return self.solve_adaptive_ffd(seqs)
        else:
            raise ValueError(f"Unknown heuristic: {heuristic}")

    # ---- Heterogeneous BFD/FFD heuristic ----

    def _generate_gpu_partitions(self) -> List[List[int]]:
        """
        Generate all valid partitions of N GPUs into groups where each group
        size is a power of 2 (≥ min_parallel_size, ≤ max_parallel_size).

        E.g. for N=8, min=1, max=8:
          [8], [4,4], [4,2,2], [2,2,2,2], [2,2,4], ...
        Partitions are sorted descending to avoid duplicates.
        """
        min_ps = max(1, self.min_parallel_size)
        max_ps = min(self.N, self.max_parallel_size)

        # Collect valid group sizes (powers of 2)
        valid_sizes = []
        ps = min_ps
        while ps <= max_ps:
            valid_sizes.append(ps)
            ps *= 2

        partitions = []

        def _partition(remaining: int, max_allowed: int, current: List[int]):
            if remaining == 0:
                partitions.append(current[:])
                return
            for sz in valid_sizes:
                if sz > remaining or sz > max_allowed:
                    continue
                current.append(sz)
                _partition(remaining - sz, sz, current)  # descending order to avoid duplicates
                current.pop()

        _partition(self.N, max_ps, [])
        return partitions

    def _strategies_for_group_size(self, group_size: int) -> List[ParallelStrategy]:
        """Generate all possible strategies for a given group size.

        For group_size=16 (gpn=8): ulysses×16, ring×16,
            usp(sp2×cp8,cf), usp(sp2×cp8,hf), usp(sp4×cp4,cf), usp(sp4×cp4,hf), ...
        """
        strats = []
        gpn = self.costmodel.gpus_per_node
        for at in self.allowed_attn_types:
            if at in ("ulysses", "ring"):
                strats.append(ParallelStrategy(at, group_size))
            elif at == "usp":
                sp = 2
                while sp <= group_size // 2:
                    if group_size % sp == 0:
                        cp = group_size // sp
                        if cp >= 2:
                            strats.append(ParallelStrategy("usp", group_size, sp_size=sp,
                                                           cp_size=cp, placement="context_first"))
                            if group_size > gpn:
                                strats.append(ParallelStrategy("usp", group_size, sp_size=sp,
                                                               cp_size=cp, placement="head_first"))
                    sp *= 2
        return strats

    def solve_heterogeneous_bfd(self, seqs: List[Sequence]) -> Optional[Dict]:
        """
        Heterogeneous BFD: enumerate all valid GPU partitions and strategy
        assignments, assign sequences with BFD, pick the best overall.

        Unlike solve_adaptive_bfd (which only tries homogeneous strategies),
        this can produce mixed groups like [ulysses×4, ring×4] in one microbatch.
        Also supports USP strategies within groups (e.g. usp(sp2×cp4) on 8 GPUs).

        Complexity: O(partitions × strategy_combos × K log K)
        For N=8, 2 attn_types: ~50 combinations — fast.
        For N=64, pruning needed (see max_hetero_combos).
        """
        partitions = self._generate_gpu_partitions()

        best_result = None
        max_combos = 500  # safety limit for large N

        combo_count = 0
        for partition in partitions:
            num_groups = len(partition)
            # For each group in the partition, get all valid strategies
            per_group_strats = [self._strategies_for_group_size(gs) for gs in partition]

            def _gen_assignments(idx, current):
                nonlocal combo_count, best_result
                if combo_count > max_combos:
                    return
                if idx == num_groups:
                    combo_count += 1
                    result = self._hetero_bfd_assign(seqs, list(current))
                    if result is not None:
                        if best_result is None or result["M"] < best_result["M"]:
                            best_result = result
                    return
                for strat in per_group_strats[idx]:
                    current.append(strat)
                    _gen_assignments(idx + 1, current)
                    current.pop()

            _gen_assignments(0, [])

        if best_result is not None:
            self._log(f"[Hetero BFD] Best: {best_result['M']:.2f} ms, "
                      f"strategies={[str(s) for s in best_result['strategies']]}")
        return best_result

    def _hetero_bfd_assign(
        self,
        seqs: List[Sequence],
        strategies: List[ParallelStrategy],
    ) -> Optional[Dict]:
        """
        Assign sequences to heterogeneous groups using Best-Fit Decreasing.

        Each group has its own strategy (attn_type, parallel_size) and thus
        its own memory capacity and time cost function.

        Args:
            seqs: list of sequences
            strategies: list of strategies, one per group

        Returns:
            result dict or None if infeasible
        """
        K = len(seqs)
        P = len(strategies)
        A = np.zeros((K, P), dtype=np.int32)

        # Compute per-group capacity
        capacities = [self.device_token_capacity * s.parallel_size for s in strategies]
        remaining = list(capacities)

        # Sort sequences descending by length
        seqs_sorted = sorted(enumerate(seqs), key=lambda x: x[1].seq, reverse=True)

        for orig_idx, seq in seqs_sorted:
            # Check if seq can fit in memory with any group's strategy
            min_ps = self._min_parallel_size(seq.seq)
            best_group = -1
            best_remaining = float('inf')
            
            for p in range(P):
                if strategies[p].parallel_size < min_ps:
                    continue
                if remaining[p] >= seq.seq:
                    # Best-fit: choose the group with the least remaining capacity after adding this seq
                    leftover = remaining[p] - seq.seq
                    if leftover < best_remaining:
                        best_remaining = leftover
                        best_group = p
            
            if best_group == -1:
                return None  # Infeasible

            A[seq.id, best_group] = 1
            remaining[best_group] -= seq.seq

        # Verify feasibility and compute max time
        M = -1
        for p in range(P):
            strat = strategies[p]
            group_tokens = sum(seqs[k].seq * A[k, p] for k in range(K))
            local_tokens = group_tokens / strat.parallel_size
            if local_tokens > self.device_token_capacity:
                return None

            group_seqs_lens = [seqs[k].seq for k in range(K) if A[k, p] > 0]
            if not group_seqs_lens:
                # Empty group — still counts GPUs but does no work
                continue

            group_time = self.costmodel.total_time(group_seqs_lens, strat)
            M = max(group_time, M)

        if M < 0:
            return None

        return {
            "seqs": seqs,
            "strategies": strategies,
            "A": A,
            "M": M,
        }

    def solve_heterogeneous_ffd(self, seqs: List[Sequence]) -> Optional[Dict]:
        """
        Heterogeneous FFD: like heterogeneous BFD but uses First-Fit Decreasing.
        Also supports USP strategies.
        """
        partitions = self._generate_gpu_partitions()

        best_result = None
        max_combos = 500

        combo_count = 0
        for partition in partitions:
            num_groups = len(partition)
            per_group_strats = [self._strategies_for_group_size(gs) for gs in partition]

            def _gen_assignments(idx, current):
                nonlocal combo_count, best_result
                if combo_count > max_combos:
                    return
                if idx == num_groups:
                    combo_count += 1
                    result = self._hetero_ffd_assign(seqs, list(current))
                    if result is not None:
                        if best_result is None or result["M"] < best_result["M"]:
                            best_result = result
                    return
                for strat in per_group_strats[idx]:
                    current.append(strat)
                    _gen_assignments(idx + 1, current)
                    current.pop()

            _gen_assignments(0, [])

        if best_result is not None:
            self._log(f"[Hetero FFD] Best: {best_result['M']:.2f} ms, "
                      f"strategies={[str(s) for s in best_result['strategies']]}")
        return best_result

    def _hetero_ffd_assign(
        self,
        seqs: List[Sequence],
        strategies: List[ParallelStrategy],
    ) -> Optional[Dict]:
        """
        Assign sequences to heterogeneous groups using First-Fit Decreasing.
        """
        K = len(seqs)
        P = len(strategies)
        A = np.zeros((K, P), dtype=np.int32)

        capacities = [self.device_token_capacity * s.parallel_size for s in strategies]
        remaining = list(capacities)

        seqs_sorted = sorted(enumerate(seqs), key=lambda x: x[1].seq, reverse=True)

        for orig_idx, seq in seqs_sorted:
            min_ps = self._min_parallel_size(seq.seq)
            placed = False

            for p in range(P):
                if strategies[p].parallel_size < min_ps:
                    continue
                if remaining[p] >= seq.seq:
                    A[seq.id, p] = 1
                    remaining[p] -= seq.seq
                    placed = True
                    break

            if not placed:
                return None

        # Verify feasibility and compute max time
        M = -1
        for p in range(P):
            strat = strategies[p]
            group_tokens = sum(seqs[k].seq * A[k, p] for k in range(K))
            local_tokens = group_tokens / strat.parallel_size
            if local_tokens > self.device_token_capacity:
                return None

            group_seqs_lens = [seqs[k].seq for k in range(K) if A[k, p] > 0]
            if not group_seqs_lens:
                continue

            group_time = self.costmodel.total_time(group_seqs_lens, strat)
            M = max(group_time, M)

        if M < 0:
            return None

        return {
            "seqs": seqs,
            "strategies": strategies,
            "A": A,
            "M": M,
        }

    # ---- Sequence bucketing ----

    def bucket_seqs(self, seqs: List[Sequence], bucket_num: int):
        """
        Bucket sequences for ILP complexity reduction.
        (Ported from FlexSP: bucket_seqs)
        
        Args:
            bucket_num: > 0 for DP bucketing, < 0 for even-distance bucketing
            
        Returns:
            (buckets, avg_error, actual_bucket_num)
        """
        if bucket_num > 0:  # DP bucketing
            bucket_error_ths = (sum(get_lens(seqs)) + self.cluster_token_capacity) / 2
            bucket_num = min(len(seqs), bucket_num)
            while True:
                buckets, avg_error = bucketing_seqs(seqs, bucket_num)
                bucket_total_token = sum(bkt.boundary * bkt.size for bkt in buckets)
                if bucket_total_token <= bucket_error_ths:
                    return buckets, avg_error, bucket_num
                bucket_num += 1
        elif bucket_num < 0:  # Even-distance bucketing
            bucket_num = -bucket_num
            max_seq = max(s.seq for s in seqs)
            seq_chunk = max_seq // bucket_num
            buckets = []
            avg_error = 0.0
            seqs_sorted = sorted(seqs)
            idx = 0
            for i in range(bucket_num):
                bound = (i + 1) * seq_chunk
                bucket = SeqBucket(bound)
                sel_seqs = []
                while idx < len(seqs_sorted) and seqs_sorted[idx].seq < bound:
                    sel_seqs.append(seqs_sorted[idx])
                    idx += 1
                if len(sel_seqs) > 0:
                    bucket.add_seqs(sel_seqs)
                    for seq in sel_seqs:
                        avg_error += (bound - seq.seq)
                    buckets.append(bucket)
            if idx < len(seqs_sorted):
                # Remaining sequences go into the last bucket
                remaining = seqs_sorted[idx:]
                bound = max_seq
                bucket = SeqBucket(bound)
                bucket.add_seqs(remaining)
                for seq in remaining:
                    avg_error += (bound - seq.seq)
                buckets.append(bucket)
            avg_error /= len(seqs)
            return buckets, avg_error, len(buckets)
        else:
            raise ValueError("bucket_num must be non-zero")

    def _get_bucket_min_parallel_size(self, buckets: List[SeqBucket]) -> List[int]:
        """Get minimum parallel_size for each bucket (based on bucket boundary)."""
        return [self._min_parallel_size(bkt.boundary) for bkt in buckets]

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

    # ---- Bucketed ILP solver (ported from FlexSP: solve_flexSP_bucket_seqs) ----

    def solve_adacpsp_bucket_ilp(
        self,
        seqs: List[Sequence],
        bucket_num: int = 16,
    ) -> Optional[Dict]:
        """
        Solve the AdaCPSP optimisation problem using ILP with sequence bucketing.
        (Ported from FlexSP: solve_flexSP_bucket_seqs)

        Instead of per-sequence binary variables, we use per-bucket integer variables:
          A[k, p] ∈ Z≥0 — number of sequences from bucket k assigned to group p

        This dramatically reduces ILP complexity for large batches:
          - Without bucketing: K×P binary variables (K = num_seqs)
          - With bucketing: B×P integer variables (B = num_buckets << K)

        Falls back to solve_adacpsp_ilp if bucket_num >= num_seqs.
        """
        try:
            from pyscipopt import Model as SCIPModel, quicksum
        except ImportError:
            self._log("[AdaCPSP] pyscipopt not available, falling back to BFD heuristic")
            return self.solve_adaptive_bfd(seqs)

        # Bucket the sequences
        buckets, avg_error, actual_bucket_num = self.bucket_seqs(seqs, bucket_num)
        if len(buckets) >= len(seqs):
            # Not enough compression, fall back to per-sequence ILP
            self._log("[AdaCPSP] Bucketing yields no compression, falling back to per-seq ILP")
            return self.solve_adacpsp_ilp(seqs, bucket_num=bucket_num)

        self._log(f"[AdaCPSP Bucket ILP] {actual_bucket_num} buckets, avg_error={avg_error:.2f}")

        strategy_options = self.get_strategy_options(seqs)
        bucket_min_ps = self._get_bucket_min_parallel_size(buckets)

        K = len(buckets)   # bucket count
        P = len(strategy_options)
        total_seq_num = sum(bkt.size for bkt in buckets)

        model = SCIPModel("AdaCPSP Bucket ILP")
        if self.hide_output:
            model.hideOutput()
        model.setParams(self.scip_param_dict)

        # Variables
        M = model.addVar(vtype="C", name="M", lb=0)
        model.setObjective(M, "minimize")

        # A[k,p] = number of seqs from bucket k assigned to group p (INTEGER, not binary)
        A = {(k, p): model.addVar(vtype="I", lb=0, name=f"A_{k}_{p}")
             for k in range(K) for p in range(P)}
        m = {p: model.addVar(vtype="B", name=f"m_{p}") for p in range(P)}

        # Constraints
        # 1. All sequences in each bucket must be assigned
        for k in range(K):
            model.addCons(quicksum(A[k, p] for p in range(P)) == buckets[k].size)
            # Prune: can't assign to groups with parallel_size < bucket's min needed
            for p in range(P):
                if strategy_options[p].parallel_size < bucket_min_ps[k]:
                    model.addCons(A[k, p] == 0)

        # 2. Group constraints
        for p in range(P):
            strat = strategy_options[p]
            ps = strat.parallel_size
            # Memory: total tokens (by bucket boundary) / parallel_size <= capacity
            model.addCons(
                quicksum(buckets[k].boundary * A[k, p] for k in range(K)) / ps
                <= self.device_token_capacity
            )
            # Time: group time <= M (using bucket boundary as proxy for seq length)
            model.addCons(
                quicksum(
                    self.costmodel.total_time_single(buckets[k].boundary, strat) * A[k, p]
                    for k in range(K)
                )
                <= M
            )
            # m[p] links
            model.addCons(quicksum(A[k, p] for k in range(K)) <= total_seq_num * m[p])
            model.addCons(quicksum(A[k, p] for k in range(K)) >= m[p])

        # 3. Device constraint
        model.addCons(quicksum(m[p] * strategy_options[p].parallel_size for p in range(P)) == self.N)

        # 4. Add warm-start initial solutions (ported from FlexSP: generate_balanced_initial_solution)
        seen_strategies = set()
        for strat in strategy_options:
            strat_key = (strat.attn_type, strat.parallel_size)
            if strat_key in seen_strategies:
                continue
            seen_strategies.add(strat_key)
            self._add_balanced_initial_solution(
                model, strategy_options, M, m, A, buckets, K, P, strat
            )

        # Solve
        if getattr(self, 'concurrent', False):
            model.solveConcurrent()
        else:
            model.optimize()
        status = model.getStatus()
        self._log(f"[AdaCPSP Bucket ILP] Status: {status}")

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
            "buckets": buckets,
            "K": K,
            "P": P,
        }

    def _add_balanced_initial_solution(
        self, model, strategy_options, M_var, m_vars, A_vars,
        buckets, K, P, target_strat: ParallelStrategy,
    ):
        """
        Add a warm-start initial solution for a given homogeneous strategy.
        (Ported from FlexSP: generate_balanced_initial_solution)

        Uses a min-heap to balance token load across groups.
        """
        # Find all groups with this strategy
        active_groups = []
        for p in range(P):
            if (strategy_options[p].attn_type == target_strat.attn_type and
                    strategy_options[p].parallel_size == target_strat.parallel_size):
                active_groups.append(p)

        if not active_groups:
            return
        
        solution = model.createSol()

        # Set m[p]
        for p in range(P):
            if p in active_groups:
                model.setSolVal(solution, m_vars[p], 1)
            else:
                model.setSolVal(solution, m_vars[p], 0)

        num_active = len(active_groups)

        if num_active == 1:
            target_group = active_groups[0]
            total_time = 0
            for k, bucket in enumerate(buckets):
                model.setSolVal(solution, A_vars[k, target_group], bucket.size)
                total_time += (self.costmodel.total_time_single(bucket.boundary, target_strat)
                               * bucket.size)
            model.setSolVal(solution, M_var, total_time)
        else:
            # Heap-based balanced assignment
            group_total_lengths = {p: 0 for p in active_groups}
            group_times = {p: 0 for p in active_groups}
            heap = [(0, p) for p in active_groups]
            heapq.heapify(heap)

            buckets_sorted = sorted(enumerate(buckets), key=lambda x: x[1].boundary, reverse=True)

            for orig_k, bucket in buckets_sorted:
                for _ in range(bucket.size):
                    assigned = False
                    tried = set()
                    while heap:
                        _, target_group = heapq.heappop(heap)
                        tried.add(target_group)

                        new_len = group_total_lengths[target_group] + bucket.boundary
                        if new_len / target_strat.parallel_size > self.device_token_capacity:
                            continue

                        current_val = model.getSolVal(solution, A_vars[orig_k, target_group]) or 0
                        model.setSolVal(solution, A_vars[orig_k, target_group], current_val + 1)

                        group_total_lengths[target_group] = new_len
                        group_times[target_group] += self.costmodel.total_time_single(
                            bucket.boundary, target_strat)

                        heapq.heappush(heap, (group_total_lengths[target_group], target_group))
                        assigned = True
                        break

                    if not assigned:
                        return  # Infeasible for this strategy

                    # Rebuild heap with remaining groups
                    heap = [(group_total_lengths[p], p) for p in active_groups if p not in tried]
                    heapq.heapify(heap)

            max_time = max(group_times.values()) if group_times else 0
            model.setSolVal(solution, M_var, max_time)

        if model.addSol(solution):
            self._log(f"  [Warm-Start] Added initial solution for {target_strat}")

    # ---- Result extraction for bucketed solver ----

    def _extract_groups_bucketed(self, result: Dict) -> List[Tuple[ParallelStrategy, List[Sequence]]]:
        """
        Extract (strategy, [sequences]) list from bucketed solver result.
        (Ported from FlexSP: show_results with bucket handling)

        When the solver uses bucket-level variables (A[k,p] = count),
        we need to randomly pop actual sequences from each bucket.
        """
        seqs = result["seqs"]
        strategies = result["strategies"]
        A = result["A"]
        buckets = result["buckets"]
        K, P = A.shape
        m = result.get("m", np.ones(P, dtype=np.int32))

        # Make deep copies of buckets to pop sequences from
        buckets_copy = deepcopy(buckets)

        groups = []
        for p in range(P):
            if m[p] == 0:
                continue
            strat = strategies[p]
            group_seqs = []
            for k in range(K):
                count = A[k, p]
                if count > 0:
                    popped = buckets_copy[k].random_pop_seqs(count)
                    if popped:
                        group_seqs.extend(popped)
            if group_seqs:
                groups.append((strat, group_seqs))

        return groups

    # ---- Homogeneous BFD/FFD for global batch (ported from FlexSP) ----

    def solve_homo_strategy_globalbatch(
        self,
        seqs_gb: List[Sequence],
        strategy: ParallelStrategy,
        heuristic: str = "bfd",
        fill_empty: bool = True,
        even_distribute: bool = True,
    ) -> Tuple[Optional[List], Optional[List]]:
        """
        Solve a global batch with a fixed homogeneous strategy using BFD/FFD.
        (Ported from FlexSP: solve_homo_sp_ffd_bfd_globalbatch)

        Handles: microbatch splitting, empty group filling, token redistribution.
        """
        group_num_per_mb = self.N // strategy.parallel_size
        group_capacity = max(
            self.device_token_capacity * strategy.parallel_size * 0.9,
            max(get_lens(seqs_gb))
        )

        # Initial bin packing across all sequences
        if heuristic == "bfd":
            A = BestFitDecreasing(seqs_gb, group_capacity)
        elif heuristic == "ffd":
            A = FirstFitDecreasing(seqs_gb, group_capacity)
        else:
            raise ValueError(f"Unknown heuristic: {heuristic}")

        if A is None:
            return None, None

        total_groups = A.shape[-1]
        mb_num = (total_groups + group_num_per_mb - 1) // group_num_per_mb
        empty_group_num = group_num_per_mb * mb_num - total_groups
        K = len(seqs_gb)

        # Pad with empty groups
        A = np.append(A, np.zeros((K, empty_group_num), dtype=np.int32), axis=1)

        def _token_lensum(groupA: np.ndarray) -> int:
            return sum(groupA[k] * seqs_gb[k].seq for k in range(K))

        if fill_empty and empty_group_num > 0:
            # Fill empty groups by stealing 1 sequence from the largest non-empty group
            group_id_0 = (mb_num - 1) * group_num_per_mb
            group_id_1 = mb_num * group_num_per_mb - empty_group_num
            group_id_2 = mb_num * group_num_per_mb
            for i in range(group_id_1, group_id_2):
                j = group_id_1 - 1
                while j >= 0:
                    if np.sum(A[:, j]) > 1:
                        k = 0
                        while A[k, j] == 0:
                            k += 1
                        break
                    j -= 1
                if j < 0:
                    break  # No group with >1 sequence found
                A[k, j], A[k, i] = 0, 1

            # Even distribution for last microbatch
            if even_distribute:
                mb_token_lensum = sum(_token_lensum(A[:, i]) for i in range(group_id_0, group_id_2))
                group_avg = mb_token_lensum // group_num_per_mb
                for i in range(group_id_0 + 1, group_id_2):
                    if _token_lensum(A[:, i]) > group_avg:
                        continue
                    for j in range(group_id_0, i):
                        k = 0
                        while (_token_lensum(A[:, i]) < group_avg and
                               _token_lensum(A[:, j]) > group_avg and
                               np.sum(A[:, j]) > 1):
                            if (A[k, j] == 1 and
                                    _token_lensum(A[:, i]) + seqs_gb[k].seq <= group_capacity):
                                A[k, j], A[k, i] = 0, 1
                            k += 1
                            if k >= K:
                                break

        # Build microbatch results
        globalbatch_groups, globalbatch_results = [], []
        sp_options = [strategy] * group_num_per_mb

        for mb_idx in range(mb_num):
            A_mb = A[:, mb_idx * group_num_per_mb:(mb_idx + 1) * group_num_per_mb]
            P = group_num_per_mb

            # Verify feasibility
            M_time = -1
            feasible = True
            for p in range(P):
                group_tokens = sum(seqs_gb[k].seq * A_mb[k, p] for k in range(K)) / strategy.parallel_size
                if group_tokens > self.device_token_capacity:
                    feasible = False
                    break
                group_time = sum(
                    self.costmodel.total_time_single(seqs_gb[k].seq, strategy) * A_mb[k, p]
                    for k in range(K)
                )
                M_time = max(group_time, M_time)

            if not feasible:
                return None, None

            result = {
                "seqs": seqs_gb,
                "strategies": sp_options,
                "A": A_mb,
                "M": M_time,
            }

            groups = []
            for p in range(P):
                group_seqs = [seqs_gb[k] for k in range(K) if A_mb[seqs_gb[k].id, p] == 1]
                groups.append((strategy, group_seqs))
            globalbatch_groups.append(groups)
            globalbatch_results.append(result)

        return globalbatch_groups, globalbatch_results

    # ---- Global batch solver ----

    def get_min_valid_microbatch_num(
        self,
        seqs_gb: List[Sequence],
        chunk_alg: str = "sort_consec",
    ) -> int:
        """
        Find the minimum number of microbatches such that each microbatch's
        total tokens fits within the cluster token capacity.
        (Ported from FlexSP: get_min_valid_microbatch_num)
        """
        total_tokens = sum(get_lens(seqs_gb))
        mb_num = max(1, int(np.ceil(total_tokens / self.cluster_token_capacity)))
        while True:
            seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)
            valid = True
            for seqs_mb in seqs_mb_all:
                if sum(get_lens(seqs_mb)) >= self.cluster_token_capacity:
                    valid = False
                    break
            if valid:
                break
            mb_num += 1
        return mb_num

    def solve_globalbatch(
        self,
        seqs_gb: List[Sequence],
        chunk_alg: str = "sort_consec",
        method: str = "adaptive_bfd",
        bucket_num: int = 16,
        _max_mb_retries: int = 20,
        log_context: Optional[str] = None,
    ) -> Tuple[List, List]:
        """
        Solve for a global batch:
          1. Determine minimum microbatch count
          2. Split into microbatches
          3. Solve each microbatch

        Methods:
          - "adaptive_bfd": BFD heuristic, try all strategies, pick best (fast)
          - "adaptive_ffd": FFD heuristic, try all strategies, pick best (fast)
          - "hetero_bfd": Heterogeneous BFD (tries mixed strategy groups)
          - "hetero_ffd": Heterogeneous FFD (tries mixed strategy groups)
          - "ilp": ILP per-sequence (exact, slow for large K)
          - "bucket_ilp": ILP with sequence bucketing (exact, faster for large K)
        """
        old_log_context = self._log_context
        if log_context is not None:
            self._log_context = log_context

        try:
            # --- Solver cache lookup ---
            cache_key = (method, tuple(sorted(s.seq for s in seqs_gb)))
            if cache_key in self._cache:
                self._cache_hits += 1
                self._log(f"[AdaCPSP] Cache HIT ({self._cache_hits}/{self._cache_hits+self._cache_misses})")
                cached_groups, cached_results = self._cache[cache_key]
                # Deep copy and remap IDs to current seqs
                from copy import deepcopy
                return deepcopy(cached_groups), deepcopy(cached_results)
            self._cache_misses += 1

            mb_num = self.get_min_valid_microbatch_num(seqs_gb, chunk_alg)

            self._log(f"[AdaCPSP] Using {mb_num} microbatch(es)")

            all_groups = []
            all_results = []

            while True:
                self._log(f"\n=========== Trying microbatch size = {mb_num} ===========")
                seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)
                feasible = True
                all_groups, all_results = [], []

                for i, seqs_mb in enumerate(seqs_mb_all):
                    # Save original IDs before re-indexing
                    orig_ids = [s.id for s in seqs_mb]
                    orig_lens = get_lens(seqs_mb)
                    seqs_mb_reindexed = [Sequence(seq=s.seq, id=j) for j, s in enumerate(seqs_mb)]

                    self._log(f"\n--- Microbatch {i} ({len(seqs_mb_reindexed)} seqs, "
                              f"{sum(orig_lens)} tokens) ---")
                    self._log(f"  Batch-local seq ids: {orig_ids}")
                    self._log(f"  Sequence lengths: {orig_lens}")

                    result = self._solve_microbatch(seqs_mb_reindexed, method, bucket_num)

                    if result is None:
                        feasible = False
                        all_groups, all_results = [], []
                        break

                    # Extract groups (handle bucketed vs non-bucketed results)
                    if "buckets" in result:
                        groups = self._extract_groups_bucketed(result)
                    else:
                        groups = self._extract_groups(result)

                    # Restore original IDs
                    for strat, group_seqs in groups:
                        for seq in group_seqs:
                            seq.id = orig_ids[seq.id]

                    all_groups.append(groups)
                    all_results.append(result)

                    if not self.hide_output:
                        for strat, group_seqs in groups:
                            seqlens = get_lens(group_seqs)
                            seq_ids = [seq.id for seq in group_seqs]
                            t = self.costmodel.total_time(seqlens, strat)
                            self._log(
                                f"  Group ({strat}): {len(group_seqs)} seqs, "
                                f"tokens={sum(seqlens)}, time={t:.2f} ms, "
                                f"batch_seq_ids={seq_ids}, seqlens={seqlens}"
                            )

                if feasible:
                    self._log(f"\n=========== Success with microbatch size = {mb_num} ! ===========")
                    break

                self._log(f"\n=========== Failed microbatch size = {mb_num} ! ===========")
                mb_num += 1
                if mb_num > _max_mb_retries + self.get_min_valid_microbatch_num(seqs_gb, chunk_alg):
                    self._log(f"[AdaCPSP] Too many retries, giving up")
                    return [], []

            # --- Store in cache ---
            self._cache[cache_key] = (all_groups, all_results)

            return all_groups, all_results
        finally:
            self._log_context = old_log_context

    def _solve_microbatch(
        self,
        seqs: List[Sequence],
        method: str,
        bucket_num: int = 16,
    ) -> Optional[Dict]:
        """Solve a single microbatch with the specified method."""
        if method == "adaptive_bfd":
            return self.solve_adaptive_bfd(seqs)
        elif method == "adaptive_ffd":
            return self.solve_adaptive_ffd(seqs)
        elif method == "hetero_bfd":
            return self.solve_heterogeneous_bfd(seqs)
        elif method == "hetero_ffd":
            return self.solve_heterogeneous_ffd(seqs)
        elif method == "ilp":
            return self.solve_adacpsp_ilp(seqs, bucket_num=bucket_num)
        elif method == "bucket_ilp":
            return self.solve_adacpsp_bucket_ilp(seqs, bucket_num=bucket_num)
        else:
            raise ValueError(f"Unknown method: {method}")

    # ---- Multiprocessing global batch solver ----
    # (Ported from FlexSP: solve_flexSP_globalbatch_mp)

    def solve_globalbatch_mp(
        self,
        seqs_gb: List[Sequence],
        chunk_alg: str = "sort_consec",
        method: str = "bucket_ilp",
        bucket_num: int = 16,
    ) -> Tuple[List, List]:
        """
        Solve global batch with parallel microbatch solving via multiprocessing.
        (Ported from FlexSP: solve_flexSP_globalbatch_mp)

        Each microbatch is solved in a separate process.
        """
        mb_num = self.get_min_valid_microbatch_num(seqs_gb, chunk_alg)

        while True:
            self._log(f"\n=========== Trying microbatch size = {mb_num} (MP) ===========")
            seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)

            # Serialize sequences for multiprocessing
            seqs_mb_serialized = [_serialize_seqs(seqs_mb) for seqs_mb in seqs_mb_all]

            manager = mp.Manager()
            stop_flag = manager.Value('i', 0)

            pool = mp.Pool(processes=min(mb_num, mp.cpu_count()))

            async_results = [
                pool.apply_async(_mp_worker, args=(
                    seqs_mb_ser, stop_flag, self.hide_output,
                    method, bucket_num,
                    self.N, self.mem_limit_gb,
                    self.min_parallel_size, self.max_parallel_size,
                    self.allowed_attn_types,
                    self.scip_param_dict,
                    self.costmodel.N, self.costmodel.h, self.costmodel.l,
                    self.costmodel.p, self.costmodel.zero_stage,
                    self.costmodel.act_per_token,
                    self.costmodel.piecewise,
                    self.costmodel.alltoall_bw, self.costmodel.p2p_bw,
                    self.costmodel.gpus_per_node,
                    self.costmodel.alltoall_bw_consec, self.costmodel.alltoall_bw_strided,
                    self.costmodel.p2p_bw_consec, self.costmodel.p2p_bw_strided,
                    self.costmodel.alltoall_linear_consec, self.costmodel.alltoall_linear_strided,
                    self.costmodel.p2p_linear_consec, self.costmodel.p2p_linear_strided,
                ))
                for seqs_mb_ser in seqs_mb_serialized
            ]

            processed = [False] * len(async_results)
            globalbatch_groups, globalbatch_results = [], []
            feasible = True

            try:
                while not all(processed):
                    for i, ar in enumerate(async_results):
                        if not processed[i] and ar.ready():
                            result = ar.get()
                            if result is None:
                                feasible = False
                                stop_flag.value = 1
                                pool.terminate()
                                pool.join()
                                break
                            else:
                                groups_ser, result_M = result
                                groups = _deserialize_strategy_groups(groups_ser)
                                # Restore original IDs
                                orig_ids = [s.id for s in seqs_mb_all[i]]
                                for strat, group_seqs in groups:
                                    for seq in group_seqs:
                                        seq.id = orig_ids[seq.id]
                                globalbatch_groups.append(groups)
                                globalbatch_results.append({"M": result_M})
                                processed[i] = True
                    if stop_flag.value == 1:
                        feasible = False
                        break
            except Exception as e:
                pool.terminate()
                pool.join()
                raise e

            if feasible:
                self._log(f"\n=========== Success with microbatch size = {mb_num} (MP) ! ===========")
                pool.close()
                pool.join()
                return globalbatch_groups, globalbatch_results

            self._log(f"\n=========== Failed microbatch size = {mb_num} (MP) ! ===========")
            try:
                pool.close()
            except Exception:
                pass
            pool.join()
            mb_num += 1

        return [], []

    def solve_globalbatch_mp_gbmb(
        self,
        seqs_gb: List[Sequence],
        chunk_alg: str = "sort_consec",
        method: str = "bucket_ilp",
        bucket_num: int = 16,
        mb_option_num: int = 5,
    ) -> Tuple[List, List]:
        """
        Solve global batch with parallel exploration of multiple microbatch numbers.
        (Ported from FlexSP: solve_flexSP_globalbatch_mp_gbmb)

        Tries mb_num, mb_num+1, ..., mb_num+mb_option_num-1 in parallel
        and picks the best (lowest total time).
        """
        mb_num_base = self.get_min_valid_microbatch_num(seqs_gb, chunk_alg)
        mb_num_options = [mb_num_base + i for i in range(mb_option_num)]

        seqs_gb_ser = _serialize_seqs(seqs_gb)

        manager = mp.Manager()
        result_dict = manager.dict()
        processes = []

        for mb_num in mb_num_options:
            p = mp.Process(
                target=_mp_gbmb_worker,
                args=(
                    seqs_gb_ser, mb_num, self.hide_output,
                    method, bucket_num, chunk_alg,
                    self.N, self.mem_limit_gb,
                    self.min_parallel_size, self.max_parallel_size,
                    self.allowed_attn_types,
                    self.scip_param_dict,
                    self.costmodel.N, self.costmodel.h, self.costmodel.l,
                    self.costmodel.p, self.costmodel.zero_stage,
                    self.costmodel.act_per_token,
                    self.costmodel.piecewise,
                    self.costmodel.alltoall_bw, self.costmodel.p2p_bw,
                    result_dict,
                    self.costmodel.gpus_per_node,
                    self.costmodel.alltoall_bw_consec, self.costmodel.alltoall_bw_strided,
                    self.costmodel.p2p_bw_consec, self.costmodel.p2p_bw_strided,
                    self.costmodel.alltoall_linear_consec, self.costmodel.alltoall_linear_strided,
                    self.costmodel.p2p_linear_consec, self.costmodel.p2p_linear_strided,
                )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Pick the best result
        feasible_results = []
        for mb_num, (gb_time, gb_groups_ser, gb_results) in result_dict.items():
            if gb_groups_ser is not None and gb_results is not None:
                gb_groups = [_deserialize_strategy_groups(g) for g in gb_groups_ser]
                feasible_results.append((mb_num, gb_time, gb_groups, gb_results))

        if feasible_results:
            best = min(feasible_results, key=lambda x: x[1])
            best_mb, best_time, best_groups, best_results = best
            self._log(f"\n=========== Best microbatch size = {best_mb}, time = {best_time:.2f} ===========")
            return best_groups, best_results
        else:
            self._log("\n=========== All mb_num attempts failed! ===========")
            return [], []

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
                sp_for_mem = strat.sp_size if strat.attn_type in ("ulysses", "usp") else 1
                mem = self.costmodel.total_memory(seqlens, strat.parallel_size, sp_size=sp_for_mem)
                print(f"  [{strat}] {len(group_seqs)} seqs, "
                      f"tokens={sum(seqlens)}, local_tokens={sum(seqlens)//strat.parallel_size}, "
                      f"time={t:.2f} ms, mem={mem:.1f} MB, seqlens={seqlens}")
        print(f"\nTotal time: {total_time:.2f} ms ({len(all_groups)} microbatches)")
        return total_time


# ──────────────────────────────────────────────────────────
# Multiprocessing helper functions
# ──────────────────────────────────────────────────────────

def _serialize_seqs(seqs: List[Sequence]) -> List[Tuple[int, int]]:
    return [(s.seq, s.id) for s in seqs]

def _deserialize_seqs(seqs_ser: List[Tuple[int, int]]) -> List[Sequence]:
    return [Sequence(seq=s, id=i) for s, i in seqs_ser]

def _serialize_strategy_groups(groups):
    """Serialize [(ParallelStrategy, [Sequence])] for IPC."""
    return [
        (strat.attn_type, strat.parallel_size, strat.sp_size, strat.cp_size,
         strat.placement, _serialize_seqs(seqs))
        for strat, seqs in groups
    ]

def _deserialize_strategy_groups(groups_ser):
    """Deserialize 6-tuple strategy groups from IPC."""
    result = []
    for item in groups_ser:
        if len(item) == 5:
            at, ps, sp, cp, seqs_ser = item
            placement = "context_first"
        else:
            at, ps, sp, cp, placement, seqs_ser = item
        strat = ParallelStrategy(at, ps, sp_size=sp, cp_size=cp, placement=placement)
        result.append((strat, _deserialize_seqs(seqs_ser)))
    return result

def _reconstruct_optimizer(
    cluster_size, mem_limit_gb, min_parallel_size, max_parallel_size,
    allowed_attn_types, scip_param_dict, hide_output,
    cm_N, cm_h, cm_l, cm_p, cm_zero, cm_act, cm_piecewise, cm_a2a_bw, cm_p2p_bw,
    cm_gpus_per_node=8,
    cm_a2a_bw_consec=None, cm_a2a_bw_strided=None,
    cm_p2p_bw_consec=None, cm_p2p_bw_strided=None,
    cm_a2a_lin_consec=None, cm_a2a_lin_strided=None,
    cm_p2p_lin_consec=None, cm_p2p_lin_strided=None,
):
    """Reconstruct AdaCPSPOptimizer in a worker process."""
    costmodel = AdaCPSPCostModel(
        cluster_size=cm_N, hidden_size=cm_h, layer_num=cm_l,
        param_size_B=cm_p, zero_stage=cm_zero, act_per_token=cm_act,
        piecewise_compute_coeffs=cm_piecewise,
        alltoall_bandwidth_dict_gbs=cm_a2a_bw,
        p2p_bandwidth_dict_gbs=cm_p2p_bw,
        gpus_per_node=cm_gpus_per_node,
        alltoall_bw_consec=cm_a2a_bw_consec,
        alltoall_bw_strided=cm_a2a_bw_strided,
        p2p_bw_consec=cm_p2p_bw_consec,
        p2p_bw_strided=cm_p2p_bw_strided,
        alltoall_linear_consec=cm_a2a_lin_consec,
        alltoall_linear_strided=cm_a2a_lin_strided,
        p2p_linear_consec=cm_p2p_lin_consec,
        p2p_linear_strided=cm_p2p_lin_strided,
    )
    return AdaCPSPOptimizer(
        cluster_size=cluster_size,
        memory_limit_gb=mem_limit_gb,
        costmodel=costmodel,
        hide_output=hide_output,
        scip_param_dict=scip_param_dict,
        allowed_attn_types=allowed_attn_types,
        max_parallel_size=max_parallel_size,
        min_parallel_size=min_parallel_size,
    )


def _mp_worker(
    seqs_mb_ser, stop_flag, hide_output,
    method, bucket_num,
    cluster_size, mem_limit_gb,
    min_parallel_size, max_parallel_size,
    allowed_attn_types, scip_param_dict,
    cm_N, cm_h, cm_l, cm_p, cm_zero, cm_act, cm_piecewise, cm_a2a_bw, cm_p2p_bw,
    cm_gpus_per_node=8,
    cm_a2a_bw_consec=None, cm_a2a_bw_strided=None,
    cm_p2p_bw_consec=None, cm_p2p_bw_strided=None,
    cm_a2a_lin_consec=None, cm_a2a_lin_strided=None,
    cm_p2p_lin_consec=None, cm_p2p_lin_strided=None,
):
    """Worker function for solve_globalbatch_mp."""
    if stop_flag.value == 1:
        return None

    seqs_mb = _deserialize_seqs(seqs_mb_ser)
    seqs_mb = [Sequence(seq=s.seq, id=j) for j, s in enumerate(seqs_mb)]

    optimizer = _reconstruct_optimizer(
        cluster_size, mem_limit_gb, min_parallel_size, max_parallel_size,
        allowed_attn_types, scip_param_dict, hide_output,
        cm_N, cm_h, cm_l, cm_p, cm_zero, cm_act, cm_piecewise, cm_a2a_bw, cm_p2p_bw,
        cm_gpus_per_node, cm_a2a_bw_consec, cm_a2a_bw_strided,
        cm_p2p_bw_consec, cm_p2p_bw_strided,
        cm_a2a_lin_consec, cm_a2a_lin_strided,
        cm_p2p_lin_consec, cm_p2p_lin_strided,
    )

    result = optimizer._solve_microbatch(seqs_mb, method, bucket_num)
    if result is None:
        stop_flag.value = 1
        return None

    if "buckets" in result:
        groups = optimizer._extract_groups_bucketed(result)
    else:
        groups = optimizer._extract_groups(result)

    groups_ser = _serialize_strategy_groups(groups)
    return (groups_ser, result["M"])


def _mp_gbmb_worker(
    seqs_gb_ser, mb_num, hide_output,
    method, bucket_num, chunk_alg,
    cluster_size, mem_limit_gb,
    min_parallel_size, max_parallel_size,
    allowed_attn_types, scip_param_dict,
    cm_N, cm_h, cm_l, cm_p, cm_zero, cm_act, cm_piecewise, cm_a2a_bw, cm_p2p_bw,
    result_dict,
    cm_gpus_per_node=8,
    cm_a2a_bw_consec=None, cm_a2a_bw_strided=None,
    cm_p2p_bw_consec=None, cm_p2p_bw_strided=None,
    cm_a2a_lin_consec=None, cm_a2a_lin_strided=None,
    cm_p2p_lin_consec=None, cm_p2p_lin_strided=None,
):
    """Worker function for solve_globalbatch_mp_gbmb."""
    seqs_gb = _deserialize_seqs(seqs_gb_ser)
    seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)

    optimizer = _reconstruct_optimizer(
        cluster_size, mem_limit_gb, min_parallel_size, max_parallel_size,
        allowed_attn_types, scip_param_dict, hide_output,
        cm_N, cm_h, cm_l, cm_p, cm_zero, cm_act, cm_piecewise, cm_a2a_bw, cm_p2p_bw,
        cm_gpus_per_node, cm_a2a_bw_consec, cm_a2a_bw_strided,
        cm_p2p_bw_consec, cm_p2p_bw_strided,
        cm_a2a_lin_consec, cm_a2a_lin_strided,
        cm_p2p_lin_consec, cm_p2p_lin_strided,
    )

    gb_groups, gb_results = [], []
    feasible = True

    for seqs_mb in seqs_mb_all:
        orig_ids = [s.id for s in seqs_mb]
        seqs_mb_reindexed = [Sequence(seq=s.seq, id=j) for j, s in enumerate(seqs_mb)]

        result = optimizer._solve_microbatch(seqs_mb_reindexed, method, bucket_num)
        if result is None:
            feasible = False
            break

        if "buckets" in result:
            groups = optimizer._extract_groups_bucketed(result)
        else:
            groups = optimizer._extract_groups(result)

        # Restore original IDs
        for strat, group_seqs in groups:
            for seq in group_seqs:
                seq.id = orig_ids[seq.id]

        gb_groups.append(_serialize_strategy_groups(groups))
        gb_results.append({"M": result["M"]})

    if feasible:
        gb_time = sum(r["M"] for r in gb_results)
        result_dict[mb_num] = (gb_time, gb_groups, gb_results)
    else:
        result_dict[mb_num] = (float("inf"), None, None)


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
    parser.add_argument("--memory_limit_gb", type=int, default=40)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--layer_num", type=int, default=32)
    parser.add_argument("--param_size_B", type=float, default=7.0)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="github")
    parser.add_argument("--seq_limit_k", type=int, default=32)
    parser.add_argument("--iter_num", type=int, default=5)
    parser.add_argument("--start_iter", type=int, default=3)
    parser.add_argument("--method", type=str, default="adaptive_bfd",
                        choices=["adaptive_bfd", "adaptive_ffd", "hetero_bfd", "hetero_ffd", "ilp", "bucket_ilp"])
    parser.add_argument("--solve_mode", type=str, default="sequential",
                        choices=["sequential", "mp", "mp_gbmb"],
                        help="sequential: solve MBs one by one; "
                             "mp: parallel MB solving; "
                             "mp_gbmb: parallel mb_num exploration")
    parser.add_argument("--bucket_num", type=int, default=16,
                        help="Positive for DP bucketing, negative for even-dist bucketing")
    parser.add_argument("--mb_option_num", type=int, default=5,
                        help="Number of mb_num options to explore (for mp_gbmb)")
    parser.add_argument("--attn_types", type=str, nargs="+",
                        default=["ulysses", "ring", "usp"],
                        choices=["ulysses", "ring", "usp"])
    parser.add_argument("--time_limit", type=int, default=10,
                        help="SCIP solver time limit in seconds")
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
    print(f" Method: {args.method}, Solve mode: {args.solve_mode}")
    print(f" Attn types: {args.attn_types}")
    if args.method in ["ilp", "bucket_ilp"]:
        print(f" Bucket num: {args.bucket_num}, SCIP time limit: {args.time_limit}s")
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
        scip_param_dict={"limits/time": args.time_limit},
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
    total_solver_time = 0
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

        if args.solve_mode == "mp_gbmb":
            all_groups, all_results = optimizer.solve_globalbatch_mp_gbmb(
                sequences, method=args.method, bucket_num=args.bucket_num,
                mb_option_num=args.mb_option_num,
            )
        elif args.solve_mode == "mp":
            all_groups, all_results = optimizer.solve_globalbatch_mp(
                sequences, method=args.method, bucket_num=args.bucket_num,
            )
        else:
            all_groups, all_results = optimizer.solve_globalbatch(
                sequences, method=args.method, bucket_num=args.bucket_num,
            )

        elapsed = time_module.time() - start
        total_solver_time += elapsed

        if all_results:
            iter_time = sum(r["M"] for r in all_results)
        else:
            iter_time = 0
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
    print(f" Summary: {args.iter_num} iterations")
    print(f"   Total estimated time = {total_time:.2f} ms")
    print(f"   Total solver time    = {total_solver_time:.3f} s")
    print(f"   Strategy usage: {dict(strategy_counter)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

