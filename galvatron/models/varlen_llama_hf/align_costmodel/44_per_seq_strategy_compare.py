"""
For a single sequence (or list of sequences) on a single group of N=16 GPUs,
print the cost-model prediction for every viable strategy, sorted.

This is what the solver USES per-group when deciding which strategy to assign.
If the prediction here is biased, the solver picks wrong.
"""
from __future__ import annotations
import glob, json, os, sys

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
import importlib.util
_solver_path = os.path.join(REPO, "galvatron/models/varlen_llama_hf/adacpsp_solver.py")
spec = importlib.util.spec_from_file_location("adacpsp_solver", _solver_path)
_solver = importlib.util.module_from_spec(spec)
sys.modules["adacpsp_solver"] = _solver
spec.loader.exec_module(_solver)
AdaCPSPCostModel = _solver.AdaCPSPCostModel
ParallelStrategy = _solver.ParallelStrategy

cfg = os.path.join(REPO, "galvatron/models/varlen_llama_hf/configs")
attn_p = sorted(glob.glob(os.path.join(cfg, "profile_validate_*.json")), reverse=True)[0]
comm_p = sorted(glob.glob(os.path.join(cfg, "comm_profile_*.json")), reverse=True)[0]
bdec_p = sorted(glob.glob(os.path.join(cfg, "b_decomp_profile_*.json")), reverse=True)[0]

cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
    attention_json=attn_p, comm_profile_json=comm_p,
    cluster_size=16, validation_json=attn_p, gpus_per_node=8,
)
with open(bdec_p) as f:
    cm.apply_b_decomp_profile(json.load(f))

# Sequences to test
SEQS_TO_TEST = [
    [52192],          # iter 1's big seq
    [48992],          # iter 6's big seq
    [55296],          # iter 9's big seq
    [29760, 27872],   # iter 5's combined (if 1 mb)
    [65536],          # forced cell single-mb token count
]


def candidates_ps16():
    cands = []
    cands.append(("ulysses", 1, 16, "context_first"))  # ring16 (sp=1,cp=16)
    cands.append(("ring", 1, 16, "context_first"))
    cands.append(("ulysses", 16, 1, "context_first"))
    cands.append(("ulysses", 8, 1, "context_first"))  # only sp=8 with N=16 → 2 groups, but here we test single group
    for sp in [2, 4, 8]:
        for cp in [2, 4, 8, 16]:
            if sp * cp != 16:
                continue
            if sp > cp:  # also try sp>cp combos but they would have higher head padding
                cands.append(("usp", sp, cp, "context_first"))
                cands.append(("usp", sp, cp, "head_first"))
            else:
                cands.append(("usp", sp, cp, "context_first"))
                cands.append(("usp", sp, cp, "head_first"))
    return cands


def total_for(seqs, attn, sp, cp, place):
    strat = ParallelStrategy(attn_type=attn, parallel_size=sp*cp if attn=="usp" else (sp if attn=="ulysses" else cp),
                              sp_size=sp, cp_size=cp, placement=place)
    return cm.total_time(seqs, strat)


for seqs in SEQS_TO_TEST:
    total = sum(seqs)
    print(f"\n=== Sequence(s) {seqs}  total={total} ===")
    rows = []
    for attn, sp, cp, place in candidates_ps16():
        try:
            t = total_for(seqs, attn, sp, cp, place)
            rows.append((t, attn, sp, cp, place))
        except Exception as exc:
            print(f"  SKIP {attn} sp={sp} cp={cp} {place}: {exc}")
    rows.sort()
    # Break down compute / comm / residual for top 8
    print(f"  {'rank':>4s} {'attn':<8s} {'sp':>3s} {'cp':>3s} {'place':<14s} "
          f"{'total':>9s} {'compute':>9s} {'comm':>9s} {'residual':>9s}")
    for i, (t, attn, sp, cp, place) in enumerate(rows[:10]):
        strat = ParallelStrategy(attn_type=attn,
                                  parallel_size=sp*cp if attn=="usp" else (sp if attn=="ulysses" else cp),
                                  sp_size=sp, cp_size=cp, placement=place)
        if attn == "ring":
            # ring overlap or additive
            if cm.enable_overlap_model:
                compute = cm._total_time_ring_overlap(seqs, strat) - cm.p2p_ring_time(seqs, cp, cm._get_topo(place, "ring", sp, cp)) * (1 + cm.ring_bwd_comm_ratio)
                comm = cm.p2p_ring_time(seqs, cp, cm._get_topo(place, "ring", sp, cp)) * (1 + cm.ring_bwd_comm_ratio)
            else:
                fwd_compute = cm.compute_time(seqs, strat)
                compute = fwd_compute * (1 + cm.bwd_fwd_ratio)
                comm = cm.comm_time(seqs, strat)
        elif attn == "usp":
            # break down approximately
            fwd_compute = cm.compute_time(seqs, strat)
            compute = fwd_compute * (1 + cm.bwd_fwd_ratio)
            comm = max(0, t - compute - cm.residual_time(seqs, strat))
        else:
            fwd_compute = cm.compute_time(seqs, strat)
            compute = fwd_compute * (1 + cm.bwd_fwd_ratio)
            comm = cm.comm_time(seqs, strat)
        resid = cm.residual_time(seqs, strat)
        marker = " ←best" if i == 0 else ""
        print(f"  {i+1:>4d} {attn:<8s} {sp:>3d} {cp:>3d} {place:<14s} "
              f"{t:>9.0f} {compute:>9.0f} {comm:>9.0f} {resid:>9.0f}{marker}")
