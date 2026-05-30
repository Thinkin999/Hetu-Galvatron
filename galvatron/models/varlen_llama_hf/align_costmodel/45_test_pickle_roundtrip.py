"""
Verify that pickling the optimizer faithfully preserves every cost-model
parameter that drives solver decisions, and that solving with the
unpickled copy yields IDENTICAL choices to solving with the original.

This is a regression test for the bug where _reconstruct_optimizer
silently dropped residual_*_per_sp, head padding, *_overhead_ms etc.
"""
from __future__ import annotations
import glob, json, os, pickle, sys

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
import importlib.util
_solver_path = os.path.join(REPO, "galvatron/models/varlen_llama_hf/adacpsp_solver.py")
spec = importlib.util.spec_from_file_location("adacpsp_solver", _solver_path)
_solver = importlib.util.module_from_spec(spec)
sys.modules["adacpsp_solver"] = _solver
spec.loader.exec_module(_solver)
AdaCPSPCostModel = _solver.AdaCPSPCostModel
AdaCPSPOptimizer = _solver.AdaCPSPOptimizer
ParallelStrategy = _solver.ParallelStrategy
Sequence = _solver.Sequence
_pickle_optimizer_for_workers = _solver._pickle_optimizer_for_workers
_reconstruct_optimizer_from_pickle = _solver._reconstruct_optimizer_from_pickle

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

opt = AdaCPSPOptimizer(cluster_size=16, memory_limit_gb=71.4, costmodel=cm,
                       hide_output=True, allowed_attn_types=["ulysses", "ring", "usp"])

# Sanity: critical fields populated in parent
print("=== Parent optimizer state ===")
print(f"  residual_b_per_sp = {opt.costmodel.residual_b_per_sp}")
print(f"  residual_a_per_sp = {opt.costmodel.residual_a_per_sp}")
print(f"  n_heads / n_kv    = {opt.costmodel.n_heads} / {opt.costmodel.n_kv_heads}")
print(f"  bwd_fwd_ratio     = {opt.costmodel.bwd_fwd_ratio}")
print(f"  ulysses_a2a_oh_ms = {opt.costmodel.ulysses_a2a_overhead_ms}")
print(f"  usp_a2a_extra_ms  = {opt.costmodel.usp_a2a_overhead_extra_ms}")
print(f"  ring_step_oh_ms   = {opt.costmodel.ring_step_overhead_ms}")
print(f"  b_step_fb_per_sp  = {opt.costmodel.b_step_fb_per_sp}")
print(f"  b_step_fb_default_ms = {opt.costmodel.b_step_fb_default_ms}")
print()

# Pickle round-trip
print("=== Pickle round-trip ===")
blob = _pickle_optimizer_for_workers(opt)
print(f"  pickled size: {len(blob)} bytes")
opt2 = _reconstruct_optimizer_from_pickle(blob)
print(f"  residual_b_per_sp = {opt2.costmodel.residual_b_per_sp}")
print(f"  residual_a_per_sp = {opt2.costmodel.residual_a_per_sp}")
print(f"  n_heads / n_kv    = {opt2.costmodel.n_heads} / {opt2.costmodel.n_kv_heads}")
print(f"  bwd_fwd_ratio     = {opt2.costmodel.bwd_fwd_ratio}")
print(f"  ulysses_a2a_oh_ms = {opt2.costmodel.ulysses_a2a_overhead_ms}")
print(f"  usp_a2a_extra_ms  = {opt2.costmodel.usp_a2a_overhead_extra_ms}")
print(f"  ring_step_oh_ms   = {opt2.costmodel.ring_step_overhead_ms}")
print(f"  b_step_fb_per_sp  = {opt2.costmodel.b_step_fb_per_sp}")
print(f"  b_step_fb_default_ms = {opt2.costmodel.b_step_fb_default_ms}")
print()

# Compute predictions on key strategies for 52K seq - both should match
for attn, sp, cp, place in [
    ("ulysses", 16, 1, "context_first"),
    ("usp",      2, 8, "head_first"),
    ("usp",      2, 8, "context_first"),
    ("ring",     1,16, "context_first"),
]:
    strat = ParallelStrategy(attn_type=attn,
                              parallel_size=sp*cp if attn=="usp" else (sp if attn=="ulysses" else cp),
                              sp_size=sp, cp_size=cp, placement=place)
    t1 = opt.costmodel.total_time([52192], strat)
    t2 = opt2.costmodel.total_time([52192], strat)
    t1_single = opt.costmodel.total_time_single(52192, strat)
    t2_single = opt2.costmodel.total_time_single(52192, strat)
    match = "OK" if abs(t1-t2) < 1e-3 and abs(t1_single-t2_single) < 1e-3 else "MISMATCH!"
    print(f"  {attn:7s} sp={sp:2d} cp={cp:2d} {place:<14s}  "
          f"parent_total={t1:7.1f}  child_total={t2:7.1f}   "
          f"parent_single={t1_single:7.1f}  child_single={t2_single:7.1f}   {match}")
