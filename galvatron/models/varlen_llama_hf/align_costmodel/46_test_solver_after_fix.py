"""
Drive the solver directly with the iter-1 sequences and verify that after
the pickle fix, the solver picks USP sp=2 cp=8 head_first (cost-min) for
the long 52192-token sequence instead of ulysses sp=16.

If this passes, the bug is real and our fix works. Then we run the full
end-to-end bench to confirm wall-clock improvement.
"""
from __future__ import annotations
import glob, json, os, sys, time

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
import importlib.util
_solver_path = os.path.join(REPO, "galvatron/models/varlen_llama_hf/adacpsp_solver.py")
spec = importlib.util.spec_from_file_location("adacpsp_solver", _solver_path)
_solver = importlib.util.module_from_spec(spec)
sys.modules["adacpsp_solver"] = _solver
spec.loader.exec_module(_solver)
AdaCPSPCostModel = _solver.AdaCPSPCostModel
AdaCPSPOptimizer = _solver.AdaCPSPOptimizer
Sequence = _solver.Sequence

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

opt = AdaCPSPOptimizer(
    cluster_size=16, memory_limit_gb=71.4, costmodel=cm,
    hide_output=False, allowed_attn_types=["ulysses", "ring", "usp"],
)

# Use the actual iter-1 sequences from the previous benchmark
# (recorded in adacpsp_chunksauto/rank0.jsonl line 1):
ITER1_LENS = [64, 6880, 384, 960, 2528, 576, 2976, 640,
              2400, 4896, 1280, 32, 2912, 1568, 1600, 52192]
seqs = [Sequence(seq=L, id=i) for i, L in enumerate(ITER1_LENS)]

print(f"\n=== Solving iter-1 with FIXED solver (mp_gbmb mode) ===")
print(f"  num_seqs={len(seqs)}  total_tokens={sum(s.seq for s in seqs)}  max={max(s.seq for s in seqs)}")
print()

t0 = time.time()
groups_all, results_all = opt.solve_globalbatch_mp_gbmb(
    seqs, chunk_alg="sort_consec", method="ilp", bucket_num=16, mb_option_num=5,
)
elapsed = time.time() - t0
print(f"\n=== Solver done in {elapsed:.2f}s; chose {len(groups_all)} microbatches ===")

# Print decision per microbatch
total_pred_ms = 0
for i, (groups, res) in enumerate(zip(groups_all, results_all)):
    mb_M = float(res.get("M", 0))
    total_pred_ms += mb_M
    print(f"\n  MB{i}: M={mb_M:.1f}ms,  {len(groups)} group(s)")
    for strat, group_seqs in groups:
        lens = [s.seq for s in group_seqs]
        t = cm.total_time(lens, strat)
        print(f"    {strat}  total_time={t:.1f}ms  seqs={sorted(lens, reverse=True)}")

print(f"\n=== Summary ===")
print(f"  total predicted FB ms (sum of M):  {total_pred_ms:.1f}")
print(f"  cost-model 'all-in-1-mb USP sp=2 cp=8 hf' lower bound: "
      f"{cm.total_time(ITER1_LENS, _solver.ParallelStrategy('usp', 16, sp_size=2, cp_size=8, placement='head_first')):.1f}")
