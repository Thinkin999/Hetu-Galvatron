"""
Diagnose memory model: dump device_token_capacity vs. the actual tokens-per-GPU
that successfully ran in forced cells, to verify whether the solver's memory
estimate is too conservative (forcing too many microbatches).
"""
from __future__ import annotations
import glob, json, os, sys
from pathlib import Path

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
import importlib.util
_solver_path = os.path.join(REPO, "galvatron/models/varlen_llama_hf/adacpsp_solver.py")
spec = importlib.util.spec_from_file_location("adacpsp_solver", _solver_path)
_solver = importlib.util.module_from_spec(spec)
sys.modules["adacpsp_solver"] = _solver
spec.loader.exec_module(_solver)
AdaCPSPCostModel = _solver.AdaCPSPCostModel
AdaCPSPOptimizer = _solver.AdaCPSPOptimizer

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

print(f"# Cost model loaded")
print(f"# act_per_token       = {cm.act_per_token} MB / token")
print(f"# model_states_mb     = {cm.model_states_mb:.1f} MB (zero{cm.zero_stage})")
print(f"# n_heads / n_kv      = {cm.n_heads} / {cm.n_kv_heads}")
print(f"# layers / hidden     = {cm.l} / {cm.h}")
print(f"# cluster_size        = {cm.N}")
print()

# Build optimizer with various mem_limit_gb to see device_token_capacity
print(f"=== device_token_capacity sweep (= (mem_gb*1024 - model_states_mb) / act_per_token) ===")
print(f"{'mem_gb':>8s}  {'tokens/GPU':>12s}  {'tokens/cluster_16':>18s}  {'tokens/USP-cp8-grp':>20s}")
for mem in [40, 56, 72, 80, 96]:
    opt = AdaCPSPOptimizer(cluster_size=16, memory_limit_gb=mem, costmodel=cm, hide_output=True)
    cap = opt.device_token_capacity
    print(f"{mem:>8d}  {cap:>12d}  {cap*16:>18d}  {cap*8:>20d}")
print()

# Now look at forced bench JSONLs to see actual tokens/GPU that ran successfully
print(f"=== Tokens packed per GPU in forced cells (chunks=1, 16 GPUs) ===")
bench_dir = Path(REPO) / "galvatron/models/varlen_llama_hf/align_costmodel/results/ghmb_zero2_precreate_full_20260529_181618/end2end"
for cell in sorted(bench_dir.iterdir()):
    if "chunks1" not in cell.name:
        continue
    f = cell / "rank0.jsonl"
    if not f.exists(): continue
    recs = [json.loads(l) for l in open(f) if l.strip()]
    train = [r for r in recs if r.get("phase") == "train_step"]
    if not train: continue
    # In each forced 1-mb cell, the global_batch.tokens is split across 1 mb.
    # The number of GPUs participating is determined by the strategy (sp_size or cp_size or sp*cp).
    # tokens per GPU = global_tokens / parallel_size_of_the_group
    per_gpu_tokens = []
    for r in train[5:]:
        gb = r.get("global_batch", {})
        for mb in gb.get("microbatches", []):
            for g in mb.get("groups", []):
                ps = int(g.get("parallel_size", 1))
                tok = int(g.get("tokens", 0))
                per_gpu_tokens.append(tok / ps)
    if per_gpu_tokens:
        max_pgt = max(per_gpu_tokens)
        avg_pgt = sum(per_gpu_tokens) / len(per_gpu_tokens)
        n_groups = len(per_gpu_tokens)
        print(f"  {cell.name:<24s}  max_tok/GPU={max_pgt:>7.0f}  avg_tok/GPU={avg_pgt:>7.0f}  n_groups={n_groups}")

print()
print("=== Cauto multi-mb decisions (why did solver split?) ===")
cauto = bench_dir / "adacpsp_chunksauto" / "rank0.jsonl"
if cauto.exists():
    recs = [json.loads(l) for l in open(cauto) if l.strip()]
    train = [r for r in recs if r.get("phase") == "train_step"][5:]
    for i, r in enumerate(train[:8], start=5):
        gb = r.get("global_batch", {})
        total_tokens = int(gb.get("global_tokens", 0))
        mbs = gb.get("microbatches", [])
        sizes = []
        for mb in mbs:
            t = sum(int(g.get("tokens",0)) for g in mb.get("groups", []))
            sizes.append(t)
        print(f"  iter {i}: total={total_tokens} → {len(mbs)} mbs: {sizes}")

# The KEY check: does forced ulysses8_c1 successfully run with more tokens/GPU
# than what solver's device_token_capacity allows?  If yes, capacity is too low.
print()
print("=== KEY CHECK ===")
print("If `max_tok/GPU in ulysses8_c1` > `device_token_capacity at default 72GB`,")
print("the solver is forcing more mbs than necessary.")
