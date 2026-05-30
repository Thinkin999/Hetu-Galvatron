"""
Localize the cauto under-prediction to the straggler GROUP, using 1-microbatch
iters (where measured step fb == straggler group's actual time, since the step
is barrier-gated and there is only one microbatch).

For each 1-mb iter:
  measured_fb      = rank0 forward_backward (== straggler group actual time)
  predicted_total  = predicted_adacpsp.total_fb_ms
  straggler group  = argmax_g predicted_ms
  decompose the straggler group's cost-model prediction into:
      attn_compute, attn_comm, residual_a (a*tokens_per_gpu), residual_b (bias)
  so we can see WHAT the model thinks the straggler is made of and whether the
  linear term (which should include the huge LM head) is large enough.

CPU-only (no GPU needed).
"""
from __future__ import annotations
import glob, json, os, sys

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
import importlib.util
_p = os.path.join(REPO, "galvatron/models/varlen_llama_hf/adacpsp_solver.py")
spec = importlib.util.spec_from_file_location("adacpsp_solver", _p)
m = importlib.util.module_from_spec(spec); sys.modules["adacpsp_solver"] = m
spec.loader.exec_module(m)

cfg = os.path.join(REPO, "galvatron/models/varlen_llama_hf/configs")
attn = sorted(glob.glob(os.path.join(cfg, "profile_validate_qwen2.5-7b_*.json")), reverse=True)[0]
comm = sorted(glob.glob(os.path.join(cfg, "comm_profile_v2_*.json")), reverse=True)[0]
bdec = sorted(glob.glob(os.path.join(cfg, "b_decomp_profile_*2step*.json")), reverse=True)
cm = m.AdaCPSPCostModel.from_attention_and_comm_profiles(
    attention_json=attn, comm_profile_json=comm, cluster_size=16,
    validation_json=attn, gpus_per_node=8)
if bdec:
    with open(bdec[0]) as f:
        cm.apply_b_decomp_profile(json.load(f))
    print(f"# applied b_decomp: {os.path.basename(bdec[0])}")
print(f"# act layers L={cm.l}, n_heads={cm.n_heads}, kv={cm.n_kv_heads}, hidden={cm.h}")
print(f"# residual_a_per_sp={cm.residual_a_per_sp}")
print(f"# residual_b_per_sp={cm.residual_b_per_sp}\n")

RUN = os.path.join(REPO, "galvatron/models/varlen_llama_hf/align_costmodel/results/cauto_postfix_20260529_202459/end2end/adacpsp_chunksauto/rank0.jsonl")
recs = [json.loads(l) for l in open(RUN) if l.strip()]
train = [r for r in recs if r.get("phase") == "train_step"]

PS = m.ParallelStrategy

def decompose(group):
    attn_t = group["attn_type"]; sp = int(group["sp_size"]); cp = int(group["cp_size"])
    ps = int(group["parallel_size"]); pl = group.get("placement", "context_first")
    tok = int(group["tokens"]); nseq = int(group.get("num_sequences", 1))
    # reconstruct seqlens: if 1 seq, [tok]; else approximate equal split
    if nseq <= 1:
        seqs = [tok]
    else:
        per = max(1, tok // nseq); seqs = [per]*nseq; seqs[-1] += tok - per*nseq
    S = PS(attn_type=attn_t, parallel_size=ps, sp_size=sp, cp_size=cp, placement=pl)
    total = cm.total_time(seqs, S)
    resid = cm.residual_time(seqs, S)
    # residual split
    sp_eff = sp if attn_t in ("ulysses","usp") else 1
    a = cm.residual_a_per_sp.get(sp_eff, cm.residual_a_default_per_token)
    b = cm.residual_b_per_sp.get(sp_eff, cm.residual_b_default_ms)
    resid_a = a * (sum(seqs)/ps)
    resid_b = b
    # attn compute (fwd*(1+bwd)) – approximate via compute_time
    fwd = cm.compute_time(seqs, S)
    attn_compute = fwd * (1 + cm.bwd_fwd_ratio)
    attn_total = total - resid
    attn_comm = attn_total - attn_compute
    return dict(attn_t=attn_t, sp=sp, cp=cp, ps=ps, pl=pl, tok=tok, nseq=nseq,
                total=total, attn_compute=attn_compute, attn_comm=attn_comm,
                resid_a=resid_a, resid_b=resid_b,
                tok_per_gpu=sum(seqs)/ps)

print(f"{'it':>3} {'meas_fb':>8} {'pred_tot':>8} {'ratio':>6}  straggler-group decomposition (ms)")
print(f"{'':>3} {'':>8} {'':>8} {'':>6}  {'strat':<16} {'tok/gpu':>8} {'attn_cmp':>8} {'attn_cmm':>8} {'res_a':>7} {'res_b':>6}")
for r in train[5:]:
    it = r["loader_iter"]
    fb = float(r["timings_ms"]["forward_backward"])
    p = r.get("predicted_adacpsp", {})
    if not isinstance(p, dict): continue
    if int(p.get("n_microbatches", 0)) != 1: continue  # only 1-mb iters
    groups = p["microbatches"][0]["groups"]
    strag = max(groups, key=lambda g: float(g["predicted_ms"]))
    d = decompose(strag)
    pred_tot = float(p.get("total_fb_ms", 0))
    ratio = pred_tot / fb if fb else 0
    strat = f"{d['attn_t']}sp{d['sp']}cp{d['cp']}{d['pl'][:2]}"
    print(f"{it:>3} {fb:>8.0f} {pred_tot:>8.0f} {ratio:>5.0%}  {strat:<16} "
          f"{d['tok_per_gpu']:>8.0f} {d['attn_compute']:>8.0f} {d['attn_comm']:>8.0f} "
          f"{d['resid_a']:>7.0f} {d['resid_b']:>6.0f}")
