"""Diagnose the latency-floor fix: print _comm_v2_time for typical
ring/usp/ulysses message sizes BEFORE and AFTER the fix.

Runs without GPU. Compares against measured points to show how much error
was reduced.
"""
from __future__ import annotations
import sys, os, glob, json

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "galvatron/site_package"))

from galvatron.models.varlen_llama_hf.adacpsp_solver import AdaCPSPCostModel

cfg = os.path.join(REPO, "galvatron/models/varlen_llama_hf/configs")
attn = sorted(glob.glob(f"{cfg}/profile_validate_*.json"), reverse=True)[0]
comm = sorted(glob.glob(f"{cfg}/comm_profile_*.json"), reverse=True)[0]

cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
    attention_json=attn, comm_profile_json=comm,
    cluster_size=16, gpus_per_node=8,
)

# Test message sizes covering tiny → measured → large
test_sizes = [0.001, 0.01, 0.05, 0.1, 0.2,
              0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
              20.0, 40.0]

print("\nLatency-floor lookup behaviour by primitive (FIXED):")
print("(below 0.25 MB uses the latency floor instead of linear fit)\n")

for prim, gs, topo in [
    ("p2p_kv_pair", 8, "consecutive"),
    ("p2p_kv_pair", 4, "consecutive"),
    ("alltoall_single", 8, "consecutive"),
    ("alltoall_single", 4, "consecutive"),
    ("p2p_sendrecv", 8, "consecutive"),
]:
    print(f"--- {prim} gs={gs} topo={topo} ---")
    # Show measured points
    pdata = cm.comm_v2.get(prim, {}).get(f"gs{gs}_{topo}", {})
    pts = pdata.get("points") or []
    fit = pdata.get("linear_fit") or {}
    print(f"  measured pts: {[(round(x,3), round(y,3)) for x,y in pts[:8]]}")
    print(f"  linear_fit:   alpha={fit.get('alpha', 0):.4f}, beta={fit.get('beta', 0):.4f}")
    print(f"  {'msg_MB':>8} {'fixed_ms':>10} {'old_fit_ms':>10} {'overpred_x':>10}")
    for mb in test_sizes:
        t_new = cm._comm_v2_time(prim, gs, topo, mb)
        alpha = fit.get("alpha", 0); beta = fit.get("beta", 0)
        t_old = max(0.0, alpha * mb + beta)
        ratio = t_old / t_new if t_new and t_new > 0 else 0
        marker = "  <-- below floor" if mb < 0.25 else ""
        print(f"  {mb:>8} {t_new:>10.4f} {t_old:>10.4f} {ratio:>9.2f}x{marker}")
    print()
