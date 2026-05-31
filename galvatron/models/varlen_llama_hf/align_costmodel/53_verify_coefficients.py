"""
Verify the new residual coefficients against a measured galvatron full-model
(L=28) fwd+bwd. Builds the cost model exactly as train_dist does, installs the
new sp-independent residual `a` + recompute-aware act_per_token, and predicts
total_time for a forced ulysses:1 group, comparing to the measured [FBPROF].

Usage: python 53_verify_coefficients.py --seq 4096 --measured_ms <X> [--recompute 0]
"""
import argparse, glob, json, os, sys

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
import importlib.util
_p = os.path.join(REPO, "galvatron/models/varlen_llama_hf/adacpsp_solver.py")
spec = importlib.util.spec_from_file_location("adacpsp_solver", _p)
m = importlib.util.module_from_spec(spec); sys.modules["adacpsp_solver"] = m
spec.loader.exec_module(m)

# Validated coefficients (CLEAN_PROFILE_FINDINGS.md)
A_NO_RECOMPUTE = 0.195   # ms/token, L=28
A_RECOMPUTE = 0.252
ACT_NO_RECOMPUTE = 5.3   # MB/token
ACT_RECOMPUTE = 0.87


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--measured_ms", type=float, default=None)
    ap.add_argument("--recompute", type=int, default=0)
    args = ap.parse_args()

    cfg = os.path.join(REPO, "galvatron/models/varlen_llama_hf/configs")
    attn = sorted(glob.glob(os.path.join(cfg, "profile_validate_qwen2.5-7b_*.json")), reverse=True)[0]
    comm = sorted(glob.glob(os.path.join(cfg, "comm_profile_v2_*.json")), reverse=True)[0]
    cm = m.AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn, comm_profile_json=comm, cluster_size=8,
        validation_json=attn, gpus_per_node=8)

    a = A_RECOMPUTE if args.recompute else A_NO_RECOMPUTE
    act = ACT_RECOMPUTE if args.recompute else ACT_NO_RECOMPUTE
    # install: sp-independent residual a, no fixed b, recompute-aware act
    cm.residual_a_default_per_token = a
    cm.residual_a_per_sp = {}
    cm.residual_b_default_ms = 0.0
    cm.residual_b_per_sp = {}
    cm.act_per_token = act

    print(f"# layers L={cm.l}, bwd_fwd_ratio={cm.bwd_fwd_ratio}, a={a}, act={act}")
    S = m.ParallelStrategy(attn_type="ulysses", parallel_size=1, sp_size=1, cp_size=1,
                           placement="head_first")
    # decompose
    fwd_attn = cm.compute_time([args.seq], S)          # forward attention (LUT) × L
    attn_total = fwd_attn * (1 + cm.bwd_fwd_ratio)
    resid = cm.residual_time([args.seq], S)
    total = cm.total_time([args.seq], S)
    print(f"\n=== ulysses sp=1, seq={args.seq}, recompute={args.recompute} ===")
    print(f"  attn (LUT×(1+bwd_fwd)) = {attn_total:8.1f} ms")
    print(f"  residual (a·tok + b)   = {resid:8.1f} ms   (a·{args.seq}={a*args.seq:.0f})")
    print(f"  comm (sp=1)            = {total-attn_total-resid:8.1f} ms")
    print(f"  TOTAL predicted        = {total:8.1f} ms")
    if args.measured_ms:
        err = (total - args.measured_ms) / args.measured_ms * 100
        print(f"  MEASURED fb            = {args.measured_ms:8.1f} ms")
        print(f"  error                  = {err:+.1f}%")


if __name__ == "__main__":
    main()
