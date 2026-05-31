"""
Compare cost-model prediction vs measured fb for the n1 communication-strategy
sweep (/tmp/n1_comm/summary.tsv). Loads the cost model exactly as train_dist
(newest residual + b_decomp profiles), predicts total_time for each forced
strategy/seq, reports error. This validates the COMM side (a2a/ring) needed for
the solver to rank ulysses vs ring vs usp.
"""
import glob, json, os, sys

REPO = "/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron"
import importlib.util
_p = os.path.join(REPO, "galvatron/models/varlen_llama_hf/adacpsp_solver.py")
spec = importlib.util.spec_from_file_location("adacpsp_solver", _p)
m = importlib.util.module_from_spec(spec); sys.modules["adacpsp_solver"] = m
spec.loader.exec_module(m)


def build_cm():
    cfg = os.path.join(REPO, "galvatron/models/varlen_llama_hf/configs")
    attn = sorted(glob.glob(os.path.join(cfg, "profile_validate_qwen2.5-7b_*.json")), reverse=True)[0]
    comm = sorted(glob.glob(os.path.join(cfg, "comm_profile_v2_*.json")), reverse=True)[0]
    cm = m.AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn, comm_profile_json=comm, cluster_size=8,
        validation_json=attn, gpus_per_node=8)
    for r in sorted(glob.glob(os.path.join(cfg, "residual_profile_*.json")), reverse=True):
        d = json.load(open(r))
        if d.get("schema") == "adacpsp_residual_v1":
            cm.apply_residual_profile(d); break
    for r in sorted(glob.glob(os.path.join(cfg, "b_decomp_profile_*.json")), reverse=True):
        d = json.load(open(r))
        if d.get("schema") == "adacpsp_b_decomp_v1":
            cm.apply_b_decomp_profile(d); break
    return cm


def strat_of(s):
    if s.startswith("ulysses:"):
        sp = int(s.split(":")[1]); return m.ParallelStrategy("ulysses", sp, sp_size=sp, cp_size=1, placement="head_first")
    if s.startswith("ring:"):
        cp = int(s.split(":")[1]); return m.ParallelStrategy("ring", cp, sp_size=1, cp_size=cp, placement="context_first")
    if s.startswith("usp:"):
        a, b = s.split(":")[1].split("x"); sp, cp = int(a), int(b)
        return m.ParallelStrategy("usp", sp*cp, sp_size=sp, cp_size=cp, placement="head_first")
    raise ValueError(s)


def main():
    cm = build_cm()
    print(f"# a_default={cm.residual_a_default_per_token}, act={cm.act_per_token}, "
          f"bwd_fwd={cm.bwd_fwd_ratio}, L={cm.l}")
    rows = [l.strip().split("\t") for l in open("/tmp/n1_comm/summary.tsv") if l.strip()][1:]
    print(f"\n{'strategy':<12} {'seq':>7} {'gbs':>4} {'measured':>9} {'predicted':>10} {'err':>8}  "
          f"{'comm':>7} {'resid':>7}")
    for r in rows:
        # columns: strategy seq gbs fb before after
        if len(r) < 4 or r[3] == "NA":
            continue
        strat_s, seq_s, gbs_s, fb_s = r[0], r[1], r[2], r[3]
        seq = int(seq_s); gbs = int(gbs_s); meas = float(fb_s)
        S = strat_of(strat_s)
        seqs = [seq] * gbs   # all gbs sequences land in the single forced group
        total = cm.total_time(seqs, S)
        resid = cm.residual_time(seqs, S)
        comm = cm.comm_time(seqs, S) if hasattr(cm, "comm_time") else float("nan")
        err = (total - meas) / meas * 100
        print(f"{strat_s:<12} {seq:>7} {gbs:>4} {meas:>9.0f} {total:>10.0f} {err:>+7.1f}%  "
              f"{comm:>7.0f} {resid:>7.0f}")


if __name__ == "__main__":
    main()
