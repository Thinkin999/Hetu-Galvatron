"""Compare old vs new Ring cost-model prediction against alignment report numbers.

No GPU. Loads the same attention + comm profile JSONs that 04_align_costmodel.py
uses, builds the cost model twice (old / new), and prints predicted vs measured
for ring_p16 at seq=8192/16384/32768.

Measured numbers come from ADACPSP_COSTMODEL_ALIGNMENT_REPORT.md (lines around
'ring_p16_seq8192' etc.).
"""

import glob
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
for p in (REPO_ROOT, MODEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from adacpsp_solver import AdaCPSPCostModel, ParallelStrategy  # noqa: E402


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def latest_matching(pattern, predicate=None):
    for p in sorted(glob.glob(pattern), reverse=True):
        try:
            d = load_json(p)
        except Exception:
            continue
        if predicate is None or predicate(d):
            return p
    return ""


def build_cm(overhead_ms: float) -> AdaCPSPCostModel:
    configs_dir = os.path.join(MODEL_DIR, "configs")
    attn = latest_matching(
        os.path.join(configs_dir, "profile_validate_*.json"),
        predicate=lambda d: "attention" in d and "segments" in d.get("attention", {}),
    )
    # Prefer comm_profile_v2 (primitive-only) -- this is what the alignment
    # report numbers were produced with. Fall back to v1 if v2 absent.
    comm = latest_matching(
        os.path.join(configs_dir, "comm_profile_v2_*.json"),
        predicate=lambda d: d.get("type") == "comm_profile_v2",
    )
    if not comm:
        comm = latest_matching(
            os.path.join(configs_dir, "comm_profile_*.json"),
            predicate=lambda d: "alltoall" in d and "p2p_ring" in d,
        )
    valid = latest_matching(
        os.path.join(configs_dir, "profile_validate_*.json"),
        predicate=lambda d: "comm_validation" in d,
    )
    print(f"  using attn   = {os.path.basename(attn)}")
    print(f"  using comm   = {os.path.basename(comm)}")
    print(f"  using valid  = {os.path.basename(valid) if valid else '<none>'}")
    if not attn:
        raise SystemExit("no attention json found")
    if not comm:
        raise SystemExit("no comm json found")

    use_validation = os.environ.get("USE_VALIDATION", "0") not in ("", "0", "false")
    valid_to_use = valid if (valid and use_validation) else None
    print(f"  validation_json applied = {bool(valid_to_use)} (set USE_VALIDATION=1 to enable)")
    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attn,
        comm_profile_json=comm,
        cluster_size=16,
        gpus_per_node=8,
        validation_json=valid_to_use,
    )
    cm.ring_step_overhead_ms = overhead_ms
    return cm


def predict_ring_p16(cm: AdaCPSPCostModel, seq_len: int) -> float:
    strategy = ParallelStrategy(
        attn_type="ring", parallel_size=16, sp_size=1, cp_size=16,
        placement="context_first",
    )
    # 03_run_real_strategies uses --num-seqs GLOBAL_BATCH_SIZE (default 16),
    # which is num_seqs per attention call. Per the alignment report data
    # these were produced with GLOBAL_BATCH_SIZE=16.
    seqlens = [seq_len] * 16
    total = cm.total_time(seqlens, strategy)
    return total / cm.l


# Fresh measured values from 12_quick_ring_align.py (this commit's GPU run).
# Numbers match ADACPSP_COSTMODEL_ALIGNMENT_REPORT.md within 2%, confirming
# the GPU side is stable; any cost-model drift is in the predictor.
MEASURED = {
    8192:  69.03,
    16384: 119.49,
    32768: 259.76,
}


def main() -> None:
    cm_old = build_cm(0.0)
    cm_add = build_cm(0.5)  # current default after kv_pair comm fix

    # Probe whether the new max-style helper is in use. _ring_step_with_overhead
    # is only present if we patched the solver.
    using_max_style = hasattr(cm_add, "_ring_step_with_overhead")
    print(f"using_max_style = {using_max_style}")
    print(f"cm.l = {cm_old.l}, bwd_fwd_ratio = {cm_old.bwd_fwd_ratio}, "
          f"overlap_slowdown = {cm_old.overlap_slowdown}, "
          f"ring_step_overhead_ms = {cm_old.ring_step_overhead_ms}")

    # debug breakdown for all three seq lengths
    strategy = ParallelStrategy(
        attn_type="ring", parallel_size=16, sp_size=1, cp_size=16,
        placement="context_first",
    )
    print()
    print("Cost-model breakdown (num_seqs=16):")
    print(f"{'seq':>6} {'step_c':>9} {'bwd_step':>9} {'fwd_comm':>9} {'bwd_comm':>9} "
          f"{'ovlp_f':>9} {'ovlp_b':>9}  {'15*ovlp_f+15*ovlp_b':>9}")
    for seq in (8192, 16384, 32768):
        seqlens = [seq] * 16
        sc = cm_old._ring_step_compute_per_layer(seqlens, strategy)
        ring_topo = cm_old._get_topo(strategy.placement, "ring", strategy.sp_size, strategy.cp_size)
        fc = cm_old._p2p_fwd_comm_per_step(sum(seqlens), 16, topo=ring_topo)
        bc = cm_old._p2p_bwd_comm_per_step(sum(seqlens), 16, topo=ring_topo)
        bs = sc * cm_old.bwd_fwd_ratio
        of = cm_old._overlap_time(sc, fc)
        ob = cm_old._overlap_time(bs, bc)
        ring_sum = 15 * of + 15 * ob + sc + bs
        print(f"{seq:>6} {sc:>9.3f} {bs:>9.3f} {fc:>9.3f} {bc:>9.3f} "
              f"{of:>9.3f} {ob:>9.3f}  {ring_sum:>9.3f}")

    # ---- counterfactual: P2P fit from trace data ----
    # cost model's v2 P2P lookup at cp=16 underestimates by ~2x (verified by
    # comparing against trace nccl kernel time). Fit a latency+bandwidth model
    # from two trace points (cp=16):
    #   chunk=512  (K payload=0.5 MB):  nccl/step = 94 us
    #   chunk=2048 (K payload=2.0 MB):  nccl/step = 223 us
    # => alpha = 51 us, beta = 86 us/MB  (linear fit time = alpha + beta*payload_MB)
    #
    # For num_seqs=N the per-step K payload scales by N (one large concat'd
    # send instead of N separate sends). Backward nccl per step is ~1.5x fwd
    # (extra dKV grad transfer).
    P2P_ALPHA_US = 51.0
    P2P_BETA_US_PER_MB = 86.0
    BWD_FWD_NCCL_RATIO = 1.6  # trace: 143/94 = 1.52, 408/223 = 1.83, avg ~1.6

    def fit_fwd_nccl_ms(chunk_per_rank: int, num_seqs: int) -> float:
        # K payload one send/recv = chunk * kv_hidden * 2 bytes
        kv_hidden = 4 * 128
        payload_mb = chunk_per_rank * num_seqs * kv_hidden * 2 / 1024 / 1024
        return (P2P_ALPHA_US + P2P_BETA_US_PER_MB * payload_mb) / 1000.0

    NS = 16
    print()
    print("Counterfactual: P2P from latency+bw fit (num_seqs=16):")
    header2 = (f"{'seq':>6} {'chunk':>6} {'fc_v2':>8} {'fc_fit':>8} {'bc_fit':>8} "
               f"{'pred_v2':>9} {'pred_fit':>9} {'measured':>10} "
               f"{'err_v2':>8} {'err_fit':>8}")
    print(header2)
    print("-" * len(header2))
    for seq in (8192, 16384, 32768):
        chunk = seq // 16  # per-rank chunk size (1 sequence)
        seqlens = [seq] * NS
        sc = cm_old._ring_step_compute_per_layer(seqlens, strategy)
        ring_topo = cm_old._get_topo(strategy.placement, "ring", strategy.sp_size, strategy.cp_size)
        fc_v2 = cm_old._p2p_fwd_comm_per_step(sum(seqlens), 16, topo=ring_topo)
        bc_v2 = cm_old._p2p_bwd_comm_per_step(sum(seqlens), 16, topo=ring_topo)
        fc_fit = fit_fwd_nccl_ms(chunk, NS)
        bc_fit = fc_fit * BWD_FWD_NCCL_RATIO
        bs = sc * cm_old.bwd_fwd_ratio

        # Predicted with v2 lookup (no overhead)
        of_v2 = abs(sc - fc_v2) + 1.1 * min(sc, fc_v2)
        ob_v2 = abs(bs - bc_v2) + 1.1 * min(bs, bc_v2)
        pred_v2 = 15 * of_v2 + sc + 15 * ob_v2 + bs

        # Predicted with fit (no overhead)
        of_fit = abs(sc - fc_fit) + 1.1 * min(sc, fc_fit)
        ob_fit = abs(bs - bc_fit) + 1.1 * min(bs, bc_fit)
        pred_fit = 15 * of_fit + sc + 15 * ob_fit + bs

        m = MEASURED[seq]
        err_v2 = (pred_v2 - m) / m * 100
        err_fit = (pred_fit - m) / m * 100
        print(f"{seq:>6} {chunk:>6} {fc_v2:>8.3f} {fc_fit:>8.3f} {bc_fit:>8.3f} "
              f"{pred_v2:>9.2f} {pred_fit:>9.2f} {m:>10.2f} "
              f"{err_v2:>+7.2f}% {err_fit:>+7.2f}%")
    print()

    # Sweep a few overhead candidates to inform the pick.
    print()
    candidates = [0.0, 0.4, 0.5, 0.55, 0.6, 0.65]
    cms = {oh: build_cm(oh) for oh in candidates}
    header = f"{'seq':>6} {'measured':>10}"
    for oh in candidates:
        header += f"  {f'oh={oh:.2f}':>10}"
    print(header)
    print("-" * len(header))
    for seq in sorted(MEASURED.keys()):
        m = MEASURED[seq]
        row = f"{seq:>6} {m:>10.2f}"
        for oh in candidates:
            pred = predict_ring_p16(cms[oh], seq)
            err = (pred - m) / m * 100.0
            row += f"  {err:>+9.2f}%"
        print(row)


if __name__ == "__main__":
    main()
