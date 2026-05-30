"""
Per-iter comparison: did the post-fix solver pick a strategy that's
actually faster (wall-clock) than the forced cells on the SAME global batch?

Since dataloader is deterministic, iter k in cauto and forced cells sees
the same sequences.  This lets us answer: "for this iter, was cauto's pick
better or worse than what ulysses8/usp2x4 would have done?"
"""
import json, sys, math
from pathlib import Path

RUN = Path("/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron/galvatron/models/varlen_llama_hf/align_costmodel/results/cauto_postfix_20260529_202459/end2end")

cells = {}
for name in ["adacpsp_chunksauto", "ulysses8_chunks1", "usp2x4_chunks1"]:
    f = RUN / name / "rank0.jsonl"
    if not f.exists():
        print(f"# {name}: not yet ready")
        continue
    recs = [json.loads(l) for l in open(f) if l.strip()]
    train = [r for r in recs if r.get("phase") == "train_step"]
    cells[name] = train
    print(f"# {name}: {len(train)} train iters")

# Per-iter: meas_fb across cells
print(f"\n{'iter':>4}  {'cauto':>9}  {'uly8c1':>9}  {'usp2x4c1':>9}  {'cauto_strat':<28} {'cauto_vs_uly8':>15}")
SKIP = 5
n = max(len(v) for v in cells.values()) if cells else 0
for i in range(SKIP, n):
    row = {}
    for c, recs in cells.items():
        if i < len(recs):
            row[c] = float(recs[i]["timings_ms"]["forward_backward"])

    cauto = row.get("adacpsp_chunksauto")
    uly8 = row.get("ulysses8_chunks1")
    usp24 = row.get("usp2x4_chunks1")

    # Get cauto's first mb strat
    cauto_strat = ""
    if "adacpsp_chunksauto" in cells and i < len(cells["adacpsp_chunksauto"]):
        op = cells["adacpsp_chunksauto"][i].get("predicted_adacpsp", {})
        mbs = op.get("microbatches", [])
        if mbs:
            gs = mbs[0].get("groups", [])
            if gs:
                g = gs[0]
                cauto_strat = f"mb0:{g['attn_type']}sp{g['sp_size']}cp{g['cp_size']}{g.get('placement','')[:2]} nmb={len(mbs)}"

    vs_uly8 = ""
    if cauto and uly8:
        diff_pct = (cauto - uly8) / uly8 * 100
        vs_uly8 = f"{diff_pct:+.0f}%"

    cauto_s = f"{cauto:.0f}" if cauto else "-"
    uly8_s = f"{uly8:.0f}" if uly8 else "-"
    usp24_s = f"{usp24:.0f}" if usp24 else "-"
    print(f"{i:>4}  {cauto_s:>9}  {uly8_s:>9}  {usp24_s:>9}  {cauto_strat:<28} {vs_uly8:>15}")

# Aggregate
print()
def agg(name):
    if name not in cells: return None
    fbs = [float(r["timings_ms"]["forward_backward"]) for r in cells[name][SKIP:]]
    if not fbs: return None
    sorted_fbs = sorted(fbs); med = sorted_fbs[len(sorted_fbs)//2]
    mad = sorted([abs(x-med) for x in fbs])[len(fbs)//2]
    thr = med + 3*(mad if mad>0 else med*0.3)
    clean = [x for x in fbs if x <= thr]
    return sum(clean)/len(clean), len(clean), len(fbs)

print(f"{'cell':<22}  {'meas_avg':>9}  {'n_clean':>8}")
for name in ["adacpsp_chunksauto", "ulysses8_chunks1", "usp2x4_chunks1"]:
    r = agg(name)
    if r: print(f"{name:<22}  {r[0]:>9.0f}  {r[1]:>3d}/{r[2]:>3d}")

print()
if "adacpsp_chunksauto" in cells and "ulysses8_chunks1" in cells:
    ca = agg("adacpsp_chunksauto")[0]
    ul = agg("ulysses8_chunks1")[0]
    print(f"cauto vs ulysses8_c1:  {ca:.0f} vs {ul:.0f}  → cauto is {(ca-ul)/ul*100:+.1f}% "
          f"(positive = cauto SLOWER, solver pick is sub-optimal)")
