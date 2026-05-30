#!/usr/bin/env python3
"""Render ILP simulation results as horizontal Markdown comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


CASE_ORDER = [
    "ulysses_only",
    "ring_only",
    "ulysses_ring",
    "usp_only",
    "ulysses_ring_usp",
]


def load_records(result_dir: Path) -> List[Dict]:
    path = result_dir / "results.jsonl"
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def fmt_num(value, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_seq_pairs(ids: List[int], lengths: List[int], max_items: int = 16) -> str:
    pairs = [f"{i}:{l}" for i, l in zip(ids, lengths)]
    if len(pairs) > max_items:
        pairs = pairs[:max_items] + [f"...(+{len(ids) - max_items})"]
    return "[" + ", ".join(pairs) + "]"


def md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def time_breakdown_line(group: Dict) -> str:
    tb = group.get("time_breakdown", {})
    strategy = group.get("attn_type")
    if strategy == "ulysses":
        return (
            f"compute={fmt_num(tb.get('compute_fwd_ms'))}+{fmt_num(tb.get('compute_bwd_ms'))}, "
            f"a2a={fmt_num(tb.get('alltoall_comm_ms'))}, "
            f"topo={tb.get('topology', {}).get('alltoall')}, "
            f"q={tb.get('head_padding', {}).get('q_factor')}"
        )
    if strategy == "ring":
        return (
            f"step_compute={fmt_num(tb.get('ring_step_compute_per_layer_ms'))}, "
            f"fwd_step={fmt_num(tb.get('ring_fwd_comm_step_ms'))}, "
            f"bwd_step={fmt_num(tb.get('ring_bwd_comm_step_ms'))}, "
            f"topo={tb.get('topology', {}).get('ring')}"
        )
    if strategy == "usp":
        return (
            f"compute_step={fmt_num(tb.get('usp_compute_step_per_layer_ms'))}, "
            f"a2a/layer={fmt_num(tb.get('a2a_fwd_per_layer_ms'))}+{fmt_num(tb.get('a2a_bwd_per_layer_ms'))}, "
            f"ring/layer={fmt_num(tb.get('ring_fwd_per_layer_ms'))}+{fmt_num(tb.get('ring_bwd_per_layer_ms'))}, "
            f"topo={tb.get('topology', {}).get('alltoall')}/{tb.get('topology', {}).get('ring')}"
        )
    return ""


def group_cell(group: Dict | None) -> str:
    if not group:
        return ""
    lines = [
        f"g{group['group_idx']}: {group['parallel_size']}GPU {group['strategy']}",
        f"time={fmt_num(group['time_ms'])}ms, mem={fmt_num(group['memory_mb'] / 1024)}GB",
        f"seq_sum={group['tokens']}, local={fmt_num(group['local_tokens'])}",
        f"seq={fmt_seq_pairs(group['seq_ids'], group['seqlens'])}",
    ]
    tb = time_breakdown_line(group)
    if tb:
        lines.append(f"tb: {tb}")
    return md_escape("<br>".join(lines))


def group_metric(group: Dict | None, metric: str) -> str:
    if not group:
        return ""
    tb = group.get("time_breakdown", {})
    mem = group.get("memory_breakdown_mb", {})
    if metric == "strategy":
        return str(group.get("strategy", ""))
    if metric == "num_gpus":
        return str(group.get("parallel_size", ""))
    if metric == "sp_cp_placement":
        return f"sp={group.get('sp_size')}, cp={group.get('cp_size')}, placement={group.get('placement')}"
    if metric == "estimated_time_ms":
        return fmt_num(group.get("time_ms"))
    if metric == "estimated_memory_gb":
        return fmt_num(group.get("memory_mb", 0) / 1024)
    if metric == "memory_breakdown_gb":
        return (
            f"model={fmt_num(mem.get('model_states', 0) / 1024)}, "
            f"act={fmt_num(mem.get('activation', 0) / 1024)}, "
            f"pad={fmt_num(mem.get('activation_head_padding_extra', 0) / 1024)}"
        )
    if metric == "seq_sum_local":
        return f"seq_sum={group.get('tokens')}, local={fmt_num(group.get('local_tokens'))}"
    if metric == "sequence_ids":
        return yaml_like_inline(group.get("seq_ids", []), max_items=24)
    if metric == "sequence_lengths":
        return yaml_like_inline(group.get("seqlens", []), max_items=24)
    if metric == "topology":
        topo = tb.get("topology", {})
        return f"a2a={topo.get('alltoall')}, ring={topo.get('ring')}"
    if metric == "head_padding":
        hp = tb.get("head_padding", {})
        return f"q={hp.get('q_factor')}, kv={hp.get('kv_factor')}"
    if metric == "compute":
        if group.get("attn_type") == "ulysses":
            return f"fwd={fmt_num(tb.get('compute_fwd_ms'))}, bwd={fmt_num(tb.get('compute_bwd_ms'))}"
        if group.get("attn_type") == "ring":
            return (
                f"step/layer={fmt_num(tb.get('ring_step_compute_per_layer_ms'))}, "
                f"fwd/layer={fmt_num(tb.get('ring_fwd_per_layer_ms'))}, "
                f"bwd/layer={fmt_num(tb.get('ring_bwd_per_layer_ms'))}"
            )
        if group.get("attn_type") == "usp":
            return f"step/layer={fmt_num(tb.get('usp_compute_step_per_layer_ms'))}"
    if metric == "alltoall_comm":
        if group.get("attn_type") == "usp":
            return (
                f"qo_op={fmt_num(tb.get('a2a_qo_per_op_ms'))}, "
                f"kv_op={fmt_num(tb.get('a2a_kv_per_op_ms'))}, "
                f"fwd/layer={fmt_num(tb.get('a2a_fwd_per_layer_ms'))}, "
                f"bwd/layer={fmt_num(tb.get('a2a_bwd_per_layer_ms'))}"
            )
        return fmt_num(tb.get("alltoall_comm_ms"))
    if metric == "ring_comm":
        if group.get("attn_type") in ("ring", "usp"):
            return (
                f"fwd_step={fmt_num(tb.get('ring_fwd_comm_step_ms'))}, "
                f"bwd_step={fmt_num(tb.get('ring_bwd_comm_step_ms'))}, "
                f"fwd/layer={fmt_num(tb.get('ring_fwd_per_layer_ms'))}, "
                f"bwd/layer={fmt_num(tb.get('ring_bwd_per_layer_ms'))}"
            )
        return "0.00"
    if metric == "time_model":
        return f"{tb.get('model')}, leakage={tb.get('overlap_leakage')}"
    if metric == "formula":
        return str(tb.get("formula", ""))
    return ""


def yaml_like_inline(values: List, max_items: int = 24) -> str:
    items = list(values)
    if len(items) > max_items:
        shown = items[:max_items]
        return "[" + ", ".join(str(x) for x in shown) + f", ...(+{len(items) - max_items})]"
    return "[" + ", ".join(str(x) for x in items) + "]"


def table(headers: List[str], rows: List[List[str]]) -> List[str]:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(md_escape(cell) for cell in row) + " |")
    return out


def render_record(record: Dict) -> str:
    batch = record["batch"]
    cases = record["cases"]
    present_cases = [name for name in CASE_ORDER if name in cases]
    lines: List[str] = []

    lines.append(f"# ILP Horizontal Comparison: `{batch['batch_key']}`")
    lines.append("")
    lines.append("## Batch")
    lines.extend(table(
        ["field", "value"],
        [
            ["sampling_mode", str(batch["sampling_mode"])],
            ["max_seq", str(batch["max_seq"])],
            ["global_batch_size", str(batch["stats"]["num_seqs"])],
            ["seq_sum", str(batch["stats"]["tokens"])],
            ["max / p95 / p99", f"{batch['stats']['max']} / {fmt_num(batch['stats']['p95'])} / {fmt_num(batch['stats']['p99'])}"],
            [">=128k / >=256k / >=384k / >=512k", f"{batch['stats']['ge_128k']} / {batch['stats']['ge_256k']} / {batch['stats']['ge_384k']} / {batch['stats']['ge_512k']}"],
            ["sequences", fmt_seq_pairs(list(range(len(batch["seqlens"]))), batch["seqlens"], max_items=64)],
        ],
    ))
    lines.append("")

    baseline = cases.get("ulysses_only", {}).get("total_time_ms")
    summary_rows = []
    for metric in ["status", "total_time_ms", "speedup_vs_ulysses", "microbatch_count", "solver_wall_s"]:
        row = [metric]
        for name in present_cases:
            case = cases[name]
            if metric == "speedup_vs_ulysses":
                value = baseline / case["total_time_ms"] if baseline and case.get("total_time_ms") else None
                row.append(fmt_num(value, 4))
            elif metric == "microbatch_count":
                row.append(str(case.get("microbatch_count", "")))
            else:
                row.append(fmt_num(case.get(metric), 4) if metric.endswith("_ms") or metric.endswith("_s") else str(case.get(metric, "")))
        summary_rows.append(row)
    lines.append("## Case Summary")
    lines.extend(table(["metric"] + present_cases, summary_rows))
    lines.append("")

    max_mbs = max(cases[name].get("microbatch_count", 0) for name in present_cases)
    for mb_idx in range(max_mbs):
        lines.append(f"## Microbatch {mb_idx}")
        overview = []
        for metric in ["estimated_time_ms", "seq_sum", "num_sequences", "sequences"]:
            row = [metric]
            for name in present_cases:
                mbs = cases[name].get("details", {}).get("microbatches", [])
                if mb_idx >= len(mbs):
                    row.append("")
                    continue
                mb = mbs[mb_idx]
                if metric == "estimated_time_ms":
                    row.append(fmt_num(mb["M_ms"]))
                elif metric == "seq_sum":
                    seq_sum = sum(sum(g["seqlens"]) for g in mb["groups"])
                    row.append(str(seq_sum))
                elif metric == "num_sequences":
                    n = sum(len(g["seqlens"]) for g in mb["groups"])
                    row.append(str(n))
                else:
                    ids = []
                    lens = []
                    for g in mb["groups"]:
                        ids.extend(g["seq_ids"])
                        lens.extend(g["seqlens"])
                    order = sorted(range(len(ids)), key=lambda i: ids[i])
                    row.append(fmt_seq_pairs([ids[i] for i in order], [lens[i] for i in order], max_items=24))
            overview.append(row)
        lines.extend(table(["metric"] + present_cases, overview))
        lines.append("")

        max_groups = 0
        for name in present_cases:
            mbs = cases[name].get("details", {}).get("microbatches", [])
            if mb_idx < len(mbs):
                max_groups = max(max_groups, len(mbs[mb_idx]["groups"]))
        group_rows = []
        metrics = [
            "strategy",
            "num_gpus",
            "sp_cp_placement",
            "estimated_time_ms",
            "estimated_memory_gb",
            "memory_breakdown_gb",
            "seq_sum_local",
            "sequence_ids",
            "sequence_lengths",
            "topology",
            "head_padding",
            "compute",
            "alltoall_comm",
            "ring_comm",
            "time_model",
            "formula",
        ]
        for group_idx in range(max_groups):
            for metric in metrics:
                row = [f"group_{group_idx}.{metric}"]
                for name in present_cases:
                    mbs = cases[name].get("details", {}).get("microbatches", [])
                    if mb_idx >= len(mbs) or group_idx >= len(mbs[mb_idx]["groups"]):
                        row.append("")
                    else:
                        row.append(group_metric(mbs[mb_idx]["groups"][group_idx], metric))
                group_rows.append(row)
        lines.extend(table(["group metric"] + present_cases, group_rows))
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or (args.result_dir / "comparison_md")
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in load_records(args.result_dir):
        batch = record["batch"]
        path = out_dir / f"{batch['batch_key']}.md"
        path.write_text(render_record(record), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
