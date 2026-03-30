#!/usr/bin/env python3
"""
实验日志分析脚本
================
解析 logs/ 下的所有日志文件, 提取关键指标, 生成对比表格

用法:
    python analyze_results.py --log-dir logs/
    python analyze_results.py --log-dir logs/ --output results.md
"""

import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def parse_log(log_path: str) -> dict:
    """解析单个实验日志, 提取关键指标"""
    result = {
        "path": log_path,
        "status": "UNKNOWN",
        "avg_iter_ms": None,
        "throughput_tokens_per_sec": None,
        "peak_memory_gb": None,
        "strategy_distribution": {},
        "iter_times": [],
        "error_msg": None,
        "total_tokens": None,
    }
    
    if not os.path.exists(log_path):
        result["status"] = "NO_LOG"
        return result
    
    with open(log_path, "r", errors="replace") as f:
        content = f.read()
    lines = content.split("\n")
    
    # 检查状态
    if "STATUS: SUCCESS" in content:
        result["status"] = "SUCCESS"
    elif "STATUS: TIMEOUT" in content:
        result["status"] = "TIMEOUT"
    elif "STATUS: FAILED" in content:
        result["status"] = "FAILED"
    elif "CUDA out of memory" in content or "OutOfMemoryError" in content:
        result["status"] = "OOM"
        result["error_msg"] = "CUDA out of memory"
    elif "NCCL" in content and "error" in content.lower():
        result["status"] = "NCCL_ERROR"
    
    if "REASON: OOM" in content:
        result["status"] = "OOM"
    
    # 提取 iteration times
    # 支持多种格式:
    # "Elapsed time per iteration (ms): 1234.5"
    # "Iteration time: 1234.5 ms"
    # "[Profiler] Iteration X time: 1234.5 ms"
    iter_time_patterns = [
        r"Elapsed time per iteration \(ms\):\s*([0-9.]+)",
        r"Iteration time:\s*([0-9.]+)\s*ms",
        r"\[Profiler\].*?time:\s*([0-9.]+)\s*ms",
        r"iter\s+\d+.*?time[:\s]*([0-9.]+)\s*ms",
    ]
    
    for pattern in iter_time_patterns:
        matches = re.findall(pattern, content)
        if matches:
            result["iter_times"] = [float(m) for m in matches]
            break
    
    # 计算 avg (去掉前5个 warmup)
    if result["iter_times"]:
        warmup = min(5, len(result["iter_times"]) // 3)
        stable_times = result["iter_times"][warmup:]
        if stable_times:
            result["avg_iter_ms"] = sum(stable_times) / len(stable_times)
    
    # 提取 throughput
    throughput_patterns = [
        r"[Tt]hroughput:\s*([0-9.]+)\s*tokens/s",
        r"tokens.per.sec:\s*([0-9.]+)",
        r"throughput.*?([0-9]+\.?[0-9]*)\s*(?:tokens|tok)/s",
    ]
    for pattern in throughput_patterns:
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            result["throughput_tokens_per_sec"] = float(m.group(1))
            break
    
    # 提取 peak memory
    mem_patterns = [
        r"[Mm]ax memory:\s*([0-9.]+)\s*GB",
        r"[Pp]eak memory:\s*([0-9.]+)\s*GB",
        r"[Mm]ax allocated:\s*([0-9.]+)\s*GB",
        r"After Backward.*?([0-9.]+)\s*GB",
    ]
    for pattern in mem_patterns:
        matches = re.findall(pattern, content)
        if matches:
            result["peak_memory_gb"] = max(float(m) for m in matches)
            break
    
    # 提取策略分布 (AdaCPSP 输出)
    strategy_counts = defaultdict(int)
    strat_patterns = [
        r"\[AdaCPSP\].*?type=(\w+)",
        r"strategy.*?(\bulysses\b|\bring\b|\busp\b)",
    ]
    for pattern in strat_patterns:
        for m in re.finditer(pattern, content, re.IGNORECASE):
            strategy_counts[m.group(1).lower()] += 1
    if strategy_counts:
        total = sum(strategy_counts.values())
        result["strategy_distribution"] = {
            k: {"count": v, "pct": v / total * 100}
            for k, v in strategy_counts.items()
        }

    # 提取 placement 分布 (placement-aware USP)
    placement_counts = defaultdict(int)
    pl_pattern = r"\[AdaCPSP\].*?placement=(\w+)"
    for m in re.finditer(pl_pattern, content):
        placement_counts[m.group(1)] += 1
    if placement_counts:
        pl_total = sum(placement_counts.values())
        result["placement_distribution"] = {
            k: {"count": v, "pct": v / pl_total * 100}
            for k, v in placement_counts.items()
        }
    else:
        result["placement_distribution"] = {}

    return result


def compute_mfu(
    avg_iter_ms: float,
    model_params_B: float,
    total_tokens_per_iter: int,
    gpu_count: int = 64,
    gpu_peak_tflops: float = 148.0,  # H20 bf16 peak TFLOPS
) -> float:
    """
    计算 MFU (Model FLOPs Utilization)
    
    MFU = actual_flops / peak_flops
    actual_flops ≈ 6 * params * tokens (for transformer training)
    """
    if avg_iter_ms is None or avg_iter_ms <= 0:
        return 0.0
    
    iter_seconds = avg_iter_ms / 1000.0
    actual_flops = 6 * model_params_B * 1e9 * total_tokens_per_iter
    peak_flops = gpu_peak_tflops * 1e12 * gpu_count
    mfu = actual_flops / (peak_flops * iter_seconds) * 100
    return mfu


def format_table(headers: List[str], rows: List[List[str]], title: str = "") -> str:
    """格式化 Markdown 表格"""
    result = ""
    if title:
        result += f"\n### {title}\n\n"
    
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in col_widths) + " |"
    
    result += header_line + "\n" + sep_line + "\n"
    for row in rows:
        cells = [str(c).ljust(w) for c, w in zip(row, col_widths)]
        result += "| " + " | ".join(cells) + " |\n"
    
    return result


def generate_report(log_dir: str, index_path: Optional[str] = None) -> str:
    """生成完整的分析报告"""
    
    # 加载实验索引
    if index_path and os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        experiments = index["experiments"]
    else:
        # 从 log 文件名自动发现
        experiments = []
        for fname in sorted(os.listdir(log_dir)):
            if fname.endswith(".log") and fname != "summary.log":
                name = fname[:-4]
                experiments.append({"name": name, "log": f"logs/{name}.log"})
    
    # 解析所有日志
    results = {}
    for exp in experiments:
        name = exp["name"]
        log_path = os.path.join(log_dir, f"{name}.log")
        results[name] = parse_log(log_path)
        results[name].update(exp)
    
    report = "# AdaCPSP vs FlexSP 实验结果分析\n\n"
    report += f"日志目录: `{log_dir}`\n"
    report += f"实验总数: {len(results)}\n\n"
    
    # ── 总览表 ──
    status_counts = defaultdict(int)
    for r in results.values():
        status_counts[r["status"]] += 1
    
    report += "## 1. 实验状态总览\n\n"
    for status, count in sorted(status_counts.items()):
        emoji = {"SUCCESS": "✅", "FAILED": "❌", "OOM": "💥", "TIMEOUT": "⏰", "UNKNOWN": "❓", "NO_LOG": "📭"}.get(status, "•")
        report += f"- {emoji} {status}: {count}\n"
    report += "\n"
    
    # ── 按 (model, dataset, seq_len) 的策略对比表 ──
    report += "## 2. 策略对比 (核心结果)\n\n"
    
    # 按 model 分组
    for model in sorted(set(r.get("model", "?") for r in results.values())):
        report += f"\n### 模型: {model}\n"
        
        for dataset in sorted(set(r.get("dataset", "?") for r in results.values())):
            headers = ["Seq Len", "FlexSP (ms)", "AdaCPSP-UR (ms)", "AdaCPSP-Full (ms)", "UR Speedup", "Full Speedup", "Best Strategy Dist"]
            rows = []
            
            for seq_len in ["128k", "256k", "384k", "512k"]:
                row = [seq_len]
                times = {}
                strat_dist = {}
                
                for strat in ["flexsp", "adacpsp_ur", "adacpsp_full"]:
                    name = f"{model}_{dataset}_{seq_len}_{strat}"
                    r = results.get(name, {})
                    t = r.get("avg_iter_ms")
                    if t is not None:
                        times[strat] = t
                        row.append(f"{t:.1f}")
                    elif r.get("status") in ["OOM", "TIMEOUT"]:
                        row.append(r.get("status", "N/A"))
                    else:
                        row.append("N/A")
                    
                    if strat in ["adacpsp_ur", "adacpsp_full"] and r.get("strategy_distribution"):
                        strat_dist[strat] = r["strategy_distribution"]
                
                # 计算 speedup
                base = times.get("flexsp")
                for strat in ["adacpsp_ur", "adacpsp_full"]:
                    t = times.get(strat)
                    if base and t and t > 0:
                        speedup = base / t
                        row.append(f"{speedup:.3f}x")
                    else:
                        row.append("N/A")
                
                # 最佳策略分布
                best_strat = "adacpsp_full" if "adacpsp_full" in strat_dist else "adacpsp_ur" if "adacpsp_ur" in strat_dist else None
                if best_strat and strat_dist.get(best_strat):
                    dist_str = ", ".join(f"{k}:{v['pct']:.0f}%" for k, v in strat_dist[best_strat].items())
                    row.append(dist_str)
                else:
                    row.append("N/A")
                
                rows.append(row)
            
            if rows:
                report += format_table(headers, rows, f"数据集: {dataset}")
    
    # ── 详细结果表 ──
    report += "\n## 3. 详细结果\n\n"
    headers = ["Experiment", "Status", "Avg Iter (ms)", "Peak Mem (GB)", "Strategy Dist"]
    rows = []
    for name in sorted(results.keys()):
        r = results[name]
        status = r.get("status", "?")
        avg_ms = f"{r['avg_iter_ms']:.1f}" if r.get("avg_iter_ms") else "N/A"
        peak_mem = f"{r['peak_memory_gb']:.1f}" if r.get("peak_memory_gb") else "N/A"
        dist = ", ".join(f"{k}:{v['pct']:.0f}%" for k, v in r.get("strategy_distribution", {}).items()) or "N/A"
        rows.append([name, status, avg_ms, peak_mem, dist])
    
    report += format_table(headers, rows)
    
    # ── OOM 分析 ──
    oom_exps = [name for name, r in results.items() if r.get("status") == "OOM"]
    if oom_exps:
        report += "\n## 4. OOM 分析\n\n"
        report += "以下实验因显存不足失败:\n\n"
        for name in oom_exps:
            report += f"- `{name}`\n"
        report += "\n建议: 降低 GBS 或增加 selective_checkpoint\n"
    
    # ── Placement 对比 (head_first vs context_first) ──
    has_placement = any("adacpsp_hf" in name or "adacpsp_cf" in name for name in results)
    if has_placement:
        report += "\n## 5. Placement 对比 (Head-First vs Context-First)\n\n"
        for model in sorted(set(r.get("model", "?") for r in results.values())):
            for dataset in sorted(set(r.get("dataset", "?") for r in results.values())):
                headers = ["Seq Len", "Auto (ms)", "Head-First (ms)", "Context-First (ms)",
                           "HF vs CF", "Auto Placement Choice"]
                rows = []
                for seq_len in ["128k", "256k", "384k", "512k"]:
                    row = [seq_len]
                    times = {}
                    auto_pl = {}
                    for strat_key, col_key in [("adacpsp_full", "auto"),
                                               ("adacpsp_hf", "hf"),
                                               ("adacpsp_cf", "cf")]:
                        name = f"{model}_{dataset}_{seq_len}_{strat_key}"
                        r = results.get(name, {})
                        t = r.get("avg_iter_ms")
                        if t is not None:
                            times[col_key] = t
                            row.append(f"{t:.1f}")
                        elif r.get("status") in ["OOM", "TIMEOUT"]:
                            row.append(r.get("status", "N/A"))
                        else:
                            row.append("N/A")
                        if col_key == "auto" and r.get("placement_distribution"):
                            auto_pl = r["placement_distribution"]

                    hf_t = times.get("hf")
                    cf_t = times.get("cf")
                    if hf_t and cf_t and cf_t > 0:
                        ratio = hf_t / cf_t
                        better = "HF" if ratio < 1.0 else "CF"
                        row.append(f"{ratio:.3f}x ({better} wins)")
                    else:
                        row.append("N/A")

                    if auto_pl:
                        pl_str = ", ".join(f"{k}:{v['pct']:.0f}%" for k, v in auto_pl.items())
                        row.append(pl_str)
                    else:
                        row.append("N/A")
                    rows.append(row)

                if any(r[1] != "N/A" or r[2] != "N/A" or r[3] != "N/A" for r in rows):
                    report += format_table(headers, rows, f"{model} / {dataset}")

        # Solver placement accuracy
        report += "\n### Solver Placement 选择准确率\n\n"
        correct = 0
        total_pl = 0
        for seq_len in ["128k", "256k", "384k", "512k"]:
            for model in sorted(set(r.get("model", "?") for r in results.values())):
                for dataset in sorted(set(r.get("dataset", "?") for r in results.values())):
                    hf_name = f"{model}_{dataset}_{seq_len}_adacpsp_hf"
                    cf_name = f"{model}_{dataset}_{seq_len}_adacpsp_cf"
                    auto_name = f"{model}_{dataset}_{seq_len}_adacpsp_full"
                    hf_r = results.get(hf_name, {})
                    cf_r = results.get(cf_name, {})
                    auto_r = results.get(auto_name, {})
                    hf_t = hf_r.get("avg_iter_ms")
                    cf_t = cf_r.get("avg_iter_ms")
                    auto_t = auto_r.get("avg_iter_ms")
                    if hf_t and cf_t and auto_t:
                        actual_best = min(hf_t, cf_t)
                        total_pl += 1
                        if abs(auto_t - actual_best) / actual_best < 0.05:
                            correct += 1
        if total_pl > 0:
            report += f"准确率 (auto 在最佳 forced placement 5% 以内): {correct}/{total_pl} = {correct/total_pl*100:.1f}%\n\n"

    # ── 关键发现 ──
    section_num = 6 if has_placement else 5
    report += f"\n## {section_num}. 关键发现\n\n"
    report += "> 请在查看以上数据后, 补充分析结论\n\n"
    report += "### 需要关注的点:\n"
    report += "1. **Ring Attention 在长序列下是否有加速?** 对比 FlexSP 和 AdaCPSP-UR\n"
    report += "2. **USP 是否进一步提升?** 对比 AdaCPSP-UR 和 AdaCPSP-Full\n"
    report += "3. **GQA 比例的影响**: 7B (1:7) vs 14B/32B (1:5)\n"
    report += "4. **跨机通信的影响**: 当 parallel_size > 8 时, 需要跨机\n"
    report += "5. **策略选择分布**: solver 在长序列下是否倾向选择 Ring/USP\n"
    if has_placement:
        report += "6. **Placement 选择**: head_first vs context_first 对不同模型/序列长度的影响\n"
        report += "7. **Solver 自动选择准确率**: auto placement 是否接近最优 forced placement\n"
    
    return report


def main():
    parser = argparse.ArgumentParser(description="分析 AdaCPSP 实验日志")
    parser.add_argument("--log-dir", type=str, default="logs/", help="日志目录")
    parser.add_argument("--index", type=str, default=None, help="experiment_index.json 路径")
    parser.add_argument("--output", type=str, default=None, help="输出 markdown 文件")
    args = parser.parse_args()
    
    # 自动发现 index
    if args.index is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_index = os.path.join(script_dir, "experiment_index.json")
        if os.path.exists(default_index):
            args.index = default_index
    
    report = generate_report(args.log_dir, args.index)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()

