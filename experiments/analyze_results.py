#!/usr/bin/env python3
"""
实验结果分析脚本
================

用法:
  python analyze_results.py results/20260317_120000/

功能:
  1. 读取 summary.csv，生成对比表格
  2. 计算 AdaCPSP vs FlexSP 的加速比
  3. 按模型 / 序列长度 / 卡数 分组统计
  4. 从各实验 log 提取详细 metrics
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict


def parse_log_metrics(log_path):
    """从实验 log 中提取关键指标"""
    metrics = {
        "avg_iter_ms": None,
        "throughput_tokens_per_s": None,
        "peak_memory_gb": None,
        "mfu": None,
        "solver_time_ms": None,
        "strategy_choices": [],
    }
    
    if not os.path.exists(log_path):
        return metrics
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # 提取平均 iter 时间
        m = re.findall(r'[Aa]vg\s*(?:iter\s*)?time[:\s]*(\d+\.?\d*)\s*ms', content)
        if m:
            metrics["avg_iter_ms"] = float(m[-1])
        
        # 提取吞吐量
        m = re.findall(r'[Tt]hroughput[:\s]*(\d+\.?\d*)', content)
        if m:
            metrics["throughput_tokens_per_s"] = float(m[-1])
        
        # 提取峰值显存
        m = re.findall(r'[Pp]eak\s*(?:memory|mem)[:\s]*(\d+\.?\d*)\s*(?:GB|gb)', content)
        if m:
            metrics["peak_memory_gb"] = float(m[-1])
        
        # 提取 MFU
        m = re.findall(r'MFU[:\s]*(\d+\.?\d*)%?', content)
        if m:
            metrics["mfu"] = float(m[-1])
        
        # 提取 solver 选择的策略
        m = re.findall(r'\[AdaCPSP\].*?strategy.*?:\s*(.+)', content)
        metrics["strategy_choices"] = m[:5]  # 取前 5 条
        
        # 如果没有提取到 avg_iter，尝试从 profile 输出获取
        if metrics["avg_iter_ms"] is None:
            # 尝试 e2e time 格式
            m = re.findall(r'e2e[_\s]time[:\s]*(\d+\.?\d*)', content)
            if m:
                metrics["avg_iter_ms"] = float(m[-1])
        
        # 从 runtime profiler 格式中提取
        if metrics["avg_iter_ms"] is None:
            # RuntimeProfiler 输出格式
            m = re.findall(r'Iter\s+\d+.*?time[:\s]*(\d+\.?\d*)\s*ms', content)
            if len(m) >= 3:
                # 跳过前几个 warmup iter
                iter_times = [float(x) for x in m[3:]]
                if iter_times:
                    metrics["avg_iter_ms"] = sum(iter_times) / len(iter_times)
        
    except Exception as e:
        print(f"  Warning: failed to parse {log_path}: {e}", file=sys.stderr)
    
    return metrics


def load_summary(result_dir):
    """加载 summary.csv"""
    csv_path = os.path.join(result_dir, "summary.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)
    
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_speedup(rows):
    """计算 AdaCPSP vs FlexSP 的加速比"""
    # 按 (model, ngpus, seqlen_k, gbs) 分组
    groups = defaultdict(dict)
    for row in rows:
        key = (row["model"], row["ngpus"], row["seqlen_k"], row["gbs"])
        groups[key][row["strategy"]] = row
    
    comparisons = []
    for key, strategies in groups.items():
        if "adacpsp" in strategies and "flexsp" in strategies:
            ada = strategies["adacpsp"]
            flex = strategies["flexsp"]
            
            ada_time = float(ada.get("avg_iter_time_ms", 0) or 0)
            flex_time = float(flex.get("avg_iter_time_ms", 0) or 0)
            
            if ada_time > 0 and flex_time > 0:
                speedup = flex_time / ada_time
            else:
                speedup = None
            
            comparisons.append({
                "model": key[0],
                "ngpus": key[1],
                "seqlen_k": key[2],
                "gbs": key[3],
                "ada_status": ada["status"],
                "flex_status": flex["status"],
                "ada_time_ms": ada_time,
                "flex_time_ms": flex_time,
                "speedup": speedup,
            })
    
    return comparisons


def print_summary_table(rows):
    """打印汇总表"""
    print("\n" + "=" * 100)
    print("实验结果汇总")
    print("=" * 100)
    print(f"{'模型':<18} {'卡数':>5} {'SeqLen':>8} {'GBS':>5} {'策略':<12} {'状态':<10} {'耗时(s)':>8} {'Iter(ms)':>10} {'峰值显存':>10}")
    print("-" * 100)
    
    for row in rows:
        model = row["model"]
        ngpus = row["ngpus"]
        seqlen = row["seqlen_k"] + "K"
        gbs = row["gbs"]
        strategy = row["strategy"]
        status = row["status"]
        wall_time = row["wall_time_s"]
        iter_time = row.get("avg_iter_time_ms", "N/A")
        peak_mem = row.get("peak_memory_gb", "N/A")
        
        print(f"{model:<18} {ngpus:>5} {seqlen:>8} {gbs:>5} {strategy:<12} {status:<10} {wall_time:>8} {iter_time:>10} {peak_mem:>10}")


def print_speedup_table(comparisons):
    """打印加速比对比表"""
    if not comparisons:
        print("\n没有可对比的 AdaCPSP vs FlexSP 数据")
        return
    
    print("\n" + "=" * 90)
    print("AdaCPSP vs FlexSP 加速比")
    print("=" * 90)
    print(f"{'模型':<18} {'卡数':>5} {'SeqLen':>8} {'GBS':>5} {'FlexSP(ms)':>12} {'AdaCPSP(ms)':>12} {'加速比':>8} {'提升%':>8}")
    print("-" * 90)
    
    total_speedup = []
    for c in comparisons:
        model = c["model"]
        ngpus = c["ngpus"]
        seqlen = c["seqlen_k"] + "K"
        gbs = c["gbs"]
        flex_t = c["flex_time_ms"]
        ada_t = c["ada_time_ms"]
        
        if c["speedup"] is not None:
            speedup_str = f"{c['speedup']:.3f}x"
            improvement = (c["speedup"] - 1) * 100
            improve_str = f"{improvement:+.1f}%"
            total_speedup.append(c["speedup"])
        else:
            speedup_str = "N/A"
            improve_str = "N/A"
        
        flex_str = f"{flex_t:.1f}" if flex_t > 0 else c["flex_status"]
        ada_str = f"{ada_t:.1f}" if ada_t > 0 else c["ada_status"]
        
        print(f"{model:<18} {ngpus:>5} {seqlen:>8} {gbs:>5} {flex_str:>12} {ada_str:>12} {speedup_str:>8} {improve_str:>8}")
    
    if total_speedup:
        avg_speedup = sum(total_speedup) / len(total_speedup)
        max_speedup = max(total_speedup)
        min_speedup = min(total_speedup)
        print("-" * 90)
        print(f"{'平均':>56} {avg_speedup:.3f}x  {(avg_speedup-1)*100:+.1f}%")
        print(f"{'最大':>56} {max_speedup:.3f}x  {(max_speedup-1)*100:+.1f}%")
        print(f"{'最小':>56} {min_speedup:.3f}x  {(min_speedup-1)*100:+.1f}%")


def print_per_model_summary(rows):
    """按模型分组统计"""
    model_stats = defaultdict(lambda: {"pass": 0, "fail": 0, "oom": 0, "timeout": 0, "total": 0})
    
    for row in rows:
        model = row["model"]
        status = row["status"].upper()
        model_stats[model]["total"] += 1
        if status == "PASS":
            model_stats[model]["pass"] += 1
        elif "OOM" in status:
            model_stats[model]["oom"] += 1
        elif "TIMEOUT" in status:
            model_stats[model]["timeout"] += 1
        else:
            model_stats[model]["fail"] += 1
    
    print("\n" + "=" * 60)
    print("按模型统计")
    print("=" * 60)
    print(f"{'模型':<18} {'总计':>6} {'通过':>6} {'OOM':>6} {'超时':>6} {'失败':>6}")
    print("-" * 60)
    for model, stats in sorted(model_stats.items()):
        print(f"{model:<18} {stats['total']:>6} {stats['pass']:>6} {stats['oom']:>6} {stats['timeout']:>6} {stats['fail']:>6}")


def main():
    parser = argparse.ArgumentParser(description="分析 AdaCPSP 实验结果")
    parser.add_argument("result_dir", help="结果目录路径")
    parser.add_argument("--detailed", action="store_true", help="显示每个实验的详细 log 分析")
    args = parser.parse_args()
    
    if not os.path.isdir(args.result_dir):
        print(f"ERROR: {args.result_dir} is not a directory")
        sys.exit(1)
    
    rows = load_summary(args.result_dir)
    print(f"加载了 {len(rows)} 条实验记录")
    
    # 如果需要，从 log 中补充指标
    if args.detailed:
        for row in rows:
            log_path = row.get("log_file", "")
            if log_path and os.path.exists(log_path):
                metrics = parse_log_metrics(log_path)
                if metrics["avg_iter_ms"] and not row.get("avg_iter_time_ms"):
                    row["avg_iter_time_ms"] = str(metrics["avg_iter_ms"])
                if metrics["peak_memory_gb"] and not row.get("peak_memory_gb"):
                    row["peak_memory_gb"] = str(metrics["peak_memory_gb"])
    
    # 打印各种汇总
    print_summary_table(rows)
    print_per_model_summary(rows)
    
    comparisons = compute_speedup(rows)
    print_speedup_table(comparisons)
    
    print(f"\n结果文件: {os.path.join(args.result_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()

