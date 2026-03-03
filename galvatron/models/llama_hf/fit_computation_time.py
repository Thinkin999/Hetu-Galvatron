#!/usr/bin/env python3
"""
使用 scipy.optimize.curve_fit 拟合计算时间与序列长度的关系
拟合模型: time = a * x^2 + b * x + c
其中 x 为序列长度 (seq)
"""

import json
import re
import numpy as np
from scipy.optimize import curve_fit
import argparse


def quadratic_model(x, a, b, c):
    """二次函数模型: time = a * x^2 + b * x + c"""
    return a * x**2 + b * x + c


def extract_layertype0_data(json_file):
    """
    从JSON文件中提取 layertype_0 的数据
    返回: (seq_lengths, times) 两个numpy数组
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    seq_lengths = []
    times = []
    
    # 匹配 layertype_0_bsz*_seq* 格式的key
    pattern = re.compile(r'^layertype_0_bsz\d+_seq(\d+)$')
    
    for key, value in data.items():
        match = pattern.match(key)
        if match:
            seq_len = int(match.group(1))
            seq_lengths.append(seq_len)
            times.append(value)
            print(f"  找到: {key} -> seq={seq_len}, time={value:.4f} ms")
    
    # 按序列长度排序
    sorted_indices = np.argsort(seq_lengths)
    seq_lengths = np.array(seq_lengths)[sorted_indices]
    times = np.array(times)[sorted_indices]
    
    return seq_lengths, times


def fit_and_report(seq_lengths, times):
    """
    使用curve_fit进行拟合，并输出结果
    """
    # 进行拟合
    popt, pcov = curve_fit(quadratic_model, seq_lengths, times)
    a, b, c = popt
    
    # 计算拟合优度 R^2
    times_pred = quadratic_model(seq_lengths, a, b, c)
    ss_res = np.sum((times - times_pred) ** 2)
    ss_tot = np.sum((times - np.mean(times)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("拟合结果")
    print("=" * 60)
    print(f"拟合模型: time = a * x^2 + b * x + c")
    print(f"其中 x 为序列长度 (seq)")
    print("-" * 60)
    print(f"  a = {a:.15e}")
    print(f"  b = {b:.15e}")
    print(f"  c = {c:.15e}")
    print("-" * 60)
    print(f"拟合优度 R² = {r_squared:.8f}")
    print("=" * 60)
    
    # 输出拟合值对比
    print("\n序列长度 vs 实际值 vs 预测值:")
    print("-" * 50)
    print(f"{'seq':<12} {'实际值(ms)':<18} {'预测值(ms)':<18} {'误差(%)':<10}")
    print("-" * 50)
    for seq, actual, pred in zip(seq_lengths, times, times_pred):
        error_pct = abs(pred - actual) / actual * 100 if actual != 0 else 0
        print(f"{seq:<12} {actual:<18.4f} {pred:<18.4f} {error_pct:<10.2f}")
    print("-" * 50)
    
    return a, b, c, r_squared


def main():
    parser = argparse.ArgumentParser(description='拟合计算时间与序列长度的关系')
    parser.add_argument('--json_file', type=str, 
                        default='/home/pkuhetu/lqs/galvatron_lxy/Hetu-Galvatron/galvatron/models/llama_hf/configs/computation_profiling_bf16_qwen2.5-72b_attention.json',
                        help='输入的JSON文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出拟合参数的JSON文件路径 (可选)')
    args = parser.parse_args()
    
    print(f"读取文件: {args.json_file}")
    print("-" * 60)
    
    # 提取数据
    seq_lengths, times = extract_layertype0_data(args.json_file)
    
    if len(seq_lengths) == 0:
        print("错误: 未找到 layertype_0 的数据!")
        return
    
    print(f"\n共找到 {len(seq_lengths)} 个数据点")
    
    # 拟合并输出结果
    a, b, c, r_squared = fit_and_report(seq_lengths, times)
    
    # 可选：保存拟合结果
    if args.output:
        result = {
            "model": "time = a * x^2 + b * x + c",
            "coefficients": {
                "a": a,
                "b": b,
                "c": c
            },
            "r_squared": r_squared,
            "data_points": len(seq_lengths)
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\n拟合结果已保存至: {args.output}")


if __name__ == "__main__":
    main()

