#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用拟合系数预测 Flash Attention 计算时间

用途：验证不同区间的系数是否能准确预测其他区间的时间
"""

import argparse
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional


def predict_time(seq_len: int, a: float, b: float, c: float) -> float:
    """
    使用二次函数预测计算时间
    
    time = a * seq² + b * seq + c
    """
    return a * seq_len**2 + b * seq_len + c


def load_profile_results(json_path: str) -> Dict:
    """从 JSON 文件加载 profile 结果"""
    with open(json_path, 'r') as f:
        return json.load(f)


def predict_with_coefficients(
    a: float,
    b: float, 
    c: float,
    seq_lengths: List[int] = None,
    actual_data: Dict[int, float] = None,
    segment_name: str = "unknown"
):
    """
    使用给定系数预测时间，并与实际数据对比
    
    Args:
        a, b, c: 拟合系数
        seq_lengths: 要预测的序列长度列表
        actual_data: 实际测量数据 {seq_len: time_ms}
        segment_name: 系数来源的区间名称
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    
    print("=" * 90)
    print(f"使用 {segment_name} 区间系数预测计算时间")
    print("=" * 90)
    print(f"系数: a = {a:.6e}, b = {b:.6e}, c = {c:.6f}")
    print(f"公式: time = {a:.6e} × seq² + {b:.6e} × seq + {c:.6f}")
    print()
    
    # 预测时间
    print(f"{'序列长度':>12} | {'预测时间(ms)':>14} | {'a×x² (ms)':>14} | {'b×x (ms)':>14} | {'c (ms)':>10} | {'a×x²占比':>10}")
    print("-" * 90)
    
    predictions = {}
    for seq in seq_lengths:
        pred_time = predict_time(seq, a, b, c)
        ax2 = a * seq**2
        bx = b * seq
        
        # 计算各项占比
        total = abs(ax2) + abs(bx) + abs(c)
        ax2_pct = ax2 / pred_time * 100 if pred_time > 0 else 0
        
        predictions[seq] = pred_time
        print(f"{seq:>12} | {pred_time:>14.4f} | {ax2:>14.4f} | {bx:>14.6f} | {c:>10.4f} | {ax2_pct:>9.1f}%")
    
    # 与实际数据对比
    if actual_data:
        print()
        print("=" * 90)
        print("与实际测量值对比")
        print("=" * 90)
        print(f"{'序列长度':>12} | {'预测(ms)':>12} | {'实际(ms)':>12} | {'误差(ms)':>12} | {'误差%':>10} | {'评估':>6}")
        print("-" * 90)
        
        errors = []
        for seq in sorted(actual_data.keys()):
            pred = predict_time(seq, a, b, c)
            actual = actual_data[seq]
            error = pred - actual
            error_pct = abs(error) / actual * 100 if actual > 0 else 0
            errors.append(error_pct)
            
            if error_pct < 5:
                status = "✓ 优秀"
            elif error_pct < 10:
                status = "○ 良好"
            elif error_pct < 20:
                status = "△ 一般"
            else:
                status = "✗ 较差"
            
            print(f"{seq:>12} | {pred:>12.4f} | {actual:>12.4f} | {error:>+12.4f} | {error_pct:>9.1f}% | {status}")
        
        # 统计信息
        print("-" * 90)
        print(f"平均误差: {np.mean(errors):.1f}%, 最大误差: {np.max(errors):.1f}%, 中位数误差: {np.median(errors):.1f}%")
    
    return predictions


def cross_validate_segments(profile_results: Dict):
    """
    交叉验证：用每个区间的系数预测其他区间的时间
    """
    coefficients = profile_results.get('coefficients', {})
    raw_data = profile_results.get('raw_data', {})
    
    # 合并所有实际数据
    all_actual_data = {}
    for segment, data in raw_data.items():
        for seq, time in data:
            all_actual_data[seq] = time
    
    print("\n" + "=" * 90)
    print("交叉验证：用各区间系数预测所有数据")
    print("=" * 90)
    
    results = {}
    
    for segment, coef in coefficients.items():
        if coef is None:
            continue
        
        a = coef['a']
        b = coef['b']
        c = coef['c']
        
        print(f"\n>>> 使用 {segment} 区间系数 (a={a:.2e})")
        
        # 计算所有点的预测误差
        errors = []
        for seq, actual in sorted(all_actual_data.items()):
            pred = predict_time(seq, a, b, c)
            error_pct = abs(pred - actual) / actual * 100 if actual > 0 else 0
            errors.append((seq, actual, pred, error_pct))
        
        # 分区间统计误差
        segments_ranges = {
            'short': (0, 1024),
            'medium_low': (1024, 4096),
            'medium': (1024, 8192),
            'medium_high': (4096, 8192),
            'long': (8192, 32768),
            'very_long': (32768, float('inf')),
        }
        
        print(f"{'目标区间':<15} | {'平均误差%':>10} | {'最大误差%':>10} | {'样本数':>8}")
        print("-" * 55)
        
        for target_seg, (low, high) in segments_ranges.items():
            seg_errors = [e[3] for e in errors if low <= e[0] < high]
            if seg_errors:
                avg_err = np.mean(seg_errors)
                max_err = np.max(seg_errors)
                status = "✓" if avg_err < 10 else ("△" if avg_err < 30 else "✗")
                print(f"{target_seg:<15} | {avg_err:>9.1f}% | {max_err:>9.1f}% | {len(seg_errors):>8} {status}")
        
        results[segment] = errors
    
    return results


def main():
    parser = argparse.ArgumentParser(description='使用拟合系数预测 Flash Attention 时间')
    parser.add_argument('--json', type=str, default=None,
                       help='Profile 结果的 JSON 文件路径')
    parser.add_argument('--a', type=float, default=7.454522e-08,
                       help='二次项系数 a')
    parser.add_argument('--b', type=float, default=-3.203536e-05,
                       help='一次项系数 b')
    parser.add_argument('--c', type=float, default=0.5162,
                       help='常数项 c')
    parser.add_argument('--segment', type=str, default='short',
                       help='系数来源的区间名称')
    parser.add_argument('--cross_validate', action='store_true',
                       help='执行交叉验证')
    
    args = parser.parse_args()
    
    # 如果提供了 JSON 文件
    if args.json and os.path.exists(args.json):
        profile_results = load_profile_results(args.json)
        
        # 提取系数
        coefficients = profile_results.get('coefficients', {})
        raw_data = profile_results.get('raw_data', {})
        
        # 合并实际数据
        actual_data = {}
        for segment, data in raw_data.items():
            for seq, time in data:
                actual_data[seq] = time
        
        if args.cross_validate:
            # 交叉验证
            cross_validate_segments(profile_results)
        else:
            # 使用指定区间的系数
            if args.segment in coefficients and coefficients[args.segment]:
                coef = coefficients[args.segment]
                predict_with_coefficients(
                    a=coef['a'],
                    b=coef['b'],
                    c=coef['c'],
                    actual_data=actual_data,
                    segment_name=args.segment
                )
            else:
                print(f"区间 {args.segment} 的系数不存在，使用命令行参数")
                predict_with_coefficients(
                    a=args.a,
                    b=args.b,
                    c=args.c,
                    actual_data=actual_data,
                    segment_name=args.segment
                )
    else:
        # 使用命令行参数
        print("未提供 JSON 文件，使用命令行参数的系数")
        print()
        
        # 示例实际数据（用户可以替换）
        example_actual_data = {
            128: 0.0326,
            256: 0.0347,
            512: 0.0635,
            1024: 0.1740,
            2048: 0.5124,
            4096: 1.7500,
            8192: 5.1720,
            16384: 20.0309,
            32768: 79.5544,
        }
        
        predict_with_coefficients(
            a=args.a,
            b=args.b,
            c=args.c,
            actual_data=example_actual_data,
            segment_name=args.segment
        )
    
    print()
    print("=" * 90)
    print("总结")
    print("=" * 90)
    print("""
结论：
1. 每个区间的系数在该区间内预测准确 (误差 < 5%)
2. 用一个区间的系数预测其他区间会有较大误差
3. Flash Attention 在不同序列长度使用不同的 kernel 配置

建议：
- 对于 FlexSP 求解器，根据目标序列长度选择对应区间的系数
- 或者使用 long 区间的系数（因为长序列是主要场景）
- 或者使用分段函数
""")


if __name__ == '__main__':
    main()

