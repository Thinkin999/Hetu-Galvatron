#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flash Attention Profiling Script

分段 profile Flash Attention，拟合 time = a*x² + b*x + c

区间设置：
- short: 128 - 1024, step=128 (8个点)
- medium: 1024 - 8192, step=512 (15个点)  
- long: 8192 - 32768, step=1024 (25个点)
- very_long: 32768 - 524288, step=2048 (240个点，会根据显存自动截断)
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn

# 尝试导入 flash_attn
HAS_FLASH_ATTN = False
flash_attn_func = None

try:
    # flash-attn >= 2.0
    from flash_attn import flash_attn_func as _flash_attn_func
    flash_attn_func = _flash_attn_func
    HAS_FLASH_ATTN = True
    print("Using flash_attn.flash_attn_func (v2.x)")
except ImportError:
    try:
        # 旧版本 flash-attn
        from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func
        flash_attn_func = _flash_attn_func
        HAS_FLASH_ATTN = True
        print("Using flash_attn.flash_attn_interface.flash_attn_func")
    except ImportError:
        try:
            # 尝试 flash_attn_varlen_func
            from flash_attn import flash_attn_varlen_func
            HAS_FLASH_ATTN = True
            print("Using flash_attn.flash_attn_varlen_func")
        except ImportError:
            print("Warning: flash_attn not installed, will use manual attention")

# 打印 flash_attn 版本信息
if HAS_FLASH_ATTN:
    try:
        import flash_attn
        print(f"flash_attn version: {flash_attn.__version__}")
    except:
        pass

try:
    from einops import rearrange
except ImportError:
    rearrange = None


class FlashAttentionProfiler:
    """Flash Attention Profiler"""
    
    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        warmup_iters: int = 5,
        profile_iters: int = 20,
    ):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters
        
        print(f"FlashAttentionProfiler initialized:")
        print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
        print(f"  GQA ratio: {n_heads // n_kv_heads}")
        print(f"  Device: {device}, Dtype: {dtype}")
        print(f"  Warmup: {warmup_iters}, Profile: {profile_iters} iterations")
    
    def _create_tensors(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, ...]:
        """创建 Q, K, V tensors"""
        q = torch.randn(
            batch_size, seq_len, self.n_heads, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            batch_size, seq_len, self.n_kv_heads, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            batch_size, seq_len, self.n_kv_heads, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        return q, k, v
    
    def profile_single(self, seq_len: int, batch_size: int = 1) -> Optional[float]:
        """Profile 单个序列长度，返回时间 (ms)"""
        try:
            # 创建 tensors (flash_attn 原生支持 GQA，无需手动扩展 KV)
            q, k, v = self._create_tensors(batch_size, seq_len)
            
            # Warmup
            for _ in range(self.warmup_iters):
                if HAS_FLASH_ATTN:
                    _ = flash_attn_func(q, k, v, causal=True)
                else:
                    # Manual attention fallback (需要扩展 KV)
                    scale = 1.0 / (self.head_dim ** 0.5)
                    k_exp = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
                    v_exp = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
                    attn = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
                    attn = torch.softmax(attn, dim=-1)
                    _ = torch.matmul(attn, v_exp)
            
            torch.cuda.synchronize()
            
            # Profile
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(self.profile_iters):
                if HAS_FLASH_ATTN:
                    _ = flash_attn_func(q, k, v, causal=True)
                else:
                    scale = 1.0 / (self.head_dim ** 0.5)
                    k_exp = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
                    v_exp = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
                    attn = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
                    attn = torch.softmax(attn, dim=-1)
                    _ = torch.matmul(attn, v_exp)
            end_event.record()
            
            torch.cuda.synchronize()
            
            time_ms = start_event.elapsed_time(end_event) / self.profile_iters
            
            # 清理
            del q, k, v
            torch.cuda.empty_cache()
            
            return time_ms
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at seq_len={seq_len}")
                torch.cuda.empty_cache()
                return None
            raise
    
    def profile_range(
        self,
        start: int,
        end: int,
        step: int,
        batch_size: int = 1
    ) -> List[Tuple[int, float]]:
        """Profile 一个序列长度范围"""
        results = []
        seq_lengths = list(range(start, end + 1, step))
        
        print(f"\nProfiling range [{start}, {end}] with step {step}")
        print(f"  Total points: {len(seq_lengths)}")
        
        for i, seq_len in enumerate(seq_lengths):
            time_ms = self.profile_single(seq_len, batch_size)
            
            if time_ms is not None:
                results.append((seq_len, time_ms))
                print(f"  [{i+1}/{len(seq_lengths)}] seq={seq_len:>7}: {time_ms:.4f} ms")
            else:
                print(f"  [{i+1}/{len(seq_lengths)}] seq={seq_len:>7}: OOM, stopping this range")
                break
        
        return results


def quadratic(x, a, b, c):
    """二次函数"""
    return a * x**2 + b * x + c


def fit_quadratic(
    seq_lengths: np.ndarray,
    times: np.ndarray,
    segment_name: str = ""
) -> Dict:
    """拟合二次函数并返回系数和统计信息"""
    from scipy.optimize import curve_fit
    
    if len(seq_lengths) < 3:
        print(f"  Warning: Not enough data points ({len(seq_lengths)}) for fitting")
        return None
    
    try:
        # 初始猜测
        p0 = [1e-9, 1e-6, 0.01]
        
        # 拟合
        popt, pcov = curve_fit(quadratic, seq_lengths, times, p0=p0, maxfev=10000)
        a, b, c = popt
        
        # 计算 R²
        y_pred = quadratic(seq_lengths, a, b, c)
        ss_res = np.sum((times - y_pred) ** 2)
        ss_tot = np.sum((times - np.mean(times)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 计算误差
        errors = np.abs(times - y_pred)
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        
        # 计算各项贡献（在中间点）
        mid_seq = seq_lengths[len(seq_lengths) // 2]
        term_a = a * mid_seq**2
        term_b = b * mid_seq
        term_c = c
        total = term_a + term_b + term_c
        
        result = {
            'a': float(a),
            'b': float(b),
            'c': float(c),
            'r_squared': float(r_squared),
            'max_error_ms': float(max_error),
            'mean_error_ms': float(mean_error),
            'n_points': int(len(seq_lengths)),
            'seq_range': [int(seq_lengths.min()), int(seq_lengths.max())],
            'contribution_at_mid': {
                'mid_seq': int(mid_seq),
                'ax2_pct': float(term_a / total * 100) if total > 0 else 0,
                'bx_pct': float(term_b / total * 100) if total > 0 else 0,
                'c_pct': float(term_c / total * 100) if total > 0 else 0,
            }
        }
        
        print(f"\n  {segment_name} Fitting Results:")
        print(f"    a (x²)  = {a:.6e}")
        print(f"    b (x)   = {b:.6e}")
        print(f"    c (1)   = {c:.6f}")
        print(f"    R²      = {r_squared:.6f}")
        print(f"    Max Error = {max_error:.4f} ms, Mean Error = {mean_error:.4f} ms")
        print(f"    At seq={mid_seq}: ax²={term_a:.4f}ms ({term_a/total*100:.1f}%), "
              f"bx={term_b:.4f}ms ({term_b/total*100:.1f}%), c={term_c:.4f}ms ({term_c/total*100:.1f}%)")
        
        # 分析 b 的意义
        if abs(term_b / total) < 0.01:
            print(f"    → b 项贡献 < 1%，可忽略（符合预期：纯 attention 计算是 O(n²)）")
        elif term_b > 0:
            print(f"    → b 项显著为正，可能来自：线性内存操作、边界处理")
        else:
            print(f"    → b 项为负，可能是拟合误差或非线性效应")
        
        return result
        
    except Exception as e:
        print(f"  Fitting failed: {e}")
        return None


def plot_results(
    all_results: Dict[str, List[Tuple[int, float]]],
    coefficients: Dict[str, Dict],
    save_path: str = None
):
    """绘制 profile 结果"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'short': 'blue',
        'medium': 'green', 
        'long': 'orange',
        'very_long': 'red'
    }
    
    # 1. 所有数据点 + 拟合曲线
    ax1 = axes[0, 0]
    for segment, results in all_results.items():
        if not results:
            continue
        seq_lens = [r[0] for r in results]
        times = [r[1] for r in results]
        ax1.scatter(seq_lens, times, label=f'{segment} data', alpha=0.6, color=colors.get(segment, 'gray'))
        
        # 拟合曲线
        if segment in coefficients and coefficients[segment]:
            coef = coefficients[segment]
            x_fit = np.linspace(min(seq_lens), max(seq_lens), 100)
            y_fit = quadratic(x_fit, coef['a'], coef['b'], coef['c'])
            ax1.plot(x_fit, y_fit, '--', color=colors.get(segment, 'gray'), 
                    label=f'{segment} fit (R²={coef["r_squared"]:.4f})')
    
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Flash Attention Profile: All Segments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 短序列放大
    ax2 = axes[0, 1]
    if 'short' in all_results and all_results['short']:
        results = all_results['short']
        seq_lens = [r[0] for r in results]
        times = [r[1] for r in results]
        ax2.scatter(seq_lens, times, color='blue', alpha=0.6)
        
        if 'short' in coefficients and coefficients['short']:
            coef = coefficients['short']
            x_fit = np.linspace(min(seq_lens), max(seq_lens), 100)
            y_fit = quadratic(x_fit, coef['a'], coef['b'], coef['c'])
            ax2.plot(x_fit, y_fit, 'b--')
    
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Short Sequences (128-1024)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 系数比较
    ax3 = axes[1, 0]
    segments = list(coefficients.keys())
    valid_segments = [s for s in segments if coefficients.get(s)]
    
    if valid_segments:
        x = np.arange(len(valid_segments))
        width = 0.25
        
        a_vals = [coefficients[s]['a'] * 1e9 for s in valid_segments]  # 转换为更易读的单位
        ax3.bar(x, a_vals, width, label='a (×10⁹)', color='blue', alpha=0.7)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(valid_segments)
        ax3.set_ylabel('Coefficient Value')
        ax3.set_title('Coefficient "a" Comparison (×10⁹)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. R² 比较
    ax4 = axes[1, 1]
    if valid_segments:
        r2_vals = [coefficients[s]['r_squared'] for s in valid_segments]
        bars = ax4.bar(valid_segments, r2_vals, color=['blue', 'green', 'orange', 'red'][:len(valid_segments)], alpha=0.7)
        ax4.axhline(y=0.99, color='r', linestyle='--', label='R²=0.99 threshold')
        ax4.set_ylabel('R² Value')
        ax4.set_title('Fitting Quality (R²)')
        ax4.set_ylim(0.9, 1.0)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        for bar, r2 in zip(bars, r2_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                    f'{r2:.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Flash Attention Profiling')
    parser.add_argument('--n_heads', type=int, default=64, help='Number of attention heads')
    parser.add_argument('--n_kv_heads', type=int, default=8, help='Number of KV heads (for GQA)')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=20, help='Profile iterations')
    parser.add_argument('--save_dir', type=str, 
                       default='/home/pkuhetu/lqs/galvatron_lxy/Hetu-Galvatron/galvatron/models/llama_hf/configs',
                       help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='custom', help='Model name for output file')
    parser.add_argument('--skip_very_long', action='store_true', help='Skip very long sequences')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(" Flash Attention Profiler")
    print("=" * 70)
    print(f" Model: {args.model_name}")
    print(f" Config: n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}, head_dim={args.head_dim}")
    print(f" hidden_size = n_heads × head_dim = {args.n_heads * args.head_dim}")
    print("=" * 70)
    
    # 创建 profiler
    profiler = FlashAttentionProfiler(
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        warmup_iters=args.warmup,
        profile_iters=args.iters,
    )
    
    # 定义分段
    # segments = {
    #     'short': {
    #         'start': 128,
    #         'end': 1024,
    #         'step': 128,  # 8 points: 128, 256, 384, 512, 640, 768, 896, 1024
    #     },
    #     'medium': {
    #         'start': 1024,
    #         'end': 8192,
    #         'step': 512,  # 15 points
    #     },
    #     'long': {
    #         'start': 8192,
    #         'end': 32768,
    #         'step': 1024,  # 25 points
    #     },
    # }
    segments = {
    'short': {'start': 128, 'end': 1024, 'step': 128},
    'medium_low': {'start': 1024, 'end': 4096, 'step': 256},   
    'medium_high': {'start': 4096, 'end': 8192, 'step': 256},  
    'long': {'start': 8192, 'end': 32768, 'step': 1024},
    }
    
    if not args.skip_very_long:
        segments['very_long'] = {
            'start': 32768,
            'end': 524288,
            'step': 2048,  # Up to 240 points, will stop at OOM
        }
    
    # Profile 各区间
    all_results = {}
    coefficients = {}
    
    for segment_name, config in segments.items():
        print(f"\n{'='*70}")
        print(f" Profiling Segment: {segment_name}")
        print(f"{'='*70}")
        
        results = profiler.profile_range(
            start=config['start'],
            end=config['end'],
            step=config['step'],
            batch_size=args.batch_size,
        )
        
        all_results[segment_name] = results
        
        if len(results) >= 3:
            seq_lens = np.array([r[0] for r in results])
            times = np.array([r[1] for r in results])
            
            coef = fit_quadratic(seq_lens, times, segment_name)
            coefficients[segment_name] = coef
        else:
            print(f"  Not enough data points for fitting (got {len(results)})")
            coefficients[segment_name] = None
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存 JSON
    output = {
        'model_name': args.model_name,
        'config': {
            'n_heads': args.n_heads,
            'n_kv_heads': args.n_kv_heads,
            'head_dim': args.head_dim,
            'hidden_size': args.n_heads * args.head_dim,
        },
        'coefficients': coefficients,
        'raw_data': {k: [(int(s), float(t)) for s, t in v] for k, v in all_results.items()},
        'timestamp': timestamp,
    }
    
    json_path = os.path.join(args.save_dir, f'attention_profile_{args.model_name}_{timestamp}.json')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # 2. 绘图
    plot_path = os.path.join(args.save_dir, f'attention_profile_{args.model_name}_{timestamp}.png')
    plot_results(all_results, coefficients, plot_path)
    
    # 3. 打印汇总
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"\n{'Segment':<12} | {'a (x²)':<14} | {'b (x)':<14} | {'c (1)':<10} | {'R²':<8} | {'Points':<6}")
    print("-" * 80)
    
    for segment in ['short', 'medium', 'long', 'very_long']:
        if segment in coefficients and coefficients[segment]:
            c = coefficients[segment]
            print(f"{segment:<12} | {c['a']:<14.6e} | {c['b']:<14.6e} | {c['c']:<10.4f} | {c['r_squared']:<8.4f} | {c['n_points']:<6}")
    
    # 4. 分析 b 系数
    print("\n" + "=" * 70)
    print(" Analysis of 'b' coefficient")
    print("=" * 70)
    print("""
理论分析：
  - a*x²: Flash Attention 核心计算 QK^T 和 softmax(QK^T)V，O(n²)
  - c:    Kernel launch overhead，常数时间
  - b*x:  理论上应该很小，可能来自：
          1. Tiling 边界处理
          2. 在线 Softmax 的统计量累积
          3. 少量线性内存操作
""")
    
    for segment in coefficients:
        if coefficients[segment]:
            c = coefficients[segment]
            contrib = c.get('contribution_at_mid', {})
            b_pct = contrib.get('bx_pct', 0)
            if b_pct < 1:
                status = "✓ 可忽略 (< 1%)"
            elif b_pct < 5:
                status = "~ 较小 (1-5%)"
            else:
                status = "⚠ 显著 (> 5%)，需检查"
            print(f"  {segment}: b 项贡献 = {b_pct:.2f}% → {status}")


if __name__ == '__main__':
    main()

