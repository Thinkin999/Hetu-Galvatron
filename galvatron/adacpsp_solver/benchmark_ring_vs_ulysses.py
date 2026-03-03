#!/usr/bin/env python3
"""
Benchmark script to compare FlexSP with Ring Attention vs Ulysses-only.
针对 Qwen2.5 系列模型，在 32 GPU 集群上进行测试。

Usage:
    python benchmark_ring_vs_ulysses.py --output results.csv
"""

import argparse
import random
import time
import csv
import os
import json
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

# Import solver components
from sequence_module import Sequence, get_lens
from solver import flexSPCostModel, flexSPOptimizer, read_dataset, get_global_batch

# ============== Qwen2.5 Model Configurations ==============
# 从 meta_configs 目录读取的配置
MODEL_CONFIGS = {
    'qwen2.5-1.5b': {
        'name': 'Qwen2.5-1.5B',
        'hidden_size': 1536,
        'n_heads': 12,
        'n_kv_heads': 2,  # GQA
        'layer_num': 28,
        'ffn_dim': 8960,
        'vocab_size': 151936,
        # 估算参数（需要根据实际测量调整）
        'param_size_B': 1.5,
        'act_per_token': 2.0,
        'cpt_alpha1': 2.0e-6,
        'cpt_alpha2': 80.0e-3,
        'cpt_beta1': 300.0,
    },
    'qwen2.5-3b': {
        'name': 'Qwen2.5-3B',
        'hidden_size': 2048,
        'n_heads': 16,
        'n_kv_heads': 2,  # GQA
        'layer_num': 36,
        'ffn_dim': 11008,
        'vocab_size': 151936,
        'param_size_B': 3.0,
        'act_per_token': 2.8,
        'cpt_alpha1': 3.5e-6,
        'cpt_alpha2': 120.0e-3,
        'cpt_beta1': 450.0,
    },
    'qwen2.5-7b': {
        'name': 'Qwen2.5-7B',
        'hidden_size': 3584,
        'n_heads': 28,
        'n_kv_heads': 4,  # GQA
        'layer_num': 28,
        'ffn_dim': 18944,
        'vocab_size': 152064,
        'param_size_B': 7.0,
        'act_per_token': 4.5,
        'cpt_alpha1': 5.128e-6,
        'cpt_alpha2': 183.9576e-3,
        'cpt_beta1': 629.3563,
    },
    'qwen2.5-72b': {
        'name': 'Qwen2.5-72B',
        'hidden_size': 8192,
        'n_heads': 64,
        'n_kv_heads': 8,  # GQA
        'layer_num': 80,
        'ffn_dim': 29568,
        'vocab_size': 152064,
        'param_size_B': 72.0,
        'act_per_token': 3.5,
        'cpt_alpha1': 20.0e-6,
        'cpt_alpha2': 1000.0e-3,
        'cpt_beta1': 3500.0,
    },
}

# ============== Bandwidth Configurations ==============
# 针对 32 GPU 集群的带宽配置,还需要1 2 4卡
BANDWIDTH_CONFIGS = {
    'ib_4x': {
        # All-to-All 带宽 (GB/s)
        'alltoall': {1: 1e10, 2: 161.384, 4: 159.782, 8: 151.186, 16: 18.2204, 32: 12.5387},
        # P2P 带宽 (GB/s)
        'p2p': {1: 1e10, 2: 163.671, 4: 138.681, 8: 109.45, 16: 21.5393, 32: 11.621},
    },
    'nvlink': {
        'alltoall': {1: 1e10, 2: 150, 4: 130, 8: 110, 16: 90, 32: 70},
        'p2p': {1: 1e10, 2: 163.671, 4: 138.681, 8: 109.45, 16: 140, 32: 120},
    },
}

# ============== Datasets ==============
DATASETS = ['github', 'common_crawl', 'wikipedia']


def run_single_benchmark(
    model_config: Dict,
    dataset_name: str,
    cluster_size: int,
    memory_limit_gb: int,
    seq_limit_k: int,
    global_batch_size: int,
    bandwidth_config: Dict,
    ring_overlap_efficiency: float,
    iter_num: int,
    start_iter: int,
    method: str,
    bucket_num: int,
    enable_ring: bool,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    运行单次 benchmark（仅 Ulysses 或 Ulysses+Ring）
    """
    random.seed(0)
    
    # Create cost model
    costmodel = flexSPCostModel(
        cluster_size=cluster_size,
        hidden_size=model_config['hidden_size'],
        n_heads=model_config['n_heads'],
        n_kv_heads=model_config['n_kv_heads'],
        layer_num=model_config['layer_num'],
        param_size_B=model_config['param_size_B'],
        zero_stage=3,
        mixed_precision=True,
        act_per_token=model_config['act_per_token'],
        cpt_alpha1=model_config['cpt_alpha1'],
        cpt_alpha2=model_config['cpt_alpha2'],
        cpt_beta1=model_config['cpt_beta1'],
        alltoall_bandwidth_dict_gbs=bandwidth_config['alltoall'],
        p2p_bandwidth_dict_gbs=bandwidth_config['p2p'],
        ring_overlap_efficiency=ring_overlap_efficiency,
    )
    
    # Create optimizer
    attn_types = ['ulysses', 'ring'] if enable_ring else ['ulysses']
    optimizer = flexSPOptimizer(
        cluster_size=cluster_size,
        memory_limit_gb=memory_limit_gb,
        costmodel=costmodel,
        hide_scipoutput=True,
        hide_alloutput=not verbose,
        scip_param_dict={"limits/time": 10},
        enable_ring_attn=enable_ring,
        attn_types=attn_types,
    )
    
    # Load dataset
    seq_limit = seq_limit_k * 1000
    try:
        data = read_dataset(dataset_name, seq_limit=seq_limit, world_size=cluster_size)
    except FileNotFoundError:
        return None
    
    if len(data) < (start_iter + iter_num) * global_batch_size:
        return None
    
    # Run benchmark
    total_time = 0.0
    solve_time = 0.0
    mb_nums = []
    
    for iter_idx in range(start_iter, start_iter + iter_num):
        sequences = get_global_batch(data, iter_idx, global_batch_size)
        sequences = [Sequence(seq=seq, id=idx) for idx, seq in enumerate(sequences)]
        
        start = time.time()
        if method == 'flexSP':
            groups, results = optimizer.solve_flexSP_globalbatch_mp_gbmb(
                sequences, bucket_alg='dp', chunk_alg='sort_consec',
                mb_option_num=5, bucket_num=bucket_num
            )
        elif method == 'adaptive':
            groups, results = optimizer.homo_sp_baseline_ffd_bfd_globalbatch(
                sequences, 'bfd', sp_select_rule='adaptive'
            )
        elif method == 'static':
            # 对于 static，使用固定 sp_size = cluster_size
            groups, results = optimizer.homo_sp_baseline_ffd_bfd_globalbatch(
                sequences, 'bfd', sp_select_rule='fix_sp', sp_size=cluster_size
            )
        solve_time += time.time() - start
        
        if results:
            total_time += sum([r['M'] for r in results])
            mb_nums.append(len(results))
    
    return {
        'total_time_ms': total_time,
        'solve_time_s': solve_time,
        'avg_mb_num': np.mean(mb_nums) if mb_nums else 0,
    }


def run_comparison_benchmark(
    model_config: Dict,
    dataset_name: str,
    cluster_size: int,
    memory_limit_gb: int,
    seq_limit_k: int,
    global_batch_size: int,
    bandwidth_config: Dict,
    ring_overlap_efficiency: float,
    iter_num: int,
    start_iter: int,
    method: str,
    bucket_num: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    比较 Ring Attention vs Ulysses-only
    """
    # Ulysses-only
    result_ulysses = run_single_benchmark(
        model_config, dataset_name, cluster_size, memory_limit_gb,
        seq_limit_k, global_batch_size, bandwidth_config,
        ring_overlap_efficiency, iter_num, start_iter, method, bucket_num,
        enable_ring=False, verbose=verbose
    )
    
    # With Ring Attention
    result_ring = run_single_benchmark(
        model_config, dataset_name, cluster_size, memory_limit_gb,
        seq_limit_k, global_batch_size, bandwidth_config,
        ring_overlap_efficiency, iter_num, start_iter, method, bucket_num,
        enable_ring=True, verbose=verbose
    )
    
    if result_ulysses is None or result_ring is None:
        return None
    
    # Calculate speedup
    if result_ulysses['total_time_ms'] > 0 and result_ring['total_time_ms'] > 0:
        speedup = result_ulysses['total_time_ms'] / result_ring['total_time_ms']
    else:
        speedup = 0.0
    
    return {
        'model': model_config['name'],
        'dataset': dataset_name,
        'cluster_size': cluster_size,
        'memory_gb': memory_limit_gb,
        'seq_limit_k': seq_limit_k,
        'gbs': global_batch_size,
        'method': method,
        'n_heads': model_config['n_heads'],
        'n_kv_heads': model_config['n_kv_heads'],
        'gqa_ratio': model_config['n_heads'] / model_config['n_kv_heads'],
        'time_ulysses_ms': result_ulysses['total_time_ms'],
        'time_ring_ms': result_ring['total_time_ms'],
        'speedup': speedup,
        'improvement_pct': (speedup - 1) * 100,
        'solve_time_ulysses_s': result_ulysses['solve_time_s'],
        'solve_time_ring_s': result_ring['solve_time_s'],
        'avg_mb_ulysses': result_ulysses['avg_mb_num'],
        'avg_mb_ring': result_ring['avg_mb_num'],
    }


def print_table_header():
    """打印表格头部"""
    header = (
        f"{'Model':<15} | {'Dataset':<15} | {'SeqLimit':<8} | {'GBS':<6} | "
        f"{'Method':<10} | {'Ulysses(ms)':<12} | {'Ring(ms)':<12} | {'Speedup':<8} | {'Improv%':<8}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))


def print_table_row(result: Dict):
    """打印表格行"""
    print(
        f"{result['model']:<15} | {result['dataset']:<15} | {result['seq_limit_k']:<8} | "
        f"{result['gbs']:<6} | {result['method']:<10} | "
        f"{result['time_ulysses_ms']:<12.2f} | {result['time_ring_ms']:<12.2f} | "
        f"{result['speedup']:<8.3f} | {result['improvement_pct']:<8.2f}%"
    )


def save_results_csv(results: List[Dict], output_path: str):
    """保存结果到 CSV"""
    if not results:
        return
    
    fieldnames = list(results[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV results saved to: {output_path}")


def print_summary(results: List[Dict]):
    """打印汇总统计"""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # 按模型分组
    by_model = defaultdict(list)
    for r in results:
        by_model[r['model']].append(r)
    
    for model, model_results in by_model.items():
        valid_results = [r for r in model_results if r['speedup'] > 0]
        if not valid_results:
            continue
        
        speedups = [r['speedup'] for r in valid_results]
        improvements = [r['improvement_pct'] for r in valid_results]
        gqa_ratio = valid_results[0]['gqa_ratio']
        
        print(f"\n{model} (GQA ratio: {gqa_ratio:.1f}x):")
        print(f"  Samples:          {len(valid_results)}")
        print(f"  Avg Speedup:      {np.mean(speedups):.3f}x")
        print(f"  Max Speedup:      {np.max(speedups):.3f}x")
        print(f"  Min Speedup:      {np.min(speedups):.3f}x")
        print(f"  Avg Improvement:  {np.mean(improvements):.2f}%")
    
    # 整体统计
    all_speedups = [r['speedup'] for r in results if r['speedup'] > 0]
    if all_speedups:
        print(f"\nOverall:")
        print(f"  Total Samples:    {len(all_speedups)}")
        print(f"  Avg Speedup:      {np.mean(all_speedups):.3f}x")
        print(f"  Max Speedup:      {np.max(all_speedups):.3f}x")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Ring Attention vs Ulysses for Qwen2.5 models')
    
    # Output
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                        help='Output CSV file path')
    
    # Cluster settings (固定为 32)
    parser.add_argument('--cluster_size', type=int, default=32,
                        help='Number of GPUs (default: 32)')
    parser.add_argument('--memory_limit_gb', type=int, default=30,
                        help='Memory limit per GPU in GB')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+',
                        default=['qwen2.5-1.5b', 'qwen2.5-3b', 'qwen2.5-7b', 'qwen2.5-72b'],
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Models to benchmark')
    
    # Dataset settings
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=DATASETS,
                        help='Datasets to test')
    
    # Sequence settings (参考 solver_all.sh)
    parser.add_argument('--seq_limit_k', type=int, nargs='+', default=[32, 64, 128, 192],
                        help='Sequence length limits in K')
    
    # Batch settings
    parser.add_argument('--global_batch_size', type=int, nargs='+', default=[256, 512, 1024],
                        help='Global batch sizes to test')
    
    # Method settings
    parser.add_argument('--methods', type=str, nargs='+', default=['flexSP'],
                        choices=['flexSP', 'adaptive', 'static'],
                        help='Optimization methods to test')
    
    # Benchmark settings
    parser.add_argument('--iter_num', type=int, default=30,
                        help='Number of iterations per configuration')
    parser.add_argument('--start_iter', type=int, default=3,
                        help='Starting iteration index')
    parser.add_argument('--bucket_num', type=int, default=16,
                        help='Number of buckets for FlexSP')
    
    # Ring Attention settings
    parser.add_argument('--ring_overlap_efficiency', type=float, default=0.15,
                        help='Ring Attention overlap efficiency')
    
    # Bandwidth settings
    parser.add_argument('--bandwidth', type=str, default='ib_4x',
                        choices=list(BANDWIDTH_CONFIGS.keys()),
                        help='Bandwidth configuration')
    
    # Verbose
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    
    args = parser.parse_args()
    
    # Get bandwidth config
    bandwidth_config = BANDWIDTH_CONFIGS[args.bandwidth]
    
    # Prepare results
    results = []
    
    # Print header
    print("=" * 100)
    print("FlexSP Benchmark: Ring Attention vs Ulysses")
    print("=" * 100)
    print(f"Cluster Size:     {args.cluster_size} GPUs")
    print(f"Memory Limit:     {args.memory_limit_gb} GB/GPU")
    print(f"Models:           {args.models}")
    print(f"Datasets:         {args.datasets}")
    print(f"Seq Limits:       {args.seq_limit_k}K")
    print(f"Global Batch:     {args.global_batch_size}")
    print(f"Methods:          {args.methods}")
    print(f"Ring Overlap:     {args.ring_overlap_efficiency}")
    print(f"Bandwidth:        {args.bandwidth}")
    print("=" * 100)
    
    # Calculate total configurations
    total_configs = (
        len(args.models) * len(args.datasets) * len(args.seq_limit_k) *
        len(args.global_batch_size) * len(args.methods)
    )
    current_config = 0
    
    # Print table header
    print_table_header()
    
    for model_name in args.models:
        model_config = MODEL_CONFIGS[model_name]
        
        for dataset in args.datasets:
            for seq_limit in args.seq_limit_k:
                for gbs in args.global_batch_size:
                    for method in args.methods:
                        current_config += 1
                        
                        if args.verbose:
                            print(f"\n[{current_config}/{total_configs}] "
                                  f"{model_name} | {dataset} | seq={seq_limit}K | gbs={gbs} | {method}")
                        
                        result = run_comparison_benchmark(
                            model_config=model_config,
                            dataset_name=dataset,
                            cluster_size=args.cluster_size,
                            memory_limit_gb=args.memory_limit_gb,
                            seq_limit_k=seq_limit,
                            global_batch_size=gbs,
                            bandwidth_config=bandwidth_config,
                            ring_overlap_efficiency=args.ring_overlap_efficiency,
                            iter_num=args.iter_num,
                            start_iter=args.start_iter,
                            method=method,
                            bucket_num=args.bucket_num,
                            verbose=args.verbose,
                        )
                        
                        if result:
                            results.append(result)
                            print_table_row(result)
                        else:
                            print(f"  [Skip] {model_name} | {dataset} | {seq_limit}K | {gbs} - Dataset unavailable")
    
    # Save results
    save_results_csv(results, args.output)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 100)
    print("Benchmark Complete!")
    print("=" * 100)


if __name__ == '__main__':
    main()
