#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FlexSP 求解器验证脚本

验证内容：
1. 对比 Ulysses-only vs Ulysses+Ring 的求解结果
2. 分析策略分配（SP size, attn_type）
3. 验证 ILP 预测时间 vs 精确模型时间
4. 测试不同序列长度分布
"""

import sys
import os
import numpy as np
import random
from typing import List, Dict, Tuple

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver import flexSPCostModel, flexSPOptimizer, Sequence, get_lens

# ==================== 模型配置 ====================
QWEN_3B_CONFIG = {
    'name': 'Qwen2.5-3B',
    'hidden_size': 2048,
    'n_heads': 16,
    'n_kv_heads': 2,  # GQA
    'layer_num': 36,
    'param_size_B': 3.3970,
    'act_per_token': 4.08095825,
    'cpt_alpha1': 2.734275689835784e-09,  # 单层
    'cpt_alpha2': 0 ,#9.004521380203265e-05,  # 单层
    'cpt_beta1': 1.635357064600996e+00,   # 单层
}
#还没有写
QWEN_7B_CONFIG = {
    'name': 'Qwen2.5-3B',
    'hidden_size': 2048,
    'n_heads': 16,
    'n_kv_heads': 2,  # GQA
    'layer_num': 36,
    'param_size_B': 3.3970,
    'act_per_token': 4.08095825,
    'cpt_alpha1': 2.734275689835784e-09,  # 单层
    'cpt_alpha2': 9.004521380203265e-05,  # 单层
    'cpt_beta1': 1.635357064600996e+00,   # 单层
}
QWEN_72B_CONFIG = {
    'name': 'Qwen2.5-72B',
    'hidden_size': 8192,
    'n_heads': 64,
    'n_kv_heads': 8,  # GQA
    'layer_num': 80,
    'param_size_B': 72,
    'act_per_token': 4.4,
    'cpt_alpha1': 7.454522e-08,  # 单层
    'cpt_alpha2': 0.0,
    'cpt_beta1': 0.05,
}

# ==================== 带宽配置 ====================
BANDWIDTH_CONFIG = {
    'alltoall': {1: 1e10, 2: 161.384, 4: 159.782, 8: 151.186, 16: 18.2204, 32: 12.5387, 64: 9.6},
    'p2p': {1: 1e10, 2: 163.671, 4: 138.681, 8: 109.45, 16: 21.5393, 32: 11.621, 64: 9},
}

# ==================== 数据集采样 ====================
DATASETS_DIR = "/home/pkuhetu/lqs/flexsp/Hetu-Galvatron/galvatron/datasets"

def generate_seqlens_from_datasets(
    global_batch_size: int = 32,
    max_seqlength: int = 131072,
    dataset_name: str = "github",
    random_seed: int = None,
    filter_mode: str = "truncate",  # "skip" 跳过超长序列, "truncate" 截断到 max_seqlength
) -> List[int]:
    """
    从数据集 txt 文件中采样序列长度
    
    Args:
        global_batch_size: 采样的序列数量
        max_seqlength: 最大序列长度限制
        dataset_name: 数据集名称 (github, common_crawl 等)
        random_seed: 随机种子 (None 表示使用随机种子)
        filter_mode: 过滤模式
            - "skip": 跳过超过 max_seqlength 的序列
            - "truncate": 将超过 max_seqlength 的序列截断 (默认)
    
    Returns:
        采样的序列长度列表
    """
    import random
    
    # 构建文件路径
    dataset_path = os.path.join(DATASETS_DIR, f"{dataset_name}.txt")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    # 设置随机种子（在读取前设置，确保采样一致性）
    if random_seed is not None:
        random.seed(random_seed)
    
    # 读取所有序列长度
    all_seqlens = []
    with open(dataset_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seqlen = int(line)
                    all_seqlens.append(seqlen)
                except ValueError:
                    continue
    
    print(f"从 {dataset_name}.txt 加载了 {len(all_seqlens)} 条序列")
    
    # 先随机采样，再进行截断/过滤
    if len(all_seqlens) <= global_batch_size:
        sampled_raw = all_seqlens
    else:
        sampled_raw = random.sample(all_seqlens, global_batch_size)
    
    # 根据 filter_mode 处理采样结果
    if filter_mode == "truncate":
        # 截断超长序列到 max_seqlength
        truncated_count = sum(1 for s in sampled_raw if s > max_seqlength)
        sampled = [min(s, max_seqlength) for s in sampled_raw]
        if truncated_count > 0:
            print(f"截断了 {truncated_count} 条超长序列 (max_seqlength={max_seqlength})")
    elif filter_mode == "skip":
        # 跳过超长序列，可能导致数量不足
        sampled = [s for s in sampled_raw if s <= max_seqlength]
        skipped = len(sampled_raw) - len(sampled)
        if skipped > 0:
            print(f"跳过了 {skipped} 条超长序列 (max_seqlength={max_seqlength})")
    else:
        sampled = sampled_raw
    
    # 排序（方便查看）
    sampled.sort()
    
    if len(sampled) > 0:
        print(f"采样 {len(sampled)} 条序列: min={min(sampled)}, max={max(sampled)}, "
              f"mean={np.mean(sampled):.0f}, median={np.median(sampled):.0f}")
    else:
        print(f"警告: 采样结果为空!")
    
    return sampled


def run_solver_benchmark(
    global_batch_size: int = 32,
    max_seqlength: int = 131072,
    dataset_name: str = "github",
    num_iters: int = 5,
    config: Dict = None,
    bandwidth: Dict = None,
    cluster_size: int = 64,
    memory_limit_gb: float = 160.0,
    sp_size_min: int = 1,
    sp_size_max: int = None,  # None 表示使用 cluster_size
    verbose: bool = True,
):
    """
    运行 Solver 基准测试，对比纯 Ulysses 和 Ulysses+Ring 两种模式
    
    Args:
        global_batch_size: 每次迭代采样的序列数量
        max_seqlength: 最大序列长度
        dataset_name: 数据集名称
        num_iters: 迭代次数
        config: 模型配置
        bandwidth: 带宽配置
        cluster_size: GPU 数量
        memory_limit_gb: 单卡显存限制 (GB)
        sp_size_min: 最小 SP size (必须是 2 的幂次)
        sp_size_max: 最大 SP size (必须是 2 的幂次, None 表示使用 cluster_size)
        verbose: 是否打印详细信息
    
    Returns:
        包含所有迭代结果的字典
    """
    if config is None:
        config = QWEN_72B_CONFIG
    if bandwidth is None:
        bandwidth = BANDWIDTH_CONFIG
    if sp_size_max is None:
        sp_size_max = cluster_size
    
    # 生成有效的 SP size 列表
    valid_sp_sizes = []
    sp = 1
    while sp <= cluster_size:
        if sp >= sp_size_min and sp <= sp_size_max:
            valid_sp_sizes.append(sp)
        sp *= 2
    
    print("=" * 90)
    print(" FlexSP Solver 基准测试")
    print("=" * 90)
    print(f"模型: {config['name']}")
    print(f"集群: {cluster_size} GPUs, {memory_limit_gb} GB/GPU")
    print(f"数据集: {dataset_name}")
    print(f"Global Batch Size: {global_batch_size}")
    print(f"Max Sequence Length: {max_seqlength}")
    print(f"SP Size 范围: [{sp_size_min}, {sp_size_max}] -> {valid_sp_sizes}")
    print(f"迭代次数: {num_iters}")
    print("=" * 90)
    
    # 创建 Cost Model
    cost_model = flexSPCostModel(
        cluster_size=cluster_size,
        hidden_size=config['hidden_size'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        layer_num=config['layer_num'],
        param_size_B=config['param_size_B'],
        act_per_token=config['act_per_token'],
        cpt_alpha1=config['cpt_alpha1'],
        cpt_alpha2=config['cpt_alpha2'],
        cpt_beta1=config['cpt_beta1'],
        alltoall_bandwidth_dict_gbs=bandwidth['alltoall'],
        p2p_bandwidth_dict_gbs=bandwidth['p2p'],
        ring_overlap_efficiency=0.12,
    )
    
    # DEBUG: 打印显存容量信息
    print(f"\n[DEBUG] Memory Check:")
    print(f"  Memory Limit: {memory_limit_gb} GB")
    print(f"  Model States: {cost_model.model_states_mb:.2f} MB")
    print(f"  Act Per Token: {cost_model.act_per_token:.2f} MB")

    # [Added] Cost Model Check inside Benchmark
    print("\n[DEBUG] Cost Model Check inside Benchmark:")
    test_seq = 23895
    test_sp = 32
    u_time = cost_model.ulysses_time_whole_model(test_seq, test_sp)
    r_time = cost_model.zigzag_ring_flash_attention_time_whole_model(test_seq, test_sp)
    print(f"Seq={test_seq}, SP={test_sp}")
    print(f"  Ulysses Time: {u_time:.2f} ms")
    print(f"  Ring Time:    {r_time:.2f} ms")
    print(f"  Winner: {'Ring' if r_time < u_time else 'Ulysses'}")
    print("-" * 50)
    
    # 静态检查 SP=64 的容量
    sp_debug = 64
    cap = cost_model.token_capacity(memory_limit_gb, sp_size=sp_debug, attn_type='ulysses')
    print(f"  SP={sp_debug}, Ulysses Token Capacity (per GPU): {cap}")
    
    # 创建两种模式的 Optimizer
    optimizer_ulysses = flexSPOptimizer(
        cluster_size=cluster_size,
        memory_limit_gb=memory_limit_gb,
        costmodel=cost_model,
        hide_scipoutput=True,
        hide_alloutput=True,
        enable_ring_attn=False,
        attn_types=['ulysses'],
        sp_size_options=valid_sp_sizes,
        scip_param_dict={'limits/time': 1000},
    )
    
    optimizer_ring = flexSPOptimizer(
        cluster_size=cluster_size,
        memory_limit_gb=memory_limit_gb,
        costmodel=cost_model,
        hide_scipoutput=True,
        hide_alloutput=True,
        enable_ring_attn=True,
        attn_types=['ulysses', 'ring'],
        sp_size_options=valid_sp_sizes,
        scip_param_dict={'limits/time': 1000},
    )
    
    # 存储每次迭代的结果
    all_results = []
    
    for iter_idx in range(num_iters):
        print(f"\n{'='*90}")
        print(f" Iteration {iter_idx + 1}/{num_iters}")
        print(f"{'='*90}")
        
        # 采样序列长度 (每次使用不同的随机种子)
        seqlen_list = generate_seqlens_from_datasets(
            global_batch_size=global_batch_size,
            max_seqlength=max_seqlength,
            dataset_name=dataset_name,
            random_seed=iter_idx * 66666,  # 每次迭代使用不同种子
        )
        
        # 转换为 Sequence 对象列表
        seqs = [Sequence(length) for length in seqlen_list]
        
        # 预检查：集群总容量
        # 使用最大 SP 来估算最大可能的集群容量（因为模型状态是切分的，SP 越大 activation 显存利用率通常越高或不变）
        max_sp = cluster_size
        cap_per_gpu = cost_model.token_capacity(memory_limit_gb, sp_size=max_sp, attn_type='ulysses')
        total_cap = cap_per_gpu * cluster_size
        total_tokens = sum(seqlen_list)
        
        print(f"\n[Capacity Check]")
        print(f"  Total Tokens: {total_tokens}")
        print(f"  Cluster Capacity (Est. with SP={max_sp}): ~{total_cap}")
        if total_tokens > total_cap:
            print(f"  WARNING: Total tokens exceed cluster capacity! Solver will likely fail.")
            print(f"  建议: 减小 --global_batch_size 或增加 --device_memory")
        
        # ========== Ulysses-Only ==========
        print("\n>>> 求解 Ulysses-Only 策略...")
        try:
            results_ulysses = optimizer_ulysses.solve_flexSP(seqs)
            analysis_ulysses = analyze_solution(results_ulysses, cost_model, seqs)
            
            if analysis_ulysses['status'] == 'success':
                ulysses_status = 'success'
                ulysses_time = analysis_ulysses.get('max_exact_time', float('inf'))
            else:
                ulysses_status = 'failed'
                ulysses_time = float('inf')
                print("Ulysses 求解失败: 未找到可行解 (Results is None)")
        except Exception as e:
            print(f"Ulysses 求解异常: {e}")
            import traceback
            traceback.print_exc()
            analysis_ulysses = {'status': 'failed', 'error': str(e)}
            ulysses_status = 'failed'
            ulysses_time = float('inf')
        
        # ========== Ulysses + Ring ==========
        print("\n>>> 求解 Ulysses+Ring 策略...")
        try:
            results_ring = optimizer_ring.solve_flexSP(seqs)
            analysis_ring = analyze_solution(results_ring, cost_model, seqs)
            
            if analysis_ring['status'] == 'success':
                ring_status = 'success'
                ring_time = analysis_ring.get('max_exact_time', float('inf'))
            else:
                ring_status = 'failed'
                ring_time = float('inf')
                print("Ring 求解失败: 未找到可行解 (Results is None)")
        except Exception as e:
            print(f"Ring 求解异常: {e}")
            import traceback
            traceback.print_exc()
            analysis_ring = {'status': 'failed', 'error': str(e)}
            ring_status = 'failed'
            ring_time = float('inf')
        
        # 对比结果
        # 即使只有一个成功，也尝试打印
        if verbose:
            # 打印详细的迭代报告
            print_iteration_summary(
                iter_idx + 1,
                seqlen_list,
                analysis_ulysses,
                analysis_ring,
                verbose=True
            )
            
        if ulysses_status == 'success' and ring_status == 'success':
            speedup = ulysses_time / ring_time if ring_time > 0 else 0
        else:
            speedup = 0
            if verbose:
                print(f"\n--- Iteration {iter_idx + 1} Status ---")
                print(f"Ulysses: {ulysses_status}")
                print(f"Ring: {ring_status}")
        
        # 存储结果
        iter_result = {
            'iter': iter_idx + 1,
            'seqlens': seqlen_list,
            'ulysses': {
                'status': ulysses_status,
                'time': ulysses_time,
                'analysis': analysis_ulysses,
            },
            'ring': {
                'status': ring_status,
                'time': ring_time,
                'analysis': analysis_ring,
            },
            'speedup': speedup,
        }
        all_results.append(iter_result)
    
    # ========== 汇总统计 ==========
    print("\n" + "=" * 90)
    print(" 汇总统计")
    print("=" * 90)
    
    successful_iters = [r for r in all_results 
                        if r['ulysses']['status'] == 'success' and r['ring']['status'] == 'success']
    
    if successful_iters:
        ulysses_times = [r['ulysses']['time'] for r in successful_iters]
        ring_times = [r['ring']['time'] for r in successful_iters]
        speedups = [r['speedup'] for r in successful_iters]
        
        print(f"\n成功迭代次数: {len(successful_iters)}/{num_iters}")
        print(f"\n{'指标':<20} {'Ulysses-Only':<15} {'Ulysses+Ring':<15} {'加速比':<10}")
        print("-" * 70)
        print(f"{'平均时间 (ms)':<20} {np.mean(ulysses_times):<15.2f} {np.mean(ring_times):<15.2f} {np.mean(speedups):<10.3f}x")
        print(f"{'最小时间 (ms)':<20} {np.min(ulysses_times):<15.2f} {np.min(ring_times):<15.2f} {np.min(speedups):<10.3f}x")
        print(f"{'最大时间 (ms)':<20} {np.max(ulysses_times):<15.2f} {np.max(ring_times):<15.2f} {np.max(speedups):<10.3f}x")
        print(f"{'标准差 (ms)':<20} {np.std(ulysses_times):<15.2f} {np.std(ring_times):<15.2f} {np.std(speedups):<10.3f}")
        
        # 每次迭代的详细结果
        print(f"\n{'='*90}")
        print(" 每次迭代详情")
        print(f"{'='*90}")
        print(f"{'Iter':<6} {'SeqLen Range':<20} {'Ulysses(ms)':<15} {'Ring(ms)':<15} {'Speedup':<10}")
        print("-" * 70)
        
        for r in all_results:
            if r['ulysses']['status'] == 'success' and r['ring']['status'] == 'success':
                seqlens = r['seqlens']
                seq_range = f"[{min(seqlens)}, {max(seqlens)}]"
                print(f"{r['iter']:<6} {seq_range:<20} {r['ulysses']['time']:<15.2f} {r['ring']['time']:<15.2f} {r['speedup']:<10.3f}x")
            else:
                print(f"{r['iter']:<6} {'FAILED':<20}")
    else:
        print("所有迭代都失败了！")
    
    return all_results


def generate_test_sequences(scenario: str, num_samples: int = 32) -> List[int]:
    """
    生成测试序列长度
    
    Args:
        scenario: 场景类型
            - 'short': 短序列为主 (256 - 4096)
            - 'long': 长序列为主 (8192 - 131072)
            - 'mixed': 混合长度
            - 'extreme': 极端分布
        num_samples: 样本数量
    
    Returns:
        序列长度列表
    """
    if scenario == 'short':
        # 短序列：256 - 4096
        seqlens = [random.randint(256, 4096) for _ in range(num_samples)]
    elif scenario == 'long':
        # 长序列：8192 - 131072
        seqlens = [random.randint(8192, 131072) for _ in range(num_samples)]
    elif scenario == 'mixed':
        # 混合：50% 短 + 50% 长
        n_short = num_samples // 2
        n_long = num_samples - n_short
        seqlens = [random.randint(256, 4096) for _ in range(n_short)]
        seqlens += [random.randint(8192, 65536) for _ in range(n_long)]
    elif scenario == 'extreme':
        # 极端：少数超长 + 大量短
        n_long = max(2, num_samples // 8)
        n_short = num_samples - n_long
        seqlens = [random.randint(256, 2048) for _ in range(n_short)]
        seqlens += [random.randint(65536, 262144) for _ in range(n_long)]
    else:
        # 默认：均匀分布
        seqlens = [random.randint(512, 32768) for _ in range(num_samples)]
    
    return sorted(seqlens)


def debug_costmodel_consistency(config: Dict = None, bandwidth: Dict = None):
    """
    调试 Cost Model 一致性：比较画图函数和 solver 中的时间计算
    """
    if config is None:
        config = QWEN_72B_CONFIG
    if bandwidth is None:
        bandwidth = BANDWIDTH_CONFIG
    
    print("=" * 90)
    print(" Cost Model 一致性检查")
    print("=" * 90)
    
    # 创建 cost model (使用真实的 cpt_alpha2)
    cost_model = flexSPCostModel(
        cluster_size=64,
        hidden_size=config['hidden_size'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        layer_num=config['layer_num'],
        param_size_B=config['param_size_B'],
        act_per_token=config['act_per_token'],
        cpt_alpha1=config['cpt_alpha1'],
        cpt_alpha2=config['cpt_alpha2'],  # 注意：不是 0!
        cpt_beta1=config['cpt_beta1'],
        alltoall_bandwidth_dict_gbs=bandwidth['alltoall'],
        p2p_bandwidth_dict_gbs=bandwidth['p2p'],
        ring_overlap_efficiency=0.15,
    )
    
    print("\n[DEBUG] Cost Model Check inside Benchmark:")
    test_seq = 23895
    test_sp = 32
    u_time = cost_model.ulysses_time_whole_model(test_seq, test_sp)
    r_time = cost_model.zigzag_ring_flash_attention_time_whole_model(test_seq, test_sp)
    print(f"Seq={test_seq}, SP={test_sp}")
    print(f"  Ulysses Time: {u_time:.2f} ms")
    print(f"  Ring Time:    {r_time:.2f} ms")
    print(f"  Winner: {'Ring' if r_time < u_time else 'Ulysses'}")
    print("-" * 50)
    
    print(f"\n模型配置: {config['name']}")
    print(f"  cpt_alpha1 = {config['cpt_alpha1']:.6e}")
    print(f"  cpt_alpha2 = {config['cpt_alpha2']:.6e}")
    print(f"  cpt_beta1 = {config['cpt_beta1']:.6f}")
    print(f"  layer_num = {config['layer_num']}")
    print(f"  bwd_fwd_coe = {cost_model.bwd_fwd_coe}")
    
    test_seqlens = [1024, 4096, 16384, 65536, 131072]
    sp_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    print("\n" + "=" * 90)
    print(" Ulysses SP 时间比较")
    print("=" * 90)
    
    for sp_size in sp_sizes:
        print(f"\n--- SP={sp_size} ---")
        print(f"{'SeqLen':>10} | {'time_single':>12} | {'seqs_total':>12} | {'+bias':>12} | {'diff%':>8}")
        print("-" * 70)
        
        for seqlen in test_seqlens:
            # Solver 方式：time_single + compute_bias
            time_single = cost_model.ulysses_time_single(seqlen, sp_size)
            compute_bias = cost_model.compute_bias(sp_size, 'ulysses')
            solver_total = time_single + compute_bias
            
            # 精确模型：seqs_total_time_whole_time (单条序列)
            seqs_total = cost_model.seqs_total_time_whole_time([seqlen], sp_size, 'ulysses')
            
            # 差异
            diff_pct = abs(solver_total - seqs_total) / seqs_total * 100 if seqs_total > 0 else 0
            
            print(f"{seqlen:>10} | {time_single:>12.2f} | {seqs_total:>12.2f} | {solver_total:>12.2f} | {diff_pct:>7.2f}%")
    
    print("\n" + "=" * 90)
    print(" Ring Attention 时间比较")
    print("=" * 90)
    
    for sp_size in [2, 4, 8, 16, 32, 64]:
        print(f"\n--- SP={sp_size} ---")
        print(f"{'SeqLen':>10} | {'time_single':>12} | {'seqs_total':>12} | {'+bias':>12} | {'diff%':>8}")
        print("-" * 70)
        
        for seqlen in test_seqlens:
            # Solver 方式：ring_time_single + compute_bias
            time_single = cost_model.ring_time_single(seqlen, sp_size)
            compute_bias = cost_model.compute_bias(sp_size, 'ring')
            solver_total = time_single + compute_bias
            
            # 精确模型
            seqs_total = cost_model.seqs_total_time_whole_time([seqlen], sp_size, 'ring')
            
            # 差异
            diff_pct = abs(solver_total - seqs_total) / seqs_total * 100 if seqs_total > 0 else 0
            
            print(f"{seqlen:>10} | {time_single:>12.2f} | {seqs_total:>12.2f} | {solver_total:>12.2f} | {diff_pct:>7.2f}%")
    
    print("\n" + "=" * 90)
    print(" 多序列 Group 时间比较")
    print("=" * 90)
    
    # 测试多条序列的情况
    test_groups = [
        [1024, 2048, 4096],
        [8192, 8192, 8192, 8192],
        [65536, 32768, 16384],
    ]
    
    for sp_size in [4, 8, 16]:
        print(f"\n--- SP={sp_size} ---")
        
        for seqlens in test_groups:
            # Solver 方式
            time_single_sum = sum(cost_model.ulysses_time_single(s, sp_size) for s in seqlens)
            compute_bias = cost_model.compute_bias(sp_size, 'ulysses')
            solver_total = time_single_sum + compute_bias
            
            # 精确模型
            seqs_total = cost_model.seqs_total_time_whole_time(seqlens, sp_size, 'ulysses')
            
            diff_pct = abs(solver_total - seqs_total) / seqs_total * 100 if seqs_total > 0 else 0
            
            print(f"  {seqlens}: solver={solver_total:.2f} ms, exact={seqs_total:.2f} ms, diff={diff_pct:.2f}%")


def print_detailed_allocation(analysis: Dict, seqlen_list: List[int], strategy_name: str):
    """
    打印详细的序列分配信息
    
    展示每个 group 包含哪些序列，以及它们的长度
    """
    if analysis.get('status') != 'success':
        print(f"\n{strategy_name}: 求解失败")
        return
    
    print(f"\n{'='*90}")
    print(f" {strategy_name} - 详细分配")
    print(f"{'='*90}")
    
    groups = analysis.get('groups', [])
    
    # 按时间排序（最长时间的 group 排前面）
    sorted_groups = sorted(groups, key=lambda g: g['exact_time'], reverse=True)
    
    total_seqs = sum(g['num_seqs'] for g in groups)
    total_gpus = sum(g['sp_size'] for g in groups)
    
    print(f"总序列数: {total_seqs}, 总 Group 数: {len(groups)}, 总 GPU 使用: {total_gpus}")
    print(f"最大 Group 时间: {analysis['max_exact_time']:.2f} ms")
    print()
    
    for i, g in enumerate(sorted_groups):
        sp_size = g['sp_size']
        attn_type = g['attn_type']
        seq_lens = g['seq_lens']
        exact_time = g['exact_time']
        total_tokens = g['total_tokens']
        
        print(f"Group {i+1}: SP={sp_size}, Attn={attn_type}, Time={exact_time:.2f}ms, "
              f"Seqs={len(seq_lens)}, Tokens={total_tokens}")
        
        # 打印序列长度（按降序）
        sorted_lens = sorted(seq_lens, reverse=True)
        if len(sorted_lens) <= 10:
            seq_str = ", ".join(str(s) for s in sorted_lens)
        else:
            # 显示前5个和后5个
            seq_str = ", ".join(str(s) for s in sorted_lens[:5])
            seq_str += f" ... ({len(sorted_lens)-10} more) ... "
            seq_str += ", ".join(str(s) for s in sorted_lens[-5:])
        
        print(f"   序列长度: [{seq_str}]")
    
    print()


def print_iteration_summary(
    iter_idx: int,
    seqlen_list: List[int],
    analysis_ulysses: Dict,
    analysis_ring: Dict,
    verbose: bool = True
):
    """
    打印单次迭代的详细汇总
    """
    print(f"\n{'#'*90}")
    print(f"# Iteration {iter_idx} 详细报告")
    print(f"{'#'*90}")
    
    # 打印输入序列
    print(f"\n输入序列 ({len(seqlen_list)} 条):")
    sorted_seqs = sorted(seqlen_list, reverse=True)
    
    # 分组显示：短序列、中序列、长序列
    short = [s for s in sorted_seqs if s <= 4096]
    medium = [s for s in sorted_seqs if 4096 < s <= 32768]
    long = [s for s in sorted_seqs if s > 32768]
    
    print(f"   短序列 (<=4K): {len(short)} 条")
    print(f"   中序列 (4K-32K): {len(medium)} 条")
    print(f"   长序列 (>32K): {len(long)} 条")
    
    if verbose:
        # 完整的序列长度列表
        if len(sorted_seqs) <= 20:
            print(f"   所有长度: {sorted_seqs}")
        else:
            print(f"   最长 10 条: {sorted_seqs[:10]}")
            print(f"   最短 10 条: {sorted_seqs[-10:]}")
    
    # 打印两种策略的详细分配
    if analysis_ulysses.get('status') == 'success':
        print_detailed_allocation(analysis_ulysses, seqlen_list, "Ulysses-Only")
    
    if analysis_ring.get('status') == 'success':
        print_detailed_allocation(analysis_ring, seqlen_list, "Ulysses+Ring")
    
    # 对比
    if analysis_ulysses.get('status') == 'success' and analysis_ring.get('status') == 'success':
        time_u = analysis_ulysses['max_exact_time']
        time_r = analysis_ring['max_exact_time']
        speedup = time_u / time_r if time_r > 0 else 0
        
        print(f"\n{'='*90}")
        print(f" 时间对比")
        print(f"{'='*90}")
        print(f"Ulysses-Only: {time_u:.2f} ms")
        print(f"Ulysses+Ring: {time_r:.2f} ms")
        print(f"加速比: {speedup:.3f}x ({(speedup-1)*100:.1f}% 提升)" if speedup > 1 else f"加速比: {speedup:.3f}x")


def print_strategy_comparison(analysis_ulysses: Dict, analysis_ring: Dict):
    """打印两种策略的分配对比"""
    
    def get_strategy_summary(analysis):
        """获取策略摘要"""
        if analysis.get('status') != 'success':
            return {}
        
        summary = {}
        for g in analysis.get('groups', []):
            sp_size = g['sp_size']
            attn_type = g['attn_type']
            key = f"SP={sp_size},{attn_type}"
            
            if key not in summary:
                summary[key] = {'groups': 0, 'seqs': 0, 'gpus': 0, 'time': 0}
            summary[key]['groups'] += 1
            summary[key]['seqs'] += g['num_seqs']
            summary[key]['gpus'] += g['sp_size']
            summary[key]['time'] += g.get('time', 0)
        
        return summary
    
    summary_u = get_strategy_summary(analysis_ulysses)
    summary_r = get_strategy_summary(analysis_ring)
    
    all_keys = sorted(set(summary_u.keys()) | set(summary_r.keys()))
    
    print(f"{'策略':<20} {'Ulysses-Only':<25} {'Ulysses+Ring':<25}")
    print("-" * 75)
    
    for key in all_keys:
        u_info = summary_u.get(key, {'groups': 0, 'seqs': 0, 'gpus': 0})
        r_info = summary_r.get(key, {'groups': 0, 'seqs': 0, 'gpus': 0})
        
        u_str = f"{u_info['groups']}g / {u_info['seqs']}seq / {u_info['gpus']}gpu" if u_info['groups'] > 0 else "-"
        r_str = f"{r_info['groups']}g / {r_info['seqs']}seq / {r_info['gpus']}gpu" if r_info['groups'] > 0 else "-"
        
        print(f"{key:<20} {u_str:<25} {r_str:<25}")

# ==================== 求解器测试 ====================
def create_cost_model(config: Dict, bandwidth: Dict) -> flexSPCostModel:
    """创建 Cost Model"""
    return flexSPCostModel(
        cluster_size=64,
        hidden_size=config['hidden_size'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        layer_num=config['layer_num'],
        param_size_B=config['param_size_B'],
        act_per_token=config['act_per_token'],
        cpt_alpha1=config['cpt_alpha1'],
        cpt_alpha2=config['cpt_alpha2'],
        cpt_beta1=config['cpt_beta1'],
        alltoall_bandwidth_dict_gbs=bandwidth['alltoall'],
        p2p_bandwidth_dict_gbs=bandwidth['p2p'],
        ring_overlap_efficiency=0.15,
    )


def create_optimizer(cost_model: flexSPCostModel, enable_ring: bool) -> flexSPOptimizer:
    """创建 Optimizer"""
    return flexSPOptimizer(
        cluster_size=64,
        memory_limit_gb=40,
        costmodel=cost_model,
        hide_scipoutput=True,
        hide_alloutput=True,
        enable_ring_attn=enable_ring,
        attn_types=['ulysses', 'ring'] if enable_ring else ['ulysses'],
        scip_param_dict={'limits/time': 30},
    )

# 默认图片保存路径
PICTURES_DIR = "/home/pkuhetu/lqs/galvatron_lxy/Hetu-Galvatron/galvatron/flexsp_solver/pictures"


def ulysses_time_per_device(
    config: Dict = None,
    bandwidth: Dict = None,
    device_memory_gb: float = 40.0,
    zero3_world_size: int = 32,
    max_seq_len: int = None,  # None 则自动计算
    save_path: str = "auto",  # "auto" 自动命名, None 不保存, 或指定路径
    show_plot: bool = True
):
    """
    绘制 Ulysses SP 下单卡时间随序列长度变化的曲线
    
    横轴：序列长度
    纵轴：单卡总时间（计算 + 通信）
    
    随着序列长度增加，SP size 会根据显存限制动态调整：
    - [0, max_token]: SP=1 (无并行)
    - [max_token, 2*max_token]: SP=2
    - [2*max_token, 4*max_token_with_rep]: SP=4 (可能有 KV 复制)
    - ...
    
    参数:
        config: 模型配置字典 (默认使用 QWEN_3B_CONFIG)
        bandwidth: 带宽配置字典 (默认使用 BANDWIDTH_CONFIG)
        device_memory_gb: 单卡显存 (GB)
        zero3_world_size: ZeRO-3 并行度 (同时也是最大设备数)
        max_seq_len: 最大序列长度 (None 则自动计算为 2^k)
        save_path: 保存路径 ("auto"=自动命名并保存到默认目录, None=不保存, 或指定完整路径)
        show_plot: 是否显示图像
    
    返回:
        seq_lens: 序列长度数组
        total_times: 总时间数组
        compute_times: 计算时间数组
        comm_times: 通信时间数组
        sp_used: 使用的 SP size 数组
    """
    import matplotlib.pyplot as plt
    import math
    from datetime import datetime
    
    # 使用默认配置
    if config is None:
        config = QWEN_3B_CONFIG
    if bandwidth is None:
        bandwidth = BANDWIDTH_CONFIG
    
    # 创建 cost model
    cost_model = flexSPCostModel(
        cluster_size=zero3_world_size,
        hidden_size=config['hidden_size'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        layer_num=config['layer_num'],
        param_size_B=config['param_size_B'],
        act_per_token=config['act_per_token'],
        cpt_alpha1=config['cpt_alpha1'],
        cpt_alpha2=config['cpt_alpha2'],
        cpt_beta1=config['cpt_beta1'],
        alltoall_bandwidth_dict_gbs=bandwidth['alltoall'],
        p2p_bandwidth_dict_gbs=bandwidth['p2p'],
        ring_overlap_efficiency=0.15,
    )
    
    # 模型参数
    h = cost_model.h
    n_heads = cost_model.n_heads
    n_kv_heads = cost_model.n_kv_heads
    l = cost_model.l
    bwd_coe = cost_model.bwd_fwd_coe
    
    # 可用 SP sizes (2 的幂次，最大为 zero3_world_size)
    sp_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    sp_sizes = [s for s in sp_sizes if s <= zero3_world_size and s <= n_heads ]
    
    # ==================== 计算每个 SP size 能处理的最大序列长度 ====================
    def get_max_seq_for_sp(sp_size: int) -> int:
        """
        计算 SP=sp_size 时能处理的最大序列长度
        max_total_seq = sp_size * token_capacity(sp_size)
        """
        token_cap = cost_model.token_capacity(device_memory_gb, sp_size=sp_size, attn_type='ulysses')
        return sp_size * token_cap
    
    # 计算区间边界
    sp_boundaries = {}
    for sp_size in sp_sizes:
        sp_boundaries[sp_size] = get_max_seq_for_sp(sp_size)
    
    # ==================== 自动计算 max_seq_len ====================
    # max_seq_len = 2^k, where k = floor(log2(device_num * token_capacity_max_sp))
    if max_seq_len is None:
        max_sp = sp_sizes[-1]  # 最大 SP size
        raw_max_seq = sp_boundaries[max_sp]
        k = int(math.floor(math.log2(raw_max_seq)))
        max_seq_len = 2 ** k
        print(f"自动计算 max_seq_len: {max_sp} 卡 × {get_max_seq_for_sp(max_sp)//max_sp} tokens/卡 = {raw_max_seq}")
        print(f"k = floor(log2({raw_max_seq})) = {k}, max_seq_len = 2^{k} = {max_seq_len}")
    
    print("=" * 60)
    print(f"Ulysses SP 区间边界分析 ({config['name']})")
    print("=" * 60)
    print(f"设备显存: {device_memory_gb} GB")
    print(f"Model States: {cost_model.model_states_mb:.2f} MB")
    print(f"Base Act/Token: {cost_model.act_per_token:.4f} MB")
    print(f"n_kv_heads: {n_kv_heads}")
    print("-" * 60)
    
    for sp_size in sp_sizes:
        token_cap = cost_model.token_capacity(device_memory_gb, sp_size=sp_size, attn_type='ulysses')
        max_seq = sp_boundaries[sp_size]
        
        # 计算 KV 复制因子
        if sp_size > n_kv_heads:
            rep_factor = sp_size // n_kv_heads
            # 计算 act_per_token_effective
            extra_kv = (sp_size - n_kv_heads) * n_kv_heads * h * 2 * 2 / 1024 / 1024
            act_eff = cost_model.act_per_token + extra_kv
        else:
            rep_factor = 1
            act_eff = cost_model.act_per_token
        
        print(f"SP={sp_size:2d}: token_cap={token_cap:6d}, max_seq={max_seq:7d}, "
              f"KV复制={rep_factor}x, act_eff={act_eff:.4f} MB/token")
    
    print("=" * 60)
    
    # ==================== 确定每个序列长度需要的 SP size ====================
    def get_required_sp(seqlen: int) -> int:
        """根据序列长度确定最小需要的 SP size"""
        for sp_size in sp_sizes:
            if seqlen <= sp_boundaries[sp_size]:
                return sp_size
        return sp_sizes[-1]  # 最大 SP size
    
    # ==================== 计算 Ulysses 时间 ====================
    def compute_ulysses_time(seqlen: int, sp_size: int) -> Tuple[float, float]:
        """
        调用 cost_model 统一建模逻辑
        返回: (计算时间, 通信时间) 单位: ms
        """
        total_time = cost_model.ulysses_time_whole_model(seqlen, sp_size)
        comm_time = cost_model.ulysses_alltoall_time_per_layer(seqlen, sp_size) * cost_model.l
        compute_time = total_time - comm_time
        return compute_time, comm_time
    
    # ==================== 生成数据点 ====================
    # 为了绘制清晰的连续折线，在每个 SP size 区间内密集采样
    # 并在切换点附近更密集，以显示跳变
    
    seq_lens = []
    
    # 获取所有 SP size 切换点
    switch_points = sorted(sp_boundaries.values())
    switch_points = [s for s in switch_points if s <= max_seq_len]
    
    # 添加起点和终点
    boundaries = [1] + switch_points + [max_seq_len]
    boundaries = sorted(set(boundaries))
    
    # 在每个区间内均匀采样
    points_per_interval = 1000  # 每个区间的采样点数
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        if end <= start:
            continue
        
        # 在区间内均匀采样
        interval_points = np.linspace(start, end, points_per_interval, dtype=int)
        
        # 确保切换点被精确包含
        if i > 0:
            interval_points = np.insert(interval_points, 0, start - 1)  # 切换点前一个
        interval_points = np.append(interval_points, end)  # 切换点
        
        seq_lens.extend(interval_points)
    
    # 去重并排序
    seq_lens = sorted(set(seq_lens))
    seq_lens = [s for s in seq_lens if 1 <= s <= max_seq_len]
    seq_lens = np.array(seq_lens)
    
    print(f"生成 {len(seq_lens)} 个数据点用于绘图")
    
    total_times = []
    compute_times = []
    comm_times = []
    sp_used = []
    
    for seqlen in seq_lens:
        sp_size = get_required_sp(seqlen)
        compute_t, comm_t = compute_ulysses_time(seqlen, sp_size)
        total_times.append(compute_t + comm_t)
        compute_times.append(compute_t)
        comm_times.append(comm_t)
        sp_used.append(sp_size)
    
    total_times = np.array(total_times)
    compute_times = np.array(compute_times)
    comm_times = np.array(comm_times)
    sp_used = np.array(sp_used)
    
    # ==================== 绘图 ====================
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # ----- 主图: 时间 vs 序列长度 -----
    ax1 = axes[0]
    ax1.plot(seq_lens, total_times, 'b-', linewidth=2, label='Total Time')
    ax1.plot(seq_lens, compute_times, 'g--', linewidth=1.5, label='Compute Time')
    ax1.plot(seq_lens, comm_times, 'r--', linewidth=1.5, label='Communication Time')
    
    # 标记 SP size 切换点
    prev_sp = sp_used[0]
    switch_points = []
    for i, sp in enumerate(sp_used):
        if sp != prev_sp:
            switch_points.append((seq_lens[i], sp))
            ax1.axvline(x=seq_lens[i], color='gray', linestyle=':', alpha=0.7)
            prev_sp = sp
    
    # 在图上标注 SP size 区域
    for i, (switch_seq, sp) in enumerate(switch_points):
        y_pos = ax1.get_ylim()[1] * 0.95
        ax1.annotate(f'SP={sp}', xy=(switch_seq, y_pos), 
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title(f'Ulysses SP: Per-Device Time vs Sequence Length\n'
                  f'({config["name"]}, {device_memory_gb}GB GPU, ZeRO-3={zero3_world_size}, n_kv_heads={n_kv_heads})',
                  fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_seq_len)
    
    # ----- 副图: SP size vs 序列长度 -----
    ax2 = axes[1]
    ax2.step(seq_lens, sp_used, 'k-', linewidth=2, where='post')
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('SP Size', fontsize=12)
    ax2.set_title('Required SP Size vs Sequence Length', fontsize=14)
    ax2.set_yticks([s for s in sp_sizes if s <= max(sp_used)])
    ax2.set_yscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_seq_len)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path == "auto":
        # 自动生成文件名: ulysses_{model}_{mem}GB_{devices}gpu_{maxseq}_{timestamp}.png
        model_name = config['name'].replace(' ', '_').replace('.', '_').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ulysses_{model_name}_{int(device_memory_gb)}GB_{zero3_world_size}gpu_maxseq{max_seq_len}_{timestamp}.png"
        save_path = os.path.join(PICTURES_DIR, filename)
        
        # 确保目录存在
        os.makedirs(PICTURES_DIR, exist_ok=True)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    elif save_path:
        # 使用用户指定的路径
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    # 打印汇总信息
    print("\n" + "=" * 60)
    print("时间汇总")
    print("=" * 60)
    print(f"{'序列长度':>10} | {'SP Size':>7} | {'计算时间':>12} | {'通信时间':>12} | {'总时间':>12}")
    print("-" * 60)
    
    # 打印几个关键点
    key_points = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
    for seqlen in key_points:
        if seqlen <= max_seq_len:
            idx = np.searchsorted(seq_lens, seqlen)
            if idx < len(seq_lens):
                print(f"{seqlen:>10} | {sp_used[idx]:>7} | {compute_times[idx]:>10.2f} ms | "
                      f"{comm_times[idx]:>10.2f} ms | {total_times[idx]:>10.2f} ms")
    
    return seq_lens, total_times, compute_times, comm_times, sp_used


def ring_time_per_device(
    config: Dict = None,
    bandwidth: Dict = None,
    device_memory_gb: float = 40.0,
    zero3_world_size: int = 32,
    max_seq_len: int = None,
    save_path: str = "auto",
    show_plot: bool = True
):
    """
    绘制 Ring Attention 下单卡时间随序列长度变化的曲线
    
    Ring Attention 特点：
    1. 没有 KV 复制，token capacity 更大
    2. P2P 通信可以与计算 overlap（当计算 > 通信时）
    3. 前向/反向的 overlap 条件不同
    
    参数:
        config: 模型配置字典
        bandwidth: 带宽配置字典
        device_memory_gb: 单卡显存 (GB)
        zero3_world_size: ZeRO-3 并行度
        max_seq_len: 最大序列长度 (None 则自动计算)
        save_path: 保存路径 ("auto"=自动命名, None=不保存)
        show_plot: 是否显示图像
    """
    import matplotlib.pyplot as plt
    import math
    from datetime import datetime
    
    if config is None:
        config = QWEN_3B_CONFIG
    if bandwidth is None:
        bandwidth = BANDWIDTH_CONFIG
    
    # 创建 cost model
    cost_model = flexSPCostModel(
        cluster_size=zero3_world_size,
        hidden_size=config['hidden_size'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        layer_num=config['layer_num'],
        param_size_B=config['param_size_B'],
        act_per_token=config['act_per_token'],
        cpt_alpha1=config['cpt_alpha1'],
        cpt_alpha2=config['cpt_alpha2'],
        cpt_beta1=config['cpt_beta1'],
        alltoall_bandwidth_dict_gbs=bandwidth['alltoall'],
        p2p_bandwidth_dict_gbs=bandwidth['p2p'],
        ring_overlap_efficiency=0.15,
    )
    
    # 模型参数
    h = cost_model.h
    n_heads = cost_model.n_heads
    n_kv_heads = cost_model.n_kv_heads
    l = cost_model.l
    bwd_coe = cost_model.bwd_fwd_coe
    overlap_eff = cost_model.ring_overlap_efficiency
    
    # 可用 SP sizes
    sp_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    sp_sizes = [s for s in sp_sizes if s <= zero3_world_size and s <= n_heads]
    
    # ==================== Ring 的 Token Capacity（无 KV 复制）====================
    def get_ring_token_capacity(sp_size: int) -> int:
        """Ring Attention 没有 KV 复制，使用 base token capacity"""
        return cost_model.token_capacity(device_memory_gb, sp_size=sp_size, attn_type='ring')
    
    def get_max_seq_for_sp(sp_size: int) -> int:
        token_cap = get_ring_token_capacity(sp_size)
        return sp_size * token_cap
    
    # 计算区间边界
    sp_boundaries = {}
    for sp_size in sp_sizes:
        sp_boundaries[sp_size] = get_max_seq_for_sp(sp_size)
    
    # 自动计算 max_seq_len
    if max_seq_len is None:
        max_sp = sp_sizes[-1]
        raw_max_seq = sp_boundaries[max_sp]
        k = int(math.floor(math.log2(raw_max_seq)))
        max_seq_len = 2 ** k
        print(f"Ring: 自动计算 max_seq_len = 2^{k} = {max_seq_len}")
    
    print("=" * 70)
    print(f"Ring Attention 区间边界分析 ({config['name']})")
    print("=" * 70)
    print(f"设备显存: {device_memory_gb} GB, n_kv_heads: {n_kv_heads}")
    print(f"Ring 特点: 无 KV 复制, P2P 通信可与计算 overlap")
    print("-" * 70)
    
    for sp_size in sp_sizes:
        token_cap = get_ring_token_capacity(sp_size)
        max_seq = sp_boundaries[sp_size]
        print(f"SP={sp_size:2d}: token_cap={token_cap:6d}, max_seq={max_seq:7d}")
    
    print("=" * 70)
    
    # ==================== 确定 SP size ====================
    def get_required_sp(seqlen: int) -> int:
        for sp_size in sp_sizes:
            if seqlen <= sp_boundaries[sp_size]:
                return sp_size
        return sp_sizes[-1]
    
    # ==================== 计算 Ring 时间 ====================
    def compute_ring_time_detail(seqlen: int, sp_size: int) -> Dict:
        """
        调用 cost_model 统一建模逻辑
        """
        total_time = cost_model.zigzag_ring_flash_attention_time_whole_model(seqlen, sp_size)
        
        # 为了分解组件，我们在测试脚本中本地复现与 solver.py 一致的逻辑
        if sp_size == 1:
            compute_total = (cost_model.cpt_alpha1 * (seqlen ** 2) + 
                            cost_model.cpt_alpha2 * seqlen + 
                            cost_model.cpt_beta1) * (1 + cost_model.bwd_fwd_coe) * cost_model.l
            return {
                'compute_total': compute_total,
                'comm_total': 0.0,
                'total': compute_total,
                'fwd_compute': compute_total / (1 + cost_model.bwd_fwd_coe),
                'bwd_compute': compute_total * cost_model.bwd_fwd_coe / (1 + cost_model.bwd_fwd_coe),
                'fwd_overlap_effective': True,
                'bwd_overlap_effective': True,
                'linear_total': cost_model.cpt_alpha2 * seqlen * (1 + cost_model.bwd_fwd_coe) * cost_model.l,
                'kernel_overhead': cost_model.cpt_beta1 * (1 + cost_model.bwd_fwd_coe) * cost_model.l,
            }
        
        local_seq = seqlen / sp_size
        head_dim = cost_model.h // cost_model.n_heads
        p2p_bw = cost_model.p2p_bandwidth_dict_gbs.get(sp_size, 100)
        
        # 计算量分解 (Zigzag)
        quad_time = 0.5 * cost_model.cpt_alpha1 * (local_seq ** 2)
        attn_0 = quad_time + cost_model.cpt_alpha2 * local_seq
        attn_others = quad_time + 0.75 * cost_model.cpt_alpha2 * local_seq
        
        # 理论计算总和 (不含 bias)
        fwd_comp = attn_0 + (sp_size - 1) * attn_others
        bwd_comp = cost_model.bwd_fwd_coe * fwd_comp
        
        # 通信量 (per step)
        kv_size_mb = 2 * local_seq * cost_model.n_kv_heads * head_dim * 2 / 1024 / 1024
        comm_step = kv_size_mb / p2p_bw
        
        bias = cost_model.compute_bias(sp_size, 'ring')
        compute_total = (fwd_comp + bwd_comp) * cost_model.l + bias
        
        # 反向 overlap 判断：优化后中间步骤最大通信量为 2x comm_step (KV_bf16 + dKV_bf16)
        # 风险提醒：精度转换带来的误差累积风险
        bwd_attn_others = cost_model.bwd_fwd_coe * attn_others
        
        return {
            'compute_total': compute_total,
            'comm_total': max(0, total_time - compute_total),
            'total': total_time,
            'fwd_compute': fwd_comp * cost_model.l,
            'bwd_compute': bwd_comp * cost_model.l,
            'fwd_overlap_effective': attn_others >= comm_step,
            'bwd_overlap_effective': bwd_attn_others >= 2 * comm_step,
            'linear_total': cost_model.cpt_alpha2 * local_seq * (1 + cost_model.bwd_fwd_coe) * cost_model.l,
            'kernel_overhead': bias,
        }
    
    # ==================== 计算 overlap 临界点 ====================
    def find_overlap_critical_points(sp_size: int) -> Dict:
        """
        找到前向和反向 overlap 从通信主导变为计算主导的临界序列长度
        """
        if sp_size == 1:
            return {'fwd_critical': 0, 'bwd_critical': 0}
        
        p2p_bw = cost_model.p2p_bandwidth_dict_gbs.get(sp_size, 100)
        
        # 修正: 使用 head_dim = hidden_size / n_heads
        head_dim = h // n_heads
        
        # 前向临界点: 0.5 * alpha1 * local_seq² = comm_per_step
        # (cpt_beta1 不参与 overlap，所以不包含在比较中)
        # comm_per_step = kv_size_mb / p2p_bw (ms, 因为 GB/s = MB/ms)
        # kv_size_mb = 2 * local_seq * n_kv_heads * head_dim * 2 / 1024 / 1024
        
        # kv_coef: MB per token for K+V
        kv_coef = 2 * n_kv_heads * head_dim * 2 / 1024 / 1024  # MB per token
        
        # 前向: 0.5 * alpha1 * local_seq² = kv_coef * local_seq / p2p_bw
        # => local_seq = kv_coef / (p2p_bw * 0.5 * alpha1)
        if cost_model.cpt_alpha1 > 0:
            fwd_local_critical = kv_coef / (p2p_bw * 0.5 * cost_model.cpt_alpha1)
            fwd_critical = int(max(0, fwd_local_critical) * sp_size)
        else:
            fwd_critical = 0
        
        # 反向: bwd_coe * 0.5 * alpha1 * local_seq² = 2 * kv_coef * local_seq / p2p_bw
        # => local_seq = 2 * kv_coef / (p2p_bw * bwd_coe * 0.5 * alpha1)
        if cost_model.cpt_alpha1 > 0 and bwd_coe > 0:
            bwd_local_critical = 2 * kv_coef / (p2p_bw * bwd_coe * 0.5 * cost_model.cpt_alpha1)
            bwd_critical = int(max(0, bwd_local_critical) * sp_size)
        else:
            bwd_critical = 0
        
        return {'fwd_critical': fwd_critical, 'bwd_critical': bwd_critical}
    
    # ==================== 生成数据点 ====================
    seq_lens = []
    switch_points = sorted(sp_boundaries.values())
    switch_points = [s for s in switch_points if s <= max_seq_len]
    
    # 添加 overlap 临界点
    overlap_criticals = []
    for sp_size in sp_sizes:
        criticals = find_overlap_critical_points(sp_size)
        if 0 < criticals['fwd_critical'] <= max_seq_len:
            overlap_criticals.append(criticals['fwd_critical'])
        if 0 < criticals['bwd_critical'] <= max_seq_len:
            overlap_criticals.append(criticals['bwd_critical'])
    
    boundaries = [1] + switch_points + overlap_criticals + [max_seq_len]
    boundaries = sorted(set(boundaries))
    
    points_per_interval = 500
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        interval_points = np.linspace(start, end, points_per_interval, dtype=int)
        if i > 0:
            interval_points = np.insert(interval_points, 0, max(1, start - 1))
        interval_points = np.append(interval_points, end)
        seq_lens.extend(interval_points)
    
    seq_lens = sorted(set(seq_lens))
    seq_lens = [s for s in seq_lens if 1 <= s <= max_seq_len]
    seq_lens = np.array(seq_lens)
    
    print(f"生成 {len(seq_lens)} 个数据点用于绘图")
    
    # 计算时间
    total_times = []
    compute_times = []
    comm_times = []
    sp_used = []
    fwd_overlap_flags = []
    bwd_overlap_flags = []
    
    for seqlen in seq_lens:
        sp_size = get_required_sp(seqlen)
        result = compute_ring_time_detail(seqlen, sp_size)
        total_times.append(result['total'])
        compute_times.append(result['compute_total'])
        comm_times.append(result['total'] - result['compute_total'])
        sp_used.append(sp_size)
        fwd_overlap_flags.append(result['fwd_overlap_effective'])
        bwd_overlap_flags.append(result['bwd_overlap_effective'])
    
    total_times = np.array(total_times)
    compute_times = np.array(compute_times)
    comm_times = np.array(comm_times)
    sp_used = np.array(sp_used)
    
    # ==================== 绘图 ====================
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    
    # ----- 主图: 时间 vs 序列长度 -----
    ax1 = axes[0]
    ax1.plot(seq_lens, total_times, 'b-', linewidth=2, label='Total Time')
    ax1.plot(seq_lens, compute_times, 'g--', linewidth=1.5, label='Compute Time')
    ax1.plot(seq_lens, comm_times, 'r--', linewidth=1.5, label='Communication Time (effective)')
    
    # 标记 SP size 切换点
    prev_sp = sp_used[0]
    for i, sp in enumerate(sp_used):
        if sp != prev_sp:
            ax1.axvline(x=seq_lens[i], color='gray', linestyle=':', alpha=0.7)
            y_pos = ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else total_times.max() * 0.95
            ax1.annotate(f'SP={sp}', xy=(seq_lens[i], y_pos), 
                        fontsize=10, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            prev_sp = sp
    
    # 标记 overlap 临界点
    for sp_size in sp_sizes[1:]:  # 跳过 SP=1
        criticals = find_overlap_critical_points(sp_size)
        sp_start = sp_boundaries.get(sp_sizes[sp_sizes.index(sp_size) - 1], 0) if sp_sizes.index(sp_size) > 0 else 0
        sp_end = sp_boundaries.get(sp_size, max_seq_len)
        
        # 前向临界点
        fwd_crit = criticals['fwd_critical']
        if sp_start < fwd_crit <= sp_end and fwd_crit <= max_seq_len:
            ax1.axvline(x=fwd_crit, color='green', linestyle='--', alpha=0.5)
            ax1.annotate(f'FWD overlap\n(SP={sp_size})', xy=(fwd_crit, total_times.max() * 0.7),
                        fontsize=8, ha='center', color='green')
        
        # 反向临界点
        bwd_crit = criticals['bwd_critical']
        if sp_start < bwd_crit <= sp_end and bwd_crit <= max_seq_len:
            ax1.axvline(x=bwd_crit, color='red', linestyle='--', alpha=0.5)
            ax1.annotate(f'BWD overlap\n(SP={sp_size})', xy=(bwd_crit, total_times.max() * 0.5),
                        fontsize=8, ha='center', color='red')
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title(f'Ring Attention: Per-Device Time vs Sequence Length\n'
                  f'({config["name"]}, {device_memory_gb}GB GPU, ZeRO-3={zero3_world_size})',
                  fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_seq_len)
    
    # ----- 副图: SP size -----
    ax2 = axes[1]
    ax2.step(seq_lens, sp_used, 'k-', linewidth=2, where='post')
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('SP Size', fontsize=12)
    ax2.set_title('Required SP Size vs Sequence Length', fontsize=14)
    ax2.set_yticks([s for s in sp_sizes if s <= max(sp_used)])
    ax2.set_yscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_seq_len)
    
    # ----- Overlap 状态图 -----
    ax3 = axes[2]
    # 将布尔转为数值：0=通信主导, 1=计算主导
    fwd_overlap_numeric = np.array([1 if f else 0 for f in fwd_overlap_flags])
    bwd_overlap_numeric = np.array([1 if b else 0 for b in bwd_overlap_flags])
    
    ax3.fill_between(seq_lens, 0, fwd_overlap_numeric, alpha=0.3, color='green', label='FWD: Compute Dominant')
    ax3.fill_between(seq_lens, 0, -bwd_overlap_numeric, alpha=0.3, color='red', label='BWD: Compute Dominant')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Sequence Length', fontsize=12)
    ax3.set_ylabel('Overlap Effective', fontsize=12)
    ax3.set_title('Overlap Status (1=Compute Dominant, can overlap; 0=Comm Dominant, cannot overlap)', fontsize=12)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['BWD Compute', 'Comm Dominant', 'FWD Compute'])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_seq_len)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path == "auto":
        model_name = config['name'].replace(' ', '_').replace('.', '_').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ring_{model_name}_{int(device_memory_gb)}GB_{zero3_world_size}gpu_maxseq{max_seq_len}_{timestamp}.png"
        save_path = os.path.join(PICTURES_DIR, filename)
        os.makedirs(PICTURES_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    elif save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("Ring Attention 时间汇总")
    print("=" * 80)
    print(f"{'序列长度':>10} | {'SP':>4} | {'计算时间':>12} | {'通信时间':>12} | {'总时间':>12} | {'FWD OL':>7} | {'BWD OL':>7}")
    print("-" * 80)
    
    key_points = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
    for seqlen in key_points:
        if seqlen <= max_seq_len:
            idx = np.searchsorted(seq_lens, seqlen)
            if idx < len(seq_lens):
                fwd_ol = "Yes" if fwd_overlap_flags[idx] else "No"
                bwd_ol = "Yes" if bwd_overlap_flags[idx] else "No"
                print(f"{seqlen:>10} | {sp_used[idx]:>4} | {compute_times[idx]:>10.2f} ms | "
                      f"{comm_times[idx]:>10.2f} ms | {total_times[idx]:>10.2f} ms | {fwd_ol:>7} | {bwd_ol:>7}")
    
    return seq_lens, total_times, compute_times, comm_times, sp_used


def compare_ulysses_ring(
    config: Dict = None,
    bandwidth: Dict = None,
    device_memory_gb: float = 40.0,
    zero3_world_size: int = 32,
    max_seq_len: int = None,
    save_path: str = "auto",
    show_plot: bool = True
):
    """
    在同一张图上对比 Ulysses 和 Ring Attention
    """
    import matplotlib.pyplot as plt
    import math
    from datetime import datetime
    
    if config is None:
        config = QWEN_3B_CONFIG
    if bandwidth is None:
        bandwidth = BANDWIDTH_CONFIG
    
    # 创建 cost model
    cost_model = flexSPCostModel(
        cluster_size=zero3_world_size,
        hidden_size=config['hidden_size'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        layer_num=config['layer_num'],
        param_size_B=config['param_size_B'],
        act_per_token=config['act_per_token'],
        cpt_alpha1=config['cpt_alpha1'],
        cpt_alpha2=config['cpt_alpha2'],
        cpt_beta1=config['cpt_beta1'],
        alltoall_bandwidth_dict_gbs=bandwidth['alltoall'],
        p2p_bandwidth_dict_gbs=bandwidth['p2p'],
        ring_overlap_efficiency=0.12,
    )
    
    h = cost_model.h
    n_heads = cost_model.n_heads
    n_kv_heads = cost_model.n_kv_heads
    l = cost_model.l
    bwd_coe = cost_model.bwd_fwd_coe
    overlap_eff = cost_model.ring_overlap_efficiency
    
    sp_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    sp_sizes = [s for s in sp_sizes if s <= zero3_world_size and s <= n_heads]
    
    # ==================== Ulysses 边界 ====================
    def get_ulysses_max_seq(sp_size):
        token_cap = cost_model.token_capacity(device_memory_gb, sp_size=sp_size, attn_type='ulysses')
        return sp_size * token_cap
    
    ulysses_boundaries = {sp: get_ulysses_max_seq(sp) for sp in sp_sizes}
    
    # ==================== Ring 边界 ====================
    def get_ring_max_seq(sp_size):
        token_cap = cost_model.token_capacity(device_memory_gb, sp_size=sp_size, attn_type='ring')
        return sp_size * token_cap
    
    ring_boundaries = {sp: get_ring_max_seq(sp) for sp in sp_sizes}
    
    # 自动计算 max_seq_len
    if max_seq_len is None:
        max_sp = sp_sizes[-1]
        raw_max = max(ulysses_boundaries[max_sp], ring_boundaries[max_sp])
        k = int(math.floor(math.log2(raw_max)))
        max_seq_len = 2 ** k
    
    print("=" * 70)
    print(f"Ulysses vs Ring 对比 ({config['name']})")
    print("=" * 70)
    print(f"{'SP Size':>8} | {'Ulysses max_seq':>15} | {'Ring max_seq':>15}")
    print("-" * 50)
    for sp in sp_sizes:
        print(f"{sp:>8} | {ulysses_boundaries[sp]:>15} | {ring_boundaries[sp]:>15}")
    print("=" * 70)
    
    # ==================== 计算时间 ====================
    def get_ulysses_sp(seqlen):
        for sp in sp_sizes:
            if seqlen <= ulysses_boundaries[sp]:
                return sp
        return sp_sizes[-1]
    
    def get_ring_sp(seqlen):
        for sp in sp_sizes:
            if seqlen <= ring_boundaries[sp]:
                return sp
        return sp_sizes[-1]
    
    def compute_ulysses_time(seqlen, sp_size):
        """Ulysses 时间计算，直接引用 cost_model"""
        total = cost_model.ulysses_time_whole_model(seqlen, sp_size)
        comm = cost_model.ulysses_alltoall_time_per_layer(seqlen, sp_size) * cost_model.l
        return total - comm, comm
    
    def compute_ring_time(seqlen, sp_size):
        """Ring Attention 时间计算，直接引用 cost_model"""
        total = cost_model.zigzag_ring_flash_attention_time_whole_model(seqlen, sp_size)
        
        # 估算理论计算时间以提取有效通信
        if sp_size == 1:
            comp = (cost_model.cpt_alpha1 * (seqlen ** 2) + 
                    cost_model.cpt_alpha2 * seqlen + 
                    cost_model.cpt_beta1) * (1 + cost_model.bwd_fwd_coe) * cost_model.l
            return comp, 0.0
            
        local_seq = seqlen / sp_size
        quad = 0.5 * cost_model.cpt_alpha1 * (local_seq ** 2)
        attn_0 = quad + cost_model.cpt_alpha2 * local_seq
        attn_others = quad + 0.75 * cost_model.cpt_alpha2 * local_seq
        fwd_comp = attn_0 + (sp_size - 1) * attn_others
        bias = cost_model.compute_bias(sp_size, 'ring')
        compute_theoretical = fwd_comp * (1 + cost_model.bwd_fwd_coe) * cost_model.l + bias
        
        return compute_theoretical, max(0, total - compute_theoretical)
    
    # 生成数据点
    all_boundaries = sorted(set(list(ulysses_boundaries.values()) + list(ring_boundaries.values())))
    all_boundaries = [b for b in all_boundaries if b <= max_seq_len]
    boundaries = [1] + all_boundaries + [max_seq_len]
    boundaries = sorted(set(boundaries))
    
    seq_lens = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end > start:
            seq_lens.extend(np.linspace(start, end, 300, dtype=int))
    seq_lens = sorted(set(seq_lens))
    seq_lens = np.array([s for s in seq_lens if 1 <= s <= max_seq_len])
    
    # 计算
    ulysses_times = []
    ring_times = []
    ulysses_sp = []
    ring_sp = []
    
    for seqlen in seq_lens:
        u_sp = get_ulysses_sp(seqlen)
        r_sp = get_ring_sp(seqlen)
        
        u_compute, u_comm = compute_ulysses_time(seqlen, u_sp)
        r_compute, r_comm = compute_ring_time(seqlen, r_sp)
        
        ulysses_times.append(u_compute + u_comm)
        ring_times.append(r_compute + r_comm)
        ulysses_sp.append(u_sp)
        ring_sp.append(r_sp)
    
    ulysses_times = np.array(ulysses_times)
    ring_times = np.array(ring_times)
    
    # ==================== 绘图 ====================
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1 = axes[0]
    ax1.plot(seq_lens, ulysses_times, 'b-', linewidth=2, label='Ulysses SP')
    ax1.plot(seq_lens, ring_times, 'r-', linewidth=2, label='Ring Attention')
    
    # 标记 Ring 更优的区域
    better_ring = ring_times < ulysses_times
    ax1.fill_between(seq_lens, 0, ulysses_times.max(), where=better_ring, 
                     alpha=0.1, color='red', label='Ring Better')
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title(f'Ulysses vs Ring Attention: Per-Device Time Comparison\n'
                  f'({config["name"]}, {device_memory_gb}GB GPU, ZeRO-3={zero3_world_size})',
                  fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_seq_len)
    
    # 加速比
    ax2 = axes[1]
    speedup = ulysses_times / np.maximum(ring_times, 1e-6)
    ax2.plot(seq_lens, speedup, 'g-', linewidth=2)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(seq_lens, 1.0, speedup, where=(speedup > 1), alpha=0.3, color='green', label='Ring Faster')
    ax2.fill_between(seq_lens, 1.0, speedup, where=(speedup < 1), alpha=0.3, color='red', label='Ulysses Faster')
    
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Speedup (Ulysses/Ring)', fontsize=12)
    ax2.set_title('Ring Attention Speedup over Ulysses', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_seq_len)
    
    plt.tight_layout()
    
    # 保存
    if save_path == "auto":
        model_name = config['name'].replace(' ', '_').replace('.', '_').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"compare_ulysses_ring_{model_name}_{int(device_memory_gb)}GB_{zero3_world_size}gpu_{timestamp}.png"
        save_path = os.path.join(PICTURES_DIR, filename)
        os.makedirs(PICTURES_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    elif save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    # 打印汇总
    print("\n" + "=" * 90)
    print("Ulysses vs Ring 时间对比")
    print("=" * 90)
    print(f"{'序列长度':>10} | {'Ulysses SP':>10} | {'Ring SP':>8} | {'Ulysses(ms)':>12} | {'Ring(ms)':>12} | {'加速比':>8}")
    print("-" * 90)
    
    key_points = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
    for seqlen in key_points:
        if seqlen <= max_seq_len:
            idx = np.searchsorted(seq_lens, seqlen)
            if idx < len(seq_lens):
                sp_ratio = ulysses_times[idx] / ring_times[idx] if ring_times[idx] > 0 else 0
                print(f"{seqlen:>10} | {ulysses_sp[idx]:>10} | {ring_sp[idx]:>8} | "
                      f"{ulysses_times[idx]:>10.2f} ms | {ring_times[idx]:>10.2f} ms | {sp_ratio:>8.2f}x")
    
    return seq_lens, ulysses_times, ring_times


def analyze_overlap_conditions(
    config: Dict = None,
    bandwidth: Dict = None,
    zero3_world_size: int = 32,
):
    """
    分析 Ring Attention 的 overlap 条件
    
    解答问题：如何让 attention 计算量变大，从而可以 overlap 通信？
    """
    if config is None:
        config = QWEN_3B_CONFIG
    if bandwidth is None:
        bandwidth = BANDWIDTH_CONFIG
    
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    h = config['hidden_size']
    head_dim = h // n_heads
    alpha1 = config['cpt_alpha1']
    
    print("=" * 100)
    print("Ring Attention Overlap 条件分析")
    print("=" * 100)
    print(f"\n模型配置: {config['name']}")
    print(f"  n_heads = {n_heads}, n_kv_heads = {n_kv_heads}, head_dim = {head_dim}")
    print(f"  hidden_size = {h}")
    print(f"  cpt_alpha1 = {alpha1:.2e} ms/token²")
    
    print("\n" + "=" * 100)
    print("Overlap 临界条件推导")
    print("=" * 100)
    
    print("""
对于 Ring Attention 的前向传播中间步（half attention）：
  
  计算时间 = 0.5 * α₁ * local_seq²
  通信时间 = kv_size / p2p_bw = (2 * local_seq * n_kv_heads * head_dim * 2 bytes) / p2p_bw

Overlap 有效的条件：计算时间 > 通信时间

  0.5 * α₁ * local_seq² > 2 * local_seq * n_kv_heads * head_dim * 2 / 1024² / p2p_bw

简化（两边除以 local_seq）：

  0.5 * α₁ * local_seq > kv_coef / p2p_bw

其中 kv_coef = 4 * n_kv_heads * head_dim / 1024² (MB/token)

所以：
  
  local_seq_critical = kv_coef / (0.5 * α₁ * p2p_bw)
  total_seq_critical = local_seq_critical * sp_size
""")
    
    # 计算不同 SP size 下的临界序列长度
    # 前向临界条件 (Zigzag Others): 0.5 * alpha1 * (S/sp)^2 + 0.75 * alpha2 * (S/sp) >= kv_coef * (S/sp) / p2p_bw
    # 简化为: 0.5 * alpha1 * (S/sp) + 0.75 * alpha2 >= kv_coef / p2p_bw
    kv_coef = 4 * n_kv_heads * head_dim / 1024 / 1024  # MB/token
    print(f"kv_coef = 4 * {n_kv_heads} * {head_dim} / 1024² = {kv_coef:.6f} MB/token")
    
    sp_sizes = [2, 4, 8, 16, 32]
    sp_sizes = [s for s in sp_sizes if s <= zero3_world_size and s <= n_heads]
    
    print(f"\n{'SP Size':>8} | {'P2P BW (GB/s)':>14} | {'临界 local_seq':>16} | {'临界 total_seq':>16} | {'是否可行':>10}")
    print("-" * 80)
    
    for sp_size in sp_sizes:
        p2p_bw = bandwidth['p2p'].get(sp_size, 100)  # GB/s = MB/ms
        
        if alpha1 > 0:
            rhs = kv_coef / p2p_bw - 0.75 * config.get('cpt_alpha2', 0.0)
            local_critical = rhs / (0.5 * alpha1)
            total_critical = local_critical * sp_size
            
            # 检查是否可行（在合理的序列长度范围内）
            feasible = "✓ 可行" if total_critical < 262144 else "✗ 序列太长"
        else:
            local_critical = float('inf')
            total_critical = float('inf')
            feasible = "✗ α₁=0"
        
        print(f"{sp_size:>8} | {p2p_bw:>14.2f} | {max(0, local_critical):>16.0f} | {max(0, total_critical):>16.0f} | {feasible:>10}")
    
    print("\n" + "=" * 100)
    print("如何让 Ring Attention 的 Overlap 更有效？")
    print("=" * 100)
    
    print("""
从公式 local_seq_critical = kv_coef / (0.5 * α₁ * p2p_bw) 可以得出：

1. 【增大 α₁】→ 使用更大的模型
   - 更大的模型有更密集的 attention 计算
   - α₁ 与 head_dim 和计算复杂度相关
   - 例如：Qwen-7B, Qwen-14B, Qwen-72B 有更大的 α₁

2. 【减小 kv_coef】→ 使用更激进的 GQA/MQA
   - kv_coef = 4 * n_kv_heads * head_dim / 1024²
   - 减少 n_kv_heads 会线性减少通信量
   - 例如：n_kv_heads = 1 (MQA) 比 n_kv_heads = 2 通信量减半

3. 【增大 p2p_bw】→ 使用更快的网络
   - 节点内使用 NVLink (例如 300+ GB/s)
   - 限制 SP size 以避免跨节点通信
   - 例如：SP=8 节点内可能有 100+ GB/s，但 SP=32 跨节点只有 10 GB/s

4. 【增大 local_seq】→ 使用更小的 SP size
   - 每个 rank 处理更长的序列
   - 这受限于显存，但 Ring Attention 没有 KV 复制，显存效率更高
   - 策略：在显存允许的情况下，使用尽可能小的 SP size
""")
    
    # 分析不同模型规模的影响
    print("\n" + "=" * 100)
    print("不同模型规模的 Overlap 临界点对比")
    print("=" * 100)
    
    model_configs = [
        ("Qwen-3B (当前)", 16, 2, 128, 2.73e-9),
        ("Qwen-7B (估计)", 28, 4, 128, 3.5e-9),
        ("Qwen-14B (估计)", 40, 8, 128, 4.5e-9),
        ("Qwen-72B (估计)", 64, 8, 128, 6.0e-9),
    ]
    
    print(f"{'模型':>20} | {'n_heads':>8} | {'n_kv_heads':>10} | {'head_dim':>8} | {'α₁':>12} | {'SP=8 临界seq':>14}")
    print("-" * 90)
    
    for name, nh, nkv, hdim, a1 in model_configs:
        kv_c = 4 * nkv * hdim / 1024 / 1024
        p2p = bandwidth['p2p'].get(8, 100)
        if a1 > 0:
            crit = kv_c / (0.5 * a1 * p2p) * 8
        else:
            crit = float('inf')
        print(f"{name:>20} | {nh:>8} | {nkv:>10} | {hdim:>8} | {a1:>12.2e} | {crit:>14.0f}")
    
    print("\n结论：")
    print("  - 更大的模型（α₁ 更大）overlap 临界点更低，更容易实现有效 overlap")
    print("  - GQA 的 n_kv_heads 越小，通信量越小，overlap 更容易")
    print("  - Ring Attention 在长序列 + 大模型 + 高带宽场景下优势明显")
    print("  - 短序列场景下，Ulysses 可能更优（因为 Ring 无法 overlap）")
    
    return {
        'kv_coef': kv_coef,
        'sp_critical_seqlens': {sp: kv_coef / (0.5 * alpha1 * bandwidth['p2p'].get(sp, 100)) * sp 
                                for sp in sp_sizes if alpha1 > 0}
    }


def compare_sp_boundaries(
    config: Dict = None,
    bandwidth: Dict = None,
    device_memory_gb: float = 40.0,
    zero3_world_size: int = 32,
    max_seq_len: int = None,
    save_path: str = "auto",
    show_plot: bool = True
):
    """
    对比 Ulysses 和 Ring Attention 的 SP size 切换点
    
    展示：
    1. 两者的 token_capacity 差异（由于 GQA KV 复制）
    2. 同一序列长度需要的 SP size 差异
    3. SP size 区间对比图
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import math
    from datetime import datetime
    
    if config is None:
        config = QWEN_3B_CONFIG
    if bandwidth is None:
        bandwidth = BANDWIDTH_CONFIG
    
    # 创建 cost model
    cost_model = flexSPCostModel(
        cluster_size=zero3_world_size,
        hidden_size=config['hidden_size'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        layer_num=config['layer_num'],
        param_size_B=config['param_size_B'],
        act_per_token=config['act_per_token'],
        cpt_alpha1=config['cpt_alpha1'],
        cpt_alpha2=config.get('cpt_alpha2', 0.0),
        cpt_beta1=config['cpt_beta1'],
        alltoall_bandwidth_dict_gbs=bandwidth['alltoall'],
        p2p_bandwidth_dict_gbs=bandwidth['p2p'],
        ring_overlap_efficiency=0.15,
    )
    
    h = cost_model.h
    n_heads = cost_model.n_heads
    n_kv_heads = cost_model.n_kv_heads
    head_dim = h // n_heads
    
    # 可用 SP sizes
    sp_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    sp_sizes = [s for s in sp_sizes if s <= zero3_world_size and s <= n_heads]
    
    print("=" * 100)
    print(f"Ulysses vs Ring Attention: SP Size 切换点对比 ({config['name']})")
    print("=" * 100)
    print(f"设备显存: {device_memory_gb} GB")
    print(f"Model States: {cost_model.model_states_mb:.2f} MB")
    print(f"Base Act/Token: {cost_model.act_per_token:.4f} MB")
    print(f"n_heads: {n_heads}, n_kv_heads: {n_kv_heads}, head_dim: {head_dim}")
    print("-" * 100)
    
    # ==================== 计算边界 ====================
    ulysses_boundaries = {}  # SP size -> max_seq
    ring_boundaries = {}
    ulysses_token_caps = {}  # SP size -> token_capacity
    ring_token_caps = {}
    
    for sp_size in sp_sizes:
        # Ulysses
        u_token_cap = cost_model.token_capacity(device_memory_gb, sp_size=sp_size, attn_type='ulysses')
        ulysses_token_caps[sp_size] = u_token_cap
        ulysses_boundaries[sp_size] = sp_size * u_token_cap
        
        # Ring
        r_token_cap = cost_model.token_capacity(device_memory_gb, sp_size=sp_size, attn_type='ring')
        ring_token_caps[sp_size] = r_token_cap
        ring_boundaries[sp_size] = sp_size * r_token_cap
    
    # ==================== 打印详细对比 ====================
    print("\n" + "=" * 100)
    print("Token Capacity 和 Max Sequence Length 对比")
    print("=" * 100)
    print(f"{'SP':>4} | {'Ulysses token_cap':>18} | {'Ring token_cap':>16} | "
          f"{'Ulysses max_seq':>16} | {'Ring max_seq':>14} | {'KV复制':>8}")
    print("-" * 100)
    
    for sp_size in sp_sizes:
        kv_rep = sp_size // n_kv_heads if sp_size > n_kv_heads else 1
        print(f"{sp_size:>4} | {ulysses_token_caps[sp_size]:>18} | {ring_token_caps[sp_size]:>16} | "
              f"{ulysses_boundaries[sp_size]:>16} | {ring_boundaries[sp_size]:>14} | {kv_rep:>6}x")
    
    # ==================== 计算 SP size 区间 ====================
    print("\n" + "=" * 100)
    print("SP Size 使用区间对比")
    print("=" * 100)
    
    def get_sp_intervals(boundaries, sp_sizes):
        """计算每个 SP size 对应的序列长度区间"""
        intervals = {}
        prev_max = 0
        for sp_size in sp_sizes:
            max_seq = boundaries[sp_size]
            if max_seq > prev_max:
                intervals[sp_size] = (prev_max + 1, max_seq)
                prev_max = max_seq
            else:
                intervals[sp_size] = None  # 这个 SP size 不会被使用
        return intervals
    
    ulysses_intervals = get_sp_intervals(ulysses_boundaries, sp_sizes)
    ring_intervals = get_sp_intervals(ring_boundaries, sp_sizes)
    
    print(f"\n{'SP':>4} | {'Ulysses 区间':>30} | {'Ring 区间':>30}")
    print("-" * 70)
    
    for sp_size in sp_sizes:
        u_interval = ulysses_intervals.get(sp_size)
        r_interval = ring_intervals.get(sp_size)
        
        u_str = f"[{u_interval[0]:>7}, {u_interval[1]:>7}]" if u_interval else "不使用"
        r_str = f"[{r_interval[0]:>7}, {r_interval[1]:>7}]" if r_interval else "不使用"
        
        print(f"{sp_size:>4} | {u_str:>30} | {r_str:>30}")
    
    # ==================== 计算 max_seq_len ====================
    if max_seq_len is None:
        max_sp = sp_sizes[-1]
        raw_max = max(ulysses_boundaries[max_sp], ring_boundaries[max_sp])
        k = int(math.floor(math.log2(raw_max)))
        max_seq_len = 2 ** k
        print(f"\n自动计算 max_seq_len = 2^{k} = {max_seq_len}")
    
    # ==================== 分析差异 ====================
    print("\n" + "=" * 100)
    print("关键序列长度的 SP Size 选择差异")
    print("=" * 100)
    
    def get_required_sp(seqlen, boundaries, sp_sizes):
        for sp in sp_sizes:
            if seqlen <= boundaries[sp]:
                return sp
        return sp_sizes[-1]
    
    # 找出差异点
    difference_points = []
    test_seqlens = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    test_seqlens = [s for s in test_seqlens if s <= max_seq_len]
    
    print(f"{'序列长度':>10} | {'Ulysses SP':>12} | {'Ring SP':>10} | {'差异':>15}")
    print("-" * 55)
    
    for seqlen in test_seqlens:
        u_sp = get_required_sp(seqlen, ulysses_boundaries, sp_sizes)
        r_sp = get_required_sp(seqlen, ring_boundaries, sp_sizes)
        
        diff = "相同" if u_sp == r_sp else f"Ulysses需要{u_sp//r_sp}x更大SP"
        if u_sp != r_sp:
            difference_points.append((seqlen, u_sp, r_sp))
        
        print(f"{seqlen:>10} | {u_sp:>12} | {r_sp:>10} | {diff:>15}")
    
    # ==================== 通信量对比 ====================
    print("\n" + "=" * 120)
    print("通信量对比分析 (单层，前向+反向，每个 rank 的通信量)")
    print("=" * 120)
    print("\n说明:")
    print("  - Ulysses: 每个 rank 通信量 = (Q+K+V+O) * (SP-1) / SP² * 2 (fwd+bwd)")
    print("  - Ring: 每个 rank 通信量 = KV * [(SP-1) + 2*SP + 1] (fwd+bwd)")
    print("  - GQA 下 Ulysses 需要复制 KV，Ring 不需要")
    print("=" * 120)
    
    def compute_ulysses_comm_volume(seqlen: int, sp_size: int) -> float:
        """
        计算 Ulysses AlltoAll 的总通信量 (MB, 单层, 前向+反向)
        调用 cost_model 统一逻辑 (已包含 2x 因子)
        """
        if sp_size == 1:
            return 0.0
        v = cost_model.alltoall_bandwidth_dict_gbs[sp_size]
        # solver.py 中 ulysses_alltoall_time_per_layer 返回的是 ms，乘以带宽得到 MB
        return cost_model.ulysses_alltoall_time_per_layer(seqlen, sp_size) * v
        
        # Q+O tensor: 2 * seqlen * hidden_size * 2 bytes (bf16)
    def compute_ring_comm_volume(seqlen: int, sp_size: int) -> float:
        """
        计算 Ring Attention P2P 的总通信量 (MB)
        包含 FP32 梯度的开销和 Stack 导致的提升
        NOTE: 优化方案 - 发送前转为 BF16 传输
        风险提醒：环路梯度累加过程中频繁精度转换可能导致误差累积
        """
        if sp_size == 1:
            return 0.0
        
        local_seq = seqlen / sp_size
        head_dim = cost_model.h // cost_model.n_heads
        kv_per_step = 2 * local_seq * cost_model.n_kv_heads * head_dim * 2 / 1024 / 1024  # MB (bf16)
        
        if sp_size == 2:
            # FWD: 1x, BWD: (1 + 1 + 1) = 3x (dKV 已转为 BF16)
            multipliers = (sp_size - 1) + 3
        else:
            # FWD: sp-1, BWD: 1 + (sp-2)*2 + 1 + 1 (dKV 已转为 BF16)
            multipliers = (sp_size - 1) + (1 + (sp_size - 2) * 2 + 1 + 1)
            
        return multipliers * kv_per_step
    
    print(f"\n{'序列长度':>10} | {'Ulysses SP':>12} | {'Ring SP':>10} | "
          f"{'Ulysses通信(MB)':>18} | {'Ring通信(MB)':>16} | {'通信量比':>12}")
    print("-" * 120)
    
    comm_comparisons = []
    for seqlen in test_seqlens:
        u_sp = get_required_sp(seqlen, ulysses_boundaries, sp_sizes)
        r_sp = get_required_sp(seqlen, ring_boundaries, sp_sizes)
        
        u_comm = compute_ulysses_comm_volume(seqlen, u_sp)
        r_comm = compute_ring_comm_volume(seqlen, r_sp)
        
        comm_ratio = u_comm / r_comm if r_comm > 0 else float('inf')
        comm_comparisons.append((seqlen, u_sp, r_sp, u_comm, r_comm, comm_ratio))
        
        print(f"{seqlen:>10} | {u_sp:>12} | {r_sp:>10} | "
              f"{u_comm:>16.2f} MB | {r_comm:>14.2f} MB | {comm_ratio:>10.2f}x")
    
    # ==================== 通信量详细分解示例 ====================
    print("\n" + "=" * 120)
    print("通信量详细分解示例 (序列长度 32768)")
    print("=" * 120)
    
    example_seq = 32768
    if example_seq <= max_seq_len:
        u_sp_ex = get_required_sp(example_seq, ulysses_boundaries, sp_sizes)
        r_sp_ex = get_required_sp(example_seq, ring_boundaries, sp_sizes)
        
        print(f"\n--- Ulysses SP={u_sp_ex} ---")
        print("\n【步骤 1: 计算原始 Tensor 大小】")
        
        # Q+O 原始大小
        qo_size_total = 2 * example_seq * h * 2 / 1024 / 1024  # MB
        print(f"Q tensor: {example_seq} * {h} * 2 bytes = {qo_size_total/2:.2f} MB")
        print(f"O tensor: {example_seq} * {h} * 2 bytes = {qo_size_total/2:.2f} MB")
        print(f"Q+O 合计: {qo_size_total:.2f} MB")
        
        # K+V 原始大小
        kv_size_total = 2 * example_seq * n_kv_heads * head_dim * 2 / 1024 / 1024  # MB
        print(f"\nK tensor: {example_seq} * {n_kv_heads} * {head_dim} * 2 bytes = {kv_size_total/2:.2f} MB")
        print(f"V tensor: {example_seq} * {n_kv_heads} * {head_dim} * 2 bytes = {kv_size_total/2:.2f} MB")
        print(f"K+V 合计: {kv_size_total:.2f} MB")
        
        # KV 复制
        if u_sp_ex > n_kv_heads:
            kv_rep = u_sp_ex // n_kv_heads
            kv_size_total_with_rep = kv_size_total * kv_rep
            print(f"\nGQA 复制: SP={u_sp_ex} > n_kv_heads={n_kv_heads}")
            print(f"K+V (复制 {kv_rep}x): {kv_size_total:.2f} * {kv_rep} = {kv_size_total_with_rep:.2f} MB")
        else:
            kv_size_total_with_rep = kv_size_total
        
        total_tensor_size = qo_size_total + kv_size_total_with_rep
        print(f"\n【Q+K+V+O 总大小】: {qo_size_total:.2f} + {kv_size_total_with_rep:.2f} = {total_tensor_size:.2f} MB")
        
        print(f"\n【步骤 2: 计算 AlltoAll 通信量】")
        print(f"每个 rank 的通信量公式: total_size * (SP-1) / SP²")
        print(f"                      = {total_tensor_size:.2f} * ({u_sp_ex}-1) / {u_sp_ex}²")
        
        alltoall_fwd = total_tensor_size * (u_sp_ex - 1) / u_sp_ex / u_sp_ex
        print(f"                      = {alltoall_fwd:.2f} MB (前向 4 次 AlltoAll)")
        
        print(f"\n【步骤 3: 前向+反向】")
        print(f"前向: Q scatter + K scatter + V scatter + O gather = 4 次")
        print(f"反向: dO scatter + dQ gather + dK gather + dV gather = 4 次")
        print(f"前向通信量: {alltoall_fwd:.2f} MB")
        print(f"反向通信量: {alltoall_fwd:.2f} MB (与前向相同)")
        print(f"总通信量: {alltoall_fwd:.2f} * 2 = {alltoall_fwd * 2:.2f} MB/层")
        
        print(f"\n--- Ring Attention SP={r_sp_ex} ---")
        
        local_seq = example_seq / r_sp_ex
        kv_per_step = 2 * local_seq * n_kv_heads * head_dim * 2 / 1024 / 1024
        print(f"local_seq: {example_seq} / {r_sp_ex} = {local_seq:.0f}")
        print(f"KV per step: 2 * {local_seq:.0f} * {n_kv_heads} * {head_dim} * 2 bytes = {kv_per_step:.2f} MB")
        
        fwd_steps = r_sp_ex - 1
        fwd_comm = fwd_steps * kv_per_step
        print(f"前向传输: {fwd_steps} 步 * {kv_per_step:.2f} MB/步 = {fwd_comm:.2f} MB")
        
        bwd_steps = r_sp_ex
        bwd_comm = bwd_steps * 2 * kv_per_step + kv_per_step
        print(f"反向传输: {bwd_steps} 步 * (2 * {kv_per_step:.2f}) MB + {kv_per_step:.2f} MB (最后dKV) = {bwd_comm:.2f} MB")
        
        ring_total = fwd_comm + bwd_comm
        print(f"总通信量: {fwd_comm:.2f} + {bwd_comm:.2f} = {ring_total:.2f} MB/层")
        
        print(f"\n{'='*60}")
        print(f"【对比总结 (序列长度 {example_seq})】")
        print(f"{'='*60}")
        print(f"Ulysses SP={u_sp_ex}: {alltoall_fwd * 2:.2f} MB/层")
        print(f"Ring SP={r_sp_ex}:    {ring_total:.2f} MB/层")
        print(f"通信量比: {(alltoall_fwd * 2) / ring_total:.2f}x")
        print(f"\nUlysses 通信量更大的原因:")
        print(f"  1. AlltoAll 需要传输 Q+K+V+O")
        print(f"  2. Ring 只需要传输 K+V")
        print(f"  3. Ulysses 在 GQA 下需要复制 KV ({kv_rep}x)" if u_sp_ex > n_kv_heads else "  3. Ulysses 没有 KV 复制")
        print(f"  4. Ulysses 使用了更大的 SP size ({u_sp_ex} vs {r_sp_ex})")
    
    # ==================== 绘图 ====================
    fig, axes = plt.subplots(3, 1, figsize=(16, 15))
    
    # ----- 图1: SP Size vs 序列长度 -----
    ax1 = axes[0]
    
    # 生成数据点
    seq_lens = np.linspace(1, max_seq_len, 2000, dtype=int)
    ulysses_sp_used = [get_required_sp(s, ulysses_boundaries, sp_sizes) for s in seq_lens]
    ring_sp_used = [get_required_sp(s, ring_boundaries, sp_sizes) for s in seq_lens]
    
    ax1.step(seq_lens, ulysses_sp_used, 'b-', linewidth=2.5, label='Ulysses SP', where='post')
    ax1.step(seq_lens, ring_sp_used, 'r--', linewidth=2.5, label='Ring Attention SP', where='post')
    
    # 标记切换点
    for sp_size in sp_sizes[1:]:  # 跳过 SP=1
        # Ulysses 切换点
        u_switch = ulysses_boundaries.get(sp_sizes[sp_sizes.index(sp_size) - 1], 0)
        if 0 < u_switch <= max_seq_len:
            ax1.axvline(x=u_switch, color='blue', linestyle=':', alpha=0.4)
        
        # Ring 切换点
        r_switch = ring_boundaries.get(sp_sizes[sp_sizes.index(sp_size) - 1], 0)
        if 0 < r_switch <= max_seq_len:
            ax1.axvline(x=r_switch, color='red', linestyle=':', alpha=0.4)
    
    # 标注差异区域
    sp_diff = np.array(ulysses_sp_used) - np.array(ring_sp_used)
    ax1.fill_between(seq_lens, 0, sp_sizes[-1], where=(sp_diff > 0),
                     alpha=0.15, color='orange', label='Ulysses needs larger SP')
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('SP Size', fontsize=12)
    ax1.set_title(f'SP Size Selection: Ulysses vs Ring Attention\n'
                  f'({config["name"]}, {device_memory_gb}GB, n_kv_heads={n_kv_heads})',
                  fontsize=14)
    ax1.set_yscale('log', base=2)
    ax1.set_yticks(sp_sizes)
    ax1.set_yticklabels([str(s) for s in sp_sizes])
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_seq_len)
    ax1.set_ylim(0.8, sp_sizes[-1] * 1.2)
    
    # ----- 图2: Token Capacity 对比 (条形图) -----
    ax2 = axes[1]
    
    x = np.arange(len(sp_sizes))
    width = 0.35
    
    ulysses_caps = [ulysses_token_caps[sp] for sp in sp_sizes]
    ring_caps = [ring_token_caps[sp] for sp in sp_sizes]
    
    bars1 = ax2.bar(x - width/2, ulysses_caps, width, label='Ulysses', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, ring_caps, width, label='Ring Attention', color='red', alpha=0.7)
    
    # 标注数值
    for bar, cap in zip(bars1, ulysses_caps):
        ax2.annotate(f'{cap}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8, rotation=45)
    for bar, cap in zip(bars2, ring_caps):
        ax2.annotate(f'{cap}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8, rotation=45)
    
    # 标记 KV 复制倍数
    for i, sp in enumerate(sp_sizes):
        if sp > n_kv_heads:
            kv_rep = sp // n_kv_heads
            ax2.annotate(f'KV复制:{kv_rep}x', xy=(x[i], 0), ha='center', va='top',
                        fontsize=9, color='orange', fontweight='bold')
    
    ax2.set_xlabel('SP Size', fontsize=12)
    ax2.set_ylabel('Token Capacity (per device)', fontsize=12)
    ax2.set_title(f'Token Capacity Comparison: Ulysses vs Ring\n'
                  f'(Ring has no KV replication, higher capacity when SP > n_kv_heads={n_kv_heads})',
                  fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'SP={sp}' for sp in sp_sizes])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ----- 图3: 通信量对比 (连续曲线) -----
    ax3 = axes[2]
    
    # 使用高密度采样以获得连续曲线
    seq_lens_cont = np.linspace(1, max_seq_len, 2000, dtype=int)
    ulysses_comms_cont = []
    ring_comms_cont = []
    
    for s in seq_lens_cont:
        u_sp = get_required_sp(s, ulysses_boundaries, sp_sizes)
        r_sp = get_required_sp(s, ring_boundaries, sp_sizes)
        ulysses_comms_cont.append(compute_ulysses_comm_volume(s, u_sp))
        ring_comms_cont.append(compute_ring_comm_volume(s, r_sp))
    
    ax3.plot(seq_lens_cont, ulysses_comms_cont, 'b-', linewidth=2, label='Ulysses AlltoAll')
    ax3.plot(seq_lens_cont, ring_comms_cont, 'r--', linewidth=2, label='Ring P2P')
    
    # 标注离散的关键点及其 SP size
    test_seq_points = np.array([s for s in test_seqlens if s <= max_seq_len])
    for seq in test_seq_points:
        u_sp = get_required_sp(seq, ulysses_boundaries, sp_sizes)
        r_sp = get_required_sp(seq, ring_boundaries, sp_sizes)
        u_comm = compute_ulysses_comm_volume(seq, u_sp)
        r_comm = compute_ring_comm_volume(seq, r_sp)
        
        # 画出离散点
        ax3.plot(seq, u_comm, 'bo', markersize=4)
        ax3.plot(seq, r_comm, 'rs', markersize=4)
        
        # 标注文字 (保持原有标注逻辑，但减少文字重叠感)
        ax3.annotate(f'SP={u_sp}\n{u_comm:.0f}MB', 
                    xy=(seq, u_comm), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='lightblue', alpha=0.5))
        
        ax3.annotate(f'SP={r_sp}\n{r_comm:.0f}MB', 
                    xy=(seq, r_comm), xytext=(0, -20),
                    textcoords='offset points', ha='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='lightcoral', alpha=0.5))

    # 标注差异区域
    better_ring_comm = np.array(ring_comms_cont) < np.array(ulysses_comms_cont)
    ax3.fill_between(seq_lens_cont, 0, max(max(ulysses_comms_cont), max(ring_comms_cont)) * 1.1, 
                     where=better_ring_comm, alpha=0.1, color='green', label='Ring has lower comm volume')
    
    y_max = max(max(ulysses_comms_cont), max(ring_comms_cont))
    ax3.set_xlabel('Sequence Length', fontsize=12)
    ax3.set_ylabel('Communication Volume (MB per layer, fwd+bwd)', fontsize=12)
    ax3.set_title(f'Communication Volume Comparison (Continuous Curve)\n'
                  f'Ulysses: AlltoAll (Q+K+V+O, KV复制), Ring: P2P (KV only, 无复制, BF16优化)',
                  fontsize=14)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_seq_len * 1.05)
    ax3.set_ylim(0, y_max * 1.3)
    
    plt.tight_layout()
    
    # 保存
    if save_path == "auto":
        model_name = config['name'].replace(' ', '_').replace('.', '_').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sp_boundaries_{model_name}_{int(device_memory_gb)}GB_{zero3_world_size}gpu_{timestamp}.png"
        save_path = os.path.join(PICTURES_DIR, filename)
        os.makedirs(PICTURES_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图片已保存: {save_path}")
    elif save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图片已保存: {save_path}")
    
    if show_plot:
        plt.show()
    
    # ==================== 总结差异的影响 ====================
    print("\n" + "=" * 100)
    print("差异影响分析")
    print("=" * 100)
    
    if difference_points:
        print(f"\n发现 {len(difference_points)} 个序列长度存在 SP size 选择差异:")
        for seqlen, u_sp, r_sp in difference_points:
            # 计算通信开销差异
            print(f"\n  序列长度 {seqlen}:")
            print(f"    - Ulysses 需要 SP={u_sp}, Ring 只需 SP={r_sp}")
            print(f"    - Ulysses 使用了 {u_sp//r_sp}x 更多的设备")
            print(f"    - 原因: GQA 下 Ulysses 需要 KV 复制，降低了 token_capacity")
    else:
        print("所有测试序列长度的 SP size 选择相同")
    
    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)
    
    # 计算平均通信量节省
    if len(comm_comparisons) > 0:
        avg_comm_ratio = np.mean([c[5] for c in comm_comparisons if c[5] < float('inf')])
        print(f"\n通信量统计:")
        print(f"  - 平均通信量比 (Ulysses/Ring): {avg_comm_ratio:.2f}x")
        print(f"  - Ring Attention 平均节省 {(avg_comm_ratio - 1) / avg_comm_ratio * 100:.1f}% 通信量")
    
    print(f"""
Ring Attention 的三大优势:

1. 【无 KV 复制】
   - 当 SP size > n_kv_heads={n_kv_heads} 时，Ulysses 需要复制 KV
   - Ring Attention 不需要复制，token_capacity 更大
   - 同一序列长度，Ring 可以使用更小的 SP size

2. 【更低通信量】
   - Ulysses: AlltoAll 需要传输 Q+O+KV，且 KV 可能复制
   - Ring: P2P 只传输 KV（无复制），且可以 overlap
   - 长序列和大 SP 下，Ring 通信量显著更低

3. 【更好的负载均衡】
   - 更少的设备参与，释放设备给其他序列
   - 更小的 SP size 提供更多灵活性
   - P2P 通信可与计算 overlap，减少有效通信时间
    """)
    
    return {
        'ulysses_boundaries': ulysses_boundaries,
        'ring_boundaries': ring_boundaries,
        'ulysses_token_caps': ulysses_token_caps,
        'ring_token_caps': ring_token_caps,
        'ulysses_intervals': ulysses_intervals,
        'ring_intervals': ring_intervals,
        'difference_points': difference_points,
        'comm_comparisons': comm_comparisons,  # (seqlen, u_sp, r_sp, u_comm, r_comm, ratio)
    }


def debug_cost_comparison(cost_model: flexSPCostModel, test_seqlens: List[int] = None):
    """调试：比较 Ulysses 和 Ring Attention 在不同配置下的时间估算"""
    if test_seqlens is None:
        test_seqlens = [2048, 4096, 8192, 16384, 32768, 65536]
    
    sp_sizes = [2, 4, 8, 16, 32]
    
    print("\n" + "="*100)
    print(" Cost Model 调试：Ulysses vs Ring Attention 详细分解 (同步最新逻辑)")
    print("="*100)
    print(f"模型参数: n_heads={cost_model.n_heads}, n_kv_heads={cost_model.n_kv_heads}, hidden_size={cost_model.h}")
    print(f"层数: {cost_model.l}, bwd_fwd_coe: {cost_model.bwd_fwd_coe}")
    print(f"cpt_alpha1={cost_model.cpt_alpha1:.2e}, cpt_alpha2={cost_model.cpt_alpha2:.4f}, cpt_beta1={cost_model.cpt_beta1:.4f}")
    
    for seqlen in test_seqlens:
        print(f"\n{'='*100}")
        print(f" 序列长度: {seqlen}")
        print(f"{'='*100}")
        
        for sp_size in sp_sizes:
            if seqlen / sp_size < 128:  # 跳过 local_seq 太小的情况
                continue
            
            u_total = cost_model.ulysses_time_whole_model(seqlen, sp_size)
            r_total = cost_model.zigzag_ring_flash_attention_time_whole_model(seqlen, sp_size)
            
            u_comm = cost_model.ulysses_alltoall_time_per_layer(seqlen, sp_size) * cost_model.l
            
            print(f"\n--- SP={sp_size} ---")
            print(f"  Ulysses 总时间: {u_total:.2f} ms (通信占比: {u_comm/u_total*100:.1f}%)")
            print(f"  Ring 总时间:    {r_total:.2f} ms")
            
            ratio = r_total / u_total if u_total > 0 else 0
            winner = "Ring ✓" if ratio < 1 else "Ulysses ✓"
            print(f"  加速比 (Ring/Ulysses): {ratio:.3f} → {winner}")


def analyze_solution(results: Dict, cost_model: flexSPCostModel, seqs: List[Sequence]) -> Dict:
    """分析求解结果"""
    if results is None:
        return {'status': 'failed'}
    
    A = results['A']
    sp_options = results['sp_options']
    M = results['M']
    m = results['m']
    
    K, P = A.shape
    
    # 分析每个 group
    groups = []
    for p in range(P):
        if m[p] == 0:
            continue
        
        sp_option = sp_options[p]
        if isinstance(sp_option, tuple):
            sp_size, attn_type = sp_option
        else:
            sp_size, attn_type = sp_option, 'ulysses'
        
        group_seqs = [seqs[k] for k in range(K) if A[k, p] == 1]
        if not group_seqs:
            continue
        
        group_lens = get_lens(group_seqs)
        # 修正方法名: _whole_time -> _whole_model
        exact_time = cost_model.seqs_total_time_whole_model(group_lens, sp_size, attn_type or 'ulysses')
        
        # 用 ILP 近似模型计算时间
        approx_time = sum([cost_model.total_time_single(seq, sp_size, attn_type or 'ulysses') 
                          for seq in group_lens]) + cost_model.compute_bias(sp_size, attn_type or 'ulysses')
        
        groups.append({
            'sp_size': sp_size,
            'attn_type': attn_type or 'ulysses',
            'num_seqs': len(group_seqs),
            'seq_lens': group_lens,
            'total_tokens': sum(group_lens),
            'exact_time': exact_time,
            'approx_time': approx_time,
        })
    
    # 计算整体指标
    max_exact_time = max([g['exact_time'] for g in groups]) if groups else 0
    max_approx_time = max([g['approx_time'] for g in groups]) if groups else 0
    
    return {
        'status': 'success',
        'ilp_objective': M,
        'max_exact_time': max_exact_time,
        'max_approx_time': max_approx_time,
        'num_groups': len(groups),
        'groups': groups,
    }


def print_analysis(analysis: Dict, title: str, verbose: bool = True):
    """打印分析结果"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    
    if analysis['status'] == 'failed':
        print("求解失败！")
        return
    
    print(f"ILP 目标值 (M): {analysis['ilp_objective']:.2f} ms")
    print(f"精确模型最大时间: {analysis['max_exact_time']:.2f} ms")
    print(f"ILP vs 精确 误差: {abs(analysis['ilp_objective'] - analysis['max_exact_time']) / analysis['max_exact_time'] * 100:.2f}%")
    
    # ========== GPU 分配统计 ==========
    # 统计每种 (sp_size, attn_type) 组合使用了多少 GPU
    gpu_usage = {}  # (sp_size, attn_type) -> num_groups
    for g in analysis['groups']:
        key = (g['sp_size'], g['attn_type'])
        gpu_usage[key] = gpu_usage.get(key, 0) + 1
    
    print(f"\n========== 32 GPU 集群分配 ==========")
    total_gpus = 0
    for (sp_size, attn_type), num_groups in sorted(gpu_usage.items()):
        gpus_used = sp_size * num_groups
        total_gpus += gpus_used
        print(f"  SP={sp_size:<2} {attn_type:<8}: {num_groups:>2} 个 group × {sp_size} GPU = {gpus_used:>2} GPUs")
    print(f"  {'总计':<14}: {total_gpus} GPUs")
    
    # ========== 按 SP size 和 attn_type 分组统计 ==========
    print(f"\n========== 策略分配详情 ==========")
    
    # 按 (sp_size, attn_type) 分组
    grouped = {}
    for g in analysis['groups']:
        key = (g['sp_size'], g['attn_type'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(g)
    
    for (sp_size, attn_type), groups in sorted(grouped.items()):
        print(f"\n--- SP={sp_size}, Attn={attn_type} ({len(groups)} 个 group, 占用 {sp_size * len(groups)} GPUs) ---")
        for i, g in enumerate(groups):
            seq_str = str(g['seq_lens']) if len(g['seq_lens']) <= 5 else f"{g['seq_lens'][:3]}...共{len(g['seq_lens'])}条"
            print(f"  Group {i+1}: {g['num_seqs']} 条序列, 总Token={g['total_tokens']}, Time={g['exact_time']:.2f}ms")
            print(f"          序列长度: {seq_str}")
    
    if verbose:
        print(f"\n========== 完整 Group 列表 ==========")
        print(f"{'#':<3} {'SP':<4} {'Attn':<8} {'Seqs':<5} {'TotalToken':<12} {'Time(ms)':<10} {'序列长度列表'}")
        print("-" * 90)
    for i, g in enumerate(analysis['groups']):
            seq_str = str(g['seq_lens']) if len(g['seq_lens']) <= 6 else f"{g['seq_lens'][:4]}...+{len(g['seq_lens'])-4}条"
            print(f"{i:<3} {g['sp_size']:<4} {g['attn_type']:<8} {g['num_seqs']:<5} {g['total_tokens']:<12} {g['exact_time']:<10.2f} {seq_str}")


def run_comparison(seq_lengths: List[int], scenario_name: str):
    """运行 Ulysses-only vs Ulysses+Ring 对比"""
    
    print(f"\n{'#'*70}")
    print(f"# 场景: {scenario_name}")
    print(f"# 序列数量: {len(seq_lengths)}")
    print(f"# 序列长度范围: [{min(seq_lengths)}, {max(seq_lengths)}]")
    print(f"# 总 Token 数: {sum(seq_lengths)}")
    print(f"# 序列长度分布: {sorted(seq_lengths, reverse=True)[:10]}..." if len(seq_lengths) > 10 else f"# 序列长度: {sorted(seq_lengths, reverse=True)}")
    print(f"{'#'*70}")
    
    # 创建序列对象
    seqs = [Sequence(length) for length in seq_lengths]
    
    # 创建 Cost Model
    cost_model = create_cost_model(QWEN_3B_CONFIG, BANDWIDTH_CONFIG)
    
    # ========== Ulysses Only ==========
    print("\n>>> 测试 Ulysses-Only 策略...")
    optimizer_ulysses = create_optimizer(cost_model, enable_ring=False)
    results_ulysses = optimizer_ulysses.solve_flexSP(seqs)
    analysis_ulysses = analyze_solution(results_ulysses, cost_model, seqs)
    print_analysis(analysis_ulysses, "Ulysses-Only 求解结果", verbose=False)
    
    # ========== Ulysses + Ring ==========
    print("\n>>> 测试 Ulysses+Ring 策略...")
    optimizer_ring = create_optimizer(cost_model, enable_ring=True)
    results_ring = optimizer_ring.solve_flexSP(seqs)
    analysis_ring = analyze_solution(results_ring, cost_model, seqs)
    print_analysis(analysis_ring, "Ulysses+Ring 求解结果", verbose=False)
    
    # ========== 对比分析 ==========
    print(f"\n{'='*70}")
    print(" 对比总结")
    print(f"{'='*70}")
    
    if analysis_ulysses['status'] == 'success' and analysis_ring['status'] == 'success':
        time_ulysses = analysis_ulysses['max_exact_time']
        time_ring = analysis_ring['max_exact_time']
        speedup = time_ulysses / time_ring if time_ring > 0 else 0
        
        print(f"\n--- 时间对比 ---")
        print(f"Ulysses-Only 最大 Group 时间: {time_ulysses:.2f} ms")
        print(f"Ulysses+Ring 最大 Group 时间: {time_ring:.2f} ms")
        print(f"加速比: {speedup:.3f}x")
        if speedup > 1:
            print(f"✅ Ring 策略节省时间: {(1 - 1/speedup) * 100:.2f}%")
        elif speedup < 1:
            print(f"❌ Ring 策略增加时间: {(1/speedup - 1) * 100:.2f}%")
        else:
            print(f"⚪ 两种策略时间相同")
        
        # GPU 分配对比
        print(f"\n--- GPU 分配对比 ---")
        
        def get_gpu_summary(analysis):
            summary = {}
            for g in analysis['groups']:
                key = (g['sp_size'], g['attn_type'])
                if key not in summary:
                    summary[key] = {'groups': 0, 'seqs': 0, 'gpus': 0}
                summary[key]['groups'] += 1
                summary[key]['seqs'] += g['num_seqs']
                summary[key]['gpus'] += g['sp_size']
            return summary
        
        summary_u = get_gpu_summary(analysis_ulysses)
        summary_r = get_gpu_summary(analysis_ring)
        
        print(f"{'策略配置':<20} {'Ulysses-Only':<20} {'Ulysses+Ring':<20}")
        print("-" * 60)
        
        all_keys = set(summary_u.keys()) | set(summary_r.keys())
        for key in sorted(all_keys):
            u_info = summary_u.get(key, {'groups': 0, 'seqs': 0, 'gpus': 0})
            r_info = summary_r.get(key, {'groups': 0, 'seqs': 0, 'gpus': 0})
            key_str = f"SP={key[0]},{key[1]}"
            u_str = f"{u_info['groups']}g/{u_info['seqs']}seq/{u_info['gpus']}gpu" if u_info['groups'] > 0 else "-"
            r_str = f"{r_info['groups']}g/{r_info['seqs']}seq/{r_info['gpus']}gpu" if r_info['groups'] > 0 else "-"
            print(f"{key_str:<20} {u_str:<20} {r_str:<20}")
    
    return {
        'scenario': scenario_name,
        'ulysses': analysis_ulysses,
        'ring': analysis_ring,
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FlexSP 求解器验证')
    parser.add_argument('--mode', type=str, default='compare', 
                        choices=['ulysses', 'ring', 'compare', 'sp_boundaries', 'overlap', 'debug', 'solver', 'benchmark', 'costmodel_check', 'all'],
                        help='运行模式: ulysses=Ulysses曲线, ring=Ring曲线, compare=对比图, sp_boundaries=SP切换点对比, overlap=Overlap分析, debug=调试, solver=求解器, benchmark=数据集基准测试, costmodel_check=Cost Model一致性检查, all=全部绘图')
    parser.add_argument('--device_memory', type=float, default=80,
                        help='单卡显存 (GB)')
    parser.add_argument('--zero3', type=int, default=64,
                        help='ZeRO-3 并行度 (同时也是 cluster_size)')
    parser.add_argument('--max_seq', type=int, default=524288,
                        help='最大序列长度')
    parser.add_argument('--save', type=str, default='auto',
                        help='保存图像路径 ("auto"=自动命名保存到默认目录, "none"=不保存, 或指定完整路径)')
    
    # Benchmark 模式专用参数
    parser.add_argument('--global_batch_size', type=int, default=32,
                        help='每次迭代采样的序列数量 (benchmark 模式)')
    parser.add_argument('--dataset', type=str, default='github',
                        help='数据集名称 (github, common_crawl 等)')
    parser.add_argument('--num_iters', type=int, default=5,
                        help='迭代次数 (benchmark 模式)')
    parser.add_argument('--sp_min', type=int, default=1,
                        help='最小 SP size (必须是 2 的幂次)')
    parser.add_argument('--sp_max', type=int, default=0,
                        help='最大 SP size (必须是 2 的幂次, 0 表示使用 cluster_size)')
    parser.add_argument('--verbose', action='store_true',
                        help='打印详细信息')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" FlexSP 求解器验证")
    print(" 模型: Qwen2.5-3B (GQA: n_heads=16, n_kv_heads=2)")
    print(" 集群: 32 GPUs, 40GB/GPU")
    print("="*70)
    
    max_seq = args.max_seq if args.max_seq > 0 else None
    save_path = None if args.save.lower() == 'none' else args.save
    
    if args.mode == 'ulysses':
        # ========== 绘制 Ulysses SP 时间曲线 ==========
        print("\n>>> 绘制 Ulysses SP 时间曲线...")
        ulysses_time_per_device(
            config=QWEN_72B_CONFIG,
            bandwidth=BANDWIDTH_CONFIG,
            device_memory_gb=args.device_memory,
            zero3_world_size=args.zero3,
            max_seq_len=max_seq,
            save_path=save_path,
            show_plot=True
        )
        return
    
    elif args.mode == 'ring':
        # ========== 绘制 Ring Attention 时间曲线 ==========
        print("\n>>> 绘制 Ring Attention 时间曲线...")
        ring_time_per_device(
            config=QWEN_72B_CONFIG,
            bandwidth=BANDWIDTH_CONFIG,
            device_memory_gb=args.device_memory,
            zero3_world_size=args.zero3,
            max_seq_len=max_seq,
            save_path=save_path,
            show_plot=True
        )
        return
    
    elif args.mode == 'compare':
        # ========== Ulysses vs Ring 对比 ==========
        print("\n>>> 绘制 Ulysses vs Ring 对比图...")
        compare_ulysses_ring(
            config=QWEN_72B_CONFIG,
            bandwidth=BANDWIDTH_CONFIG,
            device_memory_gb=args.device_memory,
            zero3_world_size=args.zero3,
            max_seq_len=max_seq,
            save_path=save_path,
            show_plot=True
        )
        return
    
    elif args.mode == 'sp_boundaries':
        # ========== SP Size 切换点对比 ==========
        print("\n>>> 绘制 SP Size 切换点对比图...")
        compare_sp_boundaries(
            config=QWEN_72B_CONFIG,
            bandwidth=BANDWIDTH_CONFIG,
            device_memory_gb=args.device_memory,
            zero3_world_size=args.zero3,
            max_seq_len=max_seq,
            save_path=save_path,
            show_plot=True
        )
        return
    
    elif args.mode == 'overlap':
        # ========== Overlap 条件分析 ==========
        print("\n>>> 分析 Ring Attention Overlap 条件...")
        analyze_overlap_conditions(
            config=QWEN_72B_CONFIG,
            bandwidth=BANDWIDTH_CONFIG,
            zero3_world_size=args.zero3,
        )
        return
    
    elif args.mode == 'all':
        # ========== 全部绘图 ==========
        print("\n>>> 绘制全部图表...")
        
        print("\n--- 1. Ulysses SP 时间曲线 ---")
        ulysses_time_per_device(
            config=QWEN_72B_CONFIG,
            bandwidth=BANDWIDTH_CONFIG,
            device_memory_gb=args.device_memory,
            zero3_world_size=args.zero3,
            max_seq_len=max_seq,
            save_path=save_path,
            show_plot=False
        )
        
        print("\n--- 2. Ring Attention 时间曲线 ---")
        ring_time_per_device(
            config=QWEN_72B_CONFIG,
            bandwidth=BANDWIDTH_CONFIG,
            device_memory_gb=args.device_memory,
            zero3_world_size=args.zero3,
            max_seq_len=max_seq,
            save_path=save_path,
            show_plot=False
        )
        
        print("\n--- 3. Ulysses vs Ring 对比图 ---")
        compare_ulysses_ring(
            config=QWEN_72B_CONFIG,
            bandwidth=BANDWIDTH_CONFIG,
            device_memory_gb=args.device_memory,
            zero3_world_size=args.zero3,
            max_seq_len=max_seq,
            save_path=save_path,
            show_plot=False
        )
        
        print("\n--- 4. SP Size 切换点对比图 ---")
        compare_sp_boundaries(
            config=QWEN_72B_CONFIG,
            bandwidth=BANDWIDTH_CONFIG,
            device_memory_gb=args.device_memory,
            zero3_world_size=args.zero3,
            max_seq_len=max_seq,
            save_path=save_path,
            show_plot=True  # 最后一张图显示
        )
        return
    
    elif args.mode == 'debug':
        # ========== 调试时间模型 ==========
        cost_model = create_cost_model(QWEN_72B_CONFIG, BANDWIDTH_CONFIG)
        debug_cost_comparison(cost_model, test_seqlens=[8192, 32768, 65536])
        return
    
    elif args.mode == 'costmodel_check':
        # ========== Cost Model 一致性检查 ==========
        print("\n>>> 检查 Cost Model 一致性...")
        debug_costmodel_consistency(config=QWEN_72B_CONFIG, bandwidth=BANDWIDTH_CONFIG)
        return
    
    elif args.mode == 'benchmark':#bench mark这一部分
        # ========== 数据集基准测试 ==========
        print("\n>>> 运行数据集基准测试...")
        sp_max = args.sp_max if args.sp_max > 0 else None
        run_solver_benchmark(
            global_batch_size=args.global_batch_size,
            max_seqlength=args.max_seq,
            dataset_name=args.dataset,
            num_iters=args.num_iters,
            config=QWEN_72B_CONFIG,
            bandwidth=BANDWIDTH_CONFIG,
            cluster_size=args.zero3,
            memory_limit_gb=args.device_memory,
            sp_size_min=args.sp_min,
            sp_size_max=sp_max,
            verbose=args.verbose,
        )
        return
    
    elif args.mode == 'solver':
        # ========== 求解器测试 (使用预定义场景) ==========
        cost_model = create_cost_model(QWEN_72B_CONFIG, BANDWIDTH_CONFIG)
        debug_cost_comparison(cost_model, test_seqlens=[8192, 32768, 65536])
        
        print("\n" + "="*70)
        print(" 开始求解器测试 (预定义场景)...")
    # print("="*70)
    
    # # 测试场景
    # scenarios = [
    #     ('short', '短序列为主'),
    #     ('long', '长序列为主'),
    # ]
    
    # results = []
    # for scenario, name in scenarios:
    #     seq_lengths = generate_test_sequences(scenario, num_samples=32)
    #     result = run_comparison(seq_lengths, name)
    #     results.append(result)
    
    # # 最终汇总
    # print("\n\n" + "="*70)
    # print(" 所有场景汇总")
    # print("="*70)
    # print(f"{'场景':<25} {'Ulysses(ms)':<15} {'Ring(ms)':<15} {'加速比':<10}")
    # print("-"*70)
    
    # for r in results:
    #     if r['ulysses']['status'] == 'success' and r['ring']['status'] == 'success':
    #         t_u = r['ulysses']['max_exact_time']
    #         t_r = r['ring']['max_exact_time']
    #         speedup = t_u / t_r if t_r > 0 else 0
    #         print(f"{r['scenario']:<25} {t_u:<15.2f} {t_r:<15.2f} {speedup:<10.3f}x")
    #     return


if __name__ == "__main__":
    main()

