from typing import List, Dict, Union, Literal
import numpy as np
from collections import Counter
from pyscipopt import Model, quicksum, multidict
import random
import multiprocessing as mp
import galvatron.flexsp_solver
# from sequence_module_py import Sequence, SeqBucket, print_seqs, get_lens, bucketing_seqs, chunk_globalbatch
from sequence_module import Sequence, SeqBucket, print_seqs, get_lens, bucketing_seqs, chunk_globalbatch
import argparse
import time

class flexSPCostModel():
    def __init__(self, 
                 cluster_size: int = 16,
                 hidden_size: int = 4096,
                 n_heads: int = 32,
                 n_kv_heads: int = 8,
                 layer_num: int = 32,
                 param_size_B: float = 7,#GB
                 zero_stage: int = 3,
                 mixed_precision: bool = True,
                 act_per_token: float = 4.71,
                 cpt_alpha1: float = 5.128 * 1e-6,#前向拟合时间，因此我们需要
                 cpt_alpha2: float = 183.9576 * 1e-3,
                 cpt_beta1: float = 629.3563,
                 bwd_fwd_coe: float = 2.0,
                 alltoall_bandwidth_dict_gbs: Dict = {1: 1e10, 2: 154, 4: 137, 8: 121, 16: 8.7},
                 p2p_bandwidth_dict_gbs: Dict = {1: 1e10, 2: 154, 4: 137, 8: 121, 16: 8.7},
                 ring_overlap_efficiency: float = 0.15,  # Ring Attention 通信与计算重叠效率
                 # ===== Double-Ring modeling knobs (LoongTrain-style) =====
                 # 说明：
                 # - 你现有的 ring 模型只用一个 p2p_bandwidth_dict_gbs，等价于“所有 P2P 都用同一类链路带宽”。
                 # - Double-Ring 需要区分内圈(intra-node, NVLINK) 和外圈(inter-node, NIC/IB)。
                 # 这里我们用两个带宽参数来做可调建模；如果你手里有 profile 曲线，可以把它们设成测得值。
                 p2p_intra_bandwidth_gbs: float = 600.0,  # 近似 NVLINK 级别有效带宽 (GB/s)，可按集群/实现校准
                 p2p_inter_bandwidth_gbs: float = 50.0,   # 近似 跨节点 P2P 单“rail”有效带宽 (GB/s)，可按集群/实现校准
                 num_nics_per_node: int = 4,              # 每节点 NIC 数；LoongTrain 建议 w≈NIC 数
                 ):
        self.N = cluster_size
        self.h = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.l = layer_num
        self.p = param_size_B
        self.zero_stage = zero_stage
        self.act_per_token = act_per_token
        self.cpt_alpha1 = cpt_alpha1 # * 1e-6#为什么他这个会这么大呢 我们这相当于profile出来的是前向时间
        self.cpt_alpha2 = cpt_alpha2 # * 1e-3
        self.cpt_beta1 = cpt_beta1 #现在全都改成了整个模型了
        self.bwd_fwd_coe = bwd_fwd_coe
        self.zero_ratio = {
            0: 1,
            1: (6/8 * (1/self.N) + 2/8) if mixed_precision else (2/4 * (1/self.N) + 2/4),
            2: (7/8 * (1/self.N) + 1/8) if mixed_precision else (3/4 * (1/self.N) + 1/4),
            3: 1/self.N
        }[self.zero_stage]#B Billison
        self.model_states_mb = param_size_B * 16 * self.zero_ratio * 1024#这里的16是什么意思 转换为mb
        # print(self.model_states_mb)
        self.alltoall_bandwidth_dict_gbs = alltoall_bandwidth_dict_gbs
        self.p2p_bandwidth_dict_gbs = p2p_bandwidth_dict_gbs  # Ring Attention P2P 带宽
        self.ring_overlap_efficiency = ring_overlap_efficiency  # Ring Attention 通信重叠效率
        self.p2p_intra_bandwidth_gbs = p2p_intra_bandwidth_gbs
        self.p2p_inter_bandwidth_gbs = p2p_inter_bandwidth_gbs
        self.num_nics_per_node = num_nics_per_node
        
    def check_costmodel(self, seqlen: Union[int,List[int]], sp_size: int = 1, attn_type: str = 'ulysses'):
        if not isinstance(seqlen, List):
            seqlen = [seqlen]
        print(f'\n[Seqlen = {seqlen}, SP = {sp_size}, AttnType = {attn_type}]')
        print(f'Activation Size: {self.activation_size(seqlen, sp_size, attn_type)}')
        print(f'Total Memory: {self.total_memory(seqlen, sp_size, attn_type)}')
        print(f'Computation Time: {self.compute_time(seqlen, sp_size)}')
        if attn_type == 'ulysses':
            print(f'AlltoAll Time: {self.ulysses_alltoall_time(seqlen, sp_size)}')
        else:
            print(f'Ring P2P Time: {self.ring_p2p_time(seqlen, sp_size)}')
        print(f'Total Time: {self.total_time(seqlen, sp_size, attn_type)}')
        comm_time = self.ulysses_alltoall_time(seqlen, sp_size) if attn_type == 'ulysses' else self.ring_p2p_time(seqlen, sp_size)
        print(f'Comm Ratio: {comm_time/self.total_time(seqlen, sp_size, attn_type)*100: .2f}')
        
    ### ======== Memory Cost Related ========
    def activation_size(self, seqlen: Union[int,List[int]], sp_size: int = 1, attn_type: str = 'ulysses'):
        if isinstance(seqlen, List):
            seqlen = sum(seqlen)
        
        base = self.act_per_token * seqlen
        
        if attn_type == 'ring':
            # Ring Attention: 无 KV 复制，显存完美切分
            return base / sp_size
        else:
            # Ulysses
            if self.n_heads == self.n_kv_heads or sp_size <= self.n_kv_heads:
                return base / sp_size
            else:
                # GQA/MQA 且 sp_size > n_kv_heads: KV 需要复制
                # 修正: 使用 head_dim = hidden_size / n_heads
                head_dim = self.h // self.n_heads
                extra = seqlen * (sp_size - self.n_kv_heads) * self.n_kv_heads * head_dim * 2 * 2 / 1024 / 1024
                return (base + extra) / sp_size

    def total_memory(self, seqlen: Union[int,List[int]] = 0, sp_size: int = 1, attn_type: str = 'ulysses'):
        return self.model_states_mb + self.activation_size(seqlen, sp_size, attn_type)
    
    def memory_capacity(self, memory_limit_gb: int, seqlen: Union[int,List[int]] = 0, sp_size: int = 1, attn_type: str = 'ulysses'):
        return memory_limit_gb * 1024 - self.total_memory(seqlen, sp_size, attn_type) #计算还有多少空余

    def token_capacity(self, memory_limit_gb: int, seqlen = 0, sp_size = 1,attn_type: str = 'ulysses'):#所以这里的token capacity已经考虑了sp size了
        available_mem = memory_limit_gb * 1024 - self.model_states_mb
        
        if attn_type == 'ulysses' and sp_size > self.n_kv_heads:
            # 修正: 使用 head_dim = hidden_size / n_heads
            head_dim = self.h // self.n_heads
            act_per_token_effective = self.act_per_token + (sp_size - self.n_kv_heads) * self.n_kv_heads * head_dim * 2 * 2 / 1024 / 1024     
        else:
           act_per_token_effective = self.act_per_token   
        return int(available_mem / act_per_token_effective)
        

    #Ulysses SP commnication time & computation time
    def ulysses_fwd_compute_time_single_per_layer(self, seqlen: int, sp_size: int = 1): # ms
        return (self.cpt_alpha1 * (seqlen ** 2) + self.cpt_alpha2 * seqlen) / sp_size

    def ulysses_fwd_compute_time_per_layer(self, seqlen: Union[int,List[int]], sp_size: int = 1): # ms
        if not isinstance(seqlen, List):
            seqlen = [seqlen]
        cpt_times = [self.ulysses_fwd_compute_time_single_per_layer(seq, sp_size) for seq in seqlen]
        return sum(cpt_times) + self.cpt_beta1

    def ulysses_bwd_compute_time_per_layer(self, seqlen: Union[int,List[int]], sp_size: int = 1):
        return self.bwd_fwd_coe * self.ulysses_fwd_compute_time_per_layer(seqlen, sp_size)

    def ulysses_alltoall_time_per_layer(self, seqlen: Union[int,List[int]] = 0, sp_size: int = 1): # ms
        if sp_size == 1:
            return 0
        if isinstance(seqlen, List):
            seqlen = sum(seqlen)
        v = self.alltoall_bandwidth_dict_gbs[sp_size]
        head_dim = self.h // self.n_heads
        
        # Q+O: seqlen * hidden_size * 2 (bf16) * 2 (Q和O)
        QO_tensor_size = 2 * self.h * seqlen * 2 / 1024 / 1024  # MB
        
        # K+V: seqlen * n_kv_heads * head_dim * 2 (bf16) * 2 (K和V)
        # 当 sp_size > n_kv_heads 时需要复制
        effective_kv_heads = self.n_kv_heads if self.n_kv_heads >= sp_size else sp_size
        KV_tensor_size = 2 * effective_kv_heads * head_dim * seqlen * 2 / 1024 / 1024  # MB
        
        all2all_tensor_size = QO_tensor_size + KV_tensor_size
        # AlltoAll: 每个 rank 发送 total * (sp_size-1) / sp_size
        return 2 * all2all_tensor_size * (sp_size - 1) / sp_size / sp_size / v

    def ulysses_time_whole_model(self, seqlen: Union[int,List[int]] = 0, sp_size: int = 1):
        ulysses_time_per_layer = self.ulysses_bwd_compute_time_per_layer(seqlen, sp_size) + self.ulysses_fwd_compute_time_per_layer(seqlen, sp_size) + self.ulysses_alltoall_time_per_layer(seqlen, sp_size)
        return self.l * ulysses_time_per_layer

    def zigzag_ring_flash_attention_time_per_layer(self, seqlen: Union[int, List[int]] = 0, sp_size: int = 1):
        '''
        以下我们先计算单层的前向的计算与通信，之后我们再计算单层的后向的计算和通信
        分别考虑前向和后向的overlap情况
        最后再进行加总
        '''
        if not isinstance(seqlen, List):
            seqlen = [seqlen]
        
        # 计算 sum(seq^2) 和 sum(seq)
        sum_seq_sq = sum([s ** 2 for s in seqlen])
        sum_seq = sum(seqlen)
        
        if sp_size == 1:
            return (self.cpt_alpha1 * sum_seq_sq + self.cpt_alpha2 * sum_seq + self.cpt_beta1) * (self.bwd_fwd_coe + 1)
        
        p2p_bandwidth = self.p2p_bandwidth_dict_gbs[sp_size]
        head_dim = self.h // self.n_heads

        # local_seqlen: 每张卡上的序列长度（zigzag 切分后）
        local_seqlen = [s / sp_size for s in seqlen]
        sum_local_seq = sum(local_seqlen)
        sum_local_seq_sq = sum([s ** 2 for s in local_seqlen])
        
        # 通信时间 per step (修正: 使用 head_dim)
        # KV: 2 * local_seq * n_kv_heads * head_dim * 2 bytes
        KV_tensor_p2p_time_per_step = 2 * sum_local_seq * self.n_kv_heads * head_dim * 2 / 1024 / 1024 / p2p_bandwidth
        
        # 纯 attention 计算时间（不含 kernel overhead，参与 overlap）
        # Zigzag 优化下：
        # Step 0: Sq=S, Skv=S, causal=True. Quadratic=0.5, Linear=1.0 (相对于非causal S*S)
        # Others: Sq=S, Skv=S/2 (或相反), causal=False. Quadratic=0.5, Linear=0.75
        quad_time = 0.5 * self.cpt_alpha1 * sum_local_seq_sq
        attn_time_0 = quad_time + self.cpt_alpha2 * sum_local_seq
        attn_time_others = quad_time + 0.75 * self.cpt_alpha2 * sum_local_seq
        
        # Kernel overhead（不参与 overlap，最后单独加）
        # 修正：cpt_beta1 (约0.5ms) 远大于真实的 Kernel Launch (约0.02ms)。
        # 这说明 cpt_beta1 包含了大量不随 Ring Loop 倍增的固定开销 (如 Python overhead, setup 等)。
        # 因此，我们假设每次 Ring Loop 只增加一小部分真实 Launch 开销 (0.05ms)，而不是完整的 cpt_beta1。
        base_overhead = self.cpt_beta1
        loop_overhead = 0.05 # 50us per extra loop
        total_overhead_base = base_overhead + loop_overhead * (sp_size - 1)
        
        kernel_overhead = total_overhead_base * (1 + self.bwd_fwd_coe)
        
        def overlap_time(attn_time, comm_time):
            # 通信主导: 总时间 = 通信时间
            # 计算主导: 总时间 = 计算时间 + overlap_efficiency * 通信时间
            if comm_time > attn_time:
                return comm_time
            else:
                return attn_time + self.ring_overlap_efficiency * comm_time

        # 前向时间（纯 attention 计算与通信的 overlap）
        # FWD: Step 0 与 1x 通信重叠; 中间 Step 与 1x 重叠; 最后一步(Step SP-1)无通信
        fwd_time_per_layer = overlap_time(attn_time_0, KV_tensor_p2p_time_per_step) + \
            (sp_size - 2) * overlap_time(attn_time_others, KV_tensor_p2p_time_per_step) + \
            attn_time_others
        
        # 反向时间
        # ==============================================================================================
        # ⚠️ 精度风险与通信优化标注：
        # 当前建模假设：dKV (梯度) 在发送前从 FP32 强制转换为 BF16 进行传输，本地累加仍维持 FP32。
        # 优化效果：中间步骤通信倍率从 4x 降至 2x (K_bf16 + dK_bf16)，显著降低长序列反向传播耗时。
        # 潜在风险：环路中经过 SP_size 次精度截断，可能导致梯度误差累积，请在训练时密切关注收敛曲线。
        # ==============================================================================================
        bwd_attn_0 = self.bwd_fwd_coe * attn_time_0
        bwd_attn_others = self.bwd_fwd_coe * attn_time_others
        
        if sp_size == 2:
            # Step 0: 1x comm, Step 1: 1x comm (dKV only), Drain: 1x comm
            bwd_time_per_layer = overlap_time(bwd_attn_0, KV_tensor_p2p_time_per_step) + \
                overlap_time(bwd_attn_others, KV_tensor_p2p_time_per_step) + \
                KV_tensor_p2p_time_per_step
        else:
            # Step 0: 1x comm
            # Steps 1 to SP-2: 2x comm (stack KV+dKV in bf16)
            # Step SP-1: 1x comm (dKV in bf16)
            # Drain: 1x comm (dKV in bf16)
            bwd_time_per_layer = overlap_time(bwd_attn_0, KV_tensor_p2p_time_per_step) + \
                (sp_size - 2) * overlap_time(bwd_attn_others, 2 * KV_tensor_p2p_time_per_step) + \
                overlap_time(bwd_attn_others, KV_tensor_p2p_time_per_step) + \
                KV_tensor_p2p_time_per_step
        
        # 总时间 = 前向attention(含overlap) + 反向attention(含overlap) + kernel_overhead
        total_time_per_layer = fwd_time_per_layer + bwd_time_per_layer + kernel_overhead
        return total_time_per_layer

    def double_ring_flash_attention_time_per_layer(
        self,
        seqlen: Union[int, List[int]] = 0,
        sp_size: int = 1,
        inner_ring_size: int = 4,
        # 可选：覆盖默认带宽（方便你在 notebook/脚本里扫参）
        p2p_intra_bandwidth_gbs: float = None,
        p2p_inter_bandwidth_gbs: float = None,
        num_nics_per_node: int = None,
    ):
        """
        Double-Ring Attention 建模（对应 LoongTrain 论文里的 Double-Ring-Attention 思路）。

        你现在的 `zigzag_ring_flash_attention_time_per_layer()` 是“单环 Ring-Attention”的建模：
        - 每个 micro-step 需要一次 P2P 传 KV chunk（最后一步无下一步通信）
        - 通信链路被抽象成一个带宽 p2p_bandwidth_dict_gbs[sp_size]

        Double-Ring 的结构（按论文描述）：
        - 把 CP(=sp_size) 划分成多个 inner rings，每个 ring 大小 w=inner_ring_size
        - 外圈 outer ring 在 inner rings 之间传 KV；内圈 inner ring 在同 ring 内滚动 KV
        - forward 每个 outer step：
            - 触发 1 次 outer-ring P2P（跨节点，期望用多 NIC）
            - 触发 (w-1) 次 inner-ring P2P（尽量同节点 NVLINK）
          （最后一个 outer step 无需再做 outer-ring P2P）

        本实现的关键“可讨论假设”（你可以和我一起改）：
        1) 仍沿用你 ring 模型的 zigzag 计算拆分（attn_time_0 / attn_time_others），并假设
           除了全局第一个 micro-step 用 attn_time_0，其余 micro-step 用 attn_time_others。
        2) inner-ring P2P 与 outer-ring P2P 的数据量都按 “KV chunk” 建模，大小同你 ring 模型：
           KV_chunk_MB = 2 * local_seq * n_kv_heads * head_dim * 2 bytes / (1024^2)
           （2 表示 K/V 两个张量，2 bytes 表示 bf16/fp16）
        3) outer-ring 的“多 NIC 并行”用一个简单的带宽放大系数来近似：
           eff_outer_bw = p2p_inter_bw * min(w, num_nics_per_node)
           直觉：w 越接近 NIC 数，越能把多 rail 吃满；w 继续变大不再增加可用 rail。
           这不是严格的网络模型，但很适合用 profile 去拟合校准。

        返回：单层（FWD+BWD+overhead）时间（ms）。
        """

        if not isinstance(seqlen, List):
            seqlen = [seqlen]

        if sp_size == 1:
            sum_seq_sq = sum([s ** 2 for s in seqlen])
            sum_seq = sum(seqlen)
            return (self.cpt_alpha1 * sum_seq_sq + self.cpt_alpha2 * sum_seq + self.cpt_beta1) * (self.bwd_fwd_coe + 1)

        if sp_size % inner_ring_size != 0:
            raise ValueError(f"Double-Ring requires sp_size % inner_ring_size == 0, got sp_size={sp_size}, w={inner_ring_size}")

        w = inner_ring_size
        outer_steps = sp_size // w

        intra_bw = self.p2p_intra_bandwidth_gbs if p2p_intra_bandwidth_gbs is None else p2p_intra_bandwidth_gbs
        inter_bw = self.p2p_inter_bandwidth_gbs if p2p_inter_bandwidth_gbs is None else p2p_inter_bandwidth_gbs
        nics = self.num_nics_per_node if num_nics_per_node is None else num_nics_per_node

        head_dim = self.h // self.n_heads

        # local_seqlen: 每张卡上的序列长度（沿用你 zigzag ring 的切分假设：均匀切分）
        local_seqlen = [s / sp_size for s in seqlen]
        sum_local_seq = sum(local_seqlen)
        sum_local_seq_sq = sum([s ** 2 for s in local_seqlen])

        # KV chunk 传输时间（ms）
        # 注意：这里与 ring 模型保持一致：只建模 KV 的 payload，不额外加 send/recv 双向系数。
        kv_chunk_mb = 2 * sum_local_seq * self.n_kv_heads * head_dim * 2 / 1024 / 1024  # MB
        T_p2p_inner = kv_chunk_mb / intra_bw  # ms (GB/s ~ MB/ms)
        eff_outer_bw = inter_bw * max(1, min(w, nics))
        T_p2p_outer = kv_chunk_mb / eff_outer_bw

        # 纯 attention 计算时间（参与 overlap），沿用 zigzag 的拆分
        quad_time = 0.5 * self.cpt_alpha1 * sum_local_seq_sq
        attn_time_0 = quad_time + self.cpt_alpha2 * sum_local_seq
        attn_time_others = quad_time + 0.75 * self.cpt_alpha2 * sum_local_seq

        # overhead：沿用 ring 模型的“base + 每额外 loop 少量开销”
        base_overhead = self.cpt_beta1
        loop_overhead = 0.05  # 50us per extra micro-step loop
        total_overhead_base = base_overhead + loop_overhead * (sp_size - 1)
        kernel_overhead = total_overhead_base * (1 + self.bwd_fwd_coe)

        def overlap_time(attn_time, comm_time):
            # 与 ring 模型保持一致：计算主导时只“吃掉”一部分通信（ring_overlap_efficiency）
            if comm_time > attn_time:
                return comm_time
            else:
                return attn_time + self.ring_overlap_efficiency * comm_time

        # =========================
        # Forward（Double-Ring）
        # =========================
        # 思路：把 sp_size 个 micro-steps 分组到 outer_steps 个 outer step。
        # - 每个 outer step：有 (w-1) 次 inner P2P +（除最后 outer step 外）1 次 outer P2P
        # - 计算：总共 sp_size 个 micro-step，其中第一个 micro-step 用 attn_time_0，其他用 attn_time_others
        fwd_time = 0.0

        # Outer step 0: 第一个 micro-step 使用 attn_time_0，并与 outer P2P（若 outer_steps>1）重叠
        if outer_steps > 1:
            fwd_time += overlap_time(attn_time_0, T_p2p_outer)
        else:
            # 只有一个 outer step，则没有 outer-ring P2P
            fwd_time += attn_time_0

        # 剩余 (w-1) 个 inner micro-steps：每步与 inner P2P 重叠（最后一个 inner step 无需为“下一步”通信，
        # 这里按论文计数 (w-1) 次 inner P2P，等价于 (w-1) 步走 overlap，其余 1 步纯计算）
        if w >= 2:
            fwd_time += (w - 1) * overlap_time(attn_time_others, T_p2p_inner)

        # 中间 outer steps（1 ... outer_steps-2）：每个都有 1 次 outer P2P + (w-1) 次 inner P2P
        for _ in range(max(0, outer_steps - 2)):
            fwd_time += overlap_time(attn_time_others, T_p2p_outer)
            if w >= 2:
                fwd_time += (w - 1) * overlap_time(attn_time_others, T_p2p_inner)

        # 最后一个 outer step：无 outer P2P，只剩 (w) 个 micro-steps 的计算；按同样方式近似 (w-1) 次 inner P2P overlap
        if outer_steps >= 2:
            if w >= 2:
                fwd_time += (w - 1) * overlap_time(attn_time_others, T_p2p_inner)
            # 最终还需要 1 个 micro-step 的纯计算（没有下一步通信）
            fwd_time += attn_time_others
        else:
            # outer_steps==1 的情况：我们已经计入了 attn_time_0 和 (w-1)*overlap(others,inner)，还缺最后一步纯计算
            fwd_time += attn_time_others

        # =========================
        # Backward（Double-Ring）
        # =========================
        # 这里先给一个“和你 ring 模型同风格”的近似：用 bwd_fwd_coe 缩放计算，并保留你对通信倍率的假设：
        # - 大多数中间步骤需要传 KV + dKV => 2x KV payload
        # - 最后与 drain 需要 1x（dKV）
        #
        # Double-Ring 的 inner/outer P2P 都承担“搬运 chunk”的角色，因此这里把倍率同样作用到两类链路上。
        bwd_attn_0 = self.bwd_fwd_coe * attn_time_0
        bwd_attn_others = self.bwd_fwd_coe * attn_time_others

        # 经验：中间阶段 2x payload（KV + dKV），边界阶段 1x payload
        comm_1x_inner = T_p2p_inner
        comm_1x_outer = T_p2p_outer
        comm_2x_inner = 2.0 * T_p2p_inner
        comm_2x_outer = 2.0 * T_p2p_outer

        # 把 backward 也按 outer step 分组：第一组用 bwd_attn_0，其余用 bwd_attn_others
        bwd_time = 0.0

        # outer step 0: 用 bwd_attn_0，与 outer comm（若存在）按 1x 近似重叠
        if outer_steps > 1:
            bwd_time += overlap_time(bwd_attn_0, comm_1x_outer)
        else:
            bwd_time += bwd_attn_0

        # inner 的 (w-1) 次通信：把它视为“中间步骤”，用 2x（和你 ring 模型的中间段一致）
        if w >= 2:
            bwd_time += (w - 1) * overlap_time(bwd_attn_others, comm_2x_inner)

        # 中间 outer steps：outer comm 2x（中间段），inner comm 2x
        for _ in range(max(0, outer_steps - 2)):
            bwd_time += overlap_time(bwd_attn_others, comm_2x_outer)
            if w >= 2:
                bwd_time += (w - 1) * overlap_time(bwd_attn_others, comm_2x_inner)

        # 最后 outer step：无 outer comm，inner comm 1x（尾段），再加最后一步纯计算
        if outer_steps >= 2:
            if w >= 2:
                bwd_time += (w - 1) * overlap_time(bwd_attn_others, comm_1x_inner)
            # 最后一步计算 + drain（把 drain 近似成 1x outer comm；如果你后面想更精细，我们可以把 drain 拆出来）
            bwd_time += bwd_attn_others + comm_1x_outer
        else:
            bwd_time += bwd_attn_others + comm_1x_outer

        return fwd_time + bwd_time + kernel_overhead

    def zigzag_ring_flash_attention_time_whole_model(self, seqlen: Union[int, List[int]] = 0, sp_size: int = 1):
        return self.l * self.zigzag_ring_flash_attention_time_per_layer(seqlen, sp_size)

    def double_ring_flash_attention_time_whole_model(
        self,
        seqlen: Union[int, List[int]] = 0,
        sp_size: int = 1,
        inner_ring_size: int = 4,
        **kwargs,
    ):
        return self.l * self.double_ring_flash_attention_time_per_layer(
            seqlen=seqlen, sp_size=sp_size, inner_ring_size=inner_ring_size, **kwargs
        )

    def seqs_total_time_whole_model(self, seqlen: Union[int, List[int]] = 0, sp_size: int = 1, attn_type: str = "ulysses"):
        if attn_type == "ulysses":
            return self.ulysses_time_whole_model(seqlen, sp_size)
        elif attn_type == "double_ring":
            # 默认 inner_ring_size=4（常见 4 NIC/节点）；如果你想扫 w，在调用处传 inner_ring_size 即可
            return self.double_ring_flash_attention_time_whole_model(seqlen, sp_size, inner_ring_size=4)
        else:
            return self.zigzag_ring_flash_attention_time_whole_model(seqlen, sp_size)

    # ==================== ILP 约束所需的单序列时间贡献函数 ====================
    # 这些函数用于 ILP 中的 quicksum(time_single(seq[k]) * A[k,p])
    
    def total_time_single(self, seqlen: int, sp_size: int = 1, attn_type: str = 'ulysses'):
        """
        单条序列对 SP group 总时间的贡献（用于 ILP 约束）
        
        注意：这是一个近似，假设每条序列的时间贡献可以独立计算
        实际执行时会调用 seqs_total_time_whole_model 来获取更准确的时间
        """
        if attn_type == 'ulysses' or attn_type is None:
            return self.ulysses_time_single(seqlen, sp_size)
        else:  # ring
            return self.ring_time_single(seqlen, sp_size)
    
    def ulysses_time_single(self, seqlen: int, sp_size: int = 1):
        """
        Ulysses: 单条序列的时间贡献（不含固定开销）
        时间 = (前向 + 反向) * 层数 + alltoall 通信
        """
        # 计算时间 per layer: (alpha1 * seq^2 + alpha2 * seq) / sp
        compute_per_layer = (self.cpt_alpha1 * (seqlen ** 2) + self.cpt_alpha2 * seqlen) / sp_size
        # 前向 + 反向
        compute_total = compute_per_layer * (1 + self.bwd_fwd_coe) * self.l
        
        # alltoall 通信时间
        alltoall_time = self.ulysses_alltoall_time_per_layer(seqlen, sp_size) * self.l
        
        return compute_total + alltoall_time
    
    def ring_time_single(self, seqlen: int, sp_size: int = 1):
        """
        Ring Attention: 单条序列的时间贡献（不含固定开销 cpt_beta1）
        
        使用精确的 overlap 计算：每步取 max(计算时间, 通信时间)
        注意：cpt_beta1 (kernel overhead) 在 compute_bias 中统一添加
        """
        if sp_size == 1:
            # 无 SP 时，与 Ulysses 相同（不含 cpt_beta1）
            compute_per_layer = (self.cpt_alpha1 * (seqlen ** 2) + self.cpt_alpha2 * seqlen)
            return compute_per_layer * (1 + self.bwd_fwd_coe) * self.l
        
        local_seq = seqlen / sp_size
        p2p_bandwidth = self.p2p_bandwidth_dict_gbs.get(sp_size, 100)
        head_dim = self.h // self.n_heads
        
        # 纯 Flash Attention 计算（每步，参与 overlap，不含 kernel overhead）
        # Zigzag 模式下：
        # Step 0: Sq=S, Skv=S, causal=True. Quadratic=0.5, Linear=1.0 (相对于非causal S*S)
        # Others: Sq=S, Skv=S/2 (或相反), causal=False. Quadratic=0.5, Linear=0.75
        quad_time = 0.5 * self.cpt_alpha1 * (local_seq ** 2)
        attn_0 = max(0, quad_time + self.cpt_alpha2 * local_seq)
        attn_others = max(0, quad_time + 0.75 * self.cpt_alpha2 * local_seq)
        
        # P2P 通信（每步，修正: 使用 head_dim）
        kv_size = 2 * local_seq * self.n_kv_heads * head_dim * 2 / 1024 / 1024  # MB (K + V)
        comm_per_step = kv_size / p2p_bandwidth
        
        def overlap_time(compute, comm):
            # 通信主导: 总时间 = 通信时间
            # 计算主导: 总时间 = 计算时间 + overlap_efficiency * 通信时间
            if comm > compute:
                return comm
            else:
                return compute + self.ring_overlap_efficiency * comm
        
        # 前向时间（每层，纯 attention 与通信的 overlap）
        # FWD: Step 0 与 1x 通信重叠; 中间 Step 与 1x 重叠; 最后一步(Step SP-1)无通信
        fwd_time_per_layer = overlap_time(attn_0, comm_per_step) + \
            (sp_size - 2) * overlap_time(attn_others, comm_per_step) + attn_others
        
        # 反向时间（每层）
        # 反向
        # ==============================================================================================
        # ⚠️ 精度风险与通信优化标注：
        # 当前建模假设：dKV (梯度) 在发送前从 FP32 强制转换为 BF16 进行传输，本地累加仍维持 FP32。
        # 优化效果：中间步骤通信倍率从 4x 降至 2x (K_bf16 + dK_bf16)，显著降低长序列反向传播耗时。
        # 潜在风险：环路中经过 SP_size 次精度截断，可能导致梯度误差累积，请在训练时密切关注收敛曲线。
        # ==============================================================================================
        bwd_attn_0 = self.bwd_fwd_coe * attn_0
        bwd_attn_others = self.bwd_fwd_coe * attn_others
        
        if sp_size == 2:
            # Step 0: 1x comm, Step 1: 1x comm (dKV), Drain: 1x comm
            bwd_time_per_layer = overlap_time(bwd_attn_0, comm_per_step) + \
                overlap_time(bwd_attn_others, comm_per_step) + comm_per_step
        else:
            # Step 0: 1x, Steps 1 to SP-2: 2x (BF16 Stack), Step SP-1: 1x, Drain: 1x
            bwd_time_per_layer = overlap_time(bwd_attn_0, comm_per_step) + \
                (sp_size - 2) * overlap_time(bwd_attn_others, 2 * comm_per_step) + \
                overlap_time(bwd_attn_others, comm_per_step) + comm_per_step
        
        # 每层总时间（不含 kernel overhead）
        time_per_layer = fwd_time_per_layer + bwd_time_per_layer
        
        return time_per_layer * self.l
    
    def compute_bias(self, sp_size: int = 1, attn_type: str = 'ulysses'):
        """
        SP group 的固定开销（kernel launch 等）
        这部分与序列数量无关，只加一次
        
        Ulysses: 每层 1 次 kernel → cpt_beta1 * l * (1 + bwd_fwd_coe)
        Ring: 每层 sp_size 次 kernel → cpt_beta1 * sp_size * l * (1 + bwd_fwd_coe)
        """
        if attn_type == 'ring' and sp_size > 1:
            return self.cpt_beta1 * sp_size * self.l * (1 + self.bwd_fwd_coe)
        else:
            return self.cpt_beta1 * self.l * (1 + self.bwd_fwd_coe)

class flexSPOptimizer():
    def __init__(self, 
                 cluster_size: int,
                 memory_limit_gb: int, 
                 costmodel: flexSPCostModel,
                 hide_scipoutput: bool = False,
                 hide_alloutput: bool = False,
                 concurrent: bool = False,
                 scip_param_dict: dict = {},
                 strategy: Literal['flexSP', 'adaptive_bfd', 'fix_sp_bfd'] = 'flexSP',
                 redist_without_empty_group: bool = False,
                 enable_ring_attn: bool = True,  # 是否启用 Ring Attention 策略
                 attn_types: List[str] = None,  # 支持的 attention 类型列表
                 sp_size_options: List[int] = None,  # 可选的 SP size 列表 (必须是 2 的幂次)
                #  seq_bucket_size: int = 1024,
                 ):
        self.N = cluster_size
        self.mem_limit_gb = memory_limit_gb
        self.mem_limit_mb = memory_limit_gb * 1024
        self.costmodel = costmodel
        self.device_token_capacity = self.costmodel.token_capacity(self.mem_limit_gb)#那我们就假设他用的是最大的
        self.cluster_token_capacity = self.device_token_capacity * self.N
        # self.group_pool = {}
        # self.global_group_set = [] #indicate wheter a group is created
        self.hide_scipoutput = hide_scipoutput or hide_alloutput
        self.hide_alloutput = hide_alloutput
        self.concurrent = concurrent#并发
        self.scip_param_dict = scip_param_dict
        self.strategy = strategy
        self.redist_without_empty_group = redist_without_empty_group#是否保留空分组
        self.enable_ring_attn = enable_ring_attn
        # 支持的 attention 类型：ulysses (All-to-All) 和 ring (Ring Attention)
        if attn_types is None:
            self.attn_types = ['ulysses', 'ring'] if enable_ring_attn else ['ulysses']
        else:
            self.attn_types = attn_types
        
        # SP size 选项：用户可以自定义可选的 SP size 列表
        # 如果为 None，则使用默认的 [1, 2, 4, ..., cluster_size]
        if sp_size_options is not None:
            # 验证 sp_size_options 是否合法
            for sp in sp_size_options:
                if sp < 1 or sp > cluster_size:
                    raise ValueError(f"SP size {sp} 超出范围 [1, {cluster_size}]")
                if sp != 1 and (sp & (sp - 1)) != 0:
                    raise ValueError(f"SP size {sp} 不是 2 的幂次")
            self.sp_size_options = sorted(sp_size_options)
        else:
            # 默认：[1, 2, 4, ..., cluster_size]
            self.sp_size_options = None
        
    def token_info(self, seqs: List[Sequence]):
        print('\n============= Token Info =============')
        # print_seqs(seqs)
        print('Device Token Capacity: %d'%self.device_token_capacity)
        print('Cluster Token Capacity: %d'%(self.device_token_capacity*self.N))
        print('Total Token Number: %d'%sum(get_lens(seqs)))
        print('Total Sequence Number: %d'%len(seqs))

    def total_tokens(self, seqs: List[Sequence]):
        return sum(get_lens(seqs))
        #feasibility到底是在检验什么
        #这个feasibility 没办法使用
    def judge_feasibility(self, seqs, A, K, P, sp):
        if A is None:
            return -1
        M = -1
        for p in range(P):
            group_token= sum([seqs[k].seq * A[k, p] for k in range(K)]) / sp
            if group_token > self.device_token_capacity:
                return -1
            group_time = sum([self.costmodel.total_time_single(seqs[k].seq, sp) * A[k, p] for k in range(K)]) + self.costmodel.compute_bias()
            M = max(group_time, M)
        return M
        #同构的sp group然后采用均分
    def solve_homo_sp_even(self, seqs: List[Sequence], sp_size: int, group_num: int):
        sp_options = [sp_size] * group_num
        K, P = len(seqs), len(sp_options)
        A = np.zeros(shape=(K, P), dtype=np.int32)
        num = int(np.ceil(K / group_num))
        for group in range(group_num):
            start = num*group
            end = num*(group+1) if group < group_num - 1 else K
            A[start:end, group] = 1
        results = None
        M = self.judge_feasibility(seqs, A, K, P, sp_size)
        if M > 0:
            results = {
                'seqs': seqs,
                'sp_options': sp_options,
                'A': A,
                'M': M,
            }
        return results
    
    def solve_homo_sp_ffd_bfd(self, seqs: List[Sequence], sp_size: int, group_num: int, type: str = 'bfd'):
        sp_options = [sp_size] * group_num
        K, P = len(seqs), len(sp_options)
        if type == 'bfd':
            from galvatron.adacpsp_solver.utils import BestFitDecreasing
            A = BestFitDecreasing(seqs, self.device_token_capacity * sp_size, group_num)
        elif type == 'ffd':
            from galvatron.adacpsp_solver.utils import FirstFitDecreasing
            A = FirstFitDecreasing(seqs, self.device_token_capacity * sp_size, group_num)
        results = None
        M = self.judge_feasibility(seqs, A, K, P, sp_size)
        if M > 0:
            results = {
                'seqs': seqs,
                'sp_options': sp_options,
                'A': A,
                'M': M,
            }
        return results
    
    def solve_homo_sp_lp(self, seqs: List[Sequence], sp_size: int, group_num: int, return_groups: bool = False):
        sp_options = [sp_size] * group_num
        K, P = len(seqs), len(sp_options)
    
        ### ======== Define Optimization Problem ========
        model = Model("Homo-SP LP Optimization Problem")
        
        if self.hide_scipoutput:
            model.hideOutput()
        model.setParams(self.scip_param_dict)#scip param dict这里怎么设置

        # Optimization Target
        M = model.addVar(vtype="C", name="M", lb=0)  # Maximum of the execution time of SP groups
        model.setObjective(M, "minimize")
        #增加一个变量，lower bound设置为0，然后是连续变量
        # A[k,p] denotes whether the k^th sequence is put in the p^th SP group
        A = {(k, p): model.addVar(vtype="B", name=f"A_{k}_{p}") for k in range(K) for p in range(P)}
        #A里面包含着非常多的选项
        ### ======== Constraints of LP Problem ========
        # Each sequence can only be put into one SP group
        for k in range(K):
            model.addCons(quicksum(A[k, p] for p in range(P)) == 1)#每个sequence只能被放在一个里面
        
        for p in range(P):#对于每一个选择来说，
            sp_size = sp_options[p]#这个地方不只是加入sp的选项，还要加入cp的选项
            # SP group memory capacity limitation
            model.addCons(quicksum(seqs[k].seq * A[k, p] for k in range(K)) / sp_size <= self.device_token_capacity)
            #他要能满足才行
            # SP group execution time limitation
            model.addCons(quicksum(self.costmodel.total_time_single(seqs[k].seq, sp_size) * A[k, p] for k in range(K)) + self.costmodel.compute_bias() <= M)
            
            # Each SP group is occupied
            model.addCons(quicksum(A[k, p] for k in range(K)) >= 1)#这里是什么意思?
            #对于每一个选项来说，
        ### ======== Solve Optimization Problem ========
        if self.concurrent:
            model.solveConcurrent()
        else:
            model.optimize()
        status = model.getStatus()
        if not self.hide_alloutput:
            print("Model status:", status)
        if status not in ["optimal", "timelimit"]:
            results = None
        else:
            # Optimization Results
            results = {
                'seqs': seqs,
                'sp_options': sp_options,
                'A': np.zeros(shape=(K, P), dtype=np.int32),
                'M':model.getVal(M),
            }
            for p in range(P):
                for k in range(K):
                    results['A'][k, p] = round(model.getVal(A[k, p]))
            if return_groups:
                groups = []
                for p in range(P):
                    sp_size, group_seqs = sp_options[p], []
                    for k in range(K):
                        if results['A'][k, p] > 0:
                            group_seqs.append(seqs[k])
                    groups.append((sp_size, group_seqs, 1))
                results = groups
        return results
        
    def homo_sp_baseline_random(self, seqs: List[Sequence]):
        seqs = random.sample(seqs, len(seqs))
        baseline_results = {}
        sp = 1
        while sp <= self.N:
            baseline_results[sp] = self.solve_homo_sp_even(seqs, sp, self.N // sp)
            sp *= 2
        return baseline_results
        
    def homo_sp_baseline_ffd_bfd(self, seqs: List[Sequence], type: str = 'bfd'):
        baseline_results = {}
        sp = 1
        while sp <= self.N:
            baseline_results[sp] = self.solve_homo_sp_ffd_bfd(seqs, sp, self.N // sp, type=type)
            sp *= 2
        return baseline_results
    
    def homo_sp_baseline_lp(self, seqs: List[Sequence]):
        baseline_results = {}
        sp = 1
        while sp <= self.N:
            baseline_results[sp] = self.solve_homo_sp_lp(seqs, sp, self.N // sp)
            sp *= 2
        return baseline_results
    #我们在这里没有区分ring attention和ulysses的两种情况
    def get_sp_options(self, device_num, sp_max=0, cnt_max=0):
        """
        生成 virtual SP group 选项列表。
        现在返回 (sp_size, attn_type) 元组列表，支持 Ulysses 和 Ring Attention。
        
        如果设置了 self.sp_size_options，则只使用其中的 SP size。
        """
        sp_options = []
        
        # 确定要遍历的 SP size 列表
        if self.sp_size_options is not None:
            sp_sizes_to_use = [sp for sp in self.sp_size_options if sp <= device_num]
        else:
            # 默认：[1, 2, 4, ..., device_num]
            sp_sizes_to_use = []
            sp = 1
            while sp <= device_num:
                sp_sizes_to_use.append(sp)
                sp *= 2
        
        for sp in sp_sizes_to_use:
            if sp_max and cnt_max and sp < sp_max:
                max_num = int((device_num-(sp_max * cnt_max))//sp)#排除掉给最大的还剩多少
            else:
                max_num = int(device_num//sp)
            
            if max_num <= 0:
                continue
                
            # 为每种 attention 类型生成对应的 sp_options
            if sp == 1:
                sp_options.extend([(sp, None)] * max_num)
            else:
                for attn_type in self.attn_types:
                    sp_options.extend([(sp, attn_type)] * max_num)
        
        return sp_options  # 生成 virtual 集合，现在是 (sp_size, attn_type) 元组列表
        #用来进行计算
    def calculate_min_sp(self, seq, capacity):
        min_sp = int(np.ceil(seq / self.device_token_capacity))
        log_2 = np.log(min_sp)/np.log(2)
        # print(log_2, int(np.ceil(log_2)))
        min_sp = 2 ** int(np.ceil(log_2))
        return min_sp
        
    def get_seq_min_sp_size(self, seqs):
        return [self.calculate_min_sp(sq.seq, self.device_token_capacity) for sq in seqs]#计算每一个sequence的最小 sp size
    #我还没有理解这里的桶是社么意思
    def get_bucket_min_sp_size(self, buckets):#这个有点问题，因为桶的最小，不是真的最小，因为有多条序列
        return [self.calculate_min_sp(bkt.boundary, self.device_token_capacity) for bkt in buckets]
        
    def get_sp_options_prune(self, seqs):
        seq_min_sp = self.get_seq_min_sp_size(seqs)#他计算的是每一条
        sp_max = max(seq_min_sp)#然后计算这个最大的
        cnt_max = Counter(seq_min_sp)[sp_max]#然后这个最大的有几个？
        sp_options = self.get_sp_options(self.N, sp_max, cnt_max) 
        # print(f"sp_max: {sp_max}, cnt_max: {cnt_max}")
        if not self.hide_alloutput:
            print(f"seq_min_sp: {seq_min_sp}")
            print(f"sp_options: {sp_options}")
        return sp_options
        
    # 辅助函数：从 sp_options 中提取 sp_size
    def _get_sp_size(self, sp_option):
        """从 sp_option 中提取 sp_size，兼容旧格式（int）和新格式（tuple）"""
        if isinstance(sp_option, tuple):
            return sp_option[0]
        return sp_option
    
    # 辅助函数：从 sp_options 中提取 attn_type
    def _get_attn_type(self, sp_option):
        """从 sp_option 中提取 attn_type，兼容旧格式（int）和新格式（tuple）"""
        if isinstance(sp_option, tuple):
            return sp_option[1]
        return 'ulysses'  # 默认使用 ulysses
    
    # No bucket
    def solve_flexSP(self, seqs: List[Sequence]):
        sp_options = self.get_sp_options_prune(seqs)#获得sp options
        seq_min_sp_size = self.get_seq_min_sp_size(seqs)#然后获得每一个sequence的最小sp size
        
        K, P = len(seqs), len(sp_options)#有多少序列，有多少选择，sp options里面肯定是包含
        total_seq_num = K

        ### ======== Define Optimization Problem ========
        model = Model("FlexSP Optimization Problem")
        
        if self.hide_scipoutput:
            model.hideOutput()
        model.setParams(self.scip_param_dict)
        
        # Optimization Target
        M = model.addVar(vtype="C", name="M", lb=0)  # Maximum of the execution time of SP groups
        model.setObjective(M, "minimize")
        
        # A[k,p] denotes whether the k^th sequence is put in the p^th SP group
        A = {(k, p): model.addVar(vtype="B", name=f"A_{k}_{p}") for k in range(K) for p in range(P)}
        
        # m[p] denotes whether the p^th SP group is occupied
        m = {p: model.addVar(vtype="B", name=f"m_{p}") for p in range(P)}
        #是否被使用
        ### ======== Constraints of LP Problem ========
        # Each sequence can only be put into one SP group
        for k in range(K):
            model.addCons(quicksum(A[k, p] for p in range(P)) == 1)#只能选择一个
            
            # Sequences cannot be put into group with sp size less than seq_min_sp_size
            for p in range(P):
                sp_size = self._get_sp_size(sp_options[p])
                if sp_size < seq_min_sp_size[k]:
                    model.addCons(A[k, p] == 0)
        
        for p in range(P):
            sp_size = self._get_sp_size(sp_options[p])
            attn_type = self._get_attn_type(sp_options[p])
            
            # SP group memory capacity limitation (考虑 attn_type 对显存的影响)
            token_capacity = self.costmodel.token_capacity(self.mem_limit_gb, sp_size=sp_size, attn_type=attn_type)
            model.addCons(quicksum(seqs[k].seq * A[k, p] for k in range(K)) / sp_size <= token_capacity)

            # SP group execution time limitation (使用对应的 attn_type)
            # 使用 Big-M 方法：当 m[p]=0（group 未使用）时，约束不限制 M
            # 当 m[p]=1 时，约束为：sum(time) + bias <= M
            # 当 m[p]=0 时，约束为：0 + bias - BIG_M <= M，总是满足
            BIG_M = 1e9  # 足够大的常数
            compute_bias = self.costmodel.compute_bias(sp_size, attn_type)
            model.addCons(quicksum(self.costmodel.total_time_single(seqs[k].seq, sp_size, attn_type) * A[k, p] for k in range(K)) + compute_bias - BIG_M * (1 - m[p]) <= M)
            #在这里从修改total time single修改为总的时间
            # Ensure no sequence is assigned to group p if m[p] == 0
            model.addCons(quicksum(A[k, p] for k in range(K)) <= total_seq_num * m[p])
            
            # Ensure there are sequences assigned to group p if m[p] == 1
            model.addCons(quicksum(A[k, p] for k in range(K)) >= m[p])
        
        # Device number constraint (只使用 sp_size)
        model.addCons(quicksum(m[p] * self._get_sp_size(sp_options[p]) for p in range(P)) == self.N)

        ### ======== Solve Optimization Problem ========
        if self.concurrent:
            model.solveConcurrent()
        else:
            model.optimize()
        status = model.getStatus()
        if not self.hide_alloutput:
            print("Model status:", status)
        if status not in ["optimal", "timelimit"]:
            return None
        if not self.hide_alloutput:
            print("Optimal value (minimum maximum group execution time):", model.getVal(M))
        # Optimization Results
        results = {
            'seqs': seqs,
            'sp_options': sp_options,
            'A': np.zeros(shape=(K, P), dtype=np.int32),
            'M': model.getVal(M),#这里会有一个最长时间，我们到时候需要看这个最长时间来自于谁
            'm': np.zeros(shape=(P), dtype=np.int32),
        }
        
        for p in range(P):
            results['m'][p] = round(model.getVal(m[p]))
            for k in range(K):
                results['A'][k, p] = round(model.getVal(A[k, p]))
        return results#在这里获得了一个
    
    def bucket_seqs(self, seqs: List[Sequence], bucket_num: int):
        if bucket_num > 0: # for bucket_alg=dp
            bucket_error_ths = (sum(get_lens(seqs)) + self.cluster_token_capacity)/2
            bucket_num = min(len(seqs), bucket_num)
            while True:
                buckets, avg_error = bucketing_seqs(seqs, bucket_num)
                bucket_total_token = sum([bucket.boundary * bucket.size for bucket in buckets])
                if bucket_total_token <= bucket_error_ths:
                    return buckets, avg_error, bucket_num
                bucket_num += 1
        elif bucket_num < 0: # for bucket_alg=even_dist
            bucket_num  = -bucket_num
            max_seq = max(seq.seq for seq in seqs)
            seq_chunk = max_seq // bucket_num
            buckets = []
            avg_error = 0.
            seqs.sort()
            idx = 0
            real_bucket_num = 0 
            for i in range(bucket_num):
                bound = (i + 1) * seq_chunk
                bucket = SeqBucket(bound)
                sel_seqs = []
                while idx < len(seqs) and seqs[idx].seq < bound:
                    sel_seqs.append(seqs[idx])
                    idx += 1
                if len(sel_seqs) > 0:
                    bucket.add_seqs(sel_seqs)
                    for seq in sel_seqs:
                        avg_error += (bound - seq.seq)
                    buckets.append(bucket)
                    real_bucket_num += 1
            avg_error /= len(seqs)
            return buckets, avg_error, real_bucket_num
        else:
            raise NotImplementedError()
    
    def solve_flexSP_bucket_seqs(self, seqs: List[Sequence], bucket_num: int = 10):
        sp_options = self.get_sp_options_prune(seqs)

        # bucket_num = min(len(seqs), bucket_num)
        # buckets, avg_error = bucketing_seqs(seqs, bucket_num)
        buckets, avg_error, bucket_num = self.bucket_seqs(seqs, bucket_num)
        if len(buckets) >= len(seqs):
            # print('[Warning] Fallen back into solver w.o. sequence bucketing!')
            return self.solve_flexSP(seqs)
        if not self.hide_alloutput:
            print( "Bucket Num:", bucket_num, "Average total error:", avg_error)
            for bucket in buckets:
                bucket.print()
        bucket_min_sp_size = self.get_bucket_min_sp_size(buckets)
        
        K, P = len(buckets), len(sp_options)
        total_seq_num = sum([bucket.size for bucket in buckets])

        ### ======== Define Optimization Problem ========
        model = Model("FlexSP Optimization Problem")
        
        if self.hide_scipoutput:
            model.hideOutput()
        model.setParams(self.scip_param_dict)
        
        # Optimization Target
        M = model.addVar(vtype="C", name="M", lb=0)  # Maximum of the execution time of SP groups
        model.setObjective(M, "minimize")
        
        # A[k,p] denotes the number of the sequences in k^th bucket that are put in the p^th SP group
        A = {(k, p): model.addVar(vtype="I", lb=0, name=f"A_{k}_{p}") for k in range(K) for p in range(P)}
        
        # m[p] denotes whether the p^th SP group is occupied
        m = {p: model.addVar(vtype="B", name=f"m_{p}") for p in range(P)}

        ### ======== Constraints of LP Problem ========
        # All sequences in each bucket are put into groups
        for k in range(K):
            model.addCons(quicksum(A[k, p] for p in range(P)) == buckets[k].size)
            
            # Sequences cannot be put into group with sp size less than seq_min_sp_size
            for p in range(P):
                sp_size = self._get_sp_size(sp_options[p])
                if sp_size < bucket_min_sp_size[k]:
                    model.addCons(A[k, p] == 0)
        
        for p in range(P):
            sp_size = self._get_sp_size(sp_options[p])
            attn_type = self._get_attn_type(sp_options[p])
            
            # SP group memory capacity limitation (考虑 attn_type 对显存的影响)
            token_capacity = self.costmodel.token_capacity(self.mem_limit_gb, sp_size=sp_size, attn_type=attn_type)
            model.addCons(quicksum(buckets[k].boundary * A[k, p] for k in range(K)) / sp_size <= token_capacity)

            # SP group execution time limitation (使用对应的 attn_type)
            # 使用 Big-M 方法：当 m[p]=0（group 未使用）时，约束不限制 M
            BIG_M = 1e9
            compute_bias = self.costmodel.compute_bias(sp_size, attn_type)
            model.addCons(quicksum(self.costmodel.total_time_single(buckets[k].boundary, sp_size, attn_type) * A[k, p] for k in range(K)) + compute_bias - BIG_M * (1 - m[p]) <= M)
            
            # Ensure no sequence is assigned to group p if m[p] == 0
            model.addCons(quicksum(A[k, p] for k in range(K)) <= total_seq_num * m[p])
            
            # Ensure there are sequences assigned to group p if m[p] == 1
            model.addCons(quicksum(A[k, p] for k in range(K)) >= m[p])
        
        # Device number constraint (只使用 sp_size)
        model.addCons(quicksum(m[p] * self._get_sp_size(sp_options[p]) for p in range(P)) == self.N)

        # Add initial feasible solution
        from galvatron.flexsp_solver.utils import generate_balanced_initial_solution
        # 获取唯一的 sp_size 列表用于初始解
        unique_sp_sizes = list(set([self._get_sp_size(opt) for opt in sp_options]))
        for sp_size in unique_sp_sizes:
            # 使用默认的 ulysses token capacity 作为初始解参考
            generate_balanced_initial_solution(model, sp_options, M, m, A, buckets, K, P, sp_size, self.hide_alloutput, self.device_token_capacity, self.costmodel, self._get_sp_size)

        ### ======== Solve Optimization Problem ========
        if self.concurrent:
            model.solveConcurrent()
        else:
            model.optimize()
        status = model.getStatus()
        if not self.hide_alloutput:
            print("Model status:", status)
        if status not in ["optimal", "timelimit"]:
            return None
        if not self.hide_alloutput:
            print("Optimal value (minimum maximum group execution time):", model.getVal(M))
        
        # Optimization Results
        results = {
            'K': K,
            'P': P,
            'sp_options': sp_options,
            'A': np.zeros(shape=(K, P), dtype=np.int32),
            'm': np.zeros(shape=(P), dtype=np.int32),
            'M': model.getVal(M),
            'seqs': seqs,
            'buckets': buckets,
        }
        
        # for k in range(K):
        #     for p in range(P):
        #         print(model.getVal(A[k, p]), end=' ')
        #     print()
        
        for p in range(P):
            results['m'][p] = round(model.getVal(m[p]))
            for k in range(K):
                results['A'][k, p] = round(model.getVal(A[k, p]))
        return results

    def get_total_time(self, seqs, sp_size, m=None, attn_type='ulysses'):
        seqlens = get_lens(seqs)
        if m is not None:
            return sum([self.costmodel.total_time_single(seq, sp_size, attn_type) for seq in seqlens]) / m + self.costmodel.compute_bias(sp_size, attn_type)
        else:
            return self.costmodel.total_time(seqlens, sp_size, attn_type)

    def get_groups_total_token(self, groups):
        total_token = 0
        # 兼容新格式 (sp_size, attn_type, seqs) 和旧格式 (sp_size, seqs)
        for group in groups:
            if len(group) == 3:
                sp_size, attn_type, seqs = group
            else:
                sp_size, seqs = group
            total_token += sum(get_lens(seqs))
        return total_token

    def print_group_seqs_info(self, seqs, sp_size, m=None, attn_type='ulysses'):
        seqlens = get_lens(seqs)
        if m is not None and m > 1:
            memory = self.costmodel.total_memory([sum(seqlens)/m], sp_size, attn_type)
            total_time = sum([self.costmodel.total_time_single(seq, sp_size, attn_type) for seq in seqlens]) / m + self.costmodel.compute_bias(sp_size, attn_type)
            total_token = sum(seqlens) / sp_size / m
        else:
            memory = self.costmodel.total_memory(seqlens, sp_size, attn_type)
            total_time = self.costmodel.total_time(seqlens, sp_size, attn_type)
            total_token = sum(seqlens) / sp_size
        attn_str = f", Attn = {attn_type}" if attn_type != 'ulysses' else ""
        if m is not None:
            print(f"SP group: SP = {sp_size}{attn_str}, Num = {m}, Time = {total_time:.2f}, Token Num = {total_token}, Memory = {memory:.1f}, Sequences = {seqlens}")
        else:
            print(f"SP group: SP = {sp_size}{attn_str}, Time = {total_time:.2f}, Token Num = {total_token}, Memory = {memory:.1f}, Sequences = {seqlens}")

    def get_min_valid_microbatch_num(self, seqs_gb: List[Sequence], chunk_alg: str = 'sort_consec'):
        total_tokens = sum(get_lens(seqs_gb))
        mb_num = int(np.ceil(total_tokens / self.cluster_token_capacity))
        while True:
            seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)
            valid = True
            for seqs_mb in seqs_mb_all:
                if self.total_tokens(seqs_mb) >= self.cluster_token_capacity:
                    valid = False
                    break
            if valid:
                break
            mb_num += 1
        return mb_num

    def solve_flexSP_globalbatch(self, seqs_gb: List[Sequence], chunk_alg: str = 'sort_consec', bucket_num: int = 16):
        mb_num = self.get_min_valid_microbatch_num(seqs_gb, chunk_alg)

        globalbatch_groups, globalbatch_results = [], []
        while True:
            if not self.hide_alloutput:
                print(f'\n=========== Trying microbatch size = {mb_num} ===========')
            seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)
            feasible = True
            for seqs_mb in seqs_mb_all:
                if not self.hide_alloutput:
                    self.token_info(seqs_mb)
                results = self.solve_flexSP_bucket_seqs(seqs_mb, bucket_num = bucket_num)
                # results = self.solve_flexSP_bucket_seqs_groups(seqs_mb, bucket_num = 16)
                if results is None:
                    feasible = False
                    globalbatch_groups, globalbatch_results = [], []
                    break
                groups = self.show_results(results)
                globalbatch_groups.append(groups)
                globalbatch_results.append(results)
            if feasible:
                if not self.hide_alloutput:
                    print(f'\n=========== Success with microbatch size = {mb_num} ! ===========')
                break
            if not self.hide_alloutput:
                print(f'\n=========== Failed microbatch size = {mb_num} ! ===========')
            mb_num += 1
        return globalbatch_groups, globalbatch_results

    def solve_flexSP_globalbatch_mp(self, seqs_gb: List['Sequence'], chunk_alg: str = 'sort_consec', bucket_num: int = 16):
        from galvatron.flexsp_solver.multiprocess_utils import serialize_seqs, deserialize_seq_groups, mp_worker
        mb_num = self.get_min_valid_microbatch_num(seqs_gb, chunk_alg)
        
        while True:
            globalbatch_groups, globalbatch_results = [], []
            if not self.hide_alloutput:
                print(f'=========== Trying microbatch size = {mb_num} ===========')
            
            seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)
            seqs_mb_all = [serialize_seqs(seqs_mb) for seqs_mb in seqs_mb_all]

            manager = mp.Manager()
            stop_flag = manager.Value('i', 0)
            results_list = manager.list()
            
            pool = mp.Pool(processes=mb_num)

            async_results = [
                pool.apply_async(mp_worker, args=(
                    seqs_mb, stop_flag, 
                    self.hide_alloutput, 
                    self.solve_flexSP_bucket_seqs, 
                    self.show_results,
                    self.token_info,
                    bucket_num)) 
                for seqs_mb in seqs_mb_all
            ]
            
            processed_results = [False] * len(async_results)

            feasible = True
            try:
                while not all(processed_results):
                    for i, ar in enumerate(async_results):
                        if not processed_results[i] and ar.ready():
                            result = ar.get()
                            if result is None:
                                feasible = False
                                stop_flag.value = 1
                                pool.terminate()
                                pool.join()
                                globalbatch_groups, globalbatch_results = [], []
                                break
                            else:
                                groups, results = result
                                groups = deserialize_seq_groups(groups)
                                globalbatch_groups.append(groups)
                                globalbatch_results.append(results)
                                processed_results[i] = True
                    if stop_flag.value == 1:
                        feasible = False
                        break
            except Exception as e:
                pool.terminate()
                pool.join()
                raise e

            if feasible:
                if not self.hide_alloutput:
                    print(f'=========== Success with microbatch size = {mb_num} ! ===========')
                pool.close()
                pool.join()
                break

            if not self.hide_alloutput:
                print(f'\n=========== Failed microbatch size = {mb_num} ! ===========')
            
            pool.close()
            pool.join()
            mb_num += 1

        return globalbatch_groups, globalbatch_results
    
    def get_globalbatch_total_time(globalbatch_results: List[Dict]):
        total_time = 0.
        for result in globalbatch_results:
            total_time += result['M']
        return total_time
    
    def homo_sp_baseline_ffd_bfd_globalbatch_fix_sp(self, seqs_all_iter: List[List[Sequence]], type: str = 'bfd'):
        sp = 1
        for seqs in seqs_all_iter:
            sp_min, (groups, results) = self.homo_sp_baseline_ffd_bfd_globalbatch(seqs, type, sp_select_rule='min_sp')
            sp = max(sp, sp_min)
        return sp
        
    def homo_sp_baseline_ffd_bfd_globalbatch(self, seqs: List[Sequence], type: str = 'bfd', sp_select_rule: str = 'adaptive', sp_size: int = 0, fill_empty : bool = True, ignore_strategies : List = []):
        baseline_results = {}
        
        if sp_select_rule == 'fix_sp':
            if sp_size == 0:
                sp_size = self.fix_sp_size
            globalbatch_groups, globalbatch_results = self.solve_homo_sp_ffd_bfd_globalbatch(seqs, sp_size, self.N // sp_size, type=type, fill_empty=fill_empty, even_distribute=sp_size>=16)
            assert globalbatch_results is not None
            baseline_results[sp_size] = (globalbatch_groups, globalbatch_results)
        else:
            sp = 1
            sp_min = self.N + 1
            while sp <= self.N:
                if sp in ignore_strategies:
                    sp *= 2
                    continue
                globalbatch_groups, globalbatch_results = self.solve_homo_sp_ffd_bfd_globalbatch(seqs, sp, self.N // sp, type=type, fill_empty=fill_empty, even_distribute=sp>=16)
                if globalbatch_results is not None:
                    baseline_results[sp] = (globalbatch_groups, globalbatch_results)
                    sp_min = min(sp_min, sp)
                sp *= 2
        
        assert len(baseline_results) != 0
        if sp_select_rule == 'adaptive':
            sp_best, time_best = 0, 1e20
            for sp, (globalbatch_groups, globalbatch_results) in baseline_results.items():
                time = sum([results['M'] for results in globalbatch_results])
                if sp_best == 0 or time < time_best:
                    sp_best, time_best = sp, time
            return baseline_results[sp_best]
        elif sp_select_rule == 'min_sp':
            return baseline_results[sp_min]
        elif sp_select_rule == 'fix_sp':
            if sp_size == 0:
                sp_size = self.fix_sp_size
            assert sp_size in baseline_results
            return baseline_results[sp_size]
        elif sp_select_rule == 'all_sp':
            return baseline_results
    
    def solve_homo_sp_ffd_bfd_globalbatch(self, seqs_gb: List[Sequence], sp_size: int, group_num_per_mb: int, type: str = 'bfd', fill_empty: bool = True, even_distribute: bool = True):
        group_capacity = max(self.device_token_capacity * sp_size * 0.9, max(get_lens(seqs_gb)))
        if type == 'bfd':
            from galvatron.adacpsp_solver.utils import BestFitDecreasing
            A = BestFitDecreasing(seqs_gb, group_capacity)
        elif type == 'ffd':
            from galvatron.adacpsp_solver.utils import FirstFitDecreasing
            A = FirstFitDecreasing(seqs_gb, group_capacity)
        
        if A is None:
            return None, None
        
        group_num = A.shape[-1]
        mb_num = (group_num + group_num_per_mb - 1) // group_num_per_mb
        empty_group_num = group_num_per_mb * mb_num - group_num
        sp_options = [sp_size] * group_num_per_mb
        K, P = len(seqs_gb), len(sp_options)
        A = np.append(A, np.zeros(shape=(K, empty_group_num), dtype=np.int32), axis=1)
        
        def _token_lensum(groupA: np.ndarray) -> int:
            return sum([groupA[k] * seqs_gb[k].seq for k in range(groupA.shape[0])])

        if fill_empty:
            group_id_0, group_id_1, group_id_2 = (mb_num-1) * group_num_per_mb, mb_num * group_num_per_mb - empty_group_num, mb_num * group_num_per_mb
            if empty_group_num > 0:
                for i in range(group_id_1, group_id_2):
                    j = group_id_1 - 1
                    while j >= 0:
                        if np.sum(A[:,j]) > 1:
                            k = 0
                            while A[k,j] == 0:
                                k += 1
                            break
                        j -= 1
                    if j < 0:
                        assert False
                    A[k,j], A[k,i] = 0, 1
            
            
            even_distribute = even_distribute and (empty_group_num > 0 or self.redist_without_empty_group)
            # To evenly distribute all tokens in the last microbatch
            if even_distribute:
                mb_token_lensum = sum([_token_lensum(A[:,i]) for i in range(group_id_0, group_id_2)])
                group_token_lensum_avg = mb_token_lensum // group_num_per_mb
                for i in range(group_id_0 + 1, group_id_2):
                    if _token_lensum(A[:,i]) > group_token_lensum_avg:
                        continue
                    for j in range(group_id_0, i):
                        k = 0
                        while _token_lensum(A[:,i]) < group_token_lensum_avg and _token_lensum(A[:,j]) > group_token_lensum_avg and np.sum(A[:,j]) > 1:
                            if A[k,j] == 1 and _token_lensum(A[:,i])+seqs_gb[k].seq <= group_capacity:
                                A[k,j], A[k,i] = 0, 1
                            k += 1
        
        
        globalbatch_groups, globalbatch_results = [], []
        result = None
        feasible = True
        for i in range(mb_num):
            A_mb = A[:, i*group_num_per_mb:(i+1)*group_num_per_mb]
            M = self.judge_feasibility(seqs_gb, A_mb, K, P, sp_size)
            if M > 0:
                result = {
                    'seqs': seqs_gb,
                    'sp_options': sp_options,
                    'A': A_mb,
                    'M': M,
                }
            else:
                return None, None
            groups = []
            for j in range(group_num_per_mb):
                group_seqs = []
                for k in range(K):
                    if A_mb[seqs_gb[k].id, j] == 1:
                        group_seqs.append(seqs_gb[k])
                groups.append((sp_size, group_seqs))
            globalbatch_groups.append(groups)
            globalbatch_results.append(result)
        return globalbatch_groups, globalbatch_results

    def solve_flexSP_globalbatch_mp_gbmb(self, seqs_gb: List['Sequence'], chunk_alg: str = 'sort_consec', bucket_alg: Literal['no_bucket', 'even_dist', 'dp'] = 'dp', mb_option_num: int = 5, bucket_num: int = 16):
        from galvatron.flexsp_solver.multiprocess_utils import serialize_seqs, deserialize_seq_groups, mp_microbatch_worker

        mb_num = self.get_min_valid_microbatch_num(seqs_gb, chunk_alg)
        mb_num_options = [mb_num + i for i in range(mb_option_num)]

        seqs_gb_serialized = serialize_seqs(seqs_gb)

        manager = mp.Manager()
        result_dict = manager.dict()
        processes = []
        for mb_num in mb_num_options:
            if bucket_alg == 'dp':
                bucket_num = bucket_num
            elif bucket_alg == 'no_bucket':
                bucket_num = 1e10
            elif bucket_alg == 'even_dist':
                bucket_num = -bucket_num
            else:
                raise NotImplementedError(f"Bucketing algorithm {bucket_alg} is not implemented.")
            
            p = mp.Process(target=mp_microbatch_worker, args=(
                seqs_gb_serialized, mb_num, 
                self.hide_alloutput, 
                self.solve_flexSP_bucket_seqs, 
                self.show_results,
                self.token_info,
                result_dict,
                chunk_alg,
                bucket_num,
            ))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

        feasible_results = []
        for mb_num, (globalbatch_time, globalbatch_groups, globalbatch_results) in result_dict.items():
            if globalbatch_groups is not None and globalbatch_results is not None:
                globalbatch_groups = [deserialize_seq_groups(groups) for groups in globalbatch_groups]
                feasible_results.append((mb_num, globalbatch_time, globalbatch_groups, globalbatch_results))
        
        if feasible_results:
            best_result = min(feasible_results, key=lambda x: x[1])
            best_mb_num, best_time, best_groups, best_results = best_result
            if not self.hide_alloutput:
                print(f'=========== Best microbatch size = {best_mb_num} with time = {best_time} ===========')
            return best_groups, best_results
        else:
            if not self.hide_alloutput:
                print('=========== All mb_num attempts failed! ===========')
            return [], []


    def show_results(self, results):
        if results is None:
            print('Infeasible!')
            return
        seqs, sp_options, A, M = results['seqs'], results['sp_options'], results['A'], results['M']
        P = len(sp_options)
        m = results['m'] if 'm' in results.keys() else np.ones(shape=(P), dtype=np.int32)
        if 'buckets' in results.keys():
            buckets = results['buckets']
            K = len(buckets)
        else:
            buckets = None
            K = len(seqs)
        
        for k in range(K):
            if buckets is None:
                assert(np.sum(A[k]) == 1)
            else:
                assert(np.sum(A[k]) == buckets[k].size)
        # 使用辅助函数提取 sp_size 进行设备数量验证
        assert(sum([m[p] * self._get_sp_size(sp_options[p]) for p in range(P)]) == self.N)
        
        minimized_time, groups = -1, []
        if buckets is not None:
            bucket_minimized_time, bucketing_groups = -1, []
        group_replica = False
        for p in range(P):
            if m[p] == 0:
                continue
            elif m[p] > 1:
                group_replica = True
            # 解包 sp_option，获取 sp_size 和 attn_type
            sp_size = self._get_sp_size(sp_options[p])
            attn_type = self._get_attn_type(sp_options[p])
            group_seqs = []
            group_bucket_seqs = [] if buckets is not None else None
            for k in range(K):
                if A[k,p] > 0:
                    if buckets is None:
                        group_seqs.append(seqs[k])
                    else:
                        group_seqs.extend(buckets[k].random_pop_seqs(A[k,p]))
                        for i in range(A[k, p]):
                            group_bucket_seqs.append(Sequence(seq=buckets[k].boundary, id=-1))
            # groups 现在包含 (sp_size, attn_type, group_seqs, m[p])
            groups.append((sp_size, attn_type, group_seqs, m[p]))
            minimized_time = max(minimized_time, self.get_total_time(group_seqs, sp_size, m[p], attn_type))
            if buckets is not None:
                bucketing_groups.append((sp_size, attn_type, group_bucket_seqs, m[p]))
                bucket_minimized_time = max(bucket_minimized_time, self.get_total_time(group_bucket_seqs, sp_size, m[p], attn_type))

        if not self.hide_alloutput:
            print(f'============= Minimized Time: {minimized_time:.2f} =============')
            for sp_size, attn_type, group_seqs, m in groups:
                self.print_group_seqs_info(group_seqs, sp_size, m, attn_type)

        if group_replica:
            groups_final = []
            for sp_size, attn_type, group_seqs, m in groups:
                if m == 1:
                    groups_final.append((sp_size, attn_type, group_seqs, m))
                else:
                    group = self.solve_homo_sp_lp(group_seqs, sp_size, m, return_groups = True)
                    if group is None:
                        assert False
                    # solve_homo_sp_lp 返回的是旧格式，需要添加 attn_type
                    for g in group:
                        groups_final.append((g[0], attn_type, g[1], g[2]))
            groups = groups_final
            minimized_time = -1
            for sp_size, attn_type, group_seqs, m in groups:
                minimized_time = max(minimized_time, self.get_total_time(group_seqs, sp_size, m, attn_type))
            
            if not self.hide_alloutput:
                print(f'============= Minimized Time: {minimized_time:.2f} =============')
                for sp_size, attn_type, group_seqs, m in groups:
                    self.print_group_seqs_info(group_seqs, sp_size, m, attn_type)
        
        results['M'] = minimized_time
            
        # 返回格式：(sp_size, attn_type, group_seqs)
        groups = [(sp_size, attn_type, group_seqs) for sp_size, attn_type, group_seqs, m in groups]
        return groups

def check_costmodel():
    cluster_size = 16
    sp_size = 16
    
    costmodel = flexSPCostModel(
                 cluster_size = cluster_size,
                 hidden_size = 4096,
                 layer_num = 32,
                 param_size_B = 7, 
                 zero_stage = 3,
                 mixed_precision = True,
                 act_per_token = 4.71, 
                 cpt_alpha1 = 5.128 * 1e-6, 
                 cpt_alpha2  = 183.9576 * 1e-3, 
                 cpt_beta1 = 629.3563,
                 alltoall_bandwidth_dict_gbs = {1: 1e10, 2: 154, 4: 137, 8: 121, 16: 8, 32:6, 64:5},
                )
    
    print('\n--------------- [Single GPU Computation Time Modeling] ---------------')
    for seq in [256, 512, 1024, 2048, 4096]:
        for num in [1, 2, 4, 8, 16, 32]:
            time = costmodel.total_time([seq] * num, 1) / 1e3
            print(f'[Seq: {seq}, Num: {num}] Time: {time:.4f}')
    
    print('\n--------------- [16 GPUs EndtoEnd Time Modeling] ---------------')
    for seq in [256, 512, 1024, 2048, 4096, 8192]:
        for num in [1, 2, 4, 8, 16, 32]:
            time = costmodel.total_time([seq] * num, 16) / 1e3
            print(f'[Seq: {seq}, Num: {num}] Time: {time:.4f}')
            
    for seq in [256, 512, 1024, 2048, 4096, 8192]:
        for num in [1, 2, 4, 8, 16, 32]:
            costmodel.check_costmodel([seq] * num, 16)

def read_dataset(name = 'github', seq_limit=65536, world_size=64):
    with open('../datasets/'+name+'.txt', 'r') as file:
        lines = file.readlines()
        data = [int(l.strip('\n')) for l in lines]
    new_data = []
    idx = 0
    for d in data:
        sentence_len = d
        pad_len = ((sentence_len - 1) // (2 * world_size) + 1) * (2 * world_size) #force the length to be multiple of 2 * world_size
        if pad_len <= seq_limit - 2 * world_size:
            new_data.append(pad_len)
            idx += 1
    return new_data

def get_global_batch(data: List[int], iter: int = 0, global_batch_size: int = 64):
    return data[iter * global_batch_size: (iter + 1) * global_batch_size]


def solver_args(parser):
    group = parser.add_argument_group(title="Solver Arguments")

    group.add_argument(
        "--cluster_size", type=int, default=64,
    )
    group.add_argument(
        "--memory_limit_gb", type=int, default=28,
    )
    group.add_argument(
        "--model_size", type=str, default='gpt-7b', choices=['gpt-7b', 'gpt-13b', 'gpt-30b'],
    )
    group.add_argument(
        "--time_limit", type=int, default=5,
    )
    group.add_argument(
        "--dataset", type=str, default='github',# choices=['github', 'common_crawl', 'wikipedia'],
    )
    group.add_argument(
        "--seq_limit_k", type=int, default=128,# choices=[32, 64, 128, 256],
    )
    group.add_argument(
        "--global_batch_size", type=int, default=1024,
    )
    group.add_argument(
        "--method_type", type=str, default='adaptive', choices=['flexSP', 'adaptive', 'static']
    )
    group.add_argument(
        "--chunk_alg", type=str, default='sort_consec', choices=['sort_consec', 'distribution'],
    )
    group.add_argument(
        "--bucket_alg", type=str, default='dp', choices=['no_bucket', 'even_dist', 'dp'], help="Algorithm for FlexSP sequence bucketing"
    )
    group.add_argument(
        "--iter_num", type=int, default=30,
    )
    group.add_argument(
        "--show_strategy", type=int, default=0,
    )
    group.add_argument(
        "--start_iter", type=int, default=3,
    )
    group.add_argument(
        "--no-seq-bucket", action="store_true",
    )
    group.add_argument(
        "--redist-without-empty-group", action="store_true",
    )
    group.add_argument(
        "--enable-ring-attn", action="store_true", default=False,
        help="Enable Ring Attention as an alternative strategy to Ulysses SP"
    )
    group.add_argument(
        "--p2p-bandwidth-gbs", type=float, default=200.0,
        help="P2P bandwidth in GB/s for Ring Attention communication"
    )
    group.add_argument(
        "--ring-overlap-efficiency", type=float, default=0.5,
        help="Communication-computation overlap efficiency for Ring Attention (0.0-1.0)"
    )

    return parser

optimizer_param_dict = {
    'gpt-7b' :{
        'act_per_token' : 4.480441895,
         'cpt_alpha1' : 5.128 * 1e-6,
         'cpt_alpha2' : 183.9576 * 1e-3,
         'cpt_beta1' : 629.3563,
         'hidden_size': 4096,
         'layer_num': 32,
         'param_size_B': {4: 6.51, 192: 7.1174468994140625, 384: 7.8498687744140625},
    },
    'gpt-13b' : {
        'act_per_token': 4.220439453,
         'cpt_alpha1' : 9.2852 * 1e-6,
         'cpt_alpha2' : 306.0189 * 1e-3,
         'cpt_beta1' : 1132.5632,
         'hidden_size': 5120,
         'layer_num': 40,
         'param_size_B': {4: 12.3568, 192: 13.11605453491211, 384: 14.03158187866211},
    },
    'gpt-30b' : {
        'act_per_token' : 3.417381836,
         'cpt_alpha1' : 15.4262 * 1e-6,
         'cpt_alpha2' : 803.5742 * 1e-3,
         'cpt_beta1' : 2789.3644,
         'hidden_size': 6656,
         'layer_num': 60,
         'param_size_B': {4: 30.538, 192: 31.52513885498047, 384: 32.71532440185547},
    }
}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser = solver_args(parser)
    args = parser.parse_args()
    model_config = optimizer_param_dict[args.model_size]
    
    random.seed(0)
    
    cluster_size = args.cluster_size
    memory_limit_gb = args.memory_limit_gb

    # alltoall_bandwidth_dict_gbs = {1: 1e10, 2: 154, 4: 137, 8: 121, 16: 8, 32:6, 64:5}
    # alltoall_bandwidth_dict_gbs = {1: 1e10, 2: 105.24, 4: 94.05, 8: 86.34, 16: 19.12, 32:11.01, 64:75} # 2k seq
    # alltoall_bandwidth_dict_gbs = {1: 1e10, 2: 119.54, 4: 104.07, 8: 96.5, 16: 20.01, 32:10.53, 64:7.77} # 4k seq
    alltoall_bandwidth_dict_gbs = {1: 1e10, 2: 119.54, 4: 104.07, 8: 96.5, 16: 10.33, 32:5.94, 64:4.87} # 4k seq, 2 IB

    param_sizes = model_config['param_size_B']
    seq_limit_k = args.seq_limit_k
    if seq_limit_k in param_sizes.keys():
        param_size_B = param_sizes[seq_limit_k]
    else:
        param_size_B = (param_sizes[384]-param_sizes[192])/192*(seq_limit_k-192)+param_sizes[192]
    print('Model Size: ', param_size_B)
    
    act_per_token = model_config['act_per_token']
    if args.model_size == 'gpt-7b' and cluster_size in [16, 32]:
        act_per_token = {
            16: 2.668764648,
            # 16: 4.480441895,
            32: 4.480441895,
        }[cluster_size]

    costmodel = flexSPCostModel(
                 cluster_size = cluster_size,
                 hidden_size = model_config['hidden_size'],
                 layer_num = model_config['layer_num'],
                 param_size_B = param_size_B, 
                 zero_stage = 3,
                 mixed_precision = True,
                 act_per_token = act_per_token, 
                 cpt_alpha1 = model_config['cpt_alpha1'], 
                 cpt_alpha2  = model_config['cpt_alpha2'], 
                 cpt_beta1 = model_config['cpt_beta1'],
                 alltoall_bandwidth_dict_gbs = alltoall_bandwidth_dict_gbs,
                 p2p_bandwidth_gbs = args.p2p_bandwidth_gbs,  # Ring Attention P2P 带宽
                 ring_overlap_efficiency = args.ring_overlap_efficiency,  # Ring Attention 重叠效率
                )

    param_dict = {
                "limits/time": args.time_limit,
                }
    flexSP_optimizer = flexSPOptimizer(
                 cluster_size = cluster_size,
                 memory_limit_gb = memory_limit_gb, 
                 costmodel = costmodel,
                 hide_scipoutput = True,
                 hide_alloutput = True,
                 concurrent = False,
                 scip_param_dict = param_dict,
                 strategy = 'adaptive_bfd',
                 redist_without_empty_group = args.redist_without_empty_group,
                 enable_ring_attn = args.enable_ring_attn,  # 启用 Ring Attention 策略
    )

    print(f'Cluster Size: {cluster_size}, Model Size: {args.model_size}, Memory: {memory_limit_gb}')
    flexSP_optimizer.token_info([])
    print('='*20)
    print()
    # exit(0)

    dataset_name = args.dataset
    global_batch_size = args.global_batch_size
    seq_limit = args.seq_limit_k * 1000
    data = read_dataset(dataset_name, seq_limit = seq_limit)
    
    ignore_strategies = [32] if 'wikipedia' in args.dataset and args.seq_limit_k <= 192 else []
    
    # dataset analysis
    total_token = 0
    max_seq, max_seq_iter = -1, -1
    for iter in range(args.start_iter, args.start_iter+args.iter_num):
        sequences = get_global_batch(data, iter, global_batch_size)
        batch_token = sum(sequences)
        total_token += batch_token
        if max(sequences) > max_seq:
            max_seq = max(sequences)
            max_seq_iter = iter
    sequences = [Sequence(seq=seq, id=id) for id, seq in enumerate([max_seq] * (flexSP_optimizer.cluster_token_capacity//max_seq))]
    globalbatch_groups, globalbatch_results = flexSP_optimizer.homo_sp_baseline_ffd_bfd_globalbatch(sequences, 'bfd', sp_select_rule='min_sp', fill_empty=False, ignore_strategies=ignore_strategies)
    min_sp_size, _ = globalbatch_groups[0][0] if len(globalbatch_groups) and len(globalbatch_groups[0]) else (-1,-1)
    print('Dataset %s: Avg Tokens per Batch = %.3f Million, Max Seq = %d (%d), Min-SP = %d'%(args.dataset, total_token/args.iter_num/1024/1024, max_seq, max_seq_iter, min_sp_size))
    print('Maxseq: %d'%args.seq_limit_k, 'Total tokens: %d'%total_token)
    print()
    # exit(0)
    
    if args.method_type == 'static':
        print(f"--------------- [Baseline BFD Homo-SP] ---------------")
        total_time = 0.
        fix_sp_size = 64 # min_sp_size
        for iter in range(args.start_iter, args.start_iter+args.iter_num):
            sequences = get_global_batch(data, iter, global_batch_size)
            sequences = [Sequence(seq=seq, id=id) for id, seq in enumerate(sequences)]
            globalbatch_groups, globalbatch_results = flexSP_optimizer.homo_sp_baseline_ffd_bfd_globalbatch(sequences, 'bfd', sp_select_rule='fix_sp', sp_size=fix_sp_size, ignore_strategies=ignore_strategies)
            globalbatch_time = sum([results['M'] for results in globalbatch_results])
            total_time += globalbatch_time
            print(f'Iteration[{iter}] Final Results: sp size = {fix_sp_size}, # microbatch = {len(globalbatch_groups)}, Time = {globalbatch_time:.2f}', flush=True)
            if args.show_strategy:
                for idx, (groups, results) in enumerate(zip(globalbatch_groups, globalbatch_results)):
                    microbatch_token = flexSP_optimizer.get_groups_total_token(groups)
                    total_token = flexSP_optimizer.cluster_token_capacity
                    print(f"============= Microbatch {idx}, Time: {results['M']:.2f}, Total Token: {microbatch_token} / {total_token} =============")
                    for group in groups:
                        # 兼容新格式 (sp_size, attn_type, seqs) 和旧格式 (sp_size, seqs)
                        if len(group) == 3:
                            sp_size, attn_type, seqs = group
                        else:
                            sp_size, seqs = group
                            attn_type = 'ulysses'
                        flexSP_optimizer.print_group_seqs_info(seqs, sp_size, attn_type=attn_type)
                print()
        print(f"--------------- [Baseline BFD Homo-SP] ---------------")
        print(f"Total Time for {args.iter_num} iterations: {total_time / 1000.:.2f} s\n")
    
    if args.method_type == 'adaptive':
        print(f"--------------- [Baseline BFD Homo-SP (adaptive across global batches)] ---------------")
        total_time = 0.
        for iter in range(args.start_iter, args.start_iter+args.iter_num):
            sequences = get_global_batch(data, iter, global_batch_size)
            sequences = [Sequence(seq=seq, id=id) for id, seq in enumerate(sequences)]
            globalbatch_groups, globalbatch_results = flexSP_optimizer.homo_sp_baseline_ffd_bfd_globalbatch(sequences, 'bfd', sp_select_rule='adaptive', ignore_strategies=ignore_strategies)
            sp_size, _ = globalbatch_groups[0][0]
            globalbatch_time = sum([results['M'] for results in globalbatch_results])
            total_time += globalbatch_time
            print(f'Iteration[{iter}] Final Results: sp size = {sp_size}, # microbatch = {len(globalbatch_groups)}, Time = {globalbatch_time:.2f}', flush=True)
            if args.show_strategy:
                for idx, (groups, results) in enumerate(zip(globalbatch_groups, globalbatch_results)):
                    microbatch_token = flexSP_optimizer.get_groups_total_token(groups)
                    total_token = flexSP_optimizer.cluster_token_capacity
                    print(f"============= Microbatch {idx}, Time: {results['M']:.2f}, Total Token: {microbatch_token} / {total_token} =============")
                    for group in groups:
                        # 兼容新格式 (sp_size, attn_type, seqs) 和旧格式 (sp_size, seqs)
                        if len(group) == 3:
                            sp_size, attn_type, seqs = group
                        else:
                            sp_size, seqs = group
                            attn_type = 'ulysses'
                        flexSP_optimizer.print_group_seqs_info(seqs, sp_size, attn_type=attn_type)
                print()
        print(f"--------------- [Baseline BFD Homo-SP (adaptive across global batches)] ---------------")
        print(f"Total Time for {args.iter_num} iterations: {total_time / 1000.:.2f} s\n")

    if args.method_type == 'flexSP':
        print(f"--------------- [FlexSP] ---------------")
        total_time = 0.
        for iter in range(args.start_iter, args.start_iter+args.iter_num):
            sequences = get_global_batch(data, iter, global_batch_size)
            sequences = [Sequence(seq=seq, id=id) for id, seq in enumerate(sequences)]
            chunk_alg = args.chunk_alg
            bucket_alg = args.bucket_alg
            bucket_num = 16
            if args.no_seq_bucket:
                bucket_num = 1e10 # bucket num set as inf, fallen back into solver without sequence bucketing
            start = time.time()
            globalbatch_groups, globalbatch_results = flexSP_optimizer.solve_flexSP_globalbatch_mp_gbmb(sequences, bucket_alg=bucket_alg, chunk_alg = chunk_alg, mb_option_num = 5, bucket_num = bucket_num)
            end = time.time()

            globalbatch_time = sum([results['M'] for results in globalbatch_results])
            total_time += globalbatch_time
            print(f'Iteration[{iter}] Final Results: Search Time = {end-start:.4f}, # microbatch = {len(globalbatch_groups)}, Time = {globalbatch_time:.2f}', flush=True)
            if args.show_strategy:
                for idx, (groups, results) in enumerate(zip(globalbatch_groups, globalbatch_results)):
                    microbatch_token = flexSP_optimizer.get_groups_total_token(groups)
                    total_token = flexSP_optimizer.cluster_token_capacity
                    print(f"============= Microbatch {idx}, Time: {results['M']:.2f}, Total Token: {microbatch_token} / {total_token} =============")
                    for group in groups:
                        # 兼容新格式 (sp_size, attn_type, seqs) 和旧格式 (sp_size, seqs)
                        if len(group) == 3:
                            sp_size, attn_type, seqs = group
                        else:
                            sp_size, seqs = group
                            attn_type = 'ulysses'
                        flexSP_optimizer.print_group_seqs_info(seqs, sp_size, attn_type=attn_type)
                print()
        print(f"--------------- [FlexSP] ---------------")
        print(f"Total Time for {args.iter_num} iterations: {total_time / 1000.:.2f} s\n")
        
    exit(0)

    # test
    sequences = get_global_batch(data, 4, global_batch_size)
    sequences = [Sequence(seq=seq, id=id) for id, seq in enumerate(sequences)]   
    start = time.time()
    chunk_alg = args.chunk_alg
    bucket_num = 16
    if args.no_seq_bucket:
        bucket_num = 1e10 # bucket num set as inf, fallen back into solver without sequence bucketing
    # globalbatch_groups, globalbatch_results = flexSP_optimizer.solve_flexSP_globalbatch(sequences, chunk_alg = chunk_alg, bucket_num = bucket_num)
    # globalbatch_groups, globalbatch_results = flexSP_optimizer.solve_flexSP_globalbatch_mp(sequences, chunk_alg = chunk_alg, bucket_num = bucket_num)
    globalbatch_groups, globalbatch_results = flexSP_optimizer.solve_flexSP_globalbatch_mp_gbmb(sequences, chunk_alg = chunk_alg, mb_option_num = 5, bucket_num = bucket_num)
    end = time.time()
    print('Time cost: %.4f'%(end-start))

    globalbatch_time = sum([results['M'] for results in globalbatch_results])
    print(f'\n\nGlobalbatch Final Results: # microbatch = {len(globalbatch_groups)}, Time = {globalbatch_time:.2f}')
    for idx, (groups, results) in enumerate(zip(globalbatch_groups, globalbatch_results)):
        microbatch_token = flexSP_optimizer.get_groups_total_token(groups)
        total_token = flexSP_optimizer.cluster_token_capacity
        print(f"============= Microbatch {idx}, Time: {results['M']:.2f}, Total Token: {microbatch_token} / {total_token} =============")
        for group in groups:
            # 兼容新格式 (sp_size, attn_type, seqs) 和旧格式 (sp_size, seqs)
            if len(group) == 3:
                sp_size, attn_type, seqs = group
            else:
                sp_size, seqs = group
                attn_type = 'ulysses'
            flexSP_optimizer.print_group_seqs_info(seqs, sp_size, attn_type=attn_type)

    exit(0)

    # sequences = sorted(sequences, reverse=True)
    # sequences = sequences[-68:]


    # baseline_results_random = flexSP_optimizer.homo_sp_baseline_random(sequences)
    # baseline_results_lp = flexSP_optimizer.homo_sp_baseline_lp(sequences)
    # baseline_results_bfd = flexSP_optimizer.homo_sp_baseline_ffd_bfd(sequences, type='bfd')
    # flexSP_results = flexSP_optimizer.solve_flexSP(sequences)
    flexSP_results = flexSP_optimizer.solve_flexSP_bucket_seqs(sequences, bucket_num=10)
    # flexSP_results = flexSP_optimizer.solve_flexSP_bucket_seqs_groups(sequences, bucket_num=12)
    
    # for sp, results in baseline_results_random.items():
    #     print('\n--------------- [Baseline Random Homo-SP = %d] ---------------'%sp)
    #     flexSP_optimizer.show_results(results)
    # for sp, results in baseline_results_lp.items():
    #     print('\n--------------- [Baseline LP Homo-SP = %d] ---------------'%sp)
    #     flexSP_optimizer.show_results(results)
    # for sp, results in baseline_results_bfd.items():
    #     print('\n--------------- [Baseline BFD Homo-SP = %d] ---------------'%sp)
    #     flexSP_optimizer.show_results(results)
    print('\n--------------- [flexSP Results] ---------------')
    groups = flexSP_optimizer.show_results(flexSP_results)
    
    print('\n============= Final Results =============')
    # for sp, results in baseline_results_random.items():
    #     if results is not None:
    #         print('Baseline Random Homo-SP = %d: Minimized Time = %.2f'%(sp, results['M']))
    # for sp, results in baseline_results_lp.items():
    #     if results is not None:
    #         print('Baseline LP Homo-SP = %d: Minimized Time = %.2f'%(sp, results['M']))
    # for sp, results in baseline_results_bfd.items():
    #     if results is not None:
    #         print('Baseline BFD Homo-SP = %d: Minimized Time = %.2f'%(sp, results['M']))
    if flexSP_results is not None:
        print('flexSP: Minimized Time = %.2f'%(flexSP_results['M']))
        
    flexSP_optimizer.token_info(sequences)
    
    # for sp_size, seqs in groups:
    #     print(sp_size)
    #     print_seqs(seqs)
