"""
AdaCPSP Solver: Adaptive Context Parallel and Sequence Parallel Solver

This solver extends FlexSP to support both Ulysses SP and Context Parallel (CP) 
with dynamic selection based on sequence characteristics and cost modeling.

Key features:
- Supports Ulysses SP (AlltoAll communication)
- Supports Zigzag CP (Ring P2P communication)
- Supports combined Ulysses SP + CP (e.g., sp_size=4, cp_size=2 for 8 GPUs)
- Dynamic strategy selection per bucket based on cost model
"""

from typing import List, Dict, Union, Literal, Tuple, Optional
import numpy as np
from collections import Counter
import torch
import torch.distributed as dist


class Sequence:
    """Simple sequence class for AdaCPSP solver."""
    def __init__(self, seq_len: int, seq_id: int = -1):
        self.seq = seq_len
        self.id = seq_id
    
    def __repr__(self):
        return f"Seq(len={self.seq}, id={self.id})"


class AdaCPSPCostModel:
    """Cost model for AdaCPSP that considers both Ulysses SP and CP.
    
    Key differences from FlexSP:
    - Ulysses SP: AlltoAll communication (bandwidth-bound for large sequences)
    - CP (Zigzag Ring): P2P ring communication (latency-bound, better for very long sequences)
    """
    
    def __init__(self, 
                 cluster_size: int = 8,
                 hidden_size: int = 4096,
                 num_attention_heads: int = 32,
                 num_kv_heads: int = 8,
                 layer_num: int = 32,
                 param_size_B: float = 7,
                 zero_stage: int = 3,
                 mixed_precision: bool = True,
                 act_per_token: float = 4.71,
                 # Computation cost parameters
                 cpt_alpha1: float = 5.128e-6,  # O(n^2) coefficient for attention
                 cpt_alpha2: float = 183.9576e-3,  # O(n) coefficient
                 cpt_beta1: float = 629.3563,  # Constant overhead
                 # Communication parameters
                 alltoall_bandwidth_dict_gbs: Dict = None,  # Ulysses SP bandwidth
                 p2p_bandwidth_gbs: float = 200,  # CP P2P bandwidth
                 p2p_latency_ms: float = 0.01,  # CP P2P latency per message
                 ):
        self.N = cluster_size
        self.h = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.l = layer_num
        self.p = param_size_B
        self.zero_stage = zero_stage
        self.act_per_token = act_per_token
        
        # Computation parameters
        self.cpt_alpha1 = cpt_alpha1
        self.cpt_alpha2 = cpt_alpha2
        self.cpt_beta1 = cpt_beta1
        
        # Communication parameters
        if alltoall_bandwidth_dict_gbs is None:
            # Default bandwidth for different SP sizes (GB/s)
            alltoall_bandwidth_dict_gbs = {1: 1e10, 2: 154, 4: 137, 8: 121, 16: 8.7}
        self.alltoall_bandwidth_dict_gbs = alltoall_bandwidth_dict_gbs
        self.p2p_bandwidth_gbs = p2p_bandwidth_gbs
        self.p2p_latency_ms = p2p_latency_ms
        
        # Memory calculation
        self.zero_ratio = {
            0: 1,
            1: (6/8 * (1/self.N) + 2/8) if mixed_precision else (2/4 * (1/self.N) + 2/4),
            2: (7/8 * (1/self.N) + 1/8) if mixed_precision else (3/4 * (1/self.N) + 1/4),
            3: 1/self.N
        }[self.zero_stage]
        self.model_states_mb = param_size_B * 16 * self.zero_ratio * 1024
        
    def activation_size(self, seqlen: Union[int, List[int]], parallel_size: int = 1) -> float:
        """Calculate activation memory size in MB."""
        if isinstance(seqlen, list):
            seqlen = sum(seqlen)
        return self.act_per_token * seqlen / parallel_size
    
    def total_memory(self, seqlen: Union[int, List[int]] = 0, parallel_size: int = 1) -> float:
        """Calculate total memory usage in MB."""
        return self.model_states_mb + self.activation_size(seqlen, parallel_size)
    
    def token_capacity(self, memory_limit_gb: int, parallel_size: int = 1) -> int:
        """Calculate maximum tokens that can fit in memory."""
        available_mb = memory_limit_gb * 1024 - self.model_states_mb
        return int(available_mb / self.act_per_token * parallel_size)
    
    def compute_time_single(self, seqlen: int, parallel_size: int = 1) -> float:
        """Compute time for a single sequence in ms."""
        # Attention is O(n^2), other operations are O(n)
        return (self.cpt_alpha1 * (seqlen ** 2) + self.cpt_alpha2 * seqlen) / parallel_size
    
    def compute_time(self, seqlen: Union[int, List[int]], parallel_size: int = 1) -> float:
        """Total compute time for sequences in ms."""
        if not isinstance(seqlen, list):
            seqlen = [seqlen]
        cpt_times = [self.compute_time_single(seq, parallel_size) for seq in seqlen]
        return sum(cpt_times) + self.cpt_beta1
    
    def ulysses_alltoall_time(self, seqlen: Union[int, List[int]], sp_size: int = 1) -> float:
        """AlltoAll communication time for Ulysses SP in ms.
        
        Communication volume: 4 * 2 * layer_num * hidden_size * seqlen * 2 bytes
        (Q,K,V before attention + output after attention, forward + backward)
        """
        if sp_size == 1:
            return 0
        if isinstance(seqlen, list):
            seqlen = sum(seqlen)
        
        bandwidth = self.alltoall_bandwidth_dict_gbs.get(sp_size, 8.0)
        # AlltoAll tensor size in GB
        tensor_size_gb = 4 * 2 * self.l * self.h * seqlen * 2 / (1024 ** 3) / sp_size
        return tensor_size_gb / bandwidth * 1000  # Convert to ms
    
    def cp_p2p_time(self, seqlen: Union[int, List[int]], cp_size: int = 1) -> float:
        """P2P ring communication time for Context Parallel in ms.
        
        For zigzag ring attention:
        - Each step sends K,V chunks to next rank
        - Total (cp_size - 1) communication steps
        - Communication volume per step: 2 * hidden_size * (seqlen / cp_size) * 2 bytes
        """
        if cp_size == 1:
            return 0
        if isinstance(seqlen, list):
            seqlen = sum(seqlen)
        
        # Per-step communication size (K and V)
        chunk_size = seqlen // cp_size
        # KV size per layer: 2 * num_kv_heads * head_dim * chunk_size
        head_dim = self.h // self.num_heads
        kv_size_per_layer = 2 * self.num_kv_heads * head_dim * chunk_size * 2  # bytes (fp16)
        
        # Total communication steps: (cp_size - 1) * 2 (forward + backward)
        num_steps = (cp_size - 1) * 2
        
        # Total communication time
        total_size_gb = kv_size_per_layer * self.l * num_steps / (1024 ** 3)
        bandwidth_time = total_size_gb / self.p2p_bandwidth_gbs * 1000
        latency_time = num_steps * self.l * self.p2p_latency_ms
        
        return bandwidth_time + latency_time
    
    def total_time_ulysses(self, seqlen: Union[int, List[int]], sp_size: int = 1) -> float:
        """Total time for Ulysses SP strategy."""
        return self.compute_time(seqlen, sp_size) + self.ulysses_alltoall_time(seqlen, sp_size)
    
    def total_time_cp(self, seqlen: Union[int, List[int]], cp_size: int = 1) -> float:
        """Total time for Context Parallel strategy."""
        return self.compute_time(seqlen, cp_size) + self.cp_p2p_time(seqlen, cp_size)
    
    def total_time_combined(self, seqlen: Union[int, List[int]], 
                           sp_size: int = 1, cp_size: int = 1) -> float:
        """Total time for combined Ulysses SP + CP strategy.
        
        With combined strategy:
        - First apply Ulysses SP (AlltoAll within SP groups)
        - Then apply CP (Ring within CP groups)
        - Total parallel size = sp_size * cp_size
        """
        if sp_size == 1 and cp_size == 1:
            return self.compute_time(seqlen, 1)
        
        total_parallel = sp_size * cp_size
        compute = self.compute_time(seqlen, total_parallel)
        
        # Ulysses AlltoAll operates on (seqlen / cp_size) per SP group
        ulysses_comm = self.ulysses_alltoall_time(seqlen, sp_size) if sp_size > 1 else 0
        
        # CP P2P operates on full sequence but split across cp_size
        cp_comm = self.cp_p2p_time(seqlen, cp_size) if cp_size > 1 else 0
        
        return compute + ulysses_comm + cp_comm
    
    def get_best_strategy(self, seqlen: Union[int, List[int]], 
                         available_gpus: int,
                         memory_limit_gb: int = 80) -> Tuple[int, int, str, float]:
        """Find the best (sp_size, cp_size, strategy_type, time) for given sequence.
        
        Returns:
            Tuple of (sp_size, cp_size, strategy_type, estimated_time)
            strategy_type: 'ulysses', 'cp', or 'combined'
        """
        if isinstance(seqlen, list):
            total_seqlen = sum(seqlen)
        else:
            total_seqlen = seqlen
        
        best_time = float('inf')
        best_config = (1, 1, 'none', best_time)
        
        # Try different combinations
        sp = 1
        while sp <= available_gpus:
            cp = 1
            while sp * cp <= available_gpus:
                total_parallel = sp * cp
                
                # Check memory constraint
                if self.activation_size(total_seqlen, total_parallel) + self.model_states_mb > memory_limit_gb * 1024:
                    cp *= 2
                    continue
                
                # Calculate time for different strategies
                if sp > 1 and cp == 1:
                    # Pure Ulysses SP
                    time = self.total_time_ulysses(total_seqlen, sp)
                    strategy = 'ulysses'
                elif sp == 1 and cp > 1:
                    # Pure CP
                    time = self.total_time_cp(total_seqlen, cp)
                    strategy = 'cp'
                elif sp > 1 and cp > 1:
                    # Combined
                    time = self.total_time_combined(total_seqlen, sp, cp)
                    strategy = 'combined'
                else:
                    # No parallelism
                    time = self.compute_time(total_seqlen, 1)
                    strategy = 'none'
                
                if time < best_time:
                    best_time = time
                    best_config = (sp, cp, strategy, time)
                
                cp *= 2
            sp *= 2
        
        return best_config


class AdaCPSPConfig:
    """Configuration for a single bucket's parallel strategy.
    
    Triplet representation: (cp_size, sp_size, [seq_ids...])
    - cp_size: Context Parallel size (Zigzag Ring Attention)
    - sp_size: Ulysses SP size (AlltoAll)
    - seq_ids: List of sequence indices assigned to this group
    - Total parallel size = cp_size * sp_size
    
    Note: use_ulysses = (sp_size > 1), use_cp = (cp_size > 1)
    """
    
    def __init__(self, 
                 cp_size: int = 1,
                 sp_size: int = 1,
                 seq_ids: List[int] = None,
                 sp_group=None,
                 cp_group=None):
        self.cp_size = cp_size      # Context Parallel (Zigzag Ring)
        self.sp_size = sp_size      # Ulysses SP (AlltoAll)
        self.seq_ids = seq_ids or []
        self.sp_group = sp_group
        self.cp_group = cp_group
    
    @property
    def use_ulysses(self) -> bool:
        """Whether to use Ulysses SP (AlltoAll)."""
        return self.sp_size > 1
    
    @property
    def use_cp(self) -> bool:
        """Whether to use Context Parallel (Zigzag Ring)."""
        return self.cp_size > 1
        
    @property
    def total_parallel_size(self) -> int:
        """Total number of GPUs used by this group."""
        return self.cp_size * self.sp_size
    
    def to_triplet(self) -> Tuple[int, int, List[int]]:
        """Convert to triplet representation: (cp_size, sp_size, seq_ids)."""
        return (self.cp_size, self.sp_size, self.seq_ids)
    
    @classmethod
    def from_triplet(cls, triplet: Tuple[int, int, List[int]]) -> 'AdaCPSPConfig':
        """Create from triplet representation."""
        cp_size, sp_size, seq_ids = triplet
        return cls(cp_size=cp_size, sp_size=sp_size, seq_ids=seq_ids)
    
    def __repr__(self):
        return (f"AdaCPSPConfig(cp_size={self.cp_size}, sp_size={self.sp_size}, "
                f"seqs={len(self.seq_ids)})")


class AdaCPSPOptimizer:
    """Optimizer for AdaCPSP that assigns sequences to buckets with optimal strategies.
    
    This optimizer:
    1. Groups sequences into buckets based on length similarity
    2. For each bucket, selects the optimal (sp_size, cp_size) combination
    3. Ensures total GPU usage matches cluster size
    """
    
    def __init__(self,
                 cluster_size: int,
                 memory_limit_gb: int,
                 costmodel: AdaCPSPCostModel,
                 max_sp_size: int = None,
                 max_cp_size: int = None,
                 prefer_ulysses: bool = True,
                 hide_output: bool = False):
        self.N = cluster_size
        self.mem_limit_gb = memory_limit_gb
        self.costmodel = costmodel
        self.max_sp_size = max_sp_size or cluster_size
        self.max_cp_size = max_cp_size or cluster_size
        self.prefer_ulysses = prefer_ulysses
        self.hide_output = hide_output
        
        self.device_token_capacity = costmodel.token_capacity(memory_limit_gb)
    
    def token_info(self, seqs: List[Sequence]):
        """Print token information (following FlexSP pattern)."""
        if self.hide_output:
            return
        print('\n============= Token Info =============')
        print('Device Token Capacity: %d' % self.device_token_capacity)
        print('Cluster Token Capacity: %d' % (self.device_token_capacity * self.N))
        total_tokens = sum(s.seq for s in seqs)
        print('Total Token Number: %d' % total_tokens)
        print('Total Sequence Number: %d' % len(seqs))
        
    def get_min_parallel_size(self, seqlen: int) -> int:
        """Get minimum parallel size needed for a sequence to fit in memory."""
        min_parallel = int(np.ceil(seqlen / self.device_token_capacity))
        # Round up to power of 2
        log_2 = np.log(max(min_parallel, 1)) / np.log(2)
        return 2 ** int(np.ceil(log_2))
    
    def solve_simple(self, seqs: List[Sequence], 
                     strategy: Literal['ulysses', 'cp', 'adaptive'] = 'adaptive'
                     ) -> List[Tuple[AdaCPSPConfig, List[Sequence]]]:
        """Simple solver that assigns all sequences to groups.
        
        Args:
            seqs: List of sequences to process
            strategy: 'ulysses' (pure Ulysses SP), 'cp' (pure CP), or 'adaptive'
            
        Returns:
            List of (config, sequences) tuples for each group
        """
        if not seqs:
            return []
        
        # Sort sequences by length (descending)
        sorted_seqs = sorted(seqs, key=lambda s: s.seq, reverse=True)
        
        # Calculate minimum parallel size needed
        max_seqlen = max(s.seq for s in sorted_seqs)
        min_parallel = self.get_min_parallel_size(max_seqlen)
        
        # Determine number of groups
        num_groups = self.N // min_parallel
        if num_groups == 0:
            num_groups = 1
            min_parallel = self.N
        
        # Determine strategy for each group
        if strategy == 'ulysses':
            sp_size, cp_size = min_parallel, 1
            use_ulysses, use_cp = True, False
        elif strategy == 'cp':
            sp_size, cp_size = 1, min_parallel
            use_ulysses, use_cp = False, True
        else:  # adaptive
            # Use cost model to decide
            sp_size, cp_size, strat_type, _ = self.costmodel.get_best_strategy(
                max_seqlen, min_parallel, self.mem_limit_gb
            )
            use_ulysses = sp_size > 1
            use_cp = cp_size > 1
        
        # Distribute sequences across groups using best-fit decreasing
        groups = [[] for _ in range(num_groups)]
        group_tokens = [0] * num_groups
        capacity = self.device_token_capacity * min_parallel
        
        for seq in sorted_seqs:
            # Find best-fit group
            best_group = -1
            best_remaining = float('inf')
            
            for i in range(num_groups):
                remaining = capacity - group_tokens[i]
                if remaining >= seq.seq and remaining - seq.seq < best_remaining:
                    best_group = i
                    best_remaining = remaining - seq.seq
            
            if best_group == -1:
                # No group has space, add to least loaded
                best_group = min(range(num_groups), key=lambda i: group_tokens[i])
            
            groups[best_group].append(seq)
            group_tokens[best_group] += seq.seq
        
        # Create configs using triplet format: (cp_size, sp_size, seq_ids)
        results = []
        for group_seqs in groups:
            if group_seqs:
                config = AdaCPSPConfig(
                    cp_size=cp_size,
                    sp_size=sp_size,
                    seq_ids=[s.id for s in group_seqs]
                )
                results.append((config, group_seqs))
        
        return results
    
    def solve_heterogeneous(self, seqs: List[Sequence]
                           ) -> List[Tuple[AdaCPSPConfig, List[Sequence]]]:
        """Heterogeneous solver that allows different strategies for different groups.
        
        This solver:
        1. Sorts sequences by length
        2. Groups similar-length sequences together
        3. Assigns optimal strategy to each group
        """
        if not seqs:
            return []
        
        # Sort sequences by length
        sorted_seqs = sorted(seqs, key=lambda s: s.seq, reverse=True)
        
        # Group sequences by required parallel size
        parallel_groups = {}  # parallel_size -> list of sequences
        
        for seq in sorted_seqs:
            min_parallel = self.get_min_parallel_size(seq.seq)
            min_parallel = min(min_parallel, self.N)
            
            if min_parallel not in parallel_groups:
                parallel_groups[min_parallel] = []
            parallel_groups[min_parallel].append(seq)
        
        # Assign GPUs to each parallel group
        results = []
        remaining_gpus = self.N
        
        for parallel_size in sorted(parallel_groups.keys(), reverse=True):
            group_seqs = parallel_groups[parallel_size]
            
            if remaining_gpus < parallel_size:
                # Not enough GPUs, merge with smaller parallel group
                continue
            
            # Calculate number of groups for this parallel size
            num_groups = remaining_gpus // parallel_size
            
            # Distribute sequences
            subgroups = [[] for _ in range(num_groups)]
            capacity = self.device_token_capacity * parallel_size
            
            for i, seq in enumerate(group_seqs):
                subgroups[i % num_groups].append(seq)
            
            # Determine strategy for this parallel size
            if parallel_size > 1:
                avg_seqlen = sum(s.seq for s in group_seqs) / len(group_seqs)
                sp_size, cp_size, strat_type, _ = self.costmodel.get_best_strategy(
                    int(avg_seqlen), parallel_size, self.mem_limit_gb
                )
            else:
                sp_size, cp_size = 1, 1
                strat_type = 'none'
            
            for subgroup in subgroups:
                if subgroup:
                    # Triplet format: (cp_size, sp_size, seq_ids)
                    config = AdaCPSPConfig(
                        cp_size=cp_size,
                        sp_size=sp_size,
                        seq_ids=[s.id for s in subgroup]
                    )
                    results.append((config, subgroup))
            
            remaining_gpus -= num_groups * parallel_size
        
        return results
    
    def print_solution(self, results: List[Tuple[AdaCPSPConfig, List[Sequence]]]):
        """Print solution details.
        
        Each group is represented as triplet: (cp_size, sp_size, [seq_ids...])
        """
        if self.hide_output:
            return
        
        print("\n============= AdaCPSP Solution =============")
        print("Format: (cp_size, sp_size, [seq_ids...])")
        total_time = 0
        
        for i, (config, seqs) in enumerate(results):
            seqlens = [s.seq for s in seqs]
            total_seqlen = sum(seqlens)
            
            # Calculate time based on strategy
            if config.use_ulysses and config.use_cp:
                time = self.costmodel.total_time_combined(total_seqlen, config.sp_size, config.cp_size)
                strategy = f"Combined(CP={config.cp_size}, SP={config.sp_size})"
            elif config.use_ulysses:
                time = self.costmodel.total_time_ulysses(total_seqlen, config.sp_size)
                strategy = f"Ulysses(SP={config.sp_size})"
            elif config.use_cp:
                time = self.costmodel.total_time_cp(total_seqlen, config.cp_size)
                strategy = f"CP(CP={config.cp_size})"
            else:
                time = self.costmodel.compute_time(total_seqlen, 1)
                strategy = "None"
            
            total_time = max(total_time, time)
            
            # Print triplet representation
            triplet = config.to_triplet()
            print(f"Group {i}: {strategy}")
            print(f"  Triplet: ({triplet[0]}, {triplet[1]}, [{len(triplet[2])} seqs])")
            print(f"  Total tokens: {total_seqlen}, GPUs: {config.total_parallel_size}")
            print(f"  Seq lengths: {seqlens[:5]}{'...' if len(seqlens) > 5 else ''}")
            print(f"  Estimated time: {time:.2f} ms")
        
        print(f"\nTotal estimated time: {total_time:.2f} ms")
        print("=" * 45)


class CommunicationGroupManager:
    """Manages pre-created communication groups for AdaCPSP.
    
    Pre-creates all possible SP and CP groups at initialization to avoid
    runtime group creation overhead.
    """
    
    def __init__(self, world_size: int, max_sp_size: int = None, max_cp_size: int = None):
        self.world_size = world_size
        self.max_sp_size = max_sp_size or world_size
        self.max_cp_size = max_cp_size or world_size
        
        self.sp_groups = {}  # (start_rank, sp_size) -> group
        self.cp_groups = {}  # (start_rank, cp_size) -> group
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        self._initialized = False
    
    def initialize_groups(self):
        """Pre-create all communication groups."""
        if self._initialized:
            return
        
        if not dist.is_initialized():
            print("Warning: Distributed not initialized, skipping group creation")
            return
        
        # Create SP groups (Ulysses AlltoAll groups)
        sp_size = 1
        while sp_size <= min(self.max_sp_size, self.world_size):
            for start in range(0, self.world_size, sp_size):
                ranks = list(range(start, start + sp_size))
                group = dist.new_group(ranks)
                self.sp_groups[(start, sp_size)] = group
            sp_size *= 2
        
        # Create CP groups (Ring P2P groups)
        cp_size = 1
        while cp_size <= min(self.max_cp_size, self.world_size):
            for start in range(0, self.world_size, cp_size):
                ranks = list(range(start, start + cp_size))
                group = dist.new_group(ranks)
                self.cp_groups[(start, cp_size)] = group
            cp_size *= 2
        
        self._initialized = True
        
        if self.rank == 0:
            print(f"AdaCPSP: Created {len(self.sp_groups)} SP groups and {len(self.cp_groups)} CP groups")
    
    def get_sp_group(self, start_rank: int, sp_size: int):
        """Get SP group for given start rank and size."""
        key = (start_rank, sp_size)
        if key not in self.sp_groups:
            raise ValueError(f"SP group not found: start={start_rank}, size={sp_size}")
        return self.sp_groups[key]
    
    def get_cp_group(self, start_rank: int, cp_size: int):
        """Get CP group for given start rank and size."""
        key = (start_rank, cp_size)
        if key not in self.cp_groups:
            raise ValueError(f"CP group not found: start={start_rank}, size={cp_size}")
        return self.cp_groups[key]
    
    def get_my_groups(self, config: AdaCPSPConfig, group_start_rank: int) -> Tuple[Optional[object], Optional[object]]:
        """Get the SP and CP groups for current rank based on config."""
        sp_group = None
        cp_group = None
        
        total_size = config.sp_size * config.cp_size
        
        if self.rank >= group_start_rank and self.rank < group_start_rank + total_size:
            local_rank = self.rank - group_start_rank
            
            if config.use_ulysses and config.sp_size > 1:
                # SP group: ranks within same CP position
                cp_position = local_rank // config.sp_size
                sp_start = group_start_rank + cp_position * config.sp_size
                sp_group = self.get_sp_group(sp_start, config.sp_size)
            
            if config.use_cp and config.cp_size > 1:
                # CP group: ranks with same SP position across CP
                sp_position = local_rank % config.sp_size
                cp_ranks = [group_start_rank + sp_position + i * config.sp_size 
                           for i in range(config.cp_size)]
                cp_start = min(cp_ranks)
                cp_group = self.get_cp_group(cp_start, config.cp_size)
        
        return sp_group, cp_group

