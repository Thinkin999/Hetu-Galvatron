"""
AdaCPSP Communication Group Manager

Pre-creates all valid (sp_size, cp_size) communication group combinations
and provides a mechanism to dynamically switch model strategy per microbatch.

Design:
  - For N GPUs, valid strategies have sp * cp * dp = N
  - Phase 3 constrains dp=1 (sp * cp = N) for simplicity
  - Groups follow the convention: SP is consecutive, CP spans across SP groups
  
  Example for N=8, sp=2, cp=4:
    SP groups: [0,1], [2,3], [4,5], [6,7]
    CP groups: [0,2,4,6], [1,3,5,7]
"""

import math
import torch
import torch.distributed as dist
from typing import Dict, Tuple, Optional, List

from galvatron.core.runtime.tensor_parallel.attention import SelfAttention
from galvatron.core.runtime.tensor_parallel.attention_impl import (
    DistributedAttention,
    ZigzagRingFlashAttentionVarlen,
    FlashSelfAttentionVarlen,
)


class CommunicationGroupManager:
    """Manages pre-created communication groups for all valid (sp_size, cp_size) strategies."""
    
    def __init__(self, world_size: int, max_sp: int = None, max_cp: int = None):
        """
        Args:
            world_size: Total number of GPUs
            max_sp: Maximum SP size to consider (default: world_size)
            max_cp: Maximum CP size to consider (default: world_size)
        """
        self.world_size = world_size
        self.rank = dist.get_rank()
        self.max_sp = max_sp or world_size
        self.max_cp = max_cp or world_size
        
        # (sp_size, cp_size) -> {"sp_group": ProcessGroup, "cp_group": ProcessGroup}
        self.groups: Dict[Tuple[int, int], Dict[str, dist.ProcessGroup]] = {}
        
        self._create_all_groups()
        
        if self.rank == 0:
            strategies = list(self.groups.keys())
            print(f"[AdaCPSP GroupManager] Created groups for {len(strategies)} strategies: {strategies}")
    
    def _powers_of_two(self, max_val: int) -> List[int]:
        """Return all powers of 2 up to max_val."""
        result = []
        p = 1
        while p <= max_val:
            result.append(p)
            p *= 2
        return result
    
    def _create_all_groups(self):
        """Create communication groups for all valid (sp_size, cp_size) combinations."""
        for sp_size in self._powers_of_two(min(self.max_sp, self.world_size)):
            for cp_size in self._powers_of_two(min(self.max_cp, self.world_size // sp_size)):
                if sp_size * cp_size <= self.world_size:
                    self._create_group_for_strategy(sp_size, cp_size)
    
    def _create_group_for_strategy(self, sp_size: int, cp_size: int):
        """Create SP and CP groups for a specific (sp_size, cp_size) strategy.
        
        Rank arrangement within each DP replica (sp_size * cp_size GPUs):
            - SP groups are consecutive ranks
            - CP groups span across SP groups
            
        For dp_replica base_rank, sp=S, cp=C:
            rank = base + cp_idx * S + sp_idx
            SP group: [base + cp_idx*S + 0, base + cp_idx*S + 1, ..., base + cp_idx*S + (S-1)]
            CP group: [base + 0*S + sp_idx, base + 1*S + sp_idx, ..., base + (C-1)*S + sp_idx]
        """
        dp_size = self.world_size // (sp_size * cp_size)
        
        my_sp_group = None
        my_cp_group = None
        
        # Create SP groups (collective - all ranks must participate)
        for dp_idx in range(dp_size):
            base = dp_idx * sp_size * cp_size
            for cp_idx in range(cp_size):
                ranks = [base + cp_idx * sp_size + sp_idx for sp_idx in range(sp_size)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    my_sp_group = group
        
        # Create CP groups (collective - all ranks must participate)
        for dp_idx in range(dp_size):
            base = dp_idx * sp_size * cp_size
            for sp_idx in range(sp_size):
                ranks = [base + cp_idx * sp_size + sp_idx for cp_idx in range(cp_size)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    my_cp_group = group
        
        # For sp_size=1, sp_group should still be valid (size 1 group)
        # For cp_size=1, cp_group should still be valid (size 1 group)
        self.groups[(sp_size, cp_size)] = {
            "sp_group": my_sp_group,
            "cp_group": my_cp_group,
            "sp_size": sp_size,
            "cp_size": cp_size,
            "dp_size": dp_size,
        }
    
    def get_groups(self, sp_size: int, cp_size: int) -> Dict:
        """Get the communication groups for a specific strategy."""
        key = (sp_size, cp_size)
        if key not in self.groups:
            raise ValueError(f"Strategy (sp={sp_size}, cp={cp_size}) not pre-created. "
                           f"Available: {list(self.groups.keys())}")
        return self.groups[key]
    
    def get_all_strategies(self) -> List[Tuple[int, int]]:
        """Return all available (sp_size, cp_size) strategies."""
        return list(self.groups.keys())


def set_model_strategy(
    model: torch.nn.Module,
    sp_size: int,
    cp_size: int,
    group_manager: CommunicationGroupManager,
):
    """
    Reconfigure all attention and embedding modules in the model 
    for the given (sp_size, cp_size) strategy.
    
    This updates:
    - SelfAttention: sp_group, cp_group, sp_size, cp_size, use_ulysses, use_zigzag_cp
    - DistributedAttention: spg (sp process group)
    - ZigzagRingFlashAttentionVarlen: cp_process_group
    - LlamaEmbeddings_: sp_group, cp_group, sp_size, cp_size
    """
    groups = group_manager.get_groups(sp_size, cp_size)
    sp_group = groups["sp_group"]
    cp_group = groups["cp_group"]
    
    use_ulysses = sp_size > 1
    use_zigzag_cp = cp_size > 1
    
    for name, module in model.named_modules():
        # Update SelfAttention modules
        if isinstance(module, SelfAttention):
            module.sp_group = sp_group
            module.cp_group = cp_group
            module.sp_size = sp_size
            module.cp_size = cp_size
            module.use_ulysses = use_ulysses
            module.use_zigzag_cp = use_zigzag_cp
            
            # Update DistributedAttention's sp group and local_attention
            if hasattr(module, 'dist_attn'):
                module.dist_attn.spg = sp_group
                # Switch local attention based on new strategy
                if use_zigzag_cp and hasattr(module, 'zigzag_ring_flash_attn'):
                    module.dist_attn.local_attn = module.zigzag_ring_flash_attn
                elif hasattr(module, 'flash_attention'):
                    module.dist_attn.local_attn = module.flash_attention
            
            # Update ZigzagRingFlashAttention's cp group
            if hasattr(module, 'zigzag_ring_flash_attn'):
                module.zigzag_ring_flash_attn.cp_process_group = cp_group
        
        # Update LlamaEmbeddings_ modules
        elif hasattr(module, 'embed_tokens') and hasattr(module, 'cp_size') and hasattr(module, 'sp_size'):
            # This matches LlamaEmbeddings_ which has embed_tokens + cp_size + sp_size
            if hasattr(module, 'cp_group'):
                module.sp_group = sp_group
                module.cp_group = cp_group
                module.sp_size = sp_size
                module.cp_size = cp_size


def strategy_from_solver_result(attn_type: str, parallel_size: int) -> Tuple[int, int]:
    """
    Convert solver's (attn_type, parallel_size) to (sp_size, cp_size).
    
    Args:
        attn_type: "ulysses" or "ring"
        parallel_size: Number of GPUs for this attention type
    
    Returns:
        (sp_size, cp_size) tuple
    """
    if attn_type == "ulysses":
        return (parallel_size, 1)
    elif attn_type == "ring":
        return (1, parallel_size)
    else:
        raise ValueError(f"Unknown attn_type: {attn_type}")

