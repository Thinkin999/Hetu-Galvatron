from .hybrid_parallel_config import ModelInfo, get_hybrid_parallel_configs_api, mixed_precision_dtype
from .hybrid_parallel_model import construct_hybrid_parallel_model_api
from .initialize import init_empty_weights
from .utils import clip_grad_norm, get_optimizer_and_param_scheduler, set_megatron_args_for_dataset

# ======== FSDP patch ========
# When using expilict forward refetch, we need to set the _prefetched handle at any case.
import torch

if torch.__version__ >= "2.1.0" and torch.__version__ < "2.2.0":
    import torch.distributed.fsdp as fsdp
    from torch.distributed.fsdp._exec_order_utils import _ExecOrderData
    from torch.distributed.fsdp._runtime_utils import (
        _FSDPState,
    )
    from torch.distributed.fsdp.flat_param import (
        FlatParamHandle,
    )
    from typing import no_type_check

    @no_type_check
    def _reshard(
        state: _FSDPState,
        handle: FlatParamHandle,
        free_unsharded_flat_param: bool,
    ):
        """
        Reshards the handle. ``free_unsharded_flat_param`` indicates whether to
        free the handle's padded unsharded flat parameter.
        """
        handle.reshard(free_unsharded_flat_param)
        if state.limit_all_gathers and free_unsharded_flat_param:
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                # We don't run a even queue for freeing under torch compile atm
                # But maybe we need to? TODO(voz): Look into this
                free_event = state._device_handle.Event()
                free_event.record()
                state._free_event_queue.enqueue(free_event)
        handle.post_reshard()
        # Since we prefetch entire handles keys at a time, conservatively mark
        # the entire key as no longer prefetched once we free at least one
        # if free_unsharded_flat_param:
        handle._prefetched = False

    fsdp._runtime_utils._reshard = _reshard

    # AdaCPSP dynamically switches the active attention implementation
    # (Flash / Ulysses / Ring) per microbatch via set_model_strategy().
    # This causes the inner FSDP handles to execute in varying order across
    # iterations, triggering _check_order's strict-consistency assertion.
    # Skipping the check disables forward-prefetch bookkeeping (minor perf
    # impact) but does not affect correctness — all_gather is still invoked
    # by the regular forward path.
    _orig_check_order = _ExecOrderData._check_order

    @no_type_check
    def _check_order_for_adacpsp(self, handle, is_training):
        try:
            from megatron.training import get_args as _get_megatron_args
            args = _get_megatron_args()
            if getattr(args, "use_adaCPSP", False):
                return
        except Exception:
            pass
        return _orig_check_order(self, handle, is_training)

    _ExecOrderData._check_order = _check_order_for_adacpsp