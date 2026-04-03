import time
from typing import Any, Dict, List, Optional

import torch

from .base_profiler import BaseProfiler
from .utils import print_peak_memory, save_profiled_memory, save_profiled_time


class RuntimeProfiler(BaseProfiler):
    """Runtime profiler for monitoring memory usage and computation time during model execution."""

    def __init__(self, args: Any):
        """Initialize runtime profiler

        Args:
            args: Arguments containing profiling configuration
        """
        super().__init__(args)
        self.step_context: Dict[str, Any] = {}
        self.time_log_rank: Optional[int] = None

    def set_profiler_dist(
        self,
        path: Optional[str] = None,
        model_layer_configs: Optional[List[Dict]] = None,
        model_name: Optional[str] = None,
        profile_ranks: Optional[List[int]] = None,
        start_iter: int = 10,
        end_iter: int = 20,
        rank: Optional[int] = None,
    ) -> None:
        """Configure distributed profiling settings

        Args:
            path: Path to save profiling results
            model_layer_configs: List of layer configurations containing:
                - hidden_size: Hidden dimension size
                - layer_num: Number of layers
                - seq_len: Sequence length
            model_name: Name of the model being profiled
            profile_ranks: List of ranks to profile (default: [0, world_size-1])
            start_iter: Starting iteration for profiling
            end_iter: Ending iteration for profiling
            rank: Current process rank (default: get from torch.distributed)
        """
        rank = torch.distributed.get_rank() if rank is None else rank
        if profile_ranks is None:
            world_size = torch.distributed.get_world_size()
            profile_ranks = [0, world_size - 1]

        self.set_path(path)
        self.set_model_name(model_name)
        self.set_model_layer_configs(model_layer_configs)
        self.set_memory_profiler(rank, profile_ranks)
        self.time_log_rank = profile_ranks[-1] if profile_ranks else rank

        exit_ = self.args.exit_after_profiling if hasattr(self.args, "exit_after_profiling") else True
        self.set_time_profiler(start_iter=start_iter, end_iter=end_iter, exit=exit_)

    def set_profiler_single(self, start_iter=10, end_iter=20):
        """
        Set profiler for single process

        Args:
            start_iter: Starting iteration for profiling
            end_iter: Ending iteration for profiling
        """
        self.set_memory_profiler(0)
        exit_ = self.args.exit_after_profiling if 'exit_after_profiling' in self.args else True
        self.set_time_profiler(start_iter=start_iter, end_iter=end_iter, exit=exit_)
    
    def set_model_layer_configs(self, model_layer_configs: Optional[List[Dict]]) -> None:
        """Set model layer configurations

        Args:
            model_layer_configs: List of layer configurations containing:
                - hidden_size: Hidden dimension size
                - layer_num: Number of layers
                - seq_len: Sequence length
        """
        if model_layer_configs is None:
            return
        self.hiddensize_list = [config["hidden_size"] for config in model_layer_configs]
        self.layernum_list = [config["layer_num"] for config in model_layer_configs]
        self.seqlen_list = [config["seq_len"] for config in model_layer_configs]

    # =============== Memory Profiling ===============
    def set_memory_profiler(self, rank: int, profile_ranks: List[int] = [], max_profile_iter: int = 5) -> None:
        """Configure memory profiler settings

        Args:
            rank: Current process rank
            profile_ranks: List of ranks to profile
            max_profile_iter: Maximum number of iterations to profile
        """
        self.rank = rank
        self.profile_ranks = profile_ranks if len(profile_ranks) > 0 else [rank]
        self.mem_dict = {}
        self.max_profile_iter = max_profile_iter

    def set_time_log_rank(self, rank: int) -> None:
        self.time_log_rank = rank

    def set_step_context(self, **context: Any) -> None:
        self.step_context = {k: v for k, v in context.items() if v is not None}

    def clear_step_context(self) -> None:
        self.step_context = {}

    def _get_data_batch_size(self) -> int:
        if hasattr(self.args, "profile_data_batch_size"):
            return int(self.args.profile_data_batch_size)
        if hasattr(self.args, "global_train_batch_size"):
            return int(self.args.global_train_batch_size)
        if hasattr(self.args, "global_batch_size"):
            return int(self.args.global_batch_size)
        return 0

    def _get_scheduler_batch_size(self) -> int:
        if hasattr(self.args, "profile_scheduler_batch_size"):
            return int(self.args.profile_scheduler_batch_size)
        if hasattr(self.args, "global_batch_size"):
            return int(self.args.global_batch_size)
        return self._get_data_batch_size()

    def _get_display_iter(self, iter: int) -> int:
        return int(self.step_context.get("train_step_id", iter + 1))

    def _get_context_fields(self) -> List[str]:
        fields = [f"rank={self.rank}"]
        for key in ("loader_iter", "train_step_id", "training_batch_id", "solver_batch_id"):
            if key in self.step_context:
                fields.append(f"{key}={self.step_context[key]}")

        data_batch_size = self._get_data_batch_size()
        if data_batch_size > 0:
            fields.append(f"data_batch_size={data_batch_size}")

        scheduler_batch_size = self._get_scheduler_batch_size()
        if scheduler_batch_size > 0 and scheduler_batch_size != data_batch_size:
            fields.append(f"scheduler_batch_size={scheduler_batch_size}")

        return fields

    def _format_tag(
        self,
        tag: str,
        extra_fields: Optional[List[str]] = None,
        include_step_context: bool = True,
    ) -> str:
        fields = self._get_context_fields() if include_step_context else [f"rank={self.rank}"]
        if not include_step_context:
            data_batch_size = self._get_data_batch_size()
            if data_batch_size > 0:
                fields.append(f"data_batch_size={data_batch_size}")
            scheduler_batch_size = self._get_scheduler_batch_size()
            if scheduler_batch_size > 0 and scheduler_batch_size != data_batch_size:
                fields.append(f"scheduler_batch_size={scheduler_batch_size}")
        if extra_fields:
            fields.extend(extra_fields)
        return "".join([f"[{tag}]"] + [f"[{field}]" for field in fields])

    def _resolve_summary_iter(self, current_iter: int) -> int:
        available_iters = []
        for key in self.mem_dict:
            if key.startswith("iter_") and key.endswith("_after_backward"):
                try:
                    available_iters.append(int(key.split("_")[1]))
                except (IndexError, ValueError):
                    continue

        if current_iter in available_iters:
            return current_iter
        if available_iters:
            return max(available_iters)
        return current_iter

    def _summary_window_fields(self) -> List[str]:
        fields = [f"profiled_loader_iters=[{self.start_iter},{self.end_iter})"]
        if getattr(self.args, "use_adaCPSP", False) and not getattr(self.args, "adaCPSP_sync_solver", False):
            start_step = self.start_iter
            end_step = self.end_iter - 1
        else:
            start_step = self.start_iter + 1
            end_step = self.end_iter
        fields.append(f"profiled_train_steps=[{start_step},{end_step}]")
        fields.append(f"measured_steps={len(self.time_list)}")
        return fields

    def profile_memory(self, iter: int, stage: str = "") -> None:
        """Profile memory usage at different stages of training

        Args:
            iter: Current iteration number
            stage: Profiling stage ("Before Forward", "After Forward", "After Backward")
        """
        args, rank = self.args, self.rank
        profile_ranks, mem_dict = self.profile_ranks, self.mem_dict
        max_profile_iter = self.max_profile_iter

        if args.profile and rank in profile_ranks and iter <= max_profile_iter:
            local_rank = args.local_rank if hasattr(args, "local_rank") else 0
            profile_type = args.profile_type if hasattr(args, "profile_type") else "allocated"
            stage_label = f"{self._format_tag('Profile')} {stage}" if stage else self._format_tag("Profile")

            if stage == "Before Forward":
                torch.cuda.reset_peak_memory_stats(local_rank)
                _, cur_mem = print_peak_memory("\n" + stage_label, local_rank, profile_type)
                mem_dict[f"iter_{iter}_before_forward"] = cur_mem
            elif stage == "After Forward":
                _, cur_mem = print_peak_memory(stage_label, local_rank, profile_type)
                mem_dict[f"iter_{iter}_after_forward"] = cur_mem
            elif stage == "After Backward":
                max_mem, cur_mem = print_peak_memory(stage_label, local_rank, profile_type)
                mem_dict[f"iter_{iter}_after_backward"] = cur_mem
                mem_dict[f"iter_{iter}_after_backward_max"] = max_mem
            else:
                print_peak_memory(stage_label, local_rank, profile_type)

    def post_profile_memory(self, iter: int) -> None:
        """Post-process and save memory profiling results

        Args:
            iter: Current iteration number
        """
        args, rank = self.args, self.rank
        profile_ranks, mem_dict = self.profile_ranks, self.mem_dict
        max_profile_iter = self.max_profile_iter

        if args.profile and iter == max_profile_iter:
            if rank in profile_ranks:
                summary_iter = self._resolve_summary_iter(iter)
                after_backward_key = f"iter_{summary_iter}_after_backward"
                after_backward_max_key = f"iter_{summary_iter}_after_backward_max"
                after_forward_key = f"iter_{summary_iter}_after_forward"
                before_forward_key = f"iter_{summary_iter}_before_forward"

                if after_backward_key not in mem_dict or after_backward_max_key not in mem_dict:
                    return

                # Calculate memory statistics
                mem_dict["model_states"] = mem_dict[after_backward_key]

                if (not hasattr(args, "pipeline_type") or args.pipeline_type == "gpipe") and after_forward_key in mem_dict and before_forward_key in mem_dict:
                    mem_dict["model_states_and_activation"] = mem_dict[after_forward_key]
                    mem_dict["activation"] = (
                        mem_dict[after_forward_key]
                        - mem_dict[before_forward_key]
                    )

                mem_dict["model_states_and_peak_activation"] = mem_dict[after_backward_max_key]
                mem_dict["peak_activation"] = (
                    mem_dict[after_backward_max_key]
                    - mem_dict[after_backward_key]
                )

                # Print results
                time.sleep(0.2 * rank)
                print(self._format_tag("Profiled memory", [f"summary_iter={summary_iter}"]) + ":")
                for key, val in mem_dict.items():
                    print(f"\t{key}: {val:.2f} MB")

                # Save results if requested
                if hasattr(args, "save_profiled_memory") and args.save_profiled_memory:
                    assert self.layernum_list is not None
                    world_size = torch.distributed.get_world_size()
                    memory_config_path = self.memory_profiling_path()

                    save_profiled_memory(
                        memory_config_path,
                        args.pp_deg,
                        args.global_tp_deg,
                        world_size,
                        self.layernum_list,
                        args.global_train_batch_size,
                        rank,
                        mem_dict["model_states"],
                        mem_dict["activation"],
                        mem_dict["peak_activation"],
                        args.global_checkpoint,
                        args.sequence_parallel,
                        args.vocab_tp,
                        self.seqlen_list,
                    )

            if hasattr(args, "save_profiled_memory") and args.save_profiled_memory:
                exit(0)

    # =============== Time Profiling ===============
    def set_time_profiler(self, start_iter: int, end_iter: int, exit: bool = False) -> None:
        """Configure time profiler settings

        Args:
            start_iter: Starting iteration for profiling
            end_iter: Ending iteration for profiling
            exit: Whether to exit after profiling
        """
        self.start_iter = start_iter
        self.end_iter = end_iter
        assert end_iter > start_iter, "End iteration must be greater than start iteration"

        self.exit = exit
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.time_list = []
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_size = 1

    def profile_time_start(self, iter: int) -> None:
        """Start timing for current iteration

        Args:
            iter: Current iteration number
        """
        if not self.args.profile:
            return

        if iter >= self.start_iter and iter < self.end_iter:
            torch.cuda.synchronize()
            self.start.record()
        elif iter == self.end_iter:
            self._process_time_results()

    def profile_time_end(
        self,
        iter: int,
        loss: Optional[torch.Tensor] = None,
        learning_rate: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> None:
        """End timing for current iteration and log results

        Args:
            iter: Current iteration number
            loss: Training loss value
            learning_rate: Current learning rate
            grad_norm: Gradient norm
        """
        if not self.args.profile:
            return

        if iter >= self.start_iter and iter < self.end_iter:
            self.end.record()
            torch.cuda.synchronize()
            iter_time = self.start.elapsed_time(self.end) / 1e3
            self.time_list.append(iter_time)

            time_log_rank = self.world_size - 1 if self.time_log_rank is None else self.time_log_rank
            if self.rank == time_log_rank:
                self._log_iteration_stats(iter, iter_time, loss, learning_rate, grad_norm)

    def profile_time_python(self, iter: int) -> None:
        """Profile time using Python's time module (coarse timing)

        Args:
            iter: Current iteration number
        """
        if not self.args.profile:
            return

        if iter == self.start_iter:
            self.total_start_time = time.time()
        elif iter == self.end_iter:
            self.total_end_time = time.time()
            avg_time = (self.total_end_time - self.total_start_time) / (self.end_iter - self.start_iter)
            time_log_rank = self.world_size - 1 if self.time_log_rank is None else self.time_log_rank
            if self.rank == time_log_rank:
                print(self._format_tag("Profile Summary", self._summary_window_fields(), include_step_context=False))
                print(f"Average iteration time is: {avg_time:.4f} s")

            args = self.args
            if hasattr(args, "profile_forward") and args.profile_forward:
                assert self.layernum_list is not None
                time_config_path = self.time_profiling_path()
                save_profiled_time(
                    time_config_path, avg_time, args.global_train_batch_size, self.layernum_list, self.seqlen_list
                )

            if self.exit:
                exit(0)
            else:
                self.start_iter, self.end_iter = self.end_iter, (self.end_iter - self.start_iter + self.end_iter)
                self.total_start_time = time.time()

    def _process_time_results(self) -> None:
        """Process and save time profiling results"""
        avg_time = sum(self.time_list) / len(self.time_list)
        time_log_rank = self.world_size - 1 if self.time_log_rank is None else self.time_log_rank
        if self.rank == time_log_rank:
            print(self._format_tag("Profile Summary", self._summary_window_fields(), include_step_context=False))
            print(f"Average iteration time is: {avg_time:.4f} s")

        args = self.args
        if hasattr(args, "profile_forward") and args.profile_forward:
            assert self.layernum_list is not None
            time_config_path = self.time_profiling_path()
            save_profiled_time(
                time_config_path, avg_time * 1e3, args.global_train_batch_size, self.layernum_list, self.seqlen_list
            )

        if self.exit:
            exit(0)
        else:
            self.time_list = []
            self.start_iter, self.end_iter = self.end_iter, (self.end_iter - self.start_iter + self.end_iter)
            torch.cuda.synchronize()
            self.start.record()

    def _log_iteration_stats(
        self,
        iter: int,
        iter_time: float,
        loss: Optional[torch.Tensor],
        learning_rate: Optional[float],
        grad_norm: Optional[float],
    ) -> None:
        """Log iteration statistics

        Args:
            iter: Current iteration number
            iter_time: Iteration time in seconds
            loss: Training loss value
            learning_rate: Current learning rate
            grad_norm: Gradient norm
        """
        if loss is None:
            print(iter_time)
        else:
            display_iter = self._get_display_iter(iter)
            data_batch_size = self._get_data_batch_size()
            consumed_samples = display_iter * data_batch_size if data_batch_size > 0 else display_iter
            loss_value = loss.item() if hasattr(loss, "item") else float(loss)
            log_parts = [
                "| Iteration: {:6d} | Consumed samples: {:12d} | ",
                "Elapsed time per iteration (ms): {:.1f} | ",
                "Learning rate: {:.6e} | Loss: {:.6e} | ",
                "grad norm: {:.2f} |",
            ]
            message = "".join(log_parts)
            print(self._format_tag("Profile Step"))
            print(
                message.format(
                    display_iter,
                    consumed_samples,
                    iter_time * 1e3,
                    self.args.lr if learning_rate is None else learning_rate,
                    loss_value,
                    0.0 if grad_norm is None else grad_norm,
                )
            )
