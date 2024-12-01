# Quick Start

## Profiling with Galvatron
The first step to use Galvatron is to profile the hardware environment and the model computation time. Galvatron will automatically save the profiled results into config files.

(1) Firstly, to profile the hardward environment, ```cd galvatron/profile_hardware```,  write the host address into ```hostfile```, set ```NUM_NODES, NUM_GPUS_PER_NODE, MPI_PATH``` in ```scripts/profile_hardware.sh``` and run:
``` shell
sh scripts/profile_hardware.sh
```

Galvatron will call [nccl-tests](https://github.com/NVIDIA/nccl-tests) to profile the communication bandwidth.

(2) Secondly, to profile the model computation time and memory usage, ```cd galvatron/models/model_name``` and run:
``` shell
sh scripts/profile_computation.sh
sh scripts/profile_memory.sh
```

## Parallelism Optimizing with Galvatron
After profiling the environments, Galvatron is able to automatically optimize the parallelism strategy for the given Transformer model. Given the memory budget, Galvatron provides the fine-grained hybrid parallel strategy with maximum throughput. The optimized parallelism strategy will be saved in `galvatron/models/model_name/configs` for the training. You can train the model with the provided optimal strategy to obtain the optimal throughput. 

To conduct parallelim optimization, ```cd galvatron/models/model_name```, customize ```NUM_NODES, NUM_GPUS_PER_NODE, MEMORY``` in ```scripts/search_dist.sh```, run:

``` shell
sh scripts/search_dist.sh
```

See more usage details of the customized parallelism optimization in [Galvatron Model Usage](../4_galvatron_model_usage/galvatron_model_usage.html#parallelism-optimizing-with-galvatron).

## Training with Galvatron
Galvatron provides a simple way to train Transformer models in fined-grained hybrid parallelism fashion. You can either train Transformer models with the searched optimal parallel strategy by specifying argument ```galvatron_config_path``` to obtain the optimal throughput, or use any parallel strategies as they like. Galvatron support two hybrid parallel config modes, including JSON config mode and GLOBAL config mode. Ypi can specify parallel strategies by modifying only a few arguments. 

To train the model with Galvatron, ```cd galvatron/models/model_name```, set ```NUM_NODES, NUM_GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, NODE_RANK```,  and run:
``` shell
sh scripts/train_dist.sh
```

See detailed guidance and more customized training options in [Galvatron Model Usage](../4_galvatron_model_usage/galvatron_model_usage.html#training-with-galvatron).