# Align CostModel

这套目录现在是一个**纯 attention benchmark harness**，目标不是跑完整 AdaCPSP 训练系统，而是做一条更干净的对齐链路：

1. 单卡 `attention compute profile`
2. 多卡 `topology-aware communication profile`
3. 多卡真实 `ulysses / ring / usp` attention 实现 benchmark
4. 用 `AdaCPSPCostModel` 做预测，并和 benchmark 结果对齐

这里的第 3 步**不调用** `train_dist_adacpsp.py`。  
它会：

- 手动 `init_process_group`
- 手动建 outer group
- 手动建 `sp_group / cp_group`
- 直接调用底层 attention 实现
  - `FlashSelfAttentionVarlen`
  - `DistributedAttention`
  - `ZigzagRingFlashAttentionVarlen`

这样可以把：

- solver
- packing
- dataloader
- optimizer
- FSDP / ZeRO

这些系统噪声排除掉，只对齐 attention compute + communication。

## 文件说明

- `00_common.sh`
  - 公共环境变量、模型 meta config 加载、结果目录、`torchrun` 前缀、benchmark case 枚举
- `01_profile_attention.sh`
  - 单卡 attention profile
- `02_profile_comm.sh`
  - 多卡 topology-aware communication profile
- `03_benchmark_attention.py`
  - 纯 attention benchmark harness
- `03_run_real_strategies.sh`
  - 多卡 benchmark wrapper，循环跑 `local / ulysses / ring / usp`
- `04_align_costmodel.py`
  - 加载 profile 与 benchmark 结果，调用 `AdaCPSPCostModel` 做对齐分析
- `05_run_all.sh`
  - 一键串起整个流程

## 默认实验假设

- 主要面向 `16` 卡对齐实验
- 建议典型拓扑：`2 x 8 GPUs`
- 默认先做“固定长度 batch”对齐，而不是混入真实 varlen dataset 噪声
- benchmark 会按 `BENCH_GROUP_SIZES` 和 world size 枚举：
  - `local(size=1)`
  - `ulysses(size=P)`
  - `ring(size=P)`
  - `usp(sp, cp), sp * cp = P`
  - `group_topology = consecutive / strided`
  - `placement = context_first / head_first`（仅 USP）

## 运行前准备

### 1. 统一所有节点的环境变量

在所有参与实验的节点上，使用相同的：

```bash
export MODEL_NAME=qwen2.5-7b
export NNODES=2
export NPROC_PER_NODE=8
export MASTER_ADDR=<master_ip_or_hostname>
export MASTER_PORT=29500
export ALIGN_RUN_ID=align_qwen25_7b_16gpu_001
```

然后每个节点设置自己的：

```bash
export NODE_RANK=0   # 主节点
export NODE_RANK=1   # 第二个节点
```

### 2. 可选参数

```bash
export GLOBAL_BATCH_SIZE=16
export ALIGN_SEQ_LENGTHS="2048 4096 8192 16384"
export WARMUP_ITERS=5
export MEASURE_ITERS=20
export ACROSS_GROUP_AGG=p90
export BENCH_GROUP_SIZES="1 2 4 8 16"
```

说明：

- `GLOBAL_BATCH_SIZE` 在这里表示 benchmark 的全局序列条数
- `ALIGN_SEQ_LENGTHS` 是每条序列的固定长度候选
- `BENCH_GROUP_SIZES` 控制要 benchmark 哪些 group size

## 推荐执行方式

### 方式 A：一步一步执行

### Step 1. attention profile

只在主节点执行：

```bash
bash galvatron/models/varlen_llama_hf/align_costmodel/01_profile_attention.sh
```

### Step 2. communication profile

在所有节点执行相同命令：

```bash
bash galvatron/models/varlen_llama_hf/align_costmodel/02_profile_comm.sh
```

### Step 3. 纯 attention benchmark

在所有节点执行相同命令：

```bash
bash galvatron/models/varlen_llama_hf/align_costmodel/03_run_real_strategies.sh
```

### Step 4. cost model 对齐分析

只在主节点执行：

```bash
python3 galvatron/models/varlen_llama_hf/align_costmodel/04_align_costmodel.py \
  --result-dir galvatron/models/varlen_llama_hf/align_costmodel/results/${ALIGN_RUN_ID} \
  --world-size $((NNODES * NPROC_PER_NODE)) \
  --gpus-per-node "${NPROC_PER_NODE}"
```

---

### 方式 B：一键执行

在所有节点执行相同命令：

```bash
bash galvatron/models/varlen_llama_hf/align_costmodel/05_run_all.sh
```

其中：

- `01_profile_attention.sh` 会自动只在 `NODE_RANK=0` 真正执行
- `04_align_costmodel.py` 会自动只在 `NODE_RANK=0` 执行

## 输出结果

结果目录：

```text
galvatron/models/varlen_llama_hf/align_costmodel/results/${ALIGN_RUN_ID}/
```

重点文件：

- `summary/run_metadata.txt`
  - 本次实验的模型、集群和参数记录
- `summary/attention_profile_path.txt`
  - attention profile 的路径
- `summary/comm_profile_path.txt`
  - communication profile 的路径
- `summary/bench_*.json`
  - 每个 benchmark case 的真实测量结果
- `summary/costmodel_alignment.csv`
  - 对齐后的结构化结果
- `summary/costmodel_alignment.md`
  - 对齐结果摘要，方便直接阅读

日志目录：

- `logs/01_profile_attention_*.log`
- `logs/02_profile_comm_*.log`
- `logs/03_bench_*.log`

## 对齐口径说明

这套 harness 默认测的是：

- `fixed-length sequences`
- `single attention implementation only`
- `manual communication group construction`
- `one-layer forward + backward wall time`

所以 `04_align_costmodel.py` 输出的核心对齐口径是：

- `predicted_total_per_layer_ms`
- `predicted_comm_per_layer_ms`
- `measured_per_layer_ms`
- `error_pct`

它会：

1. 读取 attention profile
2. 读取 `comm_profile_*.json`
3. 构建 `AdaCPSPCostModel`
4. 对每个 benchmark case 构造对应 `ParallelStrategy`
5. 根据每个 outer group 的 `group_num_seqs` 重建 workload
6. 计算每个 group 的：
   - `costmodel.total_time(...)`
   - `costmodel.comm_time(...)`
7. 取所有 group 的 `max(...)`，再换算成每层时间

因此这套实验回答的问题是：

- **cost model 是否能准确拟合单一策略、固定长度 workload 下的 attention 实际代价**

而不是：

- 完整训练系统的最终 iteration time
- solver 自动搜索后的端到端收益

## 进一步扩展

如果后面你想升级到更接近真实 workload 的版本，可以逐步加入：

- 真实 varlen batch，而不是固定长度序列
- 异构 flat partition 的 benchmark
- 更细粒度的 forward / backward 分离计时
- 让 `predicted_comm_ms` 与 primitive comm profile 再做单独对齐
