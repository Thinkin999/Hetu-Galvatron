# AdaCPSP CostModel Alignment Brief

## 1. End-to-End CostModel 对齐总规划

我们的最终目标不是只预测 attention 时间，而是得到一个能较准确预测训练 step time 和加速比的 end-to-end cost model。

完整训练 step 可以拆成：

```text
T_step
  = T_attention_wrapper
  + T_non_attention_compute
  + T_optimizer_and_runtime
```

其中：

```text
T_attention_wrapper
  = attention compute
  + Ulysses / Ring / USP communication
  + overlap / wrapper overhead

T_non_attention_compute
  = QKV projection
  + O projection
  + MLP / FFN
  + norm / residual

T_optimizer_and_runtime
  = optimizer step
  + FSDP / ZeRO common communication
  + dataloader / Python / scheduling overhead
```

因此对齐顺序应该是：

```text
Step 1: 先对齐 attention wrapper
  local / Ulysses / Ring / USP 的 measured wrapper time
  vs cost model predicted attention time

Step 2: 再对齐 communication primitive
  alltoall_single(message_MB) -> time_ms
  p2p_sendrecv(message_MB) -> time_ms

Step 3: 在 attention + communication 可解释后，再看完整 train step
  residual = measured_step_time - aligned_attention_wrapper_prediction

Step 4: residual 再用于拟合非 attention 部分
  residual ≈ MLP/projection/norm + optimizer/runtime
```

这一步很关键：如果 attention 和 communication 还没有对齐，就直接拟合 end-to-end residual，会把所有误差混在一起，得不到可解释的模型。

## 2. 当前 CostModel 原本的问题

当前 AdaCPSP CostModel 原本主要建模 attention wrapper，而不是完整 transformer layer。

它包含：

- local FlashAttention compute
- Ulysses All-to-All communication
- Ring P2P communication
- USP 的 All-to-All + Ring P2P
- attention backward/forward ratio
- Ring/USP overlap

它不包含：

- QKV projection
- O projection
- MLP / FFN
- norm / residual
- optimizer
- FSDP / ZeRO 公共通信

所以当前阶段不能直接用它预测完整训练 step。

## 3. 我们现在正在做什么

我们现在处于第一阶段：

```text
对齐 attention wrapper cost model
```

具体做法是：

1. 使用 `03_benchmark_attention.py` 测真实 attention wrapper：

```text
local
ring
usp
ulysses
```

2. 使用 `AdaCPSPCostModel` 预测同一批 case。

3. 比较：

```text
measured attention wrapper time
vs
predicted attention wrapper time
```

这样可以判断：

```text
attention compute 是否准
communication profile 是否准
overlap model 是否准
```

## 4. 已完成的几个关键校准

### 4.1 Attention backward/forward ratio

原模型默认：

```text
bwd_fwd_ratio = 2.0
forward + backward = 3.0 * forward
```

我们新增脚本：

```text
align_costmodel/07_profile_attention_bwd_fwd_ratio.py
```

测得 qwen2.5-7b local attention wrapper：

```text
attention backward ≈ 3.1-3.3 * forward
attention forward + backward ≈ 4.1-4.3 * forward
```

因此临时改为：

```text
bwd_fwd_ratio = 3.2
```

效果：

```text
local attention wrapper error:
  约 25-30% -> 约 2-5%
```

说明 local attention compute 已基本对齐。

### 4.2 通信 profile v2

旧通信 profile 同时包含：

```text
raw
model
interp
linear_fit
bandwidth_dict
```

语义混杂，不容易判断到底在预测什么。

我们新增：

```text
profile_comm_v2.py
align_costmodel/07_profile_comm_v2.sh
```

v2 的原则是：

```text
profile 只测 primitive communication
cost model 负责把模型语义转换成 message_MB
```

当前 v2 只测：

```text
alltoall_single(group_size, topology, message_MB) -> time_ms
p2p_sendrecv(group_size, topology, message_MB) -> time_ms
```

这让通信建模更清晰：

```text
Ulysses / USP 的 A2A 查 alltoall_single
Ring / USP 的 P2P 查 p2p_sendrecv
```

### 4.3 Ring P2P 口径修正

旧 Ring P2P 模型把：

```text
K + V
```

近似成一个串行大 tensor 通信，因此严重高估 Ring time。

真实 Ring step 是：

```text
send K
send V
recv K
recv V
```

K/V 是并发 P2P ops。

因此现在用：

```text
single_kv_MB
```

去查：

```text
p2p_sendrecv(single_kv_MB)
```

而不是用：

```text
K_plus_V_MB
```

效果：

```text
Ring attention wrapper error:
  约 80-100% -> 约 15%
```

### 4.4 Overlap 公式语义修正

旧公式叫：

```text
leaky_max
```

容易误解为“溢出”。

我们改成更清楚的 overlap slowdown 形式：

```text
tail = abs(compute - comm)
overlap = min(compute, comm)
time = tail + overlap_slowdown * overlap
```

含义：

```text
overlap_slowdown = 1.0: 完美 overlap
overlap_slowdown = 1.1: overlap 区间慢 10%
overlap_slowdown = 2.0: 完全不 overlap
```

注意：这个改名/改公式是语义澄清，数学上和旧公式等价，因此不会单独降低误差。

## 5. 当前对齐结果

使用 attention ratio + comm profile v2 后：

```text
local error ≈ 3.5%
ring error ≈ 14.7%
usp error ≈ 20.2%
```

其中：

```text
local 已基本对齐
ring 长序列非常准，短序列仍低估
usp 长序列改善明显，但还需要等 Ulysses/A2A 对齐后再分析
```

Ring 的具体情况：

```text
ring_p16_seq8192:
  measured = 68.61
  predicted = 48.91
  error = 28.7%

ring_p16_seq16384:
  measured = 117.58
  predicted = 100.14
  error = 14.8%

ring_p16_seq32768:
  measured = 256.72
  predicted = 255.39
  error = 0.5%
```

说明：

```text
Ring 长序列已经很好。
短序列还缺少 wrapper / per-step overhead。
```

## 6. 当前未解决的问题

### 6.1 Ring 短序列低估

短序列下，Ring wrapper 有额外开销：

```text
kernel launch
P2P scheduling
step-level Python/CUDA gap
small-message latency
```

我们尝试过一个临时项：

```text
ring_step_overhead_ms ≈ 0.43
```

可以把 Ring error 降到：

```text
约 6%
```

但这还是经验项。后续更合理的形式应该是：

```text
ring_step_overhead_ms = f(local_seq, cp_size)
```

### 6.2 Ulysses 还没对齐

Qwen2.5-7B 的：

```text
n_heads = 28
n_kv_heads = 4
```

导致 `ulysses:16` benchmark 目前跑不通，需要 head padding 支持。

临时方案是用可整除 head 数跑 Ulysses wrapper，先验证 A2A 行为。

### 6.3 USP 暂时不继续深入

USP 同时包含：

```text
All-to-All
Ring P2P
overlap
head padding
placement
```

因此在 Ulysses/A2A 和 Ring 都对齐前，不适合直接调 USP。

## 7. 下一步

短期优先级：

```text
1. 固定 comm_profile_v2 作为主方向。
2. 继续验证 Ring 不同 cp_size:
   ring:2 / ring:4 / ring:8 / ring:16
3. 拟合 ring_step_overhead_ms = f(local_seq, cp_size)。
4. 临时跑可整除 head 的 Ulysses benchmark，验证 A2A。
5. 等 Ring 与 A2A 都对齐后，再回到 USP。
6. 最后才进入完整 train step residual。
```

最终目标：

```text
用清晰的 primitive profile + 可解释的 overlap/overhead 校准，
替换旧的 raw/model/interp/bandwidth 混合 profile 逻辑。
```

