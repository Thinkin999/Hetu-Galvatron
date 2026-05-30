# ILP Horizontal Comparison: `real_random_max131072_b0`

## Batch
| field | value |
| --- | --- |
| sampling_mode | real_random |
| max_seq | 131072 |
| global_batch_size | 64 |
| seq_sum | 338784 |
| max / p95 / p99 | 98784 / 22195.20 / 60258.24 |
| >=128k / >=256k / >=384k / >=512k | 0 / 0 / 0 / 0 |
| sequences | [0:1280, 1:6464, 2:5600, 3:768, 4:320, 5:1664, 6:736, 7:512, 8:7232, 9:4960, 10:8320, 11:992, 12:4224, 13:1280, 14:960, 15:1824, 16:1760, 17:34816, 18:1920, 19:192, 20:64, 21:32, 22:1472, 23:896, 24:384, 25:192, 26:512, 27:1888, 28:98784, 29:1344, 30:2720, 31:64, 32:6656, 33:10272, 34:96, 35:96, 36:2176, 37:1184, 38:128, 39:1536, 40:9536, 41:896, 42:23136, 43:64, 44:2016, 45:352, 46:16864, 47:2944, 48:512, 49:640, 50:1280, 51:1472, 52:2560, 53:2464, 54:64, 55:5024, 56:10528, 57:480, 58:37632, 59:1152, 60:1664, 61:256, 62:192, 63:736] |

## Case Summary
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| status | ok | ok | ok | ok | ok |
| total_time_ms | 9189.0268 | 4822.8564 | 4822.8564 | 6064.9592 | 4822.8564 |
| speedup_vs_ulysses | 1.0000 | 1.9053 | 1.9053 | 1.5151 | 1.9053 |
| microbatch_count | 2 | 2 | 2 | 2 | 2 |
| solver_wall_s | 1.9931 | 2.1935 | 5.8101 | 9.7763 | 6.4532 |

## Microbatch 0
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| estimated_time_ms | 8387.81 | 4298.57 | 4298.57 | 4881.55 | 4298.57 |
| seq_sum | 171232 | 171232 | 171232 | 171232 | 171232 |
| num_sequences | 3 | 3 | 3 | 3 | 3 |
| sequences | [17:34816, 28:98784, 58:37632] | [17:34816, 28:98784, 58:37632] | [17:34816, 28:98784, 58:37632] | [17:34816, 28:98784, 58:37632] | [17:34816, 28:98784, 58:37632] |

| group metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| group_0.strategy | ulysses×4 | ring×4 | ulysses×4 | usp(sp4×cp4,cf) | ulysses×8 |
| group_0.num_gpus | 4 | 4 | 4 | 16 | 8 |
| group_0.sp_cp_placement | sp=4, cp=1, placement=head_first | sp=1, cp=4, placement=context_first | sp=4, cp=1, placement=head_first | sp=4, cp=4, placement=context_first | sp=8, cp=1, placement=head_first |
| group_0.estimated_time_ms | 1555.60 | 1267.84 | 1383.46 | 4294.40 | 2710.34 |
| group_0.estimated_memory_gb | 43.38 | 43.38 | 40.66 | 48.39 | 42.10 |
| group_0.memory_breakdown_gb | model=7.00, act=36.38, pad=0.00 | model=7.00, act=36.38, pad=0.00 | model=7.00, act=33.66, pad=0.00 | model=7.00, act=41.39, pad=0.00 | model=7.00, act=35.10, pad=0.08 |
| group_0.seq_sum_local | seq_sum=37632, local=9408.00 | seq_sum=37632, local=9408.00 | seq_sum=34816, local=8704.00 | seq_sum=171232, local=10702.00 | seq_sum=72448, local=9056.00 |
| group_0.sequence_ids | [58] | [58] | [17] | [28, 58, 17] | [58, 17] |
| group_0.sequence_lengths | [37632] | [37632] | [34816] | [98784, 37632, 34816] | [37632, 34816] |
| group_0.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided |
| group_0.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=2.0, kv=2.0 |
| group_0.compute | fwd=378.72, bwd=757.45 | step/layer=3.23, fwd/layer=13.21, bwd/layer=26.41 | fwd=324.95, bwd=649.90 | step/layer=6.68 | fwd=703.67, bwd=1407.35 |
| group_0.alltoall_comm | 419.44 | 0.00 | 408.61 | qo_op=10.79, kv_op=2.48, fwd/layer=26.54, bwd/layer=26.54 | 599.32 |
| group_0.ring_comm | 0.00 | fwd_step=1.00, bwd_step=1.99, fwd/layer=13.21, bwd/layer=26.41 | 0.00 | fwd_step=1.07, bwd_step=2.13, fwd/layer=27.04, bwd/layer=54.09 | 0.00 |
| group_0.time_model | additive, leakage=None | overlap, leakage=0.1 | additive, leakage=None | overlap, leakage=0.1 | additive, leakage=None |
| group_0.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_1.strategy | ulysses×4 | ring×4 | ulysses×4 |  | ring×8 |
| group_1.num_gpus | 4 | 4 | 4 |  | 8 |
| group_1.sp_cp_placement | sp=4, cp=1, placement=head_first | sp=1, cp=4, placement=context_first | sp=4, cp=1, placement=head_first |  | sp=1, cp=8, placement=context_first |
| group_1.estimated_time_ms | 1383.46 | 1097.80 | 1555.60 |  | 4298.57 |
| group_1.estimated_memory_gb | 40.66 | 40.66 | 43.38 |  | 54.75 |
| group_1.memory_breakdown_gb | model=7.00, act=33.66, pad=0.00 | model=7.00, act=33.66, pad=0.00 | model=7.00, act=36.38, pad=0.00 |  | model=7.00, act=47.75, pad=0.00 |
| group_1.seq_sum_local | seq_sum=34816, local=8704.00 | seq_sum=34816, local=8704.00 | seq_sum=37632, local=9408.00 |  | seq_sum=98784, local=12348.00 |
| group_1.sequence_ids | [17] | [17] | [58] |  | [28] |
| group_1.sequence_lengths | [34816] | [34816] | [37632] |  | [98784] |
| group_1.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided |  | a2a=strided, ring=consecutive |
| group_1.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |  | q=1.0, kv=1.0 |
| group_1.compute | fwd=324.95, bwd=649.90 | step/layer=2.79, fwd/layer=11.44, bwd/layer=22.87 | fwd=378.72, bwd=757.45 |  | step/layer=5.41, fwd/layer=44.78, bwd/layer=89.55 |
| group_1.alltoall_comm | 408.61 | 0.00 | 419.44 |  | 0.00 |
| group_1.ring_comm | 0.00 | fwd_step=0.96, bwd_step=1.92, fwd/layer=11.44, bwd/layer=22.87 | 0.00 |  | fwd_step=2.12, bwd_step=4.24, fwd/layer=44.78, bwd/layer=89.55 |
| group_1.time_model | additive, leakage=None | overlap, leakage=0.1 | additive, leakage=None |  | overlap, leakage=0.1 |
| group_1.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |  | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_2.strategy | ulysses×8 | ring×8 | ring×8 |  |  |
| group_2.num_gpus | 8 | 8 | 8 |  |  |
| group_2.sp_cp_placement | sp=8, cp=1, placement=head_first | sp=1, cp=8, placement=context_first | sp=1, cp=8, placement=context_first |  |  |
| group_2.estimated_time_ms | 8387.81 | 4298.57 | 4298.57 |  |  |
| group_2.estimated_memory_gb | 54.86 | 54.75 | 54.75 |  |  |
| group_2.memory_breakdown_gb | model=7.00, act=47.86, pad=0.11 | model=7.00, act=47.75, pad=0.00 | model=7.00, act=47.75, pad=0.00 |  |  |
| group_2.seq_sum_local | seq_sum=98784, local=12348.00 | seq_sum=98784, local=12348.00 | seq_sum=98784, local=12348.00 |  |  |
| group_2.sequence_ids | [28] | [28] | [28] |  |  |
| group_2.sequence_lengths | [98784] | [98784] | [98784] |  |  |
| group_2.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |  |  |
| group_2.head_padding | q=2.0, kv=2.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |  |  |
| group_2.compute | fwd=2561.14, bwd=5122.29 | step/layer=5.41, fwd/layer=44.78, bwd/layer=89.55 | step/layer=5.41, fwd/layer=44.78, bwd/layer=89.55 |  |  |
| group_2.alltoall_comm | 704.37 | 0.00 | 0.00 |  |  |
| group_2.ring_comm | 0.00 | fwd_step=2.12, bwd_step=4.24, fwd/layer=44.78, bwd/layer=89.55 | fwd_step=2.12, bwd_step=4.24, fwd/layer=44.78, bwd/layer=89.55 |  |  |
| group_2.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 |  |  |
| group_2.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) |  |  |

## Microbatch 1
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| estimated_time_ms | 801.22 | 524.28 | 524.28 | 1183.41 | 524.28 |
| seq_sum | 167552 | 167552 | 167552 | 167552 | 167552 |
| num_sequences | 61 | 61 | 61 | 61 | 61 |
| sequences | [0:1280, 1:6464, 2:5600, 3:768, 4:320, 5:1664, 6:736, 7:512, 8:7232, 9:4960, 10:8320, 11:992, 12:4224, 13:1280, 14:960, 15:1824, 16:1760, 18:1920, 19:192, 20:64, 21:32, 22:1472, 23:896, 24:384, ...(+37)] | [0:1280, 1:6464, 2:5600, 3:768, 4:320, 5:1664, 6:736, 7:512, 8:7232, 9:4960, 10:8320, 11:992, 12:4224, 13:1280, 14:960, 15:1824, 16:1760, 18:1920, 19:192, 20:64, 21:32, 22:1472, 23:896, 24:384, ...(+37)] | [0:1280, 1:6464, 2:5600, 3:768, 4:320, 5:1664, 6:736, 7:512, 8:7232, 9:4960, 10:8320, 11:992, 12:4224, 13:1280, 14:960, 15:1824, 16:1760, 18:1920, 19:192, 20:64, 21:32, 22:1472, 23:896, 24:384, ...(+37)] | [0:1280, 1:6464, 2:5600, 3:768, 4:320, 5:1664, 6:736, 7:512, 8:7232, 9:4960, 10:8320, 11:992, 12:4224, 13:1280, 14:960, 15:1824, 16:1760, 18:1920, 19:192, 20:64, 21:32, 22:1472, 23:896, 24:384, ...(+37)] | [0:1280, 1:6464, 2:5600, 3:768, 4:320, 5:1664, 6:736, 7:512, 8:7232, 9:4960, 10:8320, 11:992, 12:4224, 13:1280, 14:960, 15:1824, 16:1760, 18:1920, 19:192, 20:64, 21:32, 22:1472, 23:896, 24:384, ...(+37)] |

| group metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| group_0.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_0.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_0.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_0.estimated_time_ms | 146.40 | 146.42 | 454.71 | 280.44 | 266.58 |
| group_0.estimated_memory_gb | 66.65 | 69.00 | 71.35 | 71.97 | 68.50 |
| group_0.memory_breakdown_gb | model=7.00, act=59.65, pad=0.00 | model=7.00, act=62.00, pad=0.00 | model=7.00, act=64.35, pad=0.00 | model=7.00, act=64.97, pad=0.00 | model=7.00, act=61.50, pad=0.00 |
| group_0.seq_sum_local | seq_sum=15424, local=15424.00 | seq_sum=16032, local=16032.00 | seq_sum=16640, local=16640.00 | seq_sum=16800, local=16800.00 | seq_sum=15904, local=15904.00 |
| group_0.sequence_ids | [5, 60, 22, 51, 59, 14, 23, 41, 6, 63, 49, 26, 48, 57, 4, 61, 19, 25, 62, 34, 35, 20, 31, 43, ...(+2)] | [13, 50, 37, 59, 11, 14, 23, 41, 3, 6, 63, 49, 7, 26, 48, 57, 24, 45, 4, 61, 19, 25, 62, 38, ...(+7)] | [56, 47, 18, 37, 20] | [8, 36, 22, 29, 41, 6, 63, 49, 7, 57, 24, 34, 31, 21] | [8, 22, 29, 23, 3, 6, 49, 7, 57, 24, 45, 4, 61, 19, 62, 38] |
| group_0.sequence_lengths | [1664, 1664, 1472, 1472, 1152, 960, 896, 896, 736, 736, 640, 512, 512, 480, 320, 256, 192, 192, 192, 96, 96, 64, 64, 64, ...(+2)] | [1280, 1280, 1184, 1152, 992, 960, 896, 896, 768, 736, 736, 640, 512, 512, 512, 480, 384, 352, 320, 256, 192, 192, 192, 128, ...(+7)] | [10528, 2944, 1920, 1184, 64] | [7232, 2176, 1472, 1344, 896, 736, 736, 640, 512, 480, 384, 96, 64, 32] | [7232, 1472, 1344, 896, 768, 736, 640, 512, 480, 384, 352, 320, 256, 192, 192, 128] |
| group_0.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_0.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_0.compute | fwd=48.80, bwd=97.60 | fwd=48.81, bwd=97.62 | fwd=151.57, bwd=303.14 | fwd=93.48, bwd=186.96 | fwd=88.86, bwd=177.72 |
| group_0.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_0.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_0.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_0.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_1.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_1.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_1.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_1.estimated_time_ms | 390.51 | 202.07 | 425.89 | 230.87 | 146.16 |
| group_1.estimated_memory_gb | 68.88 | 60.71 | 70.11 | 71.97 | 61.45 |
| group_1.memory_breakdown_gb | model=7.00, act=61.88, pad=0.00 | model=7.00, act=53.71, pad=0.00 | model=7.00, act=63.11, pad=0.00 | model=7.00, act=64.97, pad=0.00 | model=7.00, act=54.45, pad=0.00 |
| group_1.seq_sum_local | seq_sum=16000, local=16000.00 | seq_sum=13888, local=13888.00 | seq_sum=16320, local=16320.00 | seq_sum=16800, local=16800.00 | seq_sum=14080, local=14080.00 |
| group_1.sequence_ids | [40, 52, 44, 16, 38] | [2, 53, 39, 22, 51, 29] | [33, 6, 63, 49, 26, 48, 57, 24, 45, 4, 61, 19, 25, 62, 38, 34, 35, 31, 43, 54, 21] | [2, 44, 27, 5, 60, 39, 11, 26, 48, 62, 35, 20, 43] | [53, 36, 18, 27, 5, 39, 13, 59] |
| group_1.sequence_lengths | [9536, 2560, 2016, 1760, 128] | [5600, 2464, 1536, 1472, 1472, 1344] | [10272, 736, 736, 640, 512, 512, 480, 384, 352, 320, 256, 192, 192, 192, 128, 96, 96, 64, 64, 64, 32] | [5600, 2016, 1888, 1664, 1664, 1536, 992, 512, 512, 192, 96, 64, 64] | [2464, 2176, 1920, 1888, 1664, 1536, 1280, 1152] |
| group_1.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_1.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_1.compute | fwd=130.17, bwd=260.34 | fwd=67.36, bwd=134.72 | fwd=141.96, bwd=283.92 | fwd=76.96, bwd=153.91 | fwd=48.72, bwd=97.44 |
| group_1.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_1.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_1.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_1.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_2.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_2.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_2.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_2.estimated_time_ms | 22.91 | 174.95 | 147.42 | 264.03 | 397.50 |
| group_2.estimated_memory_gb | 16.03 | 68.88 | 69.74 | 71.97 | 71.47 |
| group_2.memory_breakdown_gb | model=7.00, act=9.03, pad=0.00 | model=7.00, act=61.88, pad=0.00 | model=7.00, act=62.74, pad=0.00 | model=7.00, act=64.97, pad=0.00 | model=7.00, act=64.47, pad=0.00 |
| group_2.seq_sum_local | seq_sum=2336, local=2336.00 | seq_sum=16000, local=16000.00 | seq_sum=16224, local=16224.00 | seq_sum=16800, local=16800.00 | seq_sum=16672, local=16672.00 |
| group_2.sequence_ids | [15, 7] | [30, 52, 18, 27, 15, 16, 5, 60] | [5, 60, 39, 22, 51, 29, 13, 50, 11, 14, 23, 41, 3] | [1, 47, 15, 16, 13, 23, 3, 45, 61, 25, 54] | [10, 2, 52, 25] |
| group_2.sequence_lengths | [1824, 512] | [2720, 2560, 1920, 1888, 1824, 1760, 1664, 1664] | [1664, 1664, 1536, 1472, 1472, 1344, 1280, 1280, 992, 960, 896, 896, 768] | [6464, 2944, 1824, 1760, 1280, 896, 768, 352, 256, 192, 64] | [8320, 5600, 2560, 192] |
| group_2.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_2.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_2.compute | fwd=7.64, bwd=15.28 | fwd=58.32, bwd=116.63 | fwd=49.14, bwd=98.28 | fwd=88.01, bwd=176.02 | fwd=132.50, bwd=265.00 |
| group_2.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_2.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_2.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_2.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_3.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_3.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_3.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_3.estimated_time_ms | 425.90 | 321.97 | 285.07 | 424.78 | 308.08 |
| group_3.estimated_memory_gb | 64.91 | 67.76 | 71.35 | 71.84 | 71.35 |
| group_3.memory_breakdown_gb | model=7.00, act=57.91, pad=0.00 | model=7.00, act=60.76, pad=0.00 | model=7.00, act=64.35, pad=0.00 | model=7.00, act=64.84, pad=0.00 | model=7.00, act=64.35, pad=0.00 |
| group_3.seq_sum_local | seq_sum=14976, local=14976.00 | seq_sum=15712, local=15712.00 | seq_sum=16640, local=16640.00 | seq_sum=16768, local=16768.00 | seq_sum=16640, local=16640.00 |
| group_3.sequence_ids | [56, 27, 0, 50] | [8, 55, 36, 0] | [55, 9, 12, 0, 59] | [10, 32, 51, 4] | [32, 55, 15, 60, 51] |
| group_3.sequence_lengths | [10528, 1888, 1280, 1280] | [7232, 5024, 2176, 1280] | [5024, 4960, 4224, 1280, 1152] | [8320, 6656, 1472, 320] | [6656, 5024, 1824, 1664, 1472] |
| group_3.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_3.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_3.compute | fwd=141.97, bwd=283.93 | fwd=107.32, bwd=214.64 | fwd=95.02, bwd=190.05 | fwd=141.59, bwd=283.19 | fwd=102.69, bwd=205.39 |
| group_3.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_3.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_3.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_3.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_4.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_4.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_4.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_4.estimated_time_ms | 320.50 | 339.40 | 58.15 | 393.16 | 443.81 |
| group_4.estimated_memory_gb | 70.11 | 63.31 | 28.66 | 69.12 | 70.61 |
| group_4.memory_breakdown_gb | model=7.00, act=63.11, pad=0.00 | model=7.00, act=56.31, pad=0.00 | model=7.00, act=21.66, pad=0.00 | model=7.00, act=62.12, pad=0.00 | model=7.00, act=63.61, pad=0.00 |
| group_4.seq_sum_local | seq_sum=16320, local=16320.00 | seq_sum=14560, local=14560.00 | seq_sum=5600, local=5600.00 | seq_sum=16064, local=16064.00 | seq_sum=16448, local=16448.00 |
| group_4.sequence_ids | [8, 12, 47, 18] | [10, 12, 44] | [44, 15, 16] | [40, 52, 53, 37, 19, 38] | [33, 47, 30, 26] |
| group_4.sequence_lengths | [7232, 4224, 2944, 1920] | [8320, 4224, 2016] | [2016, 1824, 1760] | [9536, 2560, 2464, 1184, 192, 128] | [10272, 2944, 2720, 512] |
| group_4.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_4.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_4.compute | fwd=106.83, bwd=213.67 | fwd=113.13, bwd=226.26 | fwd=19.38, bwd=38.77 | fwd=131.05, bwd=262.11 | fwd=147.94, bwd=295.87 |
| group_4.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_4.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_4.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_4.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_5.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_5.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_5.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_5.estimated_time_ms | 472.80 | 480.01 | 189.20 | 192.55 | 433.04 |
| group_5.estimated_memory_gb | 71.10 | 69.62 | 34.97 | 62.44 | 70.24 |
| group_5.memory_breakdown_gb | model=7.00, act=64.10, pad=0.00 | model=7.00, act=62.62, pad=0.00 | model=7.00, act=27.97, pad=0.00 | model=7.00, act=55.44, pad=0.00 | model=7.00, act=63.24, pad=0.00 |
| group_5.seq_sum_local | seq_sum=16576, local=16576.00 | seq_sum=16192, local=16192.00 | seq_sum=7232, local=7232.00 | seq_sum=14336, local=14336.00 | seq_sum=16352, local=16352.00 |
| group_5.sequence_ids | [33, 9, 29] | [40, 32] | [8] | [55, 30, 18, 0, 50, 59, 14] | [56, 0, 37, 11, 14, 41, 48] |
| group_5.sequence_lengths | [10272, 4960, 1344] | [9536, 6656] | [7232] | [5024, 2720, 1920, 1280, 1280, 1152, 960] | [10528, 1280, 1184, 992, 960, 896, 512] |
| group_5.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_5.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_5.compute | fwd=157.60, bwd=315.20 | fwd=160.00, bwd=320.00 | fwd=63.07, bwd=126.13 | fwd=64.18, bwd=128.37 | fwd=144.35, bwd=288.69 |
| group_5.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_5.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_5.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_5.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_6.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ring×2 |
| group_6.num_gpus | 1 | 1 | 1 | 1 | 2 |
| group_6.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first |
| group_6.estimated_time_ms | 64.04 | 478.09 | 119.74 | 437.74 | 416.33 |
| group_6.estimated_memory_gb | 27.05 | 66.89 | 45.36 | 63.06 | 63.12 |
| group_6.memory_breakdown_gb | model=7.00, act=20.05, pad=0.00 | model=7.00, act=59.89, pad=0.00 | model=7.00, act=38.36, pad=0.00 | model=7.00, act=56.06, pad=0.00 | model=7.00, act=56.12, pad=0.00 |
| group_6.seq_sum_local | seq_sum=5184, local=5184.00 | seq_sum=15488, local=15488.00 | seq_sum=9920, local=9920.00 | seq_sum=14496, local=14496.00 | seq_sum=29024, local=14512.00 |
| group_6.sequence_ids | [30, 53] | [56, 9] | [30, 52, 53, 36] | [33, 12] | [40, 1, 9, 12, 44, 16, 31] |
| group_6.sequence_lengths | [2720, 2464] | [10528, 4960] | [2720, 2560, 2464, 2176] | [10272, 4224] | [9536, 6464, 4960, 4224, 2016, 1760, 64] |
| group_6.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive |
| group_6.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_6.compute | fwd=21.35, bwd=42.70 | fwd=159.36, bwd=318.73 | fwd=39.91, bwd=79.83 | fwd=145.91, bwd=291.82 | step/layer=2.13, fwd/layer=4.34, bwd/layer=8.67 |
| group_6.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_6.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | fwd_step=0.71, bwd_step=1.42, fwd/layer=4.34, bwd/layer=8.67 |
| group_6.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | overlap, leakage=0.1 |
| group_6.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_7.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ring×2 |
| group_7.num_gpus | 1 | 1 | 1 | 1 | 2 |
| group_7.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first |
| group_7.estimated_time_ms | 220.70 | 192.91 | 435.90 | 478.09 | 511.28 |
| group_7.estimated_memory_gb | 59.10 | 43.38 | 65.53 | 66.89 | 39.67 |
| group_7.memory_breakdown_gb | model=7.00, act=52.10, pad=0.00 | model=7.00, act=36.38, pad=0.00 | model=7.00, act=58.53, pad=0.00 | model=7.00, act=59.89, pad=0.00 | model=7.00, act=32.67, pad=0.00 |
| group_7.seq_sum_local | seq_sum=13472, local=13472.00 | seq_sum=9408, local=9408.00 | seq_sum=15136, local=15136.00 | seq_sum=15488, local=15488.00 | seq_sum=16896, local=8448.00 |
| group_7.sequence_ids | [1, 36, 39, 37, 11, 3, 45] | [1, 47] | [40, 2] | [56, 9] | [46, 21] |
| group_7.sequence_lengths | [6464, 2176, 1536, 1184, 992, 768, 352] | [6464, 2944] | [9536, 5600] | [10528, 4960] | [16864, 32] |
| group_7.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive |
| group_7.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_7.compute | fwd=73.57, bwd=147.14 | fwd=64.30, bwd=128.61 | fwd=145.30, bwd=290.60 | fwd=159.36, bwd=318.73 | step/layer=2.64, fwd/layer=5.33, bwd/layer=10.65 |
| group_7.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_7.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | fwd_step=0.45, bwd_step=0.91, fwd/layer=5.33, bwd/layer=10.65 |
| group_7.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | overlap, leakage=0.1 |
| group_7.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_8.strategy | ulysses×1 | ring×2 | ulysses×1 | usp(sp2×cp2,cf) | ring×2 |
| group_8.num_gpus | 1 | 2 | 1 | 4 | 2 |
| group_8.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first | sp=1, cp=1, placement=head_first | sp=2, cp=2, placement=context_first | sp=1, cp=2, placement=context_first |
| group_8.estimated_time_ms | 226.53 | 204.85 | 177.56 | 1183.41 | 33.36 |
| group_8.estimated_memory_gb | 53.03 | 26.86 | 41.28 | 29.37 | 11.64 |
| group_8.memory_breakdown_gb | model=7.00, act=46.03, pad=0.00 | model=7.00, act=19.86, pad=0.00 | model=7.00, act=34.28, pad=0.00 | model=7.00, act=22.37, pad=0.00 | model=7.00, act=4.64, pad=0.00 |
| group_8.seq_sum_local | seq_sum=11904, local=11904.00 | seq_sum=10272, local=5136.00 | seq_sum=8864, local=8864.00 | seq_sum=23136, local=5784.00 | seq_sum=2400, local=1200.00 |
| group_8.sequence_ids | [2, 55, 13] | [33] | [1, 27, 7] | [42] | [50, 63, 34, 35, 20, 43, 54] |
| group_8.sequence_lengths | [5600, 5024, 1280] | [10272] | [6464, 1888, 512] | [23136] | [1280, 736, 96, 96, 64, 64, 64] |
| group_8.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |
| group_8.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_8.compute | fwd=75.51, bwd=151.02 | step/layer=1.05, fwd/layer=2.13, bwd/layer=4.27 | fwd=59.19, bwd=118.38 | step/layer=2.39 | step/layer=0.17, fwd/layer=0.35, bwd/layer=0.70 |
| group_8.alltoall_comm | 0.00 | 0.00 | 0.00 | qo_op=4.66, kv_op=0.98, fwd/layer=11.27, bwd/layer=11.27 | 0.00 |
| group_8.ring_comm | 0.00 | fwd_step=0.31, bwd_step=0.63, fwd/layer=2.13, bwd/layer=4.27 | 0.00 | fwd_step=0.34, bwd_step=0.68, fwd/layer=4.81, bwd/layer=9.62 | fwd_step=0.15, bwd_step=0.29, fwd/layer=0.35, bwd/layer=0.70 |
| group_8.time_model | additive, leakage=None | overlap, leakage=0.1 | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 |
| group_8.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_9.strategy | ulysses×1 | ring×2 | ulysses×1 | usp(sp2×cp2,cf) | ring×4 |
| group_9.num_gpus | 1 | 2 | 1 | 4 | 4 |
| group_9.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first | sp=1, cp=1, placement=head_first | sp=2, cp=2, placement=context_first | sp=1, cp=4, placement=context_first |
| group_9.estimated_time_ms | 408.02 | 508.44 | 408.02 | 805.94 | 524.28 |
| group_9.estimated_memory_gb | 64.91 | 39.61 | 64.91 | 23.30 | 29.37 |
| group_9.memory_breakdown_gb | model=7.00, act=57.91, pad=0.00 | model=7.00, act=32.61, pad=0.00 | model=7.00, act=57.91, pad=0.00 | model=7.00, act=16.30, pad=0.00 | model=7.00, act=22.37, pad=0.00 |
| group_9.seq_sum_local | seq_sum=14976, local=14976.00 | seq_sum=16864, local=8432.00 | seq_sum=14976, local=14976.00 | seq_sum=16864, local=4216.00 | seq_sum=23136, local=5784.00 |
| group_9.sequence_ids | [10, 32] | [46] | [10, 32] | [46] | [42] |
| group_9.sequence_lengths | [8320, 6656] | [16864] | [8320, 6656] | [16864] | [23136] |
| group_9.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |
| group_9.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_9.compute | fwd=136.01, bwd=272.01 | step/layer=2.63, fwd/layer=5.30, bwd/layer=10.59 | fwd=136.01, bwd=272.01 | step/layer=1.31 | step/layer=1.31, fwd/layer=5.46, bwd/layer=10.92 |
| group_9.alltoall_comm | 0.00 | 0.00 | 0.00 | qo_op=3.49, kv_op=0.81, fwd/layer=8.61, bwd/layer=8.61 | 0.00 |
| group_9.ring_comm | 0.00 | fwd_step=0.45, bwd_step=0.90, fwd/layer=5.30, bwd/layer=10.59 | 0.00 | fwd_step=0.27, bwd_step=0.55, fwd/layer=2.65, bwd/layer=5.31 | fwd_step=0.80, bwd_step=1.60, fwd/layer=5.46, bwd/layer=10.92 |
| group_9.time_model | additive, leakage=None | overlap, leakage=0.1 | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 |
| group_9.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_10.strategy | ulysses×2 | ring×4 | ring×2 |  |  |
| group_10.num_gpus | 2 | 4 | 2 |  |  |
| group_10.sp_cp_placement | sp=2, cp=1, placement=head_first | sp=1, cp=4, placement=context_first | sp=1, cp=2, placement=context_first |  |  |
| group_10.estimated_time_ms | 630.04 | 524.28 | 508.44 |  |  |
| group_10.estimated_memory_gb | 40.35 | 29.37 | 39.61 |  |  |
| group_10.memory_breakdown_gb | model=7.00, act=33.35, pad=0.00 | model=7.00, act=22.37, pad=0.00 | model=7.00, act=32.61, pad=0.00 |  |  |
| group_10.seq_sum_local | seq_sum=17248, local=8624.00 | seq_sum=23136, local=5784.00 | seq_sum=16864, local=8432.00 |  |  |
| group_10.sequence_ids | [46, 24] | [42] | [46] |  |  |
| group_10.sequence_lengths | [16864, 384] | [23136] | [16864] |  |  |
| group_10.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |  |  |
| group_10.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |  |  |
| group_10.compute | fwd=158.31, bwd=316.62 | step/layer=1.31, fwd/layer=5.46, bwd/layer=10.92 | step/layer=2.63, fwd/layer=5.30, bwd/layer=10.59 |  |  |
| group_10.alltoall_comm | 155.11 | 0.00 | 0.00 |  |  |
| group_10.ring_comm | 0.00 | fwd_step=0.80, bwd_step=1.60, fwd/layer=5.46, bwd/layer=10.92 | fwd_step=0.45, bwd_step=0.90, fwd/layer=5.30, bwd/layer=10.59 |  |  |
| group_10.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 |  |  |
| group_10.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) |  |  |
| group_11.strategy | ulysses×4 |  | ring×4 |  |  |
| group_11.num_gpus | 4 |  | 4 |  |  |
| group_11.sp_cp_placement | sp=4, cp=1, placement=head_first |  | sp=1, cp=4, placement=context_first |  |  |
| group_11.estimated_time_ms | 801.22 |  | 524.28 |  |  |
| group_11.estimated_memory_gb | 29.37 |  | 29.37 |  |  |
| group_11.memory_breakdown_gb | model=7.00, act=22.37, pad=0.00 |  | model=7.00, act=22.37, pad=0.00 |  |  |
| group_11.seq_sum_local | seq_sum=23136, local=5784.00 |  | seq_sum=23136, local=5784.00 |  |  |
| group_11.sequence_ids | [42] |  | [42] |  |  |
| group_11.sequence_lengths | [23136] |  | [23136] |  |  |
| group_11.topology | a2a=consecutive, ring=strided |  | a2a=strided, ring=consecutive |  |  |
| group_11.head_padding | q=1.0, kv=1.0 |  | q=1.0, kv=1.0 |  |  |
| group_11.compute | fwd=145.85, bwd=291.69 |  | step/layer=1.31, fwd/layer=5.46, bwd/layer=10.92 |  |  |
| group_11.alltoall_comm | 363.69 |  | 0.00 |  |  |
| group_11.ring_comm | 0.00 |  | fwd_step=0.80, bwd_step=1.60, fwd/layer=5.46, bwd/layer=10.92 |  |  |
| group_11.time_model | additive, leakage=None |  | overlap, leakage=0.1 |  |  |
| group_11.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |  | sum_layers(fwd/bwd leaky overlap across ring steps) |  |  |
