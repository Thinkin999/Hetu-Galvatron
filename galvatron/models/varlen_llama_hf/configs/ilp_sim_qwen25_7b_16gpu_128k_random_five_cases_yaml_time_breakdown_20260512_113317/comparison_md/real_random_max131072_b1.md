# ILP Horizontal Comparison: `real_random_max131072_b1`

## Batch
| field | value |
| --- | --- |
| sampling_mode | real_random |
| max_seq | 131072 |
| global_batch_size | 64 |
| seq_sum | 317728 |
| max / p95 / p99 | 59328 / 23758.40 / 57594.24 |
| >=128k / >=256k / >=384k / >=512k | 0 / 0 / 0 / 0 |
| sequences | [0:864, 1:37952, 2:1440, 3:59328, 4:2144, 5:32, 6:32, 7:64, 8:1536, 9:512, 10:768, 11:832, 12:32, 13:608, 14:9664, 15:1792, 16:8000, 17:256, 18:1632, 19:18944, 20:3200, 21:64, 22:192, 23:2528, 24:64, 25:6240, 26:448, 27:11200, 28:5952, 29:960, 30:2272, 31:128, 32:14208, 33:64, 34:2240, 35:320, 36:2656, 37:1696, 38:96, 39:160, 40:288, 41:2048, 42:56576, 43:64, 44:2112, 45:3424, 46:832, 47:768, 48:384, 49:96, 50:160, 51:384, 52:24608, 53:64, 54:512, 55:3872, 56:7136, 57:1760, 58:6240, 59:2112, 60:704, 61:512, 62:352, 63:1600] |

## Case Summary
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| status | ok | ok | ok | ok | ok |
| total_time_ms | 4158.6582 | 3317.4056 | 3317.4056 | 4775.2202 | 3317.4056 |
| speedup_vs_ulysses | 1.0000 | 1.2536 | 1.2536 | 0.8709 | 1.2536 |
| microbatch_count | 2 | 2 | 2 | 2 | 2 |
| solver_wall_s | 4.4055 | 1.8757 | 5.8126 | 11.7087 | 38.2289 |

## Microbatch 0
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| estimated_time_ms | 3295.75 | 2732.54 | 2732.54 | 3494.35 | 2732.54 |
| seq_sum | 153856 | 153856 | 153856 | 153856 | 153856 |
| num_sequences | 3 | 3 | 3 | 3 | 3 |
| sequences | [1:37952, 3:59328, 42:56576] | [1:37952, 3:59328, 42:56576] | [1:37952, 3:59328, 42:56576] | [1:37952, 3:59328, 42:56576] | [1:37952, 3:59328, 42:56576] |

| group metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| group_0.strategy | ulysses×4 | ring×4 | ring×4 | usp(sp2×cp4,cf) | ring×4 |
| group_0.num_gpus | 4 | 4 | 4 | 8 | 4 |
| group_0.sp_cp_placement | sp=4, cp=1, placement=head_first | sp=1, cp=4, placement=context_first | sp=1, cp=4, placement=context_first | sp=2, cp=4, placement=context_first | sp=1, cp=4, placement=context_first |
| group_0.estimated_time_ms | 3034.45 | 1287.94 | 2732.54 | 2401.68 | 2732.54 |
| group_0.estimated_memory_gb | 61.70 | 43.69 | 61.70 | 35.68 | 61.70 |
| group_0.memory_breakdown_gb | model=7.00, act=54.70, pad=0.00 | model=7.00, act=36.69, pad=0.00 | model=7.00, act=54.70, pad=0.00 | model=7.00, act=28.68, pad=0.00 | model=7.00, act=54.70, pad=0.00 |
| group_0.seq_sum_local | seq_sum=56576, local=14144.00 | seq_sum=37952, local=9488.00 | seq_sum=56576, local=14144.00 | seq_sum=59328, local=7416.00 | seq_sum=56576, local=14144.00 |
| group_0.sequence_ids | [42] | [1] | [42] | [3] | [42] |
| group_0.sequence_lengths | [56576] | [37952] | [56576] | [59328] | [56576] |
| group_0.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |
| group_0.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_0.compute | fwd=847.39, bwd=1694.77 | step/layer=3.28, fwd/layer=13.42, bwd/layer=26.83 | step/layer=7.02, fwd/layer=28.46, bwd/layer=56.93 | step/layer=3.85 | step/layer=7.02, fwd/layer=28.46, bwd/layer=56.93 |
| group_0.alltoall_comm | 492.29 | 0.00 | 0.00 | qo_op=5.87, kv_op=1.15, fwd/layer=14.04, bwd/layer=14.04 | 0.00 |
| group_0.ring_comm | 0.00 | fwd_step=1.00, bwd_step=2.00, fwd/layer=13.42, bwd/layer=26.83 | fwd_step=1.25, bwd_step=2.50, fwd/layer=28.46, bwd/layer=56.93 | fwd_step=0.89, bwd_step=1.78, fwd/layer=15.66, bwd/layer=31.31 | fwd_step=1.25, bwd_step=2.50, fwd/layer=28.46, bwd/layer=56.93 |
| group_0.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 | overlap, leakage=0.1 | overlap, leakage=0.1 |
| group_0.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_1.strategy | ulysses×4 | ring×4 | ring×4 | usp(sp2×cp4,cf) | usp(sp2×cp2,cf) |
| group_1.num_gpus | 4 | 4 | 4 | 8 | 4 |
| group_1.sp_cp_placement | sp=4, cp=1, placement=head_first | sp=1, cp=4, placement=context_first | sp=1, cp=4, placement=context_first | sp=2, cp=4, placement=context_first | sp=2, cp=2, placement=context_first |
| group_1.estimated_time_ms | 3295.75 | 2732.54 | 1287.94 | 3386.70 | 2318.22 |
| group_1.estimated_memory_gb | 64.36 | 61.70 | 43.69 | 52.69 | 43.69 |
| group_1.memory_breakdown_gb | model=7.00, act=57.36, pad=0.00 | model=7.00, act=54.70, pad=0.00 | model=7.00, act=36.69, pad=0.00 | model=7.00, act=45.69, pad=0.00 | model=7.00, act=36.69, pad=0.00 |
| group_1.seq_sum_local | seq_sum=59328, local=14832.00 | seq_sum=56576, local=14144.00 | seq_sum=37952, local=9488.00 | seq_sum=94528, local=11816.00 | seq_sum=37952, local=9488.00 |
| group_1.sequence_ids | [3] | [42] | [1] | [42, 1] | [1] |
| group_1.sequence_lengths | [59328] | [56576] | [37952] | [56576, 37952] | [37952] |
| group_1.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |
| group_1.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_1.compute | fwd=930.96, bwd=1861.92 | step/layer=7.02, fwd/layer=28.46, bwd/layer=56.93 | step/layer=3.28, fwd/layer=13.42, bwd/layer=26.83 | step/layer=5.15 | step/layer=6.20 |
| group_1.alltoall_comm | 502.88 | 0.00 | 0.00 | qo_op=9.14, kv_op=1.62, fwd/layer=21.51, bwd/layer=21.51 | qo_op=7.41, kv_op=1.37, fwd/layer=17.56, bwd/layer=17.56 |
| group_1.ring_comm | 0.00 | fwd_step=1.25, bwd_step=2.50, fwd/layer=28.46, bwd/layer=56.93 | fwd_step=1.00, bwd_step=2.00, fwd/layer=13.42, bwd/layer=26.83 | fwd_step=1.13, bwd_step=2.25, fwd/layer=20.94, bwd/layer=41.88 | fwd_step=0.50, bwd_step=0.99, fwd/layer=12.44, bwd/layer=24.89 |
| group_1.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 | overlap, leakage=0.1 | overlap, leakage=0.1 |
| group_1.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) |
| group_2.strategy | ulysses×8 | ring×8 | ring×8 |  | usp(sp4×cp2,cf) |
| group_2.num_gpus | 8 | 8 | 8 |  | 8 |
| group_2.sp_cp_placement | sp=8, cp=1, placement=head_first | sp=1, cp=8, placement=context_first | sp=1, cp=8, placement=context_first |  | sp=4, cp=2, placement=context_first |
| group_2.estimated_time_ms | 1617.00 | 1702.84 | 1702.84 |  | 2689.91 |
| group_2.estimated_memory_gb | 25.39 | 35.68 | 35.68 |  | 35.68 |
| group_2.memory_breakdown_gb | model=7.00, act=18.39, pad=0.04 | model=7.00, act=28.68, pad=0.00 | model=7.00, act=28.68, pad=0.00 |  | model=7.00, act=28.68, pad=0.00 |
| group_2.seq_sum_local | seq_sum=37952, local=4744.00 | seq_sum=59328, local=7416.00 | seq_sum=59328, local=7416.00 |  | seq_sum=59328, local=7416.00 |
| group_2.sequence_ids | [1] | [3] | [3] |  | [3] |
| group_2.sequence_lengths | [37952] | [59328] | [59328] |  | [59328] |
| group_2.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |  | a2a=strided, ring=consecutive |
| group_2.head_padding | q=2.0, kv=2.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |  | q=1.0, kv=1.0 |
| group_2.compute | fwd=385.09, bwd=770.19 | step/layer=2.07, fwd/layer=17.74, bwd/layer=35.48 | step/layer=2.07, fwd/layer=17.74, bwd/layer=35.48 |  | step/layer=7.41 |
| group_2.alltoall_comm | 461.72 | 0.00 | 0.00 |  | qo_op=7.81, kv_op=2.05, fwd/layer=19.73, bwd/layer=19.73 |
| group_2.ring_comm | 0.00 | fwd_step=1.74, bwd_step=3.48, fwd/layer=17.74, bwd/layer=35.48 | fwd_step=1.74, bwd_step=3.48, fwd/layer=17.74, bwd/layer=35.48 |  | fwd_step=0.41, bwd_step=0.82, fwd/layer=14.87, bwd/layer=29.73 |
| group_2.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 |  | overlap, leakage=0.1 |
| group_2.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) |  | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) |

## Microbatch 1
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| estimated_time_ms | 862.91 | 584.87 | 584.87 | 1280.87 | 584.87 |
| seq_sum | 163872 | 163872 | 163872 | 163872 | 163872 |
| num_sequences | 61 | 61 | 61 | 61 | 61 |
| sequences | [0:864, 2:1440, 4:2144, 5:32, 6:32, 7:64, 8:1536, 9:512, 10:768, 11:832, 12:32, 13:608, 14:9664, 15:1792, 16:8000, 17:256, 18:1632, 19:18944, 20:3200, 21:64, 22:192, 23:2528, 24:64, 25:6240, ...(+37)] | [0:864, 2:1440, 4:2144, 5:32, 6:32, 7:64, 8:1536, 9:512, 10:768, 11:832, 12:32, 13:608, 14:9664, 15:1792, 16:8000, 17:256, 18:1632, 19:18944, 20:3200, 21:64, 22:192, 23:2528, 24:64, 25:6240, ...(+37)] | [0:864, 2:1440, 4:2144, 5:32, 6:32, 7:64, 8:1536, 9:512, 10:768, 11:832, 12:32, 13:608, 14:9664, 15:1792, 16:8000, 17:256, 18:1632, 19:18944, 20:3200, 21:64, 22:192, 23:2528, 24:64, 25:6240, ...(+37)] | [0:864, 2:1440, 4:2144, 5:32, 6:32, 7:64, 8:1536, 9:512, 10:768, 11:832, 12:32, 13:608, 14:9664, 15:1792, 16:8000, 17:256, 18:1632, 19:18944, 20:3200, 21:64, 22:192, 23:2528, 24:64, 25:6240, ...(+37)] | [0:864, 2:1440, 4:2144, 5:32, 6:32, 7:64, 8:1536, 9:512, 10:768, 11:832, 12:32, 13:608, 14:9664, 15:1792, 16:8000, 17:256, 18:1632, 19:18944, 20:3200, 21:64, 22:192, 23:2528, 24:64, 25:6240, ...(+37)] |

| group metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| group_0.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_0.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_0.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_0.estimated_time_ms | 177.70 | 346.25 | 271.61 | 265.84 | 241.69 |
| group_0.estimated_memory_gb | 66.15 | 70.98 | 70.11 | 71.97 | 69.37 |
| group_0.memory_breakdown_gb | model=7.00, act=59.15, pad=0.00 | model=7.00, act=63.98, pad=0.00 | model=7.00, act=63.11, pad=0.00 | model=7.00, act=64.97, pad=0.00 | model=7.00, act=62.37, pad=0.00 |
| group_0.seq_sum_local | seq_sum=15296, local=15296.00 | seq_sum=16544, local=16544.00 | seq_sum=16320, local=16320.00 | seq_sum=16800, local=16800.00 | seq_sum=16128, local=16128.00 |
| group_0.sequence_ids | [55, 59, 2, 11, 46, 10, 47, 60, 9, 54, 61, 26, 48, 62, 17, 22, 31, 38, 49, 7, 21, 24, 33, 43, ...(+4)] | [56, 28, 10, 60, 13, 9, 26, 51, 12] | [28, 55, 20, 41, 0, 51] | [58, 23, 30, 59, 46, 13, 9, 61, 51, 50, 31, 38, 7, 21, 24, 33, 43, 5, 6, 12] | [25, 36, 0, 10, 47, 60, 13, 9, 54, 48, 51, 62, 35, 40, 17, 22, 50, 31, 5] |
| group_0.sequence_lengths | [3872, 2112, 1440, 832, 832, 768, 768, 704, 512, 512, 512, 448, 384, 352, 256, 192, 128, 96, 96, 64, 64, 64, 64, 64, ...(+4)] | [7136, 5952, 768, 704, 608, 512, 448, 384, 32] | [5952, 3872, 3200, 2048, 864, 384] | [6240, 2528, 2272, 2112, 832, 608, 512, 512, 384, 160, 128, 96, 64, 64, 64, 64, 64, 32, 32, 32] | [6240, 2656, 864, 768, 768, 704, 608, 512, 512, 384, 384, 352, 320, 288, 256, 192, 160, 128, 32] |
| group_0.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_0.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_0.compute | fwd=59.23, bwd=118.47 | fwd=115.42, bwd=230.83 | fwd=90.54, bwd=181.07 | fwd=88.61, bwd=177.23 | fwd=80.56, bwd=161.13 |
| group_0.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_0.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_0.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_0.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_1.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_1.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_1.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_1.estimated_time_ms | 156.08 | 476.63 | 259.20 | 484.78 | 482.75 |
| group_1.estimated_memory_gb | 64.67 | 66.77 | 71.10 | 71.97 | 71.72 |
| group_1.memory_breakdown_gb | model=7.00, act=57.67, pad=0.00 | model=7.00, act=59.77, pad=0.00 | model=7.00, act=64.10, pad=0.00 | model=7.00, act=64.97, pad=0.00 | model=7.00, act=64.72, pad=0.00 |
| group_1.seq_sum_local | seq_sum=14912, local=14912.00 | seq_sum=15456, local=15456.00 | seq_sum=16576, local=16576.00 | seq_sum=16800, local=16800.00 | seq_sum=16736, local=16736.00 |
| group_1.sequence_ids | [34, 4, 44, 15, 57, 37, 18, 13, 35, 40, 39, 50] | [27, 4, 37, 62, 53] | [58, 45, 15, 2, 29, 46, 60, 54, 40, 22, 38, 49] | [27, 4, 29, 0, 60, 54, 62, 53] | [27, 63, 8, 2, 29] |
| group_1.sequence_lengths | [2240, 2144, 2112, 1792, 1760, 1696, 1632, 608, 320, 288, 160, 160] | [11200, 2144, 1696, 352, 64] | [6240, 3424, 1792, 1440, 960, 832, 704, 512, 288, 192, 96, 96] | [11200, 2144, 960, 864, 704, 512, 352, 64] | [11200, 1600, 1536, 1440, 960] |
| group_1.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_1.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_1.compute | fwd=52.03, bwd=104.05 | fwd=158.88, bwd=317.75 | fwd=86.40, bwd=172.80 | fwd=161.59, bwd=323.18 | fwd=160.92, bwd=321.83 |
| group_1.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_1.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_1.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_1.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_2.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×2 |
| group_2.num_gpus | 1 | 1 | 1 | 1 | 2 |
| group_2.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=2, cp=1, placement=head_first |
| group_2.estimated_time_ms | 261.91 | 168.09 | 252.04 | 705.22 | 302.90 |
| group_2.estimated_memory_gb | 66.28 | 69.49 | 69.62 | 71.84 | 39.17 |
| group_2.memory_breakdown_gb | model=7.00, act=59.28, pad=0.00 | model=7.00, act=62.49, pad=0.00 | model=7.00, act=62.62, pad=0.00 | model=7.00, act=64.84, pad=0.00 | model=7.00, act=32.17, pad=0.00 |
| group_2.seq_sum_local | seq_sum=15328, local=15328.00 | seq_sum=16160, local=16160.00 | seq_sum=16192, local=16192.00 | seq_sum=16768, local=16768.00 | seq_sum=16640, local=8320.00 |
| group_2.sequence_ids | [58, 45, 20, 63, 0] | [36, 59, 18, 63, 8, 29, 0, 11, 46, 47, 54, 48, 35, 40, 22, 50, 31, 7, 21, 24, 33, 43, 5, 6] | [25, 23, 4, 59, 37, 9, 61, 48, 7] | [32, 37, 48, 35, 39] | [56, 55, 30, 37, 11, 46] |
| group_2.sequence_lengths | [6240, 3424, 3200, 1600, 864] | [2656, 2112, 1632, 1600, 1536, 960, 864, 832, 832, 768, 512, 384, 320, 288, 192, 160, 128, 64, 64, 64, 64, 64, 32, 32] | [6240, 2528, 2144, 2112, 1696, 512, 512, 384, 64] | [14208, 1696, 384, 320, 160] | [7136, 3872, 2272, 1696, 832, 832] |
| group_2.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_2.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_2.compute | fwd=87.30, bwd=174.61 | fwd=56.03, bwd=112.06 | fwd=84.01, bwd=168.03 | fwd=235.07, bwd=470.15 | fwd=50.49, bwd=100.98 |
| group_2.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 151.43 |
| group_2.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_2.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_2.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_3.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ring×2 |
| group_3.num_gpus | 1 | 1 | 1 | 1 | 2 |
| group_3.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first |
| group_3.estimated_time_ms | 310.06 | 265.78 | 132.47 | 197.63 | 515.26 |
| group_3.estimated_memory_gb | 66.65 | 70.85 | 53.78 | 71.60 | 65.47 |
| group_3.memory_breakdown_gb | model=7.00, act=59.65, pad=0.00 | model=7.00, act=63.85, pad=0.00 | model=7.00, act=46.78, pad=0.00 | model=7.00, act=64.60, pad=0.00 | model=7.00, act=58.47, pad=0.00 |
| group_3.seq_sum_local | seq_sum=15424, local=15424.00 | seq_sum=16512, local=16512.00 | seq_sum=12096, local=12096.00 | seq_sum=16704, local=16704.00 | seq_sum=30240, local=15120.00 |
| group_3.sequence_ids | [25, 28, 30, 29] | [58, 20, 30, 41, 15, 61, 17, 38, 49] | [36, 30, 57, 18, 63, 10, 26, 62, 17, 39, 24, 43, 5, 12] | [45, 20, 44, 41, 18, 63, 8, 10, 40, 49] | [32, 34, 4, 44, 59, 41, 15, 57, 18, 24, 43, 53] |
| group_3.sequence_lengths | [6240, 5952, 2272, 960] | [6240, 3200, 2272, 2048, 1792, 512, 256, 96, 96] | [2656, 2272, 1760, 1632, 1600, 768, 448, 352, 256, 160, 64, 64, 32, 32] | [3424, 3200, 2112, 2048, 1632, 1600, 1536, 768, 288, 96] | [14208, 2240, 2144, 2112, 2112, 2048, 1792, 1760, 1632, 64, 64, 64] |
| group_3.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive |
| group_3.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_3.compute | fwd=103.35, bwd=206.71 | fwd=88.59, bwd=177.18 | fwd=44.16, bwd=88.31 | fwd=65.88, bwd=131.76 | step/layer=2.65, fwd/layer=5.37, bwd/layer=10.73 |
| group_3.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_3.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | fwd_step=0.73, bwd_step=1.47, fwd/layer=5.37, bwd/layer=10.73 |
| group_3.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | overlap, leakage=0.1 |
| group_3.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_4.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ring×2 |
| group_4.num_gpus | 1 | 1 | 1 | 1 | 2 |
| group_4.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first |
| group_4.estimated_time_ms | 416.30 | 411.11 | 248.05 | 395.45 | 477.19 |
| group_4.estimated_memory_gb | 67.02 | 67.51 | 57.74 | 70.73 | 69.99 |
| group_4.memory_breakdown_gb | model=7.00, act=60.02, pad=0.00 | model=7.00, act=60.51, pad=0.00 | model=7.00, act=50.74, pad=0.00 | model=7.00, act=63.73, pad=0.00 | model=7.00, act=62.99, pad=0.00 |
| group_4.seq_sum_local | seq_sum=15520, local=15520.00 | seq_sum=15648, local=15648.00 | seq_sum=13120, local=13120.00 | seq_sum=16480, local=16480.00 | seq_sum=32576, local=16288.00 |
| group_4.sequence_ids | [16, 56, 51] | [14, 55, 44] | [56, 34, 8, 11, 13, 35, 50, 31, 21, 53, 6] | [16, 25, 15, 17, 22] | [14, 58, 28, 45, 20, 23, 61, 26, 39, 38, 49, 7, 21, 33, 6, 12] |
| group_4.sequence_lengths | [8000, 7136, 384] | [9664, 3872, 2112] | [7136, 2240, 1536, 832, 608, 320, 160, 128, 64, 64, 32] | [8000, 6240, 1792, 256, 192] | [9664, 6240, 5952, 3424, 3200, 2528, 512, 448, 160, 96, 96, 64, 64, 64, 32, 32] |
| group_4.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive |
| group_4.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_4.compute | fwd=138.77, bwd=277.53 | fwd=137.04, bwd=274.07 | fwd=82.68, bwd=165.36 | fwd=131.82, bwd=263.64 | step/layer=2.45, fwd/layer=4.97, bwd/layer=9.94 |
| group_4.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_4.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | fwd_step=0.78, bwd_step=1.57, fwd/layer=4.97, bwd/layer=9.94 |
| group_4.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | overlap, leakage=0.1 |
| group_4.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_5.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ring×4 |
| group_5.num_gpus | 1 | 1 | 1 | 1 | 4 |
| group_5.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=4, placement=context_first |
| group_5.estimated_time_ms | 404.51 | 227.15 | 462.34 | 308.96 | 584.87 |
| group_5.estimated_memory_gb | 70.36 | 54.77 | 61.70 | 70.24 | 30.79 |
| group_5.memory_breakdown_gb | model=7.00, act=63.36, pad=0.00 | model=7.00, act=47.77, pad=0.00 | model=7.00, act=54.70, pad=0.00 | model=7.00, act=63.24, pad=0.00 | model=7.00, act=23.79, pad=0.00 |
| group_5.seq_sum_local | seq_sum=16384, local=16384.00 | seq_sum=12352, local=12352.00 | seq_sum=14144, local=14144.00 | seq_sum=16352, local=16352.00 | seq_sum=24608, local=6152.00 |
| group_5.sequence_ids | [14, 36, 23, 8] | [25, 45, 23, 39] | [27, 44, 47, 33] | [56, 55, 36, 34, 26] | [52] |
| group_5.sequence_lengths | [9664, 2656, 2528, 1536] | [6240, 3424, 2528, 160] | [11200, 2112, 768, 64] | [7136, 3872, 2656, 2240, 448] | [24608] |
| group_5.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive |
| group_5.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_5.compute | fwd=134.84, bwd=269.68 | fwd=75.72, bwd=151.44 | fwd=154.11, bwd=308.22 | fwd=102.99, bwd=205.97 | step/layer=1.46, fwd/layer=6.09, bwd/layer=12.18 |
| group_5.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_5.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | fwd_step=0.82, bwd_step=1.64, fwd/layer=6.09, bwd/layer=12.18 |
| group_5.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | overlap, leakage=0.1 |
| group_5.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_6.strategy | ulysses×1 | ring×2 | ring×2 | ulysses×1 | ring×4 |
| group_6.num_gpus | 1 | 2 | 2 | 1 | 4 |
| group_6.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first | sp=1, cp=2, placement=context_first | sp=1, cp=1, placement=head_first | sp=1, cp=4, placement=context_first |
| group_6.estimated_time_ms | 453.43 | 536.99 | 553.52 | 176.21 | 461.17 |
| group_6.estimated_memory_gb | 58.23 | 57.68 | 53.16 | 48.58 | 33.05 |
| group_6.memory_breakdown_gb | model=7.00, act=51.23, pad=0.00 | model=7.00, act=50.68, pad=0.00 | model=7.00, act=46.16, pad=0.00 | model=7.00, act=41.58, pad=0.00 | model=7.00, act=26.05, pad=0.00 |
| group_6.seq_sum_local | seq_sum=13248, local=13248.00 | seq_sum=26208, local=13104.00 | seq_sum=23872, local=11936.00 | seq_sum=10752, local=10752.00 | seq_sum=26944, local=6736.00 |
| group_6.sequence_ids | [27, 41] | [32, 16, 34, 57] | [32, 14] | [28, 57, 2, 11, 47] | [19, 16] |
| group_6.sequence_lengths | [11200, 2048] | [14208, 8000, 2240, 1760] | [14208, 9664] | [5952, 1760, 1440, 832, 768] | [18944, 8000] |
| group_6.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive |
| group_6.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_6.compute | fwd=151.14, bwd=302.28 | step/layer=2.76, fwd/layer=5.59, bwd/layer=11.19 | step/layer=2.85, fwd/layer=5.77, bwd/layer=11.53 | fwd=58.74, bwd=117.48 | step/layer=1.14, fwd/layer=4.80, bwd/layer=9.61 |
| group_6.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_6.ring_comm | 0.00 | fwd_step=0.65, bwd_step=1.30, fwd/layer=5.59, bwd/layer=11.19 | fwd_step=0.60, bwd_step=1.20, fwd/layer=5.77, bwd/layer=11.53 | 0.00 | fwd_step=0.85, bwd_step=1.71, fwd/layer=4.80, bwd/layer=9.61 |
| group_6.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 | additive, leakage=None | overlap, leakage=0.1 |
| group_6.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_7.strategy | ulysses×1 | ring×4 | ring×4 | ulysses×1 |  |
| group_7.num_gpus | 1 | 4 | 4 | 1 |  |
| group_7.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=4, placement=context_first | sp=1, cp=4, placement=context_first | sp=1, cp=1, placement=head_first |  |
| group_7.estimated_time_ms | 680.00 | 584.87 | 584.87 | 325.91 |  |
| group_7.estimated_memory_gb | 61.95 | 30.79 | 30.79 | 44.37 |  |
| group_7.memory_breakdown_gb | model=7.00, act=54.95, pad=0.00 | model=7.00, act=23.79, pad=0.00 | model=7.00, act=23.79, pad=0.00 | model=7.00, act=37.37, pad=0.00 |  |
| group_7.seq_sum_local | seq_sum=14208, local=14208.00 | seq_sum=24608, local=6152.00 | seq_sum=24608, local=6152.00 | seq_sum=9664, local=9664.00 |  |
| group_7.sequence_ids | [32] | [52] | [52] | [14] |  |
| group_7.sequence_lengths | [14208] | [24608] | [24608] | [9664] |  |
| group_7.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided |  |
| group_7.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |  |
| group_7.compute | fwd=226.67, bwd=453.34 | step/layer=1.46, fwd/layer=6.09, bwd/layer=12.18 | step/layer=1.46, fwd/layer=6.09, bwd/layer=12.18 | fwd=108.64, bwd=217.27 |  |
| group_7.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 |  |
| group_7.ring_comm | 0.00 | fwd_step=0.82, bwd_step=1.64, fwd/layer=6.09, bwd/layer=12.18 | fwd_step=0.82, bwd_step=1.64, fwd/layer=6.09, bwd/layer=12.18 | 0.00 |  |
| group_7.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 | additive, leakage=None |  |
| group_7.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |  |
| group_8.strategy | ulysses×4 | ring×4 | ring×4 | usp(sp2×cp2,cf) |  |
| group_8.num_gpus | 4 | 4 | 4 | 4 |  |
| group_8.sp_cp_placement | sp=4, cp=1, placement=head_first | sp=1, cp=4, placement=context_first | sp=1, cp=4, placement=context_first | sp=2, cp=2, placement=context_first |  |
| group_8.estimated_time_ms | 644.03 | 385.70 | 461.17 | 924.34 |  |
| group_8.estimated_memory_gb | 25.32 | 26.71 | 33.05 | 25.32 |  |
| group_8.memory_breakdown_gb | model=7.00, act=18.32, pad=0.00 | model=7.00, act=19.71, pad=0.00 | model=7.00, act=26.05, pad=0.00 | model=7.00, act=18.32, pad=0.00 |  |
| group_8.seq_sum_local | seq_sum=18944, local=4736.00 | seq_sum=20384, local=5096.00 | seq_sum=26944, local=6736.00 | seq_sum=18944, local=4736.00 |  |
| group_8.sequence_ids | [19] | [19, 2] | [19, 16] | [19] |  |
| group_8.sequence_lengths | [18944] | [18944, 1440] | [18944, 8000] | [18944] |  |
| group_8.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |  |
| group_8.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |  |
| group_8.compute | fwd=98.82, bwd=197.65 | step/layer=0.95, fwd/layer=4.02, bwd/layer=8.04 | step/layer=1.14, fwd/layer=4.80, bwd/layer=9.61 | step/layer=1.63 |  |
| group_8.alltoall_comm | 347.56 | 0.00 | 0.00 | qo_op=3.88, kv_op=0.87, fwd/layer=9.50, bwd/layer=9.50 |  |
| group_8.ring_comm | 0.00 | fwd_step=0.77, bwd_step=1.53, fwd/layer=4.02, bwd/layer=8.04 | fwd_step=0.85, bwd_step=1.71, fwd/layer=4.80, bwd/layer=9.61 | fwd_step=0.30, bwd_step=0.59, fwd/layer=3.30, bwd/layer=6.60 |  |
| group_8.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 | overlap, leakage=0.1 |  |
| group_8.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) |  |
| group_9.strategy | ulysses×4 |  |  | usp(sp2×cp2,cf) |  |
| group_9.num_gpus | 4 |  |  | 4 |  |
| group_9.sp_cp_placement | sp=4, cp=1, placement=head_first |  |  | sp=2, cp=2, placement=context_first |  |
| group_9.estimated_time_ms | 862.91 |  |  | 1280.87 |  |
| group_9.estimated_memory_gb | 30.79 |  |  | 30.79 |  |
| group_9.memory_breakdown_gb | model=7.00, act=23.79, pad=0.00 |  |  | model=7.00, act=23.79, pad=0.00 |  |
| group_9.seq_sum_local | seq_sum=24608, local=6152.00 |  |  | seq_sum=24608, local=6152.00 |  |
| group_9.sequence_ids | [52] |  |  | [52] |  |
| group_9.sequence_lengths | [24608] |  |  | [24608] |  |
| group_9.topology | a2a=consecutive, ring=strided |  |  | a2a=strided, ring=consecutive |  |
| group_9.head_padding | q=1.0, kv=1.0 |  |  | q=1.0, kv=1.0 |  |
| group_9.compute | fwd=164.52, bwd=329.04 |  |  | step/layer=2.69 |  |
| group_9.alltoall_comm | 369.35 |  |  | qo_op=4.93, kv_op=1.02, fwd/layer=11.90, bwd/layer=11.90 |  |
| group_9.ring_comm | 0.00 |  |  | fwd_step=0.36, bwd_step=0.71, fwd/layer=5.41, bwd/layer=10.82 |  |
| group_9.time_model | additive, leakage=None |  |  | overlap, leakage=0.1 |  |
| group_9.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |  |  | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) |  |
