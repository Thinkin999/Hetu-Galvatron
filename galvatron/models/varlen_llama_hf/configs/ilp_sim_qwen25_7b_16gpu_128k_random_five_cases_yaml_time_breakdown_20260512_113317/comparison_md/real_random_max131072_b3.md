# ILP Horizontal Comparison: `real_random_max131072_b3`

## Batch
| field | value |
| --- | --- |
| sampling_mode | real_random |
| max_seq | 131072 |
| global_batch_size | 64 |
| seq_sum | 182912 |
| max / p95 / p99 | 29792 / 10419.20 / 23159.36 |
| >=128k / >=256k / >=384k / >=512k | 0 / 0 / 0 / 0 |
| sequences | [0:16544, 1:3840, 2:384, 3:480, 4:1344, 5:416, 6:992, 7:512, 8:608, 9:704, 10:8032, 11:3040, 12:160, 13:256, 14:1280, 15:96, 16:224, 17:1792, 18:384, 19:6240, 20:544, 21:224, 22:10592, 23:3712, 24:4128, 25:1184, 26:19264, 27:192, 28:640, 29:29792, 30:384, 31:4448, 32:800, 33:1376, 34:7168, 35:320, 36:1376, 37:544, 38:512, 39:96, 40:3904, 41:1888, 42:1568, 43:544, 44:1600, 45:1536, 46:2176, 47:3552, 48:224, 49:192, 50:6944, 51:3072, 52:3424, 53:800, 54:128, 55:992, 56:9440, 57:192, 58:3296, 59:256, 60:128, 61:160, 62:832, 63:1440] |

## Case Summary
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| status | ok | ok | ok | ok | ok |
| total_time_ms | 1106.9949 | 825.0843 | 825.0843 | 1509.9671 | 825.0843 |
| speedup_vs_ulysses | 1.0000 | 1.3417 | 1.3417 | 0.7331 | 1.3417 |
| microbatch_count | 1 | 1 | 1 | 1 | 1 |
| solver_wall_s | 1.8733 | 2.7568 | 4.7666 | 17.0165 | 4.8244 |

## Microbatch 0
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| estimated_time_ms | 1106.99 | 825.08 | 825.08 | 1509.97 | 825.08 |
| seq_sum | 182912 | 182912 | 182912 | 182912 | 182912 |
| num_sequences | 64 | 64 | 64 | 64 | 64 |
| sequences | [0:16544, 1:3840, 2:384, 3:480, 4:1344, 5:416, 6:992, 7:512, 8:608, 9:704, 10:8032, 11:3040, 12:160, 13:256, 14:1280, 15:96, 16:224, 17:1792, 18:384, 19:6240, 20:544, 21:224, 22:10592, 23:3712, ...(+40)] | [0:16544, 1:3840, 2:384, 3:480, 4:1344, 5:416, 6:992, 7:512, 8:608, 9:704, 10:8032, 11:3040, 12:160, 13:256, 14:1280, 15:96, 16:224, 17:1792, 18:384, 19:6240, 20:544, 21:224, 22:10592, 23:3712, ...(+40)] | [0:16544, 1:3840, 2:384, 3:480, 4:1344, 5:416, 6:992, 7:512, 8:608, 9:704, 10:8032, 11:3040, 12:160, 13:256, 14:1280, 15:96, 16:224, 17:1792, 18:384, 19:6240, 20:544, 21:224, 22:10592, 23:3712, ...(+40)] | [0:16544, 1:3840, 2:384, 3:480, 4:1344, 5:416, 6:992, 7:512, 8:608, 9:704, 10:8032, 11:3040, 12:160, 13:256, 14:1280, 15:96, 16:224, 17:1792, 18:384, 19:6240, 20:544, 21:224, 22:10592, 23:3712, ...(+40)] | [0:16544, 1:3840, 2:384, 3:480, 4:1344, 5:416, 6:992, 7:512, 8:608, 9:704, 10:8032, 11:3040, 12:160, 13:256, 14:1280, 15:96, 16:224, 17:1792, 18:384, 19:6240, 20:544, 21:224, 22:10592, 23:3712, ...(+40)] |

| group metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| group_0.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_0.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_0.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_0.estimated_time_ms | 169.63 | 135.40 | 99.83 | 293.11 | 194.86 |
| group_0.estimated_memory_gb | 70.36 | 60.71 | 40.04 | 71.84 | 71.10 |
| group_0.memory_breakdown_gb | model=7.00, act=63.36, pad=0.00 | model=7.00, act=53.71, pad=0.00 | model=7.00, act=33.04, pad=0.00 | model=7.00, act=64.84, pad=0.00 | model=7.00, act=64.10, pad=0.00 |
| group_0.seq_sum_local | seq_sum=16384, local=16384.00 | seq_sum=13888, local=13888.00 | seq_sum=8544, local=8544.00 | seq_sum=16768, local=16768.00 | seq_sum=16576, local=16576.00 |
| group_0.sequence_ids | [58, 46, 41, 55, 62, 32, 53, 8, 20, 37, 43, 38, 5, 2, 18, 30, 35, 13, 48, 57, 61, 54] | [46, 41, 44, 63, 36, 25, 6, 43, 5, 18, 30, 13, 16, 21, 48, 27, 49, 57] | [47, 63, 25, 53, 3, 5, 13, 27, 54, 39] | [50, 23, 17, 33, 62, 9, 28, 12, 61, 54, 60, 15, 39] | [40, 51, 63, 33, 14, 53, 28, 7, 38, 3, 2, 35, 13, 59, 16, 21, 48, 57, 12, 60, 15, 39] |
| group_0.sequence_lengths | [3296, 2176, 1888, 992, 832, 800, 800, 608, 544, 544, 544, 512, 416, 384, 384, 384, 320, 256, 224, 192, 160, 128] | [2176, 1888, 1600, 1440, 1376, 1184, 992, 544, 416, 384, 384, 256, 224, 224, 224, 192, 192, 192] | [3552, 1440, 1184, 800, 480, 416, 256, 192, 128, 96] | [6944, 3712, 1792, 1376, 832, 704, 640, 160, 160, 128, 128, 96, 96] | [3904, 3072, 1440, 1376, 1280, 800, 640, 512, 512, 480, 384, 320, 256, 256, 224, 224, 224, 192, 160, 128, 96, 96] |
| group_0.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_0.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_0.compute | fwd=56.54, bwd=113.09 | fwd=45.13, bwd=90.27 | fwd=33.28, bwd=66.55 | fwd=97.70, bwd=195.41 | fwd=64.95, bwd=129.91 |
| group_0.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_0.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_0.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_0.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_1.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_1.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_1.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_1.estimated_time_ms | 311.77 | 471.24 | 245.08 | 474.91 | 86.25 |
| group_1.estimated_memory_gb | 43.51 | 71.35 | 64.42 | 71.97 | 30.76 |
| group_1.memory_breakdown_gb | model=7.00, act=36.51, pad=0.00 | model=7.00, act=64.35, pad=0.00 | model=7.00, act=57.42, pad=0.00 | model=7.00, act=64.97, pad=0.00 | model=7.00, act=23.76, pad=0.00 |
| group_1.seq_sum_local | seq_sum=9440, local=9440.00 | seq_sum=16640, local=16640.00 | seq_sum=14848, local=14848.00 | seq_sum=16800, local=16800.00 | seq_sum=6144, local=6144.00 |
| group_1.sequence_ids | [56] | [22, 40, 33, 38, 12, 15] | [50, 36, 14, 55, 32, 28, 37, 43, 18, 30, 35, 59, 49, 57] | [22, 40, 41, 21, 49] | [24, 25, 62] |
| group_1.sequence_lengths | [9440] | [10592, 3904, 1376, 512, 160, 96] | [6944, 1376, 1280, 992, 800, 640, 544, 544, 384, 384, 320, 256, 192, 192] | [10592, 3904, 1888, 224, 192] | [4128, 1184, 832] |
| group_1.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_1.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_1.compute | fwd=103.92, bwd=207.85 | fwd=157.08, bwd=314.16 | fwd=81.69, bwd=163.39 | fwd=158.30, bwd=316.61 | fwd=28.75, bwd=57.50 |
| group_1.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_1.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_1.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_1.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_2.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_2.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_2.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_2.estimated_time_ms | 352.59 | 141.67 | 180.16 | 395.73 | 244.07 |
| group_2.estimated_memory_gb | 68.13 | 58.73 | 54.40 | 71.97 | 64.05 |
| group_2.memory_breakdown_gb | model=7.00, act=61.13, pad=0.00 | model=7.00, act=51.73, pad=0.00 | model=7.00, act=47.40, pad=0.00 | model=7.00, act=64.97, pad=0.00 | model=7.00, act=57.05, pad=0.00 |
| group_2.seq_sum_local | seq_sum=15808, local=15808.00 | seq_sum=13376, local=13376.00 | seq_sum=12256, local=12256.00 | seq_sum=16800, local=16800.00 | seq_sum=14752, local=14752.00 |
| group_2.sequence_ids | [34, 19, 42, 28, 49] | [23, 42, 55, 62, 32, 53, 9, 28, 20, 37, 7, 3, 2, 35, 59, 61, 54] | [31, 24, 44, 62, 9, 20] | [56, 52, 6, 55, 3, 2, 18, 30, 35] | [19, 47, 17, 42, 6, 8] |
| group_2.sequence_lengths | [7168, 6240, 1568, 640, 192] | [3712, 1568, 992, 832, 800, 800, 704, 640, 544, 544, 512, 480, 384, 320, 256, 160, 128] | [4448, 4128, 1600, 832, 704, 544] | [9440, 3424, 992, 992, 480, 384, 384, 384, 320] | [6240, 3552, 1792, 1568, 992, 608] |
| group_2.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_2.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_2.compute | fwd=117.53, bwd=235.06 | fwd=47.22, bwd=94.45 | fwd=60.05, bwd=120.11 | fwd=131.91, bwd=263.82 | fwd=81.36, bwd=162.71 |
| group_2.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_2.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_2.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_2.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_3.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_3.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_3.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_3.estimated_time_ms | 175.45 | 408.37 | 180.61 | 914.65 | 184.35 |
| group_3.estimated_memory_gb | 33.85 | 70.85 | 54.89 | 71.97 | 57.24 |
| group_3.memory_breakdown_gb | model=7.00, act=26.85, pad=0.00 | model=7.00, act=63.85, pad=0.00 | model=7.00, act=47.89, pad=0.00 | model=7.00, act=64.97, pad=0.00 | model=7.00, act=50.24, pad=0.00 |
| group_3.seq_sum_local | seq_sum=6944, local=6944.00 | seq_sum=16512, local=16512.00 | seq_sum=12384, local=12384.00 | seq_sum=16800, local=16800.00 | seq_sum=12992, local=12992.00 |
| group_3.sequence_ids | [50] | [56, 52, 11, 8] | [1, 23, 58, 45] | [0, 59] | [31, 23, 46, 36, 20, 37, 49] |
| group_3.sequence_lengths | [6944] | [9440, 3424, 3040, 608] | [3840, 3712, 3296, 1536] | [16544, 256] | [4448, 3712, 2176, 1376, 544, 544, 192] |
| group_3.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_3.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_3.compute | fwd=58.48, bwd=116.97 | fwd=136.12, bwd=272.25 | fwd=60.20, bwd=120.40 | fwd=304.88, bwd=609.77 | fwd=61.45, bwd=122.90 |
| group_3.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_3.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_3.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_3.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_4.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_4.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_4.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_4.estimated_time_ms | 202.04 | 151.95 | 223.26 | 346.94 | 252.13 |
| group_4.estimated_memory_gb | 66.65 | 47.84 | 57.12 | 71.97 | 65.53 |
| group_4.memory_breakdown_gb | model=7.00, act=59.65, pad=0.00 | model=7.00, act=40.84, pad=0.00 | model=7.00, act=50.12, pad=0.00 | model=7.00, act=64.97, pad=0.00 | model=7.00, act=58.53, pad=0.00 |
| group_4.seq_sum_local | seq_sum=15424, local=15424.00 | seq_sum=10560, local=10560.00 | seq_sum=12960, local=12960.00 | seq_sum=16800, local=16800.00 | seq_sum=15136, local=15136.00 |
| group_4.sequence_ids | [23, 47, 51, 63, 33, 6, 7, 16, 21, 60, 15, 39] | [31, 58, 45, 14] | [19, 11, 41, 17] | [10, 31, 4, 14, 25, 7] | [50, 41, 45, 4, 55, 32, 43, 5, 30, 61, 54] |
| group_4.sequence_lengths | [3712, 3552, 3072, 1440, 1376, 992, 512, 224, 224, 128, 96, 96] | [4448, 3296, 1536, 1280] | [6240, 3040, 1888, 1792] | [8032, 4448, 1344, 1280, 1184, 512] | [6944, 1888, 1536, 1344, 992, 800, 544, 416, 384, 160, 128] |
| group_4.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_4.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_4.compute | fwd=67.35, bwd=134.69 | fwd=50.65, bwd=101.30 | fwd=74.42, bwd=148.84 | fwd=115.65, bwd=231.29 | fwd=84.04, bwd=168.09 |
| group_4.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_4.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_4.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_4.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_5.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_5.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_5.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_5.estimated_time_ms | 369.80 | 143.37 | 334.88 | 182.47 | 226.76 |
| group_5.estimated_memory_gb | 70.11 | 47.34 | 70.73 | 71.72 | 70.73 |
| group_5.memory_breakdown_gb | model=7.00, act=63.11, pad=0.00 | model=7.00, act=40.34, pad=0.00 | model=7.00, act=63.73, pad=0.00 | model=7.00, act=64.72, pad=0.00 | model=7.00, act=63.73, pad=0.00 |
| group_5.seq_sum_local | seq_sum=16320, local=16320.00 | seq_sum=10432, local=10432.00 | seq_sum=16480, local=16480.00 | seq_sum=16736, local=16736.00 | seq_sum=16480, local=16480.00 |
| group_5.sequence_ids | [10, 31, 1] | [24, 51, 17, 4, 39] | [10, 40, 42, 33, 6, 7, 15] | [51, 11, 46, 42, 63, 32, 53, 8, 20, 37, 43, 38, 5, 13, 16, 27] | [1, 52, 58, 11, 44, 9, 18, 27] |
| group_5.sequence_lengths | [8032, 4448, 3840] | [4128, 3072, 1792, 1344, 96] | [8032, 3904, 1568, 1376, 992, 512, 96] | [3072, 3040, 2176, 1568, 1440, 800, 800, 608, 544, 544, 544, 512, 416, 256, 224, 192] | [3840, 3424, 3296, 3040, 1600, 704, 384, 192] |
| group_5.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_5.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_5.compute | fwd=123.27, bwd=246.53 | fwd=47.79, bwd=95.58 | fwd=111.63, bwd=223.25 | fwd=60.82, bwd=121.65 | fwd=75.59, bwd=151.18 |
| group_5.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_5.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_5.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_5.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_6.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×2 |
| group_6.num_gpus | 1 | 1 | 1 | 1 | 2 |
| group_6.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=2, cp=1, placement=head_first |
| group_6.estimated_time_ms | 92.75 | 53.53 | 261.33 | 288.46 | 779.88 |
| group_6.estimated_memory_gb | 44.99 | 20.74 | 60.21 | 71.35 | 44.25 |
| group_6.memory_breakdown_gb | model=7.00, act=37.99, pad=0.00 | model=7.00, act=13.74, pad=0.00 | model=7.00, act=53.21, pad=0.00 | model=7.00, act=64.35, pad=0.00 | model=7.00, act=37.25, pad=0.00 |
| group_6.seq_sum_local | seq_sum=9824, local=9824.00 | seq_sum=3552, local=3552.00 | seq_sum=13760, local=13760.00 | seq_sum=16640, local=16640.00 | seq_sum=19264, local=9632.00 |
| group_6.sequence_ids | [17, 44, 45, 36, 14, 25, 9, 27, 12] | [47] | [34, 51, 4, 8, 38, 2, 48, 12, 61, 60] | [19, 24, 58, 44, 36] | [26] |
| group_6.sequence_lengths | [1792, 1600, 1536, 1376, 1280, 1184, 704, 192, 160] | [3552] | [7168, 3072, 1344, 608, 512, 384, 224, 160, 160, 128] | [6240, 4128, 3296, 1600, 1376] | [19264] |
| group_6.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_6.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_6.compute | fwd=30.92, bwd=61.84 | fwd=17.84, bwd=35.69 | fwd=87.11, bwd=174.22 | fwd=96.15, bwd=192.30 | fwd=204.18, bwd=408.36 |
| group_6.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 167.33 |
| group_6.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_6.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_6.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_7.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ring×2 |
| group_7.num_gpus | 1 | 1 | 1 | 1 | 2 |
| group_7.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first |
| group_7.estimated_time_ms | 176.98 | 232.02 | 391.27 | 319.99 | 460.07 |
| group_7.estimated_memory_gb | 55.02 | 38.56 | 66.89 | 70.85 | 56.87 |
| group_7.memory_breakdown_gb | model=7.00, act=48.02, pad=0.00 | model=7.00, act=31.56, pad=0.00 | model=7.00, act=59.89, pad=0.00 | model=7.00, act=63.85, pad=0.00 | model=7.00, act=49.87, pad=0.00 |
| group_7.seq_sum_local | seq_sum=12416, local=12416.00 | seq_sum=8160, local=8160.00 | seq_sum=15488, local=15488.00 | seq_sum=16512, local=16512.00 | seq_sum=25792, local=12896.00 |
| group_7.sequence_ids | [24, 52, 11, 4, 3] | [10, 60] | [56, 52, 46, 16, 21] | [34, 1, 47, 45, 48, 57] | [22, 10, 34] |
| group_7.sequence_lengths | [4128, 3424, 3040, 1344, 480] | [8032, 128] | [9440, 3424, 2176, 224, 224] | [7168, 3840, 3552, 1536, 224, 192] | [10592, 8032, 7168] |
| group_7.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive |
| group_7.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_7.compute | fwd=58.99, bwd=117.99 | fwd=77.34, bwd=154.68 | fwd=130.42, bwd=260.85 | fwd=106.66, bwd=213.32 | step/layer=2.36, fwd/layer=4.79, bwd/layer=9.58 |
| group_7.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_7.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | fwd_step=0.64, bwd_step=1.28, fwd/layer=4.79, bwd/layer=9.58 |
| group_7.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | overlap, leakage=0.1 |
| group_7.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_8.strategy | ulysses×2 | ring×2 | ulysses×2 | usp(sp2×cp4,cf) | ring×2 |
| group_8.num_gpus | 2 | 2 | 2 | 8 | 2 |
| group_8.sp_cp_placement | sp=2, cp=1, placement=head_first | sp=1, cp=2, placement=context_first | sp=2, cp=1, placement=head_first | sp=2, cp=4, placement=context_first | sp=1, cp=2, placement=context_first |
| group_8.estimated_time_ms | 664.79 | 762.69 | 779.88 | 1363.30 | 666.88 |
| group_8.estimated_memory_gb | 47.03 | 58.11 | 44.25 | 30.71 | 57.24 |
| group_8.memory_breakdown_gb | model=7.00, act=40.03, pad=0.00 | model=7.00, act=51.11, pad=0.00 | model=7.00, act=37.25, pad=0.00 | model=7.00, act=23.71, pad=0.00 | model=7.00, act=50.24, pad=0.00 |
| group_8.seq_sum_local | seq_sum=20704, local=10352.00 | seq_sum=26432, local=13216.00 | seq_sum=19264, local=9632.00 | seq_sum=49056, local=6132.00 | seq_sum=25984, local=12992.00 |
| group_8.sequence_ids | [0, 40, 59] | [26, 34] | [26] | [29, 26] | [0, 56] |
| group_8.sequence_lengths | [16544, 3904, 256] | [19264, 7168] | [19264] | [29792, 19264] | [16544, 9440] |
| group_8.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |
| group_8.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_8.compute | fwd=162.91, bwd=325.82 | step/layer=3.94, fwd/layer=7.94, bwd/layer=15.89 | fwd=204.18, bwd=408.36 | step/layer=1.51 | step/layer=3.44, fwd/layer=6.95, bwd/layer=13.89 |
| group_8.alltoall_comm | 176.06 | 0.00 | 167.33 | qo_op=4.92, kv_op=1.02, fwd/layer=11.86, bwd/layer=11.86 | 0.00 |
| group_8.ring_comm | 0.00 | fwd_step=0.65, bwd_step=1.31, fwd/layer=7.94, bwd/layer=15.89 | 0.00 | fwd_step=0.82, bwd_step=1.64, fwd/layer=6.29, bwd/layer=12.58 | fwd_step=0.64, bwd_step=1.29, fwd/layer=6.95, bwd/layer=13.89 |
| group_8.time_model | additive, leakage=None | overlap, leakage=0.1 | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 |
| group_8.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_9.strategy | ulysses×2 | ring×2 | ring×2 |  | ring×4 |
| group_9.num_gpus | 2 | 2 | 2 |  | 4 |
| group_9.sp_cp_placement | sp=2, cp=1, placement=head_first | sp=1, cp=2, placement=context_first | sp=1, cp=2, placement=context_first |  | sp=1, cp=4, placement=context_first |
| group_9.estimated_time_ms | 1037.95 | 723.69 | 706.12 |  | 825.08 |
| group_9.estimated_memory_gb | 64.73 | 71.91 | 59.47 |  | 35.80 |
| group_9.memory_breakdown_gb | model=7.00, act=57.73, pad=0.00 | model=7.00, act=64.91, pad=0.00 | model=7.00, act=52.47, pad=0.00 |  | model=7.00, act=28.80, pad=0.00 |
| group_9.seq_sum_local | seq_sum=29856, local=14928.00 | seq_sum=33568, local=16784.00 | seq_sum=27136, local=13568.00 |  | seq_sum=29792, local=7448.00 |
| group_9.sequence_ids | [26, 22] | [0, 50, 19, 1] | [0, 22] |  | [29] |
| group_9.sequence_lengths | [19264, 10592] | [16544, 6944, 6240, 3840] | [16544, 10592] |  | [29792] |
| group_9.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |  | a2a=strided, ring=consecutive |
| group_9.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |  | q=1.0, kv=1.0 |
| group_9.compute | fwd=268.81, bwd=537.62 | step/layer=3.73, fwd/layer=7.54, bwd/layer=15.08 | step/layer=3.64, fwd/layer=7.36, bwd/layer=14.71 |  | step/layer=2.08, fwd/layer=8.59, bwd/layer=17.19 |
| group_9.alltoall_comm | 231.52 | 0.00 | 0.00 |  | 0.00 |
| group_9.ring_comm | 0.00 | fwd_step=0.80, bwd_step=1.61, fwd/layer=7.54, bwd/layer=15.08 | fwd_step=0.67, bwd_step=1.34, fwd/layer=7.36, bwd/layer=14.71 |  | fwd_step=0.89, bwd_step=1.78, fwd/layer=8.59, bwd/layer=17.19 |
| group_9.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 |  | overlap, leakage=0.1 |
| group_9.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) |  | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_10.strategy | ulysses×4 | ring×4 | ring×4 |  |  |
| group_10.num_gpus | 4 | 4 | 4 |  |  |
| group_10.sp_cp_placement | sp=4, cp=1, placement=head_first | sp=1, cp=4, placement=context_first | sp=1, cp=4, placement=context_first |  |  |
| group_10.estimated_time_ms | 1106.99 | 825.08 | 825.08 |  |  |
| group_10.estimated_memory_gb | 35.80 | 35.80 | 35.80 |  |  |
| group_10.memory_breakdown_gb | model=7.00, act=28.80, pad=0.00 | model=7.00, act=28.80, pad=0.00 | model=7.00, act=28.80, pad=0.00 |  |  |
| group_10.seq_sum_local | seq_sum=29792, local=7448.00 | seq_sum=29792, local=7448.00 | seq_sum=29792, local=7448.00 |  |  |
| group_10.sequence_ids | [29] | [29] | [29] |  |  |
| group_10.sequence_lengths | [29792] | [29792] | [29792] |  |  |
| group_10.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |  |  |
| group_10.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |  |  |
| group_10.compute | fwd=239.24, bwd=478.47 | step/layer=2.08, fwd/layer=8.59, bwd/layer=17.19 | step/layer=2.08, fwd/layer=8.59, bwd/layer=17.19 |  |  |
| group_10.alltoall_comm | 389.28 | 0.00 | 0.00 |  |  |
| group_10.ring_comm | 0.00 | fwd_step=0.89, bwd_step=1.78, fwd/layer=8.59, bwd/layer=17.19 | fwd_step=0.89, bwd_step=1.78, fwd/layer=8.59, bwd/layer=17.19 |  |  |
| group_10.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 |  |  |
| group_10.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) |  |  |
