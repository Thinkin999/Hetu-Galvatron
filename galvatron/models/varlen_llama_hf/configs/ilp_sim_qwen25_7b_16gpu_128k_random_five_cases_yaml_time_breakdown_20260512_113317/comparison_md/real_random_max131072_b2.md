# ILP Horizontal Comparison: `real_random_max131072_b2`

## Batch
| field | value |
| --- | --- |
| sampling_mode | real_random |
| max_seq | 131072 |
| global_batch_size | 64 |
| seq_sum | 170208 |
| max / p95 / p99 | 27360 / 13537.60 / 22561.92 |
| >=128k / >=256k / >=384k / >=512k | 0 / 0 / 0 / 0 |
| sequences | [0:2784, 1:4160, 2:416, 3:1376, 4:1408, 5:416, 6:288, 7:352, 8:1088, 9:544, 10:448, 11:7232, 12:704, 13:96, 14:160, 15:64, 16:64, 17:256, 18:1088, 19:288, 20:1824, 21:14976, 22:1184, 23:288, 24:19744, 25:5376, 26:6880, 27:384, 28:2848, 29:2400, 30:256, 31:192, 32:128, 33:2464, 34:2400, 35:320, 36:1984, 37:1600, 38:1120, 39:1760, 40:11008, 41:1440, 42:96, 43:1152, 44:1984, 45:1504, 46:352, 47:6624, 48:480, 49:128, 50:13984, 51:96, 52:128, 53:1728, 54:480, 55:64, 56:416, 57:128, 58:3136, 59:27360, 60:416, 61:64, 62:3328, 63:2752] |

## Case Summary
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| status | ok | ok | ok | ok | ok |
| total_time_ms | 987.2757 | 707.1816 | 707.1816 | 1427.7913 | 707.1816 |
| speedup_vs_ulysses | 1.0000 | 1.3961 | 1.3961 | 0.6915 | 1.3961 |
| microbatch_count | 1 | 1 | 1 | 1 | 1 |
| solver_wall_s | 4.6261 | 2.4669 | 5.0577 | 17.1781 | 4.7708 |

## Microbatch 0
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| estimated_time_ms | 987.28 | 707.18 | 707.18 | 1427.79 | 707.18 |
| seq_sum | 170208 | 170208 | 170208 | 170208 | 170208 |
| num_sequences | 64 | 64 | 64 | 64 | 64 |
| sequences | [0:2784, 1:4160, 2:416, 3:1376, 4:1408, 5:416, 6:288, 7:352, 8:1088, 9:544, 10:448, 11:7232, 12:704, 13:96, 14:160, 15:64, 16:64, 17:256, 18:1088, 19:288, 20:1824, 21:14976, 22:1184, 23:288, ...(+40)] | [0:2784, 1:4160, 2:416, 3:1376, 4:1408, 5:416, 6:288, 7:352, 8:1088, 9:544, 10:448, 11:7232, 12:704, 13:96, 14:160, 15:64, 16:64, 17:256, 18:1088, 19:288, 20:1824, 21:14976, 22:1184, 23:288, ...(+40)] | [0:2784, 1:4160, 2:416, 3:1376, 4:1408, 5:416, 6:288, 7:352, 8:1088, 9:544, 10:448, 11:7232, 12:704, 13:96, 14:160, 15:64, 16:64, 17:256, 18:1088, 19:288, 20:1824, 21:14976, 22:1184, 23:288, ...(+40)] | [0:2784, 1:4160, 2:416, 3:1376, 4:1408, 5:416, 6:288, 7:352, 8:1088, 9:544, 10:448, 11:7232, 12:704, 13:96, 14:160, 15:64, 16:64, 17:256, 18:1088, 19:288, 20:1824, 21:14976, 22:1184, 23:288, ...(+40)] | [0:2784, 1:4160, 2:416, 3:1376, 4:1408, 5:416, 6:288, 7:352, 8:1088, 9:544, 10:448, 11:7232, 12:704, 13:96, 14:160, 15:64, 16:64, 17:256, 18:1088, 19:288, 20:1824, 21:14976, 22:1184, 23:288, ...(+40)] |

| group metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| group_0.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_0.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_0.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_0.estimated_time_ms | 156.59 | 164.94 | 448.33 | 161.07 | 160.30 |
| group_0.estimated_memory_gb | 68.01 | 71.72 | 62.94 | 68.50 | 51.80 |
| group_0.memory_breakdown_gb | model=7.00, act=61.01, pad=0.00 | model=7.00, act=64.72, pad=0.00 | model=7.00, act=55.94, pad=0.00 | model=7.00, act=61.50, pad=0.00 | model=7.00, act=44.80, pad=0.00 |
| group_0.seq_sum_local | seq_sum=15776, local=15776.00 | seq_sum=16736, local=16736.00 | seq_sum=14464, local=14464.00 | seq_sum=15904, local=15904.00 | seq_sum=11584, local=11584.00 |
| group_0.sequence_ids | [29, 44, 45, 41, 4, 3, 43, 8, 12, 9, 48, 54, 35, 6, 32, 49, 52, 51, 15, 55] | [36, 39, 37, 41, 43, 8, 12, 9, 48, 54, 10, 2, 5, 56, 27, 7, 46, 6, 19, 23, 17, 30, 31, 14, ...(+10)] | [40, 18, 54, 10, 2, 27, 6, 17, 42] | [33, 34, 37, 45, 4, 3, 22, 8, 18, 9, 48, 10, 13, 42, 15, 16] | [1, 62, 20, 43, 27, 6, 30, 15, 16, 61] |
| group_0.sequence_lengths | [2400, 1984, 1504, 1440, 1408, 1376, 1152, 1088, 704, 544, 480, 480, 320, 288, 128, 128, 128, 96, 64, 64] | [1984, 1760, 1600, 1440, 1152, 1088, 704, 544, 480, 480, 448, 416, 416, 416, 384, 352, 352, 288, 288, 288, 256, 256, 192, 160, ...(+10)] | [11008, 1088, 480, 448, 416, 384, 288, 256, 96] | [2464, 2400, 1600, 1504, 1408, 1376, 1184, 1088, 1088, 544, 480, 448, 96, 96, 64, 64] | [4160, 3328, 1824, 1152, 384, 288, 256, 64, 64, 64] |
| group_0.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_0.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_0.compute | fwd=52.20, bwd=104.39 | fwd=54.98, bwd=109.96 | fwd=149.44, bwd=298.89 | fwd=53.69, bwd=107.38 | fwd=53.43, bwd=106.87 |
| group_0.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_0.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_0.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_0.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_1.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_1.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_1.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_1.estimated_time_ms | 218.58 | 196.14 | 338.93 | 217.68 | 232.31 |
| group_1.estimated_memory_gb | 66.89 | 71.60 | 61.95 | 66.77 | 55.14 |
| group_1.memory_breakdown_gb | model=7.00, act=59.89, pad=0.00 | model=7.00, act=64.60, pad=0.00 | model=7.00, act=54.95, pad=0.00 | model=7.00, act=59.77, pad=0.00 | model=7.00, act=48.14, pad=0.00 |
| group_1.seq_sum_local | seq_sum=15488, local=15488.00 | seq_sum=16704, local=16704.00 | seq_sum=14208, local=14208.00 | seq_sum=15456, local=15456.00 | seq_sum=12448, local=12448.00 |
| group_1.sequence_ids | [1, 62, 0, 63, 33] | [0, 63, 33, 29, 34, 44, 38, 60, 35, 15] | [26, 47, 12] | [25, 63, 29, 39, 43, 12, 54, 2, 56] | [26, 29, 36, 22] |
| group_1.sequence_lengths | [4160, 3328, 2784, 2752, 2464] | [2784, 2752, 2464, 2400, 2400, 1984, 1120, 416, 320, 64] | [6880, 6624, 704] | [5376, 2752, 2400, 1760, 1152, 704, 480, 416, 416] | [6880, 2400, 1984, 1184] |
| group_1.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_1.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_1.compute | fwd=72.86, bwd=145.72 | fwd=65.38, bwd=130.76 | fwd=112.98, bwd=225.95 | fwd=72.56, bwd=145.12 | fwd=77.44, bwd=154.87 |
| group_1.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_1.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_1.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_1.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_2.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_2.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_2.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_2.estimated_time_ms | 768.14 | 239.32 | 223.39 | 687.76 | 251.35 |
| group_2.estimated_memory_gb | 71.60 | 70.11 | 65.16 | 71.47 | 69.25 |
| group_2.memory_breakdown_gb | model=7.00, act=64.60, pad=0.00 | model=7.00, act=63.11, pad=0.00 | model=7.00, act=58.16, pad=0.00 | model=7.00, act=64.47, pad=0.00 | model=7.00, act=62.25, pad=0.00 |
| group_2.seq_sum_local | seq_sum=16704, local=16704.00 | seq_sum=16320, local=16320.00 | seq_sum=15040, local=15040.00 | seq_sum=16672, local=16672.00 | seq_sum=16096, local=16096.00 |
| group_2.sequence_ids | [21, 60, 27, 7, 19, 23] | [25, 58, 28, 20, 53, 4] | [25, 0, 33, 34, 53, 23] | [50, 5, 27, 7, 46, 35, 6, 32, 49, 52, 57, 61] | [47, 53, 37, 45, 41, 4, 38, 48, 52, 55] |
| group_2.sequence_lengths | [14976, 416, 384, 352, 288, 288] | [5376, 3136, 2848, 1824, 1728, 1408] | [5376, 2784, 2464, 2400, 1728, 288] | [13984, 416, 384, 352, 352, 320, 288, 128, 128, 128, 128, 64] | [6624, 1728, 1600, 1504, 1440, 1408, 1120, 480, 128, 64] |
| group_2.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_2.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_2.compute | fwd=256.05, bwd=512.09 | fwd=79.77, bwd=159.55 | fwd=74.46, bwd=148.92 | fwd=229.25, bwd=458.50 | fwd=83.78, bwd=167.56 |
| group_2.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_2.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_2.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_2.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_3.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_3.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_3.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_3.estimated_time_ms | 678.24 | 347.46 | 51.51 | 438.43 | 185.20 |
| group_3.estimated_memory_gb | 68.13 | 65.04 | 24.45 | 57.24 | 71.97 |
| group_3.memory_breakdown_gb | model=7.00, act=61.13, pad=0.00 | model=7.00, act=58.04, pad=0.00 | model=7.00, act=17.45, pad=0.00 | model=7.00, act=50.24, pad=0.00 | model=7.00, act=64.97, pad=0.00 |
| group_3.seq_sum_local | seq_sum=15808, local=15808.00 | seq_sum=15008, local=15008.00 | seq_sum=4512, local=4512.00 | seq_sum=12992, local=12992.00 | seq_sum=16800, local=16800.00 |
| group_3.sequence_ids | [50, 20] | [26, 47, 45] | [28, 22, 48] | [40, 44] | [0, 33, 34, 44, 39, 12, 9, 10, 2, 5, 60, 7, 46, 19, 23, 17, 31, 14, 32, 49, 57, 42, 51] |
| group_3.sequence_lengths | [13984, 1824] | [6880, 6624, 1504] | [2848, 1184, 480] | [11008, 1984] | [2784, 2464, 2400, 1984, 1760, 704, 544, 448, 416, 416, 416, 352, 352, 288, 288, 256, 192, 160, 128, 128, 128, 96, 96] |
| group_3.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_3.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_3.compute | fwd=226.08, bwd=452.16 | fwd=115.82, bwd=231.64 | fwd=17.17, bwd=34.34 | fwd=146.14, bwd=292.29 | fwd=61.73, bwd=123.47 |
| group_3.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_3.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_3.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_3.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_4.strategy | ulysses×1 | ring×2 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_4.num_gpus | 1 | 2 | 1 | 1 | 1 |
| group_4.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_4.estimated_time_ms | 460.76 | 538.49 | 179.94 | 178.44 | 159.34 |
| group_4.estimated_memory_gb | 61.70 | 61.02 | 60.58 | 61.82 | 59.59 |
| group_4.memory_breakdown_gb | model=7.00, act=54.70, pad=0.00 | model=7.00, act=54.02, pad=0.00 | model=7.00, act=53.58, pad=0.00 | model=7.00, act=54.82, pad=0.00 | model=7.00, act=52.59, pad=0.00 |
| group_4.seq_sum_local | seq_sum=14144, local=14144.00 | seq_sum=27936, local=13968.00 | seq_sum=13856, local=13856.00 | seq_sum=14176, local=14176.00 | seq_sum=13600, local=13600.00 |
| group_4.sequence_ids | [40, 58] | [50, 11, 1, 3, 22] | [1, 58, 44, 20, 4, 5, 46, 35, 31, 61] | [62, 58, 28, 36, 41, 60, 19, 23, 31, 14, 51] | [58, 28, 63, 3, 8, 18, 54, 56, 35, 13] |
| group_4.sequence_lengths | [11008, 3136] | [13984, 7232, 4160, 1376, 1184] | [4160, 3136, 1984, 1824, 1408, 416, 352, 320, 192, 64] | [3328, 3136, 2848, 1984, 1440, 416, 288, 288, 192, 160, 96] | [3136, 2848, 2752, 1376, 1088, 1088, 480, 416, 320, 96] |
| group_4.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_4.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_4.compute | fwd=153.59, bwd=307.17 | step/layer=2.77, fwd/layer=5.61, bwd/layer=11.22 | fwd=59.98, bwd=119.96 | fwd=59.48, bwd=118.96 | fwd=53.11, bwd=106.23 |
| group_4.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_4.ring_comm | 0.00 | fwd_step=0.69, bwd_step=1.37, fwd/layer=5.61, bwd/layer=11.22 | 0.00 | 0.00 | 0.00 |
| group_4.time_model | additive, leakage=None | overlap, leakage=0.1 | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_4.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_5.strategy | ulysses×1 | ring×2 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_5.num_gpus | 1 | 2 | 1 | 1 | 1 |
| group_5.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_5.estimated_time_ms | 358.62 | 692.64 | 116.63 | 297.26 | 298.84 |
| group_5.estimated_memory_gb | 68.13 | 47.28 | 53.03 | 67.39 | 55.76 |
| group_5.memory_breakdown_gb | model=7.00, act=61.13, pad=0.00 | model=7.00, act=40.28, pad=0.00 | model=7.00, act=46.03, pad=0.00 | model=7.00, act=60.39, pad=0.00 | model=7.00, act=48.76, pad=0.00 |
| group_5.seq_sum_local | seq_sum=15808, local=15808.00 | seq_sum=20832, local=10416.00 | seq_sum=11904, local=11904.00 | seq_sum=15616, local=15616.00 | seq_sum=12608, local=12608.00 |
| group_5.sequence_ids | [26, 47, 36, 30, 61] | [24, 18] | [29, 36, 45, 41, 43, 38, 8, 9, 56, 30] | [26, 1, 0, 53, 55] | [11, 25] |
| group_5.sequence_lengths | [6880, 6624, 1984, 256, 64] | [19744, 1088] | [2400, 1984, 1504, 1440, 1152, 1120, 1088, 544, 416, 256] | [6880, 4160, 2784, 1728, 64] | [7232, 5376] |
| group_5.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_5.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_5.compute | fwd=119.54, bwd=239.08 | step/layer=3.58, fwd/layer=7.21, bwd/layer=14.43 | fwd=38.88, bwd=77.76 | fwd=99.09, bwd=198.17 | fwd=99.61, bwd=199.23 |
| group_5.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_5.ring_comm | 0.00 | fwd_step=0.54, bwd_step=1.07, fwd/layer=7.21, bwd/layer=14.43 | 0.00 | 0.00 | 0.00 |
| group_5.time_model | additive, leakage=None | overlap, leakage=0.1 | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_5.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_6.strategy | ulysses×1 | ring×4 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_6.num_gpus | 1 | 4 | 1 | 1 | 1 |
| group_6.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=4, placement=context_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_6.estimated_time_ms | 265.23 | 427.41 | 137.61 | 757.72 | 659.54 |
| group_6.estimated_memory_gb | 65.66 | 35.34 | 49.20 | 66.89 | 61.08 |
| group_6.memory_breakdown_gb | model=7.00, act=58.66, pad=0.00 | model=7.00, act=28.34, pad=0.00 | model=7.00, act=42.20, pad=0.00 | model=7.00, act=59.89, pad=0.00 | model=7.00, act=54.08, pad=0.00 |
| group_6.seq_sum_local | seq_sum=15168, local=15168.00 | seq_sum=29312, local=7328.00 | seq_sum=10912, local=10912.00 | seq_sum=15488, local=15488.00 | seq_sum=13984, local=13984.00 |
| group_6.sequence_ids | [11, 39, 22, 38, 18, 10, 2, 5, 56, 46, 31, 14, 57, 13, 42, 16] | [21, 40, 62] | [62, 63, 39, 3, 7, 19, 14, 32, 49, 52, 57, 13, 51, 15, 16, 55] | [21, 17, 30] | [50] |
| group_6.sequence_lengths | [7232, 1760, 1184, 1120, 1088, 448, 416, 416, 416, 352, 192, 160, 128, 96, 96, 64] | [14976, 11008, 3328] | [3328, 2752, 1760, 1376, 352, 288, 160, 128, 128, 128, 128, 96, 96, 64, 64, 64] | [14976, 256, 256] | [13984] |
| group_6.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_6.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_6.compute | fwd=88.41, bwd=176.82 | step/layer=1.05, fwd/layer=4.45, bwd/layer=8.90 | fwd=45.87, bwd=91.74 | fwd=252.57, bwd=505.15 | fwd=219.85, bwd=439.69 |
| group_6.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_6.ring_comm | 0.00 | fwd_step=0.89, bwd_step=1.77, fwd/layer=4.45, bwd/layer=8.90 | 0.00 | 0.00 | 0.00 |
| group_6.time_model | additive, leakage=None | overlap, leakage=0.1 | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_6.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_7.strategy | ulysses×1 | ring×4 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_7.num_gpus | 1 | 4 | 1 | 1 | 1 |
| group_7.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=4, placement=context_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_7.estimated_time_ms | 210.53 | 707.18 | 678.59 | 378.32 | 417.25 |
| group_7.estimated_memory_gb | 61.95 | 33.45 | 68.88 | 71.97 | 49.57 |
| group_7.memory_breakdown_gb | model=7.00, act=54.95, pad=0.00 | model=7.00, act=26.45, pad=0.00 | model=7.00, act=61.88, pad=0.00 | model=7.00, act=64.97, pad=0.00 | model=7.00, act=42.57, pad=0.00 |
| group_7.seq_sum_local | seq_sum=14208, local=14208.00 | seq_sum=27360, local=6840.00 | seq_sum=16000, local=16000.00 | seq_sum=16800, local=16800.00 | seq_sum=11008, local=11008.00 |
| group_7.sequence_ids | [25, 28, 34, 53, 37, 17] | [59] | [50, 37, 60] | [11, 47, 20, 38] | [40] |
| group_7.sequence_lengths | [5376, 2848, 2400, 1728, 1600, 256] | [27360] | [13984, 1600, 416] | [7232, 6624, 1824, 1120] | [11008] |
| group_7.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_7.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_7.compute | fwd=70.18, bwd=140.35 | step/layer=1.78, fwd/layer=7.37, bwd/layer=14.73 | fwd=226.20, bwd=452.39 | fwd=126.11, bwd=252.22 | fwd=139.08, bwd=278.17 |
| group_7.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_7.ring_comm | 0.00 | fwd_step=0.86, bwd_step=1.72, fwd/layer=7.37, bwd/layer=14.73 | 0.00 | 0.00 | 0.00 |
| group_7.time_model | additive, leakage=None | overlap, leakage=0.1 | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_7.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_8.strategy | ulysses×4 |  | ring×2 | usp(sp2×cp4,cf) | ulysses×2 |
| group_8.num_gpus | 4 |  | 2 | 8 | 2 |
| group_8.sp_cp_placement | sp=4, cp=1, placement=head_first |  | sp=1, cp=2, placement=context_first | sp=2, cp=4, placement=context_first | sp=2, cp=1, placement=head_first |
| group_8.estimated_time_ms | 987.28 |  | 683.53 | 1285.15 | 517.62 |
| group_8.estimated_memory_gb | 33.45 |  | 45.18 | 29.77 | 35.96 |
| group_8.memory_breakdown_gb | model=7.00, act=26.45, pad=0.00 |  | model=7.00, act=38.18, pad=0.00 | model=7.00, act=22.77, pad=0.00 | model=7.00, act=28.96, pad=0.00 |
| group_8.seq_sum_local | seq_sum=27360, local=6840.00 |  | seq_sum=19744, local=9872.00 | seq_sum=47104, local=5888.00 | seq_sum=14976, local=7488.00 |
| group_8.sequence_ids | [59] |  | [24] | [59, 24] | [21] |
| group_8.sequence_lengths | [27360] |  | [19744] | [27360, 19744] | [14976] |
| group_8.topology | a2a=consecutive, ring=strided |  | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided |
| group_8.head_padding | q=1.0, kv=1.0 |  | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_8.compute | fwd=202.45, bwd=404.90 |  | step/layer=3.53, fwd/layer=7.12, bwd/layer=14.24 | step/layer=1.38 | fwd=125.43, bwd=250.85 |
| group_8.alltoall_comm | 379.93 |  | 0.00 | qo_op=4.74, kv_op=0.99, fwd/layer=11.45, bwd/layer=11.45 | 141.35 |
| group_8.ring_comm | 0.00 |  | fwd_step=0.51, bwd_step=1.03, fwd/layer=7.12, bwd/layer=14.24 | fwd_step=0.81, bwd_step=1.62, fwd/layer=5.75, bwd/layer=11.51 | 0.00 |
| group_8.time_model | additive, leakage=None |  | overlap, leakage=0.1 | overlap, leakage=0.1 | additive, leakage=None |
| group_8.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |  | sum_layers(fwd/bwd leaky overlap across ring steps) | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_9.strategy | ulysses×4 |  | ring×2 |  | ring×2 |
| group_9.num_gpus | 4 |  | 2 |  | 2 |
| group_9.sp_cp_placement | sp=4, cp=1, placement=head_first |  | sp=1, cp=2, placement=context_first |  | sp=1, cp=2, placement=context_first |
| group_9.estimated_time_ms | 671.92 |  | 519.45 |  | 683.53 |
| group_9.estimated_memory_gb | 26.09 |  | 49.94 |  | 45.18 |
| group_9.memory_breakdown_gb | model=7.00, act=19.09, pad=0.00 |  | model=7.00, act=42.94, pad=0.00 |  | model=7.00, act=38.18, pad=0.00 |
| group_9.seq_sum_local | seq_sum=19744, local=4936.00 |  | seq_sum=22208, local=11104.00 |  | seq_sum=19744, local=9872.00 |
| group_9.sequence_ids | [24] |  | [21, 11] |  | [24] |
| group_9.sequence_lengths | [19744] |  | [14976, 7232] |  | [19744] |
| group_9.topology | a2a=consecutive, ring=strided |  | a2a=strided, ring=consecutive |  | a2a=strided, ring=consecutive |
| group_9.head_padding | q=1.0, kv=1.0 |  | q=1.0, kv=1.0 |  | q=1.0, kv=1.0 |
| group_9.compute | fwd=107.09, bwd=214.19 |  | step/layer=2.68, fwd/layer=5.41, bwd/layer=10.82 |  | step/layer=3.53, fwd/layer=7.12, bwd/layer=14.24 |
| group_9.alltoall_comm | 350.64 |  | 0.00 |  | 0.00 |
| group_9.ring_comm | 0.00 |  | fwd_step=0.56, bwd_step=1.13, fwd/layer=5.41, bwd/layer=10.82 |  | fwd_step=0.51, bwd_step=1.03, fwd/layer=7.12, bwd/layer=14.24 |
| group_9.time_model | additive, leakage=None |  | overlap, leakage=0.1 |  | overlap, leakage=0.1 |
| group_9.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |  | sum_layers(fwd/bwd leaky overlap across ring steps) |  | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_10.strategy |  |  | ring×4 |  | ring×4 |
| group_10.num_gpus |  |  | 4 |  | 4 |
| group_10.sp_cp_placement |  |  | sp=1, cp=4, placement=context_first |  | sp=1, cp=4, placement=context_first |
| group_10.estimated_time_ms |  |  | 707.18 |  | 707.18 |
| group_10.estimated_memory_gb |  |  | 33.45 |  | 33.45 |
| group_10.memory_breakdown_gb |  |  | model=7.00, act=26.45, pad=0.00 |  | model=7.00, act=26.45, pad=0.00 |
| group_10.seq_sum_local |  |  | seq_sum=27360, local=6840.00 |  | seq_sum=27360, local=6840.00 |
| group_10.sequence_ids |  |  | [59] |  | [59] |
| group_10.sequence_lengths |  |  | [27360] |  | [27360] |
| group_10.topology |  |  | a2a=strided, ring=consecutive |  | a2a=strided, ring=consecutive |
| group_10.head_padding |  |  | q=1.0, kv=1.0 |  | q=1.0, kv=1.0 |
| group_10.compute |  |  | step/layer=1.78, fwd/layer=7.37, bwd/layer=14.73 |  | step/layer=1.78, fwd/layer=7.37, bwd/layer=14.73 |
| group_10.alltoall_comm |  |  | 0.00 |  | 0.00 |
| group_10.ring_comm |  |  | fwd_step=0.86, bwd_step=1.72, fwd/layer=7.37, bwd/layer=14.73 |  | fwd_step=0.86, bwd_step=1.72, fwd/layer=7.37, bwd/layer=14.73 |
| group_10.time_model |  |  | overlap, leakage=0.1 |  | overlap, leakage=0.1 |
| group_10.formula |  |  | sum_layers(fwd/bwd leaky overlap across ring steps) |  | sum_layers(fwd/bwd leaky overlap across ring steps) |
