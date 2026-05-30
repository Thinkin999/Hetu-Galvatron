# ILP Horizontal Comparison: `real_random_max131072_b4`

## Batch
| field | value |
| --- | --- |
| sampling_mode | real_random |
| max_seq | 131072 |
| global_batch_size | 64 |
| seq_sum | 115776 |
| max / p95 / p99 | 16960 / 5868.80 / 13472.32 |
| >=128k / >=256k / >=384k / >=512k | 0 / 0 / 0 / 0 |
| sequences | [0:64, 1:64, 2:1056, 3:864, 4:4032, 5:2112, 6:352, 7:544, 8:1856, 9:1504, 10:3904, 11:736, 12:160, 13:448, 14:1376, 15:1184, 16:160, 17:1376, 18:2528, 19:5888, 20:1088, 21:416, 22:288, 23:3040, 24:96, 25:1120, 26:1056, 27:1472, 28:96, 29:2144, 30:11424, 31:1888, 32:1280, 33:1408, 34:416, 35:160, 36:576, 37:2080, 38:224, 39:2304, 40:576, 41:320, 42:992, 43:416, 44:1248, 45:3360, 46:288, 47:16960, 48:5760, 49:384, 50:64, 51:512, 52:192, 53:2112, 54:1728, 55:2304, 56:1760, 57:192, 58:1664, 59:736, 60:544, 61:4192, 62:352, 63:6336] |

## Case Summary
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| status | ok | ok | ok | ok | ok |
| total_time_ms | 579.1737 | 310.2049 | 310.2049 | 545.1061 | 310.2049 |
| speedup_vs_ulysses | 1.0000 | 1.8671 | 1.8671 | 1.0625 | 1.8671 |
| microbatch_count | 1 | 1 | 1 | 1 | 1 |
| solver_wall_s | 2.8597 | 1.6246 | 2.9963 | 1.3637 | 2.9480 |

## Microbatch 0
| metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| estimated_time_ms | 579.17 | 310.20 | 310.20 | 545.11 | 310.20 |
| seq_sum | 115776 | 115776 | 115776 | 115776 | 115776 |
| num_sequences | 64 | 64 | 64 | 64 | 64 |
| sequences | [0:64, 1:64, 2:1056, 3:864, 4:4032, 5:2112, 6:352, 7:544, 8:1856, 9:1504, 10:3904, 11:736, 12:160, 13:448, 14:1376, 15:1184, 16:160, 17:1376, 18:2528, 19:5888, 20:1088, 21:416, 22:288, 23:3040, ...(+40)] | [0:64, 1:64, 2:1056, 3:864, 4:4032, 5:2112, 6:352, 7:544, 8:1856, 9:1504, 10:3904, 11:736, 12:160, 13:448, 14:1376, 15:1184, 16:160, 17:1376, 18:2528, 19:5888, 20:1088, 21:416, 22:288, 23:3040, ...(+40)] | [0:64, 1:64, 2:1056, 3:864, 4:4032, 5:2112, 6:352, 7:544, 8:1856, 9:1504, 10:3904, 11:736, 12:160, 13:448, 14:1376, 15:1184, 16:160, 17:1376, 18:2528, 19:5888, 20:1088, 21:416, 22:288, 23:3040, ...(+40)] | [0:64, 1:64, 2:1056, 3:864, 4:4032, 5:2112, 6:352, 7:544, 8:1856, 9:1504, 10:3904, 11:736, 12:160, 13:448, 14:1376, 15:1184, 16:160, 17:1376, 18:2528, 19:5888, 20:1088, 21:416, 22:288, 23:3040, ...(+40)] | [0:64, 1:64, 2:1056, 3:864, 4:4032, 5:2112, 6:352, 7:544, 8:1856, 9:1504, 10:3904, 11:736, 12:160, 13:448, 14:1376, 15:1184, 16:160, 17:1376, 18:2528, 19:5888, 20:1088, 21:416, 22:288, 23:3040, ...(+40)] |

| group metric | ulysses_only | ring_only | ulysses_ring | usp_only | ulysses_ring_usp |
| --- | --- | --- | --- | --- | --- |
| group_0.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_0.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_0.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_0.estimated_time_ms | 102.92 | 137.62 | 183.08 | 232.09 | 151.70 |
| group_0.estimated_memory_gb | 47.59 | 64.30 | 65.66 | 69.62 | 69.62 |
| group_0.memory_breakdown_gb | model=7.00, act=40.59, pad=0.00 | model=7.00, act=57.30, pad=0.00 | model=7.00, act=58.66, pad=0.00 | model=7.00, act=62.62, pad=0.00 | model=7.00, act=62.62, pad=0.00 |
| group_0.seq_sum_local | seq_sum=10496, local=10496.00 | seq_sum=14816, local=14816.00 | seq_sum=15168, local=15168.00 | seq_sum=16192, local=16192.00 | seq_sum=16192, local=16192.00 |
| group_0.sequence_ids | [29, 53, 15, 25, 20, 3, 43, 49, 62, 41, 46, 16, 0] | [39, 56, 32, 20, 26, 42, 11, 59, 36, 40, 51, 13, 34, 49, 6, 62, 41, 22, 46, 52, 12] | [4, 18, 53, 56, 54, 9, 7, 13, 6, 35] | [48, 39, 53, 8, 54, 44, 15] | [53, 14, 20, 26, 42, 3, 11, 59, 36, 40, 7, 60, 13, 21, 34, 43, 49, 6, 62, 41, 22, 46, 38, 52, ...(+8)] |
| group_0.sequence_lengths | [2144, 2112, 1184, 1120, 1088, 864, 416, 384, 352, 320, 288, 160, 64] | [2304, 1760, 1280, 1088, 1056, 992, 736, 736, 576, 576, 512, 448, 416, 384, 352, 352, 320, 288, 288, 192, 160] | [4032, 2528, 2112, 1760, 1728, 1504, 544, 448, 352, 160] | [5760, 2304, 2112, 1856, 1728, 1248, 1184] | [2112, 1376, 1088, 1056, 992, 864, 736, 736, 576, 576, 544, 544, 448, 416, 416, 416, 384, 352, 352, 320, 288, 288, 224, 192, ...(+8)] |
| group_0.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_0.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_0.compute | fwd=34.31, bwd=68.61 | fwd=45.87, bwd=91.75 | fwd=61.03, bwd=122.05 | fwd=77.36, bwd=154.73 | fwd=50.57, bwd=101.13 |
| group_0.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_0.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_0.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_0.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_1.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_1.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_1.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_1.estimated_time_ms | 165.73 | 150.49 | 147.40 | 156.94 | 203.51 |
| group_1.estimated_memory_gb | 51.43 | 63.92 | 66.65 | 66.03 | 69.87 |
| group_1.memory_breakdown_gb | model=7.00, act=44.43, pad=0.00 | model=7.00, act=56.92, pad=0.00 | model=7.00, act=59.65, pad=0.00 | model=7.00, act=59.03, pad=0.00 | model=7.00, act=62.87, pad=0.00 |
| group_1.seq_sum_local | seq_sum=11488, local=11488.00 | seq_sum=14720, local=14720.00 | seq_sum=15424, local=15424.00 | seq_sum=15264, local=15264.00 | seq_sum=16256, local=16256.00 |
| group_1.sequence_ids | [4, 10, 37, 27] | [23, 37, 31, 17, 44, 15, 25, 3, 7, 60, 21, 43] | [39, 8, 58, 33, 17, 44, 20, 3, 11, 36, 40, 34, 43, 62, 46, 52, 50] | [23, 18, 20, 11, 59, 36, 40, 7, 60, 51, 13, 21, 34, 43, 49, 6, 62, 41, 22, 46, 52, 57, 12, 35] | [4, 23, 29, 5, 8, 58, 44, 16] |
| group_1.sequence_lengths | [4032, 3904, 2080, 1472] | [3040, 2080, 1888, 1376, 1248, 1184, 1120, 864, 544, 544, 416, 416] | [2304, 1856, 1664, 1408, 1376, 1248, 1088, 864, 736, 576, 576, 416, 416, 352, 288, 192, 64] | [3040, 2528, 1088, 736, 736, 576, 576, 544, 544, 512, 448, 416, 416, 416, 384, 352, 352, 320, 288, 288, 192, 192, 160, 160] | [4032, 3040, 2144, 2112, 1856, 1664, 1248, 160] |
| group_1.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_1.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_1.compute | fwd=55.24, bwd=110.49 | fwd=50.16, bwd=100.33 | fwd=49.13, bwd=98.27 | fwd=52.31, bwd=104.62 | fwd=67.84, bwd=135.67 |
| group_1.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_1.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_1.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_1.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_2.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_2.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_2.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_2.estimated_time_ms | 75.12 | 205.92 | 205.06 | 242.74 | 200.41 |
| group_2.estimated_memory_gb | 33.85 | 65.78 | 69.74 | 71.60 | 45.98 |
| group_2.memory_breakdown_gb | model=7.00, act=26.85, pad=0.00 | model=7.00, act=58.78, pad=0.00 | model=7.00, act=62.74, pad=0.00 | model=7.00, act=64.60, pad=0.00 | model=7.00, act=38.98, pad=0.00 |
| group_2.seq_sum_local | seq_sum=6944, local=6944.00 | seq_sum=15200, local=15200.00 | seq_sum=16224, local=16224.00 | seq_sum=16704, local=16704.00 | seq_sum=10080, local=10080.00 |
| group_2.sequence_ids | [31, 8, 54, 22, 38, 52, 57, 12, 35, 24, 28, 1] | [4, 10, 18, 8, 9, 14] | [10, 45, 55, 37, 31, 32, 42, 41, 28] | [63, 58, 9, 27, 14, 17, 2, 26, 3] | [19, 61] |
| group_2.sequence_lengths | [1888, 1856, 1728, 288, 224, 192, 192, 160, 160, 96, 96, 64] | [4032, 3904, 2528, 1856, 1504, 1376] | [3904, 3360, 2304, 2080, 1888, 1280, 992, 320, 96] | [6336, 1664, 1504, 1472, 1376, 1376, 1056, 1056, 864] | [5888, 4192] |
| group_2.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_2.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_2.compute | fwd=25.04, bwd=50.08 | fwd=68.64, bwd=137.28 | fwd=68.35, bwd=136.71 | fwd=80.91, bwd=161.82 | fwd=66.80, bwd=133.61 |
| group_2.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_2.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_2.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_2.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_3.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_3.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_3.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_3.estimated_time_ms | 12.69 | 297.88 | 131.63 | 503.99 | 154.70 |
| group_3.estimated_memory_gb | 12.07 | 69.87 | 53.16 | 71.84 | 41.40 |
| group_3.memory_breakdown_gb | model=7.00, act=5.07, pad=0.00 | model=7.00, act=62.87, pad=0.00 | model=7.00, act=46.16, pad=0.00 | model=7.00, act=64.84, pad=0.00 | model=7.00, act=34.40, pad=0.00 |
| group_3.seq_sum_local | seq_sum=1312, local=1312.00 | seq_sum=16256, local=16256.00 | seq_sum=11936, local=11936.00 | seq_sum=16768, local=16768.00 | seq_sum=8896, local=8896.00 |
| group_3.sequence_ids | [44, 50] | [19, 48, 54, 27, 33] | [23, 29, 5, 2, 59, 60, 51, 21, 49, 22, 38, 57, 12, 0, 1] | [30, 29, 37, 25] | [48, 56, 17] |
| group_3.sequence_lengths | [1248, 64] | [5888, 5760, 1728, 1472, 1408] | [3040, 2144, 2112, 1056, 736, 544, 512, 416, 384, 288, 224, 192, 160, 64, 64] | [11424, 2144, 2080, 1120] | [5760, 1760, 1376] |
| group_3.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_3.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_3.compute | fwd=4.23, bwd=8.46 | fwd=99.29, bwd=198.59 | fwd=43.88, bwd=87.75 | fwd=168.00, bwd=335.99 | fwd=51.57, bwd=103.13 |
| group_3.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_3.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_3.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_3.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_4.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_4.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_4.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_4.estimated_time_ms | 38.90 | 213.28 | 92.53 | 103.84 | 83.90 |
| group_4.estimated_memory_gb | 23.95 | 50.93 | 32.62 | 37.07 | 34.60 |
| group_4.memory_breakdown_gb | model=7.00, act=16.95, pad=0.00 | model=7.00, act=43.93, pad=0.00 | model=7.00, act=25.62, pad=0.00 | model=7.00, act=30.07, pad=0.00 | model=7.00, act=27.60, pad=0.00 |
| group_4.seq_sum_local | seq_sum=4384, local=4384.00 | seq_sum=11360, local=11360.00 | seq_sum=6624, local=6624.00 | seq_sum=7776, local=7776.00 | seq_sum=7136, local=7136.00 |
| group_4.sequence_ids | [33, 17, 59, 51, 6] | [63, 45, 58] | [61, 14, 26] | [10, 5, 56] | [18, 39, 55] |
| group_4.sequence_lengths | [1408, 1376, 736, 512, 352] | [6336, 3360, 1664] | [4192, 1376, 1056] | [3904, 2112, 1760] | [2528, 2304, 2304] |
| group_4.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_4.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_4.compute | fwd=12.97, bwd=25.93 | fwd=71.09, bwd=142.18 | fwd=30.84, bwd=61.69 | fwd=34.61, bwd=69.23 | fwd=27.97, bwd=55.93 |
| group_4.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_4.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_4.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_4.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_5.strategy | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 | ulysses×1 |
| group_5.num_gpus | 1 | 1 | 1 | 1 | 1 |
| group_5.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_5.estimated_time_ms | 469.37 | 129.78 | 37.64 | 46.26 | 17.29 |
| group_5.estimated_memory_gb | 60.58 | 44.37 | 22.59 | 23.21 | 13.68 |
| group_5.memory_breakdown_gb | model=7.00, act=53.58, pad=0.00 | model=7.00, act=37.37, pad=0.00 | model=7.00, act=15.59, pad=0.00 | model=7.00, act=16.21, pad=0.00 | model=7.00, act=6.68, pad=0.00 |
| group_5.seq_sum_local | seq_sum=13856, local=13856.00 | seq_sum=9664, local=9664.00 | seq_sum=4032, local=4032.00 | seq_sum=4192, local=4192.00 | seq_sum=1728, local=1728.00 |
| group_5.sequence_ids | [30, 14, 2] | [61, 55, 53, 2] | [27, 15, 25, 16, 24] | [55, 31] | [54] |
| group_5.sequence_lengths | [11424, 1376, 1056] | [4192, 2304, 2112, 1056] | [1472, 1184, 1120, 160, 96] | [2304, 1888] | [1728] |
| group_5.topology | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_5.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_5.compute | fwd=156.46, bwd=312.91 | fwd=43.26, bwd=86.52 | fwd=12.55, bwd=25.09 | fwd=15.42, bwd=30.84 | fwd=5.76, bwd=11.53 |
| group_5.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_5.ring_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_5.time_model | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None | additive, leakage=None |
| group_5.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_6.strategy | ulysses×1 | ring×2 | ring×2 | ulysses×1 | ulysses×1 |
| group_6.num_gpus | 1 | 2 | 2 | 1 | 1 |
| group_6.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first | sp=1, cp=2, placement=context_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_6.estimated_time_ms | 37.70 | 248.15 | 248.15 | 270.40 | 35.31 |
| group_6.estimated_memory_gb | 21.85 | 29.09 | 29.09 | 65.78 | 22.10 |
| group_6.memory_breakdown_gb | model=7.00, act=14.85, pad=0.00 | model=7.00, act=22.09, pad=0.00 | model=7.00, act=22.09, pad=0.00 | model=7.00, act=58.78, pad=0.00 | model=7.00, act=15.10, pad=0.00 |
| group_6.seq_sum_local | seq_sum=3840, local=3840.00 | seq_sum=11424, local=5712.00 | seq_sum=11424, local=5712.00 | seq_sum=15200, local=15200.00 | seq_sum=3904, local=3904.00 |
| group_6.sequence_ids | [56, 58, 34] | [30] | [30] | [19, 61, 45, 42, 38, 16, 24, 28, 0, 1, 50] | [9, 32, 25] |
| group_6.sequence_lengths | [1760, 1664, 416] | [11424] | [11424] | [5888, 4192, 3360, 992, 224, 160, 96, 96, 64, 64, 64] | [1504, 1280, 1120] |
| group_6.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_6.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_6.compute | fwd=12.57, bwd=25.13 | step/layer=1.28, fwd/layer=2.58, bwd/layer=5.17 | step/layer=1.28, fwd/layer=2.58, bwd/layer=5.17 | fwd=90.13, bwd=180.26 | fwd=11.77, bwd=23.54 |
| group_6.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_6.ring_comm | 0.00 | fwd_step=0.34, bwd_step=0.67, fwd/layer=2.58, bwd/layer=5.17 | fwd_step=0.34, bwd_step=0.67, fwd/layer=2.58, bwd/layer=5.17 | 0.00 | 0.00 |
| group_6.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 | additive, leakage=None | additive, leakage=None |
| group_6.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_7.strategy | ulysses×1 | ring×2 | ring×2 | ulysses×1 | ulysses×1 |
| group_7.num_gpus | 1 | 2 | 2 | 1 | 1 |
| group_7.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first | sp=1, cp=2, placement=context_first | sp=1, cp=1, placement=head_first | sp=1, cp=1, placement=head_first |
| group_7.estimated_time_ms | 49.61 | 48.64 | 90.69 | 90.81 | 8.97 |
| group_7.estimated_memory_gb | 29.40 | 13.25 | 19.25 | 32.99 | 11.08 |
| group_7.memory_breakdown_gb | model=7.00, act=22.40, pad=0.00 | model=7.00, act=6.25, pad=0.00 | model=7.00, act=12.25, pad=0.00 | model=7.00, act=25.99, pad=0.00 | model=7.00, act=4.08, pad=0.00 |
| group_7.seq_sum_local | seq_sum=5792, local=5792.00 | seq_sum=3232, local=1616.00 | seq_sum=6336, local=3168.00 | seq_sum=6720, local=6720.00 | seq_sum=1056, local=1056.00 |
| group_7.sequence_ids | [9, 42, 11, 36, 40, 7, 13, 21] | [5, 38, 57, 16, 35, 24, 28, 0, 1, 50] | [63] | [4, 33, 32] | [2] |
| group_7.sequence_lengths | [1504, 992, 736, 576, 576, 544, 448, 416] | [2112, 224, 192, 160, 160, 96, 96, 64, 64, 64] | [6336] | [4032, 1408, 1280] | [1056] |
| group_7.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided | a2a=consecutive, ring=strided |
| group_7.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_7.compute | fwd=16.54, bwd=33.07 | step/layer=0.25, fwd/layer=0.51, bwd/layer=1.01 | step/layer=0.46, fwd/layer=0.94, bwd/layer=1.89 | fwd=30.27, bwd=60.54 | fwd=2.99, bwd=5.98 |
| group_7.alltoall_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| group_7.ring_comm | 0.00 | fwd_step=0.16, bwd_step=0.33, fwd/layer=0.51, bwd/layer=1.01 | fwd_step=0.23, bwd_step=0.46, fwd/layer=0.94, bwd/layer=1.89 | 0.00 | 0.00 |
| group_7.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 | additive, leakage=None | additive, leakage=None |
| group_7.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_8.strategy | ulysses×1 | ring×2 | ring×2 | usp(sp2×cp4,cf) | ulysses×1 |
| group_8.num_gpus | 1 | 2 | 2 | 8 | 1 |
| group_8.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=2, placement=context_first | sp=1, cp=2, placement=context_first | sp=2, cp=4, placement=context_first | sp=1, cp=1, placement=head_first |
| group_8.estimated_time_ms | 231.15 | 23.62 | 157.47 | 545.11 | 53.74 |
| group_8.estimated_memory_gb | 55.76 | 11.15 | 29.52 | 15.20 | 28.16 |
| group_8.memory_breakdown_gb | model=7.00, act=48.76, pad=0.00 | model=7.00, act=4.15, pad=0.00 | model=7.00, act=22.52, pad=0.00 | model=7.00, act=8.20, pad=0.00 | model=7.00, act=21.16, pad=0.00 |
| group_8.seq_sum_local | seq_sum=12608, local=12608.00 | seq_sum=2144, local=1072.00 | seq_sum=11648, local=5824.00 | seq_sum=16960, local=2120.00 | seq_sum=5472, local=5472.00 |
| group_8.sequence_ids | [19, 61, 18] | [29] | [19, 48] | [47] | [37, 27, 33, 51] |
| group_8.sequence_lengths | [5888, 4192, 2528] | [2144] | [5888, 5760] | [16960] | [2080, 1472, 1408, 512] |
| group_8.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive | a2a=consecutive, ring=strided |
| group_8.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |
| group_8.compute | fwd=77.05, bwd=154.10 | step/layer=0.10, fwd/layer=0.25, bwd/layer=0.49 | step/layer=0.80, fwd/layer=1.64, bwd/layer=3.28 | step/layer=0.38 | fwd=17.91, bwd=35.83 |
| group_8.alltoall_comm | 0.00 | 0.00 | 0.00 | qo_op=1.94, kv_op=0.59, fwd/layer=5.06, bwd/layer=5.06 | 0.00 |
| group_8.ring_comm | 0.00 | fwd_step=0.14, bwd_step=0.28, fwd/layer=0.25, bwd/layer=0.49 | fwd_step=0.34, bwd_step=0.68, fwd/layer=1.64, bwd/layer=3.28 | fwd_step=0.61, bwd_step=1.21, fwd/layer=2.31, bwd/layer=4.61 | 0.00 |
| group_8.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 | overlap, leakage=0.1 | additive, leakage=None |
| group_8.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) | layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap) | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_9.strategy | ulysses×1 | ring×4 | ring×4 |  | ulysses×1 |
| group_9.num_gpus | 1 | 4 | 4 |  | 1 |
| group_9.sp_cp_placement | sp=1, cp=1, placement=head_first | sp=1, cp=4, placement=context_first | sp=1, cp=4, placement=context_first |  | sp=1, cp=1, placement=head_first |
| group_9.estimated_time_ms | 124.34 | 310.20 | 310.20 |  | 289.73 |
| group_9.estimated_memory_gb | 29.27 | 23.40 | 23.40 |  | 71.47 |
| group_9.memory_breakdown_gb | model=7.00, act=22.27, pad=0.00 | model=7.00, act=16.40, pad=0.00 | model=7.00, act=16.40, pad=0.00 |  | model=7.00, act=64.47, pad=0.00 |
| group_9.seq_sum_local | seq_sum=5760, local=5760.00 | seq_sum=16960, local=4240.00 | seq_sum=16960, local=4240.00 |  | seq_sum=16672, local=16672.00 |
| group_9.sequence_ids | [48] | [47] | [47] |  | [63, 10, 45, 31, 15] |
| group_9.sequence_lengths | [5760] | [16960] | [16960] |  | [6336, 3904, 3360, 1888, 1184] |
| group_9.topology | a2a=consecutive, ring=strided | a2a=strided, ring=consecutive | a2a=strided, ring=consecutive |  | a2a=consecutive, ring=strided |
| group_9.head_padding | q=1.0, kv=1.0 | q=1.0, kv=1.0 | q=1.0, kv=1.0 |  | q=1.0, kv=1.0 |
| group_9.compute | fwd=41.45, bwd=82.90 | step/layer=0.75, fwd/layer=3.23, bwd/layer=6.46 | step/layer=0.75, fwd/layer=3.23, bwd/layer=6.46 |  | fwd=96.58, bwd=193.15 |
| group_9.alltoall_comm | 0.00 | 0.00 | 0.00 |  | 0.00 |
| group_9.ring_comm | 0.00 | fwd_step=0.72, bwd_step=1.44, fwd/layer=3.23, bwd/layer=6.46 | fwd_step=0.72, bwd_step=1.44, fwd/layer=3.23, bwd/layer=6.46 |  | 0.00 |
| group_9.time_model | additive, leakage=None | overlap, leakage=0.1 | overlap, leakage=0.1 |  | additive, leakage=None |
| group_9.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm | sum_layers(fwd/bwd leaky overlap across ring steps) | sum_layers(fwd/bwd leaky overlap across ring steps) |  | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |
| group_10.strategy | ulysses×1 |  |  |  | ring×2 |
| group_10.num_gpus | 1 |  |  |  | 2 |
| group_10.sp_cp_placement | sp=1, cp=1, placement=head_first |  |  |  | sp=1, cp=2, placement=context_first |
| group_10.estimated_time_ms | 186.14 |  |  |  | 248.15 |
| group_10.estimated_memory_gb | 45.36 |  |  |  | 29.09 |
| group_10.memory_breakdown_gb | model=7.00, act=38.36, pad=0.00 |  |  |  | model=7.00, act=22.09, pad=0.00 |
| group_10.seq_sum_local | seq_sum=9920, local=9920.00 |  |  |  | seq_sum=11424, local=5712.00 |
| group_10.sequence_ids | [63, 55, 32] |  |  |  | [30] |
| group_10.sequence_lengths | [6336, 2304, 1280] |  |  |  | [11424] |
| group_10.topology | a2a=consecutive, ring=strided |  |  |  | a2a=strided, ring=consecutive |
| group_10.head_padding | q=1.0, kv=1.0 |  |  |  | q=1.0, kv=1.0 |
| group_10.compute | fwd=62.05, bwd=124.09 |  |  |  | step/layer=1.28, fwd/layer=2.58, bwd/layer=5.17 |
| group_10.alltoall_comm | 0.00 |  |  |  | 0.00 |
| group_10.ring_comm | 0.00 |  |  |  | fwd_step=0.34, bwd_step=0.67, fwd/layer=2.58, bwd/layer=5.17 |
| group_10.time_model | additive, leakage=None |  |  |  | overlap, leakage=0.1 |
| group_10.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |  |  |  | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_11.strategy | ulysses×1 |  |  |  | ring×4 |
| group_11.num_gpus | 1 |  |  |  | 4 |
| group_11.sp_cp_placement | sp=1, cp=1, placement=head_first |  |  |  | sp=1, cp=4, placement=context_first |
| group_11.estimated_time_ms | 153.38 |  |  |  | 310.20 |
| group_11.estimated_memory_gb | 55.02 |  |  |  | 23.40 |
| group_11.memory_breakdown_gb | model=7.00, act=48.02, pad=0.00 |  |  |  | model=7.00, act=16.40, pad=0.00 |
| group_11.seq_sum_local | seq_sum=12416, local=12416.00 |  |  |  | seq_sum=16960, local=4240.00 |
| group_11.sequence_ids | [45, 23, 39, 5, 26, 60] |  |  |  | [47] |
| group_11.sequence_lengths | [3360, 3040, 2304, 2112, 1056, 544] |  |  |  | [16960] |
| group_11.topology | a2a=consecutive, ring=strided |  |  |  | a2a=strided, ring=consecutive |
| group_11.head_padding | q=1.0, kv=1.0 |  |  |  | q=1.0, kv=1.0 |
| group_11.compute | fwd=51.13, bwd=102.25 |  |  |  | step/layer=0.75, fwd/layer=3.23, bwd/layer=6.46 |
| group_11.alltoall_comm | 0.00 |  |  |  | 0.00 |
| group_11.ring_comm | 0.00 |  |  |  | fwd_step=0.72, bwd_step=1.44, fwd/layer=3.23, bwd/layer=6.46 |
| group_11.time_model | additive, leakage=None |  |  |  | overlap, leakage=0.1 |
| group_11.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |  |  |  | sum_layers(fwd/bwd leaky overlap across ring steps) |
| group_12.strategy | ulysses×4 |  |  |  |  |
| group_12.num_gpus | 4 |  |  |  |  |
| group_12.sp_cp_placement | sp=4, cp=1, placement=head_first |  |  |  |  |
| group_12.estimated_time_ms | 579.17 |  |  |  |  |
| group_12.estimated_memory_gb | 23.40 |  |  |  |  |
| group_12.memory_breakdown_gb | model=7.00, act=16.40, pad=0.00 |  |  |  |  |
| group_12.seq_sum_local | seq_sum=16960, local=4240.00 |  |  |  |  |
| group_12.sequence_ids | [47] |  |  |  |  |
| group_12.sequence_lengths | [16960] |  |  |  |  |
| group_12.topology | a2a=consecutive, ring=strided |  |  |  |  |
| group_12.head_padding | q=1.0, kv=1.0 |  |  |  |  |
| group_12.compute | fwd=79.75, bwd=159.49 |  |  |  |  |
| group_12.alltoall_comm | 339.93 |  |  |  |  |
| group_12.ring_comm | 0.00 |  |  |  |  |
| group_12.time_model | additive, leakage=None |  |  |  |  |
| group_12.formula | compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm |  |  |  |  |
