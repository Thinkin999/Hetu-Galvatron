import os
import json
import numpy as np
from scipy.optimize import nnls
import sys
import cvxpy as cp
from typing import Dict,Any, List
REPO_ROOT = "/home/pkuhetu/lqs"
sys.path.append(os.path.join(REPO_ROOT, "galvatron_lxy/Hetu-Galvatron"))

from galvatron.core.search_engine.cost_model import TimeCostModel
from galvatron.core.search_engine.cost_model_args import (
    ModelArgs, TrainArgs, ParallelArgs, ProfileModelArgs, ProfileHardwareArgs
)



def fit_sequence_quadratic_nonneg():
    prof_path = os.path.join(
        REPO_ROOT,
        "galvatron_lxy/Hetu-Galvatron/galvatron/models/llama_hf/configs/"
        "computation_profiling_bf16_qwen2.5-3b.json"
    )
    with open(prof_path) as f:
        data = json.load(f)

    xs, ys = [], []
    for k, v in data.items():
        if k.startswith("layertype_0_bsz1_seq"):
            seq = int(k.split("seq")[1])
            xs.append(seq)
            ys.append(float(v))
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    # 构造 Vandermonde 矩阵  [x^2, x, 1]
    A = np.vstack([xs**2, xs, np.ones_like(xs)]).T
    b = ys
    # nnls 只接受非负，且最小二乘
    coeffs, rnorm = nnls(A, b)
    a_opt, b_opt, c_opt = map(float, coeffs)
    print(f"a: {a_opt:e}, b: {b_opt:e}, c: {c_opt:e}")
    return {"mode": "sequence_quadratic", "popt": [a_opt, b_opt, c_opt]}
def fit_all2all_linear_from_sp_time(sp_time_json_path: str, group_size: int) -> Dict[str, Any]:
    
    assert os.path.exists(sp_time_json_path), f"File not found: {sp_time_json_path}"
    with open(sp_time_json_path, "r") as f:
        data = json.load(f)

    prefix = f"all2all_size_{group_size}_"
    xs_mb: List[float] = []
    ys_ms: List[float] = []

    for k, v in data.items():
        # 形如: all2all_size_8_32MB_time
        if not k.startswith(prefix) or not k.endswith("_time"):
            continue
        tail = k[len(prefix):]                 # e.g. "32MB_time"
        mb_str = tail.split("MB_")[0]          # "32"
        try:
            mb_val = float(mb_str)
            time_ms = float(v)                 # 值即为毫秒
            xs_mb.append(mb_val)
            ys_ms.append(time_ms)
        except Exception:
            continue

    assert len(xs_mb) >= 2, f"Not enough all2all points for group_size={group_size}"

    xs = np.array(xs_mb, dtype=np.float64)
    ys = np.array(ys_ms, dtype=np.float64)

    # 线性拟合: time_ms = a * message_mb + b
    a, b = np.polyfit(xs, ys, deg=1).tolist()

    return [a, b]
       



def cp_metrics(seq_len: int, head_dim: int,num_attention_heads:int,num_query_groups:int,bsz: int, cp_size: int, a: float, b: float, c: float,
                mixed_precision = True):
    MB = 1024.0 * 1024.0
    pp_bandwidth = 78.0232 * 1.024
    bytes_kv = 2 if mixed_precision else 4
    bytes_fp32 = 4
   
    x = seq_len 
    # compute per-step (与 cost_model 一致)：seq->seq/p，再乘 (bsz/tp)
    comp_fwd = (a * x * x + b * x + c) /cp_size
    comp_bwd = comp_fwd * 2
   
    Q_elems_per_step = (bsz * seq_len * head_dim * num_attention_heads) / cp_size
    K_elems_per_step = (bsz * seq_len * head_dim * num_query_groups) / cp_size
    V_elems_per_step = (bsz * seq_len * head_dim * num_query_groups) / cp_size
   
    # 前向：以 K/V 大小为主
    comm_fwd_ms = ( K_elems_per_step + V_elems_per_step) * bytes_kv / MB / pp_bandwidth * (cp_size - 1)
    # 反向首段：K/V 相关
    comm_bwd_0_ms = ( K_elems_per_step + V_elems_per_step) * bytes_kv / MB / pp_bandwidth
    # 反向中段/尾段：以 Q/O 梯度大小为主
    comm_bwd_mid_ms = 2*( K_elems_per_step + V_elems_per_step) * bytes_fp32 / MB / pp_bandwidth * (cp_size - 1)
    comm_bwd_tail_ms = ( K_elems_per_step + V_elems_per_step) * bytes_fp32 / MB / pp_bandwidth
    comm_bwd_ms = comm_bwd_0_ms + comm_bwd_mid_ms + comm_bwd_tail_ms
    # overlap 函数（等价于 cost_model 的 overlap_comp_comm_time）
    def overlap_comp_comm_time(comp, comm):
        if comp >= comm:
            return comp + 0.1 * comm
        return comm

    # forward 总时间（每层）
    fwd_per_layer_ms = overlap_comp_comm_time(comp_fwd, comm_fwd_ms)
    
    bwd_per_layer_ms = overlap_comp_comm_time(comp_bwd, comm_bwd_ms)
   
    total_ms = (fwd_per_layer_ms + bwd_per_layer_ms) 
    comm = comm_fwd_ms + comm_bwd_ms
    return  comm, total_ms


# Ulysses 手工建模：前向二次/ sp_size、反向2x、8次 all-to-all（bytes/带宽）

from typing import Tuple

def ulysses_metrics(
    seq_len: int,
    head_dim: int,
    num_attention_heads:int,
    num_query_groups:int,
    bsz: int,
    a_comp: float,
    b_comp: float,
    c_comp: float,
    sp_size: int,
    a_comm: float,
    b_comm: float,
    mixed_precision: bool = True
) -> Tuple[float, float, float]:
   
    p = float(sp_size)
    b = float(bsz)
    s = float(seq_len)
    head_dim = float(head_dim)
    elem_bytes = 2.0 if mixed_precision else 4.0

    # 计算时间（每层）
    fwd_ms_raw = (a_comp * s * s + b_comp * s + c_comp) / p
    bwd_ms_raw = 2.0 * fwd_ms_raw
    comp_ms = fwd_ms_raw + bwd_ms_raw

    # Ulysses 通信元素量（FWD+BWD 合计）
    q_elems = 2.0 * b * s * head_dim * num_attention_heads * (p - 1.0) / (p * p)
    k_elems = 2.0 * b * s * head_dim * (p - 1.0) / p
    v_elems = 2.0 * b * s * head_dim *(p - 1.0) / p
    o_elems = 2.0 * b * s * head_dim * num_attention_heads * (p - 1.0) / (p * p)

    total_elems = q_elems + k_elems + v_elems + o_elems
    total_mb = total_elems * elem_bytes / 1024.0 / 1024.0

    # all2all 线性模型求通信时间（MB 与 ms）
    comm_ms = a_comm * total_mb + b_comm * 8

    total_ms = comp_ms + comm_ms
    return comp_ms, comm_ms, total_ms


def main():
    a, b, c = fit_sequence_quadratic_nonneg()
# def main():
#     # #记录普通的单卡前向时间的，已经验证了可以正确计算
#     # prof_path = os.path.join(REPO_ROOT, "galvatron_lxy/Hetu-Galvatron/galvatron/models/llama_hf/configs/computation_profiling_bf16_qwen2.5-3b.json")
#     # with open(prof_path, "r") as f:
#     #     data = json.load(f)
#     # real_results = {}   
#     # for k, v in data.items():
#     #     if k.startswith("layertype_0_bsz1_seq"):
#     #         seq = int(k.split("seq")[1])
#     #         real_results[seq] = float(v)
    
#     #a,b,c = fit_sequence_quadratic()["popt"]
#     a = 2.1930476567007057e-09 
#     b = 0
#     c = 0
#     #b = 0.0011911824608909084
#     #c = -1.1176326223782225
#     # for i in range(2048,16384 + 2048,2048):
#     #     time = a*i*i+b*i+c
#     #     diff = (real_results[i] - time)/real_results[i] 
#     #     print("seq = ", i,"fitting time = ", time,"real time = ",real_results[i] ,"diff = ",diff)
#     a_comm, b_comm = fit_all2all_linear_from_sp_time("/home/pkuhetu/lqs/galvatron_lxy/Hetu-Galvatron/galvatron/profile_hardware/hardware_configs/sp_time_1nodes_8gpus_per_node.json",8)
#     print("a_comm = ",a_comm,"b_comm = ",b_comm)
#     for i in range(4096,65536 + 2048,2048):
#         ulysses_comp_ms, ulysses_comm_ms, ulysses_total_ms = ulysses_metrics(i,128,16,2,1,a,b,c,8,a_comm,b_comm)
#         ulysses_diff = ulysses_comm_ms / ulysses_total_ms 

#         cp_comm,cp_total_ms = cp_metrics(i, 128,16,2,1,8,a,b,c)
#         cp_better = ulysses_total_ms > cp_total_ms 
#         print("seq = ", i,"ulysses total time = ", ulysses_total_ms,"communication part = ",ulysses_diff,"cp_total_ms",cp_total_ms,"cp better",cp_better)

#     ###ulysses 通信量计算

if __name__ == "__main__":
    main()
