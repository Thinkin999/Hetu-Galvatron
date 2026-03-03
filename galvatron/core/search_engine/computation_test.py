def calculate_s_with_replication(c, p, b, d, n_h, B_a2a, n_kv, coe, B_p2p):
    """
    计算两函数交点的横坐标s
    
    参数说明：
    c, p, b, d, n_h, B_a2a, n_kv, coe, B_p2p -- 公式中的各个参数（需为数值型）
    
    返回值：
    s -- 交点的横坐标，若分母为0则返回None并提示错误
    """
    # 计算分母部分
    denominator = -b * p + (4 * d * (n_h + p)) / B_a2a - (4 * d * n_kv * (coe - 1) * p**2) / B_p2p
    
    # 避免除零错误
    if denominator == 0:
        print("错误：分母为0，无唯一解（可能无交点或无穷多交点）")
        return None
    
    # 计算s的值
    s = (c * p) / denominator
    return s
def calculate_s_without_replication(c, p, b, d, n_h, B_a2a,n_kv, coe, B_p2p):
    """
    计算新公式下交点的横坐标s
    
    参数说明：
    c, p, b, d, n_h, n_kv, B_a2a, coe, B_p2p -- 公式中的各个参数（需为数值型）
    
    返回值：
    s -- 交点的横坐标，若分母为0则返回None并提示错误
    """
    # 计算分子部分
    numerator = c * (p + 1) * p
    
    # 计算分母部分
    denominator = -b * p + (4 * d * (n_h + n_kv)) / B_a2a - (4 * p * (coe - 1) * d) / B_p2p
    
    # 避免除零错误
    if denominator == 0:
        print("错误：分母为0，无唯一解（可能无交点或无穷多交点）")
        return None
    
    # 计算s的值
    s = numerator / denominator
    return s

# ------------------------------
# 使用示例（请替换为你的实际参数值）
# ------------------------------
if __name__ == "__main__":
    # 示例参数（请根据实际情况修改）
    B = 1
    #c = -1.1176326223782225
    a = 2.1930476567007057e-09 
    c = -1.1176326223782225
    p = 8
    b =  0.0011911824608909084
    d = 128
    n_h = 16
    B_a2a = 1/(0.004567077071548219)
    n_kv = 2
    coe = 1.1
    B_p2p = 8
    
    # 计算s
    result_with = calculate_s_with_replication(c, p, b, d, n_h, B_a2a, n_kv, coe, B_p2p)
    result_without = calculate_s_without_replication(c, p, b, d, n_h, B_a2a, n_kv, coe, B_p2p)
    if 1 is not None:
        print(f"交点横坐标s with的值为：{result_with:.6f}")
        print(f"交点横坐标s without的值为：{result_without:.6f}")