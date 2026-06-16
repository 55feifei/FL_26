"""FedAvg 聚合算法（McMahan et al., 2017）。

全局模型 = 各客户端本地模型按"本地样本数"加权平均：
    W_global = Σ_k (n_k / Σ n) · W_k
样本多的节点对全局模型贡献更大，这就是"联邦平均"的核心。
"""

from collections import OrderedDict


def fedavg(updates):
    """对一组客户端更新做加权平均。

    参数:
        updates: list[(state_dict, num_samples)]
    返回:
        聚合后的 state_dict
    """
    if not updates:
        raise ValueError("没有可聚合的客户端更新")

    total = sum(n for _, n in updates)
    if total <= 0:
        raise ValueError("样本总数为 0，无法聚合")

    ref_sd = updates[0][0]
    avg = OrderedDict()
    for key in ref_sd.keys():
        acc = None
        for sd, n in updates:
            weighted = sd[key].float() * (float(n) / total)
            acc = weighted if acc is None else acc + weighted
        # 还原原始 dtype（例如整型 buffer），避免类型漂移
        avg[key] = acc.to(ref_sd[key].dtype)
    return avg
