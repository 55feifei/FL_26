"""FedAvg 聚合算法（McMahan et al., 2017）。

全局模型 = 各客户端本地模型按"本地样本数"加权平均：
    W_global = Σ_k (n_k / Σ n) · W_k
样本多的节点对全局模型贡献更大，这就是"联邦平均"的核心。

鲁棒性：真机上部分客户端（尤其 armv7l 树莓派的非官方 torch 构建）可能产生
NaN/Inf 权重。直接平均会让 NaN 污染整个全局模型（准确率塌到随机水平）。
因此聚合前先剔除含 NaN/Inf 的更新；若本轮全部无效，返回 None 让调用方保留上一轮模型。
"""

from collections import OrderedDict

import torch


def _is_finite_state_dict(sd):
    """判断一个 state_dict 的所有张量是否都是有限值（无 NaN / Inf）。"""
    for v in sd.values():
        if not torch.isfinite(v.float()).all().item():
            return False
    return True


def fedavg(updates):
    """对一组客户端更新做加权平均（自动剔除含 NaN/Inf 的坏更新）。

    参数:
        updates: list[(state_dict, num_samples)]
    返回:
        (聚合后的 state_dict 或 None, dropped)
        - 当所有更新都含 NaN/Inf 时返回 (None, dropped)，调用方应保留上一轮模型；
        - dropped 为被剔除的坏更新数量。
    """
    if not updates:
        raise ValueError("没有可聚合的客户端更新")

    valid = [(sd, n) for sd, n in updates if _is_finite_state_dict(sd)]
    dropped = len(updates) - len(valid)
    if not valid:
        return None, dropped

    total = sum(n for _, n in valid)
    if total <= 0:
        raise ValueError("样本总数为 0，无法聚合")

    ref_sd = valid[0][0]
    avg = OrderedDict()
    for key in ref_sd.keys():
        acc = None
        for sd, n in valid:
            weighted = sd[key].float() * (float(n) / total)
            acc = weighted if acc is None else acc + weighted
        # 还原原始 dtype（例如整型 buffer），避免类型漂移
        avg[key] = acc.to(ref_sd[key].dtype)
    return avg, dropped
