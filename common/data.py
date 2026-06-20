"""数据加载与联邦分片（基于 torchvision）。

- load_mnist(): 用 torchvision 自动下载并做标准预处理（ToTensor + Normalize）。
- partition_iid(): 把训练集随机等分给各客户端，模拟"数据分散在不同边缘节点"。
- partition_noniid(): 预留给提高部分（Non-IID 研究），每个客户端只拿到少数类别。

注意：各客户端用相同 seed 做划分，保证切分一致、互不重叠，且每个节点只能看到
自己那一份数据 —— 这正是联邦学习"数据不出本地"的体现。
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _mnist_transform(channels=1):
    t = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 全局均值/方差
    ]
    if channels == 3:
        # 把单通道复制成 3 通道，绕开部分 armv7l 树莓派 torch 的"单通道卷积"bug
        t.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    return transforms.Compose(t)


def load_mnist(data_dir="./data", train=True, download=True, channels=1):
    """加载 MNIST（首次会自动下载到 data_dir/MNIST/）。channels=3 时复制为三通道。"""
    return datasets.MNIST(
        root=data_dir, train=train, download=download,
        transform=_mnist_transform(channels),
    )


def partition_iid(dataset, num_clients, client_id, seed=42):
    """IID 划分：随机打乱后等分，返回第 client_id 份的 Subset。"""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    shards = np.array_split(idx, num_clients)
    return Subset(dataset, shards[client_id].tolist())


def partition_noniid(dataset, num_clients, client_id, classes_per_client=2, seed=42):
    """Non-IID 划分（McMahan shard 法，预留给提高部分）。

    将数据按标签排序后切成 num_clients*classes_per_client 个分片，
    每个客户端随机领取 classes_per_client 个分片，于是只见到少数类别。
    """
    labels = np.asarray(dataset.targets)
    order = np.argsort(labels, kind="stable")
    shards = np.array_split(order, num_clients * classes_per_client)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(shards))
    chosen = perm[client_id * classes_per_client:(client_id + 1) * classes_per_client]
    idx = np.concatenate([shards[s] for s in chosen])
    return Subset(dataset, idx.tolist())


def partition_imbalanced(dataset, num_clients, client_id, ratios, seed=42):
    """不均衡 IID 划分：按 ratios 指定的比例将数据集分配给各客户端（内容 IID，数量不均）。

    参数:
        ratios: 各客户端样本比例列表，长度须等于 num_clients，各元素之和应为 1.0。
                例如 [0.9, 0.1] 表示 client 0 获得 90%，client 1 获得 10%。
    """
    assert len(ratios) == num_clients, "ratios 长度须等于 num_clients"
    n = len(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    sizes = [int(r * n) for r in ratios]
    sizes[-1] = n - sum(sizes[:-1])   # 修正浮点舍入，保证总和 = n
    start, splits = 0, []
    for s in sizes:
        splits.append(idx[start:start + s])
        start += s
    return Subset(dataset, splits[client_id].tolist())


def partition_noniid_dirichlet(dataset, num_clients, client_id, alpha=0.5, seed=42):
    """Dirichlet 分布 Non-IID 划分（比 McMahan shard 法更灵活，是学术界主流方案）。

    对每个类别 c，从 Dir(alpha) 采样各客户端的分配比例，alpha 越小越 Non-IID：
      alpha → 0  : 每个客户端只拥有极少数类别的样本（极度 Non-IID）
      alpha = 0.1: 高度 Non-IID，各客户端数据高度偏斜
      alpha = 0.5: 中等 Non-IID，有明显类别偏斜但每类都有少量样本
      alpha → ∞  : 接近 IID，每个客户端的类别分布趋向均匀

    参数:
        alpha: Dirichlet 浓度参数，控制异质程度。
    """
    labels = np.asarray(dataset.targets)
    num_classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0].copy()
        rng.shuffle(class_idx)
        n_c = len(class_idx)
        # 从 Dirichlet 采样各客户端对类别 c 的分配比例
        props = rng.dirichlet(np.ones(num_clients) * alpha)
        counts = (props * n_c).astype(int)
        counts[-1] = n_c - int(counts[:-1].sum())   # 修正舍入，保证总和 = n_c
        counts = np.maximum(counts, 0)
        start = 0
        for k, cnt in enumerate(counts):
            client_indices[k].extend(class_idx[start:start + int(cnt)].tolist())
            start += int(cnt)

    return Subset(dataset, client_indices[client_id])


def make_loader(dataset, batch_size, shuffle, num_workers=0):
    """构建 DataLoader。树莓派上 num_workers 默认 0（多进程加载在 Pi 上易出问题）。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)
