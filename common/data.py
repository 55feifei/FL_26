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


def _mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 全局均值/方差
    ])


def load_mnist(data_dir="./data", train=True, download=True):
    """加载 MNIST（首次会自动下载到 data_dir/MNIST/）。"""
    return datasets.MNIST(
        root=data_dir, train=train, download=download,
        transform=_mnist_transform(),
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


def make_loader(dataset, batch_size, shuffle, num_workers=0):
    """构建 DataLoader。树莓派上 num_workers 默认 0（多进程加载在 Pi 上易出问题）。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)
