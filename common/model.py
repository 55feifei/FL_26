"""模型定义（纯 PyTorch）。

提供两个模型：
- MLP：两层全连接，最轻量，在树莓派上训练最快，适合快速跑通联邦闭环。
- SimpleCNN：两层卷积 + 两层全连接，MNIST 上精度更高（>98%），是默认模型。

build_model() 是统一工厂，后续若扩展 CIFAR-10 / MobileNet 只需在此添加分支。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """最简单的两层全连接网络。"""

    def __init__(self, in_dim=28 * 28, num_classes=10, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleCNN(nn.Module):
    """经典小型卷积网络：conv-pool-conv-pool-fc-fc。

    通过 input_size 自适应计算展平维度（MNIST 28 -> 7；CIFAR 32 -> 8），
    便于后续切换数据集而无需改网络结构。
    """

    def __init__(self, in_channels=1, num_classes=10, input_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        feat = input_size // 4  # 经过两次 2x2 池化
        self.fc1 = nn.Linear(32 * feat * feat, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))   # 14 -> 7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(name="cnn", dataset="mnist"):
    """根据名称与数据集构建模型。"""
    in_channels = 1 if dataset == "mnist" else 3
    input_size = 28 if dataset == "mnist" else 32
    num_classes = 10

    if name == "mlp":
        in_dim = in_channels * input_size * input_size
        return MLP(in_dim, num_classes)
    if name == "cnn":
        return SimpleCNN(in_channels, num_classes, input_size)
    raise ValueError(f"未知模型: {name}（可选 'cnn' | 'mlp'）")
