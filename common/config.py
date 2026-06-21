"""集中管理联邦学习系统的所有超参数与运行配置。

服务器是"轮次"与训练流程的真相来源（current_round 驱动整个同步过程），
客户端与服务器共享同一套默认超参数（通过命令行参数覆盖即可）。
"""

from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # ===== 联邦设置 =====
    num_clients: int = 2          # 客户端总数 = 树莓派数量
    rounds: int = 15              # 通信轮数 T（全局聚合次数）

    # ===== 本地训练 =====
    local_epochs: int = 1         # 每轮本地训练 epoch 数 E
    local_steps: int = 0          # >0 时每轮只训练这么多个 batch（控时用），=0 表示按 epoch 跑满
    batch_size: int = 32
    lr: float = 0.01
    momentum: float = 0.9

    # ===== 模型 / 数据 =====
    model: str = "mlp"            # "mlp" | "cnn" | "deepcnn" | "resnet"（默认 mlp：部分 armv7l 树莓派 torch 构建的 conv 不可靠）
    channels: int = 1             # 输入通道数；CNN 在 armv7l Pi 上需设 3（绕开单通道卷积 bug）；CIFAR-10 须为 3
    dataset: str = "mnist"        # "mnist" | "cifar10"
    norm: str = "group"           # deepcnn/resnet 归一化方式："group"(FL 推荐) | "batch"
    partition: str = "iid"        # "iid" | "shard"(=noniid) | "dirichlet" | "imbalanced"
    classes_per_client: int = 2   # shard 方式：每客户端分到的类别数（越小越 Non-IID）
    alpha: float = 0.5            # dirichlet 方式：浓度参数（越小越 Non-IID）
    ratios: tuple = None          # imbalanced 方式：各客户端样本比例，如 (0.9, 0.1)
    data_dir: str = "./data"
    seed: int = 42

    # ===== 服务 =====
    host: str = "0.0.0.0"
    port: int = 5000
    eval_batch_size: int = 256
    results_dir: str = "./results"

    def to_dict(self):
        return asdict(self)
