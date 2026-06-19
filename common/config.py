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
    model: str = "mlp"            # "cnn" | "mlp"（默认 mlp：部分 armv7l 树莓派 torch 构建的 conv 不可靠）
    dataset: str = "mnist"        # 预留扩展: "mnist" | "cifar10"
    partition: str = "iid"        # 预留扩展: "iid" | "noniid"
    data_dir: str = "./data"
    seed: int = 42

    # ===== 服务 =====
    host: str = "0.0.0.0"
    port: int = 5000
    eval_batch_size: int = 256
    results_dir: str = "./results"

    def to_dict(self):
        return asdict(self)
