"""量化树莓派上 Conv2d 的偶发 NaN 率，并检验缓解手段（单线程 / MLP）。

已确认：torch 1.4.0a0 的 Conv2d 对有限输入+有限权重也会"偶发"算出 NaN
（同一前向有时正常有时 NaN）。本脚本重复多次前向统计 NaN 出现比例，并对比：
  - CNN 默认线程 vs 单线程（看 set_num_threads(1) 能否消除）
  - MLP（不含卷积，作为可靠替代方案的验证）

在树莓派 fl 目录下运行：
    python3 scripts/stress_conv.py
把输出发回。0/N 表示该配置稳定可用。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common.model import build_model
from common.data import load_mnist, partition_iid, make_loader


def trial(model_name, threads, x, runs=30):
    torch.set_num_threads(threads)
    nan = 0
    for _ in range(runs):
        model = build_model(model_name, "mnist")   # 每次全新随机权重
        with torch.no_grad():
            out = model(x)
        if not torch.isfinite(out).all():
            nan += 1
    print(f"  {model_name:3s} threads={threads}: {nan}/{runs} 次前向出现 NaN"
          f"  {'<-- 不稳定' if nan else '(稳定)'}")


def main():
    ds = load_mnist("./data", train=True, download=True)
    x, _ = next(iter(make_loader(partition_iid(ds, 2, 0), 32, shuffle=True)))
    print(f"torch={torch.__version__}  默认线程={torch.get_num_threads()}\n")
    trial("cnn", torch.get_num_threads(), x)
    trial("cnn", 1, x)
    trial("mlp", torch.get_num_threads(), x)
    trial("mlp", 1, x)
    print("\n判读：若 cnn threads=1 为 0/30 → 单线程可救活 CNN；"
          "若仅 mlp 为 0/30 → 用 --model mlp 最稳。")


if __name__ == "__main__":
    main()
