"""树莓派真实数据通路自检 —— 定位 NaN 到底从哪一级进来。

check_torch.py 用随机数据训练正常，但真机客户端在第 1 步就 NaN，说明问题出在
「真实 MNIST 数据通路」或「跨版本权重加载」上。本脚本用实际的 load_mnist 取一个
真实 batch，逐级检查：输入 -> 前向输出 -> loss -> 梯度，哪一级先出现 NaN/Inf。

在树莓派 fl 目录下运行：
    python scripts/check_data.py

判读：
  - 「输入」就含 NaN/Inf 或 min/max 异常 -> torchvision/数据加载的问题（老版本 bug 或
    下载损坏）：重下数据，或改用手动解析 MNIST 的加载方式。
  - 输入正常但「前向输出」NaN -> 该 torch 构建对真实数据布局的卷积有问题（常见于
    非连续内存）：脚本会顺便测试 x.contiguous() 是否能解决。
  - 输入/前向正常但「loss」NaN -> 标签问题（越界/类型）。
  - 前面都正常但「梯度」NaN -> 反向算子问题。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from common.model import build_model
from common.data import load_mnist, partition_iid, make_loader


def report(tag, t):
    fin = bool(torch.isfinite(t).all())
    print(f"  {tag}: shape={tuple(t.shape)} dtype={t.dtype} "
          f"min={t.float().min():.4f} max={t.float().max():.4f} isfinite={fin}")
    return fin


def forward_loss_grad(model, x, y, label):
    print(f"\n[{label}]")
    model.zero_grad()
    out = model(x)
    if not report("前向输出", out):
        return
    loss = nn.CrossEntropyLoss()(out, y)
    print(f"  loss={loss.item()}  isfinite={bool(torch.isfinite(loss))}")
    if not torch.isfinite(loss):
        return
    loss.backward()
    gfin = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"  梯度 isfinite={gfin}")


def main():
    try:
        import torchvision
        print(f"torch={torch.__version__}  torchvision={torchvision.__version__}\n")
    except Exception:
        print(f"torch={torch.__version__}  torchvision=?\n")

    ds = load_mnist("./data", train=True, download=True)
    loader = make_loader(partition_iid(ds, 2, 0), 32, shuffle=True)
    x, y = next(iter(loader))

    print("---- 真实 MNIST 一个 batch ----")
    report("输入 x", x)
    print(f"  x 是否连续(contiguous)={x.is_contiguous()}")
    print(f"  标签 y: dtype={y.dtype} min={int(y.min())} max={int(y.max())} -> {y.tolist()}")

    model = build_model("cnn", "mnist")

    # 1) 真实数据 + 原始张量
    forward_loss_grad(model, x, y, "CNN / 真实数据 / 原张量")
    # 2) 真实数据 + contiguous（排查非连续内存触发的算子 bug）
    forward_loss_grad(model, x.contiguous(), y, "CNN / 真实数据 / x.contiguous()")
    # 3) 同形状随机数据做对照
    forward_loss_grad(model, torch.randn_like(x), y, "CNN / 随机数据（对照）")
    # 4) MLP 跑真实数据（若 CNN 坏而 MLP 好，可切 --model mlp）
    forward_loss_grad(build_model("mlp", "mnist"), x, y, "MLP / 真实数据")

    print("\n（依据脚本顶部「判读」说明对照结论，把本输出发回即可。）")


if __name__ == "__main__":
    main()
