"""验证「输入通道=1 的卷积在 armv7l 树莓派上有 bug」并测试「复制成 3 通道」能否修复。

背景：
  - 知乎经验：armv7l 上 in_channels=1 的卷积第一层 grad=None、网络不训练（静默 bug）；
  - 我们实测：in_channels=1 的卷积前向偶发 NaN；
  两者触发条件相同（单通道卷积路径），疑似同源。复制成 3 通道可绕开该路径。

本脚本对比 in_channels=1 vs 3（把 MNIST 单通道 repeat 成 3 通道）：
  ① 前向 NaN 率  ② 训练时 conv1.weight.grad 是否为 None  ③ loss 是否真的下降

在树莓派 fl 目录下运行：
    python3 scripts/test_3channel.py
若 3 通道这一行：NaN=0/20、grad 不为 None、loss 明显下降 → 复制 3 通道即可在本机用 CNN。
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from common.model import SimpleCNN
from common.data import load_mnist, make_loader


def to3(x):
    """[N,1,28,28] -> [N,3,28,28]，用 repeat 生成连续内存的三通道。"""
    return x.repeat(1, 3, 1, 1)


def forward_nan_rate(loader, in_ch, runs=20):
    nan = 0
    for _ in range(runs):
        m = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=28)
        x, _ = next(iter(loader))
        if in_ch == 3:
            x = to3(x)
        with torch.no_grad():
            out = m(x)
        if not torch.isfinite(out).all():
            nan += 1
    return nan, runs


def train_check(loader, in_ch, steps=60):
    m = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=28)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    first_loss = last_loss = None
    grad_none = False
    for i, (x, y) in enumerate(loader):
        if in_ch == 3:
            x = to3(x)
        opt.zero_grad()
        loss = crit(m(x), y)
        loss.backward()
        if m.conv1.weight.grad is None:
            grad_none = True
        opt.step()
        lv = loss.item()
        if first_loss is None:
            first_loss = lv
        last_loss = lv
        if not math.isfinite(lv):
            break
        if i + 1 >= steps:
            break
    return first_loss, last_loss, grad_none


def main():
    ds = load_mnist("./data", train=True, download=True)
    loader = make_loader(ds, 32, shuffle=True)
    print(f"torch={torch.__version__}  线程={torch.get_num_threads()}\n")
    for ch in (1, 3):
        nan, runs = forward_nan_rate(loader, ch)
        fl, ll, gn = train_check(loader, ch)
        verdict = "[OK]" if (nan == 0 and not gn and fl and ll < fl - 0.1) else "[异常]"
        print(f"in_channels={ch}: 前向NaN {nan}/{runs} | conv1.grad为None={gn} | "
              f"loss {fl:.3f} -> {ll:.3f}  {verdict}")
    print("\n判读：若 in_channels=3 行为 [OK]（NaN=0、grad不为None、loss下降），"
          "则把 MNIST 复制成 3 通道即可在本机用 CNN，无需刷 64 位。")


if __name__ == "__main__":
    main()
