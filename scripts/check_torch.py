"""树莓派 PyTorch 健康自检。

真机部署若出现 loss=NaN / 全局准确率塌到 ~0.098，多半是该树莓派上的
armv7l 非官方 torch 构建在某些算子（卷积/池化/多线程）上算出了 NaN。
本脚本在随机数据上对 MLP 和 CNN 各做 20 步训练，分别用「默认线程数」和「单线程」，
报告是否出现 NaN，从而快速定位问题与可行的绕过方式。

在树莓派 fl 目录下运行：
    python scripts/check_torch.py

判读：
  - CNN 出 NaN 而 MLP 正常  → 该构建的卷积/池化算子有问题：改用 --model mlp，
    或换一个 torch wheel。
  - 单线程正常、默认线程 NaN → 多线程数值问题：客户端设 torch.set_num_threads(1)
    或环境变量 OMP_NUM_THREADS=1。
  - 全都 NaN              → 该 torch 构建整体不可靠：更换 wheel，或先用 PC 多开
    client 完成演示。
  - 全部 ✅               → torch 本身没问题，NaN 另有原因（请回看客户端日志）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common.model import build_model


def trial(name, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = build_model(name, "mnist")
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = torch.nn.CrossEntropyLoss()
    for step in range(20):
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if not torch.isfinite(loss).item():
            print(f"  [{name:3s} threads={threads}] 第 {step} 步 loss=NaN/Inf  [FAIL]")
            return False
    finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    flag = "[OK]" if finite else "[FAIL] 参数含NaN"
    print(f"  [{name:3s} threads={threads}] 20 步训练完成，参数finite={finite}  {flag}")
    return finite


def main():
    default_threads = torch.get_num_threads()
    print(f"torch 版本={torch.__version__}  默认线程数={default_threads}\n")
    for name in ("mlp", "cnn"):
        for threads in sorted({default_threads, 1}, reverse=True):
            trial(name, threads)
    print("\n（依据脚本顶部「判读」说明对照结论。）")


if __name__ == "__main__":
    main()
