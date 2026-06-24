"""树莓派本地 PyTorch 健康验证：在本机独立训练一个 MNIST 手写数字识别系统。

不连服务器、不涉及联邦，纯本地「训练 + 测试」，用来判断本机 PyTorch 能否正常训练。
默认同时测 MLP 和 CNN 作对比：
  - 两者都能 loss 下降、测试集准确率高 -> 本机 PyTorch 正常（可用于 CNN/CIFAR-10/MobileNet）。
  - CNN 出现 NaN 或准确率≈10%（随机），而 MLP 正常 -> 本机 torch 的卷积坏了
    （armv7l 老构建的已知问题）；需换 64 位官方 torch 才能做卷积网络。
  - 刷完 64 位官方 torch 后，再跑本脚本，期望 MLP 和 CNN 都 [OK]。

用法（在 fl 目录下）：
  python3 scripts/train_local.py                       # 默认 mlp+cnn 快速训练
  python3 scripts/train_local.py --model cnn           # 只测 CNN
  python3 scripts/train_local.py --epochs 2 --max-batches 0   # 跑满整个 epoch（更慢更准）

  # 复刻 FL 真机配置脱离联邦单测轻量化网络（与 fl_client 同样 --channels/--threads/--norm）：
  python3 scripts/train_local.py --model mobilenet  --channels 3 --threads 1
  python3 scripts/train_local.py --model squeezenet --channels 3 --threads 1
  # 对比：深度卷积坏(mobilenet 学不动) vs 标准卷积好(squeezenet/cnn 正常收敛)
  #       即可在本机坐实是 armv7l 的「分组/深度卷积」路径有问题。
"""

import os
import sys
import time
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common.model import build_model
from common.data import load_mnist, make_loader


def train_and_eval(model_name, train_loader, test_loader, args, device):
    print(f"\n===== 训练模型: {model_name} =====", flush=True)
    model = build_model(model_name, "mnist", channels=args.channels, norm=args.norm).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    crit = torch.nn.CrossEntropyLoss()

    nan_hit = False
    t0 = time.time()
    model.train()
    for ep in range(args.epochs):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            lv = loss.item()
            if i % 5 == 0:
                print(f"  epoch {ep + 1} batch {i:4d}  loss={lv:.4f}", flush=True)
            if not math.isfinite(lv):
                print(f"  !! batch {i} loss={lv}（NaN/Inf）—— 本机 {model_name} 训练异常", flush=True)
                nan_hit = True
                break
            if args.max_batches and (i + 1) >= args.max_batches:
                break
        if nan_hit:
            break
    dt = time.time() - t0

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    ok = (not nan_hit) and acc > 0.5
    print(f"  用时 {dt:.1f}s  测试集准确率 = {acc:.4f}  -> {'[OK] 正常' if ok else '[FAIL] 异常'}",
          flush=True)
    return model_name, acc, ok, nan_hit


def main():
    ap = argparse.ArgumentParser(description="本地训练 MNIST 验证 PyTorch 是否正常")
    ap.add_argument("--model", default="both",
                    choices=["mlp", "cnn", "deepcnn", "resnet", "mobilenet", "squeezenet", "both"],
                    help="both=mlp+cnn 对照；其余为单模型（含轻量化 mobilenet/squeezenet）")
    ap.add_argument("--channels", type=int, default=1, choices=[1, 3],
                    help="输入通道；armv7l 上设 3 把 MNIST 复制为三通道，与 fl_client 保持一致")
    ap.add_argument("--norm", default="group", choices=["batch", "group"],
                    help="deepcnn/resnet/mobilenet/squeezenet 的归一化方式（与 fl_client 一致）")
    ap.add_argument("--threads", type=int, default=0,
                    help=">0 时设置 torch 线程数；armv7l conv 偶发 NaN 时可设 1")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=200,
                    help=">0 时每个 epoch 只训这么多 batch（控时）；0=跑满整个 epoch")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args()

    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    print(f"torch={torch.__version__}  线程={torch.get_num_threads()}  device=cpu  "
          f"channels={args.channels}  norm={args.norm}  max_batches={args.max_batches or '满epoch'}")

    train_set = load_mnist(args.data_dir, train=True, download=True, channels=args.channels)
    test_set = load_mnist(args.data_dir, train=False, download=True, channels=args.channels)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True)
    test_loader = make_loader(test_set, 256, shuffle=False)

    names = ["mlp", "cnn"] if args.model == "both" else [args.model]
    results = [train_and_eval(n, train_loader, test_loader, args, device) for n in names]

    print("\n================ 结论 ================")
    for name, acc, ok, nan_hit in results:
        tag = "正常" if ok else ("NaN异常" if nan_hit else "准确率过低")
        print(f"  {name:3s}: 准确率={acc:.4f}  -> {tag}")

    ok_map = {n: ok for n, _, ok, _ in results}
    if all(ok_map.values()):
        print("\n本机 PyTorch 完全正常，可用于 CNN / CIFAR-10 / MobileNet。")
    elif ok_map.get("mlp") and ok_map.get("cnn") is False:
        print("\n卷积(CNN)异常、全连接(MLP)正常 —— 典型 armv7l 老 torch 构建的 conv bug。")
        print("MNIST 演示请用 --model mlp；要做 CNN/CIFAR-10 需换 64 位官方 torch。")
    else:
        print("\n本机 torch 训练存在异常，请检查安装（建议换 64 位官方 torch）。")


if __name__ == "__main__":
    main()
