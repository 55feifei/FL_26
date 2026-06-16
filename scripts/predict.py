"""加载训练好的全局模型，在 MNIST 测试集上评估并展示若干样例预测。

用法（在 fl 目录下）：
    python scripts/predict.py                       # 默认读 results/global_model.pth
    python scripts/predict.py results/global_model.pth
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common.model import build_model
from common.data import load_mnist, make_loader


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "results/global_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"找不到模型文件：{ckpt_path}，请先完成训练。")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ckpt["model"], ckpt["dataset"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"已加载模型：{ckpt_path}")
    print(f"  model={ckpt['model']}  dataset={ckpt['dataset']}  "
          f"round={ckpt.get('round')}  训练时记录准确率={ckpt.get('accuracy')}")

    test_set = load_mnist("./data", train=False, download=True)
    loader = make_loader(test_set, 256, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"测试集整体准确率：{correct / total:.4f}  ({correct}/{total})")

    # 随机展示 10 个样本的预测 vs 真实标签
    x, y = next(iter(make_loader(test_set, 10, shuffle=True)))
    with torch.no_grad():
        pred = model(x).argmax(1)
    print("样例预测：", pred.tolist())
    print("真实标签：", y.tolist())


if __name__ == "__main__":
    main()
