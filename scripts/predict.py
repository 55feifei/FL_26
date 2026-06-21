"""加载训练好的全局模型，在对应数据集（MNIST / CIFAR-10）测试集上评估并展示样例预测。

数据集、通道、归一化方式均从 checkpoint 元信息读取，自动重建匹配的模型。

服务器默认按实验名把产物存到 results/{dataset}_{model}[_{norm}]/ 子目录。本脚本不带
参数时会自动在 results/ 下搜索 global_model.pth 并选最新的一个；也可显式指定路径。

用法（在 fl 目录下）：
    python scripts/predict.py                                  # 自动选 results/ 下最新模型
    python scripts/predict.py results/cifar10_resnet_group/global_model.pth
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common.model import build_model
from common.data import load_dataset, make_loader


def resolve_ckpt(arg):
    """确定要评估的 checkpoint 路径。

    显式给了路径就用它；否则优先 results/global_model.pth（旧的扁平布局），
    再退而搜索 results/ 下所有子目录里的 global_model.pth，取修改时间最新的一个。
    """
    if arg:
        return arg
    # 收集 results/ 下所有 global_model.pth（含扁平布局与各实验子目录），取修改时间最新的
    # 一个。normpath 统一路径分隔符以去重（glob 在 Windows 上可能混用 / 和 \）。
    cands = sorted(
        {os.path.normpath(p) for p in glob.glob("results/**/global_model.pth", recursive=True)},
        key=os.path.getmtime, reverse=True,
    )
    if not cands:
        return "results/global_model.pth"  # 交给调用方报"找不到"
    if len(cands) > 1:
        print("results/ 下发现多个模型，默认评估最新的一个（要评估别的请把路径作为参数传入）：")
        for c in cands:
            print(f"   {c}")
    return cands[0]


def main():
    ckpt_path = resolve_ckpt(sys.argv[1] if len(sys.argv) > 1 else None)
    if not os.path.exists(ckpt_path):
        print(f"找不到模型文件：{ckpt_path}，请先完成训练。")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    channels = ckpt.get("channels")
    norm = ckpt.get("norm", "group")
    model = build_model(ckpt["model"], ckpt["dataset"], channels=channels, norm=norm)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"已加载模型：{ckpt_path}")
    print(f"  model={ckpt['model']}  dataset={ckpt['dataset']}  norm={norm}  "
          f"round={ckpt.get('round')}  训练时记录准确率={ckpt.get('accuracy')}")

    test_set = load_dataset(ckpt["dataset"], "./data", train=False, download=True, channels=channels)
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
