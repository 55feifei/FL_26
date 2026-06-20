"""预下载数据集到 ./data（首次联网时执行一次即可）。

推荐流程：先在 PC 上跑本脚本下载好，再把对应数据目录用 scp 拷到每台树莓派
（MNIST -> data/MNIST/，CIFAR-10 -> data/cifar-10-batches-py/）—— 这样可绕开
树莓派上旧版 torchvision 下载源失效的问题。

用法（在 fl 目录下）：
    python scripts/prepare_data.py                   # 下载 MNIST 到 ./data
    python scripts/prepare_data.py cifar10           # 下载 CIFAR-10 到 ./data
    python scripts/prepare_data.py cifar10 ./data    # 指定数据集与目录
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import load_dataset


def main():
    args = [a for a in sys.argv[1:]]
    dataset = "mnist"
    data_dir = "./data"
    # 解析位置参数：[dataset] [data_dir]，dataset 可省略（默认 mnist）
    for a in args:
        if a.lower() in ("mnist", "cifar10"):
            dataset = a.lower()
        else:
            data_dir = a
    print(f"下载 {dataset} 到 {os.path.abspath(data_dir)} ...")
    # download 阶段不做数据增强（augment 仅影响 transform，不影响落盘文件）
    load_dataset(dataset, data_dir, train=True, download=True, augment=False)
    load_dataset(dataset, data_dir, train=False, download=True)
    sub = "MNIST" if dataset == "mnist" else "cifar-10-batches-py"
    print(f"完成。可将 data/{sub} 目录拷贝到树莓派复用。")


if __name__ == "__main__":
    main()
