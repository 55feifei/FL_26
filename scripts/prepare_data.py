"""预下载 MNIST 数据集到 ./data（首次联网时执行一次即可）。

推荐流程：先在 PC 上跑本脚本下载好，再把整个 data/MNIST 目录用 scp 拷到每台
树莓派 —— 这样可绕开树莓派上旧版 torchvision 下载源失效的问题。

用法（在 fl 目录下）：
    python scripts/prepare_data.py            # 下载到 ./data
    python scripts/prepare_data.py ./data     # 指定目录
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import load_mnist


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    print(f"下载 MNIST 到 {os.path.abspath(data_dir)} ...")
    load_mnist(data_dir, train=True, download=True)
    load_mnist(data_dir, train=False, download=True)
    print("完成。可将 data/MNIST 目录拷贝到树莓派复用。")


if __name__ == "__main__":
    main()
