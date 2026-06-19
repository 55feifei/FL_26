"""逐层定位树莓派前向 NaN（不需要服务器）。

已确认：服务器下发的权重传到 Pi 后逐位正确且有限，但用它前向得到 NaN。
本脚本在 Pi 本地复现并逐层检查，回答：
  1) 是「加载权重(load_state_dict/序列化往返)」触发的，还是卷积算子本身？
  2) NaN 最先出现在哪一层（conv1/pool/conv2/fc1...）？
  3) 单线程是否能避免？
  4) MLP 是否完全正常（→ 能否用 --model mlp 绕过）？

在树莓派 fl 目录下运行：
    python3 scripts/diagnose_forward.py
把输出发回即可。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from common.model import build_model
from common.data import load_mnist, partition_iid, make_loader
from common.serialize import state_dict_to_bytes, bytes_to_state_dict


def fin(t):
    return bool(torch.isfinite(t).all())


def step_cnn(model, x, tag):
    """逐层跑 SimpleCNN 的前向并报告每层是否有限，返回最先 NaN 的层名。"""
    first_bad = None
    z = model.conv1(x); ok = fin(z); first_bad = first_bad or (None if ok else "conv1")
    z = model.pool(F.relu(z)); ok2 = fin(z); first_bad = first_bad or (None if ok2 else "pool1")
    z = model.conv2(z); ok3 = fin(z); first_bad = first_bad or (None if ok3 else "conv2")
    z = model.pool(F.relu(z)); ok4 = fin(z); first_bad = first_bad or (None if ok4 else "pool2")
    z = torch.flatten(z, 1)
    z = model.fc1(z); ok5 = fin(z); first_bad = first_bad or (None if ok5 else "fc1")
    z = model.fc2(F.relu(z)); ok6 = fin(z); first_bad = first_bad or (None if ok6 else "fc2")
    print(f"  [{tag}] conv1={ok} pool1={ok2} conv2={ok3} pool2={ok4} fc1={ok5} fc2={ok6}"
          f"  -> 最先NaN: {first_bad or '无(全部正常)'}")
    return first_bad


def run_cnn(no_grad):
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        torch.manual_seed(0)
        m_fresh = build_model("cnn", "mnist")
        step_cnn(m_fresh, X, f"全新权重 no_grad={no_grad}")

        # 序列化往返后 load_state_dict（复现真实客户端的加载路径）
        sd = bytes_to_state_dict(state_dict_to_bytes(m_fresh.state_dict()))
        m_reload = build_model("cnn", "mnist")
        m_reload.load_state_dict(sd)
        step_cnn(m_reload, X, f"加载往返权重 no_grad={no_grad}")


def main():
    global X
    ds = load_mnist("./data", train=True, download=True)
    X, _ = next(iter(make_loader(partition_iid(ds, 2, 0), 32, shuffle=True)))
    print(f"torch={torch.__version__}  默认线程={torch.get_num_threads()}  输入finite={fin(X)}\n")

    print("=== CNN / 默认线程 ===")
    run_cnn(no_grad=False)
    run_cnn(no_grad=True)

    print("\n=== CNN / 单线程(set_num_threads(1)) ===")
    torch.set_num_threads(1)
    run_cnn(no_grad=False)
    run_cnn(no_grad=True)

    print("\n=== MLP / 加载往返权重（验证能否用 --model mlp 绕过）===")
    torch.set_num_threads(torch.get_num_threads())
    m = build_model("mlp", "mnist")
    sd = bytes_to_state_dict(state_dict_to_bytes(m.state_dict()))
    m2 = build_model("mlp", "mnist"); m2.load_state_dict(sd)
    with torch.no_grad():
        print(f"  MLP 前向 finite = {fin(m2(X))}")


if __name__ == "__main__":
    main()
