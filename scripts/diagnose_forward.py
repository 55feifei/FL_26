"""定位"加载权重后前向变 NaN"的根因（不需要服务器，在树莓派本地复现）。

核心矛盾：
  - check_data.py：全新初始化的模型前向正常；
  - probe_transfer.py：加载(load_state_dict)同样数值的权重后前向 NaN。
两者权重数值相同，唯一区别是是否经过「序列化往返 + load_state_dict（内部 torch.from_numpy）」。

本脚本在 Pi 上对同一组权重做三种加载方式并逐层比对，回答：
  1) 往返前后权重是否逐位完全相同？（验证序列化无损）
  2) 全新权重 vs 加载权重，各层前向是否 NaN？最先在哪层 NaN？
  3) 用「clone().contiguous() 干净拷贝」加载能否避免 NaN？（这就是潜在修复）

在树莓派 fl 目录下运行：
    python3 scripts/diagnose_forward.py
把输出整段发回。
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


def per_layer(model, x):
    """逐层跑 SimpleCNN，返回最先出现 NaN 的层名（None 表示全部正常）。"""
    z = model.conv1(x)
    if not fin(z): return "conv1"
    z = model.pool(F.relu(z))
    if not fin(z): return "pool1"
    z = model.conv2(z)
    if not fin(z): return "conv2"
    z = model.pool(F.relu(z))
    if not fin(z): return "pool2"
    z = torch.flatten(z, 1)
    z = model.fc1(z)
    if not fin(z): return "fc1"
    z = model.fc2(F.relu(z))
    if not fin(z): return "fc2"
    return None


def main():
    ds = load_mnist("./data", train=True, download=True)
    X, _ = next(iter(make_loader(partition_iid(ds, 2, 0), 32, shuffle=True)))
    print(f"torch={torch.__version__}  线程={torch.get_num_threads()}  输入finite={fin(X)}\n")

    # 全新模型（= check_data 场景，自带原生权重）
    torch.manual_seed(0)
    m_fresh = build_model("cnn", "mnist")
    fresh_sd = m_fresh.state_dict()

    # 序列化往返：Pi -> bytes -> Pi（内部 from_numpy）
    rt_sd = bytes_to_state_dict(state_dict_to_bytes(fresh_sd))

    # 1) 权重往返一致性（你提的校验）
    print("[1] 往返前后权重是否逐位相同：")
    all_same = True
    for k in fresh_sd:
        same = torch.equal(fresh_sd[k].float(), rt_sd[k].float())
        all_same = all_same and same
        if not same:
            print(f"    不一致层: {k}")
    print(f"    结论: 全部逐位相同 = {all_same}\n")

    # 2) 三种加载方式的前向对比
    print("[2] 各方式前向最先 NaN 的层（None=全部正常）：")
    print(f"    A 全新原生权重           : {per_layer(m_fresh, X)}")

    m_rt = build_model("cnn", "mnist")
    m_rt.load_state_dict(rt_sd)
    print(f"    B 加载往返权重(from_numpy): {per_layer(m_rt, X)}")

    # 3) 潜在修复：加载前把张量 clone().contiguous() 成干净的原生张量
    clean_sd = {k: v.detach().clone().contiguous() for k, v in rt_sd.items()}
    m_clean = build_model("cnn", "mnist")
    m_clean.load_state_dict(clean_sd)
    print(f"    C 加载干净拷贝(contiguous): {per_layer(m_clean, X)}")

    print("\n判读：若 A=None 而 B=某层，则确为『加载路径』触发；"
          "若 C=None，则修复 = 反序列化时 clone().contiguous()。")


if __name__ == "__main__":
    main()
