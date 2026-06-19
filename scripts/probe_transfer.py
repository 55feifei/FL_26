"""权重传输探针：定性判断"服务器下发的权重，到本机反序列化后是否变成 NaN"。

用法（服务器必须正在运行）：
    # 在 PC 上（与服务器同版本，作为参照基准）：
    python scripts/probe_transfer.py http://127.0.0.1:5000
    # 在树莓派上（被怀疑出问题的一端）：
    python3 scripts/probe_transfer.py http://<PC_IP>:5000

对比两边输出的「conv1.weight 前5个值」：
  - 若 Pi 的值与 PC 不同、或出现 NaN/Inf  -> 序列化/传输在跨版本时损坏了权重，
    应改用版本无关的裸字节序列化（不要再用 np.savez）。
  - 若两边数值一致且全有限             -> 问题不在传输；NaN 另有来源，需继续排查。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import torch

from common.serialize import bytes_to_state_dict


def main():
    server = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000"
    print(f"GET {server}/get_model ...")
    resp = requests.get(f"{server}/get_model", timeout=120)
    data = resp.content
    print(f"收到字节数 = {len(data)}  轮次头 X-FL-Round={resp.headers.get('X-FL-Round')}")

    sd = bytes_to_state_dict(data)
    bad = [k for k, v in sd.items() if not torch.isfinite(v.float()).all()]
    print(f"反序列化层数 = {len(sd)}")
    print(f"含 NaN/Inf 的层 = {bad if bad else '无'}")

    # 打印一层样本值，便于 PC/Pi 逐位对比
    for key in ("conv1.weight", "fc1.weight"):
        if key in sd:
            vals = sd[key].flatten()[:5].tolist()
            print(f"{key} 前5个值 = {vals}")

    # 关键：载入下发权重后，用真实数据做一次前向，看 forward 是否 NaN
    try:
        import torch.nn as nn
        from common.model import build_model
        from common.data import load_mnist, partition_iid, make_loader
        model = build_model("cnn", "mnist")
        model.load_state_dict(sd)
        model.eval()
        ds = load_mnist("./data", train=True, download=True)
        x, y = next(iter(make_loader(partition_iid(ds, 2, 0), 32, shuffle=True)))
        with torch.no_grad():
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
        print(f"载入下发权重后真实数据前向: "
              f"out_isfinite={bool(torch.isfinite(out).all())}  "
              f"loss={loss.item()}  loss_isfinite={bool(torch.isfinite(loss))}")
    except Exception as e:
        print("前向测试跳过：", e)


if __name__ == "__main__":
    main()
