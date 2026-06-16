"""模型参数的网络序列化。

联邦学习中网络上传输的是模型参数（state_dict）。这里把 PyTorch 的 tensor
转成 numpy 数组，用 np.savez_compressed 打包成压缩字节流在 HTTP 上传输，
收端再还原为 state_dict。好处：
- 跨端只依赖 numpy 的二进制格式，体积小（压缩）、加载快；
- 不用 pickle，避免反序列化安全风险（allow_pickle=False）。
"""

import io
from collections import OrderedDict

import numpy as np
import torch


def state_dict_to_bytes(state_dict):
    """torch state_dict -> npz 压缩字节流。"""
    arrays = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def bytes_to_state_dict(data):
    """npz 压缩字节流 -> OrderedDict[str, Tensor]。"""
    buf = io.BytesIO(data)
    npz = np.load(buf, allow_pickle=False)
    sd = OrderedDict()
    for key in npz.files:
        sd[key] = torch.from_numpy(npz[key].copy())
    return sd
