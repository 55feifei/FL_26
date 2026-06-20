"""联邦学习客户端（运行在每台树莓派上，作为边缘节点）。

工作循环（每一轮）：
  1. GET /get_model  拉取全局模型与当前轮次号；
  2. 仅当出现"新的一轮"时，把全局权重载入本地模型，在【本地私有数据分片】上训练 E 个 epoch；
  3. POST /submit_update  上传本地训练后的模型参数 + 本地样本数；
  4. 轮询等待服务器聚合并进入下一轮（barrier），然后回到第 1 步。

原始数据始终留在本地，只有模型参数离开设备 —— 这就是联邦学习的核心。

运行（在 fl 目录下）：
  python -m client.fl_client --server http://<PC_IP>:5000 --client-id 0
  python -m client.fl_client --server http://<PC_IP>:5000 --client-id 1
"""

import os
import sys
import math
import time
import argparse

# 允许 `python client/fl_client.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import numpy as np
import torch

from common.config import Config
from common.model import build_model
from common.data import (
    load_mnist, make_loader,
    partition_iid, partition_noniid,
    partition_noniid_dirichlet, partition_imbalanced,
)
from common.serialize import state_dict_to_bytes, bytes_to_state_dict


def local_train(model, loader, cfg, device):
    """在本地数据上训练若干 epoch（或 local_steps 个 batch）。

    返回 (最后loss, 步数, finite)：finite 表示训练后模型参数是否仍为有限值。
    若中途 loss 变成 NaN/Inf（常见于 armv7l 树莓派的非官方 torch 构建），立即中断。
    加入梯度裁剪以抑制梯度爆炸导致的 NaN。
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    steps, last_loss, finite = 0, 0.0, True
    for _ in range(cfg.local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 抑制梯度爆炸
            optimizer.step()
            steps += 1
            last_loss = loss.item()
            if not math.isfinite(last_loss):
                print(f"    !! 第 {steps} 步 loss={last_loss}（NaN/Inf），中断本地训练", flush=True)
                finite = False
                return last_loss, steps, finite
            if cfg.local_steps and steps >= cfg.local_steps:
                return last_loss, steps, finite
    return last_loss, steps, finite


def parse_ratios(text):
    """把 '0.9,0.1' 解析成 [0.9, 0.1]。"""
    try:
        return [float(x) for x in text.split(",")]
    except ValueError:
        raise SystemExit(f"--ratios 解析失败：'{text}'，应形如 0.9,0.1")


def validate_partition(partition, num_clients, ratios):
    """对依赖额外参数的划分方式做前置校验，参数不合法时给出明确报错。"""
    if partition == "imbalanced":
        if ratios is None:
            raise SystemExit("--partition imbalanced 需配合 --ratios，如 --ratios 0.9,0.1")
        if len(ratios) != num_clients:
            raise SystemExit(f"--ratios 长度({len(ratios)})须等于 --num-clients({num_clients})")
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise SystemExit(f"--ratios 之和应为 1.0（当前为 {sum(ratios)}）")


def describe_partition(cfg):
    """生成人类可读的划分描述，便于在多台 Pi 上核对参数是否一致。"""
    if cfg.partition in ("shard", "noniid"):
        return f"shard(每客户端{cfg.classes_per_client}类)"
    if cfg.partition == "dirichlet":
        return f"dirichlet(alpha={cfg.alpha})"
    if cfg.partition == "imbalanced":
        return f"imbalanced(ratios={cfg.ratios})"
    return "iid"


def build_local_loader(cfg, client_id):
    """加载完整数据集后，按 client_id 取出本机的私有分片。

    支持四种划分方式（由 cfg.partition 选择），各客户端必须用【相同的划分参数与
    seed】，才能保证分片确定、互不重叠：
      iid        —— 随机等分（内容与数量都均衡）
      shard      —— McMahan shard 法，每客户端只见 classes_per_client 个类别
      dirichlet  —— Dirichlet(alpha) 软 Non-IID，alpha 越小越异质
      imbalanced —— 内容 IID 但样本数量按 ratios 不均衡
    返回 (loader, 本地样本数, 本地标签分布)。
    """
    train_set = load_mnist(cfg.data_dir, train=True, download=True, channels=cfg.channels)
    p = cfg.partition
    if p in ("shard", "noniid"):
        subset = partition_noniid(train_set, cfg.num_clients, client_id,
                                  classes_per_client=cfg.classes_per_client, seed=cfg.seed)
    elif p == "dirichlet":
        subset = partition_noniid_dirichlet(train_set, cfg.num_clients, client_id,
                                            alpha=cfg.alpha, seed=cfg.seed)
    elif p == "imbalanced":
        subset = partition_imbalanced(train_set, cfg.num_clients, client_id,
                                      ratios=list(cfg.ratios), seed=cfg.seed)
    else:
        subset = partition_iid(train_set, cfg.num_clients, client_id, seed=cfg.seed)
    loader = make_loader(subset, cfg.batch_size, shuffle=True)

    # 统计本地标签分布，便于核对 Non-IID 效果
    labels = np.asarray(train_set.targets)[subset.indices]
    uniq, cnts = np.unique(labels, return_counts=True)
    label_dist = {int(k): int(v) for k, v in zip(uniq, cnts)}
    return loader, len(subset), label_dist


def run(args):
    ratios = parse_ratios(args.ratios) if args.ratios else None
    validate_partition(args.partition, args.num_clients, ratios)

    cfg = Config(
        num_clients=args.num_clients, local_epochs=args.local_epochs,
        local_steps=args.local_steps, batch_size=args.batch_size, lr=args.lr,
        model=args.model, channels=args.channels, dataset=args.dataset,
        partition=args.partition, data_dir=args.data_dir, seed=args.seed,
        classes_per_client=args.classes_per_client, alpha=args.alpha, ratios=ratios,
    )
    device = torch.device("cpu")
    if getattr(args, "threads", 0) and args.threads > 0:
        torch.set_num_threads(args.threads)
        print(f"[client {args.client_id}] torch 线程数设为 {args.threads}", flush=True)

    print(f"[client {args.client_id}] 准备本地数据分片 ...", flush=True)
    loader, n_local, label_dist = build_local_loader(cfg, args.client_id)
    part_desc = describe_partition(cfg)  # 上报给服务器看板展示
    print(f"[client {args.client_id}] 本地样本数={n_local}  服务器={args.server}  "
          f"模型={cfg.model}  分布={part_desc}", flush=True)
    print(f"[client {args.client_id}] 本地标签分布={label_dist}", flush=True)

    model = build_model(cfg.model, cfg.dataset, channels=cfg.channels).to(device)
    last_trained = 0  # 已经完成训练的最大轮次

    while True:
        # 1) 拉取全局模型（顺带上报自身信息，让看板在首轮训练前就能显示已连接的客户端）
        try:
            resp = requests.get(
                f"{args.server}/get_model",
                params={"client_id": args.client_id,
                        "partition": part_desc, "num_samples": n_local},
                timeout=120,
            )
        except requests.RequestException as e:
            print(f"[client {args.client_id}] 连不上服务器，5s 后重试：{e}", flush=True)
            time.sleep(5)
            continue

        rnd = int(resp.headers.get("X-FL-Round", "0"))
        done = resp.headers.get("X-FL-Done", "0") == "1"

        if done:
            print(f"[client {args.client_id}] 服务器已完成全部轮次，退出。", flush=True)
            break
        if rnd <= last_trained:
            # 服务器尚未进入新一轮（正在等待其它客户端 / 正在聚合）——等待（barrier）
            time.sleep(args.poll_interval)
            continue

        # 2) 载入全局权重，在本地分片上训练
        model.load_state_dict(bytes_to_state_dict(resp.content))
        t0 = time.time()
        loss, steps, finite = local_train(model, loader, cfg, device)
        dt = time.time() - t0
        print(f"[client {args.client_id}] 第 {rnd} 轮：本地训练 {steps} 步，"
              f"loss={loss:.4f}，用时 {dt:.1f}s", flush=True)
        if not finite:
            print(f"[client {args.client_id}] [警告] 本地训练产生 NaN/Inf！这通常是本机 PyTorch 构建问题，"
                  f"请在树莓派上运行  python scripts/check_torch.py  自检"
                  f"（可尝试 --model mlp 或设 torch 单线程）。", flush=True)
            # 仍上传（服务器会剔除坏更新并保留旧模型），以免阻塞 barrier

        # 3) 上传本地模型参数
        body = state_dict_to_bytes(model.state_dict())
        try:
            r = requests.post(
                f"{args.server}/submit_update",
                params={"client_id": args.client_id, "round": rnd, "num_samples": n_local},
                data=body, timeout=120,
            )
            info = r.json()
        except requests.RequestException as e:
            print(f"[client {args.client_id}] 上传失败，3s 后重试：{e}", flush=True)
            time.sleep(3)
            continue

        if info.get("status") == "stale":
            # 轮次已过期，重新同步（不更新 last_trained）
            print(f"[client {args.client_id}] 轮次过期，重新同步。", flush=True)
            continue

        last_trained = rnd
        if info.get("done"):
            print(f"[client {args.client_id}] 服务器通知训练完成，退出。", flush=True)
            break
        # 其余情况：回到循环顶部，由 barrier 等待下一轮


def main():
    ap = argparse.ArgumentParser(description="联邦学习客户端（树莓派边缘节点）")
    ap.add_argument("--server", default="http://127.0.0.1:5000", help="服务器地址，如 http://192.168.1.100:5000")
    ap.add_argument("--client-id", type=int, required=True, help="客户端编号 0..N-1（每台树莓派一个）")
    ap.add_argument("--num-clients", type=int, default=2)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--local-steps", type=int, default=0, help=">0 时每轮只训练这么多 batch（控时）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--model", default="mlp", choices=["cnn", "mlp"],
                    help="默认 mlp（部分 armv7l 树莓派 torch 的 conv 不可靠）；torch 正常时可用 cnn")
    ap.add_argument("--channels", type=int, default=1, choices=[1, 3],
                    help="输入通道数（须与服务器一致）；用 CNN 时在 armv7l 树莓派上设 3 绕开单通道卷积 bug")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--partition", default="iid",
                    choices=["iid", "shard", "noniid", "dirichlet", "imbalanced"],
                    help="数据划分方式：iid | shard(=noniid) | dirichlet | imbalanced（须三端一致）")
    ap.add_argument("--classes-per-client", type=int, default=2,
                    help="shard 方式：每客户端分到的类别数（1 极度异质 … 越大越接近 IID）")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="dirichlet 方式：浓度参数（0.1 高度异质 / 0.5 中度 / 越大越接近 IID）")
    ap.add_argument("--ratios", type=str, default=None,
                    help="imbalanced 方式：各客户端样本比例，逗号分隔且和为 1，如 0.9,0.1")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0,
                    help=">0 时设置 torch 线程数；armv7l 树莓派 conv 偶发 NaN 时可设 1")
    ap.add_argument("--poll-interval", type=float, default=2.0, help="等待下一轮的轮询间隔(秒)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
