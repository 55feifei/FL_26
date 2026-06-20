"""非理想场景研究：Non-IID 与样本不均衡对 FedAvg 收敛的影响（单机仿真，无需网络）。

实验设计（默认 2 个虚拟客户端，模拟 2 台树莓派）：

  场景 1  iid_balanced     — IID 均衡基准（50% / 50%）
  场景 2  noniid_dir05     — Non-IID Dirichlet α=0.5（中度异质，数量均衡）
  场景 3  noniid_dir01     — Non-IID Dirichlet α=0.1（高度异质，数量均衡）
  场景 4  iid_imb_9_1      — IID 内容 + 9:1 样本不均衡
  场景 5  iid_imb_99_1     — IID 内容 + 99:1 极度不均衡
  场景 6  noniid_imb       — Non-IID（α=0.1）+ 9:1 数量不均衡（双重非理想）

输出：
  results/noniid_research/{场景名}.csv      每轮准确率 / 损失
  results/noniid_research/summary.csv       汇总对比表
  results/noniid_research/comparison.png    多场景准确率 & 损失对比曲线
  results/noniid_research/label_dist.png    各场景标签分布热图

用法：
  python scripts/simulate_noniid.py
  python scripts/simulate_noniid.py --rounds 30 --model cnn --local-epochs 2
  python scripts/simulate_noniid.py --scenarios iid_balanced noniid_dir01
"""

import os
import sys
import csv
import copy
import math
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from common.config import Config
from common.model import build_model
from common.data import (
    load_mnist, make_loader,
    partition_iid, partition_noniid,
    partition_imbalanced, partition_noniid_dirichlet,
)
from server.aggregate import fedavg


# ──────────────────────────────────────────────────────────────────
# 场景定义
# ──────────────────────────────────────────────────────────────────

# 每个场景用 dict 描述，字段说明：
#   name    : 文件名 / 命令行筛选标识
#   label   : 图例文字
#   kind    : "iid" | "noniid_dirichlet" | "noniid_shard"
#   alpha   : Dirichlet α（kind=noniid_dirichlet 时有效）
#   cpc     : classes_per_client（kind=noniid_shard 时有效）
#   ratios  : 各客户端样本比例（None → 均匀）

ALL_SCENARIOS = [
    {
        "name": "iid_balanced",
        "label": "IID 均衡（基准）",
        "kind": "iid",
        "alpha": None, "cpc": None,
        "ratios": None,
    },
    {
        "name": "noniid_dir05",
        "label": "Non-IID α=0.5（中度异质）",
        "kind": "noniid_dirichlet",
        "alpha": 0.5, "cpc": None,
        "ratios": None,
    },
    {
        "name": "noniid_dir01",
        "label": "Non-IID α=0.1（高度异质）",
        "kind": "noniid_dirichlet",
        "alpha": 0.1, "cpc": None,
        "ratios": None,
    },
    {
        "name": "iid_imb_9_1",
        "label": "IID + 9:1 数量不均衡",
        "kind": "iid",
        "alpha": None, "cpc": None,
        "ratios": [0.9, 0.1],
    },
    {
        "name": "iid_imb_99_1",
        "label": "IID + 99:1 极度不均衡",
        "kind": "iid",
        "alpha": None, "cpc": None,
        "ratios": [0.99, 0.01],
    },
    {
        "name": "noniid_imb",
        "label": "Non-IID α=0.1 + 9:1 不均衡",
        "kind": "noniid_dirichlet",
        "alpha": 0.1, "cpc": None,
        "ratios": [0.9, 0.1],
    },
]


# ──────────────────────────────────────────────────────────────────
# 数据工具
# ──────────────────────────────────────────────────────────────────

def build_client_subset(train_full, sc, num_clients, client_id, seed):
    """根据场景描述，为指定客户端构建数据子集。"""
    kind = sc["kind"]
    ratios = sc["ratios"]

    if kind == "iid":
        if ratios is None:
            return partition_iid(train_full, num_clients, client_id, seed=seed)
        else:
            return partition_imbalanced(train_full, num_clients, client_id, ratios, seed=seed)

    if kind == "noniid_dirichlet":
        subset = partition_noniid_dirichlet(
            train_full, num_clients, client_id,
            alpha=sc["alpha"], seed=seed,
        )
        if ratios is None:
            return subset
        # Non-IID + 数量不均衡：从 Dirichlet 子集中再按比例缩减
        target_n = max(1, int(ratios[client_id] * len(train_full)))
        indices = subset.indices
        rng = np.random.default_rng(seed + client_id + 999)
        if len(indices) > target_n:
            indices = rng.choice(indices, target_n, replace=False).tolist()
        return Subset(train_full, indices)

    if kind == "noniid_shard":
        subset = partition_noniid(
            train_full, num_clients, client_id,
            classes_per_client=sc["cpc"], seed=seed,
        )
        if ratios is None:
            return subset
        target_n = max(1, int(ratios[client_id] * len(train_full)))
        indices = subset.indices
        rng = np.random.default_rng(seed + client_id + 999)
        if len(indices) > target_n:
            indices = rng.choice(indices, target_n, replace=False).tolist()
        return Subset(train_full, indices)

    raise ValueError(f"未知 kind: {kind}")


def label_distribution(subset, num_classes=10):
    """返回数据子集的标签比例向量（长度=num_classes）。"""
    labels = np.asarray(subset.dataset.targets)[subset.indices]
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    total = counts.sum()
    return counts / total if total > 0 else counts


# ──────────────────────────────────────────────────────────────────
# 训练 / 评估
# ──────────────────────────────────────────────────────────────────

def local_train(model, loader, cfg, device):
    """本地训练若干 epoch（复用 client/fl_client.py 的逻辑）。"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = nn.CrossEntropyLoss()
    steps, last_loss = 0, 0.0
    for _ in range(cfg.local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            steps += 1
            last_loss = loss.item()
            if not math.isfinite(last_loss):
                return last_loss, steps, False
            if cfg.local_steps and steps >= cfg.local_steps:
                return last_loss, steps, True
    return last_loss, steps, True


def evaluate(model, test_loader, device):
    """在全局测试集上评估，返回 (accuracy, avg_loss)。"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total, loss_sum / total


# ──────────────────────────────────────────────────────────────────
# 单场景 FedAvg 仿真
# ──────────────────────────────────────────────────────────────────

def run_scenario(sc, train_full, test_loader, cfg, device, num_clients):
    """运行单个非理想场景的完整 FedAvg 仿真。

    返回:
        history       : [{round, accuracy, loss}, ...]（含 round 0 初始值）
        client_sizes  : 各客户端实际样本数列表
        client_dists  : 各客户端标签比例向量列表（用于可视化）
    """
    torch.manual_seed(cfg.seed)
    global_model = build_model(cfg.model, cfg.dataset, channels=cfg.channels).to(device)

    # 构建各客户端数据分片
    subsets = [
        build_client_subset(train_full, sc, num_clients, cid, cfg.seed)
        for cid in range(num_clients)
    ]
    client_sizes = [len(s) for s in subsets]
    client_dists = [label_distribution(s) for s in subsets]
    loaders = [make_loader(s, cfg.batch_size, shuffle=True) for s in subsets]

    # 打印数据统计
    print(f"\n  ┌─ 场景: {sc['label']}")
    for cid in range(num_clients):
        dist = client_dists[cid]
        top = sorted(enumerate(dist), key=lambda x: -x[1])[:5]
        top_str = "  ".join(f"{cls}:{p:.0%}" for cls, p in top)
        print(f"  │  Client {cid}: {client_sizes[cid]:>6} 样本  Top-5类: {top_str}")
    print(f"  │  样本比例 = {':'.join(str(n) for n in client_sizes)}", flush=True)

    # round 0：随机初始化模型基线
    acc0, loss0 = evaluate(global_model, test_loader, device)
    history = [{"round": 0, "accuracy": round(acc0, 4), "loss": round(loss0, 4)}]

    for rnd in range(1, cfg.rounds + 1):
        global_sd = copy.deepcopy(global_model.state_dict())
        updates = []
        for cid in range(num_clients):
            local_model = build_model(cfg.model, cfg.dataset, channels=cfg.channels).to(device)
            local_model.load_state_dict(copy.deepcopy(global_sd))
            local_train(local_model, loaders[cid], cfg, device)
            updates.append((local_model.state_dict(), client_sizes[cid]))

        new_sd, dropped = fedavg(updates)
        if new_sd is not None:
            global_model.load_state_dict(new_sd)
        if dropped:
            print(f"  │  [Round {rnd}] 剔除 {dropped} 个含 NaN 的更新", flush=True)

        acc, loss = evaluate(global_model, test_loader, device)
        history.append({"round": rnd, "accuracy": round(acc, 4), "loss": round(loss, 4)})
        print(f"  │  Round {rnd:>2}: acc={acc:.4f}  loss={loss:.4f}", flush=True)

    print(f"  └─ 完成  最终准确率={history[-1]['accuracy']:.4f}", flush=True)
    return history, client_sizes, client_dists


# ──────────────────────────────────────────────────────────────────
# 结果保存
# ──────────────────────────────────────────────────────────────────

def save_scenario_csv(history, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "loss"])
        writer.writeheader()
        writer.writerows(history)


def save_summary_csv(scenarios, all_results, all_sizes, path, threshold=0.90):
    """汇总各场景最终指标，写入 summary.csv。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "场景", "样本分布",
            "最终准确率", "最高准确率",
            f"首次达{threshold:.0%}的轮次",
            "最终损失",
        ])
        for sc, hist, sizes in zip(scenarios, all_results, all_sizes):
            final_acc = hist[-1]["accuracy"]
            max_acc = max(h["accuracy"] for h in hist)
            final_loss = hist[-1]["loss"]
            rnd_thresh = next(
                (h["round"] for h in hist if h["accuracy"] >= threshold), -1
            )
            rnd_thresh_str = str(rnd_thresh) if rnd_thresh >= 0 else "未达到"
            ratio_str = ":".join(str(s) for s in sizes)
            writer.writerow([
                sc["label"], ratio_str,
                f"{final_acc:.4f}", f"{max_acc:.4f}",
                rnd_thresh_str, f"{final_loss:.4f}",
            ])


# ──────────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────────

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 尝试加载支持中文的字体（Windows / macOS / Linux 均兼容的回退顺序）
    matplotlib.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


COLORS = ["#2e86de", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad", "#16a085"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]


def plot_comparison(scenarios, all_results, out_dir):
    """绘制多场景准确率 & 损失对比曲线（2 列，1 行）。"""
    try:
        plt = _setup_matplotlib()
    except Exception as e:
        print(f"matplotlib 不可用，跳过绘图：{e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_acc, ax_loss = axes

    for i, (sc, hist) in enumerate(zip(scenarios, all_results)):
        rounds = [h["round"] for h in hist]
        acc = [h["accuracy"] for h in hist]
        loss = [h["loss"] for h in hist]
        kw = dict(
            color=COLORS[i % len(COLORS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=4, linewidth=1.8,
            label=sc["label"],
        )
        ax_acc.plot(rounds, acc, **kw)
        ax_loss.plot(rounds, loss, **kw)

    for ax, ylabel, title in [
        (ax_acc, "准确率 (Accuracy)", "FedAvg 全局准确率 — 非理想场景对比"),
        (ax_loss, "平均损失 (Loss)", "FedAvg 全局损失 — 非理想场景对比"),
    ]:
        ax.set_xlabel("通信轮次 (Round)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
    ax_acc.set_ylim(0, 1.05)

    fig.tight_layout(pad=2.0)
    out = os.path.join(out_dir, "comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n对比曲线已保存：{out}")


def plot_label_distribution(scenarios, all_dists, num_clients, out_dir):
    """为每个场景绘制各客户端的标签分布热图（子图网格）。"""
    try:
        plt = _setup_matplotlib()
        import matplotlib.gridspec as gridspec
    except Exception as e:
        print(f"matplotlib 不可用，跳过标签分布图：{e}")
        return

    n_sc = len(scenarios)
    fig = plt.figure(figsize=(4 * num_clients * n_sc // 3 + 2, 6))
    gs = gridspec.GridSpec(1, n_sc, figure=fig, wspace=0.4)

    num_classes = 10
    digits = list(range(num_classes))

    for col, (sc, dists) in enumerate(zip(scenarios, all_dists)):
        ax = fig.add_subplot(gs[0, col])
        data = np.array(dists)   # shape: (num_clients, num_classes)
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=data.max())
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels([str(d) for d in digits], fontsize=8)
        ax.set_yticks(range(num_clients))
        ax.set_yticklabels([f"Client {i}" for i in range(num_clients)], fontsize=8)
        ax.set_title(sc["label"], fontsize=8, fontweight="bold", wrap=True)
        ax.set_xlabel("数字类别 (0–9)", fontsize=8)
        # 在格子内标注百分比（仅大于 5% 时显示）
        for r in range(num_clients):
            for c in range(num_classes):
                val = data[r, c]
                if val >= 0.05:
                    ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                            fontsize=6.5, color="black" if val < 0.6 else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("各场景客户端标签分布（Non-IID 程度可视化）",
                 fontsize=13, fontweight="bold", y=1.02)
    try:
        fig.tight_layout()
    except Exception:
        pass   # gridspec 与 tight_layout 偶有不兼容，忽略
    out = os.path.join(out_dir, "label_dist.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"标签分布热图已保存：{out}")


def plot_convergence_speed(scenarios, all_results, out_dir, threshold=0.90):
    """额外绘制两张图：① 最终准确率柱状图；② 首次达阈值轮次柱状图。"""
    try:
        plt = _setup_matplotlib()
    except Exception as e:
        print(f"matplotlib 不可用，跳过收敛速度图：{e}")
        return

    labels = [sc["label"] for sc in scenarios]
    final_accs = [h[-1]["accuracy"] for h in all_results]
    rounds_to_thresh = []
    for hist in all_results:
        rnd = next((h["round"] for h in hist if h["accuracy"] >= threshold), None)
        rounds_to_thresh.append(rnd)

    x = np.arange(len(labels))
    colors = COLORS[: len(labels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # ── 最终准确率 ──
    bars = ax1.bar(x, final_accs, color=colors, width=0.6, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax1.set_ylabel("最终准确率 (Accuracy)", fontsize=11)
    ax1.set_title(f"各场景最终准确率（第 {all_results[0][-1]['round']} 轮）",
                  fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.08)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, final_accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── 首次达阈值轮次 ──
    disp_rounds = [r if r is not None else (all_results[0][-1]["round"] + 2)
                   for r in rounds_to_thresh]
    bars2 = ax2.bar(x, disp_rounds, color=colors, width=0.6, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax2.set_ylabel("轮次 (Round)", fontsize=11)
    ax2.set_title(f"首次准确率 ≥ {threshold:.0%} 所需轮次（越少越好）",
                  fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar, rnd in zip(bars2, rounds_to_thresh):
        txt = str(rnd) if rnd is not None else "未达到"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 txt, ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout(pad=2.0)
    out = os.path.join(out_dir, "convergence_speed.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"收敛速度对比图已保存：{out}")


# ──────────────────────────────────────────────────────────────────
# 控制台汇总表格
# ──────────────────────────────────────────────────────────────────

def print_summary(scenarios, all_results, all_sizes, threshold=0.90):
    col_w = [26, 14, 12, 12, 14, 12]
    header = ["场景", "样本分布", "最终准确率", "最高准确率",
              f"达{threshold:.0%}轮次", "最终损失"]
    sep = "─" * (sum(col_w) + len(col_w) * 3)

    print(f"\n{'═' * len(sep)}")
    print("  FedAvg 非理想场景研究 — 汇总")
    print(f"{'═' * len(sep)}")
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    print(f"  {sep}")

    for sc, hist, sizes in zip(scenarios, all_results, all_sizes):
        final_acc = hist[-1]["accuracy"]
        max_acc = max(h["accuracy"] for h in hist)
        final_loss = hist[-1]["loss"]
        rnd_thresh = next((h["round"] for h in hist if h["accuracy"] >= threshold), None)
        rnd_str = str(rnd_thresh) if rnd_thresh is not None else "未达到"
        ratio_str = ":".join(str(s) for s in sizes)
        row = [
            sc["label"], ratio_str,
            f"{final_acc:.4f}", f"{max_acc:.4f}",
            rnd_str, f"{final_loss:.4f}",
        ]
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    print(f"{'═' * len(sep)}\n")


# ──────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Non-IID & 样本不均衡对 FedAvg 影响研究（单机仿真）")
    ap.add_argument("--rounds", type=int, default=30,
                    help="通信轮数（默认 30，更长轮次能充分展现收敛差异）")
    ap.add_argument("--num-clients", type=int, default=2,
                    help="虚拟客户端数量（默认 2，对应 2 台树莓派）")
    ap.add_argument("--local-epochs", type=int, default=1,
                    help="每轮本地训练 epoch 数（默认 1）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--model", default="mlp", choices=["mlp", "cnn"],
                    help="模型（默认 mlp，在 PC 上测试 cnn 速度也快）")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="./results/noniid_research")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="仅运行指定场景（场景名用空格分隔），默认运行全部 6 个场景。"
                         f"可选: {[s['name'] for s in ALL_SCENARIOS]}")
    return ap.parse_args()


def main():
    args = parse_args()

    # 筛选场景
    if args.scenarios:
        scenarios = [s for s in ALL_SCENARIOS if s["name"] in args.scenarios]
        if not scenarios:
            print(f"[错误] 未找到场景：{args.scenarios}，"
                  f"可选：{[s['name'] for s in ALL_SCENARIOS]}")
            sys.exit(1)
    else:
        scenarios = ALL_SCENARIOS

    cfg = Config(
        rounds=args.rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model=args.model,
        data_dir=args.data_dir,
        seed=args.seed,
    )

    device = torch.device("cpu")
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Non-IID & 样本不均衡对 FedAvg 影响研究（单机仿真）")
    print("=" * 60)
    print(f"  模型={cfg.model}  轮数={cfg.rounds}  "
          f"本地Epoch={cfg.local_epochs}  BatchSize={cfg.batch_size}")
    print(f"  客户端数={args.num_clients}  场景数={len(scenarios)}")
    print(f"  输出目录：{out_dir}")
    print("=" * 60, flush=True)

    # 加载数据集（所有场景共享，避免重复下载）
    print("\n正在加载 MNIST ...", flush=True)
    train_full = load_mnist(cfg.data_dir, train=True, download=True, channels=cfg.channels)
    test_set = load_mnist(cfg.data_dir, train=False, download=True, channels=cfg.channels)
    test_loader = make_loader(test_set, cfg.eval_batch_size, shuffle=False)
    print(f"训练集 {len(train_full)} 样本，测试集 {len(test_set)} 样本\n", flush=True)

    # 逐场景运行
    all_results, all_sizes, all_dists = [], [], []
    t_total_start = time.time()

    for idx, sc in enumerate(scenarios):
        print(f"\n{'─' * 60}")
        print(f"  场景 {idx + 1}/{len(scenarios)}: {sc['name']}", flush=True)
        t0 = time.time()

        history, sizes, dists = run_scenario(
            sc, train_full, test_loader, cfg, device, args.num_clients
        )

        elapsed = time.time() - t0
        print(f"  用时 {elapsed:.1f}s", flush=True)

        all_results.append(history)
        all_sizes.append(sizes)
        all_dists.append(dists)

        # 保存单场景 CSV
        csv_path = os.path.join(out_dir, f"{sc['name']}.csv")
        save_scenario_csv(history, csv_path)
        print(f"  CSV 已保存：{csv_path}")

    total_elapsed = time.time() - t_total_start
    print(f"\n{'═' * 60}")
    print(f"  全部场景完成，总用时 {total_elapsed:.1f}s", flush=True)

    # 汇总 CSV
    summary_path = os.path.join(out_dir, "summary.csv")
    save_summary_csv(scenarios, all_results, all_sizes, summary_path)
    print(f"  汇总 CSV 已保存：{summary_path}")

    # 打印控制台汇总表
    print_summary(scenarios, all_results, all_sizes)

    # 可视化
    print("正在生成可视化图表 ...", flush=True)
    plot_comparison(scenarios, all_results, out_dir)
    plot_label_distribution(scenarios, all_dists, args.num_clients, out_dir)
    plot_convergence_speed(scenarios, all_results, out_dir)

    print(f"\n所有输出文件位于：{os.path.abspath(out_dir)}")
    print("仿真完成！", flush=True)


if __name__ == "__main__":
    main()
