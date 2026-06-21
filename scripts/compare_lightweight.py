"""轻量化网络对比研究：MobileNet / SqueezeNet vs 普通 CNN 在受限硬件上的训练效率（发挥任务三）。

任务书要求："对比研究轻量化网络（如 MobileNet、SqueezeNet）与普通卷积网络（CNN）
在树莓派硬件资源受限条件下的训练效率差异。"

本脚本在一台机器上模拟 2 客户端 FedAvg 闭环（IID 等分 MNIST），对三种模型并排训练，
统一记录：
  - 参数量（直接 sum(p.numel())）
  - 模型大小 KB（state_dict 经 npz 压缩后字节数，即真实通信开销）
  - 每轮 wall-clock 训练耗时（含本地训练 + 评估）
  - 每轮全局准确率 / 损失（FedAvg 后在测试集上评估）
  - 首次达 90% 准确率所需轮次

输出（写到 results/lightweight_compare/）：
  {model}.csv          每轮 round / accuracy / loss / round_sec
  summary.csv          汇总对比表
  efficiency.png       2x2 子图：参数量/模型大小/每轮耗时柱状图 + 准确率收敛曲线
  convergence.png      准确率 vs 累计训练时间（揭示"同样时间预算下谁更强"）

关键开关 --threads 1：在 PC 上把 PyTorch 线程压到 1，近似还原树莓派单核负载，
让"训练耗时"指标真正可类比 Pi 真机。默认 0（不限制）。

用法：
  python scripts/compare_lightweight.py                            # 默认 cnn+mobilenet+squeezenet, 15 轮
  python scripts/compare_lightweight.py --threads 1 --rounds 15    # 模拟 Pi 单核
  python scripts/compare_lightweight.py --models cnn mobilenet     # 仅跑指定模型
  python scripts/compare_lightweight.py --rounds 5 --local-steps 30  # 快速验证
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

from common.config import Config
from common.model import build_model
from common.data import load_mnist, make_loader, partition_iid
from common.serialize import state_dict_to_bytes
from server.aggregate import fedavg


ALL_MODELS = ["cnn", "mobilenet", "squeezenet"]
MODEL_LABEL = {
    "cnn": "SimpleCNN（基准）",
    "mobilenet": "MobileNet（深度可分离卷积）",
    "squeezenet": "SqueezeNet（Fire 模块）",
}


# ─────────────────────────────────────────────────────────────
# 训练 / 评估（与 simulate_noniid.py 同款，含 NaN 中断）
# ─────────────────────────────────────────────────────────────

def local_train(model, loader, cfg, device):
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


# ─────────────────────────────────────────────────────────────
# 单模型 FedAvg 仿真（带每轮计时）
# ─────────────────────────────────────────────────────────────

def run_model_benchmark(model_name, train_full, test_loader, cfg, device, num_clients):
    """对单个模型跑完整 FedAvg 仿真，返回 (history, params, size_bytes, total_sec)。

    history: [{round, accuracy, loss, round_sec}, ...]（含 round 0 基线）
    每个 round_sec 包含本轮所有客户端的本地训练 + FedAvg 聚合 + 全局评估的时间。
    """
    torch.manual_seed(cfg.seed)
    global_model = build_model(model_name, cfg.dataset, channels=cfg.channels, norm=cfg.norm).to(device)

    # 统计静态指标（与训练流程无关）
    n_params = sum(p.numel() for p in global_model.parameters())
    size_bytes = len(state_dict_to_bytes(global_model.state_dict()))

    # 数据分片（IID 等分，与场景对比脚本对齐）
    subsets = [partition_iid(train_full, num_clients, cid, seed=cfg.seed)
               for cid in range(num_clients)]
    client_sizes = [len(s) for s in subsets]
    loaders = [make_loader(s, cfg.batch_size, shuffle=True) for s in subsets]

    print(f"\n  ┌─ 模型: {MODEL_LABEL[model_name]}")
    print(f"  │  参数量 = {n_params:,}    模型大小 = {size_bytes / 1024:.1f} KB    "
          f"客户端样本 = {client_sizes}", flush=True)

    # round 0：随机初始化基线
    acc0, loss0 = evaluate(global_model, test_loader, device)
    history = [{"round": 0, "accuracy": round(acc0, 4),
                "loss": round(loss0, 4), "round_sec": 0.0}]

    t_total_start = time.time()
    for rnd in range(1, cfg.rounds + 1):
        t_round_start = time.time()
        global_sd = copy.deepcopy(global_model.state_dict())
        updates = []
        for cid in range(num_clients):
            local_model = build_model(model_name, cfg.dataset, channels=cfg.channels,
                                      norm=cfg.norm).to(device)
            local_model.load_state_dict(copy.deepcopy(global_sd))
            local_train(local_model, loaders[cid], cfg, device)
            updates.append((local_model.state_dict(), client_sizes[cid]))

        new_sd, dropped = fedavg(updates)
        if new_sd is not None:
            global_model.load_state_dict(new_sd)
        if dropped:
            print(f"  │  [Round {rnd}] 剔除 {dropped} 个含 NaN 的更新", flush=True)

        acc, loss = evaluate(global_model, test_loader, device)
        round_sec = time.time() - t_round_start
        history.append({"round": rnd, "accuracy": round(acc, 4),
                        "loss": round(loss, 4), "round_sec": round(round_sec, 2)})
        print(f"  │  Round {rnd:>2}: acc={acc:.4f}  loss={loss:.4f}  "
              f"用时 {round_sec:.1f}s", flush=True)

    total_sec = time.time() - t_total_start
    print(f"  └─ 完成  最终准确率={history[-1]['accuracy']:.4f}  "
          f"训练总耗时={total_sec:.1f}s", flush=True)
    return history, n_params, size_bytes, total_sec


# ─────────────────────────────────────────────────────────────
# 结果保存
# ─────────────────────────────────────────────────────────────

def save_model_csv(history, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "loss", "round_sec"])
        writer.writeheader()
        writer.writerows(history)


def save_summary_csv(models, histories, params_list, sizes, totals, path, threshold=0.90):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "模型", "参数量", "模型大小(KB)", "每轮平均耗时(s)",
            "训练总耗时(s)", "最终准确率", "最高准确率",
            f"首次达{threshold:.0%}的轮次",
        ])
        for name, hist, p, sz, tot in zip(models, histories, params_list, sizes, totals):
            # round_sec 在 round 0 为 0，统计平均时排除掉
            train_rounds = [h for h in hist if h["round"] > 0]
            avg_sec = sum(h["round_sec"] for h in train_rounds) / max(1, len(train_rounds))
            final_acc = hist[-1]["accuracy"]
            max_acc = max(h["accuracy"] for h in hist)
            rnd_thresh = next(
                (h["round"] for h in hist if h["accuracy"] >= threshold), -1)
            rnd_str = str(rnd_thresh) if rnd_thresh >= 0 else "未达到"
            writer.writerow([
                MODEL_LABEL[name], p, f"{sz / 1024:.1f}", f"{avg_sec:.2f}",
                f"{tot:.1f}", f"{final_acc:.4f}", f"{max_acc:.4f}", rnd_str,
            ])


# ─────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


COLORS = ["#2e86de", "#e74c3c", "#27ae60", "#f39c12"]
MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-.", ":"]


def plot_efficiency(models, histories, params_list, sizes, out_dir):
    """2x2 子图：参数量 / 模型大小 / 每轮耗时柱状图 + 准确率收敛曲线。"""
    try:
        plt = _setup_matplotlib()
    except Exception as e:
        print(f"matplotlib 不可用，跳过 efficiency 图：{e}")
        return

    labels = [MODEL_LABEL[m] for m in models]
    colors = COLORS[: len(models)]
    x = np.arange(len(models))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_params, ax_size, ax_time, ax_acc = axes.flatten()

    # ① 参数量
    params_k = [p / 1000 for p in params_list]
    bars = ax_params.bar(x, params_k, color=colors, width=0.6, edgecolor="white")
    ax_params.set_xticks(x)
    ax_params.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax_params.set_ylabel("参数量 (千 / K)", fontsize=11)
    ax_params.set_title("① 模型参数量（越少越轻量）", fontsize=12, fontweight="bold")
    ax_params.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, params_k):
        ax_params.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + max(params_k) * 0.02,
                       f"{val:.1f}k", ha="center", va="bottom",
                       fontsize=10, fontweight="bold")

    # ② 模型大小（KB）
    sizes_kb = [s / 1024 for s in sizes]
    bars = ax_size.bar(x, sizes_kb, color=colors, width=0.6, edgecolor="white")
    ax_size.set_xticks(x)
    ax_size.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax_size.set_ylabel("模型大小 (KB)", fontsize=11)
    ax_size.set_title("② 模型序列化体积（越小通信越省）", fontsize=12, fontweight="bold")
    ax_size.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, sizes_kb):
        ax_size.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(sizes_kb) * 0.02,
                     f"{val:.1f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

    # ③ 每轮平均耗时
    avg_secs = []
    for hist in histories:
        train_rounds = [h for h in hist if h["round"] > 0]
        avg_secs.append(sum(h["round_sec"] for h in train_rounds) / max(1, len(train_rounds)))
    bars = ax_time.bar(x, avg_secs, color=colors, width=0.6, edgecolor="white")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax_time.set_ylabel("每轮平均耗时 (s)", fontsize=11)
    ax_time.set_title("③ 每轮训练耗时（越少越快）", fontsize=12, fontweight="bold")
    ax_time.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, avg_secs):
        ax_time.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(avg_secs) * 0.02,
                     f"{val:.1f}s", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

    # ④ 准确率收敛曲线
    for i, (m, hist) in enumerate(zip(models, histories)):
        rounds = [h["round"] for h in hist]
        acc = [h["accuracy"] for h in hist]
        ax_acc.plot(rounds, acc,
                    color=COLORS[i % len(COLORS)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    marker=MARKERS[i % len(MARKERS)],
                    markersize=4, linewidth=1.8,
                    label=MODEL_LABEL[m])
    ax_acc.set_xlabel("通信轮次 (Round)", fontsize=11)
    ax_acc.set_ylabel("全局准确率", fontsize=11)
    ax_acc.set_title("④ FedAvg 收敛曲线（越快爬升越好）", fontsize=12, fontweight="bold")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=9, loc="lower right")

    fig.suptitle("轻量化网络 vs 普通 CNN — 训练效率对比（MNIST，单机仿真 2 客户端 FedAvg）",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout(pad=2.0)
    out = os.path.join(out_dir, "efficiency.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n效率对比图已保存：{out}")


def plot_convergence_vs_time(models, histories, out_dir):
    """横轴=累计训练时间，纵轴=准确率。揭示"同样时间预算下谁更强"。"""
    try:
        plt = _setup_matplotlib()
    except Exception as e:
        print(f"matplotlib 不可用，跳过 convergence 图：{e}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (m, hist) in enumerate(zip(models, histories)):
        # 累计 wall-clock 时间（round 0 起点为 0）
        cum_t = [0.0]
        for h in hist[1:]:
            cum_t.append(cum_t[-1] + h["round_sec"])
        acc = [h["accuracy"] for h in hist]
        ax.plot(cum_t, acc,
                color=COLORS[i % len(COLORS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                marker=MARKERS[i % len(MARKERS)],
                markersize=5, linewidth=2.0,
                label=MODEL_LABEL[m])
    ax.set_xlabel("累计训练时间 (s, wall-clock)", fontsize=11)
    ax.set_ylabel("全局准确率", fontsize=11)
    ax.set_title("同等时间预算下各模型的收敛对比（横轴=训练秒数）",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    out = os.path.join(out_dir, "convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"训练时间-准确率图已保存：{out}")


# ─────────────────────────────────────────────────────────────
# 控制台汇总表
# ─────────────────────────────────────────────────────────────

def print_summary(models, histories, params_list, sizes, totals, threshold=0.90):
    col_w = [28, 10, 14, 14, 12, 12, 12]
    header = ["模型", "参数量", "模型大小(KB)", "每轮平均(s)",
              "总耗时(s)", "最终准确率", f"达{threshold:.0%}轮次"]
    sep = "─" * (sum(col_w) + len(col_w) * 2)

    print(f"\n{'═' * len(sep)}")
    print("  轻量化网络 vs 普通 CNN — 训练效率对比汇总（MNIST）")
    print(f"{'═' * len(sep)}")
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    print(f"  {sep}")

    for name, hist, p, sz, tot in zip(models, histories, params_list, sizes, totals):
        train_rounds = [h for h in hist if h["round"] > 0]
        avg_sec = sum(h["round_sec"] for h in train_rounds) / max(1, len(train_rounds))
        final_acc = hist[-1]["accuracy"]
        rnd_thresh = next(
            (h["round"] for h in hist if h["accuracy"] >= threshold), None)
        rnd_str = str(rnd_thresh) if rnd_thresh is not None else "未达到"
        row = [
            MODEL_LABEL[name], f"{p:,}", f"{sz / 1024:.1f}",
            f"{avg_sec:.2f}", f"{tot:.1f}",
            f"{final_acc:.4f}", rnd_str,
        ]
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    print(f"{'═' * len(sep)}\n")


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="MobileNet / SqueezeNet vs 普通 CNN 训练效率对比（单机仿真）")
    ap.add_argument("--rounds", type=int, default=15,
                    help="通信轮数（默认 15）")
    ap.add_argument("--num-clients", type=int, default=2,
                    help="虚拟客户端数量（默认 2，对应 2 台树莓派）")
    ap.add_argument("--local-epochs", type=int, default=1,
                    help="每轮本地训练 epoch 数（默认 1）")
    ap.add_argument("--local-steps", type=int, default=60,
                    help=">0 时每轮只训练这么多 batch（默认 60，控时让 PC 上单轮 < 60s）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--channels", type=int, default=3, choices=[1, 3],
                    help="输入通道数（默认 3，与 Pi CNN 实验一致；MobileNet/SqueezeNet 推荐 3）")
    ap.add_argument("--norm", default="group", choices=["batch", "group"],
                    help="归一化方式（默认 group，FL 友好）")
    ap.add_argument("--threads", type=int, default=0,
                    help=">0 时设置 torch.set_num_threads(N)，"
                         "推荐 1 以近似还原 Pi 单核负载")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="./results/lightweight_compare")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="*", default=None,
                    help=f"仅运行指定模型（空格分隔），默认全部：{ALL_MODELS}")
    return ap.parse_args()


def main():
    args = parse_args()

    # 模型筛选
    if args.models:
        models = [m for m in ALL_MODELS if m in args.models]
        unknown = set(args.models) - set(ALL_MODELS)
        if unknown:
            print(f"[警告] 忽略未知模型: {unknown}（可选：{ALL_MODELS}）")
        if not models:
            print(f"[错误] 没有有效模型，可选：{ALL_MODELS}")
            sys.exit(1)
    else:
        models = ALL_MODELS

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    cfg = Config(
        rounds=args.rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        model="cnn",            # 占位，run_model_benchmark 会按 models 列表逐个覆盖
        channels=args.channels,
        dataset="mnist",
        norm=args.norm,
        data_dir=args.data_dir,
        seed=args.seed,
    )

    device = torch.device("cpu")
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print("  轻量化网络对比研究（发挥任务三）— MNIST 单机仿真 FedAvg")
    print("=" * 64)
    pi_hint = "（≈Pi 单核口径）" if args.threads == 1 else "（PC 多核，对比 Pi 请加 --threads 1）"
    print(f"  对比模型: {models}")
    print(f"  轮数={cfg.rounds}  客户端={cfg.num_clients}  "
          f"local_epochs={cfg.local_epochs}  local_steps={cfg.local_steps}  "
          f"batch={cfg.batch_size}")
    print(f"  torch 版本={torch.__version__}  线程数={torch.get_num_threads()} {pi_hint}")
    print(f"  输出目录: {os.path.abspath(out_dir)}")
    print("=" * 64, flush=True)

    # 加载 MNIST（所有模型共享）
    print("\n正在加载 MNIST ...", flush=True)
    train_full = load_mnist(cfg.data_dir, train=True, download=True, channels=cfg.channels)
    test_set = load_mnist(cfg.data_dir, train=False, download=True, channels=cfg.channels)
    test_loader = make_loader(test_set, cfg.eval_batch_size, shuffle=False)
    print(f"训练集 {len(train_full)} 样本，测试集 {len(test_set)} 样本\n", flush=True)

    # 逐模型跑
    histories, params_list, sizes, totals = [], [], [], []
    t_global_start = time.time()
    for idx, name in enumerate(models):
        print(f"\n{'─' * 64}")
        print(f"  模型 {idx + 1}/{len(models)}: {name}", flush=True)
        hist, n_params, size_bytes, total_sec = run_model_benchmark(
            name, train_full, test_loader, cfg, device, cfg.num_clients,
        )
        histories.append(hist)
        params_list.append(n_params)
        sizes.append(size_bytes)
        totals.append(total_sec)

        csv_path = os.path.join(out_dir, f"{name}.csv")
        save_model_csv(hist, csv_path)
        print(f"  CSV 已保存：{csv_path}")

    print(f"\n{'═' * 64}")
    print(f"  全部模型完成，总用时 {time.time() - t_global_start:.1f}s", flush=True)

    # 汇总 CSV + 控制台表
    summary_path = os.path.join(out_dir, "summary.csv")
    save_summary_csv(models, histories, params_list, sizes, totals, summary_path)
    print(f"  汇总 CSV 已保存：{summary_path}")
    print_summary(models, histories, params_list, sizes, totals)

    # 可视化
    print("正在生成可视化图表 ...", flush=True)
    plot_efficiency(models, histories, params_list, sizes, out_dir)
    plot_convergence_vs_time(models, histories, out_dir)

    print(f"\n所有输出文件位于：{os.path.abspath(out_dir)}")
    print("对比完成！", flush=True)


if __name__ == "__main__":
    main()
