"""联邦学习中心服务器（运行在 PC 上）。

职责：
  1. 初始化并持有全局模型，向客户端下发；
  2. 收集各客户端上传的本地模型参数 + 本地样本数；
  3. 收齐后做 FedAvg 加权聚合，更新全局模型；
  4. 在全局测试集上评估，记录准确率/损失到 CSV；
  5. 提供网页看板（GET /）实时显示训练曲线。

同步策略（同步 FedAvg）：用"轮次号 current_round"做 barrier。服务器每轮等齐
num_clients 个更新才聚合并进入下一轮，客户端通过轮次号判断何时该训练、何时该等待。

运行：
  cd fl
  python -m server.fl_server                 # 默认 0.0.0.0:5000, 2 客户端, 15 轮
  python -m server.fl_server --rounds 20 --num-clients 2 --model cnn
"""

import os
import sys
import csv
import time
import threading
import argparse

# 允许 `python server/fl_server.py` 直接运行（把项目根目录加入模块搜索路径）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from flask import Flask, request, Response, jsonify

from common.config import Config
from common.model import build_model
from common.data import load_mnist, make_loader
from common.serialize import state_dict_to_bytes, bytes_to_state_dict
from server.aggregate import fedavg


class FLServer:
    """封装全局模型、轮次状态、聚合与评估逻辑。共享状态由 self.lock 保护。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.device = torch.device("cpu")

        torch.manual_seed(cfg.seed)
        self.model = build_model(cfg.model, cfg.dataset).to(self.device)

        self.current_round = 1          # 当前进行中的轮次（从 1 开始）
        self.done = False               # 是否已完成全部轮次
        self.round_updates = {}         # 本轮已收到的更新 {client_id: (state_dict, n)}
        self.history = []               # [{round, accuracy, loss, time}]
        self.clients = {}               # {client_id: {last_round, num_samples, last_seen}}
        self.start_time = time.time()

        # 全局测试集（统一在服务器评估，保证各轮可比）
        test_set = load_mnist(cfg.data_dir, train=False, download=True)
        self.test_loader = make_loader(test_set, cfg.eval_batch_size, shuffle=False)

        os.makedirs(cfg.results_dir, exist_ok=True)
        self.csv_path = os.path.join(cfg.results_dir, "metrics.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["round", "accuracy", "loss", "elapsed_sec"])

        # 记录 round 0（随机初始化模型）作为曲线起点
        acc0, loss0 = self.evaluate()
        self._record(0, acc0, loss0)

    # ---------- 模型快照 / 评估 ----------
    def snapshot(self):
        """原子地取出 (当前轮次, 是否完成, 客户端数, 权重字节)。"""
        with self.lock:
            return (self.current_round, self.done, self.cfg.num_clients,
                    state_dict_to_bytes(self.model.state_dict()))

    def evaluate(self):
        """在全局测试集上评估当前全局模型，返回 (accuracy, avg_loss)。"""
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss_sum += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / total, loss_sum / total

    def _record(self, rnd, acc, loss):
        elapsed = round(time.time() - self.start_time, 1)
        self.history.append({"round": rnd, "accuracy": round(acc, 4),
                             "loss": round(loss, 4), "time": elapsed})
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([rnd, f"{acc:.4f}", f"{loss:.4f}", elapsed])
        print(f"[Round {rnd:>2}] 全局准确率={acc:.4f}  损失={loss:.4f}  ({elapsed}s)", flush=True)

    # ---------- 接收客户端更新 ----------
    def submit(self, client_id, rnd, num_samples, weights_bytes):
        with self.lock:
            self.clients[client_id] = {
                "last_round": rnd,
                "num_samples": num_samples,
                "last_seen": round(time.time() - self.start_time, 1),
            }
            if self.done:
                return {"status": "done", "done": True}
            if rnd != self.current_round:
                # 客户端轮次与服务器不一致（过期/超前），让它重新同步
                return {"status": "stale", "current_round": self.current_round}

            self.round_updates[client_id] = (bytes_to_state_dict(weights_bytes), num_samples)
            received = len(self.round_updates)
            print(f"  收到 client {client_id} 第 {rnd} 轮更新 "
                  f"(n={num_samples})  [{received}/{self.cfg.num_clients}]", flush=True)

            if received >= self.cfg.num_clients:
                # 收齐 -> FedAvg 聚合 -> 评估 -> 进入下一轮
                new_sd, dropped = fedavg(list(self.round_updates.values()))
                if dropped:
                    print(f"  [警告] 本轮剔除 {dropped} 个含 NaN/Inf 的客户端更新"
                          f"（请检查该客户端树莓派的 PyTorch 是否产生 NaN）", flush=True)
                if new_sd is None:
                    print("  [警告] 本轮全部更新均为 NaN/Inf，保留上一轮全局模型不更新。", flush=True)
                else:
                    self.model.load_state_dict(new_sd)
                self.round_updates = {}
                acc, loss = self.evaluate()
                self._record(self.current_round, acc, loss)
                model_path = self.save_model()   # 每轮覆盖保存最新全局模型（早停也有最新模型）
                if self.current_round >= self.cfg.rounds:
                    self.done = True
                    self.plot_curves()
                    print(f"最终全局模型已保存：{model_path}", flush=True)
                    print("==== 训练完成 ====", flush=True)
                else:
                    self.current_round += 1

            return {"status": "ok", "received": received,
                    "current_round": self.current_round, "done": self.done}

    # ---------- 保存全局模型 ----------
    def save_model(self):
        """保存当前全局模型为 checkpoint（含元信息，便于后续加载推理）。"""
        path = os.path.join(self.cfg.results_dir, "global_model.pth")
        last = self.history[-1] if self.history else {}
        torch.save({
            "state_dict": self.model.state_dict(),
            "model": self.cfg.model,
            "dataset": self.cfg.dataset,
            "round": self.current_round,
            "accuracy": last.get("accuracy"),
        }, path)
        return path

    # ---------- 训练结束出图（供报告用）----------
    def plot_curves(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print("matplotlib 不可用，跳过出图：", e, flush=True)
            return
        rounds = [h["round"] for h in self.history]
        acc = [h["accuracy"] for h in self.history]
        loss = [h["loss"] for h in self.history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(rounds, acc, "-o", color="#2e86de")
        ax1.set_title("Global Accuracy"); ax1.set_xlabel("Round"); ax1.set_ylabel("Accuracy")
        ax1.grid(True, alpha=0.3)
        ax2.plot(rounds, loss, "-o", color="#e74c3c")
        ax2.set_title("Global Loss"); ax2.set_xlabel("Round"); ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(self.cfg.results_dir, "curves.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"训练曲线已保存：{out}", flush=True)


# ================= Flask 应用 =================
app = Flask(__name__)
server = None  # 在 main() 中初始化


@app.route("/get_model", methods=["GET"])
def get_model():
    """下发当前全局模型。轮次/完成标志放在响应头，权重为 npz 二进制 body。"""
    rnd, done, num_clients, body = server.snapshot()
    headers = {
        "X-FL-Round": str(rnd),
        "X-FL-Done": "1" if done else "0",
        "X-FL-Num-Clients": str(num_clients),
    }
    return Response(body, mimetype="application/octet-stream", headers=headers)


@app.route("/submit_update", methods=["POST"])
def submit_update():
    """接收客户端上传的本地模型参数。元数据走查询串，权重为请求 body。"""
    client_id = request.args.get("client_id", type=int)
    rnd = request.args.get("round", type=int)
    num_samples = request.args.get("num_samples", type=int)
    weights = request.get_data()
    if client_id is None or rnd is None or num_samples is None or not weights:
        return jsonify({"error": "缺少参数 client_id/round/num_samples 或权重为空"}), 400
    return jsonify(server.submit(client_id, rnd, num_samples, weights))


@app.route("/status", methods=["GET"])
def status():
    """看板轮询此接口获取训练进度与历史曲线数据。"""
    with server.lock:
        return jsonify({
            "current_round": server.current_round,
            "total_rounds": server.cfg.rounds,
            "done": server.done,
            "model": server.cfg.model,
            "dataset": server.cfg.dataset,
            "num_clients": server.cfg.num_clients,
            "history": server.history,
            "clients": server.clients,
        })


@app.route("/")
def dashboard():
    """返回看板页面（直接读文件，绕过 Jinja，页面零外部依赖）。"""
    path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser(description="联邦学习中心服务器 + 实时看板")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0，允许局域网访问）")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--rounds", type=int, default=15, help="通信轮数 T")
    ap.add_argument("--num-clients", type=int, default=2, help="客户端数量 = 树莓派数量")
    ap.add_argument("--model", default="cnn", choices=["cnn", "mlp"])
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config(
        num_clients=args.num_clients, rounds=args.rounds, model=args.model,
        dataset=args.dataset, data_dir=args.data_dir, results_dir=args.results_dir,
        seed=args.seed, host=args.host, port=args.port,
    )

    global server
    print("正在加载测试集并初始化全局模型 ...", flush=True)
    server = FLServer(cfg)
    print("=" * 48)
    print(f"联邦学习服务器已启动：模型={cfg.model}  数据集={cfg.dataset}")
    print(f"等待 {cfg.num_clients} 个客户端，共进行 {cfg.rounds} 轮 FedAvg")
    print(f"看板地址：http://127.0.0.1:{cfg.port}/  （局域网用本机 IP 替换 127.0.0.1）")
    print("=" * 48, flush=True)
    app.run(host=cfg.host, port=cfg.port, threaded=True)


if __name__ == "__main__":
    main()
