# 基于树莓派的分布式联邦学习系统

> 硬件课程设计 · 在不上传原始数据的前提下，多台树莓派作为边缘节点本地训练 MNIST，
> 仅上传模型参数到中心服务器做 **FedAvg 加权聚合**，并实时可视化全局模型准确率/损失。

---

## 1. 系统简介与原理

联邦学习核心：**“数据不动，模型动”**。原始隐私数据永远留在本地设备，节点之间只交换
模型参数，由中心服务器聚合成更强的全局模型再下发，循环迭代。

```
        ┌──────────────────────────────────────┐
        │  PC = 中心服务器 + 实时看板               │
        │  · 初始化/下发全局模型                    │
        │  · 收集各客户端参数 + 样本数 n_k          │
        │  · FedAvg:  W = Σ (n_k / Σn) · W_k       │
        │  · 全局测试集评估 → 准确率/损失          │
        │  · 网页看板实时画曲线 (http://IP:5000)   │
        └─────────▲──────────────────▲───────────┘
            同一 WiFi / 局域网         │ HTTP
     ┌────────────┘                  └────────────┐
 ┌───┴──────────────┐          ┌──────────────────┴─┐
 │ 树莓派#1 client 0 │          │ 树莓派#2 client 1   │
 │ 本地分片(MNIST一半)│          │ 本地分片(另一半)     │
 │ 本地训练 E 个epoch │          │ 本地训练 E 个epoch  │
 │ 上传参数 + n_k     │          │ 上传参数 + n_k      │
 └───────────────────┘          └─────────────────────┘
```

**一轮（round）的流程**：客户端拉取全局模型 → 在本地私有数据上训练 E 个 epoch →
上传模型参数 → 服务器收齐 2 份后 FedAvg 聚合 → 在测试集评估 → 进入下一轮。
服务器用“轮次号”做同步 barrier，等齐所有客户端才聚合并推进。

---

## 2. 目录结构

```
fl/
├── common/                 # 公共模块
│   ├── config.py           #   超参数与配置
│   ├── model.py            #   SimpleCNN / MLP（PyTorch）
│   ├── data.py             #   torchvision 加载 MNIST + 联邦分片(IID/预留Non-IID)
│   └── serialize.py        #   state_dict <-> npz 字节（网络传输）
├── server/                 # 中心服务器（跑在 PC）
│   ├── fl_server.py        #   Flask 服务 + 评估 + 记录 + 出图
│   ├── aggregate.py        #   FedAvg 加权平均
│   └── templates/
│       └── dashboard.html  #   零依赖网页看板（canvas 实时曲线）
├── client/
│   └── fl_client.py        # 客户端（跑在每台树莓派）
├── scripts/
│   ├── setup_pi.sh         # 树莓派一键装依赖（含 armv7l 版 torch）
│   ├── prepare_data.py     # 预下载 MNIST
│   ├── run_server.bat      # PC 启动服务器
│   └── run_client.sh       # 树莓派启动客户端
├── requirements-pc.txt
├── requirements-pi.txt
└── results/                # 运行后生成 metrics.csv 与 curves.png
```

---

## 3. 环境要求

| 角色              | 设备                      | 关键依赖                                                 |
| ----------------- | ------------------------- | -------------------------------------------------------- |
| 中心服务器 + 看板 | PC（Windows/Linux/Mac）   | Python 3.7、torch、torchvision、flask、numpy、matplotlib |
| 客户端 ×2        | 树莓派 4B（armv7l 32 位） | Python 3.7、torch、torchvision、numpy、requests          |

> 为与树莓派(Python 3.7)一致，PC 推荐用 conda 建 3.7 环境：
> `conda create -n FL python=3.7.3 -y && conda activate FL`

---

## 4. 快速开始：先在 PC 上单机自测（强烈建议）

真机部署前，先在一台 PC 上把整套闭环跑通，确认逻辑无误。

```bash
conda activate FL
cd fl
pip install -r requirements-pc.txt        # 首次
python scripts/prepare_data.py            # 下载 MNIST 到 ./data

# 终端 1：启动服务器（默认 15 轮、2 客户端）
python -m server.fl_server

# 终端 2：客户端 0
python -m client.fl_client --server http://127.0.0.1:5000 --client-id 0
# 终端 3：客户端 1
python -m client.fl_client --server http://127.0.0.1:5000 --client-id 1
```

浏览器打开 **http://127.0.0.1:5000/** 即可看到实时看板。训练结束后在 `results/`
得到 `metrics.csv` 和 `curves.png`。

> 想快点看到效果，可加 `--rounds 5` 给服务器、给客户端加 `--local-steps 50`（每轮只训 50 个 batch）。

---

## 5. 真机部署：PC 服务器 + 2 台树莓派

### 5.1 PC 端（中心服务器）

1. **建环境装依赖**（同上第 4 节）。
2. **下载数据**：`python scripts/prepare_data.py`。
3. **查本机局域网 IP**：Windows 执行 `ipconfig`，记下与树莓派同一网段的 IPv4（如 `192.168.1.100`）。
4. **放行防火墙端口 5000**（Windows，管理员 PowerShell）：
   ```powershell
   netsh advfirewall firewall add rule name="FL-5000" dir=in action=allow protocol=TCP localport=5000
   ```

   （或首次运行时在弹窗里允许 Python 通过防火墙。）
5. **启动服务器**：
   ```bash
   cd fl
   python -m server.fl_server --num-clients 2 --rounds 15 --model cnn
   ```

   > `--num-clients` 必须等于树莓派数量；`--model` 要和下面客户端填的完全一致。
   >
6. 浏览器打开 **http://127.0.0.1:5000/** 或 **http://<PC_IP>:5000/** 查看看板。

### 5.2 树莓派端（每台一个客户端）

> 按《树莓派使用文档》先用 SSH 连上：`ssh pi@raspberrypi.local`（密码 `123456`），
> 并通过 `sudo raspi-config` 连好 WiFi，确保和 PC 在**同一局域网/热点**。

1. **把代码拷到树莓派**（在 PC 上执行）：
   ```bash
   scp -r fl pi@<树莓派IP>:/home/pi/
   ```
2. **安装依赖**（在树莓派上）：
   ```bash
   cd ~/fl
   bash scripts/setup_pi.sh
   ```
3. **准备数据**：把 PC 上已下好的数据拷过来（推荐，避免树莓派旧版下载源失效）：
   ```bash
   # 在 PC 上执行
   scp -r fl/data pi@<树莓派IP>:/home/pi/fl/
   ```

   或在树莓派上直接 `python3 scripts/prepare_data.py`（需联网）。
4. **启动客户端**（两台树莓派分别用 id 0 / 1）：
   ```bash
   # 树莓派 #1
   python3 -m client.fl_client --server http://<PC_IP>:5000 --client-id 0 \
       --model cnn --num-clients 2 --seed 42
   # 树莓派 #2
   python3 -m client.fl_client --server http://<PC_IP>:5000 --client-id 1 \
       --model cnn --num-clients 2 --seed 42
   ```

   > ⚠ **三端一致性**：两台客户端与服务器的 `--model`、`--num-clients` 必须完全相同，否则模型参数维度对不上，聚合会失败。
   >
   > ⚠ **数据分片不重叠靠 seed 一致**：IID 划分用 `--seed`（默认 42）确定性地把训练集切成互不重叠的分片，两台 Pi 的 `--seed` 与 `--num-clients` 必须相同，才能保证 client 0 / client 1 各拿一半且无重复。若想做 Non-IID 对比，两台同时加 `--partition noniid`（仍需 seed 一致）。
   >

两台客户端连上后，服务器即开始逐轮 FedAvg，看板实时刷新曲线。

---

## 6. 运行参数说明

**服务器** `python -m server.fl_server`：

| 参数              | 默认    | 说明                               |
| ----------------- | ------- | ---------------------------------- |
| `--host`        | 0.0.0.0 | 监听地址（0.0.0.0 允许局域网访问） |
| `--port`        | 5000    | 端口                               |
| `--rounds`      | 15      | 通信轮数 T                         |
| `--num-clients` | 2       | 客户端数（=树莓派数）              |
| `--model`       | cnn     | `cnn` 或 `mlp`                 |

**客户端** `python -m client.fl_client`：

| 参数               | 默认                  | 说明                              |
| ------------------ | --------------------- | --------------------------------- |
| `--server`       | http://127.0.0.1:5000 | 服务器地址                        |
| `--client-id`    | 必填                  | 客户端编号 0..N-1                 |
| `--local-epochs` | 1                     | 每轮本地训练 epoch 数 E           |
| `--local-steps`  | 0                     | >0 时每轮只训这么多 batch（控时） |
| `--batch-size`   | 32                    |                                   |
| `--lr`           | 0.01                  |                                   |
| `--model`        | cnn                   | 需与服务器一致                    |
| `--partition`    | iid                   | `iid` 或 `noniid`（提高用）   |

> ⚠ 客户端的 `--model`、`--num-clients` 需与服务器保持一致，否则参数维度对不上。

---

## 7. 输出结果

运行时所有产物都在 `results/`（位于服务器/PC 端）：

- `results/global_model.pth`：**训练好的全局模型**。每轮聚合后都会覆盖保存最新版本，
  所以即使中途停掉也能拿到最新模型。这是一个含元信息的 checkpoint：
  `{state_dict, model, dataset, round, accuracy}`。
- `results/metrics.csv`：每轮的 `round, accuracy, loss, elapsed_sec`。
- `results/curves.png`：训练结束自动生成的准确率/损失曲线（可直接放进课程报告）。
- 网页看板：实时轮次、全局准确率/损失、两条曲线、各客户端状态表。

默认配置下 MNIST 约 15 轮后全局准确率可达 **97% 以上**。

**加载并使用训练好的模型**（推理 / 报告里的"模型验证"）：

```bash
python scripts/predict.py            # 在测试集上评估 + 展示样例预测
```

或在自己的代码里加载：

```python
import torch
from common.model import build_model
ckpt = torch.load("results/global_model.pth", map_location="cpu")
model = build_model(ckpt["model"], ckpt["dataset"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

---

## 8. 常见问题排查

**Q1. 树莓派 `pip install torch` 失败 / 没有 armv7l 包？**
官方 PyTorch 不提供 32 位 ARM 包。`setup_pi.sh` 会自动尝试社区 wheel（cp37）。若下载链接失效，可搜索关键词 “pytorch armv7l wheel cp37” 获取（如 Kashu7100/pytorch-armv7l、nmilosev/pytorch-arm-builds），下载后 `pip install xxx.whl`。
**兜底方案**：客户端代码与平台无关，若某台树莓派实在装不上，可直接在 PC 上多开两个 `fl_client` 进程完成完整演示。

**Q2. 客户端连不上服务器？**
① PC 和树莓派是否同一局域网（同一 WiFi/热点）；② PC 防火墙是否放行 5000；③ `--server` 是否填的是 **PC 的局域网 IP**（不是 127.0.0.1）；④ 在树莓派上 `ping <PC_IP>` 测连通。

**Q3. 看板打开了但没有曲线？**
曲线在每轮聚合完成后才出现。需要 **两个客户端都连上并各自完成一轮**，服务器聚合后才会画第一个点（round 0 为随机模型基线）。

**Q4. MNIST 下载失败？**
在能联网的 PC 上先 `python scripts/prepare_data.py`，再把 `data/MNIST` 拷到树莓派。

**Q5. 训练太慢？**
用 `--model mlp` 换更轻的网络，或给客户端加 `--local-steps 50` 限制每轮迭代数。

**Q6.（重要）真机一上来全局准确率就塌到 ~0.10、损失为 `nan`？**
现象：PC 单机自测正常，但部署到树莓派后第 1 轮起 `loss=nan`、准确率 ≈ 1/10（模型退化成只猜一个类）。
**根因**：部分 armv7l 树莓派的非官方 `torch` 构建（如 `torch 1.4.0a0`）**"输入通道=1 的卷积"路径有 bug**——表现为第一层反向梯度为 `None`（网络静默不训练，loss 不降）或前向偶发 NaN。MNIST 恰好是单通道，正好踩中。一旦某客户端上传坏权重，FedAvg 后全局模型即失效。**这不是序列化/网络/权重数值问题**（已用 `scripts/probe_transfer.py` 验证下发权重逐位正确）。
**定位工具**：`scripts/test_3channel.py`（一键对照 1 vs 3 通道）、`check_data.py`、`diagnose_forward.py`、`stress_conv.py`（很适合写进报告"问题分析"）。
**解决（二选一）**：
- **保留 CNN（推荐）**：把 MNIST 单通道**复制成 3 通道**绕开该 bug——三端都加 `--model cnn --channels 3` 即可，实测梯度正常、loss 正常下降、MNIST 可达 ~99%。CIFAR-10 本就是 3 通道，天然不受影响。
- **改用 MLP**：`--model mlp`（不含卷积，本项目默认值），MNIST 可达 ~97%。
> 三端 `--model` 与 `--channels` 必须一致，否则参数维度对不上。若日后换 64 位官方 torch，单通道卷积也正常，可回到 `--channels 1`。
> 已内置鲁棒性：服务器 `fedavg` 自动剔除含 NaN/Inf 的更新并保留上一轮模型；客户端本地训练出现 NaN 会 `[警告]` 并定位节点。
> 已内置的鲁棒性：服务器 `fedavg` 会自动剔除含 NaN/Inf 的更新并保留上一轮模型；客户端本地训练出现 NaN 会打印 `[警告]` 并定位到具体节点。

---

## 9. 提高部分扩展点（本版未实现，已预留接口）

- **CIFAR-10**：`common/data.py` 仿 `load_mnist` 加 `load_cifar10`（`torchvision.datasets.CIFAR10`），`build_model(dataset="cifar10")` 已自适应输入通道/尺寸。
- **Non-IID 研究**：客户端加 `--partition noniid` 即用 `partition_noniid`（已实现 shard 法）；对比 IID/Non-IID 收敛曲线。
- **MobileNet vs CNN**：在 `common/model.py` 加 MobileNet-lite 分支，对比在树莓派上的单轮耗时与精度。

---

## 10. 对应任务书要求

| 任务书要求                      | 本项目实现                                 |
| ------------------------------- | ------------------------------------------ |
| 硬件环境配置 + 多树莓派网络互通 | `setup_pi.sh` + HTTP/Flask 通信          |
| 基于 MNIST 的 FedAvg            | `aggregate.fedavg` + MNIST 闭环          |
| 模型下发 / 参数收集 / 加权聚合  | `/get_model`、`/submit_update`、FedAvg |
| 实时监控准确率曲线              | 网页看板 +`metrics.csv` + `curves.png` |
