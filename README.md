# 基于树莓派的分布式联邦学习系统

> 硬件课程设计 · 在不上传原始数据的前提下，多台树莓派作为边缘节点本地训练 MNIST，
> 仅上传模型参数到中心服务器做 **FedAvg 加权聚合**，并实时可视化全局模型准确率/损失。
>
> **发挥部分**额外实现了：① Non-IID & 样本不均衡对 FedAvg 的系统性研究（单机仿真，6 个对比场景）；
> ② 扩展至 **CIFAR-10 彩色数据集 + 更深层网络（DeepCNN / ResNet-20）** 的联邦训练，并探讨
> 联邦场景下 **GroupNorm 替代 BatchNorm** 的关键实现差异。

---

## 目录

1. [系统原理](#1-系统原理)
2. [目录结构](#2-目录结构)
3. [环境配置](#3-环境配置)
4. [PC 单机自测（推荐先跑）](#4-pc-单机自测推荐先跑)
5. [真机部署：PC + 2 台树莓派](#5-真机部署pc--2-台树莓派)
6. [运行参数说明](#6-运行参数说明)
7. [输出结果说明](#7-输出结果说明)
8. [发挥部分：Non-IID 与样本不均衡研究](#8-发挥部分non-iid-与样本不均衡研究)
9. [发挥部分：CIFAR-10 + 深层网络（DeepCNN / ResNet）](#9-发挥部分cifar-10--深层网络deepcnn--resnet)
10. [常见问题排查](#10-常见问题排查)
11. [对应任务书要求](#11-对应任务书要求)
12. [快速参考卡（所有场景）](#快速参考卡所有场景)

---

## 1. 系统原理

联邦学习核心：**"数据不动，模型动"**。原始隐私数据永远留在本地设备，
节点之间只交换模型参数，由中心服务器聚合成更强的全局模型再下发，循环迭代。

```
        ┌──────────────────────────────────────────┐
        │   PC = 中心服务器 + 实时看板              │
        │   · 初始化 / 下发全局模型                 │
        │   · 收集各客户端参数 + 样本数 n_k         │
        │   · FedAvg:  W = Σ (n_k / Σn) · W_k      │
        │   · 全局测试集评估 → 准确率 / 损失        │
        │   · 网页看板实时画曲线 (http://IP:5000)   │
        └──────────▲──────────────────▲─────────────┘
             同一局域网 / 热点         │ HTTP REST
       ┌──────────┘                   └──────────┐
  ┌────┴──────────────┐         ┌────────────────┴────┐
  │  树莓派 #1 (id=0) │         │  树莓派 #2 (id=1)   │
  │  本地分片(一半数据)│         │  本地分片(另一半)    │
  │  本地训练 E epoch  │         │  本地训练 E epoch   │
  │  上传参数 + n_k    │         │  上传参数 + n_k     │
  └───────────────────┘         └─────────────────────┘
```

**每轮（Round）流程**：
1. 客户端 `GET /get_model` 拉取当前全局模型与轮次号
2. 仅当进入新轮时，在**本地私有数据**上训练 E 个 epoch
3. `POST /submit_update` 上传参数 + 本地样本数 n_k
4. 服务器收齐全部客户端后执行 **FedAvg**，更新全局模型
5. 在全局测试集评估，记录 accuracy / loss，进入下一轮

**FedAvg 公式**（McMahan et al., 2017）：

$$W_{global} = \sum_{k=1}^{K} \frac{n_k}{\sum n} \cdot W_k$$

样本多的节点贡献权重更大。服务器用"轮次号"做同步 barrier，等齐所有客户端才聚合推进。

---

## 2. 目录结构

```
FL_26/
├── common/                     # 公共模块（服务器与客户端共享）
│   ├── config.py               #   超参数 Config 数据类
│   ├── model.py                #   MLP / SimpleCNN / DeepCNN / ResNet-20 模型 + norm_layer
│   ├── data.py                 #   MNIST/CIFAR-10 加载(load_dataset) + 四种联邦分片方法
│   │                           #     partition_iid()              IID 等分
│   │                           #     partition_noniid()           McMahan shard 法
│   │                           #     partition_imbalanced()       IID + 数量不均衡 ★新增
│   │                           #     partition_noniid_dirichlet() Dirichlet Non-IID ★新增
│   └── serialize.py            #   state_dict ↔ npz 字节（HTTP 传输）
│
├── server/                     # 中心服务器（运行在 PC）
│   ├── fl_server.py            #   Flask + FedAvg 编排 + 评估 + 记录
│   ├── aggregate.py            #   FedAvg 加权平均（含 NaN 自动剔除）
│   └── templates/
│       └── dashboard.html      #   零外部依赖网页看板（canvas 实时曲线）
│
├── client/
│   └── fl_client.py            # 客户端（运行在每台树莓派）
│
├── scripts/
│   ├── simulate_noniid.py      # ★新增 Non-IID 研究仿真脚本（6 场景对比）
│   ├── prepare_data.py         # 预下载 MNIST / CIFAR-10 到本地（prepare_data.py cifar10）
│   ├── predict.py              # 加载全局模型推理 + 评估
│   ├── train_local.py          # 树莓派本地单机训练健康检查
│   ├── check_data.py           # 验证数据分片结果
│   ├── check_torch.py          # PyTorch 版本 & 能力检测
│   ├── diagnose_forward.py     # 前向传播逐层 NaN/Inf 诊断
│   ├── probe_transfer.py       # 序列化往返验证
│   ├── stress_conv.py          # 卷积层压力测试（armv7l bug 定位）
│   ├── test_3channel.py        # 1 vs 3 通道卷积对比
│   ├── setup_pi.sh             # 树莓派一键装依赖（含 armv7l torch wheel）
│   ├── run_server.bat          # Windows 一键启动服务器
│   └── run_client.sh           # 树莓派一键启动客户端
│
├── data/                       # 数据集（运行时自动创建）
│   ├── MNIST/raw/
│   └── cifar-10-batches-py/    #   CIFAR-10（解压后）
│
├── results/                    # 运行产物（运行时自动创建，按实验名分子目录）
│   ├── mnist_mlp/              #   {dataset}_{model}[_{norm}]，不同实验互不覆盖
│   │   ├── global_model.pth    #     最新全局模型 checkpoint
│   │   ├── metrics.csv         #     每轮 accuracy / loss / 耗时
│   │   └── curves.png          #     训练曲线图
│   ├── cifar10_resnet_group/   #   例：CIFAR-10 + ResNet-20 + GroupNorm 的产物
│   └── noniid_research/        #   Non-IID 研究专用输出目录 ★新增
│       ├── iid_balanced.csv
│       ├── noniid_dir05.csv
│       ├── noniid_dir01.csv
│       ├── iid_imb_9_1.csv
│       ├── iid_imb_99_1.csv
│       ├── noniid_imb.csv
│       ├── summary.csv         #   汇总对比表
│       ├── comparison.png      #   多场景准确率 & 损失对比曲线
│       ├── label_dist.png      #   各场景标签分布热图
│       └── convergence_speed.png # 最终准确率 & 达标轮次柱状图
│
├── requirements-pc.txt
├── requirements-pi.txt
└── README.md
```

---

## 3. 环境配置

### 3.1 PC 端（中心服务器）

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows / Linux / macOS |
| Python | 3.7（建议 conda 环境） |
| 依赖 | torch、torchvision、flask、numpy、matplotlib、requests |

```bash
# 创建并激活 conda 环境（与树莓派 Python 3.7 保持一致）
conda create -n FL python=3.7.3 -y
conda activate FL

# 安装依赖
cd FL_26
pip install -r requirements-pc.txt
```

### 3.2 树莓派端（客户端）

| 项目 | 要求 |
|------|------|
| 硬件 | 树莓派 4B（armv7l 32 位）|
| Python | 3.7 |
| 依赖 | torch（armv7l 社区 wheel）、torchvision、numpy、requests、pillow |

```bash
# 在树莓派上执行一键安装（首次）
cd ~/FL_26
bash scripts/setup_pi.sh
```

> **torch armv7l 说明**：官方 PyTorch 不提供 32 位 ARM 包，`setup_pi.sh` 自动安装社区 wheel（cp37）。若链接失效，搜索 "pytorch armv7l wheel cp37" 手动下载后 `pip install xxx.whl`。
> **兜底方案**：若树莓派装不上 torch，可在 PC 上多开两个 `fl_client` 进程完成完整演示。

---

## 4. PC 单机自测（推荐先跑）

真机部署前先在 PC 上将完整闭环跑通，确认逻辑无误，约 2–3 分钟。

```bash
conda activate FL
cd FL_26

# 第一步：下载 MNIST（仅首次，约 50 MB）
python scripts/prepare_data.py

# 第二步：启动服务器（新终端）
python -m server.fl_server --rounds 5 --num-clients 2 --model mlp

# 第三步：启动客户端 0（新终端）
python3 -m client.fl_client --server http://127.0.0.1:5000 --client-id 0

# 第四步：启动客户端 1（新终端）
python3 -m client.fl_client --server http://127.0.0.1:5000 --client-id 1
```

浏览器打开 **http://127.0.0.1:5000/** 查看实时看板。训练结束后检查 `results/` 目录。

**预期结果**：MLP 模型 5 轮后准确率约 95%+，15 轮后约 97%+。

> **加速技巧**：给客户端加 `--local-steps 50`（每轮只训 50 个 batch），可大幅缩短单轮时间。

---

## 5. 真机部署：PC + 2 台树莓派

### 5.1 PC 端准备

```bash
# 1. 查本机局域网 IP（记下与树莓派同网段的 IPv4，例如 192.168.1.100）
#    Windows:
ipconfig
#    Linux/Mac:
ip addr

# 2. 放行防火墙端口 5000（Windows，管理员 PowerShell）
netsh advfirewall firewall add rule name="FL-5000" dir=in action=allow protocol=TCP localport=5000

# 3. 启动服务器（等待客户端连接）
cd FL_26
python -m server.fl_server --num-clients 2 --rounds 15 --model mlp
```

> 浏览器访问 **http://127.0.0.1:5000/** 或 **http://\<PC_IP\>:5000/** 查看看板。

### 5.2 树莓派端准备

```bash
# 在 PC 上：把代码和数据拷贝到树莓派
scp -r FL_26 pi@<树莓派IP>:/home/pi/
scp -r FL_26/data pi@<树莓派IP>:/home/pi/FL_26/   # 拷已下载的 MNIST，避免 Pi 重新下载

# SSH 连入树莓派
ssh pi@<树莓派IP>   # 默认密码 123456

# 在树莓派上安装依赖（首次）
cd ~/FL_26
bash scripts/setup_pi.sh
```

### 5.3 启动客户端

**树莓派 #1**（client-id = 0）：
```bash
python3 -m client.fl_client \
    --server http://<PC_IP>:5000 \
    --client-id 0 \
    --model mlp \
    --num-clients 2 \
    --seed 42
```

**树莓派 #2**（client-id = 1）：
```bash
python3 -m client.fl_client \
    --server http://<PC_IP>:5000 \
    --client-id 1 \
    --model mlp \
    --num-clients 2 \
    --seed 42
```

两台客户端连上后，服务器即开始逐轮 FedAvg，看板实时刷新。

> ⚠ **三端一致性**：`--model`、`--num-clients`、`--channels` 必须与服务器完全相同，否则参数维度对不上，聚合失败。
>
> ⚠ **分片可重复性**：IID 划分靠 `--seed`（默认 42）保证两台 Pi 数据切分互不重叠。`--seed` 与 `--num-clients` 必须一致。

### 5.4 CNN 模式（精度更高，需注意 armv7l 兼容性）

部分 armv7l 树莓派的非官方 torch 构建，**输入通道=1 的卷积层有梯度 bug**（表现为 loss=nan、准确率 ≈ 10%）。解决方案：把 MNIST 单通道复制为 3 通道绕开该 bug，三端均加 `--channels 3`：

```bash
# 服务器
python -m server.fl_server --model cnn --channels 3 --rounds 15 --num-clients 2

# 每台树莓派
python3 -m client.fl_client --server http://<PC_IP>:5000 \
    --client-id 0 --model cnn --channels 3 --num-clients 2
```

> MLP 模式不含卷积，不受此 bug 影响，MNIST 可达 ~97%，是最稳妥的默认选择。CNN（3通道）可达 ~99%。

---

## 6. 运行参数说明

### 服务器 `python -m server.fl_server`

| 参数 | 默认 | 说明 |
|------|------|------|
| `--host` | `0.0.0.0` | 监听地址（允许局域网访问） |
| `--port` | `5000` | 端口 |
| `--rounds` | `15` | 通信轮数 T |
| `--num-clients` | `2` | 客户端数量（= 树莓派数量） |
| `--model` | `mlp` | `mlp` / `cnn` / `deepcnn` / `resnet`（CIFAR-10 建议后两者） |
| `--channels` | `1` | 输入通道数（CNN 在 armv7l Pi 上设 `3`；CIFAR-10 须为 `3`） |
| `--dataset` | `mnist` | `mnist` / `cifar10` |
| `--norm` | `group` | deepcnn/resnet 归一化：`group`(FL 推荐) / `batch` |
| `--data-dir` | `./data` | 数据集存放路径 |
| `--results-dir` | `./results` | 产物根目录（默认在其下按实验名建子目录） |
| `--flat-results` | 关 | 加上则关闭子目录，直接写 `--results-dir` 根目录（旧行为） |
| `--seed` | `42` | 随机种子 |

> 📁 **产物自动分目录**：服务器默认把模型/指标/曲线存到 `results/{dataset}_{model}[_{norm}]/`（如 `results/cifar10_resnet_group/`、`results/mnist_mlp/`），不同数据集/模型/归一化的结果**互不覆盖**。启动时会打印实际产物目录。想回到旧的扁平布局加 `--flat-results`。

### 客户端 `python3 -m client.fl_client`

| 参数 | 默认 | 说明 |
|------|------|------|
| `--server` | `http://127.0.0.1:5000` | 服务器地址 |
| `--client-id` | **必填** | 客户端编号 `0..N-1` |
| `--num-clients` | `2` | 总客户端数（须与服务器一致） |
| `--local-epochs` | `1` | 每轮本地训练 epoch 数 E |
| `--local-steps` | `0` | >0 时每轮只训 N 个 batch（控时） |
| `--batch-size` | `32` | 批大小 |
| `--lr` | `0.01` | SGD 学习率 |
| `--model` | `mlp` | `mlp`/`cnn`/`deepcnn`/`resnet`，须与服务器一致 |
| `--channels` | `1` | 须与服务器一致（CIFAR-10 须为 `3`） |
| `--dataset` | `mnist` | `mnist`/`cifar10`，须与服务器一致 |
| `--norm` | `group` | deepcnn/resnet 归一化 `group`/`batch`，须与服务器一致 |
| `--partition` | `iid` | 数据划分：`iid` / `shard`(=`noniid`) / `dirichlet` / `imbalanced` |
| `--classes-per-client` | `2` | **shard 方式**：每客户端分到的类别数（越小越 Non-IID） |
| `--alpha` | `0.5` | **dirichlet 方式**：浓度参数（0.1 高度异质 / 0.5 中度 / 越大越接近 IID） |
| `--ratios` | 无 | **imbalanced 方式**：各客户端样本比例，逗号分隔且和为 1，如 `0.9,0.1` |
| `--seed` | `42` | 须与所有客户端一致（保证分片确定、不重叠） |
| `--threads` | `0` | >0 时设置 torch 线程数（armv7l NaN 时可设 1） |
| `--poll-interval` | `2.0` | 等待下一轮的轮询间隔（秒） |

> ⚠ **三类 Non-IID 划分**（`shard` / `dirichlet` / `imbalanced`）所有客户端必须使用**完全相同**的划分方式、对应参数（`--classes-per-client` / `--alpha` / `--ratios`）和 `--seed`，否则分片会重叠或错位。服务器只在全局测试集评估、不参与划分，无需这些参数。
>
> 💡 仅 **2 台树莓派**时，`shard` 方式的类别切分点固定（两端各约 5 个数字），`--classes-per-client` 效果不明显；想连续调节异质程度建议用 `dirichlet` 改 `--alpha`。客户端启动时会打印**本地标签分布**，可据此核对各台 Pi 实际拿到的数据。

---

## 7. 输出结果说明

所有产物默认存到**按实验名分的子目录** `results/{dataset}_{model}[_{norm}]/`（由服务器/PC 生成；启动时打印实际路径）。以 `results/cifar10_resnet_group/` 为例：

| 文件 | 说明 |
|------|------|
| `…/global_model.pth` | 最新全局模型 checkpoint，含 `{state_dict, model, dataset, channels, norm, round, accuracy}` |
| `…/metrics.csv` | 每轮 `round, accuracy, loss, elapsed_sec` |
| `…/curves.png` | 训练完成后自动生成的准确率/损失曲线 |
| 网页看板 `:5000/` | 实时轮次、准确率/损失曲线、各客户端状态 |

> 不同数据集/模型/归一化各自独立成目录，互不覆盖。加 `--flat-results` 可退回旧的扁平布局（直接写 `results/`）。

**加载并使用训练好的模型**：

```bash
# 不带参数会自动在 results/ 下找最新的模型评估；也可指定具体路径
python scripts/predict.py
python scripts/predict.py results/cifar10_resnet_group/global_model.pth
```

```python
# 在代码中加载模型
import torch
from common.model import build_model

ckpt = torch.load("results/global_model.pth", map_location="cpu")
model = build_model(ckpt["model"], ckpt["dataset"],
                    channels=ckpt.get("channels"), norm=ckpt.get("norm", "group"))
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

**预期指标**：MNIST：MLP 默认 15 轮 ≈ **97%+**，CNN（3通道）≈ **99%+**。CIFAR-10：DeepCNN/ResNet-20 约 30 轮 ≈ **80%+**（联邦、2 客户端 IID）。

---

## 8. 发挥部分：Non-IID 与样本不均衡研究

### 8.1 研究背景

现实联邦学习中，各边缘节点的数据往往**既不独立同分布（Non-IID）、数量也极不均等**。以 MNIST 手写数字识别为例：一台树莓派可能只采集到某几种数字，另一台的样本量可能是前者的 10 倍甚至 99 倍。这两种因素都会影响 FedAvg 的收敛速度与最终精度。

本节通过单机仿真实验，系统研究以下两个维度的影响：
- **标签分布异质性**（Non-IID）：用 Dirichlet 分布参数 α 连续控制
- **样本数量不均衡**：用比例参数显式控制（9:1、99:1 等）

### 8.2 新增的数据分片方法

在原有 `partition_iid`（等分 IID）和 `partition_noniid`（McMahan shard 法）基础上，新增两种方法（均在 [common/data.py](common/data.py)）：

#### `partition_imbalanced(dataset, num_clients, client_id, ratios, seed=42)`

不均衡 IID 划分：内容分布 IID（各类别比例相同），但各客户端的**样本总量**按 `ratios` 指定。

```python
# 示例：client 0 获得 90% 的训练数据，client 1 只获得 10%
from common.data import load_mnist, partition_imbalanced

train = load_mnist()
subset_0 = partition_imbalanced(train, num_clients=2, client_id=0, ratios=[0.9, 0.1])
subset_1 = partition_imbalanced(train, num_clients=2, client_id=1, ratios=[0.9, 0.1])
# len(subset_0) ≈ 54000,  len(subset_1) ≈ 6000
```

#### `partition_noniid_dirichlet(dataset, num_clients, client_id, alpha=0.5, seed=42)`

Dirichlet 分布 Non-IID 划分（学术界主流方案，参考 Yurochkin et al., 2019）。

对每个类别，从 Dir(α) 采样各客户端的分配比例：

| α 值 | Non-IID 程度 | 说明 |
|------|------------|------|
| → 0 | 极度 Non-IID | 每客户端近乎只有 1 个类别 |
| 0.1 | 高度 Non-IID | 严重类别偏斜 |
| 0.5 | 中度 Non-IID | 有明显偏斜但每类仍有少量样本 |
| → ∞ | 接近 IID | 各客户端分布趋向均匀 |

```python
from common.data import load_mnist, partition_noniid_dirichlet

train = load_mnist()
# 高度 Non-IID：Client 0 可能以数字 2,3,5,7 为主，Client 1 以 0,1,4,6,8,9 为主
subset = partition_noniid_dirichlet(train, num_clients=2, client_id=0, alpha=0.1)
```

### 8.3 仿真实验设计

脚本 [scripts/simulate_noniid.py](scripts/simulate_noniid.py) 在单台 PC 上仿真 2 个虚拟客户端（对应 2 台树莓派），无需真实网络，直接比较 6 个场景的 FedAvg 收敛行为：

| # | 场景名 | 数据分布 | 样本比例 | 研究目的 |
|---|--------|---------|---------|---------|
| 1 | `iid_balanced` | IID 均匀 | 50% / 50% | **基准**：理想条件 |
| 2 | `noniid_dir05` | Dirichlet α=0.5 | 约均衡 | Non-IID 中度异质 |
| 3 | `noniid_dir01` | Dirichlet α=0.1 | 约均衡 | Non-IID 高度异质 |
| 4 | `iid_imb_9_1` | IID 均匀 | 90% / 10% | 数量不均衡影响 |
| 5 | `iid_imb_99_1` | IID 均匀 | 99% / 1% | 极度不均衡 |
| 6 | `noniid_imb` | Dirichlet α=0.1 | 90% / 10% | **双重非理想**（最差情形） |

### 8.4 运行仿真

```bash
conda activate FL
cd FL_26

# 准备数据（若尚未下载）
python scripts/prepare_data.py

# 运行全部 6 个场景（30 轮，约 30–50 分钟）
python scripts/simulate_noniid.py

# 自定义轮数和模型
python scripts/simulate_noniid.py --rounds 30 --model mlp --local-epochs 1

# 快速验证（仅运行 2 个场景、5 轮）
python scripts/simulate_noniid.py --rounds 5 --scenarios iid_balanced noniid_dir01

# 仅运行最感兴趣的场景
python scripts/simulate_noniid.py --scenarios iid_balanced noniid_dir01 iid_imb_9_1 noniid_imb
```

### 8.5 仿真参数说明

```
python scripts/simulate_noniid.py [选项]

  --rounds        通信轮数（默认 30，建议 ≥ 20 以充分展现收敛差异）
  --num-clients   虚拟客户端数量（默认 2）
  --local-epochs  每轮本地训练 epoch 数（默认 1）
  --batch-size    批大小（默认 32）
  --lr            SGD 学习率（默认 0.01）
  --model         模型（mlp / cnn，默认 mlp）
  --data-dir      MNIST 数据目录（默认 ./data）
  --results-dir   输出目录（默认 ./results/noniid_research）
  --seed          随机种子（默认 42）
  --scenarios     仅运行指定场景（空格分隔场景名，默认全部运行）
```

### 8.6 输出文件

运行结束后，`results/noniid_research/` 目录包含：

| 文件 | 说明 |
|------|------|
| `{场景名}.csv` | 每轮 `round, accuracy, loss`（每个场景一个文件） |
| `summary.csv` | 汇总对比：最终准确率、最高准确率、首次达 90% 轮次、最终损失 |
| `comparison.png` | 多场景准确率 & 损失对比曲线（可直接放入报告） |
| `label_dist.png` | 各场景标签分布热图（直观呈现 Non-IID 程度） |
| `convergence_speed.png` | 最终准确率柱状图 + 达 90% 所需轮次柱状图 |

控制台同时打印汇总表格，例如：

```
════════════════════════════════════════════════════════════
  FedAvg 非理想场景研究 — 汇总
════════════════════════════════════════════════════════════
  场景                      样本分布       最终准确率  最高准确率  达90%轮次  最终损失
  ────────────────────────────────────────────────────────────────────────────
  IID 均衡（基准）          30000:30000    0.9808      0.9811      1          0.0755
  Non-IID α=0.5（中度异质） ~均衡          0.9773      0.9773      1          0.0794
  Non-IID α=0.1（高度异质） 37644:22356    0.965x      0.965x      3          0.11xx
  IID + 9:1 数量不均衡      54000:6000     0.97xx      0.97xx      2          0.09xx
  IID + 99:1 极度不均衡     59400:600      0.9xxx      0.9xxx      xx         0.xxxx
  Non-IID α=0.1 + 9:1 不均  ~54K:~6K      0.9xxx      0.9xxx      xx         0.xxxx
════════════════════════════════════════════════════════════
```

### 8.7 预期结论方向

根据仿真实验，预期可观察到以下规律：

**Non-IID 对收敛的影响：**
- IID 基准第 1 轮即可达 95%+，Non-IID α=0.1 第 1 轮仅约 67%（收敛起点降低 ~28 pp）
- α 越小，标签偏斜越严重，FedAvg 收敛越慢，最终准确率也有所下降
- 原因：各客户端梯度方向差异大，FedAvg 平均后的"合力"偏离最优方向（称为 **client drift**）

**样本不均衡对收敛的影响：**
- 9:1 不均衡时，少数客户端（10% 数据）贡献权重仅为 10%，对全局模型影响有限，最终精度接近基准
- 99:1 极度不均衡时，少数客户端的本地数据量过小（600 样本），每轮学习效果差，且 FedAvg 权重极低，几乎被忽略

**双重非理想（Non-IID + 不均衡）：**
- 两种不理想因素叠加，收敛速度最慢、最终精度最低
- 对应真实场景：某台树莓派只采集了少数几种数字，且样本量远少于另一台

---

## 9. 发挥部分：CIFAR-10 + 深层网络（DeepCNN / ResNet）

### 9.1 研究目标

把数据集从 MNIST（灰度、28×28、近线性可分）扩展到 **CIFAR-10**（10 类彩色自然图、32×32×3、类内差异大、背景复杂），并对**更深层的网络**实现联邦训练，验证 FedAvg 在「更难的数据 + 更深的模型」下依然能稳定聚合收敛。整套联邦协议（同步 FedAvg、参数序列化、看板）完全复用，无需改动。

### 9.2 新增的深层模型（均在 [common/model.py](common/model.py)）

| 模型名 | 结构 | 深度 | 适用 | 说明 |
|--------|------|------|------|------|
| `deepcnn` | VGG 风格 3 段 `[conv-norm-relu]×2 + pool` + FC | 6 卷积层 | CIFAR-10（Pi 也可） | 通道 64→128→256，含 Dropout(0.5)，比 SimpleCNN 更深、表达力更强 |
| `resnet` | CIFAR 版 ResNet-20（`6n+2`, n=3），3 组残差层 + 跳连 | ~20 层 | CIFAR-10 | 残差跳连解决深层退化/梯度消失，演示「带 shortcut 的深网络也能稳定 FedAvg」 |

`build_model(name, dataset, channels, norm)` 按数据集自适应输入通道（mnist=1 / cifar10=3）与尺寸（28 / 32）。两个深层模型沿用现有 `mlp` / `cnn` 同一工厂，三端只需指定相同的 `--model` 即可。

### 9.3 联邦学习的关键实现差异：GroupNorm vs BatchNorm

深层网络需要归一化层稳定训练。但 **BatchNorm 在联邦 + Non-IID 下有固有缺陷**：

- BN 维护 `running_mean/var` 并写入 `state_dict`。Non-IID 下各客户端本地 batch 分布差异巨大 → 各自的 BN 统计量严重发散；FedAvg 直接加权平均这些发散的 buffer，得到的全局统计量**与任何客户端都不匹配**，推理时归一化错位、准确率下降（即 FedBN / "Non-IID quagmire" 现象）。
- **GroupNorm** 在单样本内按通道分组归一化，**与 batch 组成无关、不维护 running 统计量** → 对 Non-IID 天然鲁棒，且 FedAvg 只平均可学习参数，聚合更干净。

因此本项目把归一化层做成可配置 `--norm {batch,group}`，**默认 `group`**（契合联邦场景）；保留 `batch` 用于对照实验。`--norm` 须三端一致。

### 9.4 运行（PC 单机多开，先跑通）

```bash
conda activate FL
cd FL_26

# 1. 下载 CIFAR-10（仅首次，约 170 MB；落盘到 data/cifar-10-batches-py/）
python scripts/prepare_data.py cifar10

# 2. 服务器（DeepCNN + GroupNorm，30 轮）
python -m server.fl_server --dataset cifar10 --model deepcnn --channels 3 \
    --norm group --rounds 30 --num-clients 2

# 3. 两个客户端（另开两个终端，超参建议 lr=0.05 batch=64）
python3 -m client.fl_client --server http://127.0.0.1:5000 --client-id 0 \
    --dataset cifar10 --model deepcnn --channels 3 --norm group \
    --num-clients 2 --lr 0.05 --batch-size 64 --local-epochs 2
python3 -m client.fl_client --server http://127.0.0.1:5000 --client-id 1 \
    --dataset cifar10 --model deepcnn --channels 3 --norm group \
    --num-clients 2 --lr 0.05 --batch-size 64 --local-epochs 2

# 换更深的 ResNet-20：三端把 --model deepcnn 改为 --model resnet 即可
```

**Windows PowerShell 版（单行，本机直接复制可用）** —— PowerShell 续行符是反引号而非 `\`，且需用 FL 环境的 python.exe，故统一写成单行：

```powershell
# 进入 fl 目录
cd f:\硬件课程设计\FL\fl

# 1. 下载/解压 CIFAR-10（仅首次）
& "F:\ANACONDA\envs\FL\python.exe" scripts\prepare_data.py cifar10

# 2. 终端 1 —— 服务器（ResNet-20 + GroupNorm，30 轮）
& "F:\ANACONDA\envs\FL\python.exe" -m server.fl_server --dataset cifar10 --model resnet --channels 3 --norm group --rounds 30 --num-clients 2

# 3. 终端 2 —— 客户端 0
& "F:\ANACONDA\envs\FL\python.exe" -m client.fl_client --server http://127.0.0.1:5000 --client-id 0 --dataset cifar10 --model resnet --channels 3 --norm group --num-clients 2 --lr 0.05 --batch-size 64 --local-epochs 2

# 4. 终端 3 —— 客户端 1（仅 --client-id 改为 1）
& "F:\ANACONDA\envs\FL\python.exe" -m client.fl_client --server http://127.0.0.1:5000 --client-id 1 --dataset cifar10 --model resnet --channels 3 --norm group --num-clients 2 --lr 0.05 --batch-size 64 --local-epochs 2
```

> 用 DeepCNN 就把三处 `--model resnet` 改成 `--model deepcnn`。想快速验证别等 30 轮：服务器加 `--rounds 3`、客户端加 `--local-steps 50`。

> ⚠ **三端一致**：`--dataset cifar10 --model <deepcnn|resnet> --channels 3 --norm <group|batch>` 必须在服务器与所有客户端完全相同。CIFAR-10 为彩色三通道，`--channels` 必须为 3。

### 9.5 BN vs GN 对照实验（报告亮点）

固定 `--model resnet`，对比 `{IID, Non-IID} × {group, batch}`，直观展示「IID 下 BN≈GN；Non-IID 下 GN 明显优于 BN」：

```bash
# 客户端加 --partition dirichlet --alpha 0.1 即切到高度 Non-IID（服务器不变）
# 例：Non-IID + BatchNorm（与 Non-IID + GroupNorm 对比）
python3 -m client.fl_client --server http://127.0.0.1:5000 --client-id 0 \
    --dataset cifar10 --model resnet --channels 3 --norm batch \
    --partition dirichlet --alpha 0.1 --num-clients 2 --lr 0.05
```

### 9.6 树莓派部署注意（armv7l）

- CIFAR-10 原生 3 通道，**不触发单通道卷积 bug**；但 ResNet/深层卷积在 armv7l 上偏慢。
- 建议 Pi 端：优先 `deepcnn`；ResNet 用 `--local-steps`（如 80）控时、`--rounds` 适当减小；偶发 NaN 时加 `--threads 1`（`fedavg` 的 NaN/Inf 剔除兜底仍生效）。
- 在 PC `python scripts/prepare_data.py cifar10` 下载后，把 `data/cifar-10-batches-py` 用 scp 拷到各 Pi，避开旧 torchvision 下载源问题。
- `predict.py` 会从 checkpoint 自动读取 `dataset/channels/norm` 并重建匹配模型，CIFAR-10 模型同样可直接评估。

---

## 10. 常见问题排查

**Q1. 树莓派 torch 安装失败？**

官方 PyTorch 不提供 32 位 ARM 包。`setup_pi.sh` 使用社区 wheel（cp37）。若链接失效，搜索 "pytorch armv7l wheel cp37"（如 Kashu7100/pytorch-armv7l），手动下载后 `pip install xxx.whl`。
**兜底方案**：在 PC 上多开两个 `fl_client` 进程模拟两台树莓派完成演示。

**Q2. 客户端连不上服务器？**

① PC 和树莓派是否在**同一局域网**（同一 WiFi/热点）
② PC 防火墙是否放行端口 5000（见第 5.1 节）
③ `--server` 填的是否是 **PC 的局域网 IP**（不是 `127.0.0.1`）
④ 在树莓派上执行 `ping <PC_IP>` 验证网络连通

**Q3. 看板打开但无曲线？**

两个客户端都连上并**各自完成第一轮本地训练后**，服务器聚合才会出第一个点（round 0 为随机模型基线，随即绘出）。

**Q4. MNIST 下载失败？**

在能联网的 PC 上先执行 `python scripts/prepare_data.py`，再用 `scp -r data/MNIST pi@<IP>:/home/pi/FL_26/` 拷到树莓派。

**Q5. 训练太慢？**

- 换轻量模型：`--model mlp`
- 限制每轮批数：客户端加 `--local-steps 50`
- 减少轮数：服务器加 `--rounds 5`（快速验证功能正确性）

**Q6. 树莓派上准确率塌到 ~10%、loss=nan？**

**根因**：部分 armv7l 非官方 torch 构建的"单通道卷积层"有梯度 bug（Conv2d in_channels=1 时反向传播异常）。MNIST 恰好是单通道，CNN 默认配置踩中此 bug。

**定位**：在树莓派上运行 `python scripts/test_3channel.py` 可一键对照 1 vs 3 通道的表现。

**解决方案（二选一）**：
- **保留 CNN（推荐）**：三端均加 `--model cnn --channels 3`，把 MNIST 单通道复制为 3 通道绕过该 bug，精度可达 ~99%
- **改用 MLP**：三端均加 `--model mlp`，无卷积层，不受影响，精度约 ~97%

已内置鲁棒性：服务器 `fedavg` 自动剔除含 NaN/Inf 的更新并保留上一轮模型；客户端本地训练出现 NaN 会打印警告并定位节点。

**Q7. simulate_noniid.py 运行报 ModuleNotFoundError？**

确保在项目根目录 `FL_26/` 下运行（不要在 `scripts/` 子目录里运行）：

```bash
cd FL_26
python scripts/simulate_noniid.py
```

**Q8. 仿真脚本图表中文字体乱码？**

Windows 系统应自带 SimHei（黑体），脚本已配置回退字体列表。若仍乱码，可在脚本顶部临时改为英文标签，或安装 `matplotlib` 中文字体包：

```bash
pip install matplotlib
# 清除 matplotlib 字体缓存
python -c "import matplotlib; print(matplotlib.get_cachedir())"
# 删除上述目录下的 fontlist-*.json 文件后重新运行
```

---

## 11. 对应任务书要求

### 基础部分

| 任务书要求 | 本项目实现 |
|-----------|-----------|
| 硬件环境配置 + 多树莓派网络互通 | `setup_pi.sh` 一键装依赖 + Flask HTTP 通信 |
| 基于 MNIST 的 FedAvg 联邦学习 | `aggregate.fedavg` + MNIST 完整闭环 |
| 模型下发 / 参数收集 / 加权聚合 | `GET /get_model`、`POST /submit_update`、FedAvg |
| 实时监控准确率曲线 | 网页看板 + `metrics.csv` + `curves.png` |
| 最终准确率 ≥ 97% | MLP 15 轮 ≈ 97%+，CNN 15 轮 ≈ 99% |

### 发挥部分

| 研究问题 | 实现方式 |
|---------|---------|
| Non-IID 对 FedAvg 收敛的影响 | Dirichlet 分布分片（α=0.1/0.5），对比 IID 基准 |
| 样本数量不均衡对收敛的影响 | 不均衡 IID 分片（9:1、99:1），对比均衡基准 |
| 双重非理想（最差情形）研究 | Dirichlet Non-IID + 不均衡数量组合 |
| 可复现的对比实验 | `simulate_noniid.py` 单机仿真，6 场景，30 轮，输出 CSV + 图表 |
| **扩展彩色数据集 CIFAR-10** | `load_cifar10` + `load_dataset` 统一入口，含数据增强（见第 9 节） |
| **更深层网络的联邦训练** | DeepCNN（6 卷积层）/ ResNet-20（残差，约 20 层），三端 `--model` 选择 |
| **联邦归一化差异（GroupNorm）** | `--norm group/batch` 可配置，默认 GroupNorm，附 BN-vs-GN 对照实验 |

---

## 快速参考卡（所有场景）

所有命令均在 `fl/` 目录下执行。Windows 上把 `python` 替换为 `& "F:\ANACONDA\envs\FL\python.exe"`。

```bash
# 下载 MNIST（仅首次，约 50 MB）
python scripts/prepare_data.py
```

---

### 场景 A — PC 单机自测·MLP（推荐先跑，最稳定）

默认模型，无卷积，不受 armv7l bug 影响，MNIST 约 97%。

```bash
# 终端 1：服务器
python -m server.fl_server --rounds 15 --num-clients 2 --model mlp

# 终端 2：客户端 0
python3 -m client.fl_client --server http://127.0.0.1:5000 --client-id 0 --model mlp

# 终端 3：客户端 1
python3 -m client.fl_client --server http://127.0.0.1:5000 --client-id 1 --model mlp
```

---

### 场景 B — PC 单机自测·CNN（精度更高，约 99%）

需加 `--channels 3`（把 MNIST 单通道复制为 3 通道，绕过 armv7l 卷积 bug，三端一致）。

```bash
# 终端 1：服务器
python -m server.fl_server --rounds 15 --num-clients 2 --model cnn --channels 3

# 终端 2：客户端 0
python3 -m client.fl_client --server http://127.0.0.1:5000 --client-id 0 --model cnn --channels 3

# 终端 3：客户端 1
python3 -m client.fl_client --server http://127.0.0.1:5000 --client-id 1 --model cnn --channels 3
```

---

### 场景 C — 真机部署·MLP（PC 服务器 + 2 台树莓派）

```bash
# PC 端（终端 1）
python -m server.fl_server --rounds 15 --num-clients 2 --model mlp

# 树莓派 #1（SSH 连入后执行）
python3 -m client.fl_client --server http://<PC_IP>:5000 \
    --client-id 0 --num-clients 2 --model mlp --seed 42

# 树莓派 #2（SSH 连入后执行）
python3 -m client.fl_client --server http://<PC_IP>:5000 \
    --client-id 1 --num-clients 2 --model mlp --seed 42
```

---

### 场景 D — 真机部署·CNN 3 通道（精度最高）

```bash
# PC 端（终端 1）
python -m server.fl_server --rounds 15 --num-clients 2 --model cnn --channels 3

# 树莓派 #1
python3 -m client.fl_client --server http://<PC_IP>:5000 \
    --client-id 0 --num-clients 2 --model cnn --channels 3 --seed 42

# 树莓派 #2
python3 -m client.fl_client --server http://<PC_IP>:5000 \
    --client-id 1 --num-clients 2 --model cnn --channels 3 --seed 42
```

---

### 场景 E — 真机三类 Non-IID 实验（树莓派可直接选择）

三种划分方式均可在真机客户端通过命令行选择。**所有树莓派的划分方式、对应参数与 `--seed` 必须完全一致**；服务器无需改动（始终用 `--rounds`/`--num-clients`/`--model` 即可）。启动后每台 Pi 会打印自己的本地标签分布，便于核对。

**E1 · McMahan shard 法**（每客户端只见少数类别，硬性 Non-IID）：

```bash
# 服务器：python -m server.fl_server --rounds 20 --num-clients 2 --model mlp
# 树莓派 #1
python3 -m client.fl_client --server http://<PC_IP>:5000 \
    --client-id 0 --num-clients 2 --model mlp \
    --partition shard --classes-per-client 2 --seed 42
# 树莓派 #2
python3 -m client.fl_client --server http://<PC_IP>:5000 \
    --client-id 1 --num-clients 2 --model mlp \
    --partition shard --classes-per-client 2 --seed 42
```

**E2 · Dirichlet 软 Non-IID**（用 `--alpha` 连续调节异质程度，2 台 Pi 时最推荐）：

```bash
# 高度异质 alpha=0.1（想要中度异质改为 0.5）
# 树莓派 #1
python3 -m client.fl_client --server http://<PC_IP>:5000 \
    --client-id 0 --num-clients 2 --model mlp \
    --partition dirichlet --alpha 0.1 --seed 42
# 树莓派 #2
python3 -m client.fl_client --server http://<PC_IP>:5000 \
    --client-id 1 --num-clients 2 --model mlp \
    --partition dirichlet --alpha 0.1 --seed 42
```

**E3 · 样本数量不均衡**（内容 IID，数量按 `--ratios` 倾斜，如 9:1）：

```bash
# 树莓派 #1（分到 90% 数据）
python3 -m client.fl_client --server http://<PC_IP>:5000 \
    --client-id 0 --num-clients 2 --model mlp \
    --partition imbalanced --ratios 0.9,0.1 --seed 42
# 树莓派 #2（分到 10% 数据）
python3 -m client.fl_client --server http://<PC_IP>:5000 \
    --client-id 1 --num-clients 2 --model mlp \
    --partition imbalanced --ratios 0.9,0.1 --seed 42
```

> `--partition noniid` 仍作为 `shard` 的别名保留（向后兼容旧命令）。
> CNN 模式同理，三端再加 `--model cnn --channels 3` 即可。
> 想在**单机**一次性对比全部 6 个场景（含双重非理想），见下方场景 F。

---

### 场景 F — 发挥部分·Non-IID 单机仿真（6 场景对比，无需网络）

```bash
# 运行全部 6 个场景（默认 30 轮，约 30–50 分钟）
python scripts/simulate_noniid.py

# 快速验证（2 个场景，5 轮，约 2 分钟）
python scripts/simulate_noniid.py --rounds 5 --scenarios iid_balanced noniid_dir01

# 自定义轮数 / 模型
python scripts/simulate_noniid.py --rounds 30 --model mlp --local-epochs 1

# 仅跑关键场景（IID 基准 + 高度 Non-IID + 双重非理想）
python scripts/simulate_noniid.py --scenarios iid_balanced noniid_dir01 noniid_imb
```

输出结果在 `results/noniid_research/`：各场景 CSV、`comparison.png`（准确率/损失对比曲线）、`label_dist.png`（标签分布热图）。

---

### 场景 G — 发挥部分·CIFAR-10 + 深层网络（DeepCNN / ResNet-20）

彩色数据集 + 更深网络的联邦训练。三端须一致 `--dataset cifar10 --channels 3 --model <deepcnn|resnet> --norm <group|batch>`（详见第 9 节）。

```bash
# 下载 CIFAR-10（仅首次，约 170 MB）
python scripts/prepare_data.py cifar10

# 终端 1：服务器（ResNet-20 + GroupNorm，30 轮）
python -m server.fl_server --dataset cifar10 --model resnet --channels 3 \
    --norm group --rounds 30 --num-clients 2

# 终端 2 / 3：两个客户端
python3 -m client.fl_client --server http://127.0.0.1:5000 --client-id 0 \
    --dataset cifar10 --model resnet --channels 3 --norm group \
    --num-clients 2 --lr 0.05 --batch-size 64 --local-epochs 2
python3 -m client.fl_client --server http://127.0.0.1:5000 --client-id 1 \
    --dataset cifar10 --model resnet --channels 3 --norm group \
    --num-clients 2 --lr 0.05 --batch-size 64 --local-epochs 2
```

> 想跑 BN vs GN 对照：把三端 `--norm group` 改成 `--norm batch`，客户端再加 `--partition dirichlet --alpha 0.1` 切到 Non-IID 对比。

---

### 工具脚本

```bash
python scripts/predict.py            # 加载全局模型，在测试集评估 + 展示样例预测
python scripts/check_torch.py        # 检测 PyTorch 版本与基本能力（在树莓派上运行）
python scripts/check_data.py         # 验证各客户端数据分片结果（样本数、类别分布）
python scripts/test_3channel.py      # 对比 1 vs 3 通道 CNN 在本机的表现（定位 armv7l bug）
python scripts/diagnose_forward.py   # 前向传播逐层 NaN/Inf 诊断
python scripts/stress_conv.py        # 卷积层压力测试（armv7l bug 定位）
```
