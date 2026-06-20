# Non-IID 数据划分方法说明

本文档说明项目中三种联邦学习 Non-IID 数据划分方法的来源、原理与效果。

---

## 一、McMahan Shard 法（`partition_noniid`）

**所在文件**：`common/data.py` · `partition_noniid()`  
**CLI 参数**：`--partition shard`（或别名 `noniid`），用 `--classes-per-client` 调节程度

### 来源

出自联邦学习奠基论文：

> McMahan et al., *"Communication-Efficient Learning of Deep Networks from Decentralized Data"*, PMLR 2017

即提出 FedAvg 算法的原文。本项目直接复现了原文的实验设置。

### 划分逻辑

```
MNIST 训练集（60000 条）按标签排序：
[0,0,...,0,  1,1,...,1,  2,2,...,  9,9,...,9]
              ↓ 切成 4 个 shard（num_clients=2 × classes_per_client=2）
   shard 0       shard 1       shard 2       shard 3
[0, 1, 2...]  [2, 3, 4...]  [4, 5, 6, 7...]  [7, 8, 9...]
              ↓ seed=42 随机排列 [3,2,1,0] 后分配
  client 0 ← shard 3 + shard 2   （数字 4~9 为主）
  client 1 ← shard 1 + shard 0   （数字 0~4 为主）
```

**seed=42、num_clients=2 时的实际分配结果**：

| 客户端 | 样本数 | 标签分布 |
|--------|--------|---------|
| client 0 | 30000 | 4(596) · 5(5421) · 6(5918) · 7(6265) · 8(5851) · 9(5949) |
| client 1 | 30000 | 0(5923) · 1(6742) · 2(5958) · 3(6131) · 4(5246) |

### 本质与效果

**硬性、离散**的 Non-IID：每个客户端只见到少数几个类别，样本数量相等。  
对 Non-IID 程度的描述是二值的（有这个类 / 没有这个类），无法连续调节异质程度。

原文用 `classes_per_client=2` 证明每人只见 2 种数字时 FedAvg 收敛明显变慢——这是联邦学习 Non-IID 研究的起点。

> **适用场景**：复现 FedAvg 原始实验、课程报告基础部分验证。

---

## 二、Dirichlet 分布法（`partition_noniid_dirichlet`）

**所在文件**：`common/data.py` · `partition_noniid_dirichlet()`  
**CLI 参数**：`--partition dirichlet`，用 `--alpha` 调节异质程度（真机客户端与仿真脚本均可用）

### 来源

出自：

> Yurochkin et al., *"Bayesian Nonparametric Federated Learning of Neural Networks"*, ICML 2019

此后成为联邦学习研究的**事实标准**，2020 年以后绝大多数 FL 论文用此方法生成 Non-IID 数据。

### 划分逻辑

对每个类别 `c`（0~9），从 Dirichlet 分布采样各客户端的分配比例：

```
对每个数字 c：
  [p₀, p₁] ~ Dirichlet([α, α])
  client 0 得到该类 p₀ 比例的样本
  client 1 得到该类 p₁ 比例的样本
```

α 是**浓度参数**，控制概率向量的集中程度：

| α 值 | Non-IID 程度 | 说明 |
|------|------------|------|
| → 0 | 极度 Non-IID | 每个类几乎全给一个客户端 |
| 0.1 | 高度 Non-IID | 严重类别偏斜，某些类只有个位数样本 |
| 0.5 | 中度 Non-IID | 明显偏斜，但每类仍有少量样本 |
| → ∞ | 接近 IID | 各客户端分布趋向均匀 |

### 本质与效果

**软性、连续**的 Non-IID：所有客户端原则上拥有全部类别，只是各类别的**比例**不同，可通过 α 连续调节异质程度。

与 shard 法相比更贴近真实世界——现实中很少出现某节点完全没有某类数据的情况。α=0.1 时，某客户端可能 80% 数据都是同一种数字；α=0.5 时偏斜存在但温和得多。

> **适用场景**：学术研究、报告发挥部分定量分析，通过"拨旋钮"连续改变 Non-IID 程度观察收敛曲线差异。

---

## 三、不均衡 IID（`partition_imbalanced`）

**所在文件**：`common/data.py` · `partition_imbalanced()`  
**CLI 参数**：`--partition imbalanced`，用 `--ratios` 指定各客户端比例（真机客户端与仿真脚本均可用）

### 来源

没有专门的奠基论文，是联邦学习研究中**解耦两种异质性**的标准做法：

- **标签分布异质性**（Non-IID）：各节点拥有不同类别的数据
- **样本数量不均衡**：各节点的数据总量差距悬殊

两者是独立问题，需要分开研究才能知道各自的影响有多大。`partition_imbalanced` 专门用于研究后者，将标签分布固定为 IID 以排除干扰。

### 划分逻辑

```python
partition_imbalanced(dataset, num_clients=2, client_id, ratios=[0.9, 0.1], seed=42)
# client 0 → 随机取 90% 的训练数据（54000 条），类别分布与全局相同
# client 1 → 随机取 10% 的训练数据（6000 条），类别分布与全局相同
```

内容完全 IID（各客户端数字 0~9 的比例均接近全局分布），**只有总量不同**。

### 本质与效果

FedAvg 按样本数加权平均（`W = Σ nₖ/Σn · Wₖ`），数据量少的客户端权重低。

| ratios | 效果 |
|--------|------|
| 50% / 50% | 理想均衡基准 |
| 90% / 10% | 少数节点权重仅 10%，影响有限 |
| 99% / 1% | 少数节点（600 条）几乎被 FedAvg 忽略，等效于单节点训练 |

极端情形（99:1）下，少数节点每轮只训练约 18 个 batch，本地模型质量差，且聚合时权重只有 1%，对全局模型贡献极小。

> **适用场景**：回答"两台树莓派数据量差距悬殊时，联邦学习还有意义吗？"

---

## 三种方法对比

| | McMahan Shard | Dirichlet | 不均衡 IID |
|---|---|---|---|
| 研究的异质维度 | 标签分布 | 标签分布 | 样本数量 |
| 异质程度控制 | 离散（`classes_per_client`） | 连续（`α`） | 连续（`ratios`） |
| 每客户端的类别 | 只有少数类 | 所有类（比例不同） | 所有类（比例相同） |
| 样本数是否均等 | 均等 | 大致均等 | **不均等**（这正是研究目标） |
| 学术主流程度 | 早期标准（2017） | 当前主流（2019+） | 配合前两种用于解耦分析 |

---

## 在本项目中的使用

### 真机 FL 系统（`fl_client.py`）

三种方法均可在树莓派客户端通过 `--partition` 选择，服务器无需改动。**所有客户端的划分方式、对应参数与 `--seed` 必须完全一致**（每台 Pi 启动时会打印本地标签分布，便于核对）。

```bash
# 服务器（三种实验通用）
python -m server.fl_server --rounds 20 --num-clients 2 --model mlp

# —— Shard 法：每客户端只见少数类别 ——
python3 -m client.fl_client --server http://<PC_IP>:5000 --client-id 0 \
    --num-clients 2 --model mlp --partition shard --classes-per-client 2 --seed 42

# —— Dirichlet：alpha 越小越异质（2 台 Pi 时最推荐用它调程度）——
python3 -m client.fl_client --server http://<PC_IP>:5000 --client-id 0 \
    --num-clients 2 --model mlp --partition dirichlet --alpha 0.1 --seed 42

# —— 不均衡 IID：内容均衡、数量按 ratios 倾斜 ——
python3 -m client.fl_client --server http://<PC_IP>:5000 --client-id 0 \
    --num-clients 2 --model mlp --partition imbalanced --ratios 0.9,0.1 --seed 42
```

> 每条命令在另一台 Pi 上把 `--client-id` 改为 `1`、其余参数保持完全相同即可。
> `--partition noniid` 作为 `shard` 的别名保留（向后兼容旧命令）。

### 单机仿真（`scripts/simulate_noniid.py`）

三种方法均可通过场景参数使用，覆盖 6 个对比场景：

```bash
# 全部场景（iid_balanced / noniid_dir05 / noniid_dir01 / iid_imb_9_1 / iid_imb_99_1 / noniid_imb）
python scripts/simulate_noniid.py --rounds 30

# 仅运行关键对比场景
python scripts/simulate_noniid.py --scenarios iid_balanced noniid_dir01 noniid_imb
```

输出在 `results/noniid_research/`：各场景 CSV、准确率/损失对比曲线（`comparison.png`）、标签分布热图（`label_dist.png`）。

---

## 参考文献

1. McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). *Communication-efficient learning of deep networks from decentralized data.* AISTATS.
2. Yurochkin, M., Agarwal, M., Ghosh, S., Greenewald, K., Hoang, N., & Khazaeni, Y. (2019). *Bayesian nonparametric federated learning of neural networks.* ICML.
