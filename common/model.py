"""模型定义（纯 PyTorch）。

提供以下模型：
- MLP：两层全连接，最轻量，在树莓派上训练最快，适合快速跑通联邦闭环。
- SimpleCNN：两层卷积 + 两层全连接，MNIST 上精度更高（>98%）。
- DeepCNN：VGG 风格三段卷积块（含 BatchNorm），面向 CIFAR-10 等彩色图，
            比 SimpleCNN 更深、表达力更强（发挥任务）。
- ResNet（ResNet-style 残差网络，约 20 层）：带跳连的更深网络，CIFAR-10 上精度
            最高，演示"对更深层网络做联邦训练"（发挥任务核心）。
- MobileNet（CIFAR-/MNIST-scaled MobileNetV1，width=0.5）：用 3x3 深度可分离卷积 +
            1x1 逐点卷积替代标准卷积，参数与计算量大幅下降，演示"轻量化网络在
            资源受限边缘端（树莓派）上的训练效率"（发挥任务三）。
- SqueezeNet（Fire 模块版）：用 squeeze 1x1 压缩通道 + expand 1x1/3x3 并行扩展的
            Fire 模块堆叠，分类头用 1x1 conv + 全局池化替代 FC，参数极少，
            与 MobileNet 一起作为"轻量化网络"代表（发挥任务三）。

build_model() 是统一工厂，按 (name, dataset) 自动适配输入通道/尺寸与类别数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """最简单的两层全连接网络。"""

    def __init__(self, in_dim=28 * 28, num_classes=10, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleCNN(nn.Module):
    """经典小型卷积网络：conv-pool-conv-pool-fc-fc。

    通过 input_size 自适应计算展平维度（MNIST 28 -> 7；CIFAR 32 -> 8），
    便于后续切换数据集而无需改网络结构。
    """

    def __init__(self, in_channels=1, num_classes=10, input_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        feat = input_size // 4  # 经过两次 2x2 池化
        self.fc1 = nn.Linear(32 * feat * feat, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))   # 14 -> 7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def norm_layer(channels, norm="group"):
    """构建归一化层。

    norm="batch" -> BatchNorm2d；norm="group" -> GroupNorm。

    联邦学习里 GroupNorm 通常优于 BatchNorm：BN 维护 running_mean/var 并写入 state_dict，
    Non-IID 下各客户端的 BN 统计量严重发散，FedAvg 直接加权平均会得到与任何客户端都不
    匹配的全局统计量，推理时归一化错位、精度下降（FedBN / Non-IID quagmire 现象）。
    GroupNorm 在单样本内按通道分组归一化，与 batch 无关、不维护 running 统计量，对
    Non-IID 天然鲁棒，聚合也更干净（只平均可学习参数）。

    GroupNorm 的分组数取能整除 channels 的较大值，保证 channels % num_groups == 0。
    """
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        num_groups = next(g for g in (32, 16, 8, 4, 2, 1) if channels % g == 0)
        return nn.GroupNorm(num_groups, channels)
    raise ValueError(f"未知归一化: {norm}（可选 'batch' | 'group'）")


class DeepCNN(nn.Module):
    """VGG 风格的深层 CNN：三段 [conv-norm-relu]x2 + maxpool，再接全连接分类头。

    相比 SimpleCNN（2 卷积层），这里有 6 个卷积层并引入归一化层，配合 CIFAR-10 的
    数据增强可达较高准确率，用来演示"更深网络的联邦训练"。
    通过 input_size 自适应展平维度（CIFAR 32 -> 4；MNIST 28 -> 3）。
    """

    def __init__(self, in_channels=3, num_classes=10, input_size=32, norm="group"):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), norm_layer(cout, norm), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), norm_layer(cout, norm), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            block(in_channels, 64),   # 32 -> 16
            block(64, 128),           # 16 -> 8
            block(128, 256),          # 8 -> 4
        )
        feat = input_size // 8        # 经过三次 2x2 池化
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * feat * feat, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    """ResNet 残差基本块：两层 3x3 卷积 + 跳连。"""

    def __init__(self, cin, cout, stride=1, norm="group"):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.n1 = norm_layer(cout, norm)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.n2 = norm_layer(cout, norm)
        # 维度或步长变化时，用 1x1 卷积对齐 shortcut 通道/尺寸
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                norm_layer(cout, norm),
            )

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """CIFAR 版 ResNet：3 组残差层，每组 num_blocks 个 BasicBlock。

    默认 num_blocks=3 即 ResNet-20（6n+2，n=3），是 CIFAR-10 的经典深层网络，
    用来在联邦场景下演示"更深、带跳连的网络"也能稳定聚合训练。
    """

    def __init__(self, in_channels=3, num_classes=10, num_blocks=3, norm="group"):
        super().__init__()
        self.norm = norm
        self.in_planes = 16
        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.n1 = norm_layer(16, norm)
        self.layer1 = self._make_layer(16, num_blocks, stride=1)
        self.layer2 = self._make_layer(32, num_blocks, stride=2)
        self.layer3 = self._make_layer(64, num_blocks, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, norm=self.norm))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        return self.fc(out)


class DWSepBlock(nn.Module):
    """深度可分离卷积块：3x3 深度卷积 (groups=cin) + 1x1 逐点卷积。

    用一对"分组卷积 + 1x1 卷积"近似标准卷积，计算量约从 D^2·Cin·Cout 降到
    D^2·Cin + Cin·Cout，是 MobileNetV1 的核心轻量化技巧。
    """

    def __init__(self, cin, cout, stride=1, norm="group"):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.n1 = norm_layer(cin, norm)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.n2 = norm_layer(cout, norm)

    def forward(self, x):
        x = F.relu(self.n1(self.dw(x)), inplace=True)
        x = F.relu(self.n2(self.pw(x)), inplace=True)
        return x


class MobileNet(nn.Module):
    """MobileNetV1 风格的轻量化网络（CIFAR-/MNIST-scaled）。

    结构：3x3 stem conv -> 7 个深度可分离块 -> 全局平均池化 -> 线性分类头。
    宽度系数 width=0.5：把所有通道砍半，使最终参数量与 SimpleCNN（~207k）同量级，
    真正体现"轻量化"。stem stride=1 保留输入分辨率；总共 3 次 stride=2 把 28x28
    压到 ~3x3（CIFAR 32x32 -> 4x4），末端 adaptive_avg_pool 把空间维度收成 1x1。
    """

    # (cout, stride)
    CFG = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]

    def __init__(self, in_channels=3, num_classes=10, width=0.5, norm="group"):
        super().__init__()

        def ch(c):
            # 取 max(8, ...) 保证最少有 8 通道，并向上取整到 8 的倍数便于 GroupNorm 分组
            return max(8, int(round(c * width / 8)) * 8)

        stem_c = ch(32)
        layers = [
            nn.Conv2d(in_channels, stem_c, 3, stride=1, padding=1, bias=False),
            norm_layer(stem_c, norm),
            nn.ReLU(inplace=True),
        ]
        cin = stem_c
        for cout, stride in self.CFG:
            cout_w = ch(cout)
            layers.append(DWSepBlock(cin, cout_w, stride=stride, norm=norm))
            cin = cout_w
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class Fire(nn.Module):
    """SqueezeNet Fire 模块：squeeze(1x1) -> expand(1x1) ‖ expand(3x3) 沿通道拼接。

    squeeze 层先用 1x1 卷积把通道数降到 sq；expand 分两支：1x1 conv 输出 ex1 通道、
    3x3 conv 输出 ex3 通道，最后 cat 得到 ex1+ex3 通道。等价于"先瘦身再分头扩张"，
    用大量 1x1 卷积换 3x3 卷积来压参数量，是 SqueezeNet 的核心思想。
    """

    def __init__(self, cin, sq, ex1, ex3, norm="group"):
        super().__init__()
        self.squeeze = nn.Conv2d(cin, sq, 1, bias=False)
        self.sn = norm_layer(sq, norm)
        self.expand1 = nn.Conv2d(sq, ex1, 1, bias=False)
        self.e1n = norm_layer(ex1, norm)
        self.expand3 = nn.Conv2d(sq, ex3, 3, padding=1, bias=False)
        self.e3n = norm_layer(ex3, norm)

    def forward(self, x):
        x = F.relu(self.sn(self.squeeze(x)), inplace=True)
        out1 = F.relu(self.e1n(self.expand1(x)), inplace=True)
        out3 = F.relu(self.e3n(self.expand3(x)), inplace=True)
        return torch.cat([out1, out3], 1)


class SqueezeNet(nn.Module):
    """SqueezeNet（CIFAR-/MNIST-scaled，使用 Fire 模块）。

    结构：3x3 stem conv + MaxPool -> 2 Fire(128) + MaxPool -> 2 Fire(256) + MaxPool ->
          2 Fire(384) -> 1x1 conv 投影到 num_classes -> 全局平均池化。
    分类头用 1x1 conv + 全局池化替代 FC（SqueezeNet 经典做法），让模型几乎全是卷积、
    参数比含大 FC 头的 SimpleCNN 还少。
    """

    def __init__(self, in_channels=3, num_classes=10, norm="group"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),
            norm_layer(64, norm),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),    # 28->14 / 32->16
        )
        self.fires = nn.Sequential(
            Fire(64, 16, 64, 64, norm),            # cout = 128
            Fire(128, 16, 64, 64, norm),           # cout = 128
            nn.MaxPool2d(2, 2, ceil_mode=True),    # 14->7 / 16->8
            Fire(128, 32, 128, 128, norm),         # cout = 256
            Fire(256, 32, 128, 128, norm),         # cout = 256
            nn.MaxPool2d(2, 2, ceil_mode=True),    # 7->4 / 8->4
            Fire(256, 48, 192, 192, norm),         # cout = 384
            Fire(384, 48, 192, 192, norm),         # cout = 384
        )
        self.classifier = nn.Conv2d(384, num_classes, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.fires(x)
        x = self.classifier(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return torch.flatten(x, 1)


def build_model(name="cnn", dataset="mnist", channels=None, norm="group"):
    """根据名称与数据集构建模型。

    自动按 dataset 推断输入通道（mnist=1, cifar10=3）与尺寸（28 / 32）。

    channels: 显式指定输入通道数（覆盖按 dataset 的推断）。
        某些 armv7l 树莓派 torch 构建的"单通道卷积"反向有 bug（第一层 grad=None/偶发 NaN），
        对 MNIST 把通道复制成 3 即可绕过——此时需 channels=3 使 conv1 与数据通道一致。
    norm: deepcnn / resnet / mobilenet / squeezenet 的归一化方式（"batch" | "group"），
        mlp / cnn 不含归一化、忽略此参数。
    """
    dataset = dataset.lower()
    in_channels = channels if channels else (1 if dataset == "mnist" else 3)
    input_size = 28 if dataset == "mnist" else 32
    num_classes = 10

    if name == "mlp":
        in_dim = in_channels * input_size * input_size
        return MLP(in_dim, num_classes)
    if name == "cnn":
        return SimpleCNN(in_channels, num_classes, input_size)
    if name == "deepcnn":
        return DeepCNN(in_channels, num_classes, input_size, norm=norm)
    if name == "resnet":
        return ResNet(in_channels, num_classes, norm=norm)
    if name == "mobilenet":
        return MobileNet(in_channels, num_classes, norm=norm)
    if name == "squeezenet":
        return SqueezeNet(in_channels, num_classes, norm=norm)
    raise ValueError(
        f"未知模型: {name}"
        "（可选 'mlp' | 'cnn' | 'deepcnn' | 'resnet' | 'mobilenet' | 'squeezenet'）"
    )
