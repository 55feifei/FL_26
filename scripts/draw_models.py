# -*- coding: utf-8 -*-
"""绘制 6 种神经网络结构图（mlp / cnn / deepcnn / resnet / squeezenet / mobilenet）。

纯 matplotlib 画方块流程图，输出 PNG 到 results/model_diagrams/。
默认按 MNIST（1×28×28）标注张量形状；deepcnn/resnet 用 CIFAR-10（3×32×32）标注，
因为它们面向彩色图（与 common/model.py 的实现一致）。

用法（本机）：
  & "F:\\ANACONDA\\envs\\FL\\python.exe" scripts/draw_models.py
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 中文字体（Windows）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "model_diagrams")
os.makedirs(OUT, exist_ok=True)

# 颜色按层类型
C = {
    "input":  "#E8EAED",
    "conv":   "#AECBFA",   # 卷积 蓝
    "dwconv": "#7BAAF7",   # 深度卷积 深蓝
    "norm":   "#A8DAB5",   # 归一化 绿
    "act":    "#FDE293",   # 激活 黄
    "pool":   "#FBBC04",   # 池化 橙
    "fc":     "#D7AEFB",   # 全连接 紫
    "flat":   "#F5F5F5",   # 展平
    "special":"#F6AEA9",   # 残差/特殊
    "concat": "#FCC9A9",   # 拼接
    "output": "#F28B82",   # 输出 红
}


def draw_flow(ax, blocks, x=0.5, w=3.0, top=1.0, gap=0.18, h=0.62, title=""):
    """在 ax 上从上到下画一列方块，blocks=[(主标签, 形状标签, 颜色key)]，返回最低 y。"""
    y = top
    centers = []
    for label, shape, ck in blocks:
        box = FancyBboxPatch((x, y - h), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.2, edgecolor="#5f6368",
                             facecolor=C[ck])
        ax.add_patch(box)
        cx, cy = x + w / 2.0, y - h / 2.0
        ax.text(cx, cy + (0.10 if shape else 0), label, ha="center", va="center",
                fontsize=9.5, fontweight="bold")
        if shape:
            ax.text(cx, cy - 0.14, shape, ha="center", va="center",
                    fontsize=8, color="#3c4043")
        centers.append((cx, y, y - h))
        y -= h + gap
    # 箭头连接
    for i in range(len(centers) - 1):
        cx, _, ybot = centers[i]
        _, ytop, _ = centers[i + 1]
        ax.add_patch(FancyArrowPatch((cx, ybot), (cx, ytop),
                     arrowstyle="-|>", mutation_scale=12,
                     linewidth=1.1, color="#5f6368"))
    if title:
        ax.text(x + w / 2.0, top + 0.25, title, ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    return y


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


# ---------------- 1. MLP ----------------
def fig_mlp():
    fig, ax = plt.subplots(figsize=(4.2, 6.5))
    ax.axis("off"); ax.set_xlim(0, 4); ax.set_ylim(-5.2, 1.6)
    draw_flow(ax, [
        ("输入 Input", "1×28×28", "input"),
        ("Flatten 展平", "→ 784", "flat"),
        ("Linear 全连接", "784 → 128", "fc"),
        ("ReLU", "", "act"),
        ("Linear 全连接", "128 → 10", "fc"),
        ("输出 logits", "10 类", "output"),
    ], title="MLP（两层全连接）")
    save(fig, "1_mlp.png")


# ---------------- 2. SimpleCNN ----------------
def fig_cnn():
    fig, ax = plt.subplots(figsize=(4.4, 8.2))
    ax.axis("off"); ax.set_xlim(0, 4); ax.set_ylim(-7.0, 1.6)
    draw_flow(ax, [
        ("输入 Input", "1×28×28", "input"),
        ("Conv 3×3, 16 + ReLU", "16×28×28", "conv"),
        ("MaxPool 2×2", "16×14×14", "pool"),
        ("Conv 3×3, 32 + ReLU", "32×14×14", "conv"),
        ("MaxPool 2×2", "32×7×7", "pool"),
        ("Flatten 展平", "→ 1568", "flat"),
        ("Linear + ReLU", "1568 → 128", "fc"),
        ("Linear", "128 → 10", "fc"),
        ("输出 logits", "10 类", "output"),
    ], title="SimpleCNN（cnn）")
    save(fig, "2_simplecnn.png")


# ---------------- 3. DeepCNN (VGG 风格) ----------------
def fig_deepcnn():
    fig, ax = plt.subplots(figsize=(4.8, 9.2))
    ax.axis("off"); ax.set_xlim(0, 4); ax.set_ylim(-8.2, 1.6)
    draw_flow(ax, [
        ("输入 Input (CIFAR-10)", "3×32×32", "input"),
        ("Block1: [Conv3×3,64+GN+ReLU]×2", "64×32×32", "conv"),
        ("MaxPool 2×2", "64×16×16", "pool"),
        ("Block2: [Conv3×3,128+GN+ReLU]×2", "128×16×16", "conv"),
        ("MaxPool 2×2", "128×8×8", "pool"),
        ("Block3: [Conv3×3,256+GN+ReLU]×2", "256×8×8", "conv"),
        ("MaxPool 2×2", "256×4×4", "pool"),
        ("Flatten 展平", "→ 4096", "flat"),
        ("Linear+ReLU+Dropout0.5", "4096 → 256", "fc"),
        ("Linear", "256 → 10", "fc"),
        ("输出 logits", "10 类", "output"),
    ], title="DeepCNN（VGG 风格，6 卷积层）")
    save(fig, "3_deepcnn.png")


# ---------------- 4. ResNet-20 ----------------
def fig_resnet():
    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    ax.axis("off"); ax.set_xlim(0, 9); ax.set_ylim(-7.6, 1.8)
    # 主干
    draw_flow(ax, [
        ("输入 Input (CIFAR-10)", "3×32×32", "input"),
        ("Conv 3×3, 16 + GN + ReLU", "16×32×32", "conv"),
        ("layer1: BasicBlock ×3", "16×32×32", "special"),
        ("layer2: BasicBlock ×3 (首块 stride2)", "32×16×16", "special"),
        ("layer3: BasicBlock ×3 (首块 stride2)", "64×8×8", "special"),
        ("Global AvgPool", "64×1×1", "pool"),
        ("Linear (FC)", "64 → 10", "fc"),
        ("输出 logits", "10 类", "output"),
    ], x=0.5, w=4.2, title="ResNet-20（6n+2, n=3）")

    # 右侧 BasicBlock 细节
    bx, bw, top, h, gap = 5.6, 3.0, 0.6, 0.55, 0.55
    ax.text(bx + bw / 2, top + 0.35, "BasicBlock 残差块", ha="center",
            fontsize=11, fontweight="bold")
    seq = [
        ("Conv 3×3 + GN + ReLU", "conv"),
        ("Conv 3×3 + GN", "conv"),
        ("⊕ 加上 shortcut", "special"),
        ("ReLU", "act"),
    ]
    y = top
    ys = []
    for lab, ck in seq:
        box = FancyBboxPatch((bx, y - h), bw, h,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.1, edgecolor="#5f6368", facecolor=C[ck])
        ax.add_patch(box)
        ax.text(bx + bw / 2, y - h / 2, lab, ha="center", va="center", fontsize=9)
        ys.append((y, y - h))
        y -= h + gap
    for i in range(len(ys) - 1):
        ax.add_patch(FancyArrowPatch((bx + bw / 2, ys[i][1]), (bx + bw / 2, ys[i + 1][0]),
                     arrowstyle="-|>", mutation_scale=11, color="#5f6368"))
    # shortcut 弧线：从输入绕到 ⊕
    ax.add_patch(FancyArrowPatch((bx, ys[0][0] - h / 2), (bx, ys[2][0] - h / 2),
                 connectionstyle="arc3,rad=-0.9", arrowstyle="-|>",
                 mutation_scale=11, color="#d93025", linestyle="--"))
    ax.text(bx - 0.95, (ys[0][0] + ys[2][0]) / 2 - h / 2, "shortcut\n跳连",
            ha="center", va="center", fontsize=8, color="#d93025")
    save(fig, "4_resnet.png")


# ---------------- 5. SqueezeNet ----------------
def fig_squeezenet():
    fig, ax = plt.subplots(figsize=(9.2, 8.8))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-8.0, 1.8)
    draw_flow(ax, [
        ("输入 Input", "1×28×28", "input"),
        ("Stem: Conv3×3,64 + GN + ReLU", "64×28×28", "conv"),
        ("MaxPool 2×2", "64×14×14", "pool"),
        ("Fire (64→128)", "128×14×14", "special"),
        ("Fire (128→128)", "128×14×14", "special"),
        ("MaxPool 2×2", "128×7×7", "pool"),
        ("Fire (128→256)", "256×7×7", "special"),
        ("Fire (256→256)", "256×7×7", "special"),
        ("MaxPool 2×2", "256×4×4", "pool"),
        ("Fire (256→384)", "384×4×4", "special"),
        ("Fire (384→384)", "384×4×4", "special"),
        ("Conv 1×1 分类头", "10×4×4", "conv"),
        ("Global AvgPool", "10 类", "output"),
    ], x=0.5, w=4.4, h=0.5, gap=0.13, title="SqueezeNet（Fire 模块版）")

    # Fire 模块细节
    bx = 6.2
    ax.text(bx + 1.5, 0.95, "Fire 模块", ha="center", fontsize=11, fontweight="bold")
    sq = FancyBboxPatch((bx + 0.6, -0.2), 1.8, 0.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=1.1, edgecolor="#5f6368", facecolor=C["conv"])
    ax.add_patch(sq)
    ax.text(bx + 1.5, 0.075, "squeeze 1×1 (sq)", ha="center", va="center", fontsize=8.5)
    e1 = FancyBboxPatch((bx - 0.1, -1.5), 1.5, 0.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=1.1, edgecolor="#5f6368", facecolor=C["conv"])
    e3 = FancyBboxPatch((bx + 1.7, -1.5), 1.5, 0.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=1.1, edgecolor="#5f6368", facecolor=C["conv"])
    ax.add_patch(e1); ax.add_patch(e3)
    ax.text(bx + 0.65, -1.225, "expand 1×1", ha="center", va="center", fontsize=8.5)
    ax.text(bx + 2.45, -1.225, "expand 3×3", ha="center", va="center", fontsize=8.5)
    cat = FancyBboxPatch((bx + 0.6, -2.7), 1.8, 0.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                         linewidth=1.1, edgecolor="#5f6368", facecolor=C["concat"])
    ax.add_patch(cat)
    ax.text(bx + 1.5, -2.425, "concat 通道拼接", ha="center", va="center", fontsize=8.5)
    for a, b in [((bx + 1.5, -0.2), (bx + 0.65, -0.95)),
                 ((bx + 1.5, -0.2), (bx + 2.45, -0.95)),
                 ((bx + 0.65, -1.5), (bx + 1.3, -2.15)),
                 ((bx + 2.45, -1.5), (bx + 1.7, -2.15))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color="#5f6368"))
    save(fig, "5_squeezenet.png")


# ---------------- 6. MobileNet ----------------
def fig_mobilenet():
    fig, ax = plt.subplots(figsize=(9.4, 8.8))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-8.0, 1.8)
    draw_flow(ax, [
        ("输入 Input", "1×28×28", "input"),
        ("Stem: Conv3×3,16 + GN + ReLU", "16×28×28", "conv"),
        ("DWSep (16→32)", "32×28×28", "special"),
        ("DWSep (32→64, s2)", "64×14×14", "special"),
        ("DWSep (64→64)", "64×14×14", "special"),
        ("DWSep (64→128, s2)", "128×7×7", "special"),
        ("DWSep (128→128)", "128×7×7", "special"),
        ("DWSep (128→256, s2)", "256×4×4", "special"),
        ("DWSep (256→256)", "256×4×4", "special"),
        ("Global AvgPool", "256", "pool"),
        ("Linear (FC)", "256 → 10", "fc"),
        ("输出 logits", "10 类", "output"),
    ], x=0.5, w=4.4, h=0.5, gap=0.13,
       title="MobileNetV1（width=0.5，通道已减半）")

    # DWSep 细节
    bx, bw, top, h, gap = 6.2, 3.2, 0.6, 0.6, 0.5
    ax.text(bx + bw / 2, top + 0.35, "深度可分离卷积块 DWSep", ha="center",
            fontsize=11, fontweight="bold")
    seq = [
        ("Depthwise Conv 3×3 (groups=Cin)", "dwconv"),
        ("GroupNorm + ReLU", "norm"),
        ("Pointwise Conv 1×1", "conv"),
        ("GroupNorm + ReLU", "norm"),
    ]
    y = top; ys = []
    for lab, ck in seq:
        box = FancyBboxPatch((bx, y - h), bw, h,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.1, edgecolor="#5f6368", facecolor=C[ck])
        ax.add_patch(box)
        ax.text(bx + bw / 2, y - h / 2, lab, ha="center", va="center", fontsize=8.5)
        ys.append((y, y - h)); y -= h + gap
    for i in range(len(ys) - 1):
        ax.add_patch(FancyArrowPatch((bx + bw / 2, ys[i][1]), (bx + bw / 2, ys[i + 1][0]),
                     arrowstyle="-|>", mutation_scale=11, color="#5f6368"))
    ax.text(bx + bw / 2, ys[-1][1] - 0.4,
            "标准卷积 → 拆成「逐通道 3×3 + 逐点 1×1」\n计算量大幅下降",
            ha="center", va="top", fontsize=8.5, color="#3c4043")
    save(fig, "6_mobilenet.png")


if __name__ == "__main__":
    fig_mlp()
    fig_cnn()
    fig_deepcnn()
    fig_resnet()
    fig_squeezenet()
    fig_mobilenet()
    print("\n全部完成，输出目录：", OUT)
