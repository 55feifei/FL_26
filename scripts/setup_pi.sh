#!/usr/bin/env bash
# =====================================================================
# 树莓派 (armv7l / 32 位系统) 依赖安装脚本
# 用法（在 fl 目录下）：  bash scripts/setup_pi.sh
#
# 说明：官方 PyTorch 不提供 32 位 ARM (armv7l) 预编译包，本脚本按优先级尝试：
#   ① piwheels / pip 默认源
#   ② 已知社区编译的 armv7l wheel（Python 3.7 / cp37）
# 若都失败，请见 README 的「PyTorch 安装疑难」一节，或改用 PC 多开客户端演示。
# =====================================================================
set -e
echo "======== 联邦学习客户端依赖安装 (Raspberry Pi) ========"

PY=python3
echo "[信息] Python 版本：$($PY --version 2>&1)"
echo "[信息] 系统架构：$(uname -m)"

echo "[1/4] 安装系统库（BLAS / 图像库，torch 运行时依赖）..."
sudo apt-get update
sudo apt-get install -y python3-pip libatlas-base-dev libopenblas-dev libjpeg-dev zlib1g-dev

echo "[2/4] 升级 pip 并安装 numpy / requests / pillow ..."
$PY -m pip install --upgrade pip
$PY -m pip install numpy requests pillow

echo "[3/4] 安装 PyTorch + torchvision（armv7l）..."
if $PY -m pip install torch torchvision >/dev/null 2>&1 && \
   $PY -c "import torch, torchvision" >/dev/null 2>&1; then
    echo "    ✓ 已通过 pip/piwheels 安装成功"
else
    echo "    pip 默认源无可用版本，改用社区 armv7l wheel ..."
    PYV=$($PY -c "import sys;print('%d%d' % (sys.version_info.major, sys.version_info.minor))")
    if [ "$PYV" = "37" ]; then
        BASE="https://github.com/Kashu7100/pytorch-armv7l/raw/main"
        TORCH_WHL="torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl"
        VISION_WHL="torchvision-0.8.0a0%2B45f960c-cp37-cp37m-linux_armv7l.whl"
        echo "    下载 $TORCH_WHL ..."
        wget -q --show-progress -O /tmp/torch.whl  "$BASE/$TORCH_WHL"
        echo "    下载 torchvision ..."
        wget -q --show-progress -O /tmp/torchvision.whl "$BASE/$VISION_WHL"
        $PY -m pip install /tmp/torch.whl /tmp/torchvision.whl
    else
        echo "    !! 当前 Python $PYV 无现成 wheel。请见 README「PyTorch 安装疑难」。"
        exit 1
    fi
fi

echo "[4/4] 验证安装 ..."
$PY -c "import torch, torchvision; print('    torch', torch.__version__, '| torchvision', torchvision.__version__)"

echo "======== 完成 ========"
echo "启动客户端示例（把 <PC_IP> 换成服务器电脑的局域网 IP）："
echo "    python3 -m client.fl_client --server http://<PC_IP>:5000 --client-id 0"
