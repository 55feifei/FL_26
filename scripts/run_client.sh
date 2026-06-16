#!/usr/bin/env bash
# ============================================================
# 树莓派端：启动联邦学习客户端
# 用法：  bash scripts/run_client.sh <服务器地址> <客户端ID> [其它参数]
# 例：    bash scripts/run_client.sh http://192.168.1.100:5000 0
#         bash scripts/run_client.sh http://192.168.1.100:5000 1 --local-epochs 2
# ============================================================
cd "$(dirname "$0")/.."
SERVER=${1:-http://127.0.0.1:5000}
CLIENT_ID=${2:-0}
shift 2 2>/dev/null || true
echo "连接服务器 $SERVER，客户端 ID = $CLIENT_ID"
python3 -m client.fl_client --server "$SERVER" --client-id "$CLIENT_ID" "$@"
