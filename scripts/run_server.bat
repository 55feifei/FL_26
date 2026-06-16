@echo off
REM ============================================================
REM PC 端：启动联邦学习中心服务器 + 实时看板
REM 用法：双击运行，或在命令行 scripts\run_server.bat
REM 额外参数会原样传给服务器，例如：
REM     scripts\run_server.bat --rounds 20 --num-clients 2
REM ============================================================
cd /d "%~dp0\.."
echo 启动联邦学习服务器（如用 conda 环境，请先 conda activate FL）...
python -m server.fl_server %*
pause
