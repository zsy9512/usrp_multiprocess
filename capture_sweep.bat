@echo off
REM ============================================================
REM  SNR sweep 一键采集 (Windows 批处理)
REM
REM  用法:
REM    capture_sweep.bat                   默认全扫 (3 runs × 6 gains)
REM    capture_sweep.bat --dry-run         只打印, 不采集
REM    capture_sweep.bat --runs 1          每档只采1次 (快速)
REM    capture_sweep.bat --gains 64 55 48  只采指定增益
REM ============================================================

cd /d "%~dp0"

python tools/batch_capture.py %*

echo.
echo 按任意键退出...
pause >nul
