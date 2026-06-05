# USRP DQPSK / BPSK-Polar 无线通信系统

基于 USRP（B210）的软件无线电物理层通信系统，支持 DQPSK 和 BPSK-Polar 两种工作模式。

## 快速开始

### BPSK PHY 层（物理层测试）

```bash
# 仿真模式：发送 200 帧 → 信道 → 接收
python sender.py --mode sim --num-frames 200 --sim-file tx_iq.npy
python sim_channel.py tx_iq.npy rx_iq.npy --snr-db 15 --freq-offset 2000
python receiver.py --mode sim --sim-file rx_iq.npy
```

### 硬件 USRP 模式

```bash
# 发送端 (MyB210_01)
python sender.py --mode hardware --freq 915e6 --gain 70 --rate 1e6 ^
    --usrp-args "name=MyB210_01" --num-frames 50

# 接收端 (MyB210)
python receiver.py --mode hardware --freq 915e6 --gain 40 --rate 1e6 ^
    --usrp-args "name=MyB210"
```

### 极化码 SGNN 译码

```bash
# 软件全闭环仿真
cd deploy
python simulate.py --checkpoint checkpoint/polar_GNN_20_iter_0_epoches_13.pt
```

## 系统架构

### BPSK PHY 帧结构

```
PSS(32) + RS(16) + Data(256 BPSK) + Guard(32 零符号) = 336 符号
         ↓ Zadoff-Chu    ↓ 已知导频      ↓ 极化码码字    ↓ RRC隔离
```

空口时间: 0.69ms @ 1Msps

### 同步方案

1. **RS 全窗口滑动相关** — 16 符号随机 BPSK, 峰尖锐, 免疫频偏
2. **PSS/RS 线性相位拟合** — 32/16 符号频偏估计
3. **频偏校正 + LLR** — BPSK 软判决

### 帧结构

| 字段 | 长度 | 说明 |
|------|------|------|
| PSS | 32 符号 | Zadoff-Chu 序列 (u=25), 帧同步 |
| RS | 16 符号 | 已知 BPSK 导频, 频偏估计 |
| Data | 256 符号 | BPSK 极化码码字 (上层接口) |
| Guard | 32 符号 | 零符号, RRC 滤波隔离 |

## 文件说明

| 文件 | 功能 |
|------|------|
| `sender.py` | BPSK PHY 发送端 (USRP burst/持续流/仿真) |
| `receiver.py` | BPSK PHY 接收端 (RS同步+PSS频偏+LLR) |
| `sim_channel.py` | 仿真信道 (AWGN+频偏+相偏+多径) |
| `deploy/sender.py` | 极化码编码 + UDP 发送 |
| `deploy/receiver.py` | UDP 接收 + SGNN 图神经网络译码 |
| `deploy/simulate.py` | 全闭环软件仿真 (BER/FER) |

## 依赖

- Python 3.9+
- numpy, scipy
- PyTorch (SGNN 译码)
- uhd (USRP 硬件驱动, 可选)
- crcmod

## 验证结果

| 场景 | SNR | 频偏 | BER |
|------|-----|------|-----|
| 仿真单帧 | 15dB | 2kHz | **0.0** |
| 硬件 50 帧 | ~40dB | ~1.2kHz | **50 帧 0 漏检** |
