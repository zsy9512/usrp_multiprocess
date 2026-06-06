# USRP BPSK-Polar 无线通信系统

基于 USRP（B210）的软件无线电物理层通信系统，支持 BPSK 和 BPSK-Polar 两种工作模式。

## 帧结构

```
STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)
   ↓ 4×16重复   ↓ Zadoff-Chu   ↓ 已知导频  ↓ 预留+CRC   ↓ 极化码码字   ↓ CRC16   ↓ RRC隔离
```

共 **496 符号**，空口时间: **1.03ms** @ 1Msps (含 RRC 成形)。

## 同步方案 (三级同步链)

| 阶段 | 方法 | 功能 | 精度 |
|------|------|------|------|
| ① STF | 延迟相关 (L=32样本) | 粗包检测 + 粗CFO (±15.6kHz范围) | 对CFO免疫 |
| ② PSS | 互相关 + 双质量判据 | 精定时 (peak_to_mean≥4, peak_to_second≥1.5) | 1样本 |
| ③ RS | 线性相位拟合 + 信道估计 | 细CFO + 公共相位 + 信道幅度 + 噪声方差 | <10Hz |

## 快速开始

### BPSK PHY 层（物理层测试）

```bash
# 仿真模式：发送 200 帧 → 信道 → 接收
python sender.py --mode sim --num-frames 200 --sim-file tx_iq.npy --save-bits
python sim_channel.py tx_iq.npy rx_iq.npy --snr-db 15 --freq-offset 2000
python receiver.py --mode sim --sim-file rx_iq.npy --tx-bits tx_iq_bits.npy
```

### 硬件 USRP 模式

```bash
# 发送端
python sender.py --mode hardware --freq 915e6 --gain 60 --num-frames 50

# 接收端
python receiver.py --mode hardware --freq 915e6 --gain 30
```

### 极化码 SGNN 译码

```bash
# 软件全闭环仿真
python polar_phy_sender.py --mode sim --num-frames 500 --sim-file test.npy --save-bits
python sim_channel.py test.npy test_rx.npy --snr-db 8 --freq-offset 2000
python polar_phy_receiver.py --mode sim --sim-file test_rx.npy --tx-info-bits test_info_bits.npy
```

### 离线测试

```bash
python test_phy_offline.py                     # 全部6项测试
python test_phy_offline.py --test A            # 仅 SNR sweep
python test_phy_offline.py --test B --frames 500   # CFO测试, 500帧
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `phy_params.py` | PHY 统一参数 (帧结构、参考序列、RRC、CRC16) |
| `sender.py` | BPSK PHY 发送端 (完整帧 + USRP/仿真) |
| `receiver.py` | BPSK PHY 接收端 (三级同步链 + CRC 验证) |
| `sim_channel.py` | 仿真信道 (AWGN+频偏+相偏+多径) |
| `polar_phy_sender.py` | Polar编码 + PHY TX |
| `polar_phy_receiver.py` | PHY RX + SGNN译码 + BER |
| `interferer.py` | 干扰器 (仿真 + USRP) |
| `test_phy_offline.py` | 离线测试套件 (6项测试矩阵) |
| `deploy/sender.py` | 极化码编码 + UDP 发送 |
| `deploy/receiver.py` | UDP 接收 + SGNN 译码 |
| `deploy/common.py` | 共享组件 (BPConv, LSTU, SGNN) |
| `deploy/simulate.py` | 全闭环软件仿真 (BER/FER) |

## 依赖

- Python 3.9+
- numpy
- PyTorch (SGNN 译码)
- uhd (USRP 硬件驱动, 可选)

## 测试矩阵

| 矩阵 | 内容 | 验收标准 |
|------|------|----------|
| A | SNR sweep (100dB→2dB) | 无噪 100% CRC, 20dB >99% |
| B | CFO ±20kHz | 0Hz+1kHz 必须通过 |
| C | 初始相位 0→π | 全部相位通过 |
| D | 多径 (1-5 tap) | 无多径+弱多径 必须通过 |
| E | 采样定时 (delay 0-3) | delay 0+1 必须通过 |
| F | 连续多帧 | >99% CRC, >95% 检测率 |
