# 极化码-PHY 融合系统设计方案

## 系统架构

```
                    ┌──────────────────────────────────────────┐
                    │           polar_phy_sender.py            │
                    │                                          │
  Info Bits ──→───→  Polar编码 ──→ BPSK ──→ 成帧 ──→ RRC ──→ USRP/文件
  (K=128)          (N=256码字)    (±1)     (PSS+RS)   (sps=2)
                    └──────────────────────────────────────────┘
                                               │
                                    ┌──────────┴──────────┐
                                    │   信道/干扰            │
                                    │ AWGN / 频偏 / 突发噪声 │
                                    └──────────┬──────────┘
                                               │
                    ┌──────────────────────────────────────────┐
                    │         polar_phy_receiver.py            │
                    │                                          │
  Info Bits ←───←── SGNN译码 ←── LLR ←── 同步+校正 ←── RRC ←── USRP/文件
  (K=128)          (20 iter)     (256 LLR)  (RS/PSS)   (匹配)
                    └──────────────────────────────────────────┘
```

## 帧结构

```
┌─────────────────────────────────────────────────────────────┐
│ PSS(32) │ RS(16) │ 数据(256 BPSK) │ Guard(32零) │
│ Zadoff-Chu │ 已知导频 │ 极化码码字(N=256) │ RRC隔离 │
└───────────┴────────┴────────────────┴──────────────┘
←──────────── 336 符号, 0.69ms @1Msps ────────────→
```

## 关键参数

| 参数 | 值 |
|------|-----|
| 极化码 | N=256, K=128, 码率 1/2 |
| 编码 | Arikan butterfly (deploy/common.py) |
| 译码 | SGNN, 20 迭代 (deploy/checkpoint/model.pt) |
| 调制 | BPSK (0→+1, 1→-1) |
| 同步 | RS 16符号滑动相关 |
| 频偏校正 | PSS 32符号/RS 16符号线性相位拟合 |
| 采样率 | 1 Msps, sps=2 |

## 文件清单

| 文件 | 功能 | 依赖 |
|------|------|------|
| `polar_phy_sender.py` | Polar编码+PHY TX | sender.py, deploy/common.py |
| `polar_phy_receiver.py` | PHY RX+SGNN+BER | receiver.py, deploy/receiver.py |
| `interferer.py` | 干扰器(仿真+USRP) | sim_channel.py |

## 仿真测试方案

```bash
# 不同 SNR 下测试 BER
for snr in 0 2 4 6 8 10 12 15; do
    python polar_phy_sender.py --mode sim --num-frames 500 --sim-file test.npy
    python sim_channel.py test.npy test_rx.npy --snr-db %snr% --freq-offset 2000
    python polar_phy_receiver.py --mode sim --sim-file test_rx.npy
done
```

## 硬件测试方案

```
三台 USRP 配置:
  USRP-1 (MyB210_01):  发送 (polar_phy_sender.py)
  USRP-2 (MyB210):     接收 (polar_phy_receiver.py)
  USRP-3 (可选):        干扰 (interferer.py, 同频突发噪声)
```

## 干扰方案

### 仿真干扰 (sim_channel.py 已支持)
- `--snr-db 10` — AWGN
- `--freq-offset 2000` — 频偏
- `--multipath "1.0,0.3@3,0.1@7"` — 多径

### 硬件干扰 (interferer.py)
- USRP-3 同频发送 BPSK 随机符号或突发噪声
- 干扰功率通过 TX gain 控制
- 突发模式: 随机间隔发送噪声脉冲
