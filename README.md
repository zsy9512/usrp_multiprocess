# USRP BPSK-Polar 无线通信系统

基于 USRP（B210）的软件无线电物理层，C++ PHY + Python 极化码编解码，stdin/stdout IPC 管道。

## 架构

```
TX: polar_encode.py → stdout → cpp/tx → USRP/文件
                                    ↓ (空中接口)
RX: USRP/文件 → cpp/rx → stdout → polar_decode.py (硬判 / --sgnn)
```

## 帧结构

```
STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)
   ↓ 4×16重复  ↓ ZC u=25   ↓ 已知导频  ↓ id+CRC   ↓ 极化码码字   ↓ CRC16  ↓ RRC隔离
```

共 **496 符号**，空口 **1.03ms** @ 1Msps (1012 样本/帧)。

## IPC 帧格式

| 方向 | 格式 | 字节/帧 |
|------|------|--------|
| Python→C++ TX | `[2B id(BE) 2B pad][32B codeword]` | 36 |
| C++ RX→Python | `[2B id(BE) 2B pad][1024B LLR(256 float32)][1B crc]` | 1029 |

## 快速开始

### C++ 编译

```bash
cd cpp && make    # 需要 g++ (MinGW/MSYS2)
```

### 理想信道端到端

```bash
python polar_encode.py --frames 20 --save-info info.npy | ./cpp/tx -o tx.bin
./cpp/rx --crc-filter tx.bin | python polar_decode.py --ref info.npy
```

### AWGN + CFO

```bash
python polar_encode.py --frames 50 --save-info info.npy | ./cpp/tx -o tx.bin
python sim_channel.py tx.bin rx.bin --snr-db 10 --freq-offset 500
./cpp/rx --crc-filter rx.bin | python polar_decode.py --ref info.npy
```

### SGNN 译码（需要 torch）

```bash
# 管道模式（理想信道）
./cpp/rx --crc-filter tx.bin | python polar_decode.py --sgnn --ref info.npy

# 文件模式（conda 环境）
./cpp/rx --crc-filter rx.bin > llr.bin
conda run -n pyg_test_py310 python sgnn_file_decode.py llr.bin --ref info.npy
```

### SGNN 闭环仿真（无 PHY）

```bash
conda run -n pyg_test_py310 python deploy/simulate.py \
  --checkpoint deploy/checkpoint/polar_GNN_20_iter_0_epoches_13.pt \
  --ebn0-range 1.0,5.0,1.0 --frames 500
```

### Python PHY 仿真（参考/调试）

```bash
python sender.py --mode sim --num-frames 50 --sim-file tx.npy
python sim_channel.py tx.npy rx.npy --snr-db 15 --freq-offset 2000
python receiver.py --mode sim --sim-file rx.npy
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `cpp/tx_main.cpp` | C++ 发送端 (stdin→组帧→RRC) |
| `cpp/rx_main.cpp` | C++ 接收端 (IQ→同步→LLR, `--crc-filter`) |
| `cpp/Makefile` | 编译脚本 |
| `phy_params.py` | Python/C++ 共用参数 (帧结构/RRC/CRC/同步门限) |
| `sender.py` | Python PHY 发送端 (参考实现, 对齐 C++) |
| `receiver.py` | Python PHY 接收端 (参考实现, 三级同步链) |
| `sim_channel.py` | AWGN/CFO/相偏/多径信道 |
| `polar_encode.py` | 极化码编码 IPC (info→codeword, 36B/帧) |
| `polar_decode.py` | 极化码译码 IPC (LLR→info, 硬判/--sgnn) |
| `sgnn_file_decode.py` | SGNN 文件模式译码 (conda 环境) |
| `deploy/common.py` | SGNN 模型定义 (BPConv/LSTU) |
| `deploy/simulate.py` | SGNN 闭环仿真 (BER/FER) |
| `deploy/checkpoint/` | SGNN 训练权重 |
| `deploy/matrices/` | A.npy (冻结掩膜), pcm.npy (校验矩阵) |

## 同步方案 (三级同步链)

| 阶段 | 方法 | 功能 |
|------|------|------|
| ① STF | 延迟相关 L=32 | 包检测 + 粗 CFO |
| ② PSS | ZC 互相关 + 双判据 | 精定时 (ptm≥4) |
| ③ RS | 相位拟合 + LS 估计 | 细 CFO + 信道 + σ² |

## 验证结果

| 场景 | 帧检出 | CRC | BER |
|------|--------|-----|-----|
| 理想信道 | 4/5 | 100% | 0/512 |
| 20dB + 300Hz CFO | 4/10 | 100% (过滤后) | 0/512 |
| SGNN 仿真 5dB | — | — | 0% |

## 依赖

- Python 3.9+, numpy
- g++ (C++17, 编译 PHY)
- PyTorch (SGNN 译码, 可选)
- uhd (USRP 硬件, 可选)
