# USRP B210 BPSK 物理层系统

单/双 B210 的 BPSK 收发系统，含 **Python 实时环回** 和 **C++ 高性能 PHY + Polar 编码** 两条链路，共用帧结构和同步算法。

## 帧结构

```
STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)
   ↓           ↓           ↓          ↓            ↓           ↓        ↓
 4×16重复    ZC u=25     已知导频   id+CRC16    256 数据位   CRC16    RRC隔离
```

| 参数 | 值 |
|------|-----|
| 符号率 | 500k sym/s (SPS=2, 1 Msps) |
| 调制 | BPSK |
| 帧长 | 496 符号 → 1012 样本 (~1.01ms) |
| 脉冲成形 | RRC, rolloff=0.35, 10 符号/边 |
| CRC | CRC16-IBM (x¹⁶+x¹⁵+x²+1) |

## 三级同步链

| 阶段 | 方法 | 估计量 | 关键参数 |
|------|------|--------|----------|
| ① STF | 延迟自相关 (L=32) | 粗包检测 + 粗 CFO | `STF_THRESHOLD=0.4`, `STF_MIN_ENERGY=0.64` |
| ② PSS | ZC 互相关 + 双判据 | 精定时 | `ptm≥3.5` (峰均比), `pts≥1.5` (峰次比) |
| ③ RS | 相位拟合 + LS 估计 | 细 CFO + 信道 h + 噪声 σ² | `rs_corr > RS_LEN×0.3` |

> **ptm (peak-to-mean)**: PSS 相关主峰 ÷ 全局均值。高→峰突出；低→信号弱或窗口噪声大。
> **pts (peak-to-second)**: 主峰 ÷ 远离的次高峰。高→timing 无歧义；低→可能锁错位置。

---

## 快速开始

### 环境

```bash
conda activate pyg_test_py310    # Python 3.10 + numpy + uhd
```

### 1. 查询设备

```bash
uhd_find_devices
# 输出示例:
#   Device 0:  serial=320F2BD  name=MyB210_01  type=B210
#   Device 1:  serial=320F33F  name=MyB210     type=B210
```

### 2. Python SMA 环回 (单 B210)

```bash
# SMA 线缆: TX/RX ←→ RX2
python loopback_test.py --serial 320F33F --gain-tx 60 --gain-rx 45 --num-frames 1000
```

预期: 1000/1000 帧检出, HDR≈100%, CRC≈94%.

### 3. 空口测试 (单 B210 或双 B210)

单 B210 自发自收 (TX/RX 发射, RX2 接收):

```bash
python loopback_test.py --serial 320F33F --gain-tx 65 --gain-rx 68 --num-frames 1000
```

双 B210 (需外部时钟同步或各自 internal):

```bash
# TX 端
python sender.py --mode hardware --freq 915e6 --gain 65 --num-frames 0 --usrp-args serial=320F2BD

# RX 端
python receiver.py --mode hardware --freq 915e6 --gain 68 --usrp-args serial=320F33F
```

### 4. 增益校准

BPSK 裸调制 SNR-BER 曲线在 3-8dB 区间极陡（悬崖效应）。3dB 增益变化可使 CRC 从 94%→12%。

**校准流程:**
```bash
# 固定 TX gain, 扫 RX gain 找 CRC 最高者 (每个值 200 帧即可)
python loopback_test.py --serial 320F33F --gain-tx 60 --gain-rx 40 --num-frames 200
python loopback_test.py --serial 320F33F --gain-tx 60 --gain-rx 43 --num-frames 200
python loopback_test.py --serial 320F33F --gain-tx 60 --gain-rx 45 --num-frames 200
python loopback_test.py --serial 320F33F --gain-tx 60 --gain-rx 47 --num-frames 200
```

**参考工作点:**

| 场景 | TX gain | RX gain | peak | CRC |
|------|---------|---------|------|-----|
| SMA 环回 | 60 | 45 | ~1.0 | ~94% |
| 空口 (5m) | 65 | 68 | ~0.8 | ~100% |

目标: `peak` 在 0.5~0.95, `clipped=0%`.

---

## 离线 IQ 分析

```bash
# 采集
python tools/loopback_capture.py --serial 320F33F --gain-tx 60 --gain-rx 45 \
    --num-frames 200 -o capture/test

# 参数扫描 (找最优 PSS 门限)
python tools/loopback_analyze.py capture/test --scan

# 逐帧分析 + 可视化
python tools/loopback_analyze.py capture/test --ptm 3.5 --pts 1.5 --plot
```

分析器输出: SNR / CFO / EVM / 定时偏 / BER / 星座图 / 帧间距。

---

## C++ 高性能 PHY (Polar 编码)

C++ PHY 通过 stdin/stdout IPC 与 Python Polar 编解码对接。

### 编译

```bash
cd cpp
# MSVC
build_uhd_msvc.bat

# MinGW/MSYS2
make
```

### 仿真链路

```bash
# 编码 → C++ TX → 信道 → C++ RX → 译码
python polar_encode.py --frames 50 --save-info info.npy | ./cpp/tx -o tx.bin
python sim_channel.py tx.bin rx.bin --snr-db 10 --freq-offset 500
./cpp/rx --crc-filter rx.bin | python polar_decode.py --ref info.npy
```

### SGNN 译码 (torch)

```bash
./cpp/rx --crc-filter rx.bin > llr.bin
conda run -n pyg_test_py310 python sgnn_file_decode.py llr.bin --ref info.npy
```

### IPC 帧格式

| 方向 | 格式 |
|------|------|
| Python→C++ TX | `[2B id(BE) 2B pad][32B codeword]` = 36B/帧 |
| C++ RX→Python | `[2B id(BE) 2B pad][1024B LLR(256×float32)][1B crc]` = 1029B/帧 |

---

## 文件索引

### Python PHY (实时环回)

| 文件 | 功能 |
|------|------|
| `phy_params.py` | **统一参数源**: 帧结构、参考序列、RRC、CRC、同步门限 |
| `loopback_test.py` | 单 B210 实时自发自收 (多进程零拷贝) |
| `sender.py` | BPSK 发送端 (含 Polar 编码, 仿真+硬件) |
| `receiver.py` | BPSK 接收端 (含三级同步链, 仿真+硬件) |
| `sim_channel.py` | AWGN / CFO / 相偏 / 多径信道仿真 |
| `test_phy_offline.py` | 离线 PHY 测试 |
| `hw_loopback.py` | 硬件环回辅助 |
| `sync_config.py` | 双 B210 时钟同步配置 |

### 离线分析工具 (`tools/`)

| 文件 | 功能 |
|------|------|
| `loopback_capture.py` | IQ 采集: B210 自发自收 → `_iq.npy` + `_bits.npy` |
| `loopback_analyze.py` | 逐帧分析: 星座图/EVM/CFO/BER/定时偏, `--scan` 参数扫描 |
| `live_spectrum.py` | 实时频谱仪 (目测信号幅度, 帮助定增益) |
| `iq_recorder.py` | 通用 IQ 录制 (纯 RX) |
| `iq_analyzer.py` | IQ 文件分析: 功率谱/STF检测/增益诊断 |

### C++ PHY (`cpp/`)

| 文件 | 功能 |
|------|------|
| `phy_dsp.h` | **C++ 统一 DSP**: 帧参数、序列生成、RRC、STF/PSS/RS 同步、LLR 解调 |
| `tx_main.cpp` | C++ 发送端 (stdin→组帧→RRC→文件) |
| `rx_main.cpp` | C++ 接收端 (文件→同步→LLR→stdout) |
| `uhd_tx_main.cpp` | C++ USRP 发送端 |
| `uhd_rx_main.cpp` | C++ USRP 接收端 |
| `Makefile` | g++ 编译 |
| `build_uhd_msvc.bat` | MSVC 编译 |

### Polar 编码 + SGNN (`deploy/`)

| 文件 | 功能 |
|------|------|
| `polar_encode.py` | Polar 编码 (info bits → codeword), stdin/stdout IPC |
| `polar_decode.py` | Polar 译码 (LLR → info), 硬判 + SGNN |
| `sgnn_file_decode.py` | SGNN 文件模式译码 |
| `deploy/common.py` | SGNN 模型定义 (BPConv/LSTU) |
| `deploy/simulate.py` | SGNN 闭环仿真 (BER/FER vs EbN0) |
| `deploy/checkpoint/` | SGNN 训练权重 |
| `deploy/matrices/` | A.npy (冻结掩膜), pcm.npy (校验矩阵) |

---

## 关键参数速查

所有参数定义在 `phy_params.py` 和 `cpp/phy_dsp.h` 中，两侧保持一致。

| 参数 | Python | C++ | 默认值 | 说明 |
|------|--------|-----|--------|------|
| SPS | `SPS` | `SPS` | 2 | 每符号采样数 |
| STF_THRESHOLD | `0.4` | `0.40f` | 0.4 | STF 归一化相关门限 |
| STF_MIN_ENERGY | `0.02*STF_DELAY` | `0.02f*STF_DELAY` | 0.64 | STF 最小能量 (3.2→0.64 修复弱信号漏检) |
| PSS ptm | `3.5` | `3.5f` | 3.5 | PSS 峰均比门限 (4.0→3.5) |
| PSS pts | `1.5` | `1.5f` | 1.5 | PSS 峰次比门限 |
| RS_CORR_THR | `0.3` | `0.3f` | 0.3 | RS 每符号最小相关性 |

---

## 已修复的 Bug

| Bug | 症状 | 修复 |
|-----|------|------|
| **双重相位补偿** | `y·exp(-jθ)/h` 相位转了 -2θ, 星座落在 Q 轴, BER=50% | 删除 `exp(-jθ)`, 仅保留 `/h` |
| **DBPSK 编解码不一致** | TX 差分编码 payload, RX 对 header 也做差分检测 | 统一改为标准 BPSK |
| **STF_MIN_ENERGY 过严** | peak<0.35 时 STF 无法检测 | 0.10→0.02 × STF_DELAY |
| **PSS ptm 过高** | 弱信号 PSS 峰均比不达标 | 4.0→3.5 |

Python (`loopback_test.py`, `receiver.py`) 和 C++ (`phy_dsp.h`, `rx_main.cpp`, `uhd_rx_main.cpp`) **双侧均已修复**。

---

## 已知限制

- BPSK 裸调制 SNR 门限 ~7dB (可用 Polar 码降到 ~3dB)
- 弱信号 (peak<0.5) 时 PSS 检出率下降 (STF 聚类选择噪声峰)
- 增益需精细调整 (SMA 和空口工作点不同)
- 单 B210 双端口 (TX/RX + RX2) 共用同一时钟，无 CFO；双 B210 需外部时钟同步
