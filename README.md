# USRP BPSK PHY 硬件环回与同步链实验

本仓库用于在 USRP B210 上验证一套窄带 BPSK 物理层链路。代码覆盖从离线仿真、Python/UHD 硬件收发、单机 loopback 采集分析，到 C++ 版本 PHY 复现和 Polar-SGNN 译码仿真。当前主链路以 1 Msps 复基带采样、2 samples/symbol、RRC 脉冲成形为基础，重点验证帧检测、同步、CFO 校正、信道估计和 CRC 检查。

## 仓库目的

这个仓库主要服务三类实验：

1. 离线 PHY 验证：生成 BPSK 帧，通过 AWGN/CFO/相偏/多径等软件信道，验证同步链和 CRC。
2. USRP 硬件验证：使用 B210 做单板自发自收、双板收发、原始 IQ 录制和离线逐帧分析。
3. 译码链路部署：在 PHY 输出的硬判/LLR 之上，连接 Polar 编码、硬判译码和 SGNN 译码仿真。

仓库中的 `.npy` 采集文件、模型 checkpoint、`.obj/.exe/.dll` 等大文件或编译产物不应作为日常代码提交内容。根目录 `.gitignore` 已按这个原则忽略新生成的本地输出。

## 组织结构

```text
.
├── phy_params.py          # 全局 PHY 参数、参考序列、RRC、CRC16
├── sender.py              # Python BPSK 发射端，支持 sim/hardware
├── receiver.py            # Python BPSK 接收端，三级同步和多进程硬件接收
├── loopback_test.py       # 单 B210 loopback，主进程 UHD 收发，子进程 PHY 处理
├── hw_loopback.py         # 双 B210 TX/RX 硬件测试
├── test_phy_offline.py    # 离线测试矩阵：SNR、CFO、相偏、多径、定时、多帧
├── burst_interferer.py    # B210 突发干扰源
├── tools/
│   ├── iq_recorder.py     # 原始 IQ 录制
│   ├── iq_analyzer.py     # IQ 幅度、频谱、STF 候选诊断
│   ├── live_spectrum.py   # 实时频谱仪
│   ├── loopback_capture.py# loopback IQ + TX bits 采集
│   └── loopback_analyze.py# loopback 逐帧同步、CRC、BER、绘图和参数扫描
├── cpp/
│   ├── phy_dsp.h          # C++ 版 PHY DSP 公共实现
│   ├── tx_main.cpp        # 离线 C++ 发射端，输出 interleaved float32 IQ
│   ├── rx_main.cpp        # 离线 C++ 接收端
│   ├── uhd_tx_main.cpp    # UHD C++ 发射端
│   ├── uhd_rx_main.cpp    # UHD C++ 接收端
│   └── README.md          # C++ 编译环境和命令
├── deploy/
│   ├── common.py          # Polar/SGNN 自包含部署工具
│   ├── simulate.py        # Polar-SGNN 闭环仿真
│   ├── matrices/          # Polar/LDPC 图结构矩阵
│   └── checkpoint/        # SGNN 权重
├── polar_encode.py        # Polar 编码器，stdout 管道输出码字
└── polar_decode.py        # Polar 硬判/SGNN 译码器，stdin 管道输入 LLR
```

## PHY 帧结构

帧结构在 `phy_params.py` 中统一定义，Python 和 C++ 代码应保持一致。

| 字段 | 符号数 | 用途 |
|------|--------|------|
| STF | 64 | 4 组重复的 16 符号 BPSK 序列，用于粗包检测和粗 CFO |
| PSS | 64 | Zadoff-Chu 序列，`u=25`，用于精定时和候选确认 |
| RS | 32 | 固定 BPSK 参考符号，用于细 CFO、公共相位、信道幅度和噪声方差估计 |
| Header | 32 | `frame_id(16 bit) + header_crc16(16 bit)`，BPSK 调制 |
| Payload | 256 | 数据比特，BPSK 调制 |
| Payload CRC | 16 | Payload 的 CRC16-IBM |
| Guard | 32 | 零符号保护间隔，用于滤波尾巴和帧间隔 |

关键参数：

```text
SPS = 2
sample_rate = 1e6
symbol_rate = 500 ksym/s
RRC rolloff = 0.35
RRC taps = 21
FRAME_SYMBOLS = 496
FRAME_RRC_SAMPLES = 1012
STF_DELAY = 32 samples
```

发射端组帧流程：

```text
Payload bits
  -> Payload CRC16
  -> Header(frame_id + header CRC16)
  -> BPSK 映射
  -> STF + PSS + RS + Header + Payload + CRC + Guard
  -> 上采样 SPS=2
  -> RRC 滤波
  -> 保存 IQ 或通过 UHD 发射
```

## 以 loopback 为例的同步链

`loopback_test.py` 是最完整的硬件闭环示例。它在同一个 B210 上启动 TX 和 RX，RX 样本进入共享内存环形缓冲，PHY 子进程从缓冲中取数据并完成逐帧同步和解调。这样 USRP 收样线程不被 Python DSP 处理阻塞，目标是降低 overflow。

### 1. STF 延迟相关：粗检测和粗 CFO

STF 由 4 组相同的 16 符号 BPSK 序列组成。由于 `SPS=2`，延迟相关间距为：

```text
L = STF_REP * SPS = 16 * 2 = 32 samples
```

接收端计算：

```text
P(d) = sum r[d+n] * conj(r[d+n+L])
E(d) = sum |r[d+n+L]|^2
M(d) = |P(d)| / E(d)
```

当 `M(d) > STF_THRESHOLD` 且局部能量超过 `STF_MIN_ENERGY` 时，该位置成为 STF 候选。候选按度量排序，并在约 128 samples 的窗口内聚类去重，只保留最强峰。粗 CFO 来自 `P(d)` 的相位：

```text
coarse_cfo = -angle(P) / (2*pi*L/sample_rate)
```

### 2. PSS 互相关：精定时和候选确认

对 STF 粗位置前后提取一段样本，做 RRC 匹配滤波并抽取到符号率。随后用 PSS 的共轭反向序列做互相关，得到 PSS 峰值位置。

PSS 候选需要同时满足两个质量判据：

```text
peak_to_mean   >= 3.5 或接收端配置值
peak_to_second >= 1.5 或接收端配置值
```

找到 PSS 峰后，帧起始符号为：

```text
frame_start_sym = pss_peak - STF_LEN
rs_start_sym = frame_start_sym + STF_LEN + PSS_LEN
```

如果帧起始为负、RS/Header/Payload/CRC 超出窗口，候选会被拒绝。

### 3. RS 相位拟合：细 CFO、信道和噪声估计

RS 段首先按 STF 估计的粗 CFO 做预补偿，然后计算：

```text
rs_tone[n] = rs_seg[n] * conj(RS[n])
```

接收端对 `angle(rs_tone)` unwrap 后做线性拟合，斜率换算为残余细 CFO。当前代码默认拒绝 `abs(fine_cfo) > 500 Hz` 的候选。通过总 CFO 补偿后的 RS 估计复信道：

```text
h = mean(rs_corrected * conj(RS))
sigma2 = var(rs_corrected / h - RS) * RS_LEN / (RS_LEN - 1)
```

`h` 同时提供幅度和公共相位；`sigma2` 用于 SNR/EVM 诊断。

### 4. Header/Payload 解调和 CRC

Header 和 Payload 都使用 BPSK 硬判决。解调前对每个符号应用总 CFO 补偿，并除以 RS 估计出的复信道 `h`：

```text
y_eq = symbol * exp(-j*2*pi*total_cfo*t) / h
bit = 1 if real(y_eq) < 0 else 0
```

Header CRC 通过表示 `frame_id` 可信；Payload CRC 通过表示 256 bit 数据字段通过端到端校验。`loopback_test.py` 还保存每个发射帧的参考 payload bits，可按 `frame_id` 统计 BER 和平均处理延迟。

## 常用测试代码

### 1. Python 仿真发射与接收

生成若干帧 IQ：

```powershell
python sender.py --mode sim --num-frames 50 --sim-file tx_iq.npy --save-bits
```

接收仿真 IQ：

```powershell
python receiver.py --mode sim --sim-file tx_iq.npy
```

### 2. 离线测试矩阵

`test_phy_offline.py` 包含：

| 测试 | 覆盖内容 |
|------|----------|
| A | SNR sweep |
| B | CFO 容忍度 |
| C | 初始相位容忍度 |
| D | 多径容忍度 |
| E | 采样定时误差 |
| F | 连续多帧 |

运行示例：

```powershell
python test_phy_offline.py --test F --frames 200
python test_phy_offline.py --test ALL --frames 200 --log-file test_results.txt
```

注意：按当前代码，A-E 的 `run_test_single()` 会读取 `receiver.get_stats()` 中尚未返回的 `total_bits/total_errors` 字段；如果运行时报 `KeyError`，需要先补齐接收端 BER 统计字段，或先使用 F 测试验证连续多帧检测和 CRC 链路。

### 3. 单 B210 loopback

硬件连接示例：

```text
TX/RX 端口 -> 衰减器或合适线缆 -> RX2 端口
```

运行：

```powershell
python loopback_test.py --serial 320F33F --freq 915e6 --gain-tx 65 --gain-rx 64 --rx-channel 0 --rx-antenna RX2 --num-frames 1000 --frame-gap-ms 5
```

常见调整：

```text
--rx-channel 0      使用 A 板通道
--rx-channel 1      使用另一个 RX 通道，适合跨通道隔离测试
--gain-tx           发射增益
--gain-rx           接收增益
--frame-gap-ms      帧间零填充，gap 太小会提高误检和处理压力
```

### 4. 双 B210 硬件测试

```powershell
python hw_loopback.py --serial-tx 320F2BD --serial-rx 320F33F --freq 915e6 --gain-tx 30 --gain-rx 30 --num-frames 50 --frame-gap-ms 2
```

若两台 B210 频偏较大，可尝试外部参考：

```powershell
python hw_loopback.py --sync-mode external_ref
```

### 5. C++ 离线收发

详见 `cpp/README.md`。基本流程：

```powershell
cd cpp
tx.exe --random --num-frames 50 -o tx_iq.bin
rx.exe tx_iq.bin
```

`polar_encode.py` 和 `polar_decode.py` 提供了 Polar 码 stdin/stdout 管道接口。当前 `cpp/rx_main.cpp` 主要打印 PHY 同步和 CRC 统计；若要直接串接 `polar_decode.py`，接收端需要输出 `[4B frame_id][256 float32 LLR][1B crc_ok]` 格式的数据流。

```powershell
python ..\polar_encode.py --frames 100 --save-info info.npy | tx.exe -o tx_iq.bin
```

## tools 使用方案

### 1. 原始 IQ 录制

用于确认硬件是否收到信号，或保存任意空口片段供离线分析：

```powershell
python tools/iq_recorder.py --serial 320F33F --freq 915e6 --gain 35 --duration 5 -o capture.npy
```

输出：

```text
capture.npy  # complex64 IQ
```

### 2. IQ 幅度、频谱和候选帧诊断

```powershell
python tools/iq_analyzer.py capture.npy
python tools/iq_analyzer.py capture.npy --plot
python tools/iq_analyzer.py capture.npy --save analysis
```

该工具会报告峰值幅度、RMS、削峰比例、频谱峰值频偏、STF 候选帧数和帧能量分布。它适合在解调失败前先判断“信号太弱、增益过高削峰、频偏太大、帧间距异常”等问题。

### 3. 实时频谱仪

```powershell
python tools/live_spectrum.py --serial 320F33F --freq 915e6 --gain 40
```

用途：

```text
观察中心频率附近是否有发射信号
粗看频偏和带宽
调 TX/RX 增益前先检查是否有明显削峰或无信号
```

### 4. loopback 采集

`loopback_capture.py` 会自发自收并保存两类文件：

```text
<prefix>_iq.npy    # 接收 IQ
<prefix>_bits.npy  # 每帧 TX payload 参考比特
```

示例：

```powershell
python tools/loopback_capture.py --serial 320F33F --rx-channel 1 --rx-antenna RX2 --gain-tx 65 --gain-rx 64 --num-frames 200 -o capture/baseline
```

采集带突发干扰的样本时，可另开终端先启动干扰源：

```powershell
python burst_interferer.py --serial 320F2BD --freq 915e6 --gain 50 --sigma-b 2.0 --p-b 0.05
python tools/loopback_capture.py --serial 320F33F --rx-channel 1 --rx-antenna RX2 -o capture/int_sb20
```

### 5. loopback 逐帧分析

基础分析：

```powershell
python tools/loopback_analyze.py capture/baseline
```

参数扫描：

```powershell
python tools/loopback_analyze.py capture/baseline --scan
```

绘图：

```powershell
python tools/loopback_analyze.py capture/baseline --plot
python tools/loopback_analyze.py capture/baseline --save-plot capture/baseline
```

对比两次采集，例如 baseline 和干扰条件：

```powershell
python tools/loopback_analyze.py capture/baseline --compare capture/int_sb20
```

常用门限：

```text
--ptm          PSS peak_to_mean 门限，默认 3.5
--pts          PSS peak_to_second 门限，默认 1.0
--stf-energy   STF 能量门限，0 表示使用 phy_params.py 默认值
--debug N      打印前 N 个候选帧的同步细节
```

## Polar/SGNN 相关

`polar_encode.py` 和 `polar_decode.py` 保留了 stdin/stdout 管道接口，便于后续把 PHY 和 Polar/SGNN 译码链路连接起来。按当前代码，`deploy/simulate.py` 是更完整的 Polar-SGNN 闭环入口，用于在纯软件中评估 AWGN 或突发噪声下的 BER/FER。

```powershell
python deploy/simulate.py --checkpoint deploy/checkpoint/polar_GNN_20_iter_0_epoches_13.pt --ebn0-range 1.0,5.0,0.5 --frames 2000
python deploy/simulate.py --checkpoint deploy/checkpoint/polar_GNN_20_iter_0_epoches_13.pt --ebn0 3.0 --sigma-b 2.0 --burst-prob 0.1 --frames 10000
```

## 推荐调试顺序

1. 先运行 `sender.py --mode sim` 和 `receiver.py --mode sim`，确认软件链路能检帧并通过 CRC。
2. 再运行 `tools/live_spectrum.py` 或 `tools/iq_recorder.py`，确认硬件侧有信号且幅度不过载。
3. 使用 `tools/iq_analyzer.py` 看幅度、频偏、STF 候选帧数和帧间距。
4. 使用 `tools/loopback_capture.py` 保存带参考比特的 loopback 数据。
5. 使用 `tools/loopback_analyze.py` 做逐帧同步、CRC、BER、SNR、CFO 和门限扫描。
6. 最后运行 `loopback_test.py` 或 `hw_loopback.py` 做实时硬件链路验证。
