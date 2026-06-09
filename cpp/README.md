# C++ BPSK PHY — 编译环境 & 使用说明

## 文件结构

```text
cpp/
├── phy_dsp.h             # 公共 PHY DSP（STF/PSS/RS/RRC/BPSK/CRC/环形缓冲）
├── loopback.cpp          # USRP loopback（对应 Python 版 loopback_test.py）
├── tx_main.cpp           # 离线发射端，输出 interleaved float32 IQ .bin
├── rx_main.cpp           # 离线接收端，读取 .bin 做完整同步链
├── uhd_tx_main.cpp       # UHD 发射端（需 Boost）
├── uhd_rx_main.cpp       # UHD 接收端（需 Boost）
├── build_loopback.bat    # 一键编译 loopback + uhd_tx + uhd_rx（MSVC + UHD + Boost）
├── build_uhd_msvc.bat    # uhd_tx/uhd_rx 专用编译脚本（旧版，功能同 build_loopback.bat）
├── npy2bin.py            # .npy(complex64) → .bin 转换工具
├── Makefile              # Linux/MinGW 离线编译
├── uhd.def               # UHD DLL 符号导出定义
└── README.md             # 本文件
```

## 编译工具链

| 工具 | 路径 |
|------|------|
| g++ (MinGW-w64 8.1.0) | `C:\ProgramData\MATLAB\SupportPackages\R2025a\3P.instrset\mingw_w64.instrset\bin\g++.exe` |
| MSVC (VS2022 BuildTools) | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe` |
| VS2022 vcvars64.bat | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat` |
| MSVC (VS2026 Community) | `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe` |
| VS2026 vcvarsall.bat | `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat` |

## 依赖库

| 库 | 路径 | 用途 |
|----|------|------|
| UHD (include) | `C:\Program Files\UHD\include\` | loopback / uhd_tx / uhd_rx 编译 |
| UHD (lib) | `C:\Program Files\UHD\lib\uhd.lib` | 链接 |
| UHD (dll) | `C:\Program Files\UHD\bin\uhd.dll` | 运行时 |
| Boost 1.66 | `E:\PhD_work\code\usrp_hardware\boost_1_66_0\boost_1_66_0` | loopback / uhd_tx / uhd_rx 编译链接 |
| Boost stage libs | `E:\PhD_work\code\usrp_hardware\boost_1_66_0\boost_1_66_0\stage64s\lib\` | 静态链接 (vc1444-mt-s-x64-1_66) |

> **Boost 状态**：`build_loopback.bat` 已配置 VS2022 BuildTools + Boost 1.66 静态库路径，可直接编译 loopback/uhd_tx/uhd_rx。MinGW 离线收发（tx_main/rx_main）**不需要** Boost。

## 编译命令

### 离线收发（无需 UHD）

```bash
set "PATH=C:\ProgramData\MATLAB\SupportPackages\R2025a\3P.instrset\mingw_w64.instrset\bin;%PATH%"
cd e:\PhD_work\code\usrp_hardware\usrp_multiprocess\cpp
g++ -std=c++17 -O3 -march=native tx_main.cpp -o tx.exe -lm
g++ -std=c++17 -O3 -march=native rx_main.cpp -o rx.exe -lm
```

Linux/macOS 可直接用 `make`。

### USRP loopback + UHD 收发（MSVC，推荐）

`build_loopback.bat` 调用 VS2022 BuildTools 的 vcvars64.bat，一次编译三个目标：

```bash
cd e:\PhD_work\code\usrp_hardware\usrp_multiprocess\cpp
build_loopback.bat
```

产物：

```text
loopback_msvc.exe   # USRP 单板自发自收（对应 loopback_test.py）
uhd_tx_msvc.exe     # UHD 发射端
uhd_rx_msvc.exe     # UHD 接收端
```

运行 loopback：

```bash
loopback_msvc.exe --args serial=320F33F --freq 915e6 --gain-tx 65 --gain-rx 64
```

### UHD 硬件收发（MinGW，需手动配置 Boost）

```bash
g++ -std=c++17 -O3 -march=native -I"C:\Program Files\UHD\include" -I<boost_include> uhd_tx_main.cpp -o uhd_tx.exe -L"C:\Program Files\UHD\lib" -luhd -lm
g++ -std=c++17 -O3 -march=native -I"C:\Program Files\UHD\include" -I<boost_include> uhd_rx_main.cpp -o uhd_rx.exe -L"C:\Program Files\UHD\lib" -luhd -lm
```

### 安装 Boost（仅 MinGW UHD 编译需要）

```bash
# 方案1: conda (网络正常时)
conda install -n pyg_test_py310 -c conda-forge boost

# 方案2: 手动下载预编译包
# https://sourceforge.net/projects/boost/files/boost-binaries/
# 选 mingw 8.1.0 对应版本, 解压到 C:\boost_1_xx_x\
```

## Python 环境

| 项目 | 值 |
|------|-----|
| 环境名 | `pyg_test_py310` |
| Python 路径 | `C:\Users\a\miniconda3\envs\pyg_test_py310\python.exe` |
| UHD Python | 待确认 (`import uhd` 需在 pyg_test_py310 下测试) |

## 验证结果

### Python 离线
```
python test_phy_offline.py --test A --frames 50
→ SNR≥15dB: CRC 100% 通过
→ CFO 0~1000Hz: 独立帧 CRC 98-100%
```

### C++ 离线
```
tx.exe --random --num-frames 50 -o tx_iq.bin
rx.exe --crc-filter tx_iq.bin
→ 49/49 frames CRC PASS (100%)
```

### C++ USRP loopback (B210, self-loop)
```
loopback_msvc.exe --args serial=320F33F --freq 915e6 --gain-tx 65 --gain-rx 64
→ CRC=998/999 (99.9%), HDR=999, BER=0.00% (0/255744)
→ 与 Python loopback_test.py 结果一致
```

## 修改摘要

### 2026-06-09
- **线程启动时序**：`proc_th` 提前到 RX 启动后立即执行（原为与 TX 同时启动），避免 1 秒噪声 backlog 导致同步失败
- **Ring backlog 清空**：TX 前通过 `skip_to` 机制清空环形缓冲中的噪声数据
- **Latency 统计修复**：仅在 `hdrOk && fid < g_tx_ts.size()` 时累计延迟，新增 `lat_count` 独立计数
- **BER 统计修复**：`g_results` 仅在 `hdrOk` 时推入，确保按 `fid` 对齐的参考比特有效

### 2026-06-07
1. **TS → Ts_sym**: symbol-domain CFO functions use `g_ts_sym` instead of `g_ts`
2. **PSS correlation**: use `conj(ref_pss[j])` (forward), not `conj(ref_pss[M-1-j])` (reversed)
3. **PSS dual-peak**: back-to-back frames produce two equal PSS peaks → accept ptm≥8 even with pts<1.5
4. **frame_id header**: unified Python/C++ format (frame_id + header_crc)
