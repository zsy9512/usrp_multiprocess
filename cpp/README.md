# C++ BPSK PHY — 编译环境 & 使用说明

## 编译工具链

| 工具 | 路径 |
|------|------|
| g++ (MinGW-w64 8.1.0) | `C:\ProgramData\MATLAB\SupportPackages\R2025a\3P.instrset\mingw_w64.instrset\bin\g++.exe` |
| MSVC cl.exe (VS2022 BuildTools) | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe` |
| VS2022 vcvars64.bat | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat` |

## 依赖库

| 库 | 路径 | 用途 |
|----|------|------|
| UHD (include) | `C:\Program Files\UHD\include\` | uhd_tx / uhd_rx 编译 |
| UHD (lib) | `C:\Program Files\UHD\lib\uhd.lib` | 链接 |
| UHD (dll) | `C:\Program Files\UHD\bin\uhd.dll` | 运行时 |
| **Boost** | ❌ **缺失！** | UHD C++ 编译需要 `boost/config.hpp` |

## 编译命令

### 离线收发（无需 UHD）

```bash
set "PATH=C:\ProgramData\MATLAB\SupportPackages\R2025a\3P.instrset\mingw_w64.instrset\bin;%PATH%"
cd e:\PhD_work\code\usrp_hardware\usrp_multiprocess\cpp
g++ -std=c++17 -O3 -march=native tx_main.cpp -o tx.exe -lm
g++ -std=c++17 -O3 -march=native rx_main.cpp -o rx.exe -lm
```

### UHD 硬件收发（需 Boost）

```bash
g++ -std=c++17 -O3 -march=native -I"C:\Program Files\UHD\include" -I<boost_include> uhd_tx_main.cpp -o uhd_tx.exe -L"C:\Program Files\UHD\lib" -luhd -lm
g++ -std=c++17 -O3 -march=native -I"C:\Program Files\UHD\include" -I<boost_include> uhd_rx_main.cpp -o uhd_rx.exe -L"C:\Program Files\UHD\lib" -luhd -lm
```

### 安装 Boost（解决 UHD 编译）

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

## 修改摘要 (2026-06-07)

### Bugs Fixed
1. **TS → Ts_sym**: symbol-domain CFO functions use `g_ts_sym` instead of `g_ts`
2. **PSS correlation**: use `conj(ref_pss[j])` (forward), not `conj(ref_pss[M-1-j])` (reversed)
3. **PSS dual-peak**: back-to-back frames produce two equal PSS peaks → accept ptm≥8 even with pts<1.5
4. **frame_id header**: unified Python/C++ format (frame_id + header_crc)

### New Files
- `sync_config.py` — B210 clock configuration (host / external_ref)
- `cpp/phy_dsp.h` — shared DSP for all C++ targets
- `cpp/uhd_tx_main.cpp` — UHD TX thin wrapper (needs Boost)
- `cpp/uhd_rx_main.cpp` — UHD RX thin wrapper (needs Boost)
