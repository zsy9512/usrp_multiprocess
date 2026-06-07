# B210 工具 & 测试指令

设备: TX=320F2BD (MyB210_01), RX=320F33F (MyB210)
连接: TX/TX/RX ← SMA+30dB衰减器 → RX/RX2

---

## C++ 发射机

```powershell
cd e:\PhD_work\code\usrp_hardware\usrp_multiprocess\cpp
uhd_tx_msvc.exe --freq 915e6 --gain 65 --rate 1e6 --num-frames 200 --frame-gap-ms 3
```

## Python 接收 (多进程默认)

```powershell
cd e:\PhD_work\code\usrp_hardware\usrp_multiprocess
C:\Users\a\miniconda3\envs\pyg_test_py310\python.exe -u receiver.py --mode hardware --freq 915e6 --gain 35 --usrp-args "serial=320F33F" --stf-threshold 0.6
```

## Python IQ 录制

```powershell
cd e:\PhD_work\code\usrp_hardware\usrp_multiprocess
C:\Users\a\miniconda3\envs\pyg_test_py310\python.exe -u tools/iq_recorder.py --serial 320F33F --freq 915e6 --gain 35 --duration 5 -o capture.npy
```

## Python 频谱仪

```powershell
cd e:\PhD_work\code\usrp_hardware\usrp_multiprocess
C:\Users\a\miniconda3\envs\pyg_test_py310\python.exe -u tools/live_spectrum.py --serial 320F33F --freq 915e6 --gain 40
```

## IQ 分析 + 增益推荐

```powershell
cd e:\PhD_work\code\usrp_hardware\usrp_multiprocess
C:\Users\a\miniconda3\envs\pyg_test_py310\python.exe tools/iq_analyzer.py capture.npy --save analysis
```

---

## 环回采集 & 突发干扰测试 (loopback_capture + loopback_analyze + burst_interferer)

### 硬件连接

```
TX: ch0 (A板 TX/RX) ── SMA ──→ RX: ch0 (A板 RX2)    同板回环, 有泄漏拖尾
TX: ch0 (A板 TX/RX) ── SMA ──→ RX: ch1 (B板 RX2)    跨板回环, 隔离 >60dB ✅
```

双板 B210 只有 channel 0 和 channel 1，分别对应 RFA 和 RFB。

### 采集流程

**终端 1 — 启动干扰源 (320F2BD)**
```powershell
cd e:\PhD_work\code\usrp_hardware\usrp_multiprocess
python burst_interferer.py --sigma-b 0.5    # 弱干扰
python burst_interferer.py --sigma-b 1.0    # 中等干扰
python burst_interferer.py --sigma-b 2.0    # 强干扰
```

**终端 2 — 环回采集 (320F33F)**
```powershell
cd e:\PhD_work\code\usrp_hardware\usrp_multiprocess

# A板回环 (同板, 有泄漏)
python tools/loopback_capture.py --serial 320F33F --rx-channel 0 --rx-antenna RX2 -o capture/baseline_a

# B板回环 (跨板, 无泄漏)
python tools/loopback_capture.py --serial 320F33F --rx-channel 1 --rx-antenna RX2 --gain-rx 64 -o capture/baseline

# 干扰下采集
python tools/loopback_capture.py --serial 320F33F --rx-channel 0 --rx-antenna RX2 -o capture/int_sb10
python tools/loopback_capture.py --serial 320F33F --rx-channel 0 --rx-antenna RX2 -o capture/int_sb20
```

### 分析

```powershell
python tools/loopback_analyze.py capture/baseline --plot
python tools/loopback_analyze.py capture/baseline --compare capture/int_sb20
```

### 参数速查

| 工具 | 关键参数 | 默认 |
|------|---------|------|
| `loopback_capture.py` | `--rx-channel` 0(A板)/1(B板), `--gain-rx` | ch1, 64dB |
| `burst_interferer.py` | `--sigma-b`, `--p-b` | 2.0, 0.05 |
| `loopback_analyze.py` | `--plot`, `--compare <prefix>` | — |
