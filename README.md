# USRP Polar Loopback

这是一个面向 USRP/B210 的 Polar 编码 loopback 实验项目。项目目标是在低信噪比和同步不稳定条件下，完整捕获接收 IQ，离线复盘同步链、信道状态和译码性能，并用 SGNN 译码器评估 Polar 链路在硬件信道中的表现。

默认链路工作在 915 MHz、1 Msps、2 samples/symbol。每个数据帧使用固定的 Polar 重复帧结构：

```text
STF(128) + PSS(64) + RS(64) + Header(32) + Payload(256) + CRC(16) + Guard(32)
= 592 symbols
```

Payload 是 N=256、K=128 的 Polar 码字，码率 R=1/2。发送端每个 `frame_id` 重复发送 5 次，离线接收端以 any-of-5 方式统计组检出率。

## 系统示意图

```text
                     +----------------------------+
                     |      loopback_capture      |
TX bits -> Polar ->  | BPSK frame -> RRC -> TX    |
                     | RX -> raw IQ capture       |
                     +-------------+--------------+
                                   |
                 +-----------------+-----------------+
                 |                 |                 |
                 v                 v                 v
      +-------------------+ +-------------------+ +------------------+
      | analyze_capture   | | polar_decode      | | live_spectrum    |
      | frame timing      | | frame timing      | | RX spectrum      |
      | PSS/RS diagnosis  | | RS channel est.   | | monitor          |
      | full-frame oracle | | Payload LLR       | +------------------+
      | channel metrics   | | Hard/SGNN BER     |
      +---------+---------+ +---------+---------+
                |                   |
                v                   v
      sync / channel report   BER / LLR report

External interference:
  burst_interferer -> USRP TX noise source
```

## 目录结构

```text
.
|-- README.md
|-- environment.yml
|-- burst_interferer.py
|-- phy_params.py
|-- deploy/
|   |-- common.py
|   |-- checkpoint/
|   |   `-- polar_GNN_20_iter_0_epoches_13.pt
|   `-- matrices/
|       |-- A.npy
|       `-- pcm.npy
`-- tools/
    |-- analyze_repeat_capture.py
    |-- batch_capture.py
    |-- live_spectrum.py
    |-- loopback_capture.py
    `-- polar_offline_decode.py
```

`capture/`、`plots/` 和 `__pycache__/` 是运行输出目录，不应提交。`deploy/` 是运行资产目录，包含 SGNN checkpoint 和矩阵文件。

## 环境安装

`environment.yml` 从 `pyg_test_py310` 导出，用于 Conda 迁移安装。推荐在目标机器创建独立环境：

```powershell
conda env create -n offline-polar-loopback -f environment.yml
```

安装后激活环境：

```powershell
conda activate offline-polar-loopback
```

如果目标机器的 CUDA/PyTorch/PyG wheel 源不可用，先完成 Conda 环境创建，再按目标 CUDA 版本补装 `torch`、`torch-geometric` 和相关 PyG wheel。

## 文件说明

### `phy_params.py`

PHY 层公共参数和基础函数：

- `SPS=2`
- `ROLLOFF=0.35`
- `RRC_NUM_SYM=10`
- `STF_LEN=128`
- `PSS_LEN=64`
- `RS_LEN=64`
- `HEADER_LEN=32`
- `PAYLOAD_LEN=256`
- `PAYLOAD_CRC_LEN=16`
- `GUARD_SYMBOLS=32`
- `FRAME_SYMBOLS=592`
- `FRAME_RRC_SAMPLES=1204`

同时提供 STF/PSS/RS 参考序列生成、RRC 滤波器设计、CRC16 和 bit/byte 转换函数。所有脚本应以这里的 PHY 常量为准。

### `deploy/common.py`

SGNN 部署代码，包含：

- Polar 编码和 BPSK 辅助函数。
- 不依赖 `torch_scatter` 的 `scatter_sum`。
- SGNN 模型结构、BPConv、LSTU 变体。
- `load_model()`：加载 `deploy/checkpoint/polar_GNN_20_iter_0_epoches_13.pt`。
- `build_graph()`：从 `deploy/matrices/pcm.npy` 构造 Tanner graph。

### `deploy/matrices/A.npy`

Polar 冻结位掩码。长度为 256，`1` 表示信息位位置，`0` 表示冻结位位置。

本项目默认直接加载该文件；如果缺失，采集、分析和译码脚本会在启动时报错。

### `deploy/matrices/pcm.npy`

SGNN 图译码使用的 parity-check matrix。

### `tools/loopback_capture.py`

硬件采集入口。该脚本生成随机信息位，执行 Polar 编码和 BPSK 成帧，通过 USRP 发射，同时接收并保存 IQ。

输出文件以前缀 `prefix` 命名：

```text
prefix_iq.npy      接收端原始 complex64 IQ
prefix_bits.npy    发送端 Polar codeword，每帧 256 bit
prefix_info.npy    发送端信息位，每帧 128 bit
prefix_meta.json   采集参数和帧结构元数据
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--serial` | `320F33F` | USRP 序列号 |
| `--rx-channel` | `1` | RX 通道 |
| `--rx-antenna` | `RX2` | RX 天线端口 |
| `--freq` | `915e6` | 中心频率 |
| `--gain-tx` | `60` | TX 增益 |
| `--gain-rx` | `30` | RX 增益 |
| `--num-frames` | `100` | 唯一帧数，每帧重复 5 次 |
| `--frame-gap-ms` | `30.0` | 不同 frame_id 组间隔 |
| `-o/--output` | `capture/test` | 输出前缀 |

示例：

```powershell
python tools\loopback_capture.py --serial 320F33F --freq 915e6 --gain-tx 60 --gain-rx 30 --rx-channel 1 --rx-antenna RX2 --num-frames 20 -o capture\smoke\snr_gain030_r0
```

### `tools/batch_capture.py`

批量采集入口。它按 RX gain 列表调用 `loopback_capture.py`，并默认调用 `analyze_repeat_capture.py` 做同步和信道诊断。

默认配置：

```text
serial=320F33F
freq=915e6
gain_tx=60
rx_channel=1
rx_antenna=RX2
num_frames=200
frame_gap_ms=30
gains=[21,23,25,27,30,40]
runs=1
```

示例：

```powershell
python tools\batch_capture.py --gains 21 25 --runs 1 --num-frames 5 --dry-run
python tools\batch_capture.py --serial 320F33F --freq 915e6 --gain-tx 60 --rx-channel 1 --rx-antenna RX2 --gains 21 23 25 27 30 40 --runs 3 --num-frames 100 --outdir capture\ebn0_tx60
```

### `tools/analyze_repeat_capture.py`

离线同步和信道诊断入口。它读取 capture 目录中的 `*_iq.npy`，重建发送端参考帧，定位第一帧，推断 repeat 组时序，并统计每组 5 个 repeat 的检出情况。

它还包含全帧 oracle 信道诊断：使用已知发送波形做全帧相关、线性相位拟合和 LS 信道估计，输出 CFO、`|h|`、相位、噪声底、SNR 和 Eb/N0。

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input_dir` | 必填 | capture 目录 |
| `--gain` | `0` | 只分析指定 RX gain，0 表示全部 |
| `--num-frames` | `40` | 分析的唯一帧数 |
| `--pss-ptm` | `2.5` | PSS peak-to-mean 门限 |
| `--pss-pts` | `1.0` | PSS peak-to-second 门限 |
| `--fixed-nf` | `None` | 固定底噪 dB，默认从 IQ 前缀测量 |
| `--plot` | 关闭 | 显示原始 I/Q 波形 |
| `--save-plot` | 空 | 保存 I/Q 波形图 |
| `--plot-samples` | `5000` | 绘图采样点数 |
| `-o/--output` | 空 | 输出 JSON |

示例：

```powershell
python tools\analyze_repeat_capture.py capture\ebn0_tx60 --gain 21 --num-frames 40
python tools\analyze_repeat_capture.py capture\ebn0_tx60 --gain 21 --num-frames 40 -o capture\ebn0_tx60\analyze_gain021.json
python tools\analyze_repeat_capture.py capture\ebn0_tx60 --gain 21 --save-plot plots --plot-samples 5000
```

### `tools/polar_offline_decode.py`

离线译码入口。它读取 capture 前缀或目录，定位帧，使用 RS 估计信道，生成 Payload LLR，并评估：

- `BPSK_BER`：256 bit codeword 硬判 BER。
- `Polar_BER`：Hard inverse Polar 信息位 BER。
- `SGNN_BER`：SGNN 输出后的信息位 BER。
- `BPSK_ora` / `SGNN_ora`：启用 `--oracle-llr` 时的全帧 oracle LLR 参考。

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `inputs` | 必填 | capture 前缀或目录 |
| `--gain` | `0` | 目录输入时只分析指定 gain |
| `--num-frames` | `100` | 译码帧数 |
| `--sgnn` | 默认启用 | 启用 SGNN |
| `--no-sgnn` | 关闭 | 禁用 SGNN |
| `--oracle-llr` | 关闭 | 启用全帧 oracle LLR |
| `--device` | `cpu` | `cpu` 或 `cuda` |
| `-o/--output` | 空 | 输出 JSON |

示例：

```powershell
python tools\polar_offline_decode.py capture\ebn0_tx60 --gain 21 --num-frames 40 --device cpu
python tools\polar_offline_decode.py capture\ebn0_tx60 --gain 21 --num-frames 40 --no-sgnn
python tools\polar_offline_decode.py capture\ebn0_tx60 --gain 21 --num-frames 40 --device cpu --oracle-llr
```

### `tools/live_spectrum.py`

实时频谱观察入口。它使用子进程通过 UHD 接收，主进程通过共享内存读取并绘制时域 I 和频谱瀑布图。

示例：

```powershell
python tools\live_spectrum.py --serial 320F33F --freq 915e6 --gain 40 --rate 1e6 --nfft 2048
```

### `burst_interferer.py`

突发干扰发射入口。它发送低幅度背景 CW，并按 Bernoulli 过程叠加复高斯突发噪声。

```text
rho_i ~ Bernoulli(p_b)
rho_i = 0: low-amplitude background tone
rho_i = 1: complex Gaussian burst noise
```

示例：

```powershell
python burst_interferer.py --serial 320F2BD --freq 915e6 --gain 20 --rate 1e6 --sps 2 --p-b 0.05 --sigma-b 0.002 --sigma-bg 0.00001 --duration 5
```

## 帧结构

```text
+---------+---------+--------+------------+--------------+----------+--------+
| STF     | PSS     | RS     | Header     | Payload      | CRC      | Guard  |
| 128 sym | 64 sym  | 64 sym | 32 sym     | 256 sym      | 16 sym   | 32 sym |
+---------+---------+--------+------------+--------------+----------+--------+
```

字段说明：

| 字段 | 符号数 | 内容 | 作用 |
| --- | ---: | --- | --- |
| STF | 128 | 8 组重复 BPSK，每组 16 符号 | 粗同步和全帧相关辅助 |
| PSS | 64 | Zadoff-Chu 序列 | 精定时和相关峰确认 |
| RS | 64 | 固定 BPSK 导频 | CFO、相位、信道和噪声估计 |
| Header | 32 | frame_id 16 bit + header CRC16 16 bit | 帧编号和头部校验 |
| Payload | 256 | Polar 编码后的 BPSK 码字 | 数据承载 |
| CRC | 16 | Payload CRC16 | Payload 校验 |
| Guard | 32 | 零符号 | 帧尾保护和噪声观察窗口 |

RRC 成形后的单帧 IQ 长度：

```text
FRAME_RRC_SAMPLES = FRAME_SYMBOLS * SPS + len(RRC) - 1
                  = 592 * 2 + 21 - 1
                  = 1204 samples
```

## 同步链

```text
RX IQ
  |
  +-> 全帧互相关定位 frame0
  |     TX reference = RandomState(42) -> Polar codeword -> frame -> RRC
  |
  +-> 回溯 repeat0
  |     repeat_stride = frame_iq_len + gap_repeat_iq
  |
  +-> 推断每个 frame group 的 5 个 repeat 位置
  |
  +-> 对每个 repeat 做 RRC 匹配滤波和符号抽样
  |
  +-> PSS 相关精定时
  |     ptm = peak / mean(corr)
  |     pts = peak / second_peak(corr)
  |
  +-> RS 信道估计
  |     rs_tone = rs_seg * conj(rs_ref)
  |     phase slope -> CFO
  |     h = mean(rs_corrected * conj(rs_ref))
  |     sigma2 = residual noise power
  |
  `-> Payload LLR
        LLR = clip(4 * real(y_equalized) / sigma2, -20, 20)
```

组检出规则：

```text
每个 frame_id 重复 5 次。
任意一个 repeat 通过 PSS/RS 同步，则该 frame group 记为检出成功。
```

## 全帧 oracle 诊断

`analyze_repeat_capture.py` 内部会用已知 TX 波形做全帧 oracle 诊断：

```text
known TX frame + RX frame
  |-- full-frame correlation
  |-- full-frame phase fit -> CFO
  |-- full-frame LS channel -> |h|, phase
  `-- prefix noise floor -> SNR / Eb/N0
```

这个功能用于标定和排错，不是实时接收机算法。它可以判断：

- 低 SNR 下物理信道是否仍可测。
- CFO 是否稳定。
- RS 估计失败是否由同步门限、导频能量或噪声底导致。
- 仿真中应该使用怎样的 CFO、`|h|`、noise floor。

`polar_offline_decode.py --oracle-llr` 进一步使用同类全帧已知信道估计生成理想 LLR，用于判断 SGNN/Polar 译码器在理想信道信息下的上限表现。

## 常用命令

参数检查：

```powershell
python tools\loopback_capture.py --help
python tools\batch_capture.py --help
python tools\analyze_repeat_capture.py --help
python tools\polar_offline_decode.py --help
python tools\live_spectrum.py --help
python burst_interferer.py --help
```

语法检查：

```powershell
python -c "import ast, pathlib; files=['burst_interferer.py','phy_params.py','tools/analyze_repeat_capture.py','tools/batch_capture.py','tools/polar_offline_decode.py','tools/loopback_capture.py','tools/live_spectrum.py','deploy/common.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8')) for f in files]; print('syntax OK')"
```

采集：

```powershell
python tools\loopback_capture.py --serial 320F33F --freq 915e6 --gain-tx 60 --gain-rx 30 --rx-channel 1 --rx-antenna RX2 --num-frames 20 -o capture\smoke\snr_gain030_r0
```

批量扫增益：

```powershell
python tools\batch_capture.py --serial 320F33F --freq 915e6 --gain-tx 60 --rx-channel 1 --rx-antenna RX2 --gains 21 23 25 27 30 40 --runs 3 --num-frames 100 --outdir capture\ebn0_tx60
```

同步和信道诊断：

```powershell
python tools\analyze_repeat_capture.py capture\ebn0_tx60 --gain 21 --num-frames 40 -o capture\ebn0_tx60\analyze_gain021.json
```

保存原始 I/Q 波形图：

```powershell
python tools\analyze_repeat_capture.py capture\ebn0_tx60 --gain 21 --save-plot plots --plot-samples 5000
```

离线译码：

```powershell
python tools\polar_offline_decode.py capture\ebn0_tx60 --gain 21 --num-frames 40 --device cpu
python tools\polar_offline_decode.py capture\ebn0_tx60 --gain 21 --num-frames 40 --no-sgnn
python tools\polar_offline_decode.py capture\ebn0_tx60 --gain 21 --num-frames 40 --device cpu --oracle-llr
```

实时频谱：

```powershell
python tools\live_spectrum.py --serial 320F33F --freq 915e6 --gain 40 --rate 1e6 --nfft 2048
```

突发干扰：

```powershell
python burst_interferer.py --serial 320F2BD --freq 915e6 --gain 20 --rate 1e6 --sps 2 --p-b 0.05 --sigma-b 0.002 --sigma-bg 0.00001 --duration 5
```

## 输出文件

单次 capture 前缀为 `capture/xxx/snr_gain021_r0` 时，常见输出为：

```text
snr_gain021_r0_iq.npy      RX IQ
snr_gain021_r0_bits.npy    TX Polar codeword bits
snr_gain021_r0_info.npy    TX information bits
snr_gain021_r0_meta.json   capture metadata
snr_gain021_r0_stats.json  analyze_repeat_capture report
```

`batch_capture.py` 会生成 `summary.json`。

## 工程约定

1. 帧结构以 `phy_params.py` 和 `tools/loopback_capture.py` 为准。
2. `deploy/matrices/A.npy` 是必需运行资产，冻结位掩码从该文件加载。
3. `capture/` 是实验输出，不提交。
4. 信道测量由 `analyze_repeat_capture.py` 的全帧 oracle 诊断承担。
5. 离线译码的理想信道参考由 `polar_offline_decode.py --oracle-llr` 承担。
6. 如果 oracle 指标好而普通译码差，优先检查 PSS/RS 同步和信道估计；如果 oracle LLR 下 SGNN 仍差，优先检查 checkpoint、`pcm.npy`、`A.npy` 和训练配置一致性。
