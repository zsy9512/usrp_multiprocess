# Burst Interferer — B210 UHD 突发干扰源

## 环境依赖

```bash
# Ubuntu 22.04 / 24.04
sudo apt install python3-numpy
sudo apt install uhd-host libuhd-dev
pip install uhd   # Python UHD bindings
```

## 用法

```bash
python3 burst_interferer.py --serial <USRP序列号> --sigma-b 2.0
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--serial` | `320F2BD` | USRP 序列号 |
| `--freq` | `915e6` | 中心频率 (Hz) |
| `--gain` | `50` | TX 增益 (dB) |
| `--rate` | `1e6` | 采样率 (Hz) |
| `--p-b` | `0.05` | 突发概率 (每个符号) |
| `--sigma-b` | `2.0` | 突发噪声标准差 |
| `--sigma-bg` | `0.001` | 背景平稳 CW 幅度 |
| `--duration` | `0` | 总时长 (s), 0=无限 |

## 信道模型

```
y_i = s_i + n_i + ρ_i · ω_i

  ρ_i ~ Bernoulli(p_b)       突发指示 (5% 概率)
  ω_i ~ N(0, σ_b²)          突发干扰
  非突发: 极低幅度 CW 平稳信号
```
