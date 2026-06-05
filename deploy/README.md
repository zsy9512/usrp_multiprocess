# Polar-SGNN USRP 部署套件

独立可移植的极化码 USRP 部署程序，包含发送端、接收端和干扰器。

## 文件结构

```
deploy/
├── common.py          # 自包含的公共模块（极化编解码、SGNN模型、序列化）
├── sender.py          # 发送端：随机生成信息比特 → 编码 → BPSK → UDP
├── receiver.py        # 接收端：UDP收采样 → LLR → SGNN译码
├── interferer.py      # 干扰器：按概率生成突发噪声 → UDP
├── matrices/          # 奇偶校验矩阵 pcm.npy + 冻结比特掩码 A.npy
├── checkpoint/        # 放置训练好的 .pt 模型文件
└── README.md
```

## 依赖

- Python ≥ 3.9
- numpy, torch (PyTorch ≥ 2.0)
- **无需 torch_scatter**（已用纯 PyTorch scatter_add 替代）

## 快速开始

### 1. 放置模型和矩阵

```bash
# 矩阵文件已包含在 deploy/matrices/ 中
# 将训练好的 checkpoint 复制到 deploy/checkpoint/：
cp models/20_256_iter_0_ssmb100_20260518_160559/polar_GNN_20_iter_0_epoches_13.pt deploy/checkpoint/
```

### 2. 发送端

```bash
python sender.py --tx-ip 192.168.10.2 --tx-port 5000
```

每帧发送 256 个 BPSK 符号（2048 字节 float32 I/Q），帧间间隔可通过 `--interval` 调整。

### 3. 接收端

```bash
python receiver.py --rx-port 5001 --checkpoint checkpoint/polar_GNN_20_iter_0_epoches_13.pt --sigma 0.5
```

- `--sigma`：噪声标准差，用于 LLR 计算。设为 0 则从接收信号自动估计。
- `--device cuda`：使用 GPU 推理。

### 4. 干扰器（独立 USRP）

```bash
# 符号级突发干扰（匹配训练时的 burst 模型）
python interferer.py --tx-ip 192.168.10.3 --tx-port 5002 \
    --sigma-b 2.0 --burst-prob 0.1 --mode burst

# 整帧连续高斯噪声
python interferer.py --tx-ip 192.168.10.3 --tx-port 5002 \
    --sigma-b 1.0 --mode continuous
```

## 系统架构

```
┌──────────┐    UDP (I/Q)     ┌──────────┐     RF      ┌──────────┐
│ sender   │ ────────────────→│ USRP TX  │ ─ ─ ─ ─ ─ →│ USRP RX  │
│ (编码)   │    port 5000     └──────────┘   信道     └──────────┘
└──────────┘                                               │
                                              UDP (I/Q)   │
                                                          ↓
┌──────────┐    UDP (I/Q)     ┌──────────┐          ┌──────────┐
│interferer│ ────────────────→│ USRP TX  │ ─ ─ ─ ─ →│ receiver │
│ (噪声)   │    port 5002     │ (干扰)   │   干扰   │ (SGNN)   │
└──────────┘                  └──────────┘          └──────────┘
```

三台 USRP 工作在同一频率上：
- USRP-A：发送编码后的 BPSK 信号
- USRP-B：按概率发送突发高斯噪声（干扰）
- USRP-C：接收叠加信号，传给 receiver.py 进行 SGNN 译码

### 数据格式

- **调制方式**：BPSK（0→+1，1→-1），仅实部承载信息
- **UDP 封装**：float32 交错 I/Q，每帧 256 符号 × 2 × 4 = 2048 字节
- **码参数**：N=256，K=128（码率 1/2），使用 Arikan 极化核

### 参数说明

| 程序 | 关键参数 | 说明 |
|------|---------|------|
| sender | `--tx-ip`, `--tx-port` | USRP TX 地址 |
| | `--interval` | 帧间间隔（秒），0=最快 |
| | `--frame-count` | 发送帧数，0=无限 |
| receiver | `--rx-ip`, `--rx-port` | 监听地址/端口 |
| | `--checkpoint` | 模型权重路径 |
| | `--sigma` | 噪声标准差（LLR 计算用） |
| | `--device` | cpu / cuda |
| interferer | `--sigma-b` | 突发噪声标准差 |
| | `--burst-prob` | 每符号突发概率 |
| | `--mode` | burst（符号级概率）/ continuous（整帧噪声） |

## 移植到其他机器

整个 `deploy/` 目录是自包含的，复制到任意机器后只需：

```bash
pip install numpy torch
python sender.py --tx-ip <USRP_IP> --tx-port <PORT>
python receiver.py --rx-port <PORT> --checkpoint <MODEL.pt>
python interferer.py --tx-ip <USRP_IP> --tx-port <PORT> --sigma-b 2.0
```
