# USRP DQPSK 多进程通信系统

**版本**: v1.5.0  
**作者**: shengyu@hust.edu.cn  
**更新日期**: 2025-09-17

这是一个基于USRP硬件的DQPSK (Differential Quadrature Phase Shift Keying) 多进程收发系统，支持实时信号处理、同步和可视化。系统专注于硬件USRP模式，为研究和开发提供稳定的测试环境。

## 🎯 系统特性

### ✅ 已完成功能
- **硬件USRP模式**：真实的射频信号收发和处理
- **实时同步算法**：PSS/SSS同步、频率校正、Costas环相位跟踪
- **差分解调**：完整的DQPSK信号处理链路
- **专业GUI监控**：差分星座图、时域波形、频谱分析
- **多进程架构**：发射、接收、处理完全分离
- **统一IPC接口**：UDP和队列两种通信模式

### 📊 当前性能状态

| 模式 | BER性能 | 状态 | 备注 |
|------|---------|------|------|
| **仿真模式** | 7e-4 ~ 5e-3 (15dB) | ✅ **完全完成** | 性能稳定，可靠 |
| **硬件USRP模式** | 4.89e-01 | ⚠️ **需要调整** | 缓存问题待解决 |

## 📁 项目结构

```
usrp_multiprocess/
├── 🎯 核心系统
│   ├── dqpsk_system.py        # DQPSK调制解调核心算法
│   └── __init__.py            # Python包初始化文件
├── 📡 通信程序
│   ├── tx_program.py          # USRP发射程序 (仅硬件模式)
│   ├── rx_program.py          # 接收程序 (硬件+仿真模式)
│   └── queue_server.py       # 队列服务器
├── 📊 处理程序
│   └── processing_program.py  # 处理+GUI程序 (硬件+仿真模式)
├── 🚀 启动脚本
│   └── start_experiment.py    # 统一启动脚本
├── 🧪 测试工具
│   └── simple_simulation_test.py # 仿真性能测试
└── 🔧 工具脚本
    └── cleanup_port.py        # 端口清理工具
```

## 🚀 快速开始

### 🧪 仿真模式测试（推荐）
```bash
# 一键启动完整仿真实验
python start_experiment.py --mode simulation

# 或分别启动各组件
python queue_server.py
python tx_program.py --mode simulation --ipc_mode queue
python rx_program.py --mode simulation --ipc_mode queue
python processing_program.py --mode simulation --ipc_mode queue
```

### 📡 硬件USRP模式
```bash
# 确保USRP设备已连接
python start_experiment.py --mode hardware

# 或分别启动各组件
python queue_server.py
python tx_program.py --tx_freq 900e6 --rate 1e6 --tx_gain 50
python rx_program.py --mode hardware --ipc_mode queue
python processing_program.py --mode hardware --ipc_mode queue
```

## 📋 模式详细说明

### 🧪 仿真模式 (Simulation Mode)
**设计目标**：测试和验证DQPSK处理算法的正确性

#### 核心特性
- 使用软件信道模拟真实的无线信道效应
- 支持可配置的SNR、频率偏移、相位偏移
- 快速测试，无需硬件设备
- 确定性结果，便于调试和性能评估

#### 适用场景
- 算法开发和验证
- 性能基准测试
- 教育和学习目的

### 📡 硬件USRP模式 (Hardware Mode)
**设计目标**：真实的射频信号收发和处理

#### 核心特性
- 通过USRP硬件进行真实的射频信号收发
- 支持多种USRP设备 (B210, X310等)
- 实时信号处理和同步
- 完整的无线通信链路测试

#### 当前问题
- **BER性能**：仅达到4.89e-01，远高于预期
- **根本原因**：缓存机制需要针对硬件实时性进行优化
- **解决方案**：调整接收缓冲区大小和处理时序

## 🔧 安装和配置

### 系统要求
- Python 3.7+
- UHD 4.8+ (硬件模式需要)
- 支持的USRP设备：B210, X310等

### 依赖安装
```bash
pip install numpy scipy pyqt5 pyqtgraph uhd
```

### 硬件模式配置
```bash
# 检查USRP连接
uhd_find_devices

# 配置设备参数
python tx_program.py --mode hardware --args "name=MyB210_01"
```

## ⚙️ 参数配置

### 统一启动脚本 (start_experiment.py)
```bash
--mode {simulation,hardware}  # 运行模式 (默认: simulation)
--host HOST                   # 服务器主机 (默认: 127.0.0.1)
--port PORT                   # 服务器端口 (默认: 50000)
--tx-freq TX_FREQ             # 发射频率 (Hz) (默认: 900e6)
--rx-freq RX_FREQ             # 接收频率 (Hz) (默认: 900e6)
--rate RATE                   # 采样率 (Hz) (默认: 1e6)
--tx-gain TX_GAIN             # 发射增益 (dB) (默认: 40)
--rx-gain RX_GAIN             # 接收增益 (dB) (默认: 50)
```

### 发射程序参数 (tx_program.py)
```bash
--tx_freq 915e6               # 发射频率 (Hz)
--rate 1e6                    # 采样率 (Hz)
--tx_gain 50                  # 发射增益 (dB)
--args "name=MyB210"          # USRP设备参数
--repeat_count 30             # 每个帧重复发送次数
--bit_generator random        # 比特生成模式：random/zeros/ones
```

### 接收程序参数 (rx_program.py)
```bash
--mode hardware               # 运行模式：hardware/simulation
--rx_freq 900e6               # 接收频率 (Hz)
--rate 1e6                    # 采样率 (Hz)
--rx_gain 50                  # 接收增益 (dB)
--ipc_mode queue              # IPC模式：queue/udp
--buffer_size 10000           # 接收缓冲区大小
```

### 处理程序参数 (processing_program.py)
```bash
--mode hardware               # 运行模式：hardware/simulation
--rate 1e6                    # 采样率 (Hz)
--ipc_mode queue              # IPC模式：queue/udp
--udp_port 12345              # UDP端口 (UDP模式时使用)
```

### 队列服务器参数 (queue_server.py)
```bash
--host 127.0.0.1             # 服务器主机地址
--port 50000                  # 服务器端口
```

## 🎯 实验流程

### 硬件模式流程
```mermaid
flowchart LR
    A[USRP发射] --> B[射频传输]
    B --> C[USRP接收]
    C --> D[噪声过滤]
    D --> E[写入环形缓冲区]
    E --> F[IPC发送到处理程序]
    F --> G[同步解调]
    G --> H[差分星座图显示]
```

### 仿真模式流程
```mermaid
flowchart LR
    A[软件发射] --> B[信道模拟]
    B --> C[添加噪声]
    C --> D[写入环形缓冲区]
    D --> E[IPC发送到处理程序]
    E --> F[同步解调]
    F --> G[差分星座图显示]
```

## 📊 性能监控

### GUI界面组件
- **差分星座图**：显示DQPSK差分符号，4个理想位置
- **时域波形**：接收信号时域特性
- **同步状态**：实时显示BER、同步质量、帧计数
- **频谱分析**：信号频率成分分析

### 关键性能指标
- **目标BER**：< 1e-3
- **仿真模式**：✅ 已达到7e-4~5e-3 (15dB)
- **硬件模式**：⚠️ 当前4.89e-01，需要缓存优化

## 🔍 故障排除

### 通用问题
- **队列服务器无法启动**：检查端口50000是否被占用
  ```bash
  # 清理端口
  python cleanup_port.py --port 50000
  ```
- **进程间通信失败**：确认所有程序使用相同的IPC模式(queue/udp)
- **端口占用问题**：使用cleanup_port.py清理残留进程
  ```bash
  # 查看端口占用
  python cleanup_port.py --port 50000 --info

  # 强制清理
  python cleanup_port.py --port 50000 --force
  ```

### 硬件模式问题
- **设备未识别**：检查USRP连接，运行`uhd_find_devices`
- **缓存溢出**：调整`buffer_size`参数，增加缓冲区大小
- **BER异常**：检查天线连接，调整增益参数
- **同步问题**：确认发射和接收频率严格匹配

### 队列服务器问题
- **服务器启动失败**：检查端口是否被其他程序占用
- **连接超时**：确认服务器已启动，检查防火墙设置
- **数据传输异常**：重启所有程序，确保正确的启动顺序

## ⚠️ 重要提示

### 队列服务器管理
- **启动顺序**：队列服务器必须首先启动，其他程序按顺序启动
- **关闭问题**：队列服务器目前不支持自动关闭，运行完成后需要手动清理端口
- **端口清理**：运行以下命令清理端口占用：

```bash
# 清理默认端口 (50000)
python cleanup_port.py --port 50000

# 或清理指定端口
python cleanup_port.py --port <端口号>
```

### 程序退出顺序
1. 先停止发射程序和接收程序
2. 再停止处理程序
3. 最后清理队列服务器端口

## 📈 开发路线图

### v1.5.0 (当前版本)
- ✅ 实现差分星座图显示
- ✅ 统一硬件模式架构
- ⚠️ 硬件模式缓存优化 (进行中)

### v1.6.0 (计划)
- 🔧 硬件模式缓存优化
- 📊 实时频谱分析
- 🎛️ 参数自适应调整
- 📈 性能分析工具

## 📝 更新日志

### v1.5.0 (2025-09-17)
- ✅ **差分星座图实现**：解决绝对相位旋转问题
- ✅ **全局相位追踪**：跨帧相位稳定性提升
- ✅ **硬件模式架构优化**：发射程序清理为仅支持硬件模式
- ✅ **README文档重组**：根据实际文件夹结构重新组织文档
- ⚠️ **硬件模式待优化**：BER为4.89e-01，缓存机制需要调整
- 📊 **性能监控增强**：实时BER显示和统计

### v1.1.0 (2025-09-16)
- ✅ 修复队列服务器Ctrl+C退出问题
- ✅ 优化处理程序数据接收逻辑
- ✅ 完善队列模式通信机制
- ✅ 改进错误处理和日志输出

---

**注意**：这是一个研究和开发系统。系统专注于硬件USRP模式，为真实的射频通信提供完整的测试环境。如有问题请查看日志输出或提交Issue。