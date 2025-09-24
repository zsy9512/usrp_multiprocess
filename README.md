# USRP DQPSK 多进程通信系统

**版本**: v1.8.0  
**作者**: shengyu@hust.edu.cn  
**更新日期**: 2025-09-24

本系统基于USRP硬件，支持DQPSK调制解调的多进程收发、同步、实时处理和可视化。自v1.8.0起，硬件时钟源同步方案已完善，单收单发模式也能稳定运行，误码率可达5e-3以下。系统架构灵活，支持仿真、硬件、自收自发三种模式，适合科研、教学和原型开发。

---

## 🎯 系统功能与特色

### ✅ 核心功能
- **硬件USRP模式**：真实射频信号收发，支持B210/X310等主流设备
- **单收单发模式**：时钟源同步后已稳定跑通，BER可达5e-3以下
- **自收自发模式**：单进程多线程，TX/RX/IPC高效协同
- **仿真模式**：软件信道模拟，算法开发与性能验证
- **多进程架构**：发射、接收、处理完全分离，支持队列/UDP通信
- **队列服务器**：自动连接与管理，支持多客户端
- **同步算法**：PSS/SSS/RS序列同步，Costas环相位跟踪
- **差分DQPSK解调**：完整物理层处理链路
- **专业GUI**：星座图、时域波形、频谱分析、实时BER统计
- **数据分析工具**：离线分析、性能评估
- **参数自适应**：支持多种采样率、增益、帧结构配置

### 🌟 系统特色
- **高性能同步**：多级同步算法（PSS定时、SSS粗频、RS细频），确保低SNR下稳定同步
- **实时处理**：滑动窗口多帧检测，连续帧处理无间断
- **差分调制**：消除绝对相位模糊，抗频率偏移能力强
- **多进程通信**：基于multiprocessing.Queue的共享内存通信，高效可靠
- **专业可视化**：PyQt5实时GUI，星座图流动显示，同步质量监控
- **模块化设计**：核心算法、通信、处理完全解耦，便于扩展和维护
- **跨平台兼容**：支持Windows/Linux，USRP硬件抽象层统一接口

### 📊 当前性能状态

| 模式             | BER性能           | 状态           | 备注 |
|------------------|------------------|----------------|------|
| **仿真模式**     | 7e-4 ~ 5e-3 (15dB)| ✅ 完全完成     | 性能稳定 |
| **硬件USRP模式** | <5e-3            | ✅ 已跑通       | 时钟源同步后单收单发稳定 |
| **自收自发模式** | 7e-4 ~ 5e-3      | ✅ 完全完成     | 性能稳定 |

## 📁 项目结构

```
usrp_multiprocess/
├── 🎯 核心算法
│   ├── dqpsk_system.py        # DQPSK调制解调核心算法
│   │   ├── USRP_DQPSK_System类：系统主类
│   │   ├── 帧结构：PSS(32)+SSS(32)+RS(64)+数据符号
│   │   ├── 同步算法：PSS定时、SSS粗频、RS细频
│   │   ├── Costas环：相位跟踪和同步
│   │   └── 差分编码/解码：4点星座图DQPSK
│   └── __init__.py            # Python包初始化文件
├── 📡 通信程序
│   ├── tx_program.py          # USRP发射程序 (仅硬件模式)
│   │   ├── 双数组机制：数据生成与发送线程解耦
│   │   ├── USRP硬件接口：发射流控制
│   │   └── 帧重复发送：支持可配置重复次数
│   ├── rx_program.py          # 接收程序 (硬件+仿真模式)
│   │   ├── 环形缓冲区：高效数据缓存
│   │   ├── IPC发送线程：多进程数据传递
│   │   └── 仿真信道：SNR/频偏/相偏模拟
│   ├── transceiver_program.py # 自收自发程序 (硬件模式)
│   │   ├── 多线程架构：TX/RX/IPC独立线程
│   │   ├── 共享USRP：TX/RX同一设备，天然同步
│   │   └── 队列通信：与处理程序无缝对接
│   └── queue_server.py       # 队列服务器
│       ├── BaseManager：多进程队列管理
│       ├── 端口检查：自动检测和清理占用
│       └── 信号处理：优雅关闭和资源清理
├── 📊 处理程序
│   └── processing_program.py  # 处理+GUI程序 (硬件+仿真模式)
│       ├── 滑动窗口检测：连续帧同步和提取
│       ├── PyQt5 GUI：专业实时可视化界面
│       ├── BER计算：实时误码率统计
│       └── 多进程通信：队列/UDP双模式支持
├── 🚀 启动脚本
│   └── start_experiment.py    # 统一启动脚本
│       ├── 一键启动：仿真/硬件/自收自发模式
│       ├── 进程管理：自动启动和监控所有组件
│       └── 优雅关闭：信号处理和资源清理
├── 🧪 测试工具
│   ├── simple_simulation_test.py # 仿真性能测试
│   └── analyze_rxdata.py      # 离线数据分析工具
├── 🔧 工具脚本
│   ├── cleanup_port.py        # 端口清理工具
│   └── usrp_scope.py          # USRP示波器/频谱仪
└── 📦 数据文件
    ├── rxdata_sim.bin         # 仿真接收数据
    ├── rxdata04.bin           # 硬件接收数据
    └── __pycache__/           # Python缓存文件
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

# 自收自发模式 (推荐用于自环测试)
python start_experiment.py --mode transceiver --tx-freq 915e6 --rx-freq 915e6 --tx-gain 50 --rx-gain 40
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

- 快速原型开发

### 📻 自收自发模式 (Transceiver Mode)
**设计目标**：简化测试流程，合并发射和接收功能

#### 核心特性
- 单程序同时处理发射和接收
- 多线程架构：TX线程、RX线程、IPC发送线程
- 环形缓冲区高效数据传递
- 自动同步发射和接收参数

#### 适用场景
- 自环测试和硬件验证
- 简化实验设置
- 快速原型开发

### 📡 硬件USRP模式 (Hardware Mode)
**设计目标**：真实的射频信号收发和处理

#### 核心特性
- 通过USRP硬件进行真实的射频信号收发
- 支持多种USRP设备 (B210, X310等)
- 实时信号处理和同步
- 完整的无线通信链路测试

#### 当前问题
- **时钟源同步**：v1.8.0已通过外部时钟源解决
- **BER性能**：稳定达到5e-3以下，满足预期

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
--mode {simulation,hardware,transceiver}  # 运行模式 (默认: simulation, transceiver为自收自发模式)
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

### 自收自发程序参数 (transceiver_program.py)
```bash
--tx_freq 915e6               # 发射频率 (Hz)
--rx_freq 915e6               # 接收频率 (Hz)
--rate 1e6                    # 采样率 (Hz)
--tx_gain 50                  # 发射增益 (dB)
--rx_gain 40                  # 接收增益 (dB)
--args "name=MyB210"          # USRP设备参数
--buffer_size 50000           # 接收缓冲区大小
--repeat_count 10             # 每个帧重复发送次数
--bit_generator random        # 比特生成模式：random/zeros/ones
--sps 2                       # 每符号采样点数
--roll_off 0.35               # 滚降系数
--record_file rxdata.bin      # 可选：保存接收原始数据的二进制文件
--queue_host 127.0.0.1        # 队列服务器主机地址
--queue_port 50000            # 队列服务器端口
--queue_authkey queue_key     # 队列服务器认证密钥
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

### 自收自发模式流程
```mermaid
flowchart LR
    A[生成DQPSK帧] --> B[USRP发射]
    B --> C[射频自环]
    C --> D[USRP接收]
    D --> E[写入环形缓冲区]
    E --> F[IPC发送到处理程序]
    F --> G[同步解调]
    G --> H[差分星座图显示]
```

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
- **硬件模式**：✅ 已达到5e-3以下（时钟源同步后）
- **自收自发模式**：✅ 已达到7e-4~5e-3

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

### 程序退出顺序
1. 先停止发射程序和接收程序
2. 再停止处理程序
3. 最后清理队列服务器端口

## 📈 开发路线图

### v1.8.0 (当前版本)
- ✅ **时钟源同步优化**：硬件模式单收单发已稳定跑通，BER<5e-3
- ✅ **系统功能完善**：全面支持仿真、硬件、自收自发三种模式
- ✅ **参数自适应**：支持多种采样率、增益、帧结构配置
- ✅ **性能分析工具**：实时频谱分析、BER统计
- 🔧 **队列服务器BUG修复**：计划修复无法正常关闭问题
- 📊 **结果合并优化**：计划合并同一帧的结果，返回最终正确解调BER
- 🤖 **深度学习接口扩展**：计划支持DL模型集成与调用

## 📝 更新日志

### v1.8.0 (2025-09-24)
- ✅ **硬件模式时钟源同步完善**，单收单发已稳定跑通，BER<5e-3
- ✅ **系统功能全面描述**，README文档升级
- 🔧 **队列服务器BUG修复**（计划）
- 📊 **结果合并优化**（计划）：同一帧结果合并，返回最终BER
- 🤖 **深度学习接口扩展**（计划）：支持DL模型集成

### v1.7.0 (2025-09-23)
- ✅ **自收自发模式完成**：transceiver_program.py 性能达到 7e-4 ~ 5e-3
- ✅ **在线同步优化**：processing_program.py 实现滑动窗口多帧检测
- ✅ **缓冲区改进**：提升最小处理样本数至8000，增加重叠至2000
- ⚠️ **硬件USRP模式**：时钟同步问题待解决，需要外部时钟源

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

### v1.0.0 (2025-09-01)
- 🎯 **初始版本**：基本的USRP通信框架

---

**注意**：本系统为研究与开发用途，专注于硬件USRP模式和高性能DQPSK通信。欢迎反馈问题或提交Issue。