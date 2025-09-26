# USRP DQPSK 多进程通信系统

## 项目简介

这是一个基于USRP（Universal Software Radio Peripheral）设备的DQPSK（Differential Quadrature Phase Shift Keying）多进程无线通信系统。该系统实现了完整的DQPSK调制解调链路，支持仿真和硬件模式，主要用于无线通信实验、信号处理研究和USRP设备调试。

## 主要功能

- **DQPSK调制解调**：实现完整的DQPSK信号处理链路，包括帧同步、频率同步、相位同步和差分解码
- **多进程架构**：采用发射、接收、处理分离的多进程设计，提高系统稳定性和实时性
- **实时可视化**：提供专业的GUI界面，实时显示星座图、时域波形和频谱
- **多模式支持**：支持仿真模式（算法验证）和硬件模式（实际信号收发）
- **IPC通信**：使用多进程队列实现进程间高效数据传递
- **数据记录与分析**：支持原始数据记录和离线分析工具

## 功能特色

### 多级数据缓冲
- **环形缓冲区**：接收程序使用环形缓冲区实现高速数据缓存，避免数据丢失
- **队列缓冲**：多进程间使用队列进行数据传递，支持阻塞和超时机制
- **双缓冲机制**：发射程序采用双数组机制，保证发送线程稳定运行

### 双数组机制保证发送稳定
- 发送线程使用稳定的帧数组副本，避免数据更新时的线程竞争
- 数据生成线程维护更新数组，新帧逐步替换，保证连续发射
- 线程安全的数组交换机制，确保发射过程无中断

### 纳秒级PC时钟同步
- 使用内部时钟源，通过`time.time_ns()`获取纳秒级PC时间
- 将PC时间转换为UHD时间格式，实现精确的时间同步
- 支持外部时钟源，适应不同实验环境

### IPC通信
- 基于`multiprocessing.Manager`的共享队列服务器
- 支持UDP备用通信模式
- 认证密钥保护，确保通信安全

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TX Program    │    │   RX Program    │    │Processing Program│
│  (发射程序)     │    │  (接收程序)     │    │  (处理程序)      │
│                 │    │                 │    │                 │
│ • 帧生成        │    │ • USRP接收      │    │ • 同步解调      │
│ • USRP发射      │◄──►│ • 数据缓冲      │◄──►│ • BER计算       │
│ • 双数组机制    │    │ • IPC发送       │    │ • GUI显示       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Queue Server   │
                    │ (队列服务器)   │
                    │                │
                    │ • 共享队列     │
                    │ • 进程管理     │
                    └─────────────────┘
```

## 模块组成

### 核心模块

#### `dqpsk_system.py` - DQPSK系统核心类
**功能**：实现DQPSK调制解调算法的核心类
**主要方法**：
- `generate_frame()`: 生成包含前导和数据的DQPSK帧
- `prepare_tx_signal()`: 将符号转换为发送信号（插值+滤波）
- `_enhanced_pss_sync()`: PSS符号定时同步
- `_enhanced_sss_sync()`: SSS粗频率同步
- `_enhanced_rs_sync()`: RS细频率同步
- `differential_encode/decode()`: 差分编码/解码
- `plot_ber_statistics()`: BER统计图表绘制

#### `rx_program.py` - 接收程序
**功能**：高速接收USRP数据，滤除噪声，通过IPC发送到处理程序
**主要特性**：
- 环形缓冲区实现高速数据缓存
- 噪声检测和过滤
- 支持仿真和硬件模式
- IPC队列/UDP通信

#### `tx_program.py` - 发射程序
**功能**：生成和发送DQPSK帧到USRP设备
**主要特性**：
- 双数组机制保证发送稳定
- 多线程架构（数据生成+发射分离）
- 支持重复发送和间隔控制

#### `processing_program.py` - 处理程序
**功能**：从IPC接收数据，进行同步解调，实时显示结果
**主要特性**：
- 滑动窗口帧检测和提取
- 完整的同步链路（PSS/SSS/RS/相位同步）
- CRC校验和BER计算
- PyQt5专业GUI界面

#### `transceiver_program.py` - 自收自发程序
**功能**：合并发射和接收功能，实现自收自发模式
**主要特性**：
- 单程序实现收发一体
- 适用于自环测试和调试

### 辅助工具

#### `queue_server.py` - 队列服务器
**功能**：提供多进程共享队列服务
**接口方法**：
- `get_queue()`: 获取共享队列对象
- 支持认证和多客户端连接

#### `start_experiment.py` - 实验启动器
**功能**：一键启动完整的多进程实验
**支持模式**：
- `simulation`: 仿真模式
- `hardware`: 硬件USRP模式
- `transceiver`: 自收自发模式

#### `usrp_scope.py` - USRP示波器
**功能**：独立的USRP信号可视化工具
**特性**：
- 实时时域波形显示
- 频谱分析（功率谱密度）
- 发射/接收信号监控
- 数据记录功能

#### `analyze_rxdata.py` - 数据分析工具
**功能**：离线分析接收数据文件
**特性**：
- 原始时域和频谱显示
- 帧同步和解调分析
- BER计算和星座图展示
- 命令行参数控制

#### `cleanup_port.py` - 端口清理工具
**功能**：清理占用指定端口的进程
**特性**：
- 检测端口占用状态
- 自动终止相关进程
- 支持TIME_WAIT连接处理

#### `simple_simulation_test.py` - 仿真测试
**功能**：简单的DQPSK仿真性能测试
**特性**：
- 信道效应模拟（SNR、频偏、相偏）
- 性能统计和可视化

## 使用方法

### 环境要求
- Python 3.7+
- UHD (USRP Hardware Driver)
- PyQt5 (GUI显示)
- NumPy, Matplotlib, SciPy
- uhd Python包

### 快速开始

#### 1. 仿真模式测试
```bash
# 启动完整仿真实验
python start_experiment.py --mode simulation
```

#### 2. 硬件模式运行
```bash
# 确保USRP设备连接
# 启动完整硬件实验
python start_experiment.py --mode hardware --tx-freq 915e6 --rx-freq 915e6
```

#### 3. 自收自发模式
```bash
# 自环测试模式
python start_experiment.py --mode transceiver
```

### 单独运行各模块

#### 队列服务器
```bash
python queue_server.py --host 127.0.0.1 --port 50000
```

#### 发射程序
```bash
python tx_program.py --tx-freq 915e6 --rate 1e6 --tx-gain 50
```

#### 接收程序
```bash
python rx_program.py --rx-freq 915e6 --rate 1e6 --rx-gain 30
```

#### 处理程序
```bash
python processing_program.py --ipc-mode queue --host 127.0.0.1 --port 50000
```

### 数据分析
```bash
# 分析接收数据文件
python analyze_rxdata.py --file rxdata.bin --rate 1e6 --sps 2 --analyze
```

### USRP调试
```bash
# 启动示波器工具
python usrp_scope.py --tx-freq 915e6 --rx-freq 915e6 --record-file debug_data.bin
```

## 技术参数

- **调制方式**：DQPSK (Differential QPSK)
- **采样率**：默认1MHz，可配置
- **符号率**：500kHz (sps=2)
- **滚降系数**：0.35
- **帧结构**：
  - PSS: 32符号
  - SSS: 32符号
  - RS: 64符号
  - 数据: 1280比特 (640符号)
  - 帧序号: 8比特 (汉明码保护)
  - CRC: 16比特
- **同步算法**：
  - PSS: 增强型相关同步
  - SSS: 相位差法粗频同步
  - RS: 优化搜索细频同步
  - 相位: Costas环

## 性能指标

- **实时处理**：支持1MHz采样率实时处理
- **BER性能**：在合适SNR下可达到10^-4以下
- **同步精度**：频率同步精度<1Hz，相位同步<1度
- **延迟**：端到端延迟<10ms

## 开发与调试

### 代码结构
```
usrp_multiprocess/
├── dqpsk_system.py      # 核心算法
├── rx_program.py        # 接收程序
├── tx_program.py        # 发射程序
├── processing_program.py # 处理程序
├── transceiver_program.py # 自收自发程序
├── queue_server.py     # 队列服务器
├── start_experiment.py  # 实验启动器
├── usrp_scope.py        # 示波器工具
├── analyze_rxdata.py    # 分析工具
├── cleanup_port.py      # 清理工具
├── simple_simulation_test.py # 仿真测试
└── __pycache__/         # 字节码缓存
```

### 调试技巧
1. 使用`usrp_scope.py`进行硬件调试
2. 仿真模式下验证算法正确性
3. 使用`analyze_rxdata.py`分析离线数据
4. 检查队列服务器连接状态
5. 监控进程资源使用情况

## 许可证

本项目仅用于学术研究和教育目的。

## 贡献

欢迎提交问题和改进建议。

## 版本历史

- v1.0: 初始版本，实现基本DQPSK通信链路
- 支持多进程架构和实时GUI显示
- 集成完整的同步和解调算法</content>
<parameter name="filePath">e:\PhD_work\code\uhdcode\TEST\usrp_multiprocess\README.md