# USRP DQPSK 多进程系统

**版本**: v1.1.0  
**作者**: shengyu@hust.edu.cn

这是一个基于USRP硬件的DQPSK (Differential Quadrature Phase Shift Keying) 多进程收发系统，支持实时信号处理、同步和可视化。

## 🎯 快速开始

### 🚀 一键启动实验（推荐）
```bash
# Windows用户：双击运行
start_experiment.bat

# 或命令行运行
python start_experiment.py --mode simulation
```

### 📖 完整实验指南
- **快速开始**: `QUICK_START.md`
- **详细指南**: `EXPERIMENT_GUIDE.md`
- **测试报告**: `TEST_REPORT.md`

### 🧪 手动启动（4个终端）
```bash
# 终端1：队列服务器
python queue_server.py --host 127.0.0.1 --port 50000

# 终端2：处理程序（GUI）
python processing_program.py --mode simulation --ipc_mode queue

# 终端3：接收程序
python rx_program.py --mode simulation --ipc_mode queue

# 终端4：发射程序
python tx_program.py --mode simulation --ipc_mode queue
```

## 📁 项目文件结构

```
usrp_multiprocess/
├── 📄 README.md              # 项目说明
├── 📄 QUICK_START.md         # 快速开始指南
├── 📄 EXPERIMENT_GUIDE.md    # 详细实验指南
├── 📄 TEST_REPORT.md         # 测试报告
├── 🖥️  start_experiment.bat   # Windows一键启动
├── 🐍 start_experiment.py    # Python启动脚本
├── 🐍 demo_experiment.py     # 演示脚本
├── 🧪 test_simulation.py     # 基础测试
├── 🧪 comprehensive_test.py  # 全面测试
├── 📊 dqpsk_test_results.png # 测试结果图表
├── 🐍 dqpsk_system.py        # 核心系统
├── 🐍 simulation_manager.py  # 仿真管理器
├── 📡 tx_program.py          # 发射程序
├── 📡 rx_program.py          # 接收程序
├── 📊 processing_program.py  # 处理程序（GUI）
└── 🔗 queue_server.py       # 队列服务器
```

## 🎯 实验流程

1. **数据生成** → 发射程序连续生成DQPSK帧
2. **信号发送** → 通过USRP硬件或仿真信道发送
3. **信号接收** → 接收端捕获并初步处理
4. **数据处理** → 同步、解调、解码
5. **实时显示** → GUI显示星座图、波形、频谱

## 📊 实时监控

启动实验后，您将看到专业的监控界面：

### GUI界面组件
- **星座图**：实时显示解调符号（应聚集在理想位置）
- **时域波形**：显示接收信号的时域特性
- **频谱分析**：显示信号的频率成分
- **状态信息**：BER、SNR、同步状态、帧计数

### 性能指标
- **BER**：< 1e-3（理想）
- **吞吐量**：> 100 帧/秒
- **同步成功率**：> 95%
- **延迟**：< 50ms

## 🚀 快速开始

### 仿真模式测试（推荐新用户）
如果您没有USRP硬件或想快速验证系统功能，请先运行仿真模式：

```bash
# 运行自动化测试
python test_simulation.py
```

**测试结果示例**:
```
==================================================
测试1: 直接同步模式
==================================================
测试参数 1: {'snr_db': 15, 'freq_offset': 0, 'phase_offset': 0}
平均BER: 0.00e+00
处理时间: 0.010s
...
============================================================
测试总结报告
============================================================
✓ 直接同步测试: 通过
✓ IPC仿真测试: 通过
✓ 对比测试: 通过
✓ 帧生成测试: 通过

所有测试完成！仿真系统运行正常。
```

### 硬件模式（需要USRP设备）
如果您有USRP硬件，请参考下面的完整说明。
## 程序功能

- **信号收发**：通过USRP设备进行DQPSK信号的发射和接收
- **实时同步**：实现PSS/SSS同步、频率校正和Costas环相位跟踪
- **差分解调**：DQPSK信号的差分解码算法
- **可视化监控**：实时显示星座图、时域波形和频谱
- **多进程架构**：各组件独立运行，提高系统稳定性和性能
- **UDP通信**：使用UDP Socket进行进程间数据传输

## 主要流程

1. **发射流程**：
   - 生成DQPSK调制信号
   - 通过USRP硬件发送到指定频率

2. **接收流程**：
   - 从USRP接收射频信号
   - 进行基础噪声过滤和IQ不平衡校正

3. **处理流程**：
   - 累积接收数据进行同步处理
   - 执行频率偏移校正和相位跟踪
   - 解调DQPSK信号并提取比特
   - 实时更新GUI显示

## 主要流程示意图

```mermaid
flowchart TD
    A[发射机<br/>tx_program.py] -->|射频信号| B[接收机<br/>rx_program.py]
    B -->|UDP数据块| C[处理程序<br/>processing_program.py]
    B -->|队列数据| D[队列服务器<br/>queue_server.py]
    D -->|共享队列| C
    subgraph "通信模式"
        A
        B
        C
    end

```

> 注：队列模式下，接收机和处理程序通过队列服务器共享同一个数据队列，发射机需提前启动。

## 通信方式

系统支持两种进程间通信方式：

### UDP通信（推荐，跨平台）
- **机制**：网络协议，Socket通信
- **优势**：跨平台兼容，异步通信，无文件I/O
- **数据格式**：[4字节长度前缀] + [Pickle序列化数据]

### 队列通信（多进程专用）
- **机制**：Python multiprocessing.Queue共享内存
- **优势**：高效的进程间数据传输，无序列化开销
- **使用**：需要先启动队列服务器，然后各程序连接

## 启动方案

分别在不同终端启动各组件：

### UDP模式（推荐）
```bash
# 终端1：发射程序
python tx_program.py --tx_freq 900e6 --rate 1e6 --tx_gain 50

# 终端2：接收程序
python rx_program.py --rx_freq 900e6 --rate 1e6 --rx_gain 50 --udp_host 127.0.0.1 --udp_port 12345

# 终端3：处理程序
python processing_program.py --rate 1e6 --udp_host 127.0.0.1 --udp_port 12345
```

### 队列模式（多进程）
```bash
# 终端1：队列服务器（必须先启动）
python queue_server.py

# 终端2：发射程序（必须提前启动）
python tx_program.py --tx_freq 900e6 --rate 1e6 --tx_gain 50

# 终端3：接收程序
python rx_program.py --rx_freq 900e6 --rate 1e6 --rx_gain 50 --ipc_mode queue

# 终端4：处理程序
python processing_program.py --rate 1e6 --ipc_mode queue
```

## 参数说明

### 通用参数
- `--rate`：采样率 (Hz)，默认1MHz
- `--freq`：中心频率 (Hz)，默认900MHz

### 发送程序参数
- `--tx_gain`：发送增益 (dB)，默认50
- `--amplitude`：信号幅度，默认0.5

### 接收程序参数
- `--rx_gain`：接收增益 (dB)，默认50
- `--ipc_mode`：IPC模式选择 (udp/queue)，默认queue

### 处理程序参数
- `--ipc_mode`：IPC模式选择 (udp/queue)，默认udp
- 解调结果直接打印到控制台（前100个bit）

### 通信参数
- `--udp_port`：UDP端口，默认自动分配
- `--udp_host`：UDP主机地址，默认127.0.0.1

## 系统要求

- Python 3.7+
- UHD 4.8(USRP Hardware Driver)
- 支持的USRP设备：B210, X310等

## 安装依赖

```bash
pip install numpy scipy pyqt5 pyqtgraph uhd
```

## 输出

- **控制台输出**：解调后的比特数据直接打印到控制台（每帧前100个bit）
- **实时GUI显示**：星座图、时域波形和频谱

## 故障排除

1. **同步失败**：检查信号强度、频率设置和增益参数
2. **通信失败**：确认端口未被占用，检查防火墙设置
3. **队列连接失败**：确保队列服务器已启动，检查认证密钥是否匹配
4. **处理程序看不到数据**：确认接收程序成功连接到队列，检查日志输出
5. **GUI不显示**：确保安装了PyQt5和pyqtgraph
6. **Ctrl+C无法退出**：队列服务器现已支持正常退出，按Ctrl+C可安全停止

---

**注意**：这是一个研究和开发系统，请在实际使用前进行充分测试。

## 更新日志

### v1.1.0 (2025-09-16)
- ✅ 修复队列服务器Ctrl+C无法退出问题
- ✅ 优化处理程序数据接收逻辑，提高响应速度
- ✅ 重命名队列服务器文件为更准确的名称
- ✅ 完善队列模式通信机制
- ✅ 改进错误处理和日志输出
- ✅ 更新文档和使用说明
