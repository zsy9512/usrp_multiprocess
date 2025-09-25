# USRP DQPSK 多进程通信系统

**版本**: v2.0.0  
**作者**: shengyu@hust.edu.cn  
**更新日期**: 2025-09-25

本系统基于USRP硬件，支持DQPSK调制解调的多进程收发、同步、实时处理和可视化。新版本采用纳秒级PC时钟同步技术，两台USRP硬件收发无需外部时钟源即可实现精确同步，支持远距离传输。自v2.0.0起，系统集成了CRC-16-CCITT数据验证和快速丢弃功能，提供完整的帧级错误检测能力。系统架构灵活，支持仿真、硬件、自收自发三种模式，适合科研、教学和原型开发。

---

##  系统功能与特色

### 核心功能
- **硬件USRP模式**：真实射频信号收发，支持B210/X310等主流设备
- **单收单发模式**：时钟源同步后已稳定跑通，BER可达5e-3以下
- **自收自发模式**：单进程多线程，TX/RX/IPC高效协同
- **仿真模式**：软件信道模拟，算法开发与性能验证
- **多进程架构**：发射、接收、处理完全分离，支持队列/UDP通信
- **队列服务**：自动连接与管理，支持多客户
- **同步算法**：PSS/SSS/RS序列同步，Costas环相位跟踪
- **差分DQPSK解调**：完整物理层处理链路
- **CRC-16数据验证**：帧级错误检测，CRC校验失败快速丢弃
- **专业GUI**：星座图、时域波形、频谱分析、实时BER统计
- **数据分析工具**：离线分析、性能评估
- **参数自适应**：支持多种采样率、增益、帧结构配置

###  系统特色
- **纳秒级时钟同步**：采用time.time_ns()获取纳秒级时间戳，两台USRP无需外部时钟源即可精确同步，支持远距离传输
- **高性能同步**：多级同步算法（PSS定时、SSS粗频、RS细频），确保低SNR下稳定同步
- **实时处理**：滑动窗口多帧检测，连续帧处理无间断
- **差分调制**：消除绝对相位模糊，抗频率偏移能力强
- **CRC-16-CCITT**：帧级数据完整性验证，CRC错误快速丢弃，提升系统可靠性
- **多进程通信**：基于multiprocessing.Queue的共享内存通信，高效可靠
- **专业可视化**：PyQt5实时GUI，星座图流动显示，同步质量监控
- **模块化设计**：核心算法、通信、处理完全解耦，便于扩展和维护
- **跨平台兼容**：支持Windows/Linux，USRP硬件抽象层统一接口

###  当前性能状态

| 模式             | BER性能           | CRC丢弃率 | 状态          | 备注 |
|------------------|------------------|-----------|----------------|------|
| **仿真模式**     | 7e-4 ~ 5e-3 (15dB)| <1%       | 完全完成     | 性能稳定，CRC有效 |
| **硬件USRP模式** | <5e-3            | <2%       | 已跑通       | 纳秒级PC时钟同步，支持远距离传输 |
| **自收自发模式** | 7e-4 ~ 5e-3      | <1%       | 完全完成     | 性能稳定，CRC有效 |

##  项目结构

`
usrp_multiprocess/
  核心算法
   dqpsk_system.py        # DQPSK调制解调核心算法
     USRP_DQPSK_System类：系统主类
     帧结构：PSS(32)+SSS(32)+RS(64)+FrameIndex(8)+CRC-16(16)+数据符号(1280)
     CRC-16-CCITT：帧级数据完整性验证和快速丢弃
     同步算法：PSS定时、SSS粗频、RS细频
     Costas环：相位跟踪和同步
     差分编码/解码、点星座图DQPSK
   __init__.py            # Python包初始化文件
  通信程序
   tx_program.py          # USRP发射程序 (仅硬件模式)
     双数组机制：数据生成与发送线程解耦
     USRP硬件接口：发射流控制
     帧重复发送：支持可配置重复次数
   rx_program.py          # 接收程序 (硬件+仿真模式)
     环形缓冲区：高效数据缓存
     IPC发送线程：多进程数据传输
     仿真信道：SNR/频偏/相偏模拟
   transceiver_program.py # 自收自发程序 (硬件模式)
     多线程架构：TX/RX/IPC独立线程
     共享USRP：TX/RX同一设备，天然同步
     队列通信：与处理程序无缝对接
   queue_server.py       # 队列服务
       BaseManager：多进程队列管理
       端口检查：自动检测和清理占用
       信号处理：优雅关闭和资源清理
  处理程序
   processing_program.py  # 处理+GUI程序 (硬件+仿真模式)
       滑动窗口检测：连续帧同步和提取
       PyQt5 GUI：专业实时可视化界面
       BER计算：实时误码率统计
       多进程通信：队列/UDP双模式支持
  启动脚本
   start_experiment.py    # 统一启动脚本
       一键启动：仿真/硬件/自收自发模式
       进程管理：自动启动和监控所有组件
       优雅关闭：信号处理和资源清理
  测试工具
   simple_simulation_test.py # 仿真性能测试
   analyze_rxdata.py      # 离线数据分析工具
  工具脚本
   cleanup_port.py        # 端口清理工具
   usrp_scope.py          # USRP示波器/频谱仪
  数据文件
     rxdata_sim.bin         # 仿真接收数据
     rxdata04.bin           # 硬件接收数据
     __pycache__/           # Python缓存文件
`

##  快速开始

### 环境要求
- Python 3.8+
- UHD 4.0+ (硬件模式)
- PyQt5 (GUI界面)
- NumPy, SciPy, Matplotlib

### 安装依赖
`ash
pip install numpy scipy matplotlib pyqt5 crcmod
`

### 运行实验

#### 仿真模式 (推荐新手)
`ash
python start_experiment.py --mode simulation
`

#### 硬件USRP模式
`ash
# 发射端
python tx_program.py

# 接收端 (另一台电脑)
python rx_program.py

# 处理端 (可选)
python processing_program.py
`

#### 自收自发模式
`ash
python start_experiment.py --mode transceiver
`

##  使用说明

### 参数配置
系统支持通过命令行参数自定义配置：

- --rate: 采样率 (默认 1e6)
- --freq: 中心频率 (默认 900e6)
- --tx-gain: 发射增益 (默认 40)
- --rx-gain: 接收增益 (默认 50)

### 数据分析
`ash
# 分析接收数据
python analyze_rxdata.py --file rxdata.bin --rate 1e6 --sps 2 --center_freq 915e6 --analyze
`

### 清理端口
`ash
python cleanup_port.py
`

##  开发与扩展

### 帧结构
`
帧格式: PSS(32) + SSS(32) + RS(64) + FrameIndex(8) + CRC-16(16) + 数据(1280符号)
- PSS: 主同步序列，用于定时同步
- SSS: 辅同步序列，用于粗频率同步
- RS: 参考序列，用于细频率同步
- FrameIndex: 帧序号，8比特
- CRC-16: 数据完整性校验
- 数据: DQPSK调制的数据符号
`

### 同步算法
1. **PSS同步**: 基于主同步序列的定时同步
2. **SSS同步**: 基于辅同步序列的粗频率估计
3. **RS同步**: 基于参考序列的细频率校正
4. **Costas环**: 相位跟踪和同步

### 扩展接口
- dqpsk_system.py: 核心算法类，可继承扩展
- queue_server.py: 通信接口，支持自定义协议
- processing_program.py: 处理逻辑，可添加新算法

##  性能优化

### 同步性能
- 多级同步确保低SNR下稳定工作
- 纳秒级时钟同步支持远距离传输
- CRC校验提升数据可靠性

### 处理效率
- 多进程架构避免阻塞
- 滑动窗口检测连续处理
- 环形缓冲区优化内存使用

##  故障排除

### 常见问题
1. **端口占用**: 运行 python cleanup_port.py 清理
2. **USRP连接失败**: 检查UHD驱动和设备连接
3. **同步失败**: 调整增益和频率参数
4. **GUI无响应**: 检查PyQt5安装和显示设置

### 日志调试
程序支持详细日志输出，可通过 --verbose 参数启用。

##  许可证

本项目仅供学术研究和教学使用，请遵守相关法律法规。

##  贡献

欢迎提交Issue和Pull Request，共同改进系统。

---

**注意**: 请确保USRP设备正确连接并配置时钟源。硬件模式需要两台USRP设备进行收发测试。
