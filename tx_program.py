import numpy as np
import uhd
import threading
import queue
import time
import argparse
import multiprocessing
from dqpsk_system import USRP_DQPSK_System
from simulation_manager import SimulationManager, SimulationIPC

def generate_bits(mode="random", num_bits=1000):
    """生成比特数据，支持不同模式"""
    np.random.seed(42)
    if mode == "random":
        return np.random.randint(0, 2, num_bits)
    elif mode == "zeros":
        return np.zeros(num_bits, dtype=int)
    elif mode == "ones":
        return np.ones(num_bits, dtype=int)
    else:
        # 默认随机
        return np.random.randint(0, 2, num_bits)

class TXProgram:
    """发射程序：生成和发送DQPSK帧，支持硬件和仿真模式"""

    def __init__(self, args):
        self.args = args
        self.running = threading.Event()
        self.tx_enabled = threading.Event()

        # 检测运行模式
        self.mode = getattr(args, 'mode', 'hardware')

        # 根据模式初始化系统
        if self.mode == "simulation":
            # 仿真模式
            self.qpsk_system = USRP_DQPSK_System(
                mode="simulation",
                center_freq=getattr(args, 'tx_freq', 900e6),
                samp_rate=getattr(args, 'rate', 1e6),
                tx_gain=getattr(args, 'tx_gain', 40),
                rx_gain=0,
                sps=2,
                roll_off=0.35,
                verbose=True
            )

            # 仿真通信队列
            self.sim_manager = None
            self.tx_queue = None

            # USRP相关（仿真模式下为None）
            self.usrp = None
            self.tx_streamer = None

        else:
            # 硬件模式
            self.qpsk_system = USRP_DQPSK_System(
                mode="hardware",
                center_freq=args.tx_freq,
                samp_rate=args.rate,
                tx_gain=args.tx_gain,
                rx_gain=0,
                sps=2,
                roll_off=0.35,
                verbose=True
            )

            # USRP相关
            self.usrp = None
            self.tx_streamer = None

            # 仿真相关（硬件模式下为None）
            self.sim_manager = None
            self.tx_queue = None

        # 缓冲区
        self.tx_buffer_queue = queue.Queue(maxsize=200)

        # 线程
        self.data_generation_thread = None
        self.tx_thread = None
        
        # 当前发射比特（用于BER计算）
        self.current_tx_bits = None

    def set_simulation_manager(self, sim_manager, tx_queue):
        """设置仿真管理器（仿真模式使用）"""
        if self.mode == "simulation":
            self.sim_manager = sim_manager
            self.tx_queue = tx_queue
            print("发射程序: 仿真管理器已设置")
        else:
            print("警告: 非仿真模式下设置仿真管理器无效")

    def set_tx_queue(self, tx_queue):
        """设置发射队列（用于IPC通信）"""
        self.tx_queue = tx_queue
        print("发射程序: 发射队列已设置")

    def _init_usrp(self):
        """初始化USRP设备"""
        try:
            # 初始化USRP
            self.usrp = uhd.usrp.MultiUSRP(self.args.args)

            # 配置参数
            self.usrp.set_tx_freq(uhd.types.TuneRequest(self.args.tx_freq))
            self.usrp.set_tx_gain(self.args.tx_gain)
            self.usrp.set_tx_rate(self.args.rate)
            # 配置时钟和时序
            self.usrp.set_clock_source("internal")
            self.usrp.set_time_source("internal")
            pc_time_sec = time.time()
            uhd_time = uhd.types.TimeSpec(pc_time_sec)
            self.usrp.set_time_now(uhd_time)
            print(f"USRP发射初始化完成: 频率={self.args.tx_freq/1e6:.1f}MHz, 增益={self.args.tx_gain}dB")

            # 创建发射流a
            tx_st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            tx_st_args.channels = [0]
            self.tx_streamer = self.usrp.get_tx_stream(tx_st_args)

            print("发射流创建完成")

        except Exception as e:
            print(f"USRP初始化失败: {str(e)}")
            raise

    def data_generation_thread_func(self):
        """数据生成线程：异步生产DQPSK帧"""
        print("数据生成线程启动")

        while self.running.is_set():
            try:
                # 生成比特
                bits = generate_bits(self.args.bit_generator, self.qpsk_system.data_bits)
                
                # 保存当前发射比特
                self.current_tx_bits = bits.copy()
                # 生成帧
                frame = self.qpsk_system.generate_frame()

                # 准备发送信号
                tx_signal = self.qpsk_system.prepare_tx_signal(frame)

                # 放入队列
                self.tx_buffer_queue.put(tx_signal, timeout=0.1)
                # 生产间隔
                time.sleep(0.6)  # 100ms间隔

            except queue.Full:
                print("发射缓冲区已满，等待消费...")
                time.sleep(0.05)
            except Exception as e:
                print(f"数据生成错误: {str(e)}")
                time.sleep(0.5)

        print("数据生成线程结束")

    def tx_thread_func(self):
        """发射线程：从队列取数据并发送"""
        print(f"发射线程启动 (模式: {self.mode})")

        if self.mode == "simulation":
            # 仿真模式发射逻辑
            self._tx_simulation_thread_func()
        else:
            # 硬件模式发射逻辑
            self._tx_hardware_thread_func()

    def _tx_simulation_thread_func(self):
        """仿真模式发射线程"""
        if self.tx_queue is None:
            print("仿真发射线程: tx_queue未设置")
            return

        frame_count = 0

        while self.running.is_set():
            try:
                # 等待发射使能
                if not self.tx_enabled.is_set():
                    time.sleep(0.01)
                    continue

                # 从队列取数据
                if self.tx_buffer_queue.empty():
                    time.sleep(0.01)
                    continue

                tx_signal = self.tx_buffer_queue.get(timeout=0.1)

                # 发送数据 - 根据模式决定发送方式
                try:
                    if self.mode == "simulation" and hasattr(self, 'tx_queue') and self.tx_queue is not None:
                        # 仿真模式使用IPC队列：发送完整帧数据包
                        frame_packet = {
                            'frame_id': frame_count,
                            'tx_signal': tx_signal,  # 完整帧信号 (1536采样点)
                            'tx_bits': self.current_tx_bits,
                            'timestamp': time.time(),
                            'frame_type': 'complete_frame',
                            'signal_length': len(tx_signal)
                        }
                        
                        self.tx_queue.put(frame_packet, timeout=1.0)
                        print(f"仿真发射: 已发送完整帧 {frame_count} (大小: {len(tx_signal)})")
                    else:
                        # 硬件模式：使用USRP发送
                        tx_md = uhd.types.TXMetadata()
                        # 重复发送
                        for burst_idx in range(self.args.repeat_count):
                            tx_md.start_of_burst = (burst_idx == 0)
                            tx_md.end_of_burst = (burst_idx == self.args.repeat_count - 1)

                            # 发送数据
                            self.tx_streamer.send(tx_signal, tx_md, timeout=0.1)

                        print(f"硬件发射: 已发送帧 {frame_count} (重复 {self.args.repeat_count} 次)")

                    frame_count += 1

                except queue.Full:
                    print("发射队列已满，丢弃数据包")
                    time.sleep(0.1)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"仿真发射错误: {str(e)}")
                time.sleep(0.1)

        print("仿真发射线程结束")

    def _tx_hardware_thread_func(self):
        """硬件模式发射线程"""
        if self.tx_streamer is None:
            print("发射线程: tx_streamer未初始化")
            return

        tx_md = uhd.types.TXMetadata()
        frame_count = 0

        while self.running.is_set():
            try:
                # 等待发射使能
                if not self.tx_enabled.is_set():
                    time.sleep(0.01)
                    continue

                # 从队列取数据
                if self.tx_buffer_queue.empty():
                    time.sleep(0.01)
                    continue

                tx_signal = self.tx_buffer_queue.get(timeout=0.1)

                # 重复发送
                for burst_idx in range(self.args.repeat_count):
                    tx_md.start_of_burst = (burst_idx == 0)
                    tx_md.end_of_burst = (burst_idx == self.args.repeat_count - 1)

                    # 发送数据
                    self.tx_streamer.send(tx_signal, tx_md, timeout=0.1)

                frame_count += 1

                #if frame_count % 50 == 0:
                print(f"已发送 {frame_count} 个帧组（每组{self.args.repeat_count}次重复）")

                # 发送间隔
                #time.sleep(0.001)  # 10ms检测间隔

            except queue.Empty:
                continue
            except Exception as e:
                print(f"发射错误: {str(e)}")
                time.sleep(0.1)

        print("发射线程结束")

    def start(self):
        """启动发射程序"""
        print(f"启动发射程序 (模式: {self.mode})...")

        # 根据模式初始化设备
        if self.mode == "simulation":
            # 仿真模式：优先使用tx_queue，如果没有则可以只使用IPC发送
            if self.tx_queue is None:
                print("仿真模式需要先设置tx_queue，线程退出")
                return
            print("仿真模式: 初始化完成")
        else:
            # 硬件模式需要USRP初始化
            self._init_usrp()

        # 设置运行标志
        self.running.set()

        # 启动数据生成线程
        self.data_generation_thread = threading.Thread(target=self.data_generation_thread_func)
        self.data_generation_thread.daemon = True
        self.data_generation_thread.start()

        # 启动发射线程
        self.tx_thread = threading.Thread(target=self.tx_thread_func)
        self.tx_thread.daemon = True
        self.tx_thread.start()

        # 使能发射
        self.tx_enabled.set()

        print("发射程序已启动，按Ctrl+C停止...")

        try:
            while self.running.is_set():
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\n收到停止信号...")

        self.stop()

    def stop(self):
        """停止发射程序"""
        print("停止发射程序...")

        self.running.clear()
        self.tx_enabled.clear()

        # 等待线程结束
        if self.data_generation_thread:
            self.data_generation_thread.join(timeout=2)
        if self.tx_thread:
            self.tx_thread.join(timeout=2)

        print("发射程序已停止")

def main():
    parser = argparse.ArgumentParser(description="USRP DQPSK发射程序")
    parser.add_argument("--mode", type=str, default="hardware", choices=["hardware", "simulation"], help="运行模式")
    parser.add_argument("--tx_freq", type=float, default=900e6, help="发射频率 (Hz)")
    parser.add_argument("--rate", type=float, default=1e6, help="采样率 (Hz)")
    parser.add_argument("--tx_gain", type=float, default=50, help="发射增益 (dB)")
    parser.add_argument("--args", type=str, default="name=MyB210", help="USRP设备参数")
    parser.add_argument("--repeat_count", type=int, default=5, help="每个帧重复发送次数")
    parser.add_argument("--bit_generator", type=str, default="random", choices=["random", "zeros", "ones"], help="比特生成模式")
    parser.add_argument("--ipc_mode", type=str, default="queue", choices=["udp", "queue"], help="IPC模式：udp 或 queue")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="队列服务器主机地址")
    parser.add_argument("--port", type=int, default=50000, help="队列服务器端口")

    args = parser.parse_args()

    # 创建发射程序
    tx_program = TXProgram(args)

    # 如果是仿真模式且使用队列IPC，连接到队列服务器
    if args.mode == "simulation" and args.ipc_mode == "queue":
        try:
            from multiprocessing.managers import BaseManager
            print(f"连接到队列服务器: {args.host}:{args.port}")

            # 注册队列管理器
            class QueueManager(BaseManager):
                pass
            QueueManager.register('get_queue')

            # 连接到服务器
            manager = QueueManager(address=(args.host, args.port), authkey=b'queue_key')
            manager.connect()

            # 获取队列对象
            ipc_queue = manager.get_queue()
            tx_program.set_tx_queue(ipc_queue)
            print("✅ 队列连接成功")

        except Exception as e:
            print(f"❌ 队列连接失败: {e}")
            print("请确保队列服务器已启动")
            return

    # 启动
    tx_program.start()

if __name__ == "__main__":
    main()