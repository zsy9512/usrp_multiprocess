import numpy as np
import uhd
import threading
import queue
import time
import argparse
import multiprocessing
from dqpsk_system import USRP_DQPSK_System

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
    """发射程序：生成和发送DQPSK帧，支持USRP硬件模式"""

    def __init__(self, args):
        self.args = args
        self.running = threading.Event()
        self.tx_enabled = threading.Event()

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

        # TX相关 - 双数组机制：发送线程使用稳定副本，数据生成线程维护更新数组
        self.qpsk_frames = []  # 发送线程使用的帧数组副本（稳定不变）
        self.update_frames = []  # 数据生成线程维护的更新数组
        self.update_lock = threading.Lock()  # 保护更新数组和交换操作
        self._pregenerate_qpsk_frames(5)  # 预生成5个帧作为初始副本

        # 线程
        self.data_generation_thread = None
        self.tx_thread = None

        # 当前发射比特（用于BER计算）
        self.current_tx_bits = None

    def _pregenerate_qpsk_frames(self, count):
        """预生成指定数量的DQPSK帧"""
        self.qpsk_frames = []
        for _ in range(count):
            frame_symbols, _ = self.qpsk_system.generate_frame(return_bits=True)
            tx_signal = self.qpsk_system.prepare_tx_signal(frame_symbols)
            self.qpsk_frames.append(tx_signal.astype(np.complex64))
        # 初始时更新数组也是相同的
        self.update_frames = self.qpsk_frames.copy()

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
            self.usrp.set_clock_source(self.args.clock_source)
            self.usrp.set_time_source(self.args.time_source)
            
            if self.args.clock_source == "internal":
                # 内部时钟模式：设置PC时间，获取纳秒级时钟信号
                pc_time_ns = time.time_ns()
                full_secs = pc_time_ns // 1000000000
                frac_secs = (pc_time_ns % 1000000000) / 1000000000.0
                uhd_time = uhd.types.TimeSpec(full_secs, frac_secs)
                self.usrp.set_time_now(uhd_time)
                print(f"时钟配置: {self.args.clock_source}, 时间源: {self.args.time_source}")
            else:
                # 外部时钟模式
                print(f"时钟配置: {self.args.clock_source}, 时间源: {self.args.time_source}")
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
        """数据生成线程：逐步更新帧数组，新帧往前挤"""
        print("数据生成线程启动")

        while self.running.is_set():
            try:
                # 生成新帧
                bits = generate_bits(self.args.bit_generator, self.qpsk_system.data_bits)
                self.current_tx_bits = bits.copy()
                frame = self.qpsk_system.generate_frame()
                tx_signal = self.qpsk_system.prepare_tx_signal(frame)

                # 线程安全地更新数组：新帧往前挤
                with self.update_lock:
                    if len(self.update_frames) >= 5:
                        # 数组满时，移除最旧的帧（挤出）
                        self.update_frames.pop(0)
                    # 添加新帧到末尾（新帧往前挤）
                    self.update_frames.append(tx_signal.astype(np.complex64))
                    
                    # 直接更新发送线程使用的数组（原子操作）
                    self.qpsk_frames = self.update_frames.copy()

                # 生产间隔
                time.sleep(0.5)

            except Exception as e:
                print(f"数据生成错误: {str(e)}")
                time.sleep(0.5)

        print("数据生成线程结束")

    def tx_thread_func(self):
        """发射线程：只管发送第一帧，不要判断，不要等待"""
        print("发射线程启动 (硬件模式)")

        # 硬件模式发射逻辑
        self._tx_hardware_thread_func()

    def _tx_hardware_thread_func(self):
        """硬件模式发射线程"""
        if self.tx_streamer is None:
            print("发射线程: tx_streamer未初始化")
            return

        tx_md = uhd.types.TXMetadata()

        # 等待初始帧
        while len(self.qpsk_frames) == 0 and self.running.is_set():
            time.sleep(0.1)

        if not self.running.is_set():
            return

        # 只管发送循环：每次发送第一帧
        while self.running.is_set():
            if self.tx_enabled.is_set():
                try:
                    # 直接取第一帧发送（无需锁，无需判断）双数组机制:直接使用更新缓冲会引起线程竞争，所以用一个副本数组间歇性去复制数据。
                    tx_signal = self.qpsk_frames[0]
                    
                    # 重复发送第一帧
                    for burst_idx in range(self.args.repeat_count):
                        tx_md.start_of_burst = (burst_idx == 0)
                        if burst_idx == self.args.repeat_count - 1:
                            tx_md.end_of_burst = True
                        self.tx_streamer.send(tx_signal, tx_md, timeout=0.1)
                        tx_md.start_of_burst = False #这里的三处tx_md.start_of_burst开关是必须设置，用于数据突发，否则会引起USRP下溢出 
                    
                    
                    # 发送完后sleep 0.5秒
                    time.sleep(0.5)

                except Exception as e:
                    print(f"发射错误: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.01)
        tx_md.end_of_burst = True
        print("发射线程结束")

    def start(self):
        """启动发射程序"""
        print("启动发射程序 (硬件模式)...")

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
    parser.add_argument("--tx_freq", type=float, default=915e6, help="发射频率 (Hz)")
    parser.add_argument("--rate", type=float, default=1e6, help="采样率 (Hz)")
    parser.add_argument("--tx_gain", type=float, default=50, help="发射增益 (dB)")
    parser.add_argument("--args", type=str, default="name=MyB210", help="USRP设备参数")
    parser.add_argument("--repeat_count", type=int, default=20, help="每个帧重复发送次数")
    parser.add_argument("--bit_generator", type=str, default="random", choices=["random", "zeros", "ones"], help="比特生成模式")
    parser.add_argument("--clock_source", type=str, default="internal", choices=["internal", "external"], help="时钟源 (internal/external)")
    parser.add_argument("--time_source", type=str, default="internal", choices=["internal", "external"], help="时间源 (internal/external)")

    args = parser.parse_args()

    # 创建发射程序
    tx_program = TXProgram(args)

    # 启动
    tx_program.start()

if __name__ == "__main__":
    main()