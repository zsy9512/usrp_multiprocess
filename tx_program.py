import numpy as np
import uhd
import threading
import queue
import time
import argparse
from dqpsk_system import USRP_DQPSK_System

def generate_bits(mode="random", num_bits=1000):
    """生成比特数据，支持不同模式"""
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
    """发射程序：生成和发送DQPSK帧"""

    def __init__(self, args):
        self.args = args
        self.running = threading.Event()
        self.tx_enabled = threading.Event()

        # 初始化DQPSK系统
        self.qpsk_system = USRP_DQPSK_System(
            mode="hardware",
            center_freq=args.tx_freq,
            samp_rate=args.rate,
            tx_gain=args.tx_gain,
            rx_gain=0,  # 发射程序不需要接收增益
            sps=2,
            roll_off=0.35,
            verbose=True
        )

        # 缓冲区
        self.tx_buffer_queue = queue.Queue(maxsize=200)

        # USRP相关
        self.usrp = None
        self.tx_streamer = None

        # 线程
        self.data_generation_thread = None
        self.tx_thread = None

    def _init_usrp(self):
        """初始化USRP设备"""
        try:
            # 初始化USRP
            self.usrp = uhd.usrp.MultiUSRP(self.args.args)

            # 配置参数
            self.usrp.set_tx_freq(uhd.types.TuneRequest(self.args.tx_freq))
            self.usrp.set_tx_gain(self.args.tx_gain)
            self.usrp.set_tx_rate(self.args.rate)

            print(f"USRP发射初始化完成: 频率={self.args.tx_freq/1e6:.1f}MHz, 增益={self.args.tx_gain}dB")

            # 创建发射流
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

                # 生成帧
                frame = self.qpsk_system.generate_frame()

                # 准备发送信号
                tx_signal = self.qpsk_system.prepare_tx_signal(frame)

                # 放入队列
                self.tx_buffer_queue.put(tx_signal, timeout=1.0)

                if self.tx_buffer_queue.qsize() % 10 == 0:
                    print(f"已生成 {self.tx_buffer_queue.qsize()} 个帧")

                # 生产间隔
                time.sleep(0.1)  # 100ms间隔

            except queue.Full:
                print("发射缓冲区已满，等待消费...")
                time.sleep(0.05)
            except Exception as e:
                print(f"数据生成错误: {str(e)}")
                time.sleep(0.1)

        print("数据生成线程结束")

    def tx_thread_func(self):
        """发射线程：从队列取数据并发送"""
        print("发射线程启动")

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

                if frame_count % 50 == 0:
                    print(f"已发送 {frame_count} 个帧组（每组{self.args.repeat_count}次重复）")

                # 发送间隔
                time.sleep(0.01)  # 10ms检测间隔

            except queue.Empty:
                continue
            except Exception as e:
                print(f"发射错误: {str(e)}")
                time.sleep(0.1)

        print("发射线程结束")

    def start(self):
        """启动发射程序"""
        print("启动发射程序...")

        # 初始化USRP
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
                time.sleep(1)
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
    parser.add_argument("--tx_freq", type=float, default=900e6, help="发射频率 (Hz)")
    parser.add_argument("--rate", type=float, default=1e6, help="采样率 (Hz)")
    parser.add_argument("--tx_gain", type=float, default=50, help="发射增益 (dB)")
    parser.add_argument("--args", type=str, default="name=MyB210", help="USRP设备参数")
    parser.add_argument("--repeat_count", type=int, default=10, help="每个帧重复发送次数")
    parser.add_argument("--bit_generator", type=str, default="random", choices=["random", "zeros", "ones"], help="比特生成模式")

    args = parser.parse_args()

    # 创建发射程序
    tx_program = TXProgram(args)

    # 启动
    tx_program.start()

if __name__ == "__main__":
    main()