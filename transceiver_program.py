#!/usr/bin/env python3
"""
自收自发程序：transceiver_program.py
-----------------------------------
基于usrp_scope的多线程架构，合并tx_program和rx_program的功能，
支持自收自发模式，通过IPC队列将接收数据发送到处理进程。

主要特性：
- 多线程架构：TX线程、RX线程、IPC发送线程
- 自收自发：同时进行发射和接收
- IPC通信：通过多进程队列发送数据到处理进程
- 环形缓冲区：高效数据传递
"""

import numpy as np
import uhd
import threading
import queue
import time
import argparse
import multiprocessing
from multiprocessing.managers import BaseManager
from dqpsk_system import USRP_DQPSK_System

class QueueManager(BaseManager):
    """队列管理器"""
    pass

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
        return np.random.randint(0, 2, num_bits)

class TransceiverProgram:
    """自收自发程序：支持发射和接收，通过IPC发送数据"""

    def __init__(self, args):
        self.args = args
        self.running = threading.Event()
        self.tx_enabled = threading.Event()
        self.rx_enabled = threading.Event()

        # 队列服务器连接参数
        self.queue_host = getattr(args, 'queue_host', '127.0.0.1')
        self.queue_port = getattr(args, 'queue_port', 50000)
        self.queue_authkey = getattr(args, 'queue_authkey', 'queue_key').encode()

        # DQPSK系统
        self.qpsk_system = USRP_DQPSK_System(
            mode="hardware",
            center_freq=args.tx_freq,
            samp_rate=args.rate,
            tx_gain=args.tx_gain,
            rx_gain=args.rx_gain,
            sps=args.sps,
            roll_off=args.roll_off,
            verbose=True
        )

        # USRP相关
        self.usrp = None
        self.tx_streamer = None
        self.rx_streamer = None

        # 环形缓冲区（用于RX数据传递到IPC）
        self.buffer_size = args.buffer_size
        self.rx_buffer = np.zeros(self.buffer_size, dtype=np.complex64)
        self.buffer_head = 0
        self.buffer_tail = 0
        self.buffer_lock = threading.Lock()

        # IPC队列
        self.ipc_queue = None
        self.queue_manager = None

        # 数据记录相关
        self.record_file = getattr(args, 'record_file', None)
        self.record_fp = None
        if self.record_file:
            try:
                self.record_fp = open(self.record_file, 'wb')
                print(f"接收数据将保存至: {self.record_file}")
            except Exception as e:
                print(f"无法打开记录文件 {self.record_file}: {e}")
                self.record_fp = None

        # TX相关 - 双数组机制：发送线程使用稳定副本，数据生成线程维护更新数组
        self.qpsk_frames = []  # 发送线程使用的帧数组副本（稳定不变）
        self.frame_index = 0
        self.update_frames = []  # 数据生成线程维护的更新数组
        self.update_lock = threading.Lock()  # 保护更新数组和交换操作
        self.swap_interval = 5  # 每生成5个新帧交换一次
        self.generated_count = 0
        self._pregenerate_qpsk_frames(5)  # 预生成5个帧作为初始副本

        # 统计信息
        self.stats = {
            'tx_frames': 0,
            'rx_samples': 0,
            'ipc_sent_blocks': 0,
            'overflow_count': 0,
            'noise_discard_count': 0
        }

    def _pregenerate_qpsk_frames(self, count):
        """预生成指定数量的DQPSK帧"""
        self.qpsk_frames = []
        for _ in range(count):
            frame_symbols, _ = self.qpsk_system.generate_frame(return_bits=True)
            tx_signal = self.qpsk_system.prepare_tx_signal(frame_symbols)
            self.qpsk_frames.append(tx_signal.astype(np.complex64))
        # 初始时更新数组也是相同的
        self.update_frames = self.qpsk_frames.copy()

    def set_ipc_queue(self, ipc_queue):
        """设置IPC队列"""
        self.ipc_queue = ipc_queue

    def connect_queue_server(self):
        """连接队列服务器"""
        try:
            print(f"🔗 连接队列服务器 {self.queue_host}:{self.queue_port}...")

            # 注册队列管理器
            QueueManager.register('get_queue')

            # 创建管理器并连接
            self.queue_manager = QueueManager(
                address=(self.queue_host, self.queue_port),
                authkey=self.queue_authkey
            )
            self.queue_manager.connect()

            # 获取队列
            self.ipc_queue = self.queue_manager.get_queue()

            print("✅ 队列服务器连接成功")
            return True

        except Exception as e:
            print(f"❌ 队列服务器连接失败: {e}")
            print("💡 请确保 queue_server.py 正在运行")
            return False

    def disconnect_queue_server(self):
        """断开队列服务器连接"""
        try:
            if self.queue_manager is not None:
                self.queue_manager.shutdown()
                self.queue_manager = None
                self.ipc_queue = None
                print("✅ 队列服务器连接已断开")
        except Exception as e:
            print(f"❌ 断开队列服务器连接时出错: {e}")

    def _init_usrp(self):
        """初始化USRP设备"""
        try:
            self.usrp = uhd.usrp.MultiUSRP(self.args.args)

            # 时钟和时序
            self.usrp.set_clock_source("internal")
            self.usrp.set_time_source("internal")
            pc_time_sec = time.time()
            uhd_time = uhd.types.TimeSpec(pc_time_sec)
            self.usrp.set_time_now(uhd_time)

            # TX配置
            self.usrp.set_tx_freq(uhd.types.TuneRequest(self.args.tx_freq))
            self.usrp.set_tx_gain(self.args.tx_gain)
            self.usrp.set_tx_rate(self.args.rate)

            # RX配置
            self.usrp.set_rx_freq(uhd.types.TuneRequest(self.args.rx_freq))
            self.usrp.set_rx_gain(self.args.rx_gain)
            self.usrp.set_rx_rate(self.args.rate)

            # 创建流
            tx_st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            tx_st_args.channels = [0]
            self.tx_streamer = self.usrp.get_tx_stream(tx_st_args)

            rx_st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            rx_st_args.channels = [0]
            self.rx_streamer = self.usrp.get_rx_stream(rx_st_args)

            print(f"USRP初始化完成: TX={self.args.tx_freq/1e6:.1f}MHz, RX={self.args.rx_freq/1e6:.1f}MHz")

        except Exception as e:
            print(f"USRP初始化失败: {e}")
            raise

    def _get_available_space(self):
        """计算环形缓冲区可用空间"""
        if self.buffer_head >= self.buffer_tail:
            return self.buffer_size - self.buffer_head + self.buffer_tail - 1
        else:
            return self.buffer_tail - self.buffer_head - 1

    def _write_to_buffer(self, samples):
        """写入数据到环形缓冲区"""
        num_samps = len(samples)
        with self.buffer_lock:
            available_space = self._get_available_space()
            if available_space >= num_samps:
                end_pos = (self.buffer_head + num_samps) % self.buffer_size
                if end_pos > self.buffer_head:
                    self.rx_buffer[self.buffer_head:end_pos] = samples
                else:
                    mid = self.buffer_size - self.buffer_head
                    self.rx_buffer[self.buffer_head:] = samples[:mid]
                    self.rx_buffer[:end_pos] = samples[mid:]
                self.buffer_head = end_pos
            else:
                self.stats['overflow_count'] += 1

    def _read_from_buffer(self, num_samples):
        """从环形缓冲区读取数据"""
        with self.buffer_lock:
            available_samples = (self.buffer_head - self.buffer_tail) % self.buffer_size
            if available_samples >= num_samples:
                end_pos = (self.buffer_tail + num_samples) % self.buffer_size
                if end_pos > self.buffer_tail:
                    data_block = self.rx_buffer[self.buffer_tail:end_pos].copy()
                else:
                    first_part = self.buffer_size - self.buffer_tail
                    data_block = np.concatenate([
                        self.rx_buffer[self.buffer_tail:],
                        self.rx_buffer[:end_pos]
                    ])
                self.buffer_tail = end_pos
                return data_block
            else:
                return None

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
                print(f"数据生成错误: {e}")
                time.sleep(0.5)

        print("数据生成线程结束")

    def tx_thread_func(self):
        """发射线程：只管发送第一帧，不要判断，不要等待"""
        print("发射线程启动")

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
                    # 直接取第一帧发送（无需锁，无需判断）
                    tx_signal = self.qpsk_frames[0]
                    
                    # 重复发送第一帧
                    for burst_idx in range(self.args.repeat_count):
                        tx_md.start_of_burst = (burst_idx == 0)
                        if burst_idx == self.args.repeat_count - 1:
                            tx_md.end_of_burst = True
                        self.tx_streamer.send(tx_signal, tx_md, timeout=0.1)
                        tx_md.start_of_burst = False
                    
                    self.stats['tx_frames'] += 1
                    
                    # 发送完后sleep 0.5秒
                    time.sleep(0.5)

                except Exception as e:
                    print(f"发射错误: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.01)
        
        tx_md.end_of_burst = True
        print("发射线程结束")

    def rx_thread_func(self):
        """接收线程"""
        print("接收线程启动")

        buffer_samps = 2048
        recv_buffer = np.zeros((1, buffer_samps), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)

        while self.running.is_set():
            if self.rx_enabled.is_set():
                try:
                    num_samps = self.rx_streamer.recv(recv_buffer, metadata, timeout=0.01)
                    if num_samps > 0:
                        samples = recv_buffer[0][:num_samps]
                        self.stats['rx_samples'] += num_samps

                        # 噪声检测：计算信号功率
                        # step = max(1, len(samples) // 100)
                        # signal_power = np.mean(np.abs(samples[::step])**2)

                        # # 功率阈值判断（避免接收太多杂波）
                        # if signal_power < 0.05:
                        #     self.stats['noise_discard_count'] += 1
                        #     continue

                        # 数据记录功能：保存接收原始数据
                        if self.record_fp is not None:
                            try:
                                samples.astype(np.complex64).tofile(self.record_fp)
                            except Exception as e:
                                print(f"写入记录文件失败: {e}")

                        # 写入环形缓冲区
                        self._write_to_buffer(samples)

                except Exception as e:
                    pass
            else:
                time.sleep(0.001)

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        print("接收线程结束")

    def ipc_send_thread_func(self):
        """IPC发送线程"""
        print("IPC发送线程启动")

        send_block_size = 3000

        while self.running.is_set():
            try:
                data_block = self._read_from_buffer(send_block_size)
                if data_block is not None and self.ipc_queue is not None:
                    self.ipc_queue.put(data_block, timeout=0.1)
                    self.stats['ipc_sent_blocks'] += 1

                time.sleep(0.01)

            except Exception as e:
                print(f"IPC发送错误: {e}")
                time.sleep(0.1)

        print("IPC发送线程结束")

    def run(self):
        """运行自收自发程序"""
        print("启动自收自发程序...")

        # 连接队列服务器
        if not self.connect_queue_server():
            print("❌ 无法连接队列服务器，程序退出")
            return

        # 初始化USRP
        self._init_usrp()

        # 设置运行标志
        self.running.set()

        # 启动线程
        data_thread = threading.Thread(target=self.data_generation_thread_func, daemon=True, name="DataGen_Thread")
        tx_thread = threading.Thread(target=self.tx_thread_func, daemon=True, name="TX_Thread")
        rx_thread = threading.Thread(target=self.rx_thread_func, daemon=True, name="RX_Thread")
        ipc_thread = threading.Thread(target=self.ipc_send_thread_func, daemon=True, name="IPC_Thread")

        data_thread.start()
        tx_thread.start()
        rx_thread.start()
        ipc_thread.start()

        # 使能TX和RX
        self.tx_enabled.set()
        self.rx_enabled.set()

        print("自收自发程序运行中，按Ctrl+C停止...")

        try:
            while self.running.is_set():
                time.sleep(1)
                print(f"统计: TX帧={self.stats['tx_frames']}, RX样本={self.stats['rx_samples']}, IPC块={self.stats['ipc_sent_blocks']}, 噪声丢弃={self.stats['noise_discard_count']}")
        except KeyboardInterrupt:
            print("\n收到停止信号...")

        self.stop()

    def stop(self):
        """停止程序"""
        print("停止自收自发程序...")

        self.running.clear()
        self.tx_enabled.clear()
        self.rx_enabled.clear()

        # 等待线程结束
        time.sleep(0.5)

        # 断开队列服务器连接
        self.disconnect_queue_server()

        # 关闭数据记录文件
        if self.record_fp is not None:
            try:
                self.record_fp.close()
                print(f"记录文件 {self.record_file} 已关闭")
            except Exception as e:
                print(f"关闭记录文件失败: {e}")
            self.record_fp = None

        time.sleep(0.1)
        print("自收自发程序已停止")

def main():
    parser = argparse.ArgumentParser(description="USRP自收自发程序")
    parser.add_argument("--tx_freq", type=float, default=915e6, help="发射频率 (Hz)")
    parser.add_argument("--rx_freq", type=float, default=915e6, help="接收频率 (Hz)")
    parser.add_argument("--rate", type=float, default=1e6, help="采样率 (Hz)")
    parser.add_argument("--tx_gain", type=float, default=50, help="发射增益 (dB)")
    parser.add_argument("--rx_gain", type=float, default=35, help="接收增益 (dB)")
    parser.add_argument("--args", type=str, default="name=MyB210", help="USRP设备参数")
    parser.add_argument("--buffer_size", type=int, default=500000, help="接收缓冲区大小")
    parser.add_argument("--repeat_count", type=int, default=50, help="每个帧重复发送次数")
    parser.add_argument("--bit_generator", type=str, default="random", choices=["random", "zeros", "ones"], help="比特生成模式")
    parser.add_argument("--sps", type=int, default=2, help="每符号采样点数")
    parser.add_argument("--roll_off", type=float, default=0.35, help="滚降系数")
    parser.add_argument("--record_file", type=str, default=None, help="可选：保存接收原始数据的二进制文件（complex64, .bin/.npy兼容）")
    parser.add_argument("--queue_host", type=str, default="127.0.0.1", help="队列服务器主机地址")
    parser.add_argument("--queue_port", type=int, default=50000, help="队列服务器端口")
    parser.add_argument("--queue_authkey", type=str, default="queue_key", help="队列服务器认证密钥")

    args = parser.parse_args()

    # 创建程序
    transceiver = TransceiverProgram(args)

    # 运行（队列连接在内部处理）
    transceiver.run()

if __name__ == "__main__":
    main()