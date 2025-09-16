import numpy as np
import uhd
import threading
import queue
import time
import argparse
import collections
import os
import pickle
import socket
import struct
import multiprocessing
from multiprocessing.managers import BaseManager
from dqpsk_system import USRP_DQPSK_System

class RXProgram:
    """接收程序：高速接收数据，滤除噪声，通过文件IPC发送到处理程序"""

    def __init__(self, args):
        self.args = args
        self.running = threading.Event()
        self.rx_enabled = threading.Event()

        # 初始化DQPSK系统
        self.qpsk_system = USRP_DQPSK_System(
            mode="hardware",
            center_freq=args.rx_freq,
            samp_rate=args.rate,
            tx_gain=0,  # 接收程序不需要发射增益
            rx_gain=args.rx_gain,
            sps=2,
            roll_off=0.35,
            verbose=True
        )

        # 环形缓冲区设计
        self.buffer_size = args.buffer_size  # 缓冲区大小
        self.rx_buffer = np.zeros(self.buffer_size, dtype=np.complex64)
        self.buffer_head = 0  # 写入指针
        self.buffer_tail = 0  # 读取指针
        self.buffer_lock = threading.Lock()  # 保护环形缓冲区
        
        # 发送块大小（连续读取的长度）
        self.send_block_size = 1000  # 每次发送1000个复数样本

        # IPC相关
        self.ipc_mode = args.ipc_mode
        self.ipc_queue = None  # 用于Queue模式
        self.udp_host = args.udp_host
        self.udp_port = args.udp_port
        self.udp_socket = None

        # 根据IPC模式初始化
        if self.ipc_mode == "udp":
            self._init_udp()
        elif self.ipc_mode == "queue":
            print("Queue IPC模式已选择，等待设置Queue对象")
        else:
            raise ValueError(f"不支持的IPC模式: {self.ipc_mode}")

        # USRP相关
        self.usrp = None
        self.rx_streamer = None

        # 统计信息
        self.rx_samples_received = 0
        self.noise_discard_count = 0
        self.overflow_count = 0

        # 线程
        self.rx_thread = None
        self.ipc_send_thread = None

    def set_queue(self, ipc_queue):
        """设置IPC Queue对象（用于Queue模式）"""
        if self.ipc_mode == "queue":
            self.ipc_queue = ipc_queue
            print("Queue对象已设置")
        else:
            print("警告: 非Queue模式下设置Queue对象无效")

    def _init_udp(self):
        """初始化UDP通信"""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print(f"UDP通信已初始化: {self.udp_host}:{self.udp_port}")
        except Exception as e:
            print(f"UDP初始化失败: {e}")
            raise

    def _get_available_space(self):
        """计算环形缓冲区的可用空间"""
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
                # 缓冲区满，丢弃数据
                self.overflow_count += 1

    def _read_from_buffer(self, num_samples):
        """从环形缓冲区读取固定长度的数据"""
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
                return None  # 数据不足

    def _init_usrp(self):
        """初始化USRP设备"""
        try:
            # 初始化USRP
            self.usrp = uhd.usrp.MultiUSRP(self.args.args)

            # 配置参数
            self.usrp.set_clock_source("internal")
            self.usrp.set_time_source("internal")
            pc_time_sec = time.time()
            uhd_time = uhd.types.TimeSpec(pc_time_sec)
            self.usrp.set_time_now(uhd_time)
            self.usrp.set_rx_freq(uhd.types.TuneRequest(self.args.rx_freq))
            self.usrp.set_rx_gain(self.args.rx_gain)
            self.usrp.set_rx_rate(self.args.rate)

            print(f"USRP接收初始化完成: 频率={self.args.rx_freq/1e6:.1f}MHz, 增益={self.args.rx_gain}dB")

            # 创建接收流
            rx_st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            rx_st_args.channels = [0]
            self.rx_streamer = self.usrp.get_rx_stream(rx_st_args)

            print("接收流创建完成")

        except Exception as e:
            print(f"USRP初始化失败: {str(e)}")
            raise

    def rx_thread_func(self):
        """接收线程：高速接收数据，滤除噪声"""
        print(f"接收线程启动: 缓冲区大小 {self.args.buffer_size}")

        if self.rx_streamer is None:
            print("接收线程: rx_streamer未初始化")
            return

        # 快速接收缓冲区
        fast_recv_buffer_size = 2048
        recv_buffer = np.zeros((1, fast_recv_buffer_size), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()

        # 启动连续接收
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)

        monitor_count = 0

        while self.running.is_set():
            if self.rx_enabled.is_set():
                try:
                    # 接收数据
                    num_samps = self.rx_streamer.recv(recv_buffer, metadata, timeout=0.5)

                    if num_samps > 0:
                        samples = recv_buffer[0][:num_samps].copy()
                        self.rx_samples_received += num_samps

                        # 噪声检测：计算信号功率
                        step = max(1, len(samples) // 100)
                        signal_power = np.mean(np.abs(samples[::step])**2)

                        # 功率阈值判断
                        if signal_power < 0.2:
                            self.noise_discard_count += 1
                            
                            continue

                        # 有效信号，连续写入环形缓冲区
                        self._write_to_buffer(samples)
                        
                        if monitor_count % 50 == 0:  # 从100改为50，更频繁地显示
                            available_samples = (self.buffer_head - self.buffer_tail) % self.buffer_size
                            print(f"接收调试: 信号功率 {signal_power:.6f}, 缓冲区使用率 {available_samples}/{self.buffer_size}")

                    # 检查UHD错误
                    if metadata.error_code != 0:
                        if monitor_count % 1000 == 0:
                            print(f"UHD错误: {metadata.error_code}")

                except Exception as e:
                    if monitor_count % 1000 == 0:
                        print(f"接收线程错误: {str(e)}")
            else:
                time.sleep(0.001)

        # 停止接收
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        print("接收线程结束")

    def ipc_send_thread_func(self):
        """IPC发送线程：从环形缓冲区连续读取固定长度数据，通过UDP或Queue发送"""
        print(f"{self.ipc_mode.upper()} IPC发送线程启动")

        while self.running.is_set():
            try:
                # 从环形缓冲区读取固定长度的数据块
                data_block = self._read_from_buffer(self.send_block_size)
                if data_block is not None:
                    if self.ipc_mode == "udp" and self.udp_socket:
                        # 使用UDP发送
                        try:
                            # 将numpy数组序列化为bytes
                            data_bytes = pickle.dumps(data_block)
                            # 添加数据长度前缀
                            length_prefix = struct.pack('!I', len(data_bytes))
                            message = length_prefix + data_bytes

                            self.udp_socket.sendto(message, (self.udp_host, self.udp_port))
                            print(f"UDP发送: 数据块大小 {len(data_block)}, 消息大小 {len(message)} bytes")
                        except Exception as e:
                            print(f"UDP发送错误: {str(e)}")
                            # 如果发送失败，可以选择重新写入缓冲区（可选）
                    elif self.ipc_mode == "queue" and self.ipc_queue:
                        # 使用Queue发送
                        try:
                            self.ipc_queue.put(data_block, timeout=1.0)
                            print(f"Queue发送: 数据块大小 {len(data_block)}")
                        except queue.Full:
                            print("Queue满，丢弃数据块")
                        except Exception as e:
                            print(f"Queue发送错误: {str(e)}")
                    else:
                        print(f"IPC模式 {self.ipc_mode} 未正确初始化")
                else:
                    # 调试信息：显示缓冲区状态
                    available_samples = (self.buffer_head - self.buffer_tail) % self.buffer_size
                    if available_samples > 0:
                        print(f"发送等待: 缓冲区有 {available_samples} 样本，需 {self.send_block_size} 样本")
                    # 没有足够的数据，等待下次循环

                time.sleep(0.01)  # 10ms发送间隔

            except Exception as e:
                print(f"IPC发送线程错误: {str(e)}")
                time.sleep(0.1)

        if self.udp_socket:
            try:
                self.udp_socket.close()
                print("UDP连接已关闭")
            except:
                pass

        print("IPC发送线程结束")

    def start(self):
        """启动接收程序"""
        print("启动接收程序...")

        # 如果是Queue模式，连接服务器获取队列
        if self.ipc_mode == "queue":
            print("连接队列服务器...")
            try:
                class QueueManager(BaseManager):
                    pass
                QueueManager.register('get_queue')

                self.queue_manager = QueueManager(address=(self.udp_host, 50000), authkey=b'queue_key')
                print("尝试连接到服务器...")
                self.queue_manager.connect()
                print("连接成功，获取队列...")
                self.ipc_queue = self.queue_manager.get_queue()
                print(f"成功连接队列服务器，获取队列对象: {self.ipc_queue}")
            except Exception as e:
                print(f"连接队列服务器失败: {e}")
                import traceback
                traceback.print_exc()
                return

        # 初始化USRP
        self._init_usrp()

        # 设置运行标志
        self.running.set()

        # 启动接收线程
        self.rx_thread = threading.Thread(target=self.rx_thread_func)
        self.rx_thread.daemon = True
        self.rx_thread.start()

        # 启动IPC发送线程
        self.ipc_send_thread = threading.Thread(target=self.ipc_send_thread_func)
        self.ipc_send_thread.daemon = True
        self.ipc_send_thread.start()

        # 使能接收
        self.rx_enabled.set()

        print("接收程序已启动，按Ctrl+C停止...")

        try:
            while self.running.is_set():
                time.sleep(1)
                # 打印统计信息
                #available_samples = (self.buffer_head - self.buffer_tail) % self.buffer_size
                #print(f"统计: 接收样本 {self.rx_samples_received}, 噪声丢弃 {self.noise_discard_count}, 缓冲区使用 {available_samples}/{self.buffer_size}")
        except KeyboardInterrupt:
            print("\n收到停止信号...")

        self.stop()

    def stop(self):
        """停止接收程序"""
        print("停止接收程序...")

        self.running.clear()
        self.rx_enabled.clear()

        # 等待线程结束
        if self.rx_thread:
            self.rx_thread.join(timeout=2)
        if self.ipc_send_thread:
            self.ipc_send_thread.join(timeout=2)

        print("接收程序已停止")

def main():
    parser = argparse.ArgumentParser(description="USRP DQPSK接收程序")
    parser.add_argument("--rx_freq", type=float, default=900e6, help="接收频率 (Hz)")
    parser.add_argument("--rate", type=float, default=1e6, help="采样率 (Hz)")
    parser.add_argument("--rx_gain", type=float, default=50, help="接收增益 (dB)")
    parser.add_argument("--args", type=str, default="name=MyB210_01", help="USRP设备参数")
    parser.add_argument("--buffer_size", type=int, default=10000, help="接收缓冲区大小")
    parser.add_argument("--udp_host", type=str, default="127.0.0.1", help="UDP通信主机地址")
    parser.add_argument("--udp_port", type=int, default=12345, help="UDP通信端口")
    parser.add_argument("--ipc_mode", type=str, default="queue", choices=["udp", "queue"], help="IPC模式：udp 或 queue")

    args = parser.parse_args()

    # 创建接收程序
    rx_program = RXProgram(args)

    # 启动
    rx_program.start()

if __name__ == "__main__":
    main()