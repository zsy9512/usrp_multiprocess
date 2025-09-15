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

        # 缓冲区设计
        self.large_pool_queue = collections.deque(maxlen=args.buffer_size)

        # IPC相关
        self.use_pipe = args.use_pipe.lower() == 'true'
        self.use_queue = getattr(args, 'use_queue', False)
        self.udp_host = args.udp_host
        self.udp_port = args.udp_port
        self.udp_socket = None
        self.queue_conn = None

        if self.use_queue:
            # 使用Queue模式 - 简化实现，直接使用UDP作为替代
            print("Queue模式启用，使用UDP作为通信方式")
            self._init_udp()
        elif self.use_pipe:
            # 使用Pipe模式
            try:
                pipe_fd = int(os.environ.get('PIPE_FD', '3'))
                import multiprocessing.connection
                self.pipe_conn = multiprocessing.connection.Connection(os.dup(pipe_fd))
                print("Pipe IPC模式已启用")
            except Exception as e:
                print(f"Pipe连接失败: {e}，回退到UDP模式")
                self.use_pipe = False
                self._init_udp()
        else:
            # 使用UDP模式
            self._init_udp()
            self.ipc_file = args.ipc_file
            self.file_lock = threading.Lock()
            self.pipe_conn = None

    def _init_udp(self):
        """初始化UDP通信"""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print(f"UDP通信已初始化: {self.udp_host}:{self.udp_port}")
        except Exception as e:
            print(f"UDP初始化失败: {e}")
            raise

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

    def _init_usrp(self):
        """初始化USRP设备"""
        try:
            # 初始化USRP
            self.usrp = uhd.usrp.MultiUSRP(self.args.args)

            # 配置参数
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
                        if signal_power < 0.01:
                            self.noise_discard_count += 1
                            monitor_count += 1
                            if monitor_count % 1000 == 0:
                                print(f"噪声抛弃: 功率 {signal_power:.6f}, 累计 {self.noise_discard_count}")
                            continue

                        # 有效信号，放入缓冲区
                        try:
                            self.large_pool_queue.append(samples)
                            if len(self.large_pool_queue) >= self.args.buffer_size:
                                # 缓冲区满时，移除最旧的数据
                                self.large_pool_queue.popleft()

                            if monitor_count % 100 == 0:
                                print(f"接收调试: 信号功率 {signal_power:.6f}, 队列大小 {len(self.large_pool_queue)}")

                        except Exception as e:
                            self.overflow_count += 1
                            if monitor_count % 1000 == 0:
                                print(f"缓冲区操作错误: {str(e)}")

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
        """IPC发送线程：异步从缓冲区取数据，通过UDP/Pipe/文件发送"""
        if self.use_pipe:
            print("Pipe IPC发送线程启动")
        else:
            print("UDP IPC发送线程启动")

        while self.running.is_set():
            try:
                if len(self.large_pool_queue) > 0:
                    # 从缓冲区取数据
                    samples = self.large_pool_queue.popleft()

                    if self.use_pipe and self.pipe_conn:
                        # 使用Pipe发送
                        try:
                            self.pipe_conn.send(samples)
                            print(f"Pipe发送: 数据块大小 {len(samples)}")
                        except Exception as e:
                            print(f"Pipe发送错误: {str(e)}")
                            # 重新放回缓冲区
                            self.large_pool_queue.appendleft(samples)
                    elif self.udp_socket:
                        # 使用UDP发送
                        try:
                            # 将numpy数组序列化为bytes
                            data_bytes = pickle.dumps(samples)
                            # 添加数据长度前缀
                            length_prefix = struct.pack('!I', len(data_bytes))
                            message = length_prefix + data_bytes

                            self.udp_socket.sendto(message, (self.udp_host, self.udp_port))
                            print(f"UDP发送: 数据块大小 {len(samples)}, 消息大小 {len(message)} bytes")
                        except Exception as e:
                            print(f"UDP发送错误: {str(e)}")
                            # 重新放回缓冲区
                            self.large_pool_queue.appendleft(samples)
                    else:
                        # 使用文件IPC发送
                        try:
                            with self.file_lock:
                                with open(self.ipc_file, 'wb') as f:
                                    pickle.dump(samples, f)
                            print(f"文件IPC发送: 数据块大小 {len(samples)}")
                        except Exception as e:
                            print(f"文件IPC发送错误: {str(e)}")
                            # 重新放回缓冲区
                            self.large_pool_queue.appendleft(samples)

                time.sleep(0.01)  # 10ms发送间隔

            except Exception as e:
                print(f"IPC发送线程错误: {str(e)}")
                time.sleep(0.1)

        if self.use_pipe and self.pipe_conn:
            try:
                self.pipe_conn.close()
                print("Pipe连接已关闭")
            except:
                pass

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
                print(f"统计: 接收样本 {self.rx_samples_received}, 噪声丢弃 {self.noise_discard_count}, 队列大小 {len(self.large_pool_queue)}")
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
    parser.add_argument("--buffer_size", type=int, default=1000, help="接收缓冲区大小")
    parser.add_argument("--ipc_file", type=str, default="rx_to_proc.pkl", help="IPC文件路径（文件模式）")
    parser.add_argument("--use_pipe", type=str, default="false", help="是否使用Pipe IPC（true/false）")
    parser.add_argument("--use_queue", action="store_true", help="是否使用Queue IPC")
    parser.add_argument("--udp_host", type=str, default="127.0.0.1", help="UDP通信主机地址")
    parser.add_argument("--udp_port", type=int, default=12345, help="UDP通信端口")

    args = parser.parse_args()

    # 创建接收程序
    rx_program = RXProgram(args)

    # 启动
    rx_program.start()

if __name__ == "__main__":
    main()