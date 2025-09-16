#!/usr/bin/env python3
"""
仿真通信管理器 - 管理进程间通信和仿真信道
"""

import multiprocessing
import queue
import time
import threading
from typing import Dict, Any, Optional
import numpy as np

class SimulationManager:
    """仿真通信管理器，处理进程间数据传递和信道仿真"""

    def __init__(self, buffer_size: int = 1000):
        """
        初始化仿真管理器

        Args:
            buffer_size: 队列缓冲区大小
        """
        self.tx_queue = multiprocessing.Queue(maxsize=buffer_size)
        self.rx_queue = multiprocessing.Queue(maxsize=buffer_size)
        self.control_queue = multiprocessing.Queue(maxsize=100)

        # 仿真参数
        self.channel_params = {
            'snr_db': 15.0,
            'freq_offset': 1000.0,
            'phase_offset': np.pi/4,
            'delay_samples': 0,
            'multipath_enabled': False,
            'fading_enabled': False
        }

        # 统计信息
        self.stats = {
            'packets_sent': 0,
            'packets_received': 0,
            'packets_dropped': 0,
            'avg_latency': 0.0,
            'start_time': time.time()
        }

        # 控制线程
        self.running = threading.Event()
        self.monitor_thread = None

    def update_channel_params(self, **params):
        """更新信道参数"""
        for key, value in params.items():
            if key in self.channel_params:
                self.channel_params[key] = value
                print(f"更新信道参数: {key} = {value}")

    def apply_channel_effects(self, signal: np.ndarray) -> np.ndarray:
        """应用信道效应到信号"""
        # 获取当前信道参数
        snr_db = self.channel_params['snr_db']
        freq_offset = self.channel_params['freq_offset']
        phase_offset = self.channel_params['phase_offset']
        delay_samples = self.channel_params['delay_samples']

        # 应用频率偏移
        if freq_offset != 0:
            n = np.arange(len(signal))
            signal = signal * np.exp(1j * 2 * np.pi * freq_offset * n / 1e6)  # 假设采样率1MHz

        # 应用相位偏移
        if phase_offset != 0:
            signal = signal * np.exp(1j * phase_offset)

        # 添加延迟
        if delay_samples > 0:
            # 简单的延迟模型
            delayed_signal = np.zeros(len(signal) + delay_samples, dtype=signal.dtype)
            delayed_signal[delay_samples:] = signal
            signal = delayed_signal[:len(signal)]

        # 添加AWGN噪声
        if snr_db < 100:  # SNR < 100dB时添加噪声
            signal_power = np.mean(np.abs(signal)**2)
            noise_power = signal_power / (10**(snr_db/10))
            noise = np.sqrt(noise_power/2) * (
                np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
            )
            signal = signal + noise

        # 多径效应（可选）
        if self.channel_params['multipath_enabled']:
            signal = self._apply_multipath(signal)

        # 衰落效应（可选）
        if self.channel_params['fading_enabled']:
            signal = self._apply_fading(signal)

        return signal

    def _apply_multipath(self, signal: np.ndarray) -> np.ndarray:
        """应用多径效应"""
        # 简化的两径模型
        path_delay = 5  # 5个采样点的延迟
        path_gain = 0.3  # 路径增益

        if len(signal) > path_delay:
            multipath_signal = np.zeros_like(signal)
            multipath_signal[path_delay:] = signal[:-path_delay] * path_gain
            signal = signal + multipath_signal

        return signal

    def _apply_fading(self, signal: np.ndarray) -> np.ndarray:
        """应用衰落效应"""
        # 简化的瑞利衰落模型
        fading_coeff = np.random.rayleigh(1.0, len(signal))
        return signal * fading_coeff

    def send_to_receiver(self, packet: Dict[str, Any]) -> bool:
        """发送数据包到接收端"""
        try:
            # 应用信道效应
            if 'rx_signal' in packet:
                packet['rx_signal'] = self.apply_channel_effects(packet['rx_signal'])

            # 发送到接收队列
            self.rx_queue.put(packet, timeout=1.0)
            self.stats['packets_sent'] += 1

            if self.stats['packets_sent'] % 100 == 0:
                print(f"仿真管理器: 已发送 {self.stats['packets_sent']} 个数据包")

            return True

        except queue.Full:
            self.stats['packets_dropped'] += 1
            print("仿真管理器: 接收队列已满，丢弃数据包")
            return False
        except Exception as e:
            print(f"仿真管理器发送错误: {e}")
            return False

    def receive_from_transmitter(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """从发射端接收数据包"""
        try:
            packet = self.tx_queue.get(timeout=timeout)
            self.stats['packets_received'] += 1
            return packet
        except queue.Empty:
            return None
        except Exception as e:
            print(f"仿真管理器接收错误: {e}")
            return None

    def monitor_performance(self):
        """监控性能统计"""
        while self.running.is_set():
            current_time = time.time()
            elapsed = current_time - self.stats['start_time']

            if elapsed > 0:
                throughput = self.stats['packets_sent'] / elapsed
                print(f"仿真管理器统计: 发送={self.stats['packets_sent']}, "
                      f"接收={self.stats['packets_received']}, "
                      f"丢弃={self.stats['packets_dropped']}, "
                      f"吞吐量={throughput:.1f} packets/s")

            time.sleep(5.0)  # 每5秒报告一次

    def start_monitoring(self):
        """启动性能监控"""
        if self.monitor_thread is None:
            self.running.set()
            self.monitor_thread = threading.Thread(target=self.monitor_performance, daemon=True)
            self.monitor_thread.start()
            print("仿真管理器: 性能监控已启动")

    def stop_monitoring(self):
        """停止性能监控"""
        self.running.clear()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            print("仿真管理器: 性能监控已停止")

    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        current_time = time.time()
        elapsed = current_time - self.stats['start_time']

        stats_copy = self.stats.copy()
        stats_copy['elapsed_time'] = elapsed
        stats_copy['throughput'] = self.stats['packets_sent'] / elapsed if elapsed > 0 else 0

        return stats_copy

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'packets_sent': 0,
            'packets_received': 0,
            'packets_dropped': 0,
            'avg_latency': 0.0,
            'start_time': time.time()
        }
        print("仿真管理器: 统计信息已重置")


class SimulationIPC:
    """仿真IPC通信助手类"""

    @staticmethod
    def create_packet(frame_id: int, tx_signal: np.ndarray, tx_bits: np.ndarray = None,
                     metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """创建标准数据包"""
        return {
            'frame_id': frame_id,
            'tx_signal': tx_signal.copy() if tx_signal is not None else None,
            'tx_bits': tx_bits.copy() if tx_bits is not None else None,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

    @staticmethod
    def extract_packet_data(packet: Dict[str, Any]) -> tuple:
        """从数据包中提取关键数据"""
        if not isinstance(packet, dict):
            raise ValueError(f"数据包必须是字典类型，收到: {type(packet)}")

        return (
            packet.get('frame_id', 0),
            packet.get('tx_signal'),
            packet.get('rx_signal'),  # 这个可能为None
            packet.get('tx_bits'),
            packet.get('timestamp', 0),
            packet.get('metadata', {})
        )