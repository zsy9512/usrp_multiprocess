#!/usr/bin/env python3
"""
专业的USRP DQPSK处理程序 - 使用PyQt5 GUI
"""

import numpy as np
import threading
import queue
import time
import argparse
import os
import pickle
import sys
import multiprocessing
from multiprocessing.managers import BaseManager
from dqpsk_system import USRP_DQPSK_System

# PyQt5 GUI imports
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
    from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread
    import pyqtgraph as pg
    PYQT_AVAILABLE = True
    print("PyQt5 GUI模块加载成功")
except ImportError as e:
    PYQT_AVAILABLE = False
    print(f"⚠ PyQt5不可用: {e}，将使用matplotlib备用方案")

import socket
import struct

class DQPSKMonitor(QMainWindow):
    """专业的DQPSK信号监控界面"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('DQPSK Processing Monitor - Professional')
        self.setGeometry(100, 100, 1200, 600)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)

        # 创建左侧面板（星座图）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 星座图标题
        constellation_title = QLabel('DQPSK Constellation (Differential)')
        constellation_title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        left_layout.addWidget(constellation_title)

        # 创建星座图
        self.constellation_plot = pg.PlotWidget()
        self.constellation_plot.setBackground('w')
        self.constellation_plot.showGrid(x=True, y=True, alpha=0.3)
        self.constellation_plot.setLabel('left', 'Quadrature')
        self.constellation_plot.setLabel('bottom', 'In-phase')
        self.constellation_plot.setXRange(-2.5, 2.5)
        self.constellation_plot.setYRange(-2.5, 2.5)
        self.constellation_plot.setAspectLocked(True)

        # 创建星座图散点
        self.constellation_scatter = pg.ScatterPlotItem(
            size=3,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 100, 255, 120)
        )
        self.constellation_plot.addItem(self.constellation_scatter)
        left_layout.addWidget(self.constellation_plot)

        # 同步质量显示
        self.sync_quality_label = QLabel('Sync Quality: --')
        self.sync_quality_label.setStyleSheet("font-size: 12px; margin: 5px;")
        left_layout.addWidget(self.sync_quality_label)

        # 创建右侧面板（时域波形）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 时域波形标题
        time_title = QLabel('Time Domain Signal (Synchronized)')
        time_title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        right_layout.addWidget(time_title)

        # 创建时域波形图
        self.time_plot = pg.PlotWidget()
        self.time_plot.setBackground('w')
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.time_plot.setLabel('left', 'Amplitude')
        self.time_plot.setLabel('bottom', 'Sample')
        self.time_plot.setXRange(0, 2500)
        self.time_plot.setYRange(-1.5, 1.52)  # 固定纵坐标范围

        # 创建时域波形曲线
        self.time_curve = self.time_plot.plot(pen=pg.mkPen('b', width=2))
        right_layout.addWidget(self.time_plot)

        # 统计信息显示
        self.stats_label = QLabel('Frames Processed: 0')
        self.stats_label.setStyleSheet("font-size: 12px; margin: 5px;")
        right_layout.addWidget(self.stats_label)

        # 添加面板到主布局
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333;
            }
        """)

        print("PyQt5专业GUI初始化完成")

    def update_constellation(self, points):
        """更新星座图"""
        if len(points) > 0:
            # 下采样以提高性能
            step = max(1, len(points) // 2000)
            display_points = points[::step]

            # 更新散点数据
            self.constellation_scatter.setData(
                x=display_points.real,
                y=display_points.imag
            )

    def update_time_domain(self, samples):
        """更新时域波形 - 实现流动显示"""
        if len(samples) > 0:
            # 初始化滑动窗口缓冲区（如果还没有）
            if not hasattr(self, 'time_buffer'):
                self.time_buffer = np.zeros(2500, dtype=np.float32)
                self.buffer_index = 0

            # 获取实部数据
            new_samples = np.real(samples)

            # 将新数据添加到缓冲区
            available_space = len(self.time_buffer) - self.buffer_index
            if len(new_samples) <= available_space:
                # 新数据可以完全放入剩余空间
                self.time_buffer[self.buffer_index:self.buffer_index + len(new_samples)] = new_samples
                self.buffer_index += len(new_samples)
            else:
                # 需要覆盖旧数据
                # 先填满剩余空间
                self.time_buffer[self.buffer_index:] = new_samples[:available_space]
                # 剩余数据覆盖开头
                remaining = len(new_samples) - available_space
                self.time_buffer[:remaining] = new_samples[available_space:]
                self.buffer_index = remaining

            # 创建横坐标（0到2499）
            x_data = np.arange(2500)

            # 显示缓冲区数据
            self.time_curve.setData(x_data, self.time_buffer)

    def update_sync_quality(self, quality):
        """更新同步质量显示"""
        if quality is not None:
            self.sync_quality_label.setText('.2f')

    def update_stats(self, frame_count):
        """更新统计信息"""
        self.stats_label.setText(f'Frames Processed: {frame_count}')

class GUIManager(QObject):
    """GUI管理器，处理多线程通信"""

    # 定义信号
    update_constellation_signal = pyqtSignal(object)
    update_time_signal = pyqtSignal(object)
    update_quality_signal = pyqtSignal(float)
    update_stats_signal = pyqtSignal(int)

    def __init__(self, gui_queue):
        super().__init__()
        self.gui_queue = gui_queue
        self.running = True

        # 连接信号到槽
        self.update_constellation_signal.connect(self._update_constellation_slot)
        self.update_time_signal.connect(self._update_time_slot)
        self.update_quality_signal.connect(self._update_quality_slot)
        self.update_stats_signal.connect(self._update_stats_slot)

    def _update_constellation_slot(self, points):
        """更新星座图槽函数"""
        if hasattr(self, 'monitor'):
            self.monitor.update_constellation(points)

    def _update_time_slot(self, samples):
        """更新时域波形槽函数"""
        if hasattr(self, 'monitor'):
            self.monitor.update_time_domain(samples)

    def _update_quality_slot(self, quality):
        """更新同步质量槽函数"""
        if hasattr(self, 'monitor'):
            self.monitor.update_sync_quality(quality)

    def _update_stats_slot(self, count):
        """更新统计信息槽函数"""
        if hasattr(self, 'monitor'):
            self.monitor.update_stats(count)

    def process_queue(self):
        """处理GUI队列中的数据"""
        try:
            while self.running:
                if not self.gui_queue.empty():
                    gui_data = self.gui_queue.get(timeout=0.1)

                    # 发送信号更新GUI
                    if 'constellation' in gui_data:
                        self.update_constellation_signal.emit(gui_data['constellation'])
                    if 'time_domain' in gui_data:
                        self.update_time_signal.emit(gui_data['time_domain'])
                    if 'sync_quality' in gui_data:
                        self.update_quality_signal.emit(gui_data['sync_quality'])
                    if 'frame_count' in gui_data:
                        self.update_stats_signal.emit(gui_data['frame_count'])

                time.sleep(0.01)
        except:
            pass

    def set_monitor(self, monitor):
        """设置监控器引用"""
        self.monitor = monitor

    def stop(self):
        """停止处理"""
        self.running = False

class ProcessingProgram:
    """同步处理程序：从IPC接收数据，进行同步解调，显示结果，支持硬件和仿真模式"""

    def __init__(self, args):
        self.args = args
        self.running = threading.Event()

        # 检测运行模式
        self.mode = getattr(args, 'mode', 'hardware')

        # 初始化DQPSK系统
        self.qpsk_system = USRP_DQPSK_System(
            mode=self.mode,  # 处理程序不需要硬件访问
            center_freq=900e6,
            samp_rate=getattr(args, 'rate', 1e6),
            sps=2,
            roll_off=0.35,
            verbose=True
        )

        # 缓冲区设计 - 简化为帧级缓冲
        self.frame_buffer = queue.Queue(maxsize=100)  # 帧缓冲区
        self.processing_queue = queue.Queue(maxsize=1000)  # 从20增加到1000

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

        # 处理相关 - 添加同步状态保持
        self.costas_loop =self.qpsk_system._init_costas_loop(loop_bw=0.01)
        self.sync_state = {
            'freq_offset': 0.0,
            'phase_offset': 0.0,
            'costas_phase': 0.0,
            'last_valid_sync': 0,
            'sync_quality_history': [],
            'global_phase_offset': 0.0  # 添加全局相位追踪
        }

        # 统计信息
        self.total_processed_frames = 0
        self.ber_history = []
        self.dropped_frames = 0  # 记录丢弃帧数量

        # GUI相关
        self.gui_queue = queue.Queue(maxsize=200)
        self.gui_app = None
        self.gui_monitor = None
        self.gui_manager = None

        # 线程
        self.ipc_receive_thread = None
        self.processing_thread = None
        self.gui_thread = None

    def set_queue(self, ipc_queue):
        """设置IPC Queue对象（用于Queue模式）"""
        if self.ipc_mode == "queue":
            self.ipc_queue = ipc_queue
            print("IPC Queue对象已设置")
        else:
            print("警告: 非Queue模式下设置Queue对象无效")

    def set_gui_queue(self, gui_queue):
        """设置GUI Queue对象"""
        self.gui_queue = gui_queue
        print("GUI Queue对象已设置")

    def _init_udp(self):
        """初始化UDP通信"""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_socket.bind((self.udp_host, self.udp_port))
            self.udp_socket.settimeout(0.1)  # 设置超时，避免阻塞
            print(f"UDP通信已初始化: {self.udp_host}:{self.udp_port}")
        except Exception as e:
            print(f"UDP初始化失败: {e}")
            raise

    def generate_bits(self, mode="random", num_bits=1000):
        """生成比特数据，支持不同模式，与发射端保持一致"""
        np.random.seed(42)  # 使用相同的种子确保一致性
        if mode == "random":
            return np.random.randint(0, 2, num_bits)
        elif mode == "zeros":
            return np.zeros(num_bits, dtype=int)
        elif mode == "ones":
            return np.ones(num_bits, dtype=int)
        else:
            # 默认随机
            return np.random.randint(0, 2, num_bits)
    def _udp_receive_thread_func(self):
        """UDP接收线程（原有的硬件模式逻辑）"""
        print("UDP IPC接收线程启动")

        while self.running.is_set():
            try:
                if self.udp_socket:
                    # 从UDP接收数据
                    try:
                        data, addr = self.udp_socket.recvfrom(65536)  # 64KB缓冲区
                        if len(data) > 4:  # 至少包含长度前缀
                            # 解析长度前缀
                            expected_length = struct.unpack('!I', data[:4])[0]
                            data_bytes = data[4:]

                            if len(data_bytes) == expected_length:
                                # 反序列化数据
                                samples = pickle.loads(data_bytes)
                                #print(f"UDP接收: 数据块大小 {len(samples)}, 来自 {addr}")

                                # 放入处理队列
                                try:
                                    self.processing_queue.put(samples, timeout=1.0)
                                except queue.Full:
                                    print("处理队列已满，丢弃数据块")
                            else:
                                print(f"UDP数据长度不匹配: 期望 {expected_length}, 实际 {len(data_bytes)}")
                    except socket.timeout:
                        # 超时，继续等待
                        pass
                    except Exception as e:
                        print(f"UDP接收错误: {str(e)}")
                        time.sleep(0.01)

            except Exception as e:
                print(f"IPC接收线程错误: {str(e)}")
                time.sleep(0.1)

        if self.udp_socket:
            try:
                self.udp_socket.close()
                print("UDP连接已关闭")
            except:
                pass

        print("IPC接收线程结束")

    def queue_receive_thread_func(self):
        """Queue接收线程：从共享队列接收数据"""
        print("Queue IPC接收线程启动")

        while self.running.is_set():
            try:
                # 从共享队列获取数据 (reduced timeout for faster polling)
                samples = self.ipc_queue.get(timeout=0.05)  # Changed from 0.1 to 0.05 for responsiveness
                #print(f"Queue接收成功: 数据块大小 {len(samples)}")

                # 放入处理队列
                try:
                    self.processing_queue.put(samples, timeout=1.0)
                except queue.Full:
                    print("处理队列已满，丢弃数据块")

            except queue.Empty:
                # 队列为空，继续等待 (no print to reduce spam)
                time.sleep(0.01)
            except Exception as e:
                print(f"Queue接收错误: {str(e)} - 检查服务器连接")
                time.sleep(0.1)

        print("Queue接收线程结束")

    def processing_thread_func(self):
        """处理线程：处理累积的样本数据"""
        print("处理线程启动: 支持样本数据")

        while self.running.is_set():
            try:
                # 从处理队列获取数据
                try:
                    data = self.processing_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # 检查数据类型
                if isinstance(data, np.ndarray):
                    # 原始样本数据模式
                    success = self._process_raw_samples(data)
                else:
                    print(f"未知数据类型: {type(data)}")
                    success = False

                if not success:
                    print("数据处理失败")

                time.sleep(0.01)  # 短暂休眠避免CPU占用过高

            except Exception as e:
                print(f"处理线程错误: {str(e)}")
                time.sleep(0.1)

        print("处理线程结束")

    def _process_raw_samples(self, samples):
        """处理原始样本数据（UDP/硬件模式）- 使用滑动窗口方法"""
        try:
            # 将新样本添加到缓冲区
            if not hasattr(self, 'raw_sample_buffer'):
                self.raw_sample_buffer = np.array([], dtype=np.complex64)
                self.raw_buffer_index = 0

            # 扩展缓冲区
            self.raw_sample_buffer = np.append(self.raw_sample_buffer, samples)
            self.raw_buffer_index += len(samples)

            min_process_samples = 8000  # 提升到8000，确保有足够数据进行滑动窗口
            overlap_samples = 2000  # 增加重叠样本

            # 当有足够数据时进行处理
            if self.raw_buffer_index >= min_process_samples:
                # 处理累积的数据
                success = self._process_accumulated_samples(min_process_samples)

                if success:
                    # 处理成功，缓冲区已在内部更新，无需额外移动
                    pass
                else:
                    # 处理失败，移动小窗口继续尝试，避免死循环
                    shift_size = min_process_samples // 4
                    remaining_samples = self.raw_buffer_index - shift_size
                    if remaining_samples > 0:
                        self.raw_sample_buffer = self.raw_sample_buffer[shift_size:self.raw_buffer_index]
                        self.raw_buffer_index = remaining_samples
                    else:
                        self.raw_sample_buffer = np.array([], dtype=np.complex64)
                        self.raw_buffer_index = 0

                return success
            else:
                return False  # 数据不足，等待更多数据

        except Exception as e:
            print(f"原始样本处理错误: {str(e)}")
            return False

    def _process_accumulated_samples(self, num_samples):
        """处理累积的样本数据 - 改进版：滑动窗口多帧检测和提取"""
        try:
            # 获取要处理的数据（整个缓冲区）
            process_data = self.raw_sample_buffer[:num_samples]

            # 滑动窗口参数
            frame_len_samples = 1536  # 假设帧长度（根据你的dqpsk_system调整）
            win_len = 1800  # 窗口大小
            step = 300  # 步长（小于帧长，保证不遗漏）
            detected_frames = []  # 存储检测到的帧信息

            # 滑动窗口检测
            for start in range(0, len(process_data) - win_len + 1, step):
                win_end = start + win_len
                window = process_data[start:win_end]

                # 匹配滤波
                filtered = np.convolve(window, self.qpsk_system.rrc_filter, mode='full')
                rx_symbols = filtered[::self.qpsk_system.sps]

                # PSS同步
                timing_offset = self.qpsk_system._enhanced_pss_sync(rx_symbols)

                # 验证同步质量
                pss_conj = np.conj(self.qpsk_system.pss[::-1])
                corr = np.correlate(rx_symbols, pss_conj, mode='full')
                sync_peak = np.max(np.abs(corr))
                sync_quality = sync_peak / (np.mean(np.abs(corr)) + 1e-12)

                # 更新同步质量历史
                self.sync_state['sync_quality_history'].append(sync_quality)
                if len(self.sync_state['sync_quality_history']) > 10:
                    self.sync_state['sync_quality_history'].pop(0)

                # 同步质量阈值判断（降低到1.0）
                avg_sync_quality = np.mean(self.sync_state['sync_quality_history'])
                if sync_quality > 1.0:  # 降低阈值
                    # 计算帧在全局缓冲区中的起始位置
                    frame_start_global = start + timing_offset * self.qpsk_system.sps
                    frame_end_global = frame_start_global + frame_len_samples

                    # 检查帧是否完整且在缓冲区内
                    if frame_start_global >= 0 and frame_end_global <= len(process_data):
                        detected_frames.append((frame_start_global, frame_end_global, sync_quality, rx_symbols, timing_offset))

            # 按同步质量排序，优先处理高质量帧
            detected_frames.sort(key=lambda x: x[2], reverse=True)

            # 处理检测到的帧
            processed_any = False
            for frame_start, frame_end, sync_quality, rx_symbols, timing_offset in detected_frames:
                try:
                    #print(f"处理帧: 位置 {frame_start}-{frame_end}, 质量 {sync_quality:.2f}")

                    # 频率同步
                    coarse_freq = self.qpsk_system._enhanced_sss_sync(rx_symbols, timing_offset)
                    fine_freq = self.qpsk_system._enhanced_rs_sync(rx_symbols, timing_offset, coarse_freq)
                    total_freq = coarse_freq + fine_freq
                    self.sync_state['freq_offset'] = 0.9 * self.sync_state['freq_offset'] + 0.1 * total_freq

                    # 频率校正
                    Ts = 1.0 / self.args.rate
                    n = np.arange(len(rx_symbols))
                    phase_correction = np.exp(-1j * 2 * np.pi * self.sync_state['freq_offset'] * n * Ts)
                    rx_corrected = rx_symbols * phase_correction

                    # 提取数据符号（现在包括帧序号、CRC和数据）
                    data_start = timing_offset + self.qpsk_system.preamble_len
                    frame_index_symbols_end = data_start + 4  # 帧序号4符号（8比特）
                    crc_symbols_end = frame_index_symbols_end + 8  # CRC 8符号（16比特）
                    data_end = crc_symbols_end + self.qpsk_system.data_symbols
                    
                    if data_start < len(rx_corrected) and data_end <= len(rx_corrected):
                        # 先提取帧序号符号
                        frame_index_symbols = rx_corrected[data_start:frame_index_symbols_end]
                        
                        # Costas环处理帧序号符号
                        frame_index_synced = self.costas_loop.process(frame_index_symbols)
                        frame_index_demod = self.qpsk_system.differential_decode(frame_index_synced)
                        frame_index_bits = self.qpsk_system._symbols_to_bits(frame_index_demod)
                        
                        # 解码并校验帧序号
                        frame_index, valid = self.qpsk_system.decode_frame_index_hamming_parity(frame_index_bits)
                        
                        if not valid:
                            self.dropped_frames += 1
                            #print(f"帧序号校验失败，丢弃帧。丢弃总数: {self.dropped_frames}")
                            continue  # 丢弃该帧，不继续处理
                        
                        # 提取CRC符号
                        crc_symbols = rx_corrected[frame_index_symbols_end:crc_symbols_end]
                        crc_synced = self.costas_loop.process(crc_symbols)
                        crc_demod = self.qpsk_system.differential_decode(crc_synced)
                        crc_bits = self.qpsk_system._symbols_to_bits(crc_demod)
                        
                        # 校验通过，继续提取数据符号
                        data_symbols = rx_corrected[crc_symbols_end:data_end]

                        # Costas环、差分解码
                        synchronized_symbols = self.costas_loop.process(data_symbols)
                        demod_symbols = self.qpsk_system.differential_decode(synchronized_symbols)
                        recv_bits = self.qpsk_system._symbols_to_bits(demod_symbols)
                        
                        # 验证CRC
                        if not self.qpsk_system.verify_crc16(recv_bits, crc_bits):
                            self.dropped_frames += 1
                            #print(f"CRC校验失败，丢弃帧。丢弃总数: {self.dropped_frames}")
                            continue  # CRC不匹配，丢弃帧
                        
                        # CRC校验通过，继续BER计算等
                        expected_bits = self.generate_bits(mode=getattr(self.args, 'bit_generator', 'random'), num_bits=len(recv_bits))
                        if len(expected_bits) == len(recv_bits):
                            errors = np.sum(expected_bits != recv_bits)
                            ber = errors / len(recv_bits)
                            self.ber_history.append(ber)
                            if len(self.ber_history) > 50:
                                self.ber_history.pop(0)
                            print(f"帧 {self.total_processed_frames + 1} BER: {ber:.2e}")

                        # 计算差分符号用于星座图
                        diff_symbols = synchronized_symbols[1:] * np.conj(synchronized_symbols[:-1])
                        if len(diff_symbols) > 10:
                            avg_phase = np.mean(np.angle(diff_symbols))
                            self.sync_state['global_phase_offset'] = 0.9 * self.sync_state['global_phase_offset'] + 0.1 * avg_phase
                        phase_correction = np.exp(-1j * self.sync_state['global_phase_offset'])
                        diff_symbols *= phase_correction
                        display_constellation = diff_symbols[-500:] if len(diff_symbols) >= 500 else diff_symbols

                        # 更新GUI
                        gui_data = {
                            'constellation': display_constellation.copy(),
                            'time_domain': rx_corrected[data_start-100:data_start+500].copy(),
                            'sync_quality': sync_quality,
                            'frame_count': self.total_processed_frames
                        }
                        self._update_gui_data(gui_data)

                        self.total_processed_frames += 1
                        processed_any = True

                        # 从缓冲区移除已处理帧的样本
                        self.raw_sample_buffer = np.concatenate([
                            self.raw_sample_buffer[:int(frame_start)],
                            self.raw_sample_buffer[int(frame_end):]
                        ])
                        self.raw_buffer_index -= (frame_end - frame_start)

                        # 由于缓冲区变化，重新调整num_samples
                        num_samples = min(num_samples, len(self.raw_sample_buffer))

                except Exception as e:
                    print(f"帧处理错误: {str(e)}")
                    continue

            return processed_any

        except Exception as e:
            print(f"累积样本处理错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _update_gui_data(self, gui_data):
        """更新GUI显示数据"""
        try:
            # 放入GUI队列
            try:
                self.gui_queue.put(gui_data, timeout=0.1)
            except queue.Full:
                # 队列满时，移除旧数据
                try:
                    self.gui_queue.get_nowait()
                    self.gui_queue.put(gui_data, timeout=0.1)
                except:
                    pass

        except Exception as e:
            print(f"GUI数据更新错误: {str(e)}")

    def gui_thread_func(self):
        """GUI线程：使用PyQt5创建专业界面"""
        print("GUI线程启动")

        if not PYQT_AVAILABLE:
            print("PyQt5不可用，跳过GUI初始化")
            return

        try:
            # 创建Qt应用（必须在主线程中）
            self.gui_app = QApplication(sys.argv)

            # 创建GUI监控器
            self.gui_monitor = DQPSKMonitor()

            # 创建GUI管理器
            self.gui_manager = GUIManager(self.gui_queue)
            self.gui_manager.set_monitor(self.gui_monitor)

            # 启动队列处理线程
            gui_process_thread = threading.Thread(target=self.gui_manager.process_queue, daemon=True)
            gui_process_thread.start()

            # 显示GUI
            self.gui_monitor.show()

            print("PyQt5 GUI启动成功")

            # 运行Qt事件循环
            self.gui_app.exec_()

        except Exception as e:
            print(f"GUI线程初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()

        print("GUI线程结束")

    def start_gui(self):
        """在主线程中启动GUI"""
        try:
            # 创建Qt应用
            self.gui_app = QApplication(sys.argv)

            # 创建GUI监控器
            self.gui_monitor = DQPSKMonitor()
            self.gui_monitor.gui_app = self.gui_app

            # 创建GUI管理器
            self.gui_manager = GUIManager(self.gui_queue)
            self.gui_manager.set_monitor(self.gui_monitor)

            # 启动队列处理线程
            gui_process_thread = threading.Thread(target=self.gui_manager.process_queue, daemon=True)
            gui_process_thread.start()

            # 显示GUI
            self.gui_monitor.show()

            print("PyQt5 GUI启动成功")

            # 运行Qt事件循环（这会阻塞主线程）
            self.gui_app.exec_()

        except Exception as e:
            print(f"GUI初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def start(self, enable_gui=True):
        """启动处理程序"""
        print(f"启动DQPSK处理程序 (模式: {self.mode})...")

        # 根据IPC模式初始化
        if self.ipc_mode == "queue" and self.ipc_queue is None:
            print("Queue模式需要先设置IPC队列")
            return
        elif self.ipc_mode == "udp":
            print("UDP模式初始化完成")

        # 设置运行标志
        self.running.set()

        # 根据IPC配置启动相应的接收线程
        if self.ipc_mode == "udp":
            # UDP模式
            self.ipc_receive_thread = threading.Thread(target=self._udp_receive_thread_func)
        elif self.ipc_mode == "queue":
            # 队列模式
            self.ipc_receive_thread = threading.Thread(target=self.queue_receive_thread_func)
        else:
            raise ValueError(f"不支持的IPC模式: {self.ipc_mode}")

        self.ipc_receive_thread.daemon = True
        self.ipc_receive_thread.start()

        # 启动处理线程
        self.processing_thread = threading.Thread(target=self.processing_thread_func)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # 如果启用GUI，启动GUI（这会阻塞主线程）
        if enable_gui and PYQT_AVAILABLE:
            print("启动GUI...")
            self.start_gui()
        elif enable_gui and not PYQT_AVAILABLE:
            print("GUI不可用，使用无GUI模式")
            # 运行主循环
            try:
                while self.running.is_set():
                    time.sleep(1)
                    print(f"统计: 已处理帧数 {self.total_processed_frames}")
            except KeyboardInterrupt:
                print("\n收到停止信号...")
        else:
            print("处理程序已启动（无GUI模式），按Ctrl+C停止...")
            # 运行主循环
            try:
                while self.running.is_set():
                    time.sleep(1)
                    print(f"统计: 已处理帧数 {self.total_processed_frames}")
            except KeyboardInterrupt:
                print("\n收到停止信号...")

        self.stop()

    def stop(self):
        """停止处理程序"""
        print("停止处理程序...")

        self.running.clear()

        # 注意：GUI相关代码已移除

        # 等待线程结束
        if self.ipc_receive_thread:
            self.ipc_receive_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)

        print("处理程序已停止")

def main():
    parser = argparse.ArgumentParser(description="USRP DQPSK处理程序")
    parser.add_argument("--mode", type=str, default="hardware", choices=["hardware", "simulation"], help="运行模式")
    parser.add_argument("--rate", type=float, default=1e6, help="采样率 (Hz)")
    parser.add_argument("--udp_host", type=str, default="127.0.0.1", help="UDP通信主机地址")
    parser.add_argument("--udp_port", type=int, default=12345, help="UDP通信端口")
    parser.add_argument("--ipc_mode", type=str, default="queue", choices=["udp", "queue"], help="IPC模式：udp 或 queue")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="队列服务器主机地址")
    parser.add_argument("--port", type=int, default=50000, help="队列服务器端口")
    parser.add_argument("--bit_generator", type=str, default="random", choices=["random", "zeros", "ones"], help="比特生成模式（与发射端保持一致）")

    args = parser.parse_args()

    # 创建处理程序
    processing_program = ProcessingProgram(args)

    # 如果是队列模式，连接到队列服务器
    if args.ipc_mode == "queue":
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
            processing_program.set_queue(ipc_queue)
            print("✅ 队列连接成功")

        except Exception as e:
            print(f"❌ 队列连接失败: {e}")
            print("请确保队列服务器已启动")
            return

    # 启动
    processing_program.start()

if __name__ == "__main__":
    main()