
# usrp_scope.py
# ---------------------------------------------
# 用途说明：
# 本文件为独立的USRP自收自发示波器/频谱仪工具，
# 主要用于无线信号链路测试、硬件调试和信号可视化。
# 可实时显示发射/接收时域波形和接收信号频谱。
# 与主程序无关，仅供实验和调试使用。
# ---------------------------------------------

import threading
import time
import numpy as np
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import uhd
import argparse
from dqpsk_system import USRP_DQPSK_System

def parse_args():
    parser = argparse.ArgumentParser(description='USRP Scope for DQPSK Transceiver')
    parser.add_argument('--args', default='name=MyB210_01', help='USRP device arguments')
    parser.add_argument('--tx_freq', type=float, default=915e6, help='Transmit frequency (Hz)')
    parser.add_argument('--rx_freq', type=float, default=915e6, help='Receive frequency (Hz)')
    parser.add_argument('--rate', type=float, default=1e6, help='Sample rate (Hz)')
    parser.add_argument('--tx_gain', type=float, default=50, help='Transmit gain (dB)')
    parser.add_argument('--rx_gain', type=float, default=30, help='Receive gain (dB)')
    parser.add_argument('--tx_chan', type=int, default=0, help='Transmit channel')
    parser.add_argument('--rx_chan', type=int, default=0, help='Receive channel')
    parser.add_argument('--clock_source', default='internal', help='Clock source')
    parser.add_argument('--time_source', default='external', help='Time source')
    parser.add_argument('--sps', type=int, default=2, help='Samples per symbol')
    parser.add_argument('--roll_off', type=float, default=0.35, help='Roll-off factor')
    parser.add_argument('--record_file', type=str, default=None, help='Optional: file to record received raw samples (complex64, .npy)')
    return parser.parse_args()

# 频谱功率谱密度计算函数（参考rx_spectrum_to_pyplot.py）
def psd(nfft: int, samples: np.ndarray) -> np.ndarray:
    window = np.hamming(nfft)
    fft = np.fft.fft(samples * window)
    window_power = sum(window * window) / nfft
    logfft = (
        20 * np.log10(np.abs(np.fft.fftshift(fft)) + 1e-12)  # 防止log(0)
        - 10 * np.log10(window_power)
        - 20 * np.log10(nfft)
        + 3
    )
    return logfft

class USRPScope:
    def __init__(self, args):
        self.args = args
        self.usrp = None
        self.tx_streamer = None
        self.rx_streamer = None
        self.running = threading.Event()
        self.tx_enabled = threading.Event()
        self.rx_enabled = threading.Event()
        self.tx_time_data = np.array([])
        self.rx_time_data = np.array([])
        # 频谱相关
        self.rx_spectrum_data = np.zeros(1024)
        self.spectrum_freqs = np.zeros(1024)
        self.spectrum_nfft = 1024
        self.spectrum_ref = 0
        self.spectrum_dyn = 60

        # 数据记录相关
        self.record_file = args.record_file
        self.record_fp = None
        if self.record_file:
            # 以二进制方式写入，后续可用numpy.fromfile读取
            self.record_fp = open(self.record_file, 'wb')
            # 文件头注释
            # 可用 np.fromfile('filename', dtype=np.complex64) 读取
        
        # 小缓冲区 + 流水线处理
        self.buffer_size = 4096  # 小缓冲区
        self.rx_buffer = np.zeros(self.buffer_size, dtype=np.complex64)
        self.buffer_index = 0
        self.buffer_lock = threading.Lock()
        
        # 队列
        self.processing_queue = queue.Queue(maxsize=64)  # 增大队列容量
        self.gui_queue = queue.Queue(maxsize=20)
        
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
        self.current_tx_freq = args.tx_freq
        self.current_rx_freq = args.rx_freq
        self.current_tx_gain = args.tx_gain
        self.current_rx_gain = args.rx_gain
        self.clock_source = args.clock_source
        self.time_source = args.time_source
        self.qpsk_frames = []
        self.frame_index = 0
        self._pregenerate_qpsk_frames(10)
        
        # 性能监控
        self.stats = {
            'rx_samples': 0,
            'processed_samples': 0,
            'overflow_count': 0
        }
        self.last_stats_time = time.time()

    def _init_usrp(self):
        self.usrp = uhd.usrp.MultiUSRP(self.args.args)
        self.usrp.set_clock_source("external")
        # #self.usrp.set_clock_rate(10e6) 
        self.usrp.set_time_source("external")
        #self.usrp.set_clock_source("internal")
        #self.usrp.set_time_source("internal")
        pc_time_sec = time.time()
        uhd_time = uhd.types.TimeSpec(pc_time_sec)
        self.usrp.set_time_now(uhd_time)
        self.usrp.set_tx_freq(uhd.types.TuneRequest(self.current_tx_freq), self.args.tx_chan)
        self.usrp.set_rx_freq(uhd.types.TuneRequest(self.current_rx_freq), self.args.rx_chan)
        self.usrp.set_tx_gain(self.current_tx_gain, self.args.tx_chan)
        self.usrp.set_rx_gain(self.current_rx_gain, self.args.rx_chan)
        self.usrp.set_tx_rate(self.args.rate)
        self.usrp.set_rx_rate(self.args.rate)
        tx_st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        tx_st_args.channels = [self.args.tx_chan]
        self.tx_streamer = self.usrp.get_tx_stream(tx_st_args)
        rx_st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        rx_st_args.channels = [self.args.rx_chan]
        self.rx_streamer = self.usrp.get_rx_stream(rx_st_args)

    def _pregenerate_qpsk_frames(self, count):
        self.qpsk_frames = []
        for _ in range(count):
            frame_symbols, _ = self.qpsk_system.generate_frame(return_bits=True)
            tx_signal = self.qpsk_system.prepare_tx_signal(frame_symbols)
            self.qpsk_frames.append(tx_signal.astype(np.complex64))

    def tx_thread(self):
        if self.tx_streamer is None:
            return
            
        tx_md = uhd.types.TXMetadata()
        tx_md.start_of_burst = True
        tx_md.end_of_burst = False
        
        frame_count = 0
        repeat_count = 50
        
        while self.running.is_set():
            if self.tx_enabled.is_set():
                try:
                    current_frame = self.qpsk_frames[self.frame_index]
                    self.frame_index = (self.frame_index + 1) % len(self.qpsk_frames)
                    
                    for burst_idx in range(repeat_count):
                        tx_md.start_of_burst = (burst_idx == 0)
                        if burst_idx == repeat_count - 1:
                            tx_md.end_of_burst = True
                        self.tx_streamer.send(current_frame, tx_md, timeout=0.1)
                        tx_md.start_of_burst = False
                        if burst_idx == 0:
                            self.tx_time_data = np.append(self.tx_time_data, np.real(current_frame[:min(100, len(current_frame))]))
                            if len(self.tx_time_data) > 1000:
                                self.tx_time_data = self.tx_time_data[-1000:]
                    
                    time.sleep(0.5)
                    frame_count += 1
                    
                except Exception as e:
                    pass
            else:
                time.sleep(0.01)
                
        tx_md.end_of_burst = True
        try:
            self.tx_streamer.send(np.zeros(10, dtype=np.complex64), tx_md, timeout=1.0)
        except:
            pass

    def rx_thread(self):
        if self.rx_streamer is None:
            return
            
        buffer_samps = min(self.rx_streamer.get_max_num_samps(), 2048)
        recv_buffer = np.zeros((1, buffer_samps), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        
        while self.running.is_set():
            if self.rx_enabled.is_set():
                try:
                    num_samps = self.rx_streamer.recv(recv_buffer, metadata, timeout=0.01)  # 短超时
                    
                    if num_samps > 0:
                        samples = recv_buffer[0][:num_samps]
                        self.stats['rx_samples'] += num_samps

                        # 数据记录功能：保存原始复数采样
                        if self.record_fp is not None:
                            samples.astype(np.complex64).tofile(self.record_fp)
                        
                        # 快速噪声检测
                        signal_power = np.mean(np.abs(samples[::20])**2)  # 减少计算量
                        if signal_power < 0.01:
                            continue
                        
                        # 立即处理小块数据
                        self._process_samples_immediately(samples)
                        
                        # 统计信息
                        current_time = time.time()
                        if current_time - self.last_stats_time >= 2.0:
                            self.last_stats_time = current_time
                            
                except Exception as e:
                    pass
            else:
                time.sleep(0.001)
                
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(stream_cmd)

    def _process_samples_immediately(self, samples):
        """立即处理样本数据 - 流水线处理"""
        try:
            # 极简处理：只提取用于显示的部分
            if len(samples) > 1000:
                # 数据量大时，抽取显示
                step = max(1, len(samples) // 500)
                display_data = np.real(samples[::step][:500])
            else:
                display_data = np.real(samples[:min(500, len(samples))])
            
            self.stats['processed_samples'] += len(samples)
            
            # 立即放入处理队列
            if not self.processing_queue.full():
                self.processing_queue.put_nowait(display_data)
            else:
                self.stats['overflow_count'] += 1

            # 频谱数据处理（只用最新一批样本计算）
            if len(samples) >= self.spectrum_nfft:
                # 取最新的nfft个点
                spectrum_samples = samples[-self.spectrum_nfft:]
                logfft = psd(self.spectrum_nfft, spectrum_samples)
                # 频率轴
                rx_rate = self.args.rate
                rx_freq = self.current_rx_freq
                freqs = (np.arange(-self.spectrum_nfft/2, self.spectrum_nfft/2, 1) / self.spectrum_nfft * rx_rate + rx_freq)
                self.rx_spectrum_data = logfft
                self.spectrum_freqs = freqs
                
        except Exception as e:
            self.stats['overflow_count'] += 1

    def processing_thread(self):
        """高速处理线程"""
        while self.running.is_set():
            try:
                # 高频检查处理队列
                if not self.processing_queue.empty():
                    display_data = self.processing_queue.get_nowait()
                    
                    # 立即放入GUI队列
                    if not self.gui_queue.full():
                        try:
                            self.gui_queue.put_nowait(display_data)
                        except:
                            pass
                    else:
                        self.stats['overflow_count'] += 1
                
                time.sleep(0.0005)  # 0.5ms检查间隔 - 非常高频
            except queue.Empty:
                time.sleep(0.0005)
            except Exception as e:
                time.sleep(0.001)

    def gui_update_thread(self):
        """高速GUI更新线程"""
        while self.running.is_set():
            try:
                if not self.gui_queue.empty():
                    data = self.gui_queue.get_nowait()
                    # 快速更新显示数据
                    self.rx_time_data = np.append(self.rx_time_data, data)
                    if len(self.rx_time_data) > 5000:  # 减小显示缓冲区
                        self.rx_time_data = self.rx_time_data[-5000:]
                else:
                    time.sleep(0.001)  # 5ms更新间隔
            except queue.Empty:
                time.sleep(0.001)
            except Exception as e:
                time.sleep(0.01)

    def update_tx_gain(self, val):
        self.current_tx_gain = val
        if self.usrp is not None:
            self.usrp.set_tx_gain(val, self.args.tx_chan)

    def update_rx_gain(self, val):
        if abs(val - self.current_rx_gain) > 0.5:
            self.current_rx_gain = val
            if self.usrp is not None:
                self.usrp.set_rx_gain(val, self.args.rx_chan)

    def start_tx(self, event=None):
        self.tx_enabled.set()

    def stop_tx(self, event=None):
        self.tx_enabled.clear()

    def start_rx(self, event=None):
        self.rx_enabled.set()

    def stop_rx(self, event=None):
        self.rx_enabled.clear()

    def run(self):
        self.running.set()
        self._init_usrp()
        
        # 按优先级启动线程
        rx_thread = threading.Thread(target=self.rx_thread, daemon=True, name="RX_Thread")
        processing_thread = threading.Thread(target=self.processing_thread, daemon=True, name="Processing_Thread")
        tx_thread = threading.Thread(target=self.tx_thread, daemon=True, name="TX_Thread")
        gui_thread = threading.Thread(target=self.gui_update_thread, daemon=True, name="GUI_Thread")
        
        # 启动顺序很重要
        rx_thread.start()
        processing_thread.start()
        tx_thread.start()
        gui_thread.start()
        
        self._run_gui()
        self.stop()


    def _run_gui(self):
        # 三个子图：TX时域、RX时域、RX频谱
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        tx_line, = axes[0].plot([], [], 'r-', label='TX', linewidth=0.5)
        rx_line, = axes[1].plot([], [], 'g-', label='RX', linewidth=0.5)
        spectrum_line, = axes[2].plot([], [], 'b-', label='RX Spectrum', linewidth=0.7)
        axes[0].set_title('TX Signal (Time Domain)')
        axes[1].set_title('RX Signal (Time Domain)')
        axes[2].set_title('RX Spectrum (Power Spectral Density)')
        for a in axes[:2]:
            a.set_xlim(0, 1000)
            a.set_ylim(-1.5, 1.5)
            a.grid(True, alpha=0.3)
            a.legend()
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('Power Spectral Density (dB)')
        axes[2].set_ylim(self.spectrum_ref - self.spectrum_dyn, self.spectrum_ref)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.subplots_adjust(bottom=0.25, hspace=0.5, top=0.93)

        ax_tx_gain = plt.axes([0.1, 0.16, 0.35, 0.025])
        ax_rx_gain = plt.axes([0.55, 0.16, 0.35, 0.025])

        self.slider_tx_gain = Slider(ax_tx_gain, 'TX Gain (dB)', 0, 89, valinit=self.current_tx_gain)
        self.slider_rx_gain = Slider(ax_rx_gain, 'RX Gain (dB)', 0, 76, valinit=self.current_rx_gain)

        self.slider_tx_gain.on_changed(self.update_tx_gain)
        self.slider_rx_gain.on_changed(self.update_rx_gain)

        ax_tx_start = plt.axes([0.1, 0.06, 0.12, 0.04])
        ax_tx_stop = plt.axes([0.25, 0.06, 0.12, 0.04])
        ax_rx_start = plt.axes([0.4, 0.06, 0.12, 0.04])
        ax_rx_stop = plt.axes([0.55, 0.06, 0.12, 0.04])

        self.btn_tx_start = Button(ax_tx_start, 'TX ON')
        self.btn_tx_stop = Button(ax_tx_stop, 'TX OFF')
        self.btn_rx_start = Button(ax_rx_start, 'RX ON')
        self.btn_rx_stop = Button(ax_rx_stop, 'RX OFF')

        self.btn_tx_start.on_clicked(self.start_tx)
        self.btn_tx_stop.on_clicked(self.stop_tx)
        self.btn_rx_start.on_clicked(self.start_rx)
        self.btn_rx_stop.on_clicked(self.stop_rx)

        def update(frame):
            # 时域
            tx = self.tx_time_data[-1000:] if len(self.tx_time_data) >= 1000 else self.tx_time_data
            rx = self.rx_time_data[-1000:] if len(self.rx_time_data) >= 1000 else self.rx_time_data
            tx_line.set_data(np.arange(len(tx)), tx)
            rx_line.set_data(np.arange(len(rx)), rx)
            # 动态调整显示范围
            if len(rx) > 0:
                rx_max = np.max(np.abs(rx)) * 1.2
                if rx_max > 0.01:
                    axes[1].set_ylim(-rx_max, rx_max)
            # 频谱
            if self.spectrum_freqs is not None and self.rx_spectrum_data is not None:
                spectrum_line.set_data(self.spectrum_freqs, self.rx_spectrum_data)
                if len(self.rx_spectrum_data) > 0:
                    # 修复警告：避免xlim低高相等
                    x0 = self.spectrum_freqs[0]
                    x1 = self.spectrum_freqs[-1]
                    if x0 != x1:
                        axes[2].set_xlim(x0, x1)
            return tx_line, rx_line, spectrum_line

        ani = animation.FuncAnimation(fig, update, interval=60, blit=True, save_count=50)
        plt.show()

    def stop(self):
        self.running.clear()
        self.tx_enabled.clear()
        self.rx_enabled.clear()
        time.sleep(0.1)
        # 关闭数据记录文件
        if self.record_fp is not None:
            self.record_fp.close()
            self.record_fp = None

if __name__ == "__main__":
    args = parse_args()
    scope = USRPScope(args)
    print("\n[INFO] USRP Scope: 可选参数 --record_file 可将接收原始数据保存为npy兼容格式，便于matlab/python后续分析。\n")
    scope.run()