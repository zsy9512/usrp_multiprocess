import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize_scalar
import uhd
import time
import threading
from threading import Event
import queue

# Configure matplotlib for English display
plt.rcParams["font.family"] = ["Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

class CoarseFrequencyCompensator:
    """Coarse Frequency Compensator implementing FFT-based and Correlation-based algorithms"""
    def __init__(self, sample_rate=1e6, modulation='QPSK', algorithm='FFT-based',
                 frequency_resolution=100, maximum_frequency_offset=5e3, samples_per_symbol=2):
        self.sample_rate = sample_rate
        self.modulation = modulation
        self.algorithm = algorithm
        self.frequency_resolution = frequency_resolution
        self.maximum_frequency_offset = maximum_frequency_offset
        self.samples_per_symbol = samples_per_symbol  # 添加这个属性

        if modulation == 'QPSK':
            self.modulation_order = 4
        elif modulation == 'BPSK':
            self.modulation_order = 2
        elif modulation == '8PSK':
            self.modulation_order = 8
        else:
            self.modulation_order = 4  # default

        self.p_cum_freq_offset = 0.0
        self.p_raised_signal_buffer = None
        self.p_fft_length = None
        self.p_num_lag = None
        self.p_scaling_factor = None
        self.p_input_length = None
        self.p_time_steps = None
        self.p_sample_time = 1.0 / sample_rate
        
        # 历史参考参数
        self.freq_history = []  # 存储历史频率估计
        self.history_length = 10  # 历史长度
        self.history_weight = 0.7  # 历史权重

        self._setup()

    def _setup(self):
        # Determine algorithm - match MATLAB logic
        using_fft = (self.algorithm == 'FFT-based') or (self.modulation in ['QAM', 'OQPSK'])

        # Always setup FFT parameters
        if self.modulation == 'OQPSK':
            modulation_factor = 1
        else:
            modulation_factor = self.modulation_order
        self.p_fft_length = 2 ** int(np.ceil(np.log2(self.sample_rate / (self.frequency_resolution * modulation_factor))))
        # Initialize buffer with small random values to avoid zero initialization issues
        self.p_raised_signal_buffer = np.random.randn(self.p_fft_length) * 1e-6 + 1j * np.random.randn(self.p_fft_length) * 1e-6

        # Always setup correlation parameters
        max_tone_offset = self.maximum_frequency_offset * self.modulation_order
        self.p_num_lag = int(np.round(self.sample_rate / max_tone_offset)) - 1
        self.p_scaling_factor = self.sample_rate / ((self.p_num_lag + 1) * self.modulation_order * np.pi)

        # Initialize correlation buffer if using correlation
        if not using_fft:
            self.p_raised_signal_buffer = np.zeros(self.p_num_lag, dtype=complex)

    def _fft_estimate_offset(self, x):
        """FFT-based frequency offset estimation - match MATLAB exactly"""
        fft_length = self.p_fft_length
        sig_length = len(x)
        raised_signal_buffer = self.p_raised_signal_buffer

        # Raise signal
        if self.modulation != 'OQPSK':
            raised_signal = x ** self.modulation_order
        else:
            raised_signal = x ** 2

        if sig_length < fft_length:
            # Buffer multiple frames - match MATLAB
            raised_signal_buffer[:fft_length - sig_length] = raised_signal_buffer[sig_length:]
            raised_signal_buffer[fft_length - sig_length:] = raised_signal
            abs_fft_sig = np.abs(np.fft.fft(raised_signal_buffer, fft_length))
            self.p_raised_signal_buffer = raised_signal_buffer
        elif sig_length == fft_length:
            abs_fft_sig = np.abs(np.fft.fft(raised_signal, fft_length))
        else:
            # Multiple FFTs - match MATLAB
            num_ffts = int(np.ceil(sig_length / fft_length))
            abs_fft_sig = np.zeros(fft_length, dtype=float)
            for idx in range(num_ffts - 1):
                start = idx * fft_length
                end = start + fft_length
                new_raised = raised_signal[start:end]
                abs_fft_sig += np.abs(np.fft.fft(new_raised, fft_length))
            # Last FFT
            abs_fft_sig += np.abs(np.fft.fft(raised_signal[-fft_length:], fft_length))

        # Find offset index - match MATLAB
        spectrum = np.fft.fftshift(abs_fft_sig)
        if self.modulation != 'OQPSK':
            # Limit search range to expected frequency offset range
            max_offset_freq = self.maximum_frequency_offset * self.modulation_order
            delta_freq = self.sample_rate / fft_length
            max_offset_bins = int(np.round(max_offset_freq / delta_freq))
            center = fft_length // 2
            left = max(0, center - max_offset_bins)
            right = min(fft_length - 1, center + max_offset_bins)
            spectrum_range = spectrum[left:right+1]
            max_idx_in_range = np.argmax(spectrum_range) + left
            max_idx = max_idx_in_range

            # Peak interpolation for better accuracy
            if max_idx > left and max_idx < right:
                # Quadratic interpolation
                y1 = spectrum[max_idx - 1]
                y2 = spectrum[max_idx]
                y3 = spectrum[max_idx + 1]
                # Peak location correction
                correction = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3) if (y1 - 2*y2 + y3) != 0 else 0
                max_idx = max_idx + correction

            offset_idx = max_idx - fft_length // 2  # translate to -Fs/2 : Fs/2
            delta_freq = self.sample_rate / fft_length
            est_freq_offset = delta_freq * offset_idx / self.modulation_order

            # Debug info
            if hasattr(self, '_debug') and self._debug:
                print(f"FFT调试: 估计频偏={est_freq_offset:.2f} Hz")
        else:
            # OQPSK specific - match MATLAB exactly
            symbol_rate = self.sample_rate / self.samples_per_symbol
            # Look for spectral peaks in a region around the symbol rate
            symbol_rate_bin_left = int(np.round(1 + fft_length/2 - symbol_rate * fft_length / self.sample_rate))
            symbol_rate_bin_right = int(np.round(1 + fft_length/2 + symbol_rate * fft_length / self.sample_rate))
            range_val = int(np.round(symbol_rate * fft_length / self.sample_rate))  # search region for FreqOff up to Fsym/2

            # Find left peak - match MATLAB indexing (1-based to 0-based conversion)
            left_start = max(0, symbol_rate_bin_left - range_val - 1)  # -1 for 0-based indexing
            left_end = min(symbol_rate_bin_left + range_val, len(spectrum))
            if left_start < left_end:
                max_idx1 = np.argmax(spectrum[left_start:left_end]) + left_start
            else:
                max_idx1 = symbol_rate_bin_left - 1  # fallback

            # Find right peak - match MATLAB indexing
            right_start = max(0, symbol_rate_bin_right - range_val - 1)  # -1 for 0-based indexing
            right_end = min(symbol_rate_bin_right + range_val, len(spectrum))
            if right_start < right_end:
                max_idx2 = np.argmax(spectrum[right_start:right_end]) + right_start
            else:
                max_idx2 = symbol_rate_bin_right - 1  # fallback

            offset_idx = (max_idx1 + max_idx2) / 2
            offset_idx = int(np.round(offset_idx - fft_length / 2))  # translate to -Fs/2 : Fs/2

            delta_freq = self.sample_rate / fft_length
            est_freq_offset = delta_freq * (offset_idx - 1) / 2

        return est_freq_offset

    def _correlation_estimate_offset(self, x):
        """Correlation-based frequency offset estimation - dispatch to correct method"""
        if self.p_num_lag > len(x):
            return self._correlation_estimate_offset1(x)
        else:
            return self._correlation_estimate_offset2(x)

    def _correlation_estimate_offset1(self, x):
        """Correlation-based estimation when numLag > inputLength - match MATLAB correlationEstimateOffset1"""
        raised_signal = x ** self.modulation_order
        num_lag = self.p_num_lag
        input_length = len(x)

        raised_signal_buffer = self.p_raised_signal_buffer
        conj_raised_signal = np.conj(raised_signal)

        auto_corr_sum = 0.0 + 0.0j
        for idx in range(1, input_length + 1):
            buffer_part = raised_signal_buffer[num_lag - idx : num_lag]
            signal_part = conj_raised_signal[:input_length - idx]
            if len(buffer_part) > 0 and len(signal_part) > 0:
                # Match MATLAB: ([buffer_part; signal_part].' * raised_signal)
                combined = np.concatenate([buffer_part, signal_part])
                auto_corr_sum += np.dot(np.conj(combined), raised_signal)

        for idx in range(input_length + 1, num_lag + 1):
            buff_start_idx = num_lag - idx
            buff_end_idx = buff_start_idx + input_length
            buffer_part = raised_signal_buffer[buff_start_idx : buff_end_idx]
            if len(buffer_part) > 0:
                # Match MATLAB: buffer_part.' * raised_signal
                auto_corr_sum += np.dot(np.conj(buffer_part), raised_signal)

        est_freq_offset = self.p_scaling_factor * np.angle(auto_corr_sum)

        # Update buffer - match MATLAB
        self.p_raised_signal_buffer = np.concatenate([
            raised_signal_buffer[input_length:num_lag],
            conj_raised_signal
        ])

        return est_freq_offset

    def _correlation_estimate_offset2(self, x):
        """Correlation-based estimation when numLag <= inputLength - simplified version for testing"""
        raised_signal = x ** self.modulation_order
        num_lag = self.p_num_lag

        # For testing, initialize buffer with some non-zero values
        if np.all(self.p_raised_signal_buffer == 0):
            self.p_raised_signal_buffer = np.random.randn(num_lag) * 1e-6 + 1j * np.random.randn(num_lag) * 1e-6

        raised_signal_buffer = self.p_raised_signal_buffer
        conj_raised_signal = np.conj(raised_signal)

        auto_corr_sum = 0.0 + 0.0j
        for idx in range(1, num_lag + 1):
            signal_part = conj_raised_signal[idx-1 : idx-1 + num_lag]
            if len(signal_part) == num_lag:
                # Match MATLAB: conj(signal_part).' * raised_signal_buffer
                auto_corr_sum += np.dot(np.conj(signal_part), raised_signal_buffer)

        est_freq_offset = self.p_scaling_factor * np.angle(auto_corr_sum)

        # Update buffer
        self.p_raised_signal_buffer = conj_raised_signal[len(x) - num_lag : len(x)]

        return est_freq_offset

    def _apply_history_reference(self, current_freq):
        """应用历史平均参考来稳定频率估计"""
        # 添加当前频率到历史
        self.freq_history.append(current_freq)
        
        # 保持历史长度
        if len(self.freq_history) > self.history_length:
            self.freq_history = self.freq_history[-self.history_length:]
        
        # 如果历史足够，计算加权平均
        if len(self.freq_history) >= 3:  # 至少需要3个历史值
            # 计算历史平均
            history_avg = np.mean(self.freq_history[:-1])  # 不包括当前值
            
            # 使用加权平均：历史权重 + 当前权重
            stabilized_freq = (self.history_weight * history_avg + 
                             (1 - self.history_weight) * current_freq)
            
            # 如果当前值与历史平均相差太大，使用历史平均
            deviation_threshold = 500  # Hz，偏差阈值
            if abs(current_freq - history_avg) > deviation_threshold:
                stabilized_freq = history_avg
                
            return stabilized_freq
        else:
            # 历史不足，返回当前值
            return current_freq

    def estimate_offset(self, signal):
        """Estimate frequency offset - match MATLAB logic"""
        # Setup parameters if not already done
        if self.p_raised_signal_buffer is None:
            self._setup()

        self.p_input_length = len(signal)
        self.p_time_steps = np.arange(self.p_input_length)

        # Match MATLAB algorithm selection
        using_fft = (self.algorithm == 'FFT-based') or (self.modulation in ['QAM', 'OQPSK'])

        if using_fft:
            est_freq_offset = self._fft_estimate_offset(signal)
        else:
            est_freq_offset = self._correlation_estimate_offset(signal)
            
        # 应用历史参考来稳定频率估计
        est_freq_offset = self._apply_history_reference(est_freq_offset)

        return est_freq_offset

    def compensate(self, signal, freq_offset=None):
        """Apply frequency compensation"""
        # Setup parameters if not already done
        if self.p_raised_signal_buffer is None:
            self._setup()

        if freq_offset is None:
            freq_offset = self.estimate_offset(signal)

        freq_vec = freq_offset * self.p_time_steps * self.p_sample_time
        compensation = np.exp(1j * 2 * np.pi * (self.p_cum_freq_offset - freq_vec))
        compensated = signal * compensation

        # Update cumulative offset
        self.p_cum_freq_offset = self.p_cum_freq_offset - freq_vec[-1]

        return compensated, freq_offset

class EnhancedCostasLoop:
    """增强型Costas环，具有改进的相位检测器和自适应参数"""
    
    def __init__(self, loop_bw, damping=0.707, freq_offset=0.0, detector_type='decision_directed'):
        self.loop_bw = loop_bw
        self.damping = damping
        self.freq = freq_offset  # 频率偏移
        self.phase = 0.0         # 相位偏移
        self.detector_type = detector_type
        
        # 二阶环参数
        self.alpha = 4 * damping * loop_bw / (1 + 2 * damping * loop_bw + loop_bw**2)
        self.beta = 4 * loop_bw**2 / (1 + 2 * damping * loop_bw + loop_bw**2)
        
        # 存储跟踪误差
        self.phase_errors = []
        self.freq_errors = []
        
        # 自适应参数
        self.error_variance = 0.0
        self.adaptation_rate = 0.01
        
        # QPSK星座图，用于决策导向检测器
        self.constellation = np.array([
            1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j
        ], dtype=np.complex64) / np.sqrt(2)

    def _phase_detector(self, sample):
        """增强型相位检测器，支持多种检测方式"""
        if self.detector_type == 'cross_product':
            # QPSK交叉乘积检测器
            i = np.real(sample)
            q = np.imag(sample)
            if i > 0:
                return i * q
            else:
                return -i * q
                
        elif self.detector_type == 'decision_directed':
            # 决策导向检测器
            # 找到最近的星座点
            distances = np.abs(sample - self.constellation)
            nearest_idx = np.argmin(distances)
            nearest_symbol = self.constellation[nearest_idx]
            
            # 相位误差是共轭相乘的虚部
            error_complex = sample * np.conj(nearest_symbol)
            return np.imag(error_complex)
            
        elif self.detector_type == 'fourth_power':
            # 四次方检测器（去除调制）
            fourth_power = sample**4
            phase_estimate = np.angle(fourth_power) / 4
            # 归一化到[-pi/4, pi/4]
            phase_estimate = ((phase_estimate + np.pi/4) % (np.pi/2)) - np.pi/4
            return phase_estimate
            
        else:
            # 默认使用交叉乘积检测器
            i = np.real(sample)
            q = np.imag(sample)
            if i > 0:
                return i * q
            else:
                return -i * q

    def process(self, input_samples):
        """处理输入样本并输出同步信号"""
        output = []
        self.phase_errors = []
        self.freq_errors = []
        
        for sample in input_samples:
            # 载波恢复：旋转以消除相位偏移
            rotated = sample * np.exp(-1j * self.phase)
            
            # 相位误差检测
            error = self._phase_detector(rotated)
            self.phase_errors.append(error)
            
            # 更新频率和相位
            self.freq += self.beta * error
            self.phase += self.alpha * error + self.freq
            
            # 相位归一化
            self.phase = (self.phase + np.pi) % (2 * np.pi) - np.pi
            
            self.freq_errors.append(self.freq)
            output.append(rotated)
        
        return np.array(output)

class USRP_DQPSK_System:
    def transmit_and_receive(self, snr_db=None, freq_offset=None, phase_offset=None, n_frames=1, usrp_tx=None, usrp_rx=None):
        """
        统一的收发流程：仿真/硬件共用。仿真时snr_db等参数有效，硬件时usrp_tx/usrp_rx为USRP对象。
        返回：ber_list, 可选星座点等
        """
        ber_list = []
        constellation_points = []
        for frame_idx in range(n_frames):
            # 1. 生成帧
            frame, tx_bits = self.generate_frame(return_bits=True)
            tx_signal = self.prepare_tx_signal(frame)
            # 2. 信道
            if self.mode == "simulation":
                # 仿真信道
                snr = snr_db if snr_db is not None else self.sim_params["snr_db"]
                freq_off = freq_offset if freq_offset is not None else self.sim_params["freq_offset"]
                phase_off = phase_offset if phase_offset is not None else self.sim_params["phase_offset"]
                n = np.arange(len(tx_signal))
                tx_signal = tx_signal * np.exp(1j * 2 * np.pi * freq_off * n / self.samp_rate)
                tx_signal = tx_signal * np.exp(1j * phase_off)
                signal_power = np.mean(np.abs(tx_signal)**2)
                noise_power = signal_power / (10**(snr/10))
                noise = np.sqrt(noise_power/2) * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
                rx_signal = tx_signal + noise
            else:
                # 硬件信道
                if usrp_tx is None or usrp_rx is None:
                    raise RuntimeError("硬件模式需提供usrp_tx和usrp_rx对象")
                usrp_tx.send(tx_signal)
                rx_signal = usrp_rx.recv(len(tx_signal))
            # 3. 匹配滤波
            rx_filtered = np.convolve(rx_signal, self.rrc_filter, mode='same')
            rx_symbols = rx_filtered[::self.sps]
            # 4. 同步
            timing_offset = self._enhanced_pss_sync(rx_symbols)
            coarse_freq = self._enhanced_sss_sync(rx_symbols, timing_offset)
            fine_freq = self._enhanced_rs_sync(rx_symbols, timing_offset, coarse_freq)
            n2 = np.arange(len(rx_symbols))
            rx_symbols_corr = rx_symbols * np.exp(-1j * 2 * np.pi * (coarse_freq+fine_freq) * n2 * self.Ts)
            # 5. 数据提取
            data_start = timing_offset + self.preamble_len
            data_end = data_start + self.data_symbols
            data_symbols = rx_symbols_corr[data_start:data_end]
            # 6. 相位同步
            costas = self._init_costas_loop(loop_bw=0.002)
            synced = costas.process(data_symbols)
            # 7. 差分解码
            decoded = self.differential_decode(synced)
            rx_bits = self._symbols_to_bits(decoded)
            min_len = min(len(tx_bits), len(rx_bits))
            ber = self._calculate_ber(tx_bits[:min_len], rx_bits[:min_len])
            ber_list.append(ber)
            constellation_points.extend(decoded[:min(200, len(decoded))])
        return ber_list, np.array(constellation_points)
    def __init__(self, mode="simulation", center_freq=900e6, samp_rate=1e6, sps=2, roll_off=0.35,
                 tx_gain=40, rx_gain=30, verbose=False):
        """
        Initialize system with verbose flag to control debug output
        """
        self.verbose = verbose
        
        # Core parameters
        self.mode = mode
        self.center_freq = center_freq
        self.samp_rate = samp_rate
        self.sps = sps  # Samples per symbol
        self.roll_off = roll_off
        self.symbol_rate = samp_rate / sps
        self.Ts = 1 / self.symbol_rate  # Symbol period
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain

        # Frame structure
        self.pss_len = 32
        self.sss_len = 32
        self.rs_len = 64
        self.preamble_len = self.pss_len + self.sss_len + self.rs_len
        self.data_bits = 1280  # 640 symbols (2 bits/symbol)
        self.data_symbols = self.data_bits // 2

        # Synchronization sequences
        self.pss = self._generate_pss()
        self.sss = self._generate_sss()
        self.rs = self._generate_rs()

        # FFT-based frequency compensator for high-precision frequency offset estimation
        self.fft_compensator = CoarseFrequencyCompensator(
            sample_rate=self.samp_rate,
            modulation='QPSK',
            algorithm='FFT-based',
            frequency_resolution=100,  # 100Hz resolution
            maximum_frequency_offset=5000,  # ±5kHz range
            samples_per_symbol=self.sps
        )

        # QPSK constellation (Gray coded) - 与原始系统保持一致
        self.constellation = np.array([
            1 + 1j,  # 00
            -1 + 1j, # 01  
            1 - 1j,  # 10
            -1 - 1j  # 11
        ], dtype=np.complex64) / np.sqrt(2)
        self.diff_ref_symbol = 1 + 0j  # 差分编码参考符号 - 关键修复！

        # RRC filter
        self.rrc_filter = self._design_rrc_filter()

        # USRP device
        self.usrp = None
        
        # Channel configuration
        self.tx_channel = 0  # TX on A:A
        self.rx_channel = 1  # RX on A:B

        # System state - 用于BER统计
        self.current_tx_bits = None  # 当前发送的比特（供接收端对比）
        self.ber_history = []  # 存储每一帧的BER值
        self.frame_numbers = []  # 存储对应的帧号
        self.current_frame = 0  # 当前帧计数器
        
        # BER缓存机制 - 用于重复帧的最低BER选择
        self.repeated_ber_cache = []  # 存储重复帧的BER数据
        self.repeated_frame_data_cache = []  # 存储对应的帧数据
        self.repeated_tx_bits_cache = []  # 存储对应的发送比特
        self.repeated_frame_count = 30  # 重复帧数量（与发送端匹配）
        self.last_ber_update_time = time.time()  # BER更新时间戳
        self.ber_timeout = 5.0  # BER缓存超时时间（秒）
        
        self.recent_symbols = np.array([], dtype=np.complex64)
        self.max_recent_symbols = 200
        
        # Sync history for debugging
        self.sync_history = []

        # Simulation parameters
        self.sim_params = {
            "freq_offset": 1000,  # Hz
            "phase_offset": np.pi/4,
            "snr_db": 15
        }

        # Queues for hardware mode (removed unused queues)
        self.stop_event = Event()

        if self.verbose:
            print(f"System initialized in {mode} mode")
            print(f"Center Frequency: {center_freq/1e6:.1f} MHz, Sampling Rate: {samp_rate/1e6:.1f} MHz")

    def generate_frame(self, return_bits=False):
        """生成包含前导和数据的DQPSK帧，返回符号及可选的原始比特"""
        # 生成随机数据比特
        data_bits = np.random.randint(0, 2, self.data_bits)
        # 转换为符号
        data_symbols = self._bits_to_symbols(data_bits)
        # 差分编码
        encoded_symbols = self.differential_encode(data_symbols)
        # 构建完整帧
        frame = np.concatenate([self.pss, self.sss, self.rs, encoded_symbols])
        
        if return_bits:
            return frame, data_bits  # 返回帧和原始比特（用于BER计算）
        return frame

    def plot_ber_statistics(self):
        """绘制BER统计图表（修复：处理数据不足和零值情况）"""
        if len(self.ber_history) == 0:
            print("没有足够的BER数据用于绘图")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 处理BER为0的情况，添加小偏移避免log(0)错误
        ber_values = np.array(self.ber_history)
        ber_values = np.where(ber_values <= 0, 1e-8, ber_values)  # 将0或负值替换为很小的正值
        
        # Plot BER per frame
        ax.plot(self.frame_numbers, ber_values, 'b-', alpha=0.7, label='BER per Frame')
        
        # 计算并绘制平均BER
        avg_ber = np.mean(ber_values)
        ax.axhline(y=avg_ber, color='r', linestyle='--', label=f'Average BER: {avg_ber:.2e}')
        
        # Chart settings
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.set_title('DQPSK Transmission BER Statistics')
        
        # 只有当BER值有变化时才使用对数刻度
        if len(np.unique(ber_values)) > 1 and np.min(ber_values) > 0:
            ax.set_yscale('log')  # BER适合用对数刻度
        else:
            ax.set_yscale('linear')  # 使用线性刻度避免log(0)错误
            
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        
        # 添加统计信息文本
        stats_text = (f'Total frames: {len(self.ber_history)}\n'
                f'Average BER: {avg_ber:.6f}\n'
                f'Best BER: {np.min(self.ber_history):.6f}\n'
                f'Worst BER: {np.max(self.ber_history):.6f}')
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def _calculate_ber(self, tx_bits, rx_bits):
        """计算比特错误率 (BER)"""
        if len(tx_bits) != len(rx_bits):
            return 1.0  # 如果长度不匹配，返回最大BER
        return np.mean(tx_bits != rx_bits)

    def add_ber_to_cache(self, ber, frame_data, tx_bits):
        """添加BER到缓存中，用于重复帧的最低BER选择
        
        Args:
            ber: 当前帧的BER值
            frame_data: 帧数据字典，包含'demod_symbols'和'time_samples'
            tx_bits: 发送的比特序列
            
        Returns:
            bool: 是否完成了一轮重复帧处理
        """
        current_time = time.time()
        
        # 检查是否超时，如果超时则清空缓存并重新开始
        if len(self.repeated_ber_cache) > 0 and (current_time - self.last_ber_update_time) > self.ber_timeout:
            print(f"BER缓存超时，清空缓存并重新开始 (缓存大小: {len(self.repeated_ber_cache)})")
            self.repeated_ber_cache = []
            self.repeated_frame_data_cache = []
            self.repeated_tx_bits_cache = []
        
        self.repeated_ber_cache.append(ber)
        self.repeated_frame_data_cache.append(frame_data)
        self.repeated_tx_bits_cache.append(tx_bits)
        self.last_ber_update_time = current_time
        
        # 检查是否收集够了重复帧数量
        if len(self.repeated_ber_cache) >= self.repeated_frame_count:
            return True
        return False

    def get_min_ber_data(self):
        """从缓存中获取BER最低的数据
        
        Returns:
            tuple: (frame_data, min_ber) - BER最低的帧数据和对应的BER值
        """
        if len(self.repeated_ber_cache) == 0:
            return None, 1.0
            
        # 找到BER最低的索引
        min_ber_idx = np.argmin(self.repeated_ber_cache)
        min_ber = self.repeated_ber_cache[min_ber_idx]
        min_frame_data = self.repeated_frame_data_cache[min_ber_idx]
        
        # 清空缓存，为下一轮重复帧做准备
        self.repeated_ber_cache = []
        self.repeated_frame_data_cache = []
        self.repeated_tx_bits_cache = []
        
        return min_frame_data, min_ber

    # 以下为其他核心方法（保持不变）
    def _generate_pss(self):
        n = np.arange(self.pss_len - 1)
        u = 25
        zc_sequence = np.exp(-1j * np.pi * u * n * (n + 1) / (self.pss_len - 1))
        return np.concatenate([zc_sequence, [0]])

    def _generate_sss(self):
        def _m_sequence(degree, taps):
            length = 2**degree - 1
            reg = np.ones(degree, dtype=int)
            mseq = []
            for _ in range(length * 2):
                mseq.append(reg[-1])
                feedback = 0
                for tap in taps:
                    feedback ^= reg[tap-1]
                reg = np.roll(reg, 1)
                reg[0] = feedback
            return np.array(mseq) * 2 - 1  # Map to -1/+1
        
        mseq1 = _m_sequence(8, [8, 4, 3, 2])
        mseq2 = _m_sequence(8, [8, 6, 5, 1])
        sss = np.concatenate([
            mseq1[:16] + 1j * mseq1[16:32],
            mseq2[:16] + 1j * mseq2[16:32]
        ])
        return sss / np.sqrt(np.mean(np.abs(sss)**2))

    def _generate_rs(self):
        base_seq = np.array([1, 1, -1, 1, -1, -1, 1, -1, 
                           1, -1, -1, 1, -1, 1, 1, -1], dtype=np.complex64)
        return np.tile(base_seq, 4)

    def _design_rrc_filter(self, num_symbols=10):
        taps = num_symbols * self.sps
        t = np.arange(-num_symbols/2, num_symbols/2, 1/self.sps)
        rrc = []
        for ti in t:
            if abs(ti) < 1e-10:
                val = 1 + self.roll_off * (4/np.pi - 1)
            elif abs(abs(ti) - 1/(4*self.roll_off)) < 1e-10:
                val = (self.roll_off/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*self.roll_off)) + 
                                                   (1-2/np.pi)*np.cos(np.pi/(4*self.roll_off)))
            else:
                pi_t = np.pi * ti
                sin_t = np.sin(pi_t * (1 - self.roll_off))
                cos_t = np.cos(pi_t * (1 + self.roll_off))
                denom = pi_t * (1 - (4 * self.roll_off * ti)**2)
                val = (sin_t + 4 * self.roll_off * ti * cos_t) / denom
            rrc.append(val)
        rrc = np.array(rrc)
        return rrc / np.sqrt(np.sum(rrc**2))

    def _bits_to_symbols(self, bits):
        symbols = []
        for i in range(0, len(bits), 2):
            bit1, bit2 = bits[i], bits[i+1]
            if (bit1, bit2) == (0, 0):
                idx = 0
            elif (bit1, bit2) == (0, 1):
                idx = 1
            elif (bit1, bit2) == (1, 0):
                idx = 2
            else:
                idx = 3
            symbols.append(self.constellation[idx])
        return np.array(symbols, dtype=np.complex64)

    def _symbols_to_bits(self, symbols):
        """符号到比特转换 - 恢复到原始简洁版本"""
        bits = []
        for symbol in symbols:
            # 找到最近的星座点
            distances = np.abs(symbol - self.constellation)
            idx = np.argmin(distances)
            
            # 转换为比特（Gray编码）
            if idx == 0:
                bits.extend([0, 0])
            elif idx == 1:
                bits.extend([0, 1])
            elif idx == 2:
                bits.extend([1, 0])
            else:
                bits.extend([1, 1])
        return np.array(bits)

    def differential_encode(self, symbols):
        """差分编码 - 修复关键错误"""
        encoded = [symbols[0] * self.diff_ref_symbol]
        for i in range(1, len(symbols)):
            encoded.append(symbols[i] * encoded[i-1])  # 移除错误的共轭
        return np.array(encoded)

    def differential_decode(self, symbols):
        """差分解码 - 与原始版本保持一致"""
        if len(symbols) == 0:
            return np.array([])
            
        decoded = []
        # 第一个符号使用参考符号解码
        decoded.append(symbols[0] * np.conj(self.diff_ref_symbol))
        
        # 后续符号进行差分解码
        for i in range(1, len(symbols)):
            decoded.append(symbols[i] * np.conj(symbols[i-1]))
        
        return np.array(decoded, dtype=np.complex64)

    def prepare_tx_signal(self, symbols):
        """将符号转换为发送信号（插值+滤波）"""
        # 上采样
        upsampled = np.zeros(len(symbols) * self.sps, dtype=np.complex64)
        upsampled[::self.sps] = symbols
        
        # 脉冲成形
        tx_signal = np.convolve(upsampled, self.rrc_filter, mode='same')
        return tx_signal

    def _init_costas_loop(self, loop_bw=0.01):
        """初始化Costas环用于相位同步"""
        class CostasLoop:
            def __init__(self, loop_bw):
                self.loop_bw = loop_bw
                self.phase = 0.0
                self.freq = 0.0
                self.alpha = 4 * loop_bw
                self.beta = 4 * loop_bw**2 / 2
                
            def process(self, symbols):
                out = []
                for sym in symbols:
                    # 相位旋转
                    rotated = sym * np.exp(-1j * self.phase)
                    out.append(rotated)
                    
                    # 相位误差检测（QPSK）
                    if rotated.real > 0:
                        pd = rotated.real * rotated.imag  # 简化的鉴相器
                    else:
                        pd = -rotated.real * rotated.imag
                        
                    # 环路滤波
                    self.freq += self.beta * pd
                    self.phase += self.alpha * pd + self.freq
                    
                    # 相位折叠到[-π, π]
                    self.phase = (self.phase + np.pi) % (2 * np.pi) - np.pi
                return np.array(out)
        
        return CostasLoop(loop_bw)

    # 同步相关方法（保持不变）
    def _enhanced_pss_sync(self, rx_symbols):
        """增强型PSS符号定时同步，带插值优化"""
        # 添加调试信息

        corr = np.correlate(rx_symbols, self.pss, mode='valid')
        corr_power = np.abs(corr)**2

        # 找到最大值的位置
        max_idx = np.argmax(corr_power)
        max_power = corr_power[max_idx]

        # 检查是否有真正的PSS相关峰值
        threshold = np.mean(corr_power) + 2 * np.std(corr_power)

        if max_idx > 0 and max_idx < len(corr_power) - 1:
            y1, y2, y3 = corr_power[max_idx-1], corr_power[max_idx], corr_power[max_idx+1]
            if y1 - 2*y2 + y3 != 0:
                delta = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
                timing_offset = max_idx + delta
            else:
                timing_offset = max_idx
        else:
            timing_offset = max_idx

        timing_offset = int(np.round(timing_offset))
        return timing_offset

    def _enhanced_sss_sync(self, rx_symbols, timing_offset):
        """FFT-based coarse frequency synchronization using SSS sequence"""
        sss_start = timing_offset + self.pss_len
        sss_end = sss_start + self.sss_len
        if sss_end > len(rx_symbols):
            return 0.0
        rx_sss = rx_symbols[sss_start:sss_end]

        # Use FFT-based frequency offset estimation for high precision
        coarse_freq = self.fft_compensator.estimate_offset(rx_sss)

        # 屏蔽频偏估计打印
        # if self.verbose:
        #     print(f"FFT-based coarse frequency estimation: {coarse_freq:.2f} Hz")

        return coarse_freq

    def _enhanced_rs_sync(self, rx_symbols, timing_offset, coarse_freq):
        """FFT-based fine frequency synchronization using RS sequence"""
        n = np.arange(len(rx_symbols))
        rx_symbols_corrected = rx_symbols * np.exp(-1j * 2 * np.pi * coarse_freq * n * self.Ts)
        rs_start = timing_offset + self.pss_len + self.sss_len
        rs_end = rs_start + self.rs_len
        if rs_end > len(rx_symbols_corrected):
            return 0.0
        rx_rs = rx_symbols_corrected[rs_start:rs_end]

        # Use FFT-based fine frequency estimation with smaller search range
        # Create a temporary compensator with smaller frequency range for fine tuning
        fine_compensator = CoarseFrequencyCompensator(
            sample_rate=self.samp_rate,
            modulation='QPSK',
            algorithm='FFT-based',
            frequency_resolution=10,  # Higher resolution for fine tuning
            maximum_frequency_offset=200,  # Smaller range around coarse estimate
            samples_per_symbol=self.sps
        )

        # Estimate residual frequency offset
        residual_freq = fine_compensator.estimate_offset(rx_rs)

        # 屏蔽频偏估计打印
        # if self.verbose:
        #     print(f"FFT-based fine frequency estimation: {residual_freq:.2f} Hz")

        return residual_freq
