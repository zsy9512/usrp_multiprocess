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
            print(tx_bits[:100])
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
            rx_filtered = np.convolve(rx_signal, self.rrc_filter, mode='full')
            print(f" rx_signal长度={len(rx_signal)}, filtered长度={len(self.rrc_filter)}, rrc_filter长度={len(self.rrc_filter)}")
            rx_symbols = rx_filtered[::self.sps]
            print(f"下采样后符号长度={len(rx_symbols)},sps={self.sps}")
            rrc_delay = (len(self.rrc_filter) - 1) // 2
            # 4. 同步
            timing_offset = self._enhanced_pss_sync(rx_symbols)
            #timing_offset -= rrc_delay // self.sps  # 右移补偿
            print(f" PSS同步偏移={timing_offset}")
            coarse_freq = self._enhanced_sss_sync(rx_symbols, timing_offset)
            print(f"SSS粗频估计={coarse_freq:.2f} Hz")
            fine_freq = self._enhanced_rs_sync(rx_symbols, timing_offset, coarse_freq)
            total_freq = coarse_freq + fine_freq
            print(f"RS细频估计={fine_freq:.2f} Hz, 总频偏={total_freq:.2f} Hz")
            n2 = np.arange(len(rx_symbols))
            rx_symbols_corr = rx_symbols * np.exp(-1j * 2 * np.pi * (coarse_freq+fine_freq) * n2 * self.Ts)
            # 5. 数据提取
            data_start = timing_offset + self.preamble_len
            data_end = data_start + self.data_symbols
            data_symbols = rx_symbols_corr[data_start:data_end]
            print(f"数据符号长度={len(data_symbols)}")
            # 6. 相位同步
            costas = self._init_costas_loop(loop_bw=0.001)
            synced = costas.process(data_symbols)
            # 7. 差分解码
            decoded = self.differential_decode(synced)
            rx_bits = self._symbols_to_bits(decoded)
            print(f"Frame length={len(tx_bits)}, decoded length={len(rx_bits)}")
            print("rx：",rx_bits[:100],"tx：",tx_bits[:100])
            min_len = min(len(tx_bits), len(rx_bits))
            ber = self._calculate_ber(tx_bits[:min_len], rx_bits[:min_len])
            print(f"Frame {frame_idx+1}/{n_frames}, BER: {ber:.6f}")
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
        self.Ts = 1.0 / self.symbol_rate  # Symbol period
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
        np.random.seed(42)
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

    def _calculate_ber(self, tx_bits, rx_bits):
        """计算误比特率"""
        if len(tx_bits) != len(rx_bits):
            if self.verbose:
                print(f"比特长度不匹配: 发送 {len(tx_bits)}, 接收 {len(rx_bits)}")
            return 0.0  # 长度不匹配时返回0（或根据需要调整）
        errors = np.sum(tx_bits != rx_bits)
        return errors / len(tx_bits)

    def plot_ber_statistics(self):
        """绘制BER统计图表（修复：处理数据不足情况）"""
        if len(self.ber_history) == 0:
            print("没有足够的BER数据用于绘图")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot BER per frame
        ax.plot(self.frame_numbers, self.ber_history, 'b-', alpha=0.7, label='BER per Frame')
        
        # 计算并绘制平均BER
        avg_ber = np.mean(self.ber_history)
        ax.axhline(y=avg_ber, color='r', linestyle='--', label=f'Average BER: {avg_ber:.6f}')
        
        # Chart settings
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.set_title('DQPSK Transmission BER Statistics')
        ax.set_yscale('log')  # BER适合用对数刻度
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        
        # 添加统计信息文本
        stats_text = (f'Total frames: {len(self.ber_history)}\n'
                f'Average BER: {avg_ber:.6f}\n'
                f'Min BER: {np.min(self.ber_history):.6f}\n'
                f'Max BER: {np.max(self.ber_history):.6f}')
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

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
        tx_signal = np.convolve(upsampled, self.rrc_filter, mode='full')
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
        corr = np.correlate(rx_symbols, self.pss, mode='valid')
        corr_power = np.abs(corr)**2
        max_idx = np.argmax(corr_power)
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
        sss_start = timing_offset + self.pss_len
        sss_end = sss_start + self.sss_len
        if sss_end > len(rx_symbols):
            return 0.0
        rx_sss = rx_symbols[sss_start:sss_end]
        # 方法1: 相位差法
        tx_phase = np.angle(self.sss)
        tx_phase_diff = np.diff(tx_phase)
        rx_phase = np.angle(rx_sss)
        rx_phase_diff = np.diff(rx_phase)
        freq_phase_diff = rx_phase_diff - tx_phase_diff
        freq_phase_diff = np.angle(np.exp(1j * freq_phase_diff))
        freq_est1 = np.mean(freq_phase_diff) / (2 * np.pi * self.Ts)
        # 方法2: 基于相关的频率估计
        freq_search = np.linspace(-10000/2, 10000/2, 500)
        corr_values = []
        for f_test in freq_search:
            n = np.arange(len(rx_sss))
            rx_corrected = rx_sss * np.exp(-1j * 2 * np.pi * f_test * n * self.Ts)
            corr = np.abs(np.sum(rx_corrected * np.conj(self.sss)))
            corr_values.append(corr)
        max_corr_idx = np.argmax(corr_values)
        freq_est2 = freq_search[max_corr_idx]
        coarse_freq = 0.3 * freq_est1 + 0.7 * freq_est2
        return coarse_freq

    def _enhanced_rs_sync(self, rx_symbols, timing_offset, coarse_freq):
        n = np.arange(len(rx_symbols))
        rx_symbols_corrected = rx_symbols * np.exp(-1j * 2 * np.pi * coarse_freq * n * self.Ts)
        rs_start = timing_offset + self.pss_len + self.sss_len
        rs_end = rs_start + self.rs_len
        if rs_end > len(rx_symbols_corrected):
            return 0.0
        rx_rs = rx_symbols_corrected[rs_start:rs_end]
        group_size = 16
        num_groups = self.rs_len // group_size
        phase_diffs = []
        for i in range(num_groups - 1):
            group1 = rx_rs[i*group_size:(i+1)*group_size]
            group2 = rx_rs[(i+1)*group_size:(i+2)*group_size]
            cross_corr = np.sum(group1 * np.conj(group2))
            phase_diff = np.angle(cross_corr)
            phase_diffs.append(phase_diff)
        avg_phase_diff = np.mean(phase_diffs)
        fine_freq1 = avg_phase_diff / (2 * np.pi * group_size * self.Ts)
        def freq_cost_function(f_test):
            n_rs = np.arange(len(rx_rs))
            rx_test = rx_rs * np.exp(-1j * 2 * np.pi * f_test * n_rs * self.Ts)
            corr = np.abs(np.sum(rx_test * np.conj(self.rs)))
            return -corr
        from scipy.optimize import minimize_scalar
        search_range = 1000
        result = minimize_scalar(freq_cost_function, 
                               bounds=(fine_freq1 - search_range, fine_freq1 + search_range),
                               method='bounded')
        fine_freq2 = result.x
        fine_freq = 0.6 * fine_freq1 + 0.4 * fine_freq2
        return fine_freq
