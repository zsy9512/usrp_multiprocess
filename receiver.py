#!/usr/bin/env python3
"""
receiver.py — BPSK PHY 接收端 (完全自包含)

帧结构 (无保护间隔):
  PSS(32) + RS(16) + Data(256) = 304 符号, RRC 成形

同步方案:
  ① 整个信号做 RRC 匹配滤波 → 符号率
  ② 符号域 PSS 互相关 → 找最强峰 → 定时
  ③ RS 线性相位拟合 → 频偏估计
  ④ 频偏校正 + LLR → 硬判决

用法:
  python receiver.py --mode sim --sim-file rx_iq.npy --tx-bits tx_bits.npy
  python receiver.py --mode hardware --freq 2.45e9 --gain 40
"""

import argparse, os, sys, time
from typing import Optional, Callable
import numpy as np

# ── 帧参数 (与 sender.py 严格一致: PSS+RS+Data+Guard) ──
PSS_LEN, RS_LEN, DATA_LEN = 32, 16, 256
GUARD_SYMBOLS = 32
FRAME_SYMBOLS = PSS_LEN + RS_LEN + DATA_LEN + GUARD_SYMBOLS  # 336

# ── PHY 函数 ──
def _ref_pss():
    n = np.arange(PSS_LEN)
    return np.exp(-1j * np.pi * 25 * n * (n + 1) / PSS_LEN).astype(np.complex64)

def _ref_rs():
    rng = np.random.RandomState(13)
    return (2 * rng.randint(0, 2, RS_LEN) - 1).astype(np.complex64)

def _design_rrc(sps=2, rolloff=0.35, num_sym=10):
    n_taps = num_sym * sps
    t = np.arange(-num_sym / 2, num_sym / 2, 1 / sps)
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 + rolloff * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * rolloff)) < 1e-12:
            h[i] = (rolloff / np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*rolloff)) + (1-2/np.pi)*np.cos(np.pi/(4*rolloff)))
        else:
            pi_t = np.pi * ti
            num = np.sin(pi_t*(1-rolloff)) + 4*rolloff*ti*np.cos(pi_t*(1+rolloff))
            den = pi_t*(1 - (4*rolloff*ti)**2)
            h[i] = num / den
    return (h / np.sqrt(np.sum(h**2))).astype(np.float32)

def _rrc_match(samples, rrc, sps):
    filt = np.convolve(samples, rrc, mode='full')
    delay = (len(rrc) - 1) // 2 + (sps // 2)
    return filt[delay::sps]

def _parabolic_interp(cp, peak):
    if peak <= 0 or peak >= len(cp) - 1: return 0.0
    y1, y2, y3 = cp[peak-1], cp[peak], cp[peak+1]
    d = y1 - 2*y2 + y3
    return 0.5*(y1-y3)/d if abs(d) > 1e-12 else 0.0

# ── 全局参考 ──
REF_PSS = _ref_pss()
REF_RS  = _ref_rs()
RRC_TX  = _design_rrc(2)

# ======================================================================
#  接收端
# ======================================================================
class BpskPhyReceiver:
    def __init__(self, samp_rate=1e6, sps=2, decoder_callback=None):
        self.samp_rate = samp_rate
        self.sps = sps
        self.Ts = sps / samp_rate
        self.decoder_callback = decoder_callback
        self.running = False
        self.win = np.zeros(1000000, dtype=np.complex64)
        self.win_len = 0
        self.win_cap = 1000000
        self.total_frames = 0
        self.total_bits = 0
        self.total_errors = 0
        self.overflow_count = 0
        self.ber_history = []
        self._last_pss = -1
        self._rs_adj_count = 0

    def start(self, mode='sim', freq=2.45e9, gain=40, sim_file='rx_iq.npy',
              usrp_args='', tx_bits_file='', snr_db=15.0, subdev='A:A',
              save_iq=''):
        self.running = True
        self.save_iq_path = save_iq
        self._iq_buffer = [] if save_iq else None
        tx_bits = np.load(tx_bits_file) if tx_bits_file and os.path.isfile(tx_bits_file) else None
        if mode == 'hardware':
            self._init_usrp(freq, gain, usrp_args, subdev)
            self._rx_loop_hardware(tx_bits)
        else:
            self._rx_loop_sim(sim_file, tx_bits)

    def stop(self):
        self.running = False

    # ── 接收核心 ──
    def _process_window(self, tx_bits):
        """处理窗口: RRC匹配 → RS同步 → PSS频偏 → 数据."""
        r = self.win[:self.win_len]
        min_samp = FRAME_SYMBOLS * self.sps
        if self.win_len < min_samp:
            return

        symbols = _rrc_match(r, RRC_TX, self.sps)

        # ── ① RS 滑动相关 (找帧) ──
        rs_corr = np.correlate(symbols, REF_RS, mode='valid')  # 复数, 保留相位
        rs_mag = np.abs(rs_corr)
        thr = np.percentile(rs_mag, 10) * 6  # 10分位噪声参考 ×6
        peaks = [i for i in range(1, len(rs_mag)-1)
                 if rs_mag[i] > thr and rs_mag[i] > rs_mag[i-1] and rs_mag[i] > rs_mag[i+1]]
        if not peaks:
            return

        if self._last_pss >= 0:
            expected = self._last_pss + FRAME_SYMBOLS
            p = None
            for cp in peaks:
                if abs(cp - expected - PSS_LEN) < 10:
                    p = cp - PSS_LEN
                    break
            if p is None:
                self._last_pss = -1
                return
        else:
            p = peaks[0] - PSS_LEN

        if p < 0 or p + PSS_LEN + RS_LEN + DATA_LEN > len(symbols):
            return

        # ── ② PSS 频偏估计 (32符号, 精度比RS的16符号高√2倍) ──
        pss_seg = symbols[p:p + PSS_LEN]
        pss_tone = pss_seg * np.conj(REF_PSS)
        pss_phase = np.unwrap(np.angle(pss_tone))
        nn32 = np.arange(PSS_LEN, dtype=np.float64)
        slope = (np.sum(nn32 * pss_phase) - np.mean(nn32) * np.sum(pss_phase)) / \
                (np.sum(nn32**2) - PSS_LEN * np.mean(nn32)**2)
        freq_est = slope / (2 * np.pi * self.Ts)

        # ── ③ RS 解决 π 相位模糊 + 质量门限 ──
        rs_seg = symbols[p + PSS_LEN:p + PSS_LEN + RS_LEN]
        rs_dot = np.dot(rs_seg, np.conj(REF_RS))  # 保留符号, 不用 abs
        rs_val = np.abs(rs_dot)
        if rs_val < 1.0:  # 低于1.0是噪声误检
            return
        if rs_dot.real < 0:
            # BPSK 相位翻转 180°, 翻转极性
            flip_phase = np.pi
        else:
            flip_phase = 0.0

        # ── ④ 数据提取 + 频偏校正 + 相位校正 ──
        data_start = p + PSS_LEN + RS_LEN
        if data_start + DATA_LEN > len(symbols):
            return
        n_data = np.arange(DATA_LEN)
        data_syms = symbols[data_start:data_start + DATA_LEN]
        correction = np.exp(-1j * (2 * np.pi * freq_est * (data_start + n_data) * self.Ts + flip_phase))
        data_corrected = data_syms * correction

        # ── ⑤ LLR + 硬判决 (幅度归一化) ──
        amp = max(np.std(data_corrected.real), 0.01)
        sigma_est = 0.5  # 匹配 SGNN 训练条件
        llr = (2.0 * data_corrected.real.astype(np.float32) / amp) / (sigma_est ** 2)
        rx_bits = (llr < 0).astype(np.int64)

        if self.total_frames < 10 or self.total_frames % 10 == 0:
            print(f"  frame={self.total_frames} "
                  f"Δf={freq_est:.0f}Hz rs={rs_val:.1f} "
                  f"std={data_corrected.real.std():.3f}")

        self.total_frames += 1
        self._last_pss = p

        frame_samps = FRAME_SYMBOLS * self.sps
        consume_to = min(self.win_len, p * self.sps + frame_samps)
        self.win_len -= consume_to
        if self.win_len > 0:
            self.win[:self.win_len] = self.win[consume_to:consume_to + self.win_len]

    # ── 仿真模式 ──
    def _rx_loop_sim(self, sim_file, tx_bits):
        if sim_file.endswith('.bin'):
            mm = np.fromfile(sim_file, dtype=np.complex64)
        else:
            mm = np.load(sim_file, mmap_mode='r')
        total = len(mm)
        pos = 0
        while pos < total and self.running:
            end = min(pos + 10000, total)
            # 追加到窗口
            n = end - pos
            if n > self.win_cap - self.win_len:
                self._compact_win()
            space = min(n, self.win_cap - self.win_len)
            self.win[self.win_len:self.win_len + space] = np.asarray(mm[pos:pos+space])
            self.win_len += space
            pos = end
            self._process_window(tx_bits)
        self._print_summary(0)

    def _init_usrp(self, freq, gain, usrp_args, subdev='A:A'):
        import uhd
        self.usrp = uhd.usrp.MultiUSRP(usrp_args)
        if subdev:
            try:
                self.usrp.set_rx_subdev_spec(subdev)  # UHD 4.8 接受字符串
                print(f"[receiver] subdev={subdev}")
            except Exception:
                pass  # 部分版本不支持, 使用默认
        self.usrp.set_rx_freq(uhd.types.TuneRequest(freq))
        self.usrp.set_rx_gain(gain)
        self.usrp.set_rx_rate(self.samp_rate)
        pc_ns = time.time_ns()
        tspec = uhd.types.TimeSpec(pc_ns // 1_000_000_000, (pc_ns % 1_000_000_000) / 1e9)
        self.usrp.set_time_now(tspec)
        self.usrp.set_clock_source('internal')
        self.usrp.set_time_source('internal')
        args = uhd.usrp.StreamArgs('fc32', 'sc16')
        args.channels = [0]
        self.rx_stream = self.usrp.get_rx_stream(args)
        self.rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))
        print(f"[receiver] USRP RX: {freq/1e6:.1f}MHz, gain={gain}dB")

    def _rx_loop_hardware(self, tx_bits):
        import uhd
        md = uhd.types.RXMetadata()
        uhd_buf = np.zeros((1, 4096), dtype=np.complex64)
        t_start = time.time()
        rx_count = 0
        try:
            while self.running:
                ns = self.rx_stream.recv(uhd_buf, md, timeout=0.1)
                if ns == 0: continue
                if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
                    self.overflow_count += 1
                    continue
                if ns > self.win_cap - self.win_len:
                    self._compact_win()
                self.win[self.win_len:self.win_len+ns] = uhd_buf[0, :ns]
                self.win_len += ns
                rx_count += 1
                # 保存原始IQ
                if self._iq_buffer is not None:
                    self._iq_buffer.append(uhd_buf[0, :ns].copy())
                    if len(self._iq_buffer) >= 2048:
                        self._flush_iq()
                # 每100个recv打印一次状态
                if rx_count % 100 == 0 and rx_count > 0:
                    elapsed = time.time() - t_start
                    print(f"  [rx] 累计{rx_count}包 win={self.win_len} "
                          f"O={self.overflow_count} t={elapsed:.1f}s", flush=True)
                # 只有窗口积累足够数据才处理 (每 ~200ms 一次)
                # TX帧间隔100ms, 200ms窗口保证至少抓到一帧
                if self.win_len >= 200000:
                    self._process_window(tx_bits)
                    self.win_len = 0  # 清空窗口,避免重复处理
        finally:
            self._flush_iq()
            self._print_summary(time.time() - t_start)
            if hasattr(self, 'rx_stream') and self.rx_stream:
                try: self.rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
                except: pass
            self.usrp = None
            self.rx_stream = None

    def _flush_iq(self):
        if self._iq_buffer:
            data = np.concatenate(self._iq_buffer)
            # 二进制追加写入 (避免 np.save 的覆盖问题)
            with open(self.save_iq_path, 'ab') as f:
                data.tofile(f)
            print(f"[save_iq] 已保存 {len(data)} 样本 → {self.save_iq_path}")
            self._iq_buffer = []

    def _compact_win(self):
        if self.win_len > self.win_cap // 2:
            shift = self.win_len - self.win_cap // 2
            self.win[:self.win_len-shift] = self.win[shift:self.win_len]
            self.win_len -= shift

    def _print_summary(self, elapsed):
        ber = self.total_errors / self.total_bits if self.total_bits > 0 else 0
        print(f"\n接收完成: {self.total_frames} 帧, "
              f"BER={self.total_errors}/{self.total_bits}={ber:.2e}, "
              f"溢出={self.overflow_count}")
        if self.ber_history:
            np.save('ber_history.npy', np.array(self.ber_history))

def main():
    p = argparse.ArgumentParser(description='BPSK PHY 接收端')
    p.add_argument('--mode', default='sim', choices=['hardware', 'sim'])
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--gain', type=float, default=30)
    p.add_argument('--rate', type=float, default=1e6)
    p.add_argument('--sim-file', default='rx_iq.npy')
    p.add_argument('--tx-bits', default='')
    p.add_argument('--snr-db', type=float, default=15.0)
    p.add_argument('--usrp-args', default='')
    p.add_argument('--subdev', default='A:A', help='子设备 (A:A=TX/RX, A:B=RX2)')
    p.add_argument('--save-iq', default='', help='保存原始IQ到文件 (.npy)')
    args = p.parse_args()
    rx = BpskPhyReceiver(samp_rate=args.rate)
    rx.start(mode=args.mode, freq=args.freq, gain=args.gain,
             sim_file=args.sim_file, tx_bits_file=args.tx_bits, snr_db=args.snr_db,
             usrp_args=args.usrp_args, subdev=args.subdev, save_iq=args.save_iq)

if __name__ == '__main__':
    main()
