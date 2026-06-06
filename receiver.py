#!/usr/bin/env python3
"""
receiver.py — BPSK PHY 接收端 (完整三级同步链)

同步方案:
  ① STF 延迟相关 → 粗包检测 + 粗 CFO (对 CFO 不敏感)
  ② PSS 互相关 → 精定时 + 峰值质量判据 (peak_to_mean, peak_to_second)
  ③ RS 线性相位拟合 → 细 CFO + 公共相位 + 信道幅度 + 噪声方差

帧结构 (符号域):
  STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)

窗口管理:
  滑动窗口, 处理完一帧后只消费对应样本, 不清空全部.

用法:
  仿真:  python receiver.py --mode sim --sim-file rx_iq.npy
  硬件:  python receiver.py --mode hardware --freq 915e6 --gain 30
  保存IQ: python receiver.py --mode hardware --save-iq capture.npy
"""
from __future__ import annotations

import argparse, os, sys, time
from typing import Optional, Tuple
import numpy as np

from phy_params import (
    SPS, TS, STF, PSS, RS, RRC, STF_REP, STF_LEN, PSS_LEN, RS_LEN,
    HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN, GUARD_SYMBOLS,
    FRAME_SYMBOLS, STF_DELAY, RRC_DELAY_SAMPLES,
    STF_THRESHOLD, STF_MIN_ENERGY,
    PSS_PEAK_TO_MEAN_THR, PSS_PEAK_TO_SECOND_THR,
    PSS_SEARCH_WIN_SAMPLES, ADVANCE_SAMPLES,
    FRAME_RRC_SAMPLES, MIN_WIN_SAMPLES,
    crc16, crc16_check, bits_to_bytes, bytes_to_bits,
)


# ======================================================================
# PHY 处理函数
# ======================================================================

# --- 参考序列 (符号域) ---
_REF_PSS = PSS.astype(np.complex64)     # (64,)
_REF_RS  = RS.astype(np.complex64)      # (32,)
_RRC     = RRC                          # (21,)

# --- 帧样本数 ---
FRAME_SAMPLES = FRAME_SYMBOLS * SPS

# --- 旁函数 ---
def _rrc_match_conj(samples: np.ndarray, rrc: np.ndarray) -> np.ndarray:
    """RRC 匹配滤波: 用 rrc[::-1] 卷积 + 抽取到符号率."""
    filt = np.convolve(samples, rrc[::-1], mode='full')
    delay = (len(rrc) - 1) // 2
    return filt[delay::SPS].astype(np.complex64)


def _stf_delay_correlation(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """STF 延迟相关: 包检测 + 粗 CFO.

    P(d) = sum_{n=0}^{L-1} r[d+n] * conj(r[d+n+L])
    E(d) = sum_{n=0}^{L-1} |r[d+n+L]|^2
    M(d) = |P(d)| / E(d)

    Args:
        samples: (N,) complex64 时域样本
    Returns:
        metric: (N - L,) float32, 归一化检测度量 M(d) ∈ [0, 1]
        P: (N - L,) complex64, 延迟相关值 (用于粗 CFO)
    """
    L = STF_DELAY  # 32 samples
    N = len(samples)
    if N <= L:
        return np.array([], dtype=np.float32), np.array([], dtype=np.complex64)

    r0 = samples[:N - L]
    rL = samples[L:]

    prod = r0 * np.conj(rL)
    ones = np.ones(L, dtype=np.float32)
    P = np.convolve(prod, ones, mode='valid')

    energy = (rL.real ** 2 + rL.imag ** 2).astype(np.float32)
    E = np.convolve(energy, ones, mode='valid')

    noise_floor = 1e-6 * L
    metric = np.abs(P) / (E + noise_floor)

    return metric.astype(np.float32), P


def _compute_coarse_cfo(P_peak: complex, L_samples: int, samp_rate: float) -> float:
    """从 STF 延迟相关峰值计算粗 CFO.

    P(d) = Σ r[n]·conj(r[n+L]), 相位 = -2πΔf·L·Ts
    故: Δf = -angle(P) / (2π·L·Ts)
    """
    phase = np.angle(P_peak)
    return -phase / (2 * np.pi * L_samples / samp_rate)


def _pss_correlation(symbols: np.ndarray) -> Tuple[np.ndarray, int, float, float]:
    """PSS 互相关: 精定时 + 质量判据.

    Returns:
        corr_mag: 相关幅度序列
        peak_idx: 峰值位置 (符号索引)
        peak_to_mean: 峰值/均值比
        peak_to_second: 峰值/次大峰比
    """
    if len(symbols) < PSS_LEN:
        return np.array([], dtype=np.float32), 0, 0.0, 0.0

    pss_conj_rev = np.conj(_REF_PSS[::-1])
    corr = np.convolve(symbols, pss_conj_rev, mode='valid')
    corr_mag = np.abs(corr)

    if len(corr_mag) < 2:
        return corr_mag.astype(np.float32), 0, 0.0, 0.0

    peak_idx = int(np.argmax(corr_mag))
    peak_val = corr_mag[peak_idx]

    mean_val = np.mean(corr_mag)
    peak_to_mean = peak_val / (mean_val + 1e-30)

    sorted_idx = np.argsort(corr_mag)[::-1]
    if len(sorted_idx) >= 2:
        second_val = 0
        for idx in sorted_idx[1:]:
            if abs(idx - peak_idx) > PSS_LEN // 2:
                second_val = corr_mag[idx]
                break
        peak_to_second = peak_val / (second_val + 1e-30)
    else:
        peak_to_second = peak_to_mean

    return corr_mag.astype(np.float32), peak_idx, float(peak_to_mean), float(peak_to_second)


def _rs_fine_cfo(symbols: np.ndarray, rs_pos: int,
                 coarse_cfo: float = 0.0) -> Tuple[float, float]:
    """RS 线性相位拟合: 粗CFO预补偿后估计残余细CFO.

    Args:
        coarse_cfo: STF粗CFO (Hz), 先补偿再拟合, 大幅提升大频偏下精度
    Returns:
        fine_cfo: 残余细 CFO (Hz)
        rs_corr: RS 相关幅度 (用于质量判定)
    """
    if rs_pos + RS_LEN > len(symbols):
        return 0.0, 0.0

    rs_seg = symbols[rs_pos:rs_pos + RS_LEN]

    # 粗 CFO 预补偿: 消除大频偏, 让 unwrap 和线性拟合工作在残余小频偏上
    if abs(coarse_cfo) > 0:
        n_rs = np.arange(RS_LEN)
        pre_comp = np.exp(-1j * 2 * np.pi * coarse_cfo * (rs_pos + n_rs) * TS)
        rs_seg = rs_seg * pre_comp

    rs_tone = rs_seg * np.conj(_REF_RS)
    rs_corr = float(np.abs(np.sum(rs_tone)))

    rs_phase = np.unwrap(np.angle(rs_tone))
    n = np.arange(RS_LEN, dtype=np.float64)
    n_mean = np.mean(n)
    phase_mean = np.mean(rs_phase)

    num = np.sum((n - n_mean) * (rs_phase - phase_mean))
    den = np.sum((n - n_mean) ** 2)
    slope = num / (den + 1e-30)

    residual_cfo = slope / (2 * np.pi * TS)
    return float(residual_cfo), rs_corr


def _rs_channel_estimate(symbols: np.ndarray, rs_pos: int,
                         fine_cfo: float,
                         coarse_cfo: float = 0.0) -> Tuple[complex, float, float]:
    """RS 信道/相位/噪声估计 (粗CFO+细CFO联合补偿).

    Returns:
        h: 复信道增益 (含公共相位)
        phase_est: 公共相位 (rad)
        sigma2: 噪声方差
    """
    if rs_pos + RS_LEN > len(symbols):
        return 1.0 + 0j, 0.0, 0.1

    rs_seg = symbols[rs_pos:rs_pos + RS_LEN]
    n_rs = np.arange(RS_LEN)
    total_cfo = coarse_cfo + fine_cfo
    rs_corrected = rs_seg * np.exp(-1j * 2 * np.pi * total_cfo * (rs_pos + n_rs) * TS)

    h = np.mean(rs_corrected * np.conj(_REF_RS))
    if abs(h) < 1e-30:
        return 1.0 + 0j, 0.0, 0.1

    rs_eq = rs_corrected / h
    # Welch 校正: 32符号LS估计, 噪声方差偏小 ~1/32, 乘 N/(N-1) 修正
    noise = rs_eq - _REF_RS
    sigma2 = float(np.var(noise)) * (RS_LEN / (RS_LEN - 1))
    phase_est = float(np.angle(h))

    return h, phase_est, max(sigma2, 1e-30)


def _demod_llr(symbols: np.ndarray, data_start: int, data_len: int,
               h: complex, phase_est: float, fine_cfo: float,
               sigma2: float, coarse_cfo: float = 0.0) -> np.ndarray:
    """BPSK 解调: 粗CFO+细CFO联合补偿 → 相位校正 → 信道均衡 → LLR.

    y = data_syms * exp(-j*2π*Δf_total*t) * exp(-j*θ) / h
    llr = 2 * real(y) / σ²
    """
    if data_start + data_len > len(symbols):
        return np.zeros(data_len, dtype=np.float32)

    data_seg = symbols[data_start:data_start + data_len]
    n = np.arange(data_len)
    total_cfo = coarse_cfo + fine_cfo
    cfo_comp = np.exp(-1j * 2 * np.pi * total_cfo * (data_start + n) * TS)

    y = data_seg * cfo_comp
    y = y * np.exp(-1j * phase_est)
    if abs(h) > 1e-30:
        y = y / h

    llr = (2.0 * y.real.astype(np.float32)) / (sigma2 + 1e-30)
    llr = np.clip(llr, -50.0, 50.0)
    return llr


def _estimate_snr(symbols: np.ndarray, rs_pos: int,
                  fine_cfo: float, h: complex,
                  coarse_cfo: float = 0.0) -> Tuple[float, float]:
    """从 RS 估计 SNR 和 EVM."""
    _, _, sigma2 = _rs_channel_estimate(symbols, rs_pos, fine_cfo, coarse_cfo)
    signal_power = abs(h) ** 2
    if sigma2 > 0:
        snr = signal_power / sigma2
        snr_db = 10 * np.log10(max(snr, 1e-30))
        evm_db = 10 * np.log10(max(sigma2 / (signal_power + 1e-30), 1e-30))
    else:
        snr_db = 50.0
        evm_db = -50.0
    return float(snr_db), float(evm_db)


def _crc_bits_to_int(crc_bits: np.ndarray) -> int:
    """16 BPSK hard bits → uint16."""
    val = 0
    for i in range(16):
        val = (val << 1) | int(crc_bits[i])
    return val


def _verify_header(header_bits: np.ndarray) -> bool:
    """验证 Header CRC."""
    if len(header_bits) < HEADER_LEN:
        return False
    reserved = header_bits[:16]
    crc_bits = header_bits[16:32]
    expected_crc = _crc_bits_to_int(crc_bits)
    header_bytes = bits_to_bytes(reserved)
    return crc16_check(header_bytes, expected_crc)


def _verify_payload_crc(payload_bits: np.ndarray, crc_bits: np.ndarray) -> bool:
    """验证 Payload CRC."""
    payload_bytes = bits_to_bytes(payload_bits)
    expected_crc = _crc_bits_to_int(crc_bits)
    return crc16_check(payload_bytes, expected_crc)


# ======================================================================
# Receiver 类
# ======================================================================

class BpskPhyReceiver:
    """BPSK PHY 接收端 (完整三级同步链)."""

    def __init__(self, samp_rate: float = 1e6, sps: int = SPS):
        self.samp_rate = samp_rate
        self.sps = sps
        self.Ts = 1.0 / samp_rate
        self.Ts_sym = sps / samp_rate

        # --- 环形缓冲 ---
        self.buf_cap = int(samp_rate)  # 1 秒缓冲
        self.buf = np.zeros(self.buf_cap, dtype=np.complex64)
        self.buf_len = 0

        # --- 统计 ---
        self.total_frames = 0
        self.crc_pass = 0
        self.header_crc_pass = 0
        self.detection_attempts = 0
        self.false_alarms = 0
        self.total_bits = 0
        self.total_errors = 0
        self.overflow_count = 0

        # --- 连续帧跟踪 ---
        self._expected_frame_start = -1

        # --- IQ 保存 ---
        self._iq_buffer = None
        self._save_iq_path = ""

        # --- 帧历史 ---
        self.tx_ref_bits = None

        # --- 运行时状态 ---
        self.running = False
        self.usrp = None
        self.rx_stream = None

    # ================================================================
    # 公共接口
    # ================================================================

    def start(self, mode: str = 'sim', freq: float = 915e6, gain: float = 30,
              sim_file: str = 'rx_iq.npy', tx_bits_file: str = '',
              snr_db: float = 15.0, usrp_args: str = '', subdev: str = 'A:A',
              save_iq: str = ''):
        """启动接收."""
        self.running = True

        if tx_bits_file and os.path.isfile(tx_bits_file):
            self.tx_ref_bits = np.load(tx_bits_file)
            print(f"[receiver] 加载 {len(self.tx_ref_bits)} 参考比特")

        if save_iq:
            self._save_iq_path = save_iq
            self._iq_buffer = []

        if mode == 'hardware':
            self._init_usrp(freq, gain, usrp_args, subdev)
            self._rx_loop_hardware()
        else:
            self._rx_loop_sim(sim_file)

    def stop(self):
        self.running = False

    # ================================================================
    # 接收处理核心
    # ================================================================

    def _process_samples(self, new_samples: np.ndarray, tx_bits=None):
        """添加新样本到缓冲, 然后尝试检测和处理帧."""
        self._append_to_buf(new_samples)
        self._detect_and_demod()

    def _append_to_buf(self, samples: np.ndarray):
        """追加样本到环形缓冲."""
        n = len(samples)
        if n > self.buf_cap - self.buf_len:
            discard = self.buf_len - self.buf_cap // 2
            if discard > 0:
                self.buf[:self.buf_len - discard] = self.buf[discard:self.buf_len]
                self.buf_len -= discard
                if self._expected_frame_start >= 0:
                    self._expected_frame_start = max(0, self._expected_frame_start - discard)
        space = min(n, self.buf_cap - self.buf_len)
        self.buf[self.buf_len:self.buf_len + space] = samples[:space]
        self.buf_len += space

    def _detect_and_demod(self):
        """检测 → 同步 → 解调 → 消费窗口."""
        r = self.buf[:self.buf_len]

        while self.running and self.buf_len >= MIN_WIN_SAMPLES:
            self.detection_attempts += 1

            # ===== 阶段 1: STF 延迟相关 → 候选位置 =====
            metric, P = _stf_delay_correlation(r)
            if len(metric) == 0:
                break

            candidates = []
            for d in range(len(metric)):
                if metric[d] > STF_THRESHOLD:
                    local_E = np.sum(np.abs(r[d + STF_DELAY:d + 2 * STF_DELAY]) ** 2)
                    if local_E > STF_MIN_ENERGY:
                        candidates.append(d)

            if not candidates:
                self._advance_window(ADVANCE_SAMPLES)
                self._expected_frame_start = -1
                continue

            frame_found = False
            for candidate_d in candidates[:32]:
                coarse_sample_pos = candidate_d

                if candidate_d < len(P):
                    coarse_cfo = _compute_coarse_cfo(
                        P[candidate_d], STF_DELAY, self.samp_rate)
                else:
                    coarse_cfo = 0.0

                # ===== 阶段 2: PSS 精定时验证 =====
                margin = PSS_SEARCH_WIN_SAMPLES
                extract_start = max(0, coarse_sample_pos - margin)
                extract_end = min(self.buf_len,
                                  coarse_sample_pos + margin + FRAME_RRC_SAMPLES)
                chunk = r[extract_start:extract_end]

                symbols = _rrc_match_conj(chunk, _RRC)
                if len(symbols) < PSS_LEN + RS_LEN:
                    continue

                _, pss_peak, ptm, pts = _pss_correlation(symbols)

                if ptm < PSS_PEAK_TO_MEAN_THR or pts < PSS_PEAK_TO_SECOND_THR:
                    continue

                pss_start = pss_peak
                frame_sym_start = pss_start - STF_LEN
                if frame_sym_start < 0:
                    continue

                rrc_delay = (len(_RRC) - 1) // 2
                frame_sample_start = (extract_start
                                      + frame_sym_start * self.sps
                                      - rrc_delay)
                if frame_sample_start < 0:
                    continue

                # ===== 阶段 3: RS 细 CFO + 相位 + 信道 =====
                rs_sym_start = frame_sym_start + STF_LEN + PSS_LEN
                fine_cfo, rs_corr = _rs_fine_cfo(symbols, rs_sym_start, coarse_cfo)

                if rs_corr < RS_LEN * 0.3:
                    continue

                h, phase_est, sigma2 = _rs_channel_estimate(
                    symbols, rs_sym_start, fine_cfo, coarse_cfo)

                # ===== 阶段 4: 解调 + CRC =====
                hdr_start = frame_sym_start + STF_LEN + PSS_LEN + RS_LEN
                hdr_llr = _demod_llr(symbols, hdr_start, HEADER_LEN,
                                     h, phase_est, fine_cfo, sigma2, coarse_cfo)
                hdr_bits = (hdr_llr < 0).astype(np.int64)
                hdr_ok = _verify_header(hdr_bits)

                pay_start = hdr_start + HEADER_LEN
                pay_llr = _demod_llr(symbols, pay_start,
                                     PAYLOAD_LEN + PAYLOAD_CRC_LEN,
                                     h, phase_est, fine_cfo, sigma2, coarse_cfo)
                pay_bits = (pay_llr < 0).astype(np.int64)
                payload_bits = pay_bits[:PAYLOAD_LEN]
                crc_bits = pay_bits[PAYLOAD_LEN:PAYLOAD_LEN + PAYLOAD_CRC_LEN]
                pay_crc_ok = _verify_payload_crc(payload_bits, crc_bits)

                # ===== 阶段 5: 指标 =====
                signal_power = abs(h) ** 2
                snr_db = float(10 * np.log10(max(signal_power / sigma2, 1e-30)))
                total_cfo = coarse_cfo + fine_cfo

                self.total_frames += 1
                if hdr_ok:
                    self.header_crc_pass += 1
                if pay_crc_ok:
                    self.crc_pass += 1

                if self.total_frames <= 5 or self.total_frames % 50 == 0:
                    print(
                        f"  frame={self.total_frames:5d}  "
                        f"ptm={ptm:.1f}  pts={pts:.1f}  "
                        f"cf0={coarse_cfo:+.0f}  cf1={fine_cfo:+.0f}  "
                        f"Δf={total_cfo:+.0f}Hz  "
                        f"θ={phase_est:.3f}rad  "
                        f"|h|={abs(h):.3f}  "
                        f"σ²={sigma2:.4f}  "
                        f"SNR={snr_db:.1f}dB  "
                        f"HDR={'OK' if hdr_ok else 'XX'}  "
                        f"CRC={'OK' if pay_crc_ok else 'XX'}",
                        flush=True)

                # ===== 阶段 6: BER 统计 (按帧序号对齐, 漏帧不导致错位) =====
                if self.tx_ref_bits is not None:
                    frame_idx = self.total_frames - 1  # 当前帧序号 (0-based)
                    start_idx = frame_idx * PAYLOAD_LEN
                    end_idx = start_idx + PAYLOAD_LEN
                    if end_idx <= len(self.tx_ref_bits):
                        ref = self.tx_ref_bits[start_idx:end_idx]
                        errs = int(np.sum(payload_bits != ref))
                        self.total_bits += PAYLOAD_LEN
                        self.total_errors += errs

                # ===== 阶段 7: 消费窗口 =====
                consume_end = frame_sample_start + FRAME_RRC_SAMPLES
                if consume_end > self.buf_len:
                    consume_end = self.buf_len
                self._consume(consume_end)
                self._expected_frame_start = -1
                frame_found = True
                break

            if not frame_found:
                self.false_alarms += len(candidates[:32])
                self._advance_window(ADVANCE_SAMPLES)
                self._expected_frame_start = -1

    def _advance_window(self, n_samples: int):
        consume = min(n_samples, self.buf_len)
        self._consume(consume)

    def _consume(self, end_idx: int):
        """消费 [0, end_idx) 的样本."""
        if end_idx <= 0:
            return
        if end_idx >= self.buf_len:
            self.buf_len = 0
        else:
            remaining = self.buf_len - end_idx
            self.buf[:remaining] = self.buf[end_idx:self.buf_len]
            self.buf_len = remaining
        if self._expected_frame_start >= 0:
            self._expected_frame_start = max(0, self._expected_frame_start - end_idx)

    def _process_window(self, tx_bits=None):
        """兼容旧接口: process_samples + detect_and_demod."""
        self._detect_and_demod()

    # ================================================================
    # 仿真模式
    # ================================================================

    def _rx_loop_sim(self, sim_file: str):
        """仿真模式: 从 .npy 文件读取 IQ."""
        if not os.path.isfile(sim_file):
            print(f"[receiver] 错误: 文件不存在 → {sim_file}")
            return

        mm = np.load(sim_file, mmap_mode='r')
        total = len(mm)
        pos = 0
        print(f"[receiver] 仿真: {sim_file}  ({total} 样本)")

        t0 = time.time()
        while pos < total and self.running:
            chunk_end = min(pos + int(self.samp_rate * 0.2), total)
            chunk = np.asarray(mm[pos:chunk_end])
            self._process_samples(chunk)
            pos = chunk_end

        elapsed = time.time() - t0
        self._print_summary(elapsed)
        self.running = False

    # ================================================================
    # 硬件模式
    # ================================================================

    def _init_usrp(self, freq: float, gain: float,
                   usrp_args: str = '', subdev: str = 'A:A'):
        import uhd
        self.usrp = uhd.usrp.MultiUSRP(usrp_args)
        if subdev:
            try:
                self.usrp.set_rx_subdev_spec(subdev)
            except Exception:
                pass
        actual_freq = self.usrp.set_rx_freq(uhd.types.TuneRequest(freq))
        actual_gain = self.usrp.set_rx_gain(gain)
        actual_rate = self.usrp.set_rx_rate(self.samp_rate)
        actual_bw   = self.usrp.set_rx_bandwidth(self.samp_rate)
        self.usrp.set_rx_antenna("RX2")
        self.usrp.set_clock_source("internal")
        self.usrp.set_time_source("internal")
        pc_ns = time.time_ns()
        tspec = uhd.types.TimeSpec(pc_ns // 1_000_000_000,
                                    (pc_ns % 1_000_000_000) / 1e9)
        self.usrp.set_time_now(tspec)
        args = uhd.usrp.StreamArgs('fc32', 'sc16')
        args.channels = [0]
        self.rx_stream = self.usrp.get_rx_stream(args)
        self.rx_stream.issue_stream_cmd(
            uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))
        print(f"[receiver] USRP RX: freq={actual_freq:.6e}Hz  "
              f"gain={actual_gain:.1f}dB  rate={actual_rate:.6e}  "
              f"bw={actual_bw:.6e}  subdev={subdev}")

    def _rx_loop_hardware(self):
        import uhd
        md = uhd.types.RXMetadata()
        uhd_buf = np.zeros((1, 4096), dtype=np.complex64)
        t_start = time.time()
        print(f"[receiver] 开始硬件接收...")
        try:
            while self.running:
                ns = self.rx_stream.recv(uhd_buf, md, timeout=0.5)
                if ns == 0:
                    continue
                if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
                    self.overflow_count += 1
                    continue

                chunk = uhd_buf[0, :ns].copy()
                self._process_samples(chunk)

                if self._iq_buffer is not None:
                    self._iq_buffer.append(chunk)
                    if len(self._iq_buffer) >= 500:
                        self._flush_iq()

        except KeyboardInterrupt:
            pass
        finally:
            self._flush_iq()
            elapsed = time.time() - t_start
            self._print_summary(elapsed)
            if hasattr(self, 'rx_stream') and self.rx_stream is not None:
                try:
                    self.rx_stream.issue_stream_cmd(
                        uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
                except Exception:
                    pass
            self.usrp = None
            self.rx_stream = None

    def _flush_iq(self):
        if self._iq_buffer and len(self._iq_buffer) > 0:
            data = np.concatenate(self._iq_buffer)
            mode = 'ab' if os.path.isfile(self._save_iq_path) else 'wb'
            with open(self._save_iq_path, mode) as f:
                np.save(f, data)
            print(f"[save_iq] 已保存 {len(data)} 样本", flush=True)
            self._iq_buffer = []

    # ================================================================
    # 统计输出
    # ================================================================

    def _print_summary(self, elapsed: float):
        print(f"\n{'='*60}")
        print(f"接收完成")
        print(f"  耗时: {elapsed:.1f}s")
        print(f"  检测尝试: {self.detection_attempts}")
        print(f"  检出帧数: {self.total_frames}")
        print(f"  Header CRC OK: {self.header_crc_pass}")
        print(f"  Payload CRC OK: {self.crc_pass}")
        print(f"  误检(虚假告警): {self.false_alarms}")
        print(f"  Overflow: {self.overflow_count}")
        if self.total_frames > 0:
            print(f"  CRC通过率: {self.crc_pass / self.total_frames * 100:.1f}%")
        if self.total_bits > 0:
            ber = self.total_errors / self.total_bits
            print(f"  BER={self.total_errors}/{self.total_bits}={ber:.2e}")
        print(f"{'='*60}")

    def get_stats(self) -> dict:
        """获取当前统计摘要."""
        return {
            'total_frames': self.total_frames,
            'crc_pass': self.crc_pass,
            'header_crc_pass': self.header_crc_pass,
            'false_alarms': self.false_alarms,
            'overflow': self.overflow_count,
            'detection_attempts': self.detection_attempts,
            'crc_rate': self.crc_pass / max(self.total_frames, 1),
            'total_bits': self.total_bits,
            'total_errors': self.total_errors,
        }


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description='BPSK PHY 接收端')
    p.add_argument('--mode', default='sim', choices=['hardware', 'sim'])
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--gain', type=float, default=30)
    p.add_argument('--rate', type=float, default=1e6)
    p.add_argument('--sim-file', default='rx_iq.npy')
    p.add_argument('--tx-bits', default='', help='参考发送比特 (.npy)')
    p.add_argument('--snr-db', type=float, default=15.0)
    p.add_argument('--usrp-args', default='')
    p.add_argument('--subdev', default='A:A', help='子设备 (A:A=TX/RX, A:B=RX2)')
    p.add_argument('--save-iq', default='', help='保存原始 IQ')
    args = p.parse_args()

    rx = BpskPhyReceiver(samp_rate=args.rate)
    rx.start(mode=args.mode, freq=args.freq, gain=args.gain,
             sim_file=args.sim_file, tx_bits_file=args.tx_bits,
             snr_db=args.snr_db, usrp_args=args.usrp_args,
             subdev=args.subdev, save_iq=args.save_iq)


if __name__ == '__main__':
    main()
