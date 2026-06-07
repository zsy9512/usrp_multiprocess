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
import multiprocessing as mp
from multiprocessing import shared_memory, Process, Event, Value
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


def _stf_cluster_peaks(metric: np.ndarray, P: np.ndarray,
                       samples: np.ndarray,
                       samp_rate: float) -> Tuple[list, list]:
    """STF 峰值聚类去重 (对齐 loopback_test 128-sample 窗).

    按 M 排序, 128-sample 窗内只保留最强峰.
    Returns:
        peaks: list[int] 聚类后峰位置
        cfos:  list[float] 对应粗 CFO
    """
    L = STF_DELAY
    N = len(samples)
    raw = []
    for d in range(len(metric)):
        if metric[d] > STF_THRESHOLD:
            local_E = np.sum(np.abs(samples[d + L:d + 2 * L]) ** 2)
            if local_E > STF_MIN_ENERGY:
                raw.append((d, metric[d], P[d]))
    if not raw:
        return [], []

    raw.sort(key=lambda x: x[1], reverse=True)
    peaks, cfos, used = [], [], set()
    for d, _, p in raw:
        if d in used:
            continue
        for dx in range(max(0, d - 128), min(len(metric), d + 128)):
            used.add(dx)
        peaks.append(d)
        cfos.append(_compute_coarse_cfo(p, L, samp_rate))
    return peaks, cfos


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
                 coarse_cfo: float = 0.0,
                 Ts_sym: float = None) -> Tuple[float, float]:
    """RS 线性相位拟合: 粗CFO预补偿后估计残余细CFO.

    Args:
        coarse_cfo: STF粗CFO (Hz), 先补偿再拟合, 大幅提升大频偏下精度
        Ts_sym:     符号时间 (s), 默认 TS*SPS = 2e-6
    Returns:
        fine_cfo: 残余细 CFO (Hz)
        rs_corr: RS 相关幅度 (用于质量判定)
    """
    if Ts_sym is None:
        Ts_sym = TS * SPS
    if rs_pos + RS_LEN > len(symbols):
        return 0.0, 0.0

    rs_seg = symbols[rs_pos:rs_pos + RS_LEN]

    # 粗 CFO 预补偿: 消除大频偏, 让 unwrap 和线性拟合工作在残余小频偏上
    if abs(coarse_cfo) > 0:
        n_rs = np.arange(RS_LEN)
        pre_comp = np.exp(-1j * 2 * np.pi * coarse_cfo * (rs_pos + n_rs) * Ts_sym)
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

    residual_cfo = slope / (2 * np.pi * Ts_sym)
    return float(residual_cfo), rs_corr


def _rs_channel_estimate(symbols: np.ndarray, rs_pos: int,
                         fine_cfo: float,
                         coarse_cfo: float = 0.0,
                         Ts_sym: float = None) -> Optional[Tuple[complex, float, float]]:
    """RS 信道/相位/噪声估计 (粗CFO+细CFO联合补偿).

    Returns:
        (h, phase_est, sigma2) or None if |h| < 1e-6
    """
    if Ts_sym is None:
        Ts_sym = TS * SPS
    if rs_pos + RS_LEN > len(symbols):
        return None

    rs_seg = symbols[rs_pos:rs_pos + RS_LEN]
    n_rs = np.arange(RS_LEN)
    total_cfo = coarse_cfo + fine_cfo
    rs_corrected = rs_seg * np.exp(-1j * 2 * np.pi * total_cfo * (rs_pos + n_rs) * Ts_sym)

    h = np.mean(rs_corrected * np.conj(_REF_RS))
    if abs(h) < 1e-6:
        return None

    rs_eq = rs_corrected / h
    # Welch 校正: 32符号LS估计, 噪声方差偏小 ~1/32, 乘 N/(N-1) 修正
    noise = rs_eq - _REF_RS
    sigma2 = float(np.var(noise)) * (RS_LEN / (RS_LEN - 1))
    phase_est = float(np.angle(h))

    return h, phase_est, max(sigma2, 1e-30)


def _bpsk_demod_hard(symbols: np.ndarray, data_start: int, data_len: int,
                     h: complex, total_cfo: float,
                     Ts_sym: float = None) -> np.ndarray:
    """BPSK 硬判决解调 (对齐 loopback_test _bpsk_demod).

    CFO全补偿 + 除以h (含相位校正) → sign(real).
    """
    if Ts_sym is None:
        Ts_sym = TS * SPS
    if data_start + data_len > len(symbols):
        return np.zeros(data_len, dtype=np.int64)

    seg = symbols[data_start:data_start + data_len]
    n = np.arange(data_len)
    cfo_comp = np.exp(-1j * 2 * np.pi * total_cfo * (data_start + n) * Ts_sym)
    y = seg * cfo_comp
    if abs(h) > 1e-30:
        y = y / h
    return (y.real < 0).astype(np.int64)


def _estimate_snr(symbols: np.ndarray, rs_pos: int,
                  fine_cfo: float, h: complex,
                  coarse_cfo: float = 0.0,
                  Ts_sym: float = None) -> Tuple[float, float]:
    """从 RS 估计 SNR 和 EVM."""
    _, _, sigma2 = _rs_channel_estimate(symbols, rs_pos, fine_cfo, coarse_cfo, Ts_sym)
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
    """验证 Header CRC (frame_id + CRC16)."""
    if len(header_bits) < HEADER_LEN:
        return False
    id_bits = header_bits[:16]
    crc_bits = header_bits[16:32]
    expected_crc = _crc_bits_to_int(crc_bits)
    id_bytes = bits_to_bytes(id_bits)
    return crc16_check(id_bytes, expected_crc)


def _verify_payload_crc(payload_bits: np.ndarray, crc_bits: np.ndarray) -> bool:
    """验证 Payload CRC."""
    payload_bytes = bits_to_bytes(payload_bits)
    expected_crc = _crc_bits_to_int(crc_bits)
    return crc16_check(payload_bytes, expected_crc)


# ======================================================================
# 共享内存收样子进程 (多进程零拷贝, 防止 FPGA overflow)
# ======================================================================

RING_CAP = 2_000_000  # 2M 样本共享内存环形缓冲


def _recv_proc(shm_name, serial, freq, gain, rate, subdev, sync_mode, settle_s,
               wr_count, has_data, running, ovf_count):
    """收样子进程: UHD recv() → shared_memory ring buffer (零拷贝, wr_count 单调递增)."""
    import uhd

    dev_args = f'serial={serial}' if serial else ''
    usrp = uhd.usrp.MultiUSRP(dev_args)
    if subdev:
        try: usrp.set_rx_subdev_spec(subdev)
        except: pass
    usrp.set_rx_freq(uhd.types.TuneRequest(freq))
    usrp.set_rx_gain(gain)
    usrp.set_rx_rate(rate)
    usrp.set_rx_bandwidth(rate)
    usrp.set_rx_antenna("RX2")

    if sync_mode in ('host', 'internal'):
        usrp.set_clock_source("internal"); usrp.set_time_source("internal")
    elif sync_mode == 'external_ref':
        usrp.set_clock_source("external"); usrp.set_time_source("internal")
        time.sleep(settle_s)
    pc_ns = time.time_ns()
    tspec = uhd.types.TimeSpec(pc_ns // 1_000_000_000, (pc_ns % 1_000_000_000) / 1e9)
    usrp.set_time_now(tspec)

    stream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
    stream_args.channels = [0]
    rx_stream = usrp.get_rx_stream(stream_args)
    rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))

    shm = shared_memory.SharedMemory(name=shm_name)
    ring = np.ndarray(RING_CAP, dtype=np.complex64, buffer=shm.buf)

    md = uhd.types.RXMetadata()
    buf = np.zeros((1, 4096), dtype=np.complex64)
    w = 0

    while running.value:
        ns = rx_stream.recv(buf, md, timeout=0.2)
        if ns == 0: continue
        if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
            ovf_count.value += 1; continue

        data = buf[0, :ns]
        end = w + ns
        if end <= RING_CAP:
            ring[w:end] = data
        else:
            n1 = RING_CAP - w
            ring[w:] = data[:n1]
            ring[:ns - n1] = data[n1:]
        w = end % RING_CAP
        wr_count.value += ns
        has_data.set()

    rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    shm.close()


# ======================================================================
# Receiver 类
# ======================================================================

class BpskPhyReceiver:
    """BPSK PHY 接收端 (完整三级同步链)."""

    def __init__(self, samp_rate: float = 1e6, sps: int = SPS,
                 stf_threshold: float = STF_THRESHOLD,
                 pss_ptm: float = 3.5,
                 pss_pts: float = PSS_PEAK_TO_SECOND_THR,
                 rs_corr_thr: float = 0.3):
        self.samp_rate = samp_rate
        self.sps = sps
        self.Ts = 1.0 / samp_rate
        self.Ts_sym = sps / samp_rate

        # --- 同步门限 ---
        self.stf_threshold = stf_threshold
        self.pss_ptm = pss_ptm
        self.pss_pts = pss_pts
        self.rs_corr_thr = rs_corr_thr

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
        self.overflow_count = 0

        # --- 连续帧跟踪 ---
        self._expected_frame_start = -1

        # --- IQ 保存 ---
        self._iq_buffer = None
        self._save_iq_path = ""

        # --- 帧历史 ---

        # --- 运行时状态 ---
        self.running = False
        self.usrp = None
        self.rx_stream = None

    # ================================================================
    # 公共接口
    # ================================================================

    def start(self, mode: str = 'sim', freq: float = 915e6, gain: float = 30,
              sim_file: str = 'rx_iq.npy',
              snr_db: float = 15.0, usrp_args: str = '', subdev: str = 'A:A',
              save_iq: str = '', sync_mode: str = 'host', settle_s: float = 1.0):
        """启动接收."""
        self.running = True

        if save_iq:
            self._save_iq_path = save_iq
            self._iq_buffer = []

        if mode == 'hardware':
            self._rx_loop_mp(freq, gain, usrp_args, subdev, sync_mode, settle_s)
        else:
            self._rx_loop_sim(sim_file)

    def stop(self):
        self.running = False

    # ================================================================
    # 接收处理核心
    # ================================================================

    def _process_samples(self, new_samples: np.ndarray):
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
        """检测 → 同步 → 解调 → 消费窗口 (对齐 loopback_test)."""
        r = self.buf[:self.buf_len]

        while self.running and self.buf_len >= MIN_WIN_SAMPLES:
            self.detection_attempts += 1

            # ===== 阶段 1: STF 延迟相关 + 峰值聚类 =====
            metric, P = _stf_delay_correlation(r)
            if len(metric) == 0:
                break

            peaks, cfos = _stf_cluster_peaks(metric, P, r, self.samp_rate)
            if not peaks:
                self._advance_window(ADVANCE_SAMPLES)
                continue

            frame_found = False
            for pi in range(min(8, len(peaks))):
                d = peaks[pi]
                coarse_cfo = cfos[pi]
                coarse = d  # 样本域 STF 检测位置

                # ===== 阶段 2: 提取窗口 + PSS 精定时 =====
                EXTRACT_EXTRA_ = 200  # 对齐 loopback_test
                extract_start = max(0, coarse - EXTRACT_EXTRA_)
                extract_end = min(self.buf_len,
                                  coarse + EXTRACT_EXTRA_ + FRAME_RRC_SAMPLES + EXTRACT_EXTRA_)
                chunk = r[extract_start:extract_end]
                if len(chunk) < PSS_LEN * self.sps:
                    continue

                symbols = _rrc_match_conj(chunk, _RRC)
                if len(symbols) < PSS_LEN + RS_LEN:
                    continue

                _, pss_peak, ptm, pts = _pss_correlation(symbols)

                if ptm < self.pss_ptm or pts < self.pss_pts:
                    continue

                fs = pss_peak - STF_LEN               # 帧起始符号索引
                if fs < 0:
                    continue

                rp = fs + STF_LEN + PSS_LEN           # RS 起始符号索引
                if rp + RS_LEN + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN > len(symbols):
                    continue

                # ===== 阶段 3: RS 细 CFO + 信道估计 =====
                fine_cfo, rs_corr = _rs_fine_cfo(symbols, rp, coarse_cfo, self.Ts_sym)
                if rs_corr < RS_LEN * self.rs_corr_thr:
                    continue
                if abs(fine_cfo) > 500:               # 细CFO超限 → 拒收
                    continue

                chan = _rs_channel_estimate(symbols, rp, fine_cfo, coarse_cfo, self.Ts_sym)
                if chan is None:
                    continue
                h, phase_est, sigma2 = chan
                sigma2 = min(sigma2, 0.5)

                # ===== 阶段 4: BPSK 硬判决解调 + CRC =====
                total_cfo = coarse_cfo + fine_cfo
                hdr_start = rp + RS_LEN
                hdr_bits = _bpsk_demod_hard(symbols, hdr_start, HEADER_LEN,
                                            h, total_cfo, self.Ts_sym)
                hdr_ok = _verify_header(hdr_bits)

                pay_start = hdr_start + HEADER_LEN
                pay_bits = _bpsk_demod_hard(symbols, pay_start,
                                            PAYLOAD_LEN + PAYLOAD_CRC_LEN,
                                            h, total_cfo, self.Ts_sym)
                payload_bits = pay_bits[:PAYLOAD_LEN]
                crc_bits = pay_bits[PAYLOAD_LEN:PAYLOAD_LEN + PAYLOAD_CRC_LEN]
                pay_crc_ok = _verify_payload_crc(payload_bits, crc_bits)

                # ===== 阶段 5: 指标 + 打印 =====
                hmag = abs(h)
                snr_db = float(10 * np.log10(max(hmag ** 2 / sigma2, 1e-30)))

                self.total_frames += 1
                if hdr_ok:
                    self.header_crc_pass += 1
                if pay_crc_ok:
                    self.crc_pass += 1

                if self.total_frames <= 5 or self.total_frames % 100 == 0:
                    print(
                        f"  frame={self.total_frames:5d}  "
                        f"ptm={ptm:.1f}  pts={pts:.1f}  "
                        f"\u0394f0={coarse_cfo:+.0f}  "
                        f"\u0394f1={fine_cfo:+.0f}  "
                        f"|h|={hmag:.3f}  SNR={snr_db:.1f}dB  "
                        f"HDR={'OK' if hdr_ok else 'XX'}  "
                        f"CRC={'OK' if pay_crc_ok else 'XX'}",
                        flush=True)

                # ===== 阶段 6: 消费窗口 =====
                consume_end = extract_start + fs * self.sps + FRAME_RRC_SAMPLES + 50
                if consume_end > self.buf_len:
                    consume_end = self.buf_len
                self._consume(consume_end)
                frame_found = True
                break

            if not frame_found:
                self.false_alarms += min(8, len(peaks))
                self._advance_window(ADVANCE_SAMPLES)

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

    def _process_window(self):
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

    def _rx_loop_mp(self, freq, gain, usrp_args, subdev, sync_mode, settle_s):
        """多进程零拷贝接收: 子进程recv → 共享内存 → 主进程PHY检测."""
        serial = ''
        if 'serial=' in usrp_args:
            serial = usrp_args.split('serial=')[1].split(',')[0]

        ctx = mp.get_context('spawn')
        shm = shared_memory.SharedMemory(create=True, size=RING_CAP * 8)
        ring = np.ndarray(RING_CAP, dtype=np.complex64, buffer=shm.buf)
        ring[:] = 0j

        wr_count = ctx.Value('Q', 0)
        has_data = ctx.Event()
        running = ctx.Value('i', 1)
        ovf_count = ctx.Value('i', 0)

        proc = ctx.Process(target=_recv_proc,
                            args=(shm.name, serial, freq, gain, self.samp_rate,
                                  subdev, sync_mode, settle_s,
                                  wr_count, has_data, running, ovf_count),
                            daemon=True)
        proc.start()
        print(f"[receiver] 收样 PID={proc.pid}  共享内存 {RING_CAP//1000}k 零拷贝")

        self.running = True
        rd_count = 0
        iq_buffer = []
        save_path = self._save_iq_path
        ovf_last = 0
        t0 = time.time()

        try:
            while proc.is_alive():
                has_data.wait(timeout=0.5)
                has_data.clear()
                wc = wr_count.value
                avail = wc - rd_count
                if avail <= 0: continue
                if avail > RING_CAP:
                    rd_count = wc - RING_CAP
                    avail = RING_CAP

                rd_count += avail

                # 分小块喂给 _process_samples (内部缓冲只有 1M, 大块会溢出)
                # 快速检幅: max < amp_thr 的纯噪声跳过, 不拷贝不处理
                CHUNK = 8192
                amp_thr = 0.002  # 远低于最弱可用信号, 仅滤纯底噪
                while avail > 0:
                    take = min(avail, CHUNK)
                    pos2 = (rd_count - avail) % RING_CAP
                    end2 = (pos2 + take) % RING_CAP
                    sign = False
                    if end2 > pos2:
                        sign = bool(np.any(np.abs(ring[pos2:pos2 + take]) > amp_thr))
                    else:
                        sign = (bool(np.any(np.abs(ring[pos2:]) > amp_thr)) or
                                bool(np.any(np.abs(ring[:end2]) > amp_thr)))
                    if sign:
                        if end2 > pos2:
                            c = ring[pos2:pos2 + take].copy()
                        else:
                            c = ring[pos2:].copy()
                            if end2 > 0:
                                c = np.concatenate([c, ring[:end2]])
                        self._process_samples(c)
                    avail -= take

                if ovf_count.value != ovf_last:
                    print(f"\n  overflow={ovf_count.value}", flush=True)
                    ovf_last = ovf_count.value

                if save_path:
                    iq_buffer.append(chunk)
                    if len(iq_buffer) >= 500:
                        self._flush_iq_buf(iq_buffer, save_path)
                        iq_buffer = []

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            running.value = 0
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
            if iq_buffer and save_path:
                self._flush_iq_buf(iq_buffer, save_path)
            self.overflow_count = ovf_count.value
            shm.close()
            shm.unlink()
            self._print_summary(time.time() - t0)

    def _flush_iq_buf(self, buf_list, path):
        if buf_list:
            data = np.concatenate(buf_list)
            mode = 'ab' if os.path.isfile(path) else 'wb'
            with open(path, mode) as f:
                np.save(f, data)
            print(f"[save_iq] {len(data)} 样本", flush=True)

    # ================================================================
    # 统计输出
    # ================================================================

    def _print_summary(self, elapsed: float):
        print(f"\n--- 结果 ---")
        print(f"  frames={self.total_frames}  "
              f"CRC={self.crc_pass}/{self.total_frames} "
              f"({self.crc_pass/max(self.total_frames,1)*100:.1f}%)  "
              f"HDR={self.header_crc_pass}  "
              f"false_alarms={self.false_alarms}",
              flush=True)

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
        }


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description='BPSK PHY 接收端')
    p.add_argument('--mode', default='sim', choices=['hardware', 'sim'])
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--gain', type=float, default=20, help='B210: 0.0~76.0dB PGA')
    p.add_argument('--rate', type=float, default=1e6)
    p.add_argument('--sps', type=int, default=SPS)
    p.add_argument('--sim-file', default='rx_iq.npy')
    p.add_argument('--snr-db', type=float, default=15.0)
    p.add_argument('--usrp-args', default='')
    p.add_argument('--subdev', default='A:A', help='子设备 (A:A=TX/RX, A:B=RX2)')
    p.add_argument('--save-iq', default='', help='保存原始 IQ')
    p.add_argument('--sync-mode', default='host', choices=['host', 'external_ref'],
                   help='时钟同步模式')
    p.add_argument('--settle', type=float, default=1.0, help='外部参考锁定时长 (s)')
    p.add_argument('--stf-threshold', type=float, default=STF_THRESHOLD,
                   help='STF 包检测门限')
    p.add_argument('--pss-ptm', type=float, default=PSS_PEAK_TO_MEAN_THR,
                   help='PSS 峰均比门限')
    p.add_argument('--pss-pts', type=float, default=PSS_PEAK_TO_SECOND_THR,
                   help='PSS 峰次比门限')
    p.add_argument('--rs-corr-thr', type=float, default=0.3,
                   help='RS 相关门限 (相对值)')
    args = p.parse_args()

    rx = BpskPhyReceiver(samp_rate=args.rate, sps=args.sps,
                         stf_threshold=args.stf_threshold,
                         pss_ptm=args.pss_ptm,
                         pss_pts=args.pss_pts,
                         rs_corr_thr=args.rs_corr_thr)
    rx.start(mode=args.mode, freq=args.freq, gain=args.gain,
             sim_file=args.sim_file,
             snr_db=args.snr_db, usrp_args=args.usrp_args,
             subdev=args.subdev, save_iq=args.save_iq,
             sync_mode=args.sync_mode, settle_s=args.settle)


if __name__ == '__main__':
    mp.freeze_support()
    main()
