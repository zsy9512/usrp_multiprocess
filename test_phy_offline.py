#!/usr/bin/env python3
"""
test_phy_offline.py — 完整离线测试套件

覆盖测试矩阵:
  A: 基础正确性 (SNR sweep)
  B: CFO 容忍度
  C: 初始相位
  D: 多径
  E: 采样定时误差
  F: 连续多帧

用法:
  python test_phy_offline.py
  python test_phy_offline.py --test A
  python test_phy_offline.py --log-file test_results.txt
  python test_phy_offline.py --test A --frames 2000
"""
from __future__ import annotations

import argparse, os, sys, time
import numpy as np

from phy_params import (
    SPS, RRC, STF_LEN, PSS_LEN, RS_LEN,
    HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN, GUARD_SYMBOLS,
    FRAME_SYMBOLS, STF_DELAY,
)
from sender import build_frame, rrc_filter, default_bit_source
from receiver import BpskPhyReceiver


# ======================================================================
# 仿真信道
# ======================================================================

class SimChannel:
    """纯 AWGN + CFO + 相偏 + 多径."""

    def __init__(self, samp_rate: float = 1e6):
        self.samp_rate = samp_rate

    def process(self, tx_signal: np.ndarray, snr_db: float = 30.0,
                freq_offset: float = 0.0, phase_offset: float = 0.0,
                multipath: str = "") -> np.ndarray:
        rx = tx_signal.copy().astype(np.complex64)

        if abs(freq_offset) > 0:
            t = np.arange(len(rx), dtype=np.float64) / self.samp_rate
            rx = rx * np.exp(1j * 2 * np.pi * freq_offset * t)

        if abs(phase_offset) > 0:
            rx = rx * np.exp(1j * phase_offset)

        if multipath:
            taps = self._parse_multipath(multipath)
            if len(taps) > 1:
                rx = np.convolve(rx, taps, mode='full')[:len(rx)]

        if snr_db < 100:
            signal_power = np.mean(np.abs(rx) ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(len(rx)) + 1j * np.random.randn(len(rx)))
            rx = rx + noise.astype(np.complex64)

        return rx

    @staticmethod
    def _parse_multipath(cfg: str) -> np.ndarray:
        if not cfg:
            return np.array([1.0], dtype=np.complex64)
        parts = cfg.split(',')
        max_delay = 0
        tap_info = []
        for part in parts:
            if '@' in part:
                gain_str, delay_str = part.split('@')
                gain = float(gain_str)
                delay = int(delay_str)
            else:
                gain = float(part)
                delay = 0
            tap_info.append((gain, delay))
            max_delay = max(max_delay, delay)
        h = np.zeros(max_delay + 1, dtype=np.complex64)
        for gain, delay in tap_info:
            h[delay] = gain
        h /= np.sqrt(np.sum(np.abs(h) ** 2))
        return h


# ======================================================================
# 测试基类
# ======================================================================

class TestLogger:
    def __init__(self, log_file: str = ""):
        self.log_file = log_file
        self.lines = []

    def log(self, msg: str = ""):
        print(msg)
        self.lines.append(msg)

    def save(self):
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.lines))
            print(f"\n日志已保存至 {self.log_file}")


def run_test_single(name: str, tx_iq: np.ndarray, ref_bits: np.ndarray,
                    snr_db: float, freq_offset: float = 0.0,
                    phase_offset: float = 0.0, multipath: str = "",
                    samp_rate: float = 1e6, logger: TestLogger = None) -> dict:
    """单条件测试."""
    if logger is None:
        logger = TestLogger()

    ch = SimChannel(samp_rate)
    rx_iq = ch.process(tx_iq, snr_db=snr_db, freq_offset=freq_offset,
                       phase_offset=phase_offset, multipath=multipath)

    rx = BpskPhyReceiver(samp_rate)
    rx.running = True
    # 设置参考比特用于 BER
    rx.tx_ref_bits = ref_bits
    rx._process_samples(rx_iq)

    stats = rx.get_stats()

    result = {
        'name': name,
        'snr_db': snr_db,
        'freq_offset_hz': freq_offset,
        'phase_offset_rad': phase_offset,
        'multipath': multipath,
        'detected': stats['total_frames'],
        'crc_pass': stats['crc_pass'],
        'crc_rate': stats['crc_rate'],
        'false_alarms': stats['false_alarms'],
        'total_bits': stats['total_bits'],
        'total_errors': stats['total_errors'],
        'pass': stats['crc_rate'] > 0.99 and stats['total_frames'] > 0,
    }

    status = 'PASS' if result['pass'] else 'FAIL'
    logger.log(
        f"  [{status}] {name:20s}  "
        f"SNR={snr_db:5.1f}dB  "
        f"CFO={freq_offset:+7.0f}Hz  "
        f"θ={phase_offset:+.2f}rad  "
        f"多径={multipath or '无':15s}  "
        f"检帧={stats['total_frames']:4d}  "
        f"CRC={stats['crc_pass']:4d}/{stats['total_frames']:d}  "
        f"误检={stats['false_alarms']:3d}")

    return result


# ======================================================================
# 测试矩阵
# ======================================================================

def test_matrix_A(logger: TestLogger, num_frames: int = 200,
                  samp_rate: float = 1e6):
    """矩阵 A: 基础正确性 (SNR sweep)."""
    logger.log(f"\n{'='*70}")
    logger.log(f"测试矩阵 A: 基础正确性 (SNR sweep)")
    logger.log(f"{'='*70}")
    logger.log(f"  帧数={num_frames}  采样率={samp_rate/1e6:.1f}Msps")

    tx_buf, ref_all = [], []
    rrc = RRC
    for _ in range(num_frames):
        bits = default_bit_source(PAYLOAD_LEN)
        frame_syms = build_frame(bits)
        tx_sig = rrc_filter(frame_syms, rrc, SPS)
        tx_buf.append(tx_sig)
        ref_all.append(bits)
    tx_iq = np.concatenate(tx_buf)
    ref_bits = np.concatenate(ref_all)

    snr_list = [100, 20, 15, 10, 8, 6, 5, 4, 3, 2]
    results = []
    for snr in snr_list:
        r = run_test_single(f"A_snr{snr}", tx_iq, ref_bits, snr, 0, 0, "",
                            samp_rate, logger)
        results.append(r)

    logger.log(f"\n  {'SNR':>6s}  {'检帧':>6s}  {'CRC':>6s}  {'CRC率':>8s}  {'状态':>6s}")
    logger.log(f"  {'-'*40}")
    passes = 0
    for r in results:
        logger.log(f"  {r['snr_db']:6.1f}  {r['detected']:6d}  {r['crc_pass']:6d}  "
                   f"{r['crc_rate']:8.4f}  {'PASS' if r['pass'] else 'FAIL':>6s}")
        if r['pass']:
            passes += 1

    a0_pass = results[0]['crc_rate'] > 0.999
    a1_pass = results[1]['crc_rate'] > 0.99
    logger.log(f"\n  验收: A0(无噪)={'PASS' if a0_pass else 'FAIL'}  "
               f"A1(20dB)={'PASS' if a1_pass else 'FAIL'}")
    logger.log(f"{'='*70}\n")

    return {'matrix': 'A', 'results': results,
            'a0_pass': a0_pass, 'a1_pass': a1_pass,
            'overall_pass': a0_pass and a1_pass}


def test_matrix_B(logger: TestLogger, num_frames: int = 200,
                  samp_rate: float = 1e6):
    """矩阵 B: CFO 容忍度."""
    logger.log(f"\n{'='*70}")
    logger.log(f"测试矩阵 B: CFO 容忍度")
    logger.log(f"{'='*70}")
    logger.log(f"  帧数={num_frames}  SNR=20dB")

    tx_buf = []
    rrc = RRC
    for _ in range(num_frames):
        bits = default_bit_source(PAYLOAD_LEN)
        frame_syms = build_frame(bits)
        tx_sig = rrc_filter(frame_syms, rrc, SPS)
        tx_buf.append(tx_sig)
    tx_iq = np.concatenate(tx_buf)

    cfo_list = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, -1000, -5000]
    results = []
    for cfo in cfo_list:
        r = run_test_single(f"B_cfo{cfo:+}", tx_iq, None, 20, cfo, 0, "",
                            samp_rate, logger)
        results.append(r)

    logger.log(f"\n  {'CFO':>8s}  {'检帧':>6s}  {'CRC':>6s}  {'CRC率':>8s}  {'状态':>6s}")
    logger.log(f"  {'-'*45}")
    for r in results:
        logger.log(f"  {r['freq_offset_hz']:+8.0f}  {r['detected']:6d}  "
                   f"{r['crc_pass']:6d}  {r['crc_rate']:8.4f}  "
                   f"{'PASS' if r['pass'] else 'FAIL':>6s}")

    b0_pass = results[0]['pass']
    b3_pass = results[3]['pass']
    b5_pass = results[5]['pass']
    logger.log(f"\n  验收: B0(0Hz)={'PASS' if b0_pass else 'FAIL'}  "
               f"B3(+1kHz)={'PASS' if b3_pass else 'FAIL'}  "
               f"B5(+5kHz)={'PASS' if b5_pass else 'FAIL'}")
    logger.log(f"{'='*70}\n")

    return {'matrix': 'B', 'results': results,
            'b0_pass': b0_pass, 'b3_pass': b3_pass, 'b5_pass': b5_pass,
            'overall_pass': b0_pass and b3_pass}


def test_matrix_C(logger: TestLogger, num_frames: int = 200,
                  samp_rate: float = 1e6):
    """矩阵 C: 初始相位."""
    logger.log(f"\n{'='*70}")
    logger.log(f"测试矩阵 C: 初始相位容忍度")
    logger.log(f"{'='*70}")
    logger.log(f"  帧数={num_frames}  SNR=20dB")

    tx_buf = []
    rrc = RRC
    for _ in range(num_frames):
        bits = default_bit_source(PAYLOAD_LEN)
        frame_syms = build_frame(bits)
        tx_sig = rrc_filter(frame_syms, rrc, SPS)
        tx_buf.append(tx_sig)
    tx_iq = np.concatenate(tx_buf)

    phase_list = [0, np.pi/8, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -np.pi/2]
    results = []
    for theta in phase_list:
        r = run_test_single(f"C_θ{theta:.3f}", tx_iq, None,
                            20, 0, theta, "", samp_rate, logger)
        results.append(r)

    logger.log(f"\n  {'θ(rad)':>10s}  {'检帧':>6s}  {'CRC':>6s}  {'CRC率':>8s}  {'状态':>6s}")
    logger.log(f"  {'-'*45}")
    for r in results:
        logger.log(f"  {r['phase_offset_rad']:10.4f}  {r['detected']:6d}  "
                   f"{r['crc_pass']:6d}  {r['crc_rate']:8.4f}  "
                   f"{'PASS' if r['pass'] else 'FAIL':>6s}")

    all_pass = all(r['pass'] for r in results)
    logger.log(f"\n  验收: 全部相位={'PASS' if all_pass else 'FAIL'}")
    logger.log(f"{'='*70}\n")

    return {'matrix': 'C', 'results': results, 'all_pass': all_pass,
            'overall_pass': all_pass}


def test_matrix_D(logger: TestLogger, num_frames: int = 200,
                  samp_rate: float = 1e6):
    """矩阵 D: 多径."""
    logger.log(f"\n{'='*70}")
    logger.log(f"测试矩阵 D: 多径容忍度")
    logger.log(f"{'='*70}")
    logger.log(f"  帧数={num_frames}  SNR=30dB")

    tx_buf = []
    rrc = RRC
    for _ in range(num_frames):
        bits = default_bit_source(PAYLOAD_LEN)
        frame_syms = build_frame(bits)
        tx_sig = rrc_filter(frame_syms, rrc, SPS)
        tx_buf.append(tx_sig)
    tx_iq = np.concatenate(tx_buf)

    mp_list = [
        ("无多径", ""),
        ("弱@1", "1.0,0.2@1"),
        ("弱@2", "1.0,0.3@2"),
        ("中@4", "1.0,0.3@4"),
        ("强@8", "1.0,0.5@8"),
        ("多径组合", "1.0,0.3@3,0.1@7"),
    ]
    results = []
    for mp_name, mp_cfg in mp_list:
        r = run_test_single(f"D_{mp_name}", tx_iq, None,
                            30, 0, 0, mp_cfg, samp_rate, logger)
        r['mp_name'] = mp_name
        results.append(r)

    logger.log(f"\n  {'多径':>12s}  {'检帧':>6s}  {'CRC':>6s}  {'CRC率':>8s}  {'状态':>6s}")
    logger.log(f"  {'-'*45}")
    for r in results:
        logger.log(f"  {r['mp_name']:>12s}  {r['detected']:6d}  "
                   f"{r['crc_pass']:6d}  {r['crc_rate']:8.4f}  "
                   f"{'PASS' if r['pass'] else 'FAIL':>6s}")

    d0_pass = results[0]['pass']
    d1_pass = results[1]['pass']
    logger.log(f"\n  验收: D0(无多径)={'PASS' if d0_pass else 'FAIL'}  "
               f"D1(弱@1)={'PASS' if d1_pass else 'FAIL'}")
    logger.log(f"{'='*70}\n")

    return {'matrix': 'D', 'results': results,
            'd0_pass': d0_pass, 'd1_pass': d1_pass,
            'overall_pass': d0_pass and d1_pass}


def test_matrix_E(logger: TestLogger, num_frames: int = 200,
                  samp_rate: float = 1e6):
    """矩阵 E: 采样定时误差."""
    logger.log(f"\n{'='*70}")
    logger.log(f"测试矩阵 E: 采样定时误差")
    logger.log(f"{'='*70}")
    logger.log(f"  帧数={num_frames}  SNR=20dB  sps={SPS}")

    tx_buf = []
    rrc = RRC
    for _ in range(num_frames):
        bits = default_bit_source(PAYLOAD_LEN)
        frame_syms = build_frame(bits)
        tx_sig = rrc_filter(frame_syms, rrc, SPS)
        tx_buf.append(tx_sig)
    tx_iq = np.concatenate(tx_buf)

    results = []
    for delay in range(SPS + 2):
        delayed = np.concatenate([np.zeros(delay, dtype=np.complex64), tx_iq])
        r = run_test_single(f"E_delay{delay}", delayed, None,
                            20, 0, 0, "", samp_rate, logger)
        r['delay'] = delay
        results.append(r)

    logger.log(f"\n  {'延迟':>6s}  {'检帧':>6s}  {'CRC':>6s}  {'CRC率':>8s}  {'状态':>6s}")
    logger.log(f"  {'-'*42}")
    for r in results:
        logger.log(f"  {r['delay']:6d}  {r['detected']:6d}  "
                   f"{r['crc_pass']:6d}  {r['crc_rate']:8.4f}  "
                   f"{'PASS' if r['pass'] else 'FAIL':>6s}")

    e0_pass = results[0]['pass']
    e1_pass = results[1]['pass']
    logger.log(f"\n  验收: E0(delay=0)={'PASS' if e0_pass else 'FAIL'}  "
               f"E1(delay=1)={'PASS' if e1_pass else 'FAIL'}")
    logger.log(f"{'='*70}\n")

    return {'matrix': 'E', 'results': results,
            'e0_pass': e0_pass, 'e1_pass': e1_pass,
            'overall_pass': e0_pass and e1_pass}


def test_matrix_F(logger: TestLogger, num_frames: int = 200,
                  samp_rate: float = 1e6):
    """矩阵 F: 连续多帧."""
    logger.log(f"\n{'='*70}")
    logger.log(f"测试矩阵 F: 连续多帧处理")
    logger.log(f"{'='*70}")
    logger.log(f"  帧数={num_frames}  SNR=30dB")

    tx_buf, ref_all = [], []
    rrc = RRC
    guard_samples = GUARD_SYMBOLS * SPS
    for _ in range(num_frames):
        bits = default_bit_source(PAYLOAD_LEN)
        frame_syms = build_frame(bits)
        tx_sig = rrc_filter(frame_syms, rrc, SPS)
        tx_buf.append(tx_sig)
        tx_buf.append(np.zeros(guard_samples, dtype=np.complex64))
        ref_all.append(bits)
    tx_iq = np.concatenate(tx_buf)
    ref_bits = np.concatenate(ref_all)

    rx = BpskPhyReceiver(samp_rate)
    rx.running = True
    rx.tx_ref_bits = ref_bits
    rx._process_samples(tx_iq)
    stats = rx.get_stats()

    detected = stats['total_frames']
    crc_pass = stats['crc_pass']
    crc_rate = stats['crc_rate']
    false_alarms = stats['false_alarms']

    f_pass = crc_rate > 0.99 and detected >= int(num_frames * 0.95)
    logger.log(f"\n  发送帧数: {num_frames}")
    logger.log(f"  检测帧数: {detected}")
    logger.log(f"  CRC通过:  {crc_pass}")
    logger.log(f"  CRC率:    {crc_rate:.4f}")
    logger.log(f"  误检:     {false_alarms}")
    logger.log(f"  状态:     {'PASS' if f_pass else 'FAIL'}")
    logger.log(f"{'='*70}\n")

    return {'matrix': 'F', 'detected': detected, 'crc_pass': crc_pass,
            'crc_rate': crc_rate, 'false_alarms': false_alarms,
            'pass': f_pass, 'overall_pass': f_pass,
            'num_frames': num_frames}


# ======================================================================
# 主测试入口
# ======================================================================

def main():
    p = argparse.ArgumentParser(description='PHY 离线测试套件')
    p.add_argument('--test', default='ALL',
                   choices=['A', 'B', 'C', 'D', 'E', 'F', 'ALL'])
    p.add_argument('--frames', type=int, default=200, help='每测试帧数')
    p.add_argument('--rate', type=float, default=1e6, help='采样率')
    p.add_argument('--log-file', default='', help='日志输出文件')
    args = p.parse_args()

    logger = TestLogger(args.log_file)
    logger.log(f"PHY 离线测试套件")
    logger.log(f"帧数/测试: {args.frames}  采样率: {args.rate/1e6:.1f}Msps")
    logger.log(f"帧参数: STF({STF_LEN})+PSS({PSS_LEN})+RS({RS_LEN})"
               f"+Header({HEADER_LEN})+Payload({PAYLOAD_LEN})"
               f"+CRC+Guard({GUARD_SYMBOLS})")
    logger.log(f"帧符号数: {FRAME_SYMBOLS}  SPS={SPS}")

    all_results = {}
    all_pass = True
    t_start = time.time()

    np.random.seed(42)

    test_map = {
        'A': ('基础正确性', test_matrix_A),
        'B': ('CFO容忍度', test_matrix_B),
        'C': ('相位容忍度', test_matrix_C),
        'D': ('多径容忍度', test_matrix_D),
        'E': ('采样定时', test_matrix_E),
        'F': ('连续多帧', test_matrix_F),
    }

    tests_to_run = test_map.keys() if args.test == 'ALL' else [args.test]

    for key in tests_to_run:
        name, func = test_map[key]
        logger.log(f"\n{'#'*70}")
        logger.log(f"# 测试 {key}: {name}")
        logger.log(f"{'#'*70}")
        result = func(logger, args.frames, args.rate)
        all_results[key] = result
        if not result.get('overall_pass', False):
            all_pass = False

    elapsed = time.time() - t_start
    logger.log(f"\n{'='*70}")
    logger.log(f"测试汇总")
    logger.log(f"{'='*70}")
    logger.log(f"  总耗时: {elapsed:.1f}s")
    for key in tests_to_run:
        r = all_results[key]
        status = 'PASS' if r.get('overall_pass', False) else 'FAIL'
        logger.log(f"  测试 {key}: {status}")

    logger.log(f"\n  总体: {'全部通过' if all_pass else '存在失败'}")
    logger.log(f"{'='*70}")

    logger.save()

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
