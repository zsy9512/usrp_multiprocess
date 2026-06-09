#!/usr/bin/env python3
"""
calibrated_simulation.py — Capture-driven calibrated simulation

从 sim_calibration.json 读取信道参数, 生成合成 RX IQ, 跑完整同步链,
输出与硬件 capture 可对比的逐帧指标。

信道模型 (2m 静止空口, 平坦衰落):
  y = |h| * e^(j*phase) * x * e^(j*2pi*CFO*t) + noise
  CFO   ~ N(mu, sigma)      — 从 gain=55 参考 capture 提取
  phase ~ U(-pi, +pi)        — PLL 随机初相
  noise ~ CN(0, noise_floor) — 每 gain 档独立实测

两种模式:
  --mode blind    全盲同步 (STF扫描→PSS→RS→解调), 与硬件公平对比
  --mode ideal    已知帧头位置, 只测 BER vs SNR (Stage 6)

用法:
  # 对比 gain=55 硬件 vs 仿真
  python tools/calibrated_simulation.py capture/20260609/sim_calibration.json --gain 55 --frames 500

  # 全 gain 扫参 + 输出对比报告
  python tools/calibrated_simulation.py capture/20260609/sim_calibration.json --all-gains -o sim_report.json
"""

import argparse, json, os, sys, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from phy_params import (SPS, STF, PSS, RS, RRC,
                        STF_LEN, PSS_LEN, RS_LEN,
                        HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN,
                        STF_DELAY, FRAME_RRC_SAMPLES, RRC_DELAY_SAMPLES,
                        GUARD_SYMBOLS, FRAME_SYMBOLS,
                        crc16, crc16_check, bits_to_bytes, bytes_to_bits)
from sender import build_frame, rrc_filter

# 复用 extract_channel_stats 的同步函数 (与 polar_loopback.py 一致)
from tools.extract_channel_stats import (
    _rrc_match, _pss_correlate, _rs_estimate, _bpsk_demod_hard, _b2i
)

SAMP_RATE = 1e6
TS_SYM = SPS / SAMP_RATE
N_POLAR = 256
K_POLAR = 128

# 加载 Polar 编码
FROZEN_PATH = os.path.join(BASE, 'deploy', 'matrices', 'A.npy')
FROZEN_MASK = np.load(FROZEN_PATH).squeeze()


def _polar_encode(u):
    cw = u.copy().ravel()
    for stage in range(1, 8):  # log2(256) = 8
        sep = N_POLAR // (1 << stage)
        for j in range(N_POLAR):
            if (j // sep) % 2 == 0:
                cw[j] = (cw[j] + cw[j + sep]) % 2
    return cw


def _build_codeword(info_bits):
    u = np.zeros(N_POLAR, dtype=np.int64)
    u[FROZEN_MASK.astype(bool)] = info_bits.ravel()
    return _polar_encode(u)


def _polar_hard_inverse(llr):
    hard_bits = (llr < 0).astype(np.int64)
    u_hat = _polar_encode(hard_bits)
    return u_hat[FROZEN_MASK.astype(bool)]


# ======================================================================
# 信道模拟
# ======================================================================

def apply_channel(tx_iq, h_mag, cfo_hz, phase_rad, noise_floor,
                  samp_rate=1e6, timing_offset_sym=0.0):
    """施加平坦信道效应.

    Args:
        tx_iq:          (N,) complex64  发送 IQ 样本
        h_mag:          float           信道幅度
        cfo_hz:         float           载波频偏 Hz
        phase_rad:      float           初始相位 rad
        noise_floor:    float           符号域噪声方差 (匹配 canonical SNR 定义)
        samp_rate:      float           采样率
        timing_offset_sym: float        定时偏 (符号), 通过整数样本移位近似

    Returns:
        (N,) complex64  接收 IQ 样本
    """
    rx = tx_iq.astype(np.complex64).copy()

    # 定时偏 (整数样本移位近似, < 0.5 sym 时影响可忽略)
    if abs(timing_offset_sym) > 0.01:
        shift = int(round(timing_offset_sym * SPS))
        if shift > 0:
            rx = np.concatenate([np.zeros(shift, dtype=np.complex64), rx[:-shift]])
        elif shift < 0:
            rx = np.concatenate([rx[-shift:], np.zeros(-shift, dtype=np.complex64)])

    # CFO + 相偏
    t = np.arange(len(rx), dtype=np.float64) / samp_rate
    rx = rx * np.exp(1j * (2 * np.pi * cfo_hz * t + phase_rad))

    # 信道幅度
    rx = rx * h_mag

    # AWGN: noise_floor 是符号域方差, 需转换到 IQ 样本域
    # RRC 匹配滤波后符号方差 = noise_floor
    # IQ 样本域噪声方差 ≈ noise_floor * SPS (上采样后噪声)
    noise_std_iq = np.sqrt(noise_floor * SPS / 2)  # /2 因为复噪声
    noise = (noise_std_iq * np.random.randn(len(rx))
             + 1j * noise_std_iq * np.random.randn(len(rx)))
    rx = rx + noise.astype(np.complex64)

    return rx


# ======================================================================
# STF 检测 (与 polar_loopback.py _stf_detect 完全一致)
# ======================================================================

def _stf_detect_one(samples, stf_threshold=0.4, stf_min_energy=0.01):
    """单窗口 STF 检测: 返回最强峰或 None."""
    L = STF_DELAY
    N = len(samples)
    if N <= L:
        return None

    r0, rL = samples[:N - L], samples[L:]
    prod = r0 * np.conj(rL)
    ones = np.ones(L, dtype=np.float32)
    P = np.convolve(prod, ones, mode='valid')
    E = np.convolve((np.abs(rL) ** 2).astype(np.float32), ones, mode='valid')
    M = np.abs(P) / (E + 1e-6 * L)

    # 找最强峰 (大于门限 + 能量足够)
    best_d, best_m, best_p = -1, 0, 0j
    for d in range(len(M)):
        if M[d] > stf_threshold:
            le = np.sum(np.abs(samples[d + L:d + 2 * L]) ** 2)
            if le > stf_min_energy and M[d] > best_m:
                best_d = d
                best_m = float(M[d])
                best_p = P[d]

    if best_d < 0:
        return None

    coarse_cfo = float(-np.angle(best_p) / (2 * np.pi * L / SAMP_RATE))
    return {'pos': best_d, 'M': best_m, 'coarse_cfo': coarse_cfo}


# ======================================================================
# 单帧仿真 + 同步
# ======================================================================

def simulate_one_frame(info_bits, frame_id, h_mag, cfo_hz, phase_rad,
                       noise_floor, timing_offset_sym=0.0,
                       stf_threshold=0.4, stf_min_energy=0.01,
                       pss_ptm=2.5, pss_pts=1.0, rs_corr_min=0.1):
    """生成一帧, 通过信道, 盲同步, 返回所有指标.

    Returns:
        dict with sync metrics (same keys as extract_channel_stats detections),
        or None if sync fails.
    """
    # ── TX: 信息比特 → Polar 编码 → 成帧 → RRC ──
    coded_bits = _build_codeword(info_bits)
    frame_syms = build_frame(coded_bits, frame_id)
    tx_iq = rrc_filter(frame_syms, RRC, SPS)

    # ── 信道 ──
    rx_iq = apply_channel(tx_iq, h_mag, cfo_hz, phase_rad, noise_floor,
                          SAMP_RATE, timing_offset_sym)

    # ── 前后加噪声 padding (模拟帧间隔) ──
    pad_before = 2000  # IQ 样本
    pad_after = 3000
    pad_noise_before = (np.sqrt(noise_floor * SPS / 2)
                        * (np.random.randn(pad_before)
                           + 1j * np.random.randn(pad_before))).astype(np.complex64)
    pad_noise_after = (np.sqrt(noise_floor * SPS / 2)
                       * (np.random.randn(pad_after)
                          + 1j * np.random.randn(pad_after))).astype(np.complex64)
    rx_with_pad = np.concatenate([pad_noise_before, rx_iq, pad_noise_after])

    # ── RRC 匹配滤波 ──
    syms = _rrc_match(rx_with_pad)
    if len(syms) < PSS_LEN + RS_LEN:
        return None

    # ── STF 检测 (盲) ──
    stf_result = _stf_detect_one(rx_with_pad, stf_threshold, stf_min_energy)
    if stf_result is None:
        return {'detected': False, 'fail_stage': 'STF'}

    # ── PSS ──
    # 提取帧窗口 (围绕 STF 位置)
    coarse = stf_result['pos']
    margin = 400
    es = max(0, coarse - margin)
    ee = min(len(rx_with_pad), coarse + margin + FRAME_RRC_SAMPLES + margin)
    syms_window = _rrc_match(rx_with_pad[es:ee])

    if len(syms_window) < PSS_LEN + RS_LEN:
        return {'detected': False, 'fail_stage': 'PSS', 'reason': 'window too short'}

    pk, ptm, pts, pval = _pss_correlate(syms_window)
    if ptm < pss_ptm or pts < pss_pts:
        return {'detected': False, 'fail_stage': 'PSS',
                'ptm': ptm, 'pts': pts}

    fs = pk - STF_LEN
    if fs < 0:
        return {'detected': False, 'fail_stage': 'PSS', 'reason': 'fs<0'}

    rp = fs + STF_LEN + PSS_LEN
    if rp + RS_LEN + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN > len(syms_window):
        return {'detected': False, 'fail_stage': 'RS', 'reason': 'frame too long'}

    # ── RS ──
    chan = _rs_estimate(syms_window, rp, stf_result['coarse_cfo'])
    if chan is None:
        return {'detected': False, 'fail_stage': 'RS'}

    if chan['rs_corr'] < RS_LEN * rs_corr_min:
        return {'detected': False, 'fail_stage': 'RS', 'reason': 'rs_corr low'}

    hmag = abs(chan['h'])
    snr_sym = float(10 * np.log10(max(hmag ** 2 / max(noise_floor, 1e-30), 1e-30)))
    snr_rs = float(10 * np.log10(max(hmag ** 2 / max(chan['sigma2'], 1e-30), 1e-30)))

    # ── BPSK 硬解调 ──
    hdr_start = rp + RS_LEN
    hdr_bits = _bpsk_demod_hard(syms_window, hdr_start, HEADER_LEN,
                                 chan['h'], chan['total_cfo'])
    hdr_ok = False
    if len(hdr_bits) >= 32:
        decoded_fid = _b2i(hdr_bits[:16])
        hdr_ok = crc16_check(bits_to_bytes(hdr_bits[:16]),
                             _b2i(hdr_bits[16:32]))
    else:
        decoded_fid = -1

    pay_start = hdr_start + HEADER_LEN
    pay_bits = _bpsk_demod_hard(syms_window, pay_start,
                                 PAYLOAD_LEN + PAYLOAD_CRC_LEN,
                                 chan['h'], chan['total_cfo'])
    crc_ok = False
    if len(pay_bits) >= PAYLOAD_LEN + PAYLOAD_CRC_LEN:
        payload = pay_bits[:PAYLOAD_LEN]
        crc_val = _b2i(pay_bits[PAYLOAD_LEN:])
        crc_ok = crc16_check(bits_to_bytes(payload), crc_val)

    # ── LLR + Polar 硬判逆变换 ──
    # LLR (对齐 polar_loopback.py _bpsk_demod_llr)
    seg = syms_window[pay_start:pay_start + PAYLOAD_LEN]
    n_arr = np.arange(PAYLOAD_LEN)
    cfo_comp = np.exp(-1j * 2 * np.pi * chan['total_cfo'] * (pay_start + n_arr) * TS_SYM)
    y_eq = seg * cfo_comp
    if abs(chan['h']) > 1e-30:
        y_eq = y_eq / chan['h']
    sigma2_out = max(float(chan['sigma2']), 1e-6)
    llr = np.clip(4.0 * y_eq.real / sigma2_out, -20.0, 20.0).astype(np.float32)

    info_hat = _polar_hard_inverse(llr)
    info_errs = int(np.sum(info_hat != info_bits.ravel()))
    coded_errs = int(np.sum(pay_bits[:PAYLOAD_LEN] != coded_bits.ravel()))

    return {
        'detected': True,
        'frame_id': frame_id,
        'decoded_fid': decoded_fid,
        'ptm': ptm,
        'pts': pts,
        'coarse_cfo': chan['coarse_cfo'],
        'fine_cfo': chan['fine_cfo'],
        'total_cfo': chan['total_cfo'],
        'phase_est': chan['phase_est'],
        'h_mag': hmag,
        'sigma2': chan['sigma2'],
        'rs_corr': chan['rs_corr'],
        'snr_symbol': snr_sym,
        'snr_rs': snr_rs,
        'noise_floor': noise_floor,
        'hdr_ok': hdr_ok,
        'crc_ok': crc_ok,
        'info_errs': info_errs,
        'coded_errs': coded_errs,
        'applied_cfo': cfo_hz,
        'applied_phase': phase_rad,
        'applied_h_mag': h_mag,
    }


# ======================================================================
# 批量仿真 + 统计
# ======================================================================

def run_simulation(calib, target_gain, n_frames=500, seed=42,
                   stf_threshold=0.4, stf_min_energy=0.01,
                   pss_ptm=2.5, pss_pts=1.0, rs_corr_min=0.1):
    """对指定 gain 档运行 N 帧仿真.

    Args:
        calib:          sim_calibration.json 内容
        target_gain:    目标 gain (如 "55.0")
        n_frames:       仿真帧数
        seed:           随机种子

    Returns:
        dict: {detections: [...], stats: {...}, config: {...}}
    """
    rng = np.random.RandomState(seed)

    ch_model = calib['channel_model']
    gain_info = calib['per_gain'].get(str(target_gain), {})

    h_mag = gain_info.get('h_mag_mean', 0.1)
    noise_floor = gain_info.get('noise_floor', 1e-6)

    # CFO 分布参数
    cfo_mean = ch_model['cfo_hz']['mean']
    cfo_std = ch_model['cfo_hz']['std']
    timing_std = ch_model.get('timing_offset_sym', {}).get('std', 0.058)

    detections = []
    failures = {'STF': 0, 'PSS': 0, 'RS': 0}

    for fi in range(n_frames):
        # 随机信息比特 (128bit)
        info_bits = (rng.rand(K_POLAR) < 0.5).astype(np.int64)

        # 采样信道参数
        cfo = rng.normal(cfo_mean, cfo_std)
        phase = rng.uniform(-np.pi, np.pi)
        timing = rng.normal(0, timing_std)

        result = simulate_one_frame(
            info_bits, fi, h_mag, cfo, phase, noise_floor, timing,
            stf_threshold, stf_min_energy, pss_ptm, pss_pts, rs_corr_min,
        )

        if result is None:
            continue
        if result.get('detected'):
            detections.append(result)
        else:
            stage = result.get('fail_stage', 'unknown')
            if stage in failures:
                failures[stage] += 1
            else:
                failures[stage] = 1

    n_det = len(detections)
    stats = {'n_frames': n_frames, 'n_detected': n_det}
    stats.update({f'fail_{k}': v for k, v in failures.items()})

    if n_det > 0:
        cfos = [d['total_cfo'] for d in detections]
        ptms = [d['ptm'] for d in detections]
        ptss = [d['pts'] for d in detections]
        snrs = [d['snr_symbol'] for d in detections]
        snrs_rs = [d['snr_rs'] for d in detections]
        hdr_ok = sum(1 for d in detections if d['hdr_ok'])
        crc_ok = sum(1 for d in detections if d['crc_ok'])
        info_errs = sum(d['info_errs'] for d in detections)
        coded_errs = sum(d['coded_errs'] for d in detections)
        info_total = n_det * K_POLAR
        coded_total = n_det * N_POLAR

        stats.update({
            'detection_rate': n_det / n_frames,
            'hdr_ok': hdr_ok,
            'crc_ok': crc_ok,
            'info_ber': info_errs / max(info_total, 1),
            'coded_ber': coded_errs / max(coded_total, 1),
            'cfo': {
                'mean': float(np.mean(cfos)),
                'std': float(np.std(cfos)),
                'min': float(np.min(cfos)),
                'max': float(np.max(cfos)),
            },
            'snr_symbol': {
                'mean': float(np.mean(snrs)),
                'std': float(np.std(snrs)),
            },
            'snr_rs': {
                'mean': float(np.mean(snrs_rs)),
                'std': float(np.std(snrs_rs)),
            },
            'ptm': {
                'mean': float(np.mean(ptms)),
                'std': float(np.std(ptms)),
            },
            'pts': {
                'mean': float(np.mean(ptss)),
                'std': float(np.std(ptss)),
            },
        })

    return {
        'detections': detections,
        'stats': stats,
        'config': {
            'target_gain': target_gain,
            'h_mag': h_mag,
            'noise_floor': noise_floor,
            'cfo_mean': cfo_mean,
            'cfo_std': cfo_std,
            'timing_std': timing_std,
            'stf_threshold': stf_threshold,
            'stf_min_energy': stf_min_energy,
            'pss_ptm': pss_ptm,
            'pss_pts': pss_pts,
            'rs_corr_min': rs_corr_min,
        },
    }


# ======================================================================
# 硬件 vs 仿真对比
# ======================================================================

def compare_with_hardware(sim_stats, hw_stats, gain_label):
    """对比仿真和硬件统计, 生成一致性报告."""
    report = {'gain': gain_label}

    if sim_stats['n_detected'] == 0:
        report['verdict'] = 'sim_no_detections'
        return report

    # 检出率
    sim_det_rate = sim_stats.get('detection_rate', 0)
    report['sim_detection_rate'] = sim_det_rate
    if 'n_detections' in hw_stats and hw_stats['n_detections'] > 0:
        hw_n = hw_stats['n_detections']
        # 硬件是 capture 有 200 帧发送, 检出率 = n_detected / 200
        hw_det_rate = hw_n / 200
        report['hw_detection_rate'] = hw_det_rate
        report['det_rate_delta'] = sim_det_rate - hw_det_rate

    # CFO 均值
    if 'cfo' in sim_stats and 'cfo' in hw_stats:
        sim_cfo = sim_stats['cfo']['mean']
        hw_cfo = hw_stats['cfo']['mean']
        report['cfo_delta_hz'] = sim_cfo - hw_cfo
        report['cfo_match'] = abs(report['cfo_delta_hz']) < 20

    # SNR
    if 'snr_symbol' in sim_stats and 'snr_symbol' in hw_stats:
        sim_snr = sim_stats['snr_symbol']['mean']
        hw_snr = hw_stats['snr_symbol']['mean']
        report['snr_delta_db'] = sim_snr - hw_snr
        report['snr_match'] = abs(report['snr_delta_db']) < 3

    # PSS ptm 对比
    if 'ptm' in sim_stats and 'ptm' in hw_stats:
        report['ptm_delta'] = sim_stats['ptm']['mean'] - hw_stats['ptm']['mean']

    # 综合判定
    checks = [report.get('cfo_match', True), report.get('snr_match', True)]
    report['parity_ok'] = all(checks)
    report['verdict'] = 'PASS' if report['parity_ok'] else 'MISMATCH'

    return report


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(
        description='Capture-driven calibrated simulation')
    p.add_argument('calib_json', help='sim_calibration.json 路径')
    p.add_argument('--gain', default='',
                   help='目标 gain 档位 (如 "55.0")')
    p.add_argument('--all-gains', action='store_true',
                   help='对所有 gain 档位仿真')
    p.add_argument('--frames', type=int, default=500,
                   help='每档仿真帧数 (默认 500)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stf-threshold', type=float, default=0.4)
    p.add_argument('--stf-energy', type=float, default=0.01)
    p.add_argument('--pss-ptm', type=float, default=2.5)
    p.add_argument('--pss-pts', type=float, default=1.0)
    p.add_argument('--rs-corr-min', type=float, default=0.1)
    p.add_argument('-o', '--output', default='',
                   help='输出 JSON (仿真+对比报告)')
    p.add_argument('--verbose', action='store_true',
                   help='打印每帧结果')
    args = p.parse_args()

    with open(args.calib_json, encoding='utf-8') as f:
        calib = json.load(f)

    # 兼容两种 JSON 结构: 直接 channel_model 或嵌套在 simulation_calibration 下
    if 'simulation_calibration' in calib:
        sim_cal = calib['simulation_calibration']
    else:
        sim_cal = calib

    print(f"加载标定: {args.calib_json}")
    ch = sim_cal['channel_model']
    print(f"  CFO: N({ch['cfo_hz']['mean']:.1f}, {ch['cfo_hz']['std']:.1f}) Hz")
    print(f"  Phase: U(-pi, +pi)")
    print(f"  Timing std: {ch['timing_offset_sym']['std']:.4f} sym")

    gains_to_run = []
    if args.all_gains:
        gains_to_run = sorted(sim_cal['per_gain'].keys(),
                              key=lambda x: float(x))
    elif args.gain:
        gains_to_run = [args.gain]
    else:
        # 默认跑参考 gain (检测数最多的那个)
        src = sim_cal.get('source_gain_db', '55.0')
        gains_to_run = [src]

    all_results = {}
    for gain in gains_to_run:
        print(f"\n{'='*60}")
        print(f"  仿真 gain={gain} dB  ({args.frames} 帧)")
        print(f"{'='*60}")

        t0 = time.time()
        result = run_simulation(
            sim_cal, gain, n_frames=args.frames, seed=args.seed,
            stf_threshold=args.stf_threshold,
            stf_min_energy=args.stf_energy,
            pss_ptm=args.pss_ptm,
            pss_pts=args.pss_pts,
            rs_corr_min=args.rs_corr_min,
        )
        elapsed = time.time() - t0

        s = result['stats']
        n_det = s['n_detected']
        print(f"  检出: {n_det}/{args.frames} ({n_det/max(args.frames,1)*100:.1f}%)  "
              f"[{elapsed:.1f}s, {args.frames/elapsed:.0f} fps]")
        print(f"  失败: STF={s.get('fail_STF',0)}  "
              f"PSS={s.get('fail_PSS',0)}  RS={s.get('fail_RS',0)}")

        if n_det > 0:
            print(f"  HDR OK: {s['hdr_ok']}/{n_det}  CRC OK: {s['crc_ok']}/{n_det}")
            print(f"  SNR: {s['snr_symbol']['mean']:.1f} ± {s['snr_symbol']['std']:.1f} dB")
            print(f"  CFO (estimated): {s['cfo']['mean']:+.1f} ± {s['cfo']['std']:.1f} Hz")
            print(f"  info BER: {s['info_ber']*100:.2f}%  "
                  f"coded BER: {s['coded_ber']*100:.2f}%")
            print(f"  ptm: {s['ptm']['mean']:.1f} ± {s['ptm']['std']:.1f}")

        # 与硬件对比
        hw_data = calib['by_gain'].get(gain, {})
        hw_stats = hw_data.get('stats', {})
        if hw_stats.get('n_detections', 0) > 0:
            report = compare_with_hardware(s, hw_stats, gain)
            print(f"\n  ── 硬件 vs 仿真对比 ──")
            print(f"  检出率: sim={report.get('sim_detection_rate',0)*100:.0f}%  "
                  f"hw={report.get('hw_detection_rate',0)*100:.0f}%")
            if 'cfo_delta_hz' in report:
                print(f"  CFO delta: {report['cfo_delta_hz']:+.1f} Hz  "
                      f"{'[OK]' if report.get('cfo_match') else '[WARN]'}")
            if 'snr_delta_db' in report:
                print(f"  SNR delta: {report['snr_delta_db']:+.1f} dB  "
                      f"{'[OK]' if report.get('snr_match') else '[WARN]'}")
            print(f"  一致性: {report['verdict']}")
        else:
            print(f"\n  (硬件 {gain} dB 无检出, 跳过对比)")

        # 保存结果 (不含逐帧数据以减小文件)
        all_results[gain] = {
            'stats': s,
            'config': result['config'],
        }

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n仿真报告 → {args.output}")


if __name__ == '__main__':
    main()
