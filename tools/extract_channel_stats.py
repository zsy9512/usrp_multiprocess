#!/usr/bin/env python3
"""
extract_channel_stats.py — 从 capture 提取信道参数, 直接用于仿真标定

目的:
  不是做帧检测/译码评估（那是 loopback_analyze.py 的事），
  而是从原始 IQ 中提取 CFO / 相偏 / |h| / noise_floor / 定时 的分布，
  输出可直接喂给 sim_channel.py 做 calibrated 低 SNR 仿真。

核心物理假设（需用数据验证）:
  - CFO、相偏、定时偏 只依赖 TX/RX 硬件链，与 RX gain 无关
  - |h| 与 RX gain 成线性关系
  - noise_floor 随 RX gain 降低而降低（绝对 ADC 功率）

工作流:
  1. 加载一个或多个 capture 目录下的 _iq.npy + _meta.json
  2. 对每个 capture:
     a. 从 IQ 前缀测量独立底噪 noise_floor
     b. 全量 STF 扫描 (低门限, 最大化检出)
     c. 逐候选 PSS+RS 同步, 记录所有同步参数
     d. 可选: 尝试解调解码 (不强制, 低 SNR 下允许失败)
  3. 按 gain 分组, 输出:
     - 每帧的 {cfo, phase, |h|, sigma2, noise_floor, ptm, pts, timing}
     - 每组 gain 的统计摘要
  4. 跨 gain 对比: CFO/phase/timing 是否一致？(验证物理假设)

用法:
  # 单个 capture
  python tools/extract_channel_stats.py capture/20260609/snr_gain064_r0

  # 整批 capture (自动按 gain 分组)
  python tools/extract_channel_stats.py capture/20260609/snr_gain*_r0

  # 输出仿真标定 JSON
  python tools/extract_channel_stats.py capture/20260609/snr_gain*_r0 -o sim_calibration.json
"""

import argparse, json, os, sys, glob
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from phy_params import (SPS, STF, PSS, RS, RRC,
                        STF_LEN, PSS_LEN, RS_LEN,
                        HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN,
                        STF_DELAY, FRAME_RRC_SAMPLES, RRC_DELAY_SAMPLES,
                        crc16, crc16_check, bits_to_bytes, bytes_to_bits)

SAMP_RATE = 1e6
TS_SYM = SPS / SAMP_RATE


# ======================================================================
# 底层同步函数 (与 polar_loopback.py 完全一致, 不依赖 loopback_analyze)
# ======================================================================

def _rrc_match(samples):
    f = np.convolve(samples, RRC[::-1], mode='full')
    return f[RRC_DELAY_SAMPLES::SPS].astype(np.complex64)


def _stf_scan(samples, stf_threshold=0.4, stf_min_energy=0.01):
    """全量 STF 扫描: 返回所有超过门限的 (position, M_value, coarse_cfo)."""
    L = STF_DELAY
    N = len(samples)
    if N <= L:
        return [], [], []

    r0, rL = samples[:N - L], samples[L:]
    prod = r0 * np.conj(rL)
    ones = np.ones(L, dtype=np.float32)
    P = np.convolve(prod, ones, mode='valid')
    E = np.convolve((np.abs(rL) ** 2).astype(np.float32), ones, mode='valid')
    M = np.abs(P) / (E + 1e-6 * L)

    peaks, metrics, cfos = [], [], []
    for d in range(len(M)):
        if M[d] > stf_threshold:
            le = np.sum(np.abs(samples[d + L:d + 2 * L]) ** 2)
            if le > stf_min_energy:
                peaks.append(d)
                metrics.append(float(M[d]))
                cfos.append(float(-np.angle(P[d]) / (2 * np.pi * L / SAMP_RATE)))
    return peaks, metrics, cfos


def _pss_correlate(syms):
    """PSS 互相关: 返回 (peak_idx, ptm, pts, peak_val)."""
    M = PSS_LEN
    if len(syms) < M:
        return -1, 0.0, 0.0, 0.0
    pss_rev = np.conj(PSS[::-1])
    c = np.abs(np.convolve(syms, pss_rev, mode='valid'))
    pk = int(np.argmax(c))
    peak_val = float(c[pk])
    ptm = peak_val / (np.mean(c) + 1e-30)

    pts = ptm
    sv = np.sort(c)[::-1]
    for v in sv[1:]:
        idx_list = np.where(np.isclose(c, v))[0]
        found = False
        for idx in idx_list:
            if abs(idx - pk) > PSS_LEN // 2:
                pts = peak_val / (v + 1e-30)
                found = True
                break
        if found:
            break
    return pk, ptm, pts, peak_val


def _rs_estimate(syms, rs_pos, coarse_cfo=0.0):
    """RS 信道估计 (完全对齐 polar_loopback.py _rs_estimate).

    返回 dict 或 None (如果任何检查失败).
    """
    if rs_pos + RS_LEN > len(syms):
        return None

    rs_seg = syms[rs_pos:rs_pos + RS_LEN].copy()
    n_rs = np.arange(RS_LEN)

    # 粗 CFO 预补偿
    if abs(coarse_cfo) > 0.0:
        pre_comp = np.exp(-1j * 2 * np.pi * coarse_cfo * (rs_pos + n_rs) * TS_SYM)
        rs_seg = rs_seg * pre_comp

    # 细 CFO: 线性相位拟合
    rs_tone = rs_seg * np.conj(RS)
    rs_corr = float(np.abs(np.sum(rs_tone)))
    rs_phase = np.unwrap(np.angle(rs_tone))
    n = np.arange(RS_LEN, dtype=np.float64)
    n_mean = np.mean(n)
    p_mean = np.mean(rs_phase)
    num = np.sum((n - n_mean) * (rs_phase - p_mean))
    den = np.sum((n - n_mean) ** 2)
    slope = num / (den + 1e-30)
    fine_cfo = float(slope / (2 * np.pi * TS_SYM))

    # 细 CFO 超限
    if abs(fine_cfo) > 500:
        return None

    # 总 CFO 补偿 + 信道估计
    total_cfo = coarse_cfo + fine_cfo
    total_comp = np.exp(-1j * 2 * np.pi * total_cfo * (rs_pos + n_rs) * TS_SYM)
    rs_corrected = syms[rs_pos:rs_pos + RS_LEN] * total_comp

    h = np.mean(rs_corrected * np.conj(RS))
    if abs(h) < 1e-6:
        return None

    # Welch 校正噪声方差
    noise = rs_corrected / h - RS
    sigma2 = max(float(np.sum(np.abs(noise) ** 2) / (RS_LEN - 1)), 1e-30)

    # RS 相关质量 (放宽到 0.1 以捕获弱信号)
    if rs_corr < RS_LEN * 0.1:
        return None

    return {
        'h': h,
        'phase_est': float(np.angle(h)),
        'sigma2': sigma2,
        'coarse_cfo': coarse_cfo,
        'fine_cfo': fine_cfo,
        'total_cfo': total_cfo,
        'rs_corr': rs_corr,
    }


def _bpsk_demod_hard(syms, data_start, data_len, h, total_cfo):
    """BPSK 硬解调."""
    if data_start + data_len > len(syms):
        return np.zeros(data_len, dtype=np.int64)
    seg = syms[data_start:data_start + data_len]
    n = np.arange(data_len)
    cfo_comp = np.exp(-1j * 2 * np.pi * total_cfo * (data_start + n) * TS_SYM)
    y = seg * cfo_comp
    if abs(h) > 1e-30:
        y = y / h
    return (y.real < 0).astype(np.int64)


def _b2i(b):
    v = 0
    for x in b:
        v = (v << 1) | int(x)
    return v


# ======================================================================
# 主提取函数
# ======================================================================

def extract_from_capture(iq_path, stf_threshold=0.4, stf_min_energy=0.01,
                         pss_ptm=2.5, pss_pts=1.0, rs_corr_min=0.1,
                         n_noise=50000):
    """从单个 capture 提取所有候选帧的信道参数.

    Args:
        iq_path: _iq.npy 文件路径
        stf_threshold, stf_min_energy: STF 检测门限 (低默认值以最大化检出)
        pss_ptm, pss_pts: PSS 质量门限 (放宽以捕获弱信号)
        rs_corr_min: RS 相关最小平均值/每符号 (放宽到 0.1)
        n_noise: 底噪测量 IQ 样本数

    Returns:
        dict with keys:
          - noise_floor: float
          - detections: list[dict]  每候选帧的信道参数
          - meta: dict  元数据 (如果有 _meta.json)
          - n_iq_samples: int
    """
    iq = np.load(iq_path)
    n_total = len(iq)

    # ── 底噪 ──
    n_nf = min(n_noise, n_total)
    noise_syms = _rrc_match(iq[:n_nf])
    noise_floor = float(np.var(noise_syms))

    # ── 元数据 ──
    meta = {}
    meta_path = iq_path.replace('_iq.npy', '_meta.json')
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # ── 全量 STF 扫描 ──
    # 分段处理以避免内存爆炸
    seg_size = 1_000_000
    overlap = FRAME_RRC_SAMPLES + 5000
    all_peaks, all_metrics, all_cfos = [], [], []

    pos = 0
    while pos < n_total:
        end = min(pos + seg_size, n_total)
        seg = iq[pos:end]
        p, m, c = _stf_scan(seg, stf_threshold, stf_min_energy)
        all_peaks.extend([x + pos for x in p])
        all_metrics.extend(m)
        all_cfos.extend(c)
        pos += seg_size - min(overlap, seg_size // 2)
        if pos >= n_total:
            break

    # ── CFO 过滤: 只保留合理 CFO 的候选 (B210 同板 < 200 Hz) ──
    # 这是关键: 假 STF 峰的 CFO 随机分布在 ±2kHz,
    # 而真帧的 CFO 集中在 0±50Hz
    valid = [i for i, c in enumerate(all_cfos) if abs(c) < 200]
    all_peaks = [all_peaks[i] for i in valid]
    all_metrics = [all_metrics[i] for i in valid]
    all_cfos = [all_cfos[i] for i in valid]

    # ── 聚类去重 ──
    # 在 FRAME_RRC_SAMPLES//2 窗口内只保留最强峰
    if all_peaks:
        arr = list(zip(all_peaks, all_cfos, all_metrics))
        arr.sort(key=lambda x: x[2], reverse=True)
        used = set()
        c_peaks, c_cfos, c_metrics = [], [], []
        win = FRAME_RRC_SAMPLES // 2
        for d, cfo, m in arr:
            if d in used:
                continue
            for dx in range(max(0, d - win), d + win + 1):
                used.add(dx)
            c_peaks.append(d)
            c_cfos.append(cfo)
            c_metrics.append(m)
    else:
        c_peaks, c_cfos, c_metrics = [], [], []

    # ── 逐候选同步 ──
    detections = []
    for d, coarse_cfo, stf_m in zip(c_peaks, c_cfos, c_metrics):
        # 提取窗口
        margin = 400
        es = max(0, d - margin)
        ee = min(n_total, d + margin + FRAME_RRC_SAMPLES + margin)
        if ee - es < FRAME_RRC_SAMPLES:
            continue

        syms = _rrc_match(iq[es:ee])
        if len(syms) < PSS_LEN + RS_LEN:
            continue

        # PSS
        pk, ptm, pts, pval = _pss_correlate(syms)
        if ptm < pss_ptm or pts < pss_pts:
            continue

        fs = pk - STF_LEN
        if fs < 0:
            continue

        rp = fs + STF_LEN + PSS_LEN
        if rp + RS_LEN + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN > len(syms):
            continue

        # RS
        chan = _rs_estimate(syms, rp, coarse_cfo)
        if chan is None:
            continue
        # 额外 RS 相关检查
        if chan['rs_corr'] < RS_LEN * rs_corr_min:
            continue

        hmag = abs(chan['h'])
        snr_sym = float(10 * np.log10(max(hmag ** 2 / max(noise_floor, 1e-30), 1e-30)))
        snr_rs = float(10 * np.log10(max(hmag ** 2 / max(chan['sigma2'], 1e-30), 1e-30)))

        # 定时偏 (PSS 二次插值)
        # 需要 pss_corr 序列 — 重新算一次代价不大
        pss_rev = np.conj(PSS[::-1])
        pss_corr = np.abs(np.convolve(syms, pss_rev, mode='valid'))
        if pk > 0 and pk < len(pss_corr) - 1:
            c0, c1, c2 = pss_corr[pk - 1], pss_corr[pk], pss_corr[pk + 1]
            denom = 2 * (c0 - 2 * c1 + c2)
            toff = float((c0 - c2) / (denom + 1e-30)) if abs(denom) > 1e-9 else 0.0
        else:
            toff = 0.0

        # 尝试解调 Header (用于验证, 不强制)
        hdr_start = rp + RS_LEN
        hdr_bits = _bpsk_demod_hard(syms, hdr_start, HEADER_LEN,
                                     chan['h'], chan['total_cfo'])
        hdr_ok = False
        frame_id = -1
        if len(hdr_bits) >= 32:
            frame_id = _b2i(hdr_bits[:16])
            hdr_ok = crc16_check(bits_to_bytes(hdr_bits[:16]),
                                 _b2i(hdr_bits[16:32]))

        # 尝试解调 Payload CRC
        pay_start = hdr_start + HEADER_LEN
        pay_bits = _bpsk_demod_hard(syms, pay_start,
                                     PAYLOAD_LEN + PAYLOAD_CRC_LEN,
                                     chan['h'], chan['total_cfo'])
        crc_ok = False
        if len(pay_bits) >= PAYLOAD_LEN + PAYLOAD_CRC_LEN:
            payload = pay_bits[:PAYLOAD_LEN]
            crc_val = _b2i(pay_bits[PAYLOAD_LEN:])
            crc_ok = crc16_check(bits_to_bytes(payload), crc_val)

        detections.append({
            'global_pos': d,
            'stf_metric': stf_m,
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
            'timing_offset': toff,
            'frame_id': frame_id,
            'hdr_ok': hdr_ok,
            'crc_ok': crc_ok,
        })

    return {
        'noise_floor': noise_floor,
        'detections': detections,
        'meta': meta,
        'n_iq_samples': n_total,
    }


# ======================================================================
# 批量处理 + 汇总
# ======================================================================

def process_captures(prefixes, **kwargs):
    """处理多个 capture, 按 gain 分组统计.

    Args:
        prefixes: list of capture 前缀 (如 ['capture/20260609/snr_gain064_r0', ...])
        **kwargs: 传递给 extract_from_capture

    Returns:
        dict: {
            'per_capture': {prefix: extract_from_capture result},
            'by_gain': {gain_db: aggregated_stats},
            'cross_gain': {  # 跨 gain 一致性检查
                'cfo_consistency': ...,
                'phase_consistency': ...,
                'timing_consistency': ...,
            },
        }
    """
    per_capture = {}
    by_gain = {}

    for prefix in prefixes:
        iq_path = prefix + '_iq.npy'
        if not os.path.isfile(iq_path):
            print(f"  [skip] 找不到 {iq_path}")
            continue

        print(f"  extracting: {os.path.basename(prefix)} ...")
        result = extract_from_capture(iq_path, **kwargs)
        per_capture[prefix] = result

        gain = result['meta'].get('gain_rx_db', 'unknown')
        if gain not in by_gain:
            by_gain[gain] = {
                'detections': [],
                'noise_floors': [],
                'prefixes': [],
            }
        by_gain[gain]['detections'].extend(result['detections'])
        by_gain[gain]['noise_floors'].append(result['noise_floor'])
        by_gain[gain]['prefixes'].append(prefix)

    # ── 每组 gain 的统计 ──
    for gain, data in by_gain.items():
        dets = data['detections']
        n = len(dets)
        if n == 0:
            data['stats'] = {'n_detections': 0}
            continue

        cfos = [d['total_cfo'] for d in dets]
        phases = [d['phase_est'] for d in dets]
        toffs = [d['timing_offset'] for d in dets]
        h_mags = [d['h_mag'] for d in dets]
        snrs = [d['snr_symbol'] for d in dets]
        snrs_rs = [d['snr_rs'] for d in dets]
        crc_ok = sum(1 for d in dets if d['crc_ok'])
        hdr_ok = sum(1 for d in dets if d['hdr_ok'])

        data['stats'] = {
            'n_detections': n,
            'crc_ok': crc_ok,
            'hdr_ok': hdr_ok,
            'cfo': {
                'mean': float(np.mean(cfos)),
                'std': float(np.std(cfos)),
                'min': float(np.min(cfos)),
                'max': float(np.max(cfos)),
                'median': float(np.median(cfos)),
            },
            'phase': {
                'mean': float(np.mean(phases)),
                'std': float(np.std(phases)),
            },
            'timing_offset': {
                'mean': float(np.mean(toffs)),
                'std': float(np.std(toffs)),
            },
            'h_mag': {
                'mean': float(np.mean(h_mags)),
                'std': float(np.std(h_mags)),
            },
            'snr_symbol': {
                'mean': float(np.mean(snrs)),
                'std': float(np.std(snrs)),
                'min': float(np.min(snrs)),
                'max': float(np.max(snrs)),
            },
            'snr_rs': {
                'mean': float(np.mean(snrs_rs)),
                'std': float(np.std(snrs_rs)),
            },
            'noise_floor': {
                'mean': float(np.mean(data['noise_floors'])),
                'std': float(np.std(data['noise_floors'])),
            },
        }

    # ── 跨 gain 一致性检查 (验证物理假设) ──
    gains_with_frames = {g: d for g, d in by_gain.items()
                         if d['stats'].get('n_detections', 0) > 0}

    cross_gain = {}
    if len(gains_with_frames) >= 2:
        # CFO 是否一致？
        cfo_means = [d['stats']['cfo']['mean'] for d in gains_with_frames.values()]
        cfo_stds = [d['stats']['cfo']['std'] for d in gains_with_frames.values()]
        cross_gain['cfo_spread_hz'] = float(np.std(cfo_means))
        cross_gain['cfo_consistent'] = cross_gain['cfo_spread_hz'] < 20

        # 定时是否一致？
        toff_means = [d['stats']['timing_offset']['mean']
                      for d in gains_with_frames.values()]
        cross_gain['timing_spread_sym'] = float(np.std(toff_means))
        cross_gain['timing_consistent'] = cross_gain['timing_spread_sym'] < 0.1

    return {
        'per_capture': {os.path.basename(k): v for k, v in per_capture.items()},
        'by_gain': {str(k): v for k, v in by_gain.items()},
        'cross_gain': cross_gain,
    }


# ======================================================================
# 仿真标定配置生成
# ======================================================================

def make_simulation_config(results):
    """从提取结果生成 sim_channel.py 可用的标定配置.

    策略:
      - CFO: 从高 SNR capture 拟合 N(mean, std)
      - 相偏: 均匀 [0, 2π) (物理必然)
      - 定时偏: 取高 SNR capture 的 std
      - noise_floor: 用目标 gain 档的实测值
      - |h|: 用目标 gain 档的实测均值
    """
    # 找最高 SNR 的 gain (检测数最多的)
    best_gain = None
    best_n = 0
    for gain_str, data in results['by_gain'].items():
        n = data['stats'].get('n_detections', 0)
        if n > best_n:
            best_n = n
            best_gain = gain_str

    if best_gain is None:
        return {'error': 'no detections in any capture'}

    ref = results['by_gain'][best_gain]['stats']

    config = {
        'description': 'Capture-driven simulation calibration',
        'source_gain_db': best_gain,
        'source_n_frames': best_n,
        'channel_model': {
            'cfo_hz': {
                'distribution': 'normal',
                'mean': ref['cfo']['mean'],
                'std': max(ref['cfo']['std'], 1.0),  # 至少 1Hz
            },
            'phase_offset_rad': {
                'distribution': 'uniform',
                'min': -np.pi,
                'max': np.pi,
            },
            'timing_offset_sym': {
                'distribution': 'normal',
                'mean': 0.0,
                'std': max(ref['timing_offset']['std'], 0.001),
            },
        },
        'per_gain': {},
    }

    for gain_str, data in results['by_gain'].items():
        s = data['stats']
        if s.get('n_detections', 0) == 0:
            config['per_gain'][gain_str] = {
                'noise_floor': float(np.mean(data['noise_floors'])),
                'h_mag_expected': None,
                'snr_expected_db': None,
                'note': 'No frames synced — noise_floor from prefix only',
            }
        else:
            config['per_gain'][gain_str] = {
                'noise_floor': float(np.mean(data['noise_floors'])),
                'h_mag_mean': s['h_mag']['mean'],
                'h_mag_std': s['h_mag']['std'],
                'snr_symbol_mean_db': s['snr_symbol']['mean'],
                'snr_symbol_std_db': s['snr_symbol']['std'],
                'snr_rs_mean_db': s['snr_rs']['mean'],
                'n_detections': s['n_detections'],
                'crc_ok': s['crc_ok'],
                'cfo_hz_mean': s['cfo']['mean'],
                'cfo_hz_std': s['cfo']['std'],
            }

    return config


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(
        description='从 capture 提取信道参数 → 仿真标定')
    p.add_argument('prefixes', nargs='+',
                   help='capture 前缀 (支持 glob) 或目录')
    p.add_argument('-o', '--output', default='',
                   help='输出 JSON (仿真标定配置)')
    p.add_argument('--stf-threshold', type=float, default=0.4,
                   help='STF 相关门限 (默认 0.4)')
    p.add_argument('--stf-energy', type=float, default=0.01,
                   help='STF 能量门限 (默认 0.01, 宽松以最大化检出)')
    p.add_argument('--pss-ptm', type=float, default=2.5,
                   help='PSS ptm 门限 (默认 2.5)')
    p.add_argument('--pss-pts', type=float, default=1.0,
                   help='PSS pts 门限 (默认 1.0)')
    p.add_argument('--rs-corr-min', type=float, default=0.1,
                   help='RS 相关最小值/每符号 (默认 0.1)')
    p.add_argument('--stats-only', action='store_true',
                   help='只输出统计摘要, 不输出逐帧数据 (减小 JSON)')
    args = p.parse_args()

    # 展开 glob
    prefixes = []
    for pat in args.prefixes:
        if '*' in pat or '?' in pat:
            # glob 模式
            matches = sorted(glob.glob(pat))
            # 去 _iq/_bits/_meta 后缀
            seen = set()
            for m in matches:
                base = m.replace('_iq.npy', '').replace('_bits.npy', '').replace('_meta.json', '')
                if base not in seen:
                    seen.add(base)
                    prefixes.append(base)
        elif os.path.isdir(pat):
            # 目录: 找所有 _iq.npy
            for f in sorted(glob.glob(os.path.join(pat, '*_iq.npy'))):
                prefixes.append(f.replace('_iq.npy', ''))
        else:
            # 直接前缀 (去掉可能的后缀)
            base = pat.replace('_iq.npy', '').replace('_bits.npy', '').replace('_meta.json', '')
            prefixes.append(base)

    if not prefixes:
        print("错误: 未找到任何 capture 文件")
        sys.exit(1)

    print(f"处理 {len(prefixes)} 个 capture:")
    for pfx in prefixes:
        print(f"  {os.path.basename(pfx)}")

    # 提取
    results = process_captures(
        prefixes,
        stf_threshold=args.stf_threshold,
        stf_min_energy=args.stf_energy,
        pss_ptm=args.pss_ptm,
        pss_pts=args.pss_pts,
        rs_corr_min=args.rs_corr_min,
    )

    # 打印统计
    print(f"\n{'='*70}")
    print(f"信道参数提取报告")
    print(f"{'='*70}")

    for gain_str in sorted(results['by_gain'].keys(), key=lambda x: float(x) if x != 'unknown' else 0):
        data = results['by_gain'][gain_str]
        s = data['stats']
        n = s.get('n_detections', 0)
        print(f"\n── gain={gain_str} dB ──")
        print(f"  底噪: {np.mean(data['noise_floors']):.2e}  "
              f"({10*np.log10(max(np.mean(data['noise_floors']), 1e-30)):.1f} dB)")
        if n == 0:
            print(f"  检出: 0 帧 (信号太弱, STF/PSS/RS 均未通过)")
            print(f"  -> noise_floor 仍可用于仿真噪声标定")
        else:
            print(f"  检出: {n} 帧  "
                  f"HDR={s['hdr_ok']}/{n}  CRC={s['crc_ok']}/{n}")
            print(f"  SNR (symbol): {s['snr_symbol']['mean']:.1f} ± {s['snr_symbol']['std']:.1f} dB")
            print(f"  SNR (RS):     {s['snr_rs']['mean']:.1f} ± {s['snr_rs']['std']:.1f} dB")
            print(f"  CFO:          {s['cfo']['mean']:+.1f} ± {s['cfo']['std']:.1f} Hz  "
                  f"[{s['cfo']['min']:+.1f}, {s['cfo']['max']:+.1f}]")
            print(f"  |h|:          {s['h_mag']['mean']:.3f} ± {s['h_mag']['std']:.3f}")
            print(f"  phase:        {s['phase']['mean']:+.3f} ± {s['phase']['std']:.3f} rad")
            print(f"  timing:       {s['timing_offset']['mean']:+.4f} ± {s['timing_offset']['std']:.4f} sym")

    # 跨 gain 一致性
    cg = results['cross_gain']
    if cg:
        print(f"\n── 跨 gain 一致性验证 (物理假设) ──")
        cfo_ok = cg.get('cfo_consistent', False)
        tim_ok = cg.get('timing_consistent', False)
        print(f"  CFO 一致性:    {'[OK]' if cfo_ok else '[WARN]'}  "
              f"(跨 gain CFO mean 散布={cg.get('cfo_spread_hz', 0):.1f} Hz)")
        print(f"  定时一致性:    {'[OK]' if tim_ok else '[WARN]'}  "
              f"(跨 gain timing mean 散布={cg.get('timing_spread_sym', 0):.4f} sym)")
        if cfo_ok and tim_ok:
            print(f"  -> 物理假设成立: CFO/定时不依赖 RX gain")
            print(f"  -> 高 SNR 测得的 CFO/phase/timing 可直接用于低 SNR 仿真")
        else:
            print(f"  -> 需要检查: 低 SNR 下同步参数可能被噪声污染")

    # 输出
    if args.output:
        if args.stats_only:
            # 只输出统计, 不输出逐帧数据
            output = {
                'by_gain': {g: {'stats': d['stats'], 'noise_floors': d['noise_floors']}
                            for g, d in results['by_gain'].items()},
                'cross_gain': results['cross_gain'],
            }
        else:
            # 完整输出 (含每帧数据) + 仿真标定
            sim_config = make_simulation_config(results)
            output = {
                'by_gain': {g: {'stats': d['stats'],
                                'noise_floors': d['noise_floors'],
                                'n_detections': len(d['detections'])}
                            for g, d in results['by_gain'].items()},
                'cross_gain': results['cross_gain'],
                'simulation_calibration': sim_config,
            }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n标定配置已保存 → {args.output}")

        if 'simulation_calibration' in output and 'error' not in output['simulation_calibration']:
            sc = output['simulation_calibration']
            print(f"\n  [sim_channel.py 参数建议]")
            print(f"  --freq-offset ~ N({sc['channel_model']['cfo_hz']['mean']:.0f}, "
                  f"{sc['channel_model']['cfo_hz']['std']:.0f}) Hz")
            print(f"  --phase-offset ~ U(-pi, pi) rad")
            for g, gc in sc['per_gain'].items():
                if gc.get('snr_symbol_mean_db') is not None:
                    print(f"  gain={g}dB: SNR~{gc['snr_symbol_mean_db']:.1f}dB  "
                          f"|h|~{gc['h_mag_mean']:.3f}")


if __name__ == '__main__':
    main()
