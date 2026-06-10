#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_repeat_capture.py — 分析 5x 重复帧 capture (any-of-N 检出 + frame_id 去重)

对应 loopback_capture.py 的 REPEAT=5 发送模式:
  每组 5 帧完全相同 (frame_id 相同), 间隔 3ms, 组间 5ms.
  接收端对每组 5 帧独立跑 STF->PSS->RS 同步,
  任意一帧通过 -> 组检出成功, 用 frame_id 去重.

用法:
  python tools/analyze_repeat_capture.py capture/low_snr
  python tools/analyze_repeat_capture.py capture/low_snr --gain 15
  python tools/analyze_repeat_capture.py capture/low_snr --all -o report.json
  python tools/analyze_repeat_capture.py capture/low_snr --plot
  python tools/analyze_repeat_capture.py capture/low_snr --save-plot plots
"""

import argparse, json, os, sys, time, glob
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from phy_params import (SPS, PSS as PSS_REF, RRC,
                        STF_LEN, PSS_LEN, RS_LEN,
                        HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN,
                        GUARD_SYMBOLS, FRAME_SYMBOLS,
                        RRC_DELAY_SAMPLES, STF_DELAY,
                        FRAME_RRC_SAMPLES,
                        crc16, bits_to_bytes, bytes_to_bits)

SAMP_RATE = 1e6
REPEAT = 5
GAP_REPEAT_IQ = int(0.003 * SAMP_RATE)  # 3000
GAP_GROUP_IQ  = int(0.005 * SAMP_RATE)  # 5000

FROZEN_PATH = os.path.join(BASE, 'deploy', 'matrices', 'A.npy')
FROZEN_MASK = np.load(FROZEN_PATH).astype(np.int64).squeeze()


def make_stf(n_reps=8, base_len=16):
    rng = np.random.RandomState(7)
    base = 2 * rng.randint(0, 2, base_len) - 1
    return np.tile(base, n_reps).astype(np.complex64)


def make_rs(n_syms=64):
    rng = np.random.RandomState(13)
    return (2 * rng.randint(0, 2, n_syms) - 1).astype(np.complex64)


def rrc_filter(symbols, rrc, sps):
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    return np.convolve(up, rrc, mode='full').astype(np.complex64)


def build_custom_frame(data_bits, frame_id, stf_syms, pss_syms, rs_syms):
    payload_crc = crc16(bits_to_bytes(data_bits))
    crc_bits = bytes_to_bits(
        np.array([(payload_crc >> 8) & 0xFF, payload_crc & 0xFF], dtype=np.uint8), 16)
    id_bytes = np.array([(frame_id >> 8) & 0xFF, frame_id & 0xFF], dtype=np.uint8)
    id_bits = bytes_to_bits(id_bytes, 16)
    header_crc = crc16(id_bytes)
    header_crc_bits = bytes_to_bits(
        np.array([(header_crc >> 8) & 0xFF, header_crc & 0xFF], dtype=np.uint8), 16)
    header_bits = np.concatenate([id_bits, header_crc_bits])

    def _bpsk(bits):
        return (1.0 - 2.0 * bits).astype(np.float32)

    return np.concatenate([
        stf_syms.astype(np.complex64),
        pss_syms.astype(np.complex64),
        rs_syms.astype(np.complex64),
        _bpsk(header_bits).astype(np.complex64),
        _bpsk(data_bits).astype(np.complex64),
        _bpsk(crc_bits).astype(np.complex64),
        np.zeros(GUARD_SYMBOLS, dtype=np.complex64),
    ])


def pss_correlate_custom(syms, pss_ref):
    m = len(pss_ref)
    if len(syms) < m:
        return -1, 0.0, 0.0, 0.0
    c = np.abs(np.convolve(syms, np.conj(pss_ref[::-1]), mode='valid'))
    pk = int(np.argmax(c))
    peak_val = float(c[pk])
    ptm = peak_val / (np.mean(c) + 1e-30)
    pts = ptm
    for v in np.sort(c)[::-1][1:]:
        for idx in np.where(np.isclose(c, v))[0]:
            if abs(idx - pk) > m // 2:
                pts = peak_val / (v + 1e-30)
                break
        if pts != ptm:
            break
    return pk, ptm, pts, peak_val


TS_SYM = SPS / SAMP_RATE


def rs_estimate_custom(syms, rs_pos, rs_ref, coarse_cfo=0.0):
    rs_len = len(rs_ref)
    if rs_pos + rs_len > len(syms):
        return None
    rs_seg = syms[rs_pos:rs_pos + rs_len].copy()
    rs_tone = rs_seg * np.conj(rs_ref)
    rs_corr = float(np.abs(np.sum(rs_tone)))
    phase = np.angle(rs_tone)
    n = np.arange(rs_len, dtype=np.float64)
    n_mean = np.mean(n); p_mean = np.mean(phase)
    den = np.sum((n - n_mean) ** 2)
    if den < 1e-30:
        return None
    slope = np.sum((n - n_mean) * (phase - p_mean)) / den
    total_cfo = float(slope / (2 * np.pi * TS_SYM))
    if abs(total_cfo) > 2000:
        return None
    total_comp = np.exp(-1j * 2 * np.pi * total_cfo * (rs_pos + n) * TS_SYM)
    rs_corrected = syms[rs_pos:rs_pos + rs_len] * total_comp
    h = np.mean(rs_corrected * np.conj(rs_ref))
    if abs(h) < 1e-6:
        return None
    noise = rs_corrected / h - rs_ref
    sigma2 = max(float(np.sum(np.abs(noise) ** 2) / max(rs_len - 1, 1)), 1e-30)
    if rs_corr < rs_len * 0.1:
        return None
    return {'h': h, 'phase_est': float(np.angle(h)), 'sigma2': sigma2,
            'coarse_cfo': 0.0, 'fine_cfo': total_cfo,
            'total_cfo': total_cfo, 'rs_corr': rs_corr}


def _local_peak(corr, center, radius=16):
    """在 center 附近找局部相关峰。"""
    if len(corr) == 0:
        return -1, 0.0
    lo = max(0, int(center) - radius)
    hi = min(len(corr), int(center) + radius + 1)
    if hi <= lo:
        return -1, 0.0
    rel = int(np.argmax(corr[lo:hi]))
    pos = lo + rel
    return pos, float(corr[pos])


def _find_first_repeat_offset(corr, repeat_stride, n_repeats=REPEAT,
                              search_radius=24, rel_threshold=0.35):
    """从 frame0 的相关峰恢复 group0/repeat0 起点。

    frame0 会连续重复 n_repeats 次，直接 argmax 可能落在任意一个 repeat。
    这里从最大峰按 repeat_stride 向前回溯，只要预期位置附近仍有足够强的
    相关峰，就把它视为更早的重复副本。
    """
    if len(corr) == 0:
        return -1, {'peak': 0.0, 'repeat_index_guess': 0, 'peaks': []}

    best_pos = int(np.argmax(corr))
    best_val = float(corr[best_pos])
    pos = best_pos
    peaks = [{'pos': best_pos, 'val': best_val}]
    repeat_index_guess = 0

    for _ in range(n_repeats - 1):
        expected = pos - repeat_stride
        if expected < 0:
            break
        prev_pos, prev_val = _local_peak(corr, expected, search_radius)
        if prev_pos < 0 or prev_val < best_val * rel_threshold:
            break
        pos = prev_pos
        repeat_index_guess += 1
        peaks.append({'pos': prev_pos, 'val': prev_val})

    peaks.sort(key=lambda x: x['pos'])
    return int(pos), {
        'peak': best_val,
        'repeat_index_guess': repeat_index_guess,
        'peaks': peaks,
    }


def detect_one_frame(syms, stf_syms, pss_syms, rs_syms,
                     pss_ptm=2.5, pss_pts=1.0, stf_energy=0.01,
                     noise_floor=None):
    """对一段符号序列跑 PSS->RS, 返回同步诊断 dict。

    noise_floor: 前缀底噪 (符号域), 为 None 时使用 rs_estimate_custom 自带的硬门限;
                 不为 None 时用自适应门限: |h_est|²/noise_floor > 3dB 即通过。
    """
    stf_len = len(stf_syms)
    rs_len = len(rs_syms)
    if len(syms) < len(pss_syms) + rs_len:
        return {'detected': False, 'fail_stage': 'short', 'n_syms': len(syms)}

    pk, ptm, pts, pv = pss_correlate_custom(syms, pss_syms)
    base = {
        'pk': int(pk), 'ptm': float(ptm), 'pts': float(pts),
        'pval': float(pv), 'n_syms': int(len(syms)),
    }
    if ptm < pss_ptm or pts < pss_pts:
        return {**base, 'detected': False, 'fail_stage': 'PSS_THRESH'}

    fs = pk - stf_len
    base['fs'] = int(fs)
    if fs < 0:
        return {**base, 'detected': False, 'fail_stage': 'FS_NEG'}

    rp = fs + stf_len + len(pss_syms)
    base['rs_pos'] = int(rp)
    if rp + rs_len > len(syms):
        return {**base, 'detected': False, 'fail_stage': 'RS_OOB'}

    chan = rs_estimate_custom(syms, rp, rs_syms)
    rs_diag = {}
    if chan is None and rp + rs_len <= len(syms):
        rs_seg = syms[rp:rp + rs_len]
        rs_tone = rs_seg * np.conj(rs_syms)
        rs_corr_d = float(np.abs(np.sum(rs_tone)))
        h_est = rs_corr_d / rs_len
        snr_rs_linear = h_est ** 2 / max(noise_floor or 1e-30, 1e-30)
        rs_diag = {'rs_corr_raw': rs_corr_d, 'h_est_rs': h_est,
                   'snr_rs_linear': snr_rs_linear,
                   'thr_corr': rs_len * 0.1}

        # 自适应重试: 如果 rs_corr 相对底噪足够强, 用简化的信道估计
        if noise_floor is not None and snr_rs_linear > 2.0:
            rs_phase_d = np.angle(rs_tone)
            n_d = np.arange(rs_len, dtype=np.float64)
            n_mean_d = np.mean(n_d); p_mean_d = np.mean(rs_phase_d)
            num_d = np.sum((n_d - n_mean_d) * (rs_phase_d - p_mean_d))
            den_d = np.sum((n_d - n_mean_d) ** 2)
            total_cfo = float((num_d / (den_d + 1e-30)) / (2 * np.pi * TS_SYM))
            if abs(total_cfo) < 2000:
                total_comp = np.exp(-1j * 2 * np.pi * total_cfo * (rp + n_d) * TS_SYM)
                rs_corrected = syms[rp:rp + rs_len] * total_comp
                h = np.mean(rs_corrected * np.conj(rs_syms))
                if abs(h) > 1e-6:
                    noise = rs_corrected / h - rs_syms
                    sigma2 = max(float(np.sum(np.abs(noise)**2) / (rs_len - 1)), 1e-30)
                    return {
                        **base, 'detected': True, 'fail_stage': 'OK',
                        'total_cfo': total_cfo, 'h_mag': float(abs(h)),
                        'sigma2': sigma2, 'rs_corr': rs_corr_d,
                    }
    if chan is None:
        return {**base, 'detected': False, 'fail_stage': 'RS_FAIL', **rs_diag}

    return {
        **base,
        'detected': True,
        'fail_stage': 'OK',
        'total_cfo': float(chan['total_cfo']),
        'h_mag': float(abs(chan['h'])),
        'sigma2': float(chan['sigma2']),
        'rs_corr': float(chan['rs_corr']),
    }


def analyze_capture(prefix, num_frames=40, pss_ptm=2.5, pss_pts=1.0,
                    fixed_nf_db=None):
    """分析单个 capture 的 5x 重复帧检出率.

    Args:
        prefix:     capture 前缀 (如 capture/low_snr/snr_gain015_r0)
        num_frames: 唯一帧数 (组数)

    Returns:
        dict: {groups_ok, total_groups, per_group_hits, ...}
    """
    iq_path = prefix + '_iq.npy'
    meta_path = prefix + '_meta.json'

    if not os.path.isfile(iq_path):
        return {'error': f'no file: {iq_path}'}

    iq = np.load(iq_path)
    n_total = len(iq)

    # 前缀噪声底噪 (符号域, 用于 SNR 计算)
    noise_floor_measured = _prefix_noise_floor(iq)
    if fixed_nf_db is not None:
        noise_floor = 10 ** (fixed_nf_db / 10)
    else:
        noise_floor = noise_floor_measured
    nf_db = 10 * np.log10(max(noise_floor, 1e-30))

    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)

    # 帧间隔从 meta 读取 (短前导: 5ms/3ms, 长前导 Polar: 30ms/5ms)
    gap_group_ms = meta.get('frame_gap_ms', 5.0)
    gap_repeat_ms = meta.get('gap_repeat_ms', 3.0)
    GAP_GROUP_IQ  = int(gap_group_ms * SAMP_RATE / 1000)
    GAP_REPEAT_IQ = int(gap_repeat_ms * SAMP_RATE / 1000)
    # STF/RS 长度也从 meta 读 (短前导: 64/32, 长前导 Polar: 128/64)
    stf_len_sym = meta.get('stf_syms', STF_LEN)
    rs_len_sym  = meta.get('rs_syms', RS_LEN)

    # 帧结构参数
    stf_reps = stf_len_sym // 16
    stf_syms = make_stf(stf_reps)
    rs_syms = make_rs(rs_len_sym)
    pss_syms = PSS_REF
    total_sym = (len(stf_syms) + len(pss_syms) + len(rs_syms)
                 + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN + GUARD_SYMBOLS)
    frame_iq_len = total_sym * SPS + len(RRC) - 1

    # TX 时序: [pad] [F0×5 + gaps] [GAP_GROUP] [F1×5 + gaps] ...
    # RX 开始采集早于 TX (~1s), 第一帧起始约在 IQ 开头后 50000 样本
    # 用全帧互相关定位第一帧 (与 channel_measure.py 相同)
    # 检测长前导 Polar 编码帧
    use_polar = stf_len_sym > 64 or rs_len_sym > 32

    # TX 时序定位
    rng = np.random.RandomState(42)
    if use_polar:
        # Polar capture: rng.rand(K_POLAR) 而不是 rng.randint
        from tools.loopback_capture import build_frame, rrc_pulse
        import math
        def _polar_enc(u):
            cw = u.copy().ravel()
            for stage in range(1, int(math.log2(256)) + 1):
                sep = 256 // (1 << stage)
                for j in range(256):
                    if (j // sep) % 2 == 0:
                        cw[j] = (cw[j] + cw[j + sep]) % 2
            return cw
        info = (rng.rand(128) < 0.5).astype(np.int64)
        u = np.zeros(256, dtype=np.int64)
        u[FROZEN_MASK.astype(bool)] = info
        coded = _polar_enc(u)
        tx_iq_ref = rrc_pulse(
            build_frame(coded, 0, stf_syms, pss_syms, rs_syms),
            RRC, SPS).astype(np.complex64)
    else:
        tx_iq_ref, _, _ = __reconstruct_first_frame(rng, stf_syms, pss_syms, rs_syms)

    # 全帧互相关找 frame0，并回溯到 group0/repeat0
    tx_rev = np.conj(tx_iq_ref[::-1])
    corr = np.abs(np.convolve(iq[:min(2000000, n_total)], tx_rev, mode='valid'))
    repeat_stride = frame_iq_len + GAP_REPEAT_IQ
    first_offset, first_info = _find_first_repeat_offset(corr, repeat_stride)
    if first_offset < 0:
        return {'error': 'empty correlation'}
    print(f"    第一帧 @ IQ[{first_offset}], corr_peak={first_info['peak']:.1f} "
          f"(argmax回溯 {first_info['repeat_index_guess']} repeats)")

    groups_ok = 0
    hits_per_group = []
    total_groups = 0
    fail_counts = {}
    pss_ptm_vals, pss_pts_vals = [], []
    rs_corr_vals, h_mag_vals, cfo_vals, sigma2_vals = [], [], [], []
    rs_fail_corr_vals, rs_fail_cfo_vals = [], []  # RS 失败帧的诊断

    for gi in range(num_frames):
        group_offset = first_offset + gi * (
            REPEAT * frame_iq_len + (REPEAT - 1) * GAP_REPEAT_IQ + GAP_GROUP_IQ)
        n_hits = 0

        for ri in range(REPEAT):
            offset = group_offset + ri * repeat_stride
            if offset + frame_iq_len > n_total:
                fail_counts['OOB'] = fail_counts.get('OOB', 0) + 1
                break

            # 提取 + RRC 匹配
            margin = 400
            es = max(0, offset - margin)
            ee = min(n_total, offset + frame_iq_len + margin)
            chunk = iq[es:ee]
            syms = np.convolve(chunk, RRC[::-1], mode='full')[RRC_DELAY_SAMPLES::SPS].astype(np.complex64)

            result = detect_one_frame(syms, stf_syms, pss_syms, rs_syms,
                                      pss_ptm, pss_pts,
                                      noise_floor=noise_floor)
            stage = result.get('fail_stage', 'unknown')
            fail_counts[stage] = fail_counts.get(stage, 0) + 1
            if 'ptm' in result:
                pss_ptm_vals.append(result['ptm'])
            if 'pts' in result:
                pss_pts_vals.append(result['pts'])
            if result.get('detected'):
                n_hits += 1
                rs_corr_vals.append(result['rs_corr'])
                h_mag_vals.append(result['h_mag'])
                cfo_vals.append(result['total_cfo'])
                sigma2_vals.append(result['sigma2'])
            elif result.get('fail_stage') == 'RS_FAIL':
                if 'rs_corr_raw' in result:
                    rs_fail_corr_vals.append(result['rs_corr_raw'])
                if 'cfo_raw' in result:
                    rs_fail_cfo_vals.append(result['cfo_raw'])

        if n_hits > 0:
            groups_ok += 1
        hits_per_group.append(n_hits)
        total_groups += 1

    # --- 全帧信道测量 (oracle-based, 独立于同步链) ---
    oracle_h_mag, oracle_cfo, oracle_phase = [], [], []
    oracle_r2 = []
    oracle_rng = np.random.RandomState(42)  # 每帧独立重建
    for gi in range(num_frames):
        group_offset = first_offset + gi * (
            REPEAT * frame_iq_len + (REPEAT - 1) * GAP_REPEAT_IQ + GAP_GROUP_IQ)
        om = _oracle_measure_one_frame(iq, group_offset, frame_iq_len,
                                       oracle_rng, gi,
                                       stf_syms, pss_syms, rs_syms,
                                       use_polar=use_polar)
        if om is not None:
            oracle_h_mag.append(om['h_mag'])
            oracle_cfo.append(om['cfo_hz'])
            oracle_phase.append(om['h_phase'])
            oracle_r2.append(om['cfo_r_squared'])

    oracle_snr_db = None
    if oracle_h_mag:
        oracle_snr_db = float(10 * np.log10(
            max(np.mean(oracle_h_mag) ** 2 / max(noise_floor, 1e-30), 1e-30)))

    diagnostics = {
        'first_offset': first_offset,
        'first_peak': first_info['peak'],
        'first_repeat_index_guess': first_info['repeat_index_guess'],
        'n_total': n_total,
        'frame_iq_len': frame_iq_len,
        'gap_repeat_iq': GAP_REPEAT_IQ,
        'gap_group_iq': GAP_GROUP_IQ,
        'fail_counts': fail_counts,
        'pss_ptm_mean': float(np.mean(pss_ptm_vals)) if pss_ptm_vals else None,
        'pss_pts_mean': float(np.mean(pss_pts_vals)) if pss_pts_vals else None,
        'rs_corr_mean': float(np.mean(rs_corr_vals)) if rs_corr_vals else None,
        'h_mag_mean': float(np.mean(h_mag_vals)) if h_mag_vals else None,
        'cfo_mean': float(np.mean(cfo_vals)) if cfo_vals else None,
        'cfo_std': float(np.std(cfo_vals)) if cfo_vals else None,
        'sigma2_mean': float(np.mean(sigma2_vals)) if sigma2_vals else None,
        # RS 失败帧诊断
        'rs_fail_corr_mean': float(np.mean(rs_fail_corr_vals)) if rs_fail_corr_vals else None,
        'rs_fail_cfo_mean': float(np.mean(rs_fail_cfo_vals)) if rs_fail_cfo_vals else None,
        # 全帧 oracle 信道参数 (用于诊断低 SNR 下物理信道)
        'oracle': {
            'noise_floor': noise_floor,
            'nf_db': nf_db,
            'snr_db': oracle_snr_db,
            'h_mag_mean': float(np.mean(oracle_h_mag)) if oracle_h_mag else None,
            'h_mag_std': float(np.std(oracle_h_mag)) if oracle_h_mag else None,
            'cfo_mean': float(np.mean(oracle_cfo)) if oracle_cfo else None,
            'cfo_std': float(np.std(oracle_cfo)) if oracle_cfo else None,
            'cfo_r2_mean': float(np.mean(oracle_r2)) if oracle_r2 else None,
            'phase_mean': float(np.mean(oracle_phase)) if oracle_phase else None,
            'phase_std': float(np.std(oracle_phase)) if oracle_phase else None,
            'n_frames': len(oracle_h_mag),
        },
    }

    return {
        'total_groups': total_groups,
        'groups_ok': groups_ok,
        'detection_rate': groups_ok / max(total_groups, 1),
        'mean_hits': float(np.mean(hits_per_group)) if hits_per_group else 0,
        'hits_distribution': {i: hits_per_group.count(i) for i in range(REPEAT + 1)},
        'diagnostics': diagnostics,
    }


def __reconstruct_first_frame(rng, stf_syms, pss_syms, rs_syms):
    """重建 frame_id=0 的 TX IQ (用于全帧互相关定位)."""
    raw = rng.randint(0, 2, PAYLOAD_LEN).astype(np.int64)
    frame_syms = build_custom_frame(raw, 0, stf_syms, pss_syms, rs_syms)
    tx_iq = rrc_filter(frame_syms, RRC, SPS)
    return tx_iq.astype(np.complex64), raw, frame_syms


# ═══════════════════════════════════════════════════════════════════════
# 前缀噪声底噪测量
# ═══════════════════════════════════════════════════════════════════════

def _prefix_noise_floor(iq, n_samples=50000):
    """从前缀 IQ 样本测量符号域噪声底噪。"""
    n = min(n_samples, len(iq))
    filt = np.convolve(iq[:n], RRC[::-1], mode='full')
    syms = filt[RRC_DELAY_SAMPLES::SPS]
    return float(np.var(syms))


def plot_iq_waveform(prefix, result, num_frames=40, save_path='', show=False,
                     n_plot_samples=5000):
    """画原始接收 IQ 的 I/Q 波形, 并标出 repeat frame 位置。"""
    diag = result.get('diagnostics', {})
    first_offset = diag.get('first_offset')
    frame_iq_len = diag.get('frame_iq_len')
    gap_repeat_iq = diag.get('gap_repeat_iq')
    gap_group_iq = diag.get('gap_group_iq')
    if first_offset is None or frame_iq_len is None:
        print("  绘图跳过: 缺少 first_offset/frame_iq_len")
        return

    iq_path = prefix + '_iq.npy'
    if not os.path.isfile(iq_path):
        print(f"  绘图跳过: 找不到 {iq_path}")
        return

    import matplotlib
    if save_path and not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    iq = np.load(iq_path)
    plot_start = max(0, int(first_offset))
    plot_end = min(len(iq), plot_start + int(n_plot_samples))
    if plot_end <= plot_start:
        print("  绘图跳过: 绘图窗口为空")
        return

    iq_win = iq[plot_start:plot_end]
    sample_idx = np.arange(plot_start, plot_end)
    t_ms = sample_idx.astype(np.float64) / SAMP_RATE * 1000.0

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t_ms, iq_win.real, lw=0.45, color='steelblue', label='I')
    ax.plot(t_ms, iq_win.imag, lw=0.45, color='darkorange', label='Q')

    repeat_stride = frame_iq_len + gap_repeat_iq
    group_stride = REPEAT * frame_iq_len + (REPEAT - 1) * gap_repeat_iq + gap_group_iq
    colors = ['tab:green', 'tab:orange', 'tab:purple', 'tab:red', 'tab:brown']

    for gi in range(num_frames):
        group_offset = first_offset + gi * group_stride
        if group_offset >= plot_end:
            break
        for ri in range(REPEAT):
            start = group_offset + ri * repeat_stride
            end = start + frame_iq_len
            if start >= plot_end:
                break
            if end < plot_start:
                continue
            c = colors[ri % len(colors)]
            if start >= plot_start:
                ax.axvline(start / SAMP_RATE * 1000.0, color=c, lw=0.6, alpha=0.7)
            if plot_start <= end < plot_end:
                ax.axvline(end / SAMP_RATE * 1000.0, color=c, lw=0.4, alpha=0.35, ls='--')

    tag = os.path.basename(prefix)
    ax.set_title(f'Raw received I/Q waveform with repeat frame markers: {tag}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_path:
        out = save_path
        _, ext = os.path.splitext(save_path)
        is_dir_target = (os.path.isdir(save_path)
                         or save_path.lower().endswith(('/', '\\'))
                         or ext == '')
        if is_dir_target:
            os.makedirs(save_path, exist_ok=True)
            out = os.path.join(save_path, f'{tag}_iq_waveform.png')
        else:
            parent = os.path.dirname(save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"  绘图 -> {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 全帧信道测量 (oracle-based, 独立于同步链)
# ═══════════════════════════════════════════════════════════════════════

TS_SYM = SPS / SAMP_RATE


def _oracle_frame_cfo(rx_syms, tx_syms_ref, ts_sym=TS_SYM):
    """全帧 496 符号线性相位拟合 -> CFO + 初始相位。"""
    mask = np.abs(tx_syms_ref) > 0.01
    if np.sum(mask) < 10:
        return 0.0, 0.0, 0.0
    idx_int = np.where(mask)[0]
    idx = idx_int.astype(np.float64)
    phase_diff = np.angle(rx_syms[idx_int] * np.conj(tx_syms_ref[idx_int]))
    phase_unwrapped = np.unwrap(phase_diff)
    n_mean = np.mean(idx)
    p_mean = np.mean(phase_unwrapped)
    num = np.sum((idx - n_mean) * (phase_unwrapped - p_mean))
    den = np.sum((idx - n_mean) ** 2)
    slope = num / (den + 1e-30)
    cfo_hz = float(slope / (2 * np.pi * ts_sym))
    phase_0 = float(p_mean - slope * n_mean)
    phase_pred = slope * idx + phase_0
    ss_res = np.sum((phase_unwrapped - phase_pred) ** 2)
    ss_tot = np.sum((phase_unwrapped - p_mean) ** 2)
    r_sq = float(1 - ss_res / (ss_tot + 1e-30))
    return cfo_hz, phase_0, r_sq


def _oracle_frame_channel(rx_syms, tx_syms_ref, cfo_hz, phase_0, ts_sym=TS_SYM):
    """全帧 LS 信道估计 -> |h|, phase, sigma2。"""
    n_sym = np.arange(len(rx_syms))
    total_phase = 2 * np.pi * cfo_hz * n_sym * ts_sym + phase_0
    rx_corrected = rx_syms * np.exp(-1j * total_phase)
    mask = np.abs(tx_syms_ref) > 0.01
    num = np.sum(rx_corrected[mask] * np.conj(tx_syms_ref[mask]))
    den = np.sum(np.abs(tx_syms_ref[mask]) ** 2)
    h = num / (den + 1e-30)
    return h, float(abs(h)), float(np.angle(h))


def _oracle_measure_one_frame(iq, expected_offset, frame_iq_len, rng, frame_id,
                               stf_syms, pss_syms, rs_syms, use_polar=False):
    """对一帧做全帧信道测量 (已知帧 + 时域相关搜索).

    独立于 PSS+RS 同步链。
    use_polar=True: 模拟 Polar 编码负载, RNG 调用匹配 loopback_capture.
    Returns dict 或 None (如果找不到帧或信号过弱).
    """
    if use_polar:
        # 匹配 loopback_capture.py 的 RNG 调用序列:
        # rng.rand(K_POLAR) -> polar_encode -> build_frame
        N_POLAR = 256; K_POLAR = 128
        import math
        def _polar_encode(u):
            cw = u.copy().ravel()
            for stage in range(1, int(math.log2(N_POLAR)) + 1):
                sep = N_POLAR // (1 << stage)
                for j in range(N_POLAR):
                    if (j // sep) % 2 == 0:
                        cw[j] = (cw[j] + cw[j + sep]) % 2
            return cw
        info = (rng.rand(K_POLAR) < 0.5).astype(np.int64)
        u = np.zeros(N_POLAR, dtype=np.int64)
        u[FROZEN_MASK.astype(bool)] = info
        coded = _polar_encode(u)
        from tools.loopback_capture import build_frame, rrc_pulse
        frame_syms = build_frame(coded, frame_id, stf_syms, pss_syms, rs_syms)
        tx_iq = rrc_pulse(frame_syms, RRC, SPS).astype(np.complex64)
    else:
        raw = rng.randint(0, 2, PAYLOAD_LEN).astype(np.int64)
        frame_syms = build_custom_frame(raw, frame_id, stf_syms, pss_syms, rs_syms)
        tx_iq = rrc_filter(frame_syms, RRC, SPS).astype(np.complex64)

    # 搜索窗口: expected_offset 前后各 margin 样本
    margin = 400
    search_lo = max(0, expected_offset - margin)
    search_hi = min(len(iq), expected_offset + margin + len(tx_iq))
    if search_hi - search_lo < len(tx_iq):
        return None

    seg = iq[search_lo:search_hi]
    tx_rev = np.conj(tx_iq[::-1])
    corr = np.abs(np.convolve(seg, tx_rev, mode='valid'))
    pk = int(np.argmax(corr))
    corr_val = float(corr[pk])
    if corr_val < 0.1:
        return None

    best_pos = search_lo + pk
    if best_pos + len(tx_iq) > len(iq):
        return None

    frame_iq = iq[best_pos:best_pos + len(tx_iq)]
    filt = np.convolve(frame_iq, RRC[::-1], mode='full')
    rx_syms = filt[RRC_DELAY_SAMPLES::SPS].astype(np.complex64)
    n_sym = min(len(rx_syms), len(frame_syms))
    if n_sym < 10:
        return None
    rx_syms = rx_syms[:n_sym]

    cfo_hz, phase_0, r_sq = _oracle_frame_cfo(rx_syms, frame_syms)
    h, h_mag, h_phase = _oracle_frame_channel(rx_syms, frame_syms, cfo_hz, phase_0)

    return {
        'frame_id': frame_id,
        'pos': best_pos,
        'corr_peak': corr_val,
        'cfo_hz': cfo_hz,
        'cfo_r_squared': r_sq,
        'phase_0_sym': phase_0,    # 帧起始前的公共相位 (相对于符号 0)
        'h_mag': h_mag,
        'h_phase': h_phase,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='5x 重复帧 capture 分析 (any-of-N 检出)')
    p.add_argument('input_dir', help='capture 目录')
    p.add_argument('--gain', type=int, default=0,
                   help='只分析指定 gain (0=全部)')
    p.add_argument('--num-frames', type=int, default=40,
                   help='每组唯一帧数 (默认 40)')
    p.add_argument('--pss-ptm', type=float, default=2.5)
    p.add_argument('--pss-pts', type=float, default=1.0)
    p.add_argument('--fixed-nf', type=float, default=None,
                   help='固定底噪 dB (如 -70), 默认 None=实测')
    p.add_argument('--plot', action='store_true',
                   help='绘制原始接收 I/Q 波形并标出 repeat frame 位置')
    p.add_argument('--save-plot', default='',
                   help='保存 I/Q 波形图; 多 capture 时建议传目录')
    p.add_argument('--plot-samples', type=int, default=5000,
                   help='绘图采样点数, 默认 5000')
    p.add_argument('-o', '--output', default='',
                   help='输出 JSON')
    args = p.parse_args()

    # 找所有 capture
    prefixes = []
    for f in sorted(glob.glob(os.path.join(args.input_dir, '*_iq.npy'))):
        pfx = f.replace('_iq.npy', '')
        if args.gain > 0 and f'gain{args.gain:03d}' not in pfx:
            continue
        prefixes.append(pfx)

    if not prefixes:
        print(f"未找到 capture: {args.input_dir}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"5x 重复帧 any-of-N 分析: {len(prefixes)} captures")
    print(f"{'='*60}")

    all_results = {}
    for pfx in prefixes:
        tag = os.path.basename(pfx)
        gain_str = str(int(tag.split('_')[1].replace('gain', '')))
        print(f"\n-- {tag} (gain={int(gain_str)} dB) --")
        r = analyze_capture(pfx, num_frames=args.num_frames,
                            pss_ptm=args.pss_ptm, pss_pts=args.pss_pts,
                            fixed_nf_db=args.fixed_nf)
        if 'error' in r:
            print(f"  错误: {r['error']}")
            continue

        print(f"  组检出: {r['groups_ok']}/{r['total_groups']} "
              f"({r['detection_rate']*100:.1f}%)")
        print(f"  平均命中: {r['mean_hits']:.1f}/{REPEAT} 帧/组")
        hits_str = ' '.join(f'{k}:{v}' for k, v in sorted(r['hits_distribution'].items()))
        print(f"  命中分布: {hits_str}")
        diag = r.get('diagnostics', {})
        if diag:
            fails = diag.get('fail_counts', {})
            fail_str = ' '.join(f'{k}:{v}' for k, v in sorted(fails.items()))
            print(f"  阶段统计: {fail_str}")
            if diag.get('pss_ptm_mean') is not None:
                print(f"  PSS均值: ptm={diag['pss_ptm_mean']:.2f} "
                      f"pts={diag['pss_pts_mean']:.2f}")
            if diag.get('rs_corr_mean') is not None:
                print(f"  RS均值: corr={diag['rs_corr_mean']:.1f} "
                      f"|h|={diag['h_mag_mean']:.4f} "
                      f"CFO={diag['cfo_mean']:+.1f}±{diag['cfo_std']:.1f}Hz")
            if diag.get('rs_fail_corr_mean') is not None:
                cfo_str = ""
                if diag.get('rs_fail_cfo_mean') is not None:
                    cfo_str = f" cfo_raw={diag['rs_fail_cfo_mean']:+.0f}Hz"
                print(f"  RS失败帧: corr_raw={diag['rs_fail_corr_mean']:.2f}{cfo_str}"
                      f"  (门限=3.2)")
        oracle = diag.get('oracle', {})
        if oracle.get('snr_db') is not None:
            eb_n0_db = oracle['snr_db'] + 3.0  # R=128/256=0.5, Eb/N0 = Es/N0 + 3dB
            pstr = f"SNR={oracle['snr_db']:.1f}dB"
            if oracle['snr_db'] is not None:
                pstr += f"  Eb/N0={eb_n0_db:.1f}dB"
            pstr += f"  CFO={oracle['cfo_mean']:+.1f}±{oracle['cfo_std']:.1f}Hz  "
            pstr += f"|h|={oracle['h_mag_mean']:.4f}  "
            pstr += f"φ={oracle['phase_mean']:+.3f}±{oracle['phase_std']:.3f}rad  "
            pstr += f"底噪={oracle['nf_db']:.1f}dB"
            print(f"  信道: {pstr}")
        if args.plot or args.save_plot:
            save_path = args.save_plot
            plot_iq_waveform(pfx, r, num_frames=args.num_frames,
                             save_path=save_path, show=args.plot,
                             n_plot_samples=args.plot_samples)
        all_results[gain_str] = r

    # 汇总
    if all_results:
        print(f"\n{'='*60}")
        print(f"  汇总")
        print(f"  {'gain':>6s}  {'groups':>8s}  {'rate':>7s}  {'hits/grp':>9s}")
        print(f"  {'-'*40}")
        for gain_str in sorted(all_results.keys(), key=int):
            r = all_results[gain_str]
            print(f"  {int(gain_str):5d}  {r['groups_ok']:4d}/{r['total_groups']:<4d}  "
                  f"{r['detection_rate']*100:5.1f}%  {r['mean_hits']:7.1f}/{REPEAT}")
        print(f"{'='*60}")

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n报告 -> {args.output}")


if __name__ == '__main__':
    main()
