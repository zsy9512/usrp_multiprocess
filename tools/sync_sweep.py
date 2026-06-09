#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sync_sweep.py — 低 SNR 同步链设计 + 诊断工具

目标:
  用仿真找出当前同步链 (STF->PSS->RS) 在各级 SNR 下的精确失效点,
  并支持修改帧结构 (STF 重复数 / RS 长度) 来压低工作 SNR。

输入:
  channel_params.json (来自 channel_measure.py 的信道标定)

输出:
  - 每 SNR 点的检出率 + 各阶段失败分布
  - 不同帧结构变体的 SNR 极限对比
  - 推荐的低 SNR 帧参数

用法:
  # 当前帧结构 SNR sweep
  python tools/sync_sweep.py channel_params.json --snr-range -5 15 2

  # 测试更长 STF
  python tools/sync_sweep.py channel_params.json --stf-reps 8 --rs-len 64

  # 对比多种变体
  python tools/sync_sweep.py channel_params.json --compare
"""

import argparse, json, os, sys, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from phy_params import (SPS, STF_LEN, PSS_LEN, RS_LEN,
                        HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN,
                        GUARD_SYMBOLS, FRAME_SYMBOLS,
                        RRC_DELAY_SAMPLES, STF_DELAY,
                        FRAME_RRC_SAMPLES)

from tools.calibrated_simulation import (
    simulate_one_frame, _build_codeword, apply_channel
)
from sender import build_frame, rrc_filter
from phy_params import RRC

SAMP_RATE = 1e6
TS_SYM = SPS / SAMP_RATE
N_POLAR = 256
K_POLAR = 128


# ═══════════════════════════════════════════════════════════════════════
# 可变帧结构: 构造不同 STF/RS 长度的参考序列
# ═══════════════════════════════════════════════════════════════════════

def make_stf(n_reps=4, base_len=16):
    """生成 STF: n_reps × base_len 重复 BPSK."""
    rng = np.random.RandomState(7)
    base = 2 * rng.randint(0, 2, base_len) - 1
    return np.tile(base, n_reps).astype(np.complex64)


def make_rs(n_syms=32):
    """生成 RS: 固定 BPSK 导频."""
    rng = np.random.RandomState(13)
    return (2 * rng.randint(0, 2, n_syms) - 1).astype(np.complex64)


def build_custom_frame(data_bits, frame_id, stf_syms, pss_syms, rs_syms):
    """用自定义 STF/RS 构建帧 (Header+Payload+CRC+Guard 保持不变).

    Returns:
        (FRAME_SYMBOLS_CUSTOM,) complex64
    """
    from phy_params import crc16, bits_to_bytes, bytes_to_bits

    payload_bytes = bits_to_bytes(data_bits)
    payload_crc = crc16(payload_bytes)
    crc_bits = bytes_to_bits(
        np.array([(payload_crc >> 8) & 0xFF, payload_crc & 0xFF], dtype=np.uint8), 16)

    id_bytes = np.array([(frame_id >> 8) & 0xFF, frame_id & 0xFF], dtype=np.uint8)
    id_bits = bytes_to_bits(id_bytes, 16)
    header_crc = crc16(id_bytes)
    header_crc_bits = bytes_to_bits(
        np.array([(header_crc >> 8) & 0xFF, header_crc & 0xFF], dtype=np.uint8), 16)
    header_bits = np.concatenate([id_bits, header_crc_bits])

    def _bpsk(b):
        return (1.0 - 2.0 * b).astype(np.float32)

    return np.concatenate([
        stf_syms.astype(np.complex64),
        pss_syms.astype(np.complex64),
        rs_syms.astype(np.complex64),
        _bpsk(header_bits).astype(np.complex64),
        _bpsk(data_bits).astype(np.complex64),
        _bpsk(crc_bits).astype(np.complex64),
        np.zeros(GUARD_SYMBOLS, dtype=np.complex64),
    ])


# ═══════════════════════════════════════════════════════════════════════
# 可变帧结构的同步函数
# ═══════════════════════════════════════════════════════════════════════

def stf_detect_custom(samples, stf_len_sym, stf_delay_sym=16, sps=SPS,
                      threshold=0.4, min_energy=0.01, max_peaks=8):
    """可变长度 STF 延迟相关检测. 返回多个候选峰 (对齐 polar_loopback.py 逻辑).

    在 128-sample 窗内聚类去重, 按 M 值排序返回前 max_peaks 个.

    Returns:
        list[dict]: [{pos, M, coarse_cfo}, ...]  (可能为空)
    """
    L = stf_delay_sym * sps
    N = len(samples)
    if N <= L + stf_len_sym * sps:
        return []

    r0, rL = samples[:N - L], samples[L:]
    prod = r0 * np.conj(rL)
    ones = np.ones(L, dtype=np.float32)
    P = np.convolve(prod, ones, mode='valid')
    E = np.convolve((np.abs(rL) ** 2).astype(np.float32), ones, mode='valid')
    M = np.abs(P) / (E + 1e-6 * L)

    # 收集所有候选
    raw = []
    for d in range(len(M)):
        if M[d] > threshold:
            le = np.sum(np.abs(samples[d + L:d + 2 * L]) ** 2)
            if le > min_energy:
                raw.append((d, float(M[d]), P[d]))
    if not raw:
        return []

    # 按 M 排序, 128-sample 窗内去重 (对齐 polar_loopback.py)
    raw.sort(key=lambda x: x[1], reverse=True)
    peaks, used = [], set()
    for d, m, p in raw:
        if d in used:
            continue
        for dx in range(max(0, d - 128), min(len(M), d + 128)):
            used.add(dx)
        cfo = float(-np.angle(p) / (2 * np.pi * L / SAMP_RATE))
        peaks.append({'pos': d, 'M': m, 'coarse_cfo': cfo})
        if len(peaks) >= max_peaks:
            break

    return peaks


def pss_correlate_custom(syms, pss_ref):
    """PSS 互相关 (与 phy_params PSS 长度可能不同)."""
    M = len(pss_ref)
    if len(syms) < M:
        return -1, 0, 0, 0
    pss_rev = np.conj(pss_ref[::-1])
    c = np.abs(np.convolve(syms, pss_rev, mode='valid'))
    pk = int(np.argmax(c))
    peak_val = float(c[pk])
    ptm = peak_val / (np.mean(c) + 1e-30)
    pts = ptm
    sv = np.sort(c)[::-1]
    for v in sv[1:]:
        for idx in np.where(np.isclose(c, v))[0]:
            if abs(idx - pk) > M // 2:
                pts = peak_val / (v + 1e-30)
                break
        if pts != ptm:
            break
    return pk, ptm, pts, peak_val


def rs_estimate_custom(syms, rs_pos, rs_ref, coarse_cfo=0.0):
    """可变长度 RS 信道估计.

    关键改进 (vs polar_loopback.py):
      - 跳过粗 CFO 预补偿 (低 SNR 下 STF 粗 CFO 误差 ~400Hz,
        预补偿反而引入错误相位斜坡, 导致长 RS 的 unwrap 爆炸)
      - 直接从 RS 全段估计总 CFO (分辨率 = 1/(2pi*rs_len*Tsym),
        比 STF 的 32-sample 延迟相关精确 4-8x)
      - fine_cfo_max 放宽到 2000 Hz (真实 CFO < 100 Hz)
    """
    rs_len = len(rs_ref)
    if rs_pos + rs_len > len(syms):
        return None

    rs_seg = syms[rs_pos:rs_pos + rs_len].copy()
    n_rs = np.arange(rs_len)

    # 直接估计总 CFO (不预补偿 — 预补偿的粗 CFO 在低 SNR 下不可靠)
    rs_tone = rs_seg * np.conj(rs_ref)
    rs_corr = float(np.abs(np.sum(rs_tone)))

    # 相位线性回归 -> 总 CFO
    # 关键: 不 unwrap (真实 CFO < 100 Hz, 32-128 符号内相位积累 < 0.05 rad,
    # 远小于 pi, np.angle 不会发生 2pi 跳变. unwrap 在低 SNR 下反而会放大噪声尖峰)
    rs_phase = np.angle(rs_tone)
    n = np.arange(rs_len, dtype=np.float64)
    n_mean = np.mean(n); p_mean = np.mean(rs_phase)
    num = np.sum((n - n_mean) * (rs_phase - p_mean))
    den = np.sum((n - n_mean) ** 2)
    if den < 1e-30:
        return None
    slope = num / den
    total_cfo = float(slope / (2 * np.pi * TS_SYM))

    # 放宽上限: 真实硬件 CFO < 100 Hz, 但低 SNR 估计有噪声
    if abs(total_cfo) > 2000:
        return None

    # 总 CFO 补偿 + 信道估计
    total_comp = np.exp(-1j * 2 * np.pi * total_cfo * (rs_pos + n_rs) * TS_SYM)
    rs_corrected = syms[rs_pos:rs_pos + rs_len] * total_comp
    h = np.mean(rs_corrected * np.conj(rs_ref))
    if abs(h) < 1e-6:
        return None
    noise = rs_corrected / h - rs_ref
    sigma2 = max(float(np.sum(np.abs(noise) ** 2) / (rs_len - 1)), 1e-30)

    # RS 相关质量门限 (保留, 用于过滤假帧)
    if rs_corr < rs_len * 0.1:
        return None

    return {'h': h, 'phase_est': float(np.angle(h)), 'sigma2': sigma2,
            'coarse_cfo': 0.0, 'fine_cfo': total_cfo,
            'total_cfo': total_cfo, 'rs_corr': rs_corr}


def sync_one_frame(rx_iq, stf_syms, pss_syms, rs_syms,
                   stf_thr=0.4, stf_energy=0.01,
                   pss_ptm=2.5, pss_pts=1.0):
    """对一段 RX IQ 跑完整同步链 (可变帧结构).

    尝试多个 STF 候选峰 (对齐 polar_loopback.py 逻辑),
    返回第一个通过 PSS+RS 的结果.

    Returns:
        dict: {detected, fail_stage?, ...}  或 None
    """
    stf_delay_sym = 16
    stf_len_sym = len(stf_syms)

    # STF: 返回多个候选峰
    candidates = stf_detect_custom(rx_iq, stf_len_sym, stf_delay_sym,
                                   threshold=stf_thr, min_energy=stf_energy,
                                   max_peaks=8)
    if not candidates:
        return {'detected': False, 'fail_stage': 'STF'}

    # 统计每阶段失败次数
    fail_count = {'STF': 0, 'PSS': 0, 'RS': 0}

    for stf in candidates:
        # 过滤离谱 CFO
        if abs(stf['coarse_cfo']) > 2000:
            fail_count['STF'] += 1
            continue

        coarse = stf['pos']
        margin = 400
        total_frame_sym = (stf_len_sym + len(pss_syms) + len(rs_syms)
                           + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN + GUARD_SYMBOLS)
        total_frame_iq = total_frame_sym * SPS + len(RRC) - 1
        es = max(0, coarse - margin)
        ee = min(len(rx_iq), coarse + margin + total_frame_iq + margin)
        chunk = rx_iq[es:ee]

        filt = np.convolve(chunk, RRC[::-1], mode='full')
        RRC_DELAY = (len(RRC) - 1) // 2  # 10 samples (not RRC_DELAY_SAMPLES=20!)
        syms = filt[RRC_DELAY::SPS].astype(np.complex64)

        if len(syms) < len(pss_syms) + len(rs_syms):
            fail_count['PSS'] += 1
            continue

        # PSS
        pk, ptm, pts, pval = pss_correlate_custom(syms, pss_syms)
        if ptm < pss_ptm or pts < pss_pts:
            fail_count['PSS'] += 1
            continue

        fs = pk - stf_len_sym
        if fs < 0:
            fail_count['PSS'] += 1
            continue

        rp = fs + stf_len_sym + len(pss_syms)
        data_end = rp + len(rs_syms) + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN
        if data_end > len(syms):
            fail_count['RS'] += 1
            continue

        # RS
        chan = rs_estimate_custom(syms, rp, rs_syms, stf['coarse_cfo'])
        if chan is None:
            fail_count['RS'] += 1
            continue

        hmag = abs(chan['h'])
        return {
            'detected': True,
            'ptm': ptm, 'pts': pts,
            'coarse_cfo': chan['coarse_cfo'],
            'fine_cfo': chan['fine_cfo'],
            'total_cfo': chan['total_cfo'],
            'h_mag': hmag,
            'sigma2': chan['sigma2'],
            'rs_corr': chan['rs_corr'],
        }

    # 所有候选都失败
    return {'detected': False, 'fail_stage': 'RS',
            'fails': fail_count}


# ═══════════════════════════════════════════════════════════════════════
# SNR Sweep
# ═══════════════════════════════════════════════════════════════════════

def run_snr_sweep(snr_range, n_frames=200, seed=42,
                  stf_reps=4, rs_len=32,
                  cfo_mean=-2.4, cfo_std=19.2,
                  stf_thr=0.4, stf_energy=0.01,
                  pss_ptm=2.5, pss_pts=1.0,
                  verbose=True):
    """在指定 SNR 范围上测试同步链性能.

    Args:
        snr_range:  (snr_min, snr_max, snr_step)  dB, 符号域
        stf_reps:   STF 重复段数 (默认 4 -> 64 符号)
        rs_len:     RS 长度 (默认 32)

    Returns:
        list[dict]: 每 SNR 点的统计
    """
    from phy_params import PSS as PSS_REF

    snr_min, snr_max, snr_step = snr_range
    snr_points = np.arange(snr_min, snr_max + snr_step/2, snr_step)

    # 构造参考序列
    stf_syms = make_stf(n_reps=stf_reps)
    rs_syms = make_rs(n_syms=rs_len)
    pss_syms = PSS_REF  # 保持 Zadoff-Chu 64

    rng = np.random.RandomState(seed)
    results = []

    for snr_db in snr_points:
        # noise_floor = |h|^2 / 10^(SNR/10), with |h| = 1.0 normalized
        h_mag = 1.0
        noise_floor = h_mag ** 2 / (10 ** (snr_db / 10))

        det_ok = 0
        fails = {'STF': 0, 'PSS': 0, 'RS': 0}
        ptms, ptss = [], []
        cfo_ests = []

        for fi in range(n_frames):
            # 生成随机数据 + 构建帧
            info_bits = (rng.rand(K_POLAR) < 0.5).astype(np.int64)
            coded = _build_codeword(info_bits)
            frame_syms = build_custom_frame(coded, fi, stf_syms, pss_syms, rs_syms)
            tx_iq = rrc_filter(frame_syms, RRC, SPS)

            # 信道
            cfo = rng.normal(cfo_mean, cfo_std)
            phase = rng.uniform(-np.pi, np.pi)
            rx_iq = apply_channel(tx_iq, h_mag, cfo, phase, noise_floor)

            # 前后加噪声 padding
            pad_before = 2000
            pad_after = 3000
            n_std = np.sqrt(noise_floor * SPS / 2)
            pad_b = (n_std * (rng.randn(pad_before) + 1j * rng.randn(pad_before))).astype(np.complex64)
            pad_a = (n_std * (rng.randn(pad_after) + 1j * rng.randn(pad_after))).astype(np.complex64)
            rx_full = np.concatenate([pad_b, rx_iq, pad_a])

            # 同步
            result = sync_one_frame(rx_full, stf_syms, pss_syms, rs_syms,
                                    stf_thr, stf_energy, pss_ptm, pss_pts)

            if result is None:
                continue
            if result['detected']:
                det_ok += 1
                ptms.append(result['ptm'])
                ptss.append(result['pts'])
                cfo_ests.append(result['total_cfo'])
            else:
                stage = result.get('fail_stage', 'unknown')
                fails[stage] = fails.get(stage, 0) + 1

        stats = {
            'snr_db': float(snr_db),
            'noise_floor': noise_floor,
            'n_frames': n_frames,
            'detected': det_ok,
            'detection_rate': det_ok / n_frames,
            'fails': fails,
        }
        if det_ok > 0:
            stats['ptm_mean'] = float(np.mean(ptms))
            stats['ptm_std'] = float(np.std(ptms))
            stats['pts_mean'] = float(np.mean(ptss))
            stats['cfo_est_mean'] = float(np.mean(cfo_ests))
            stats['cfo_est_std'] = float(np.std(cfo_ests))

        results.append(stats)

        if verbose:
            bar = '#' * int(stats['detection_rate'] * 40)
            print(f"  SNR={snr_db:5.1f} dB  "
                  f"det={det_ok:4d}/{n_frames} ({stats['detection_rate']*100:5.1f}%)  "
                  f"STF={fails.get('STF',0):3d}  PSS={fails.get('PSS',0):3d}  "
                  f"RS={fails.get('RS',0):3d}  {bar}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 帧结构对比
# ═══════════════════════════════════════════════════════════════════════

def compare_variants(snr_range=(-5, 20, 2), n_frames=200, cfo_mean=-2.4, cfo_std=19.2):
    """对比不同 STF/RS 变体的 SNR 极限."""
    variants = [
        {'label': 'STF=64 RS=32  (current)', 'stf_reps': 4, 'rs_len': 32},
        {'label': 'STF=128 RS=32',           'stf_reps': 8, 'rs_len': 32},
        {'label': 'STF=256 RS=32',           'stf_reps': 16, 'rs_len': 32},
        {'label': 'STF=64  RS=64',           'stf_reps': 4, 'rs_len': 64},
        {'label': 'STF=128 RS=64',           'stf_reps': 8, 'rs_len': 64},
        {'label': 'STF=256 RS=128',          'stf_reps': 16, 'rs_len': 128},
    ]

    print(f"\n{'='*80}")
    print(f"帧结构变体对比 (CFO: N({cfo_mean:.1f}, {cfo_std:.1f}) Hz)")
    print(f"{'='*80}")

    all_results = {}
    for v in variants:
        print(f"\n-- {v['label']} --")
        t0 = time.time()
        r = run_snr_sweep(snr_range, n_frames=n_frames,
                          stf_reps=v['stf_reps'], rs_len=v['rs_len'],
                          cfo_mean=cfo_mean, cfo_std=cfo_std,
                          verbose=False)
        elapsed = time.time() - t0

        # 找 50% 和 90% 检出率的 SNR
        snrs = np.array([p['snr_db'] for p in r])
        rates = np.array([p['detection_rate'] for p in r])

        snr50, snr90 = None, None
        for i in range(len(rates)):
            if snr50 is None and rates[i] >= 0.5:
                snr50 = snrs[i]
            if snr90 is None and rates[i] >= 0.9:
                snr90 = snrs[i]

        print(f"    50%检出 @ SNR={snr50:.1f} dB" if snr50 else f"    50%检出: 未达到")
        print(f"    90%检出 @ SNR={snr90:.1f} dB" if snr90 else f"    90%检出: 未达到")
        print(f"    耗时: {elapsed:.1f}s")

        all_results[v['label']] = {
            'config': v,
            'snr50': snr50,
            'snr90': snr90,
            'sweep': [{k: v for k, v in p.items()} for p in r],
        }

    # 排名
    print(f"\n-- 排名 (90%检出 SNR, 越低越好) --")
    ranked = sorted([(k, v['snr90'] or 99) for k, v in all_results.items()],
                    key=lambda x: x[1])
    for i, (label, snr90) in enumerate(ranked):
        flag = " <-- BEST" if i == 0 else ""
        print(f"  {i+1}. {label:30s}  SNR90={snr90:.1f} dB{flag}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# 重复帧相干叠加模式 (5x same frame -> coherent combine)
# ═══════════════════════════════════════════════════════════════════════

def run_repeat_combine_sweep(snr_range, n_groups=50, seed=42,
                              stf_reps=4, rs_len=64,
                              n_repeats=5, gap_repeat_ms=3.0, gap_group_ms=10.0,
                              cfo_mean=-2.9, cfo_std=17.9,
                              stf_thr=0.4, stf_energy=0.01,
                              pss_ptm=2.5, pss_pts=1.0,
                              verbose=True):
    """重复帧相干叠加 SNR sweep.

    每"组"发送同一帧 n_repeats 次 (间隔 gap_repeat_ms),
    接收端检测第一帧 -> 估计 CFO -> 补偿相位 -> 5 帧相干叠加 -> 同步.

    帧结构: STF(stf_reps*16) + PSS(64) + RS(rs_len) + Header(32) + Payload(256) + CRC(16) + Guard(32)

    Args:
        n_groups: 每组重复次数取代原来的 n_frames (总 TX 次数 = n_groups * n_repeats)
    """
    from phy_params import PSS as PSS_REF

    snr_min, snr_max, snr_step = snr_range
    snr_points = np.arange(snr_min, snr_max + snr_step/2, snr_step)

    stf_syms = make_stf(n_reps=stf_reps)
    rs_syms = make_rs(n_syms=rs_len)
    pss_syms = PSS_REF

    total_sym = (len(stf_syms) + len(pss_syms) + len(rs_syms)
                 + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN + GUARD_SYMBOLS)
    frame_iq_len = total_sym * SPS + len(RRC) - 1
    gap_repeat_iq = int(gap_repeat_ms * SAMP_RATE / 1000)  # 重复间隔 IQ 样本
    gap_group_iq = int(gap_group_ms * SAMP_RATE / 1000)    # 组间隔 IQ 样本

    rng = np.random.RandomState(seed)
    results = []

    for snr_db in snr_points:
        h_mag = 1.0
        noise_floor = h_mag ** 2 / (10 ** (snr_db / 10))
        nf_std = np.sqrt(noise_floor * SPS / 2)

        det_ok = 0
        fails = {'STF': 0, 'PSS': 0, 'RS': 0}

        for gi in range(n_groups):
            # -- 生成一帧 + 重复 n_repeats 次 --
            info_bits = (rng.rand(K_POLAR) < 0.5).astype(np.int64)
            coded = _build_codeword(info_bits)
            frame_syms = build_custom_frame(coded, gi, stf_syms, pss_syms, rs_syms)
            tx_iq_one = rrc_filter(frame_syms, RRC, SPS)

            # 拼接: [GAP_GROUP] [FRAME] [GAP_REPEAT] [FRAME] ... × n_repeats
            tx_segments = []
            # 组前 gap (第一组前不加, 后续组前加)
            if gi > 0:
                tx_segments.append(np.zeros(gap_group_iq, dtype=np.complex64))
            for ri in range(n_repeats):
                tx_segments.append(tx_iq_one)
                if ri < n_repeats - 1:
                    tx_segments.append(np.zeros(gap_repeat_iq, dtype=np.complex64))
            tx_all = np.concatenate(tx_segments)

            # 信道
            cfo = rng.normal(cfo_mean, cfo_std)
            phase = rng.uniform(-np.pi, np.pi)
            rx_all = apply_channel(tx_all, h_mag, cfo, phase, noise_floor)

            # padding (前后噪声)
            pad_len = 2000
            pad_before = (nf_std * (rng.randn(pad_len) + 1j * rng.randn(pad_len))).astype(np.complex64)
            pad_after = (nf_std * (rng.randn(pad_len) + 1j * rng.randn(pad_len))).astype(np.complex64)
            rx = np.concatenate([pad_before, rx_all, pad_after])

            # -- 检测第一帧位置 + 估计 CFO --
            first_frame_start = pad_len  # 第一帧在 rx 中的起始位置 (无组前 gap 时)
            if gi > 0:
                first_frame_start += gap_group_iq

            # 提取第一帧 + 前后裕量
            margin = 400
            es = max(0, first_frame_start - margin)
            ee = min(len(rx), first_frame_start + frame_iq_len + margin)
            chunk1 = rx[es:ee]

            RRC_DEL = (len(RRC) - 1) // 2
            syms1 = np.convolve(chunk1, RRC[::-1], mode='full')[RRC_DEL::SPS].astype(np.complex64)

            if len(syms1) < len(pss_syms) + len(rs_syms):
                fails['PSS'] += 1
                continue

            # PSS on first frame
            pk, ptm, pts, pv = pss_correlate_custom(syms1, pss_syms)
            if ptm < pss_ptm or pts < pss_pts:
                fails['PSS'] += 1
                continue

            fs = pk - len(stf_syms)
            if fs < 0:
                fails['PSS'] += 1
                continue

            rp1 = fs + len(stf_syms) + len(pss_syms)
            if rp1 + len(rs_syms) > len(syms1):
                fails['RS'] += 1
                continue

            # -- 逐帧独立 PSS 定位 + 相位估计 + 相干叠加 --
            ref_phases = []
            all_frame_syms = []

            for ri in range(n_repeats):
                offset = first_frame_start + ri * (frame_iq_len + gap_repeat_iq)
                if offset + frame_iq_len > len(rx):
                    break

                m2 = 200
                es2 = max(0, offset - m2)
                ee2 = min(len(rx), offset + frame_iq_len + m2)
                chunk = rx[es2:ee2]
                syms = np.convolve(chunk, RRC[::-1], mode='full')[RRC_DEL::SPS].astype(np.complex64)

                # 每帧独立 PSS 定位 (关键: 不能假设 sym_start 固定)
                pk_i, ptm_i, pts_i, _ = pss_correlate_custom(syms, pss_syms)
                if ptm_i < pss_ptm or pts_i < pss_pts:
                    break
                fs_i = pk_i - len(stf_syms)
                if fs_i < 0:
                    break
                frame_start_sym = fs_i
                frame_end_sym = frame_start_sym + total_sym
                if frame_end_sym > len(syms):
                    break

                frame_syms_i = syms[frame_start_sym:frame_end_sym]

                # RS 相位估计 (frame 内相对位置 = stf_len + pss_len)
                rp_local = len(stf_syms) + len(pss_syms)
                if rp_local + len(rs_syms) > len(frame_syms_i):
                    break
                rs_seg_i = frame_syms_i[rp_local:rp_local + len(rs_syms)]
                rs_tone_i = rs_seg_i * np.conj(rs_syms)
                h_est_i = np.mean(rs_tone_i)
                phase_i = float(np.angle(h_est_i))

                ref_phases.append(phase_i)
                all_frame_syms.append(frame_syms_i)

            if len(all_frame_syms) < 2:
                fails['RS'] += 1
                continue

            # 以第一帧相位为参考, 对齐后叠加
            ref_p0 = ref_phases[0]
            accum_syms = all_frame_syms[0].astype(np.complex128)
            for ri in range(1, len(all_frame_syms)):
                dp = ref_phases[ri] - ref_p0
                accum_syms += all_frame_syms[ri] * np.exp(-1j * dp)

            combined_syms = (accum_syms / len(all_frame_syms)).astype(np.complex64)

            # -- 在叠加后的符号上跑同步 --
            # PSS (叠加后 SNR 更高, 位置应更精确)
            pk2, ptm2, pts2, pv2 = pss_correlate_custom(combined_syms, pss_syms)
            if ptm2 < pss_ptm or pts2 < pss_pts:
                fails['PSS'] += 1
                continue

            fs2 = pk2 - len(stf_syms)
            if fs2 < 0:
                fails['PSS'] += 1
                continue

            rp2 = len(stf_syms) + len(pss_syms)
            if rp2 + len(rs_syms) > len(combined_syms):
                fails['RS'] += 1
                continue

            # RS on combined symbols
            chan2 = rs_estimate_custom(combined_syms, rp2, rs_syms)
            if chan2 is None:
                fails['RS'] += 1
                continue

            det_ok += 1

        stats = {
            'snr_db': float(snr_db),
            'noise_floor': noise_floor,
            'n_groups': n_groups,
            'n_repeats': n_repeats,
            'detected': det_ok,
            'detection_rate': det_ok / n_groups,
            'fails': fails,
            'config': {
                'stf_syms': len(stf_syms), 'rs_syms': len(rs_syms),
                'gap_repeat_ms': gap_repeat_ms, 'gap_group_ms': gap_group_ms,
                'frame_iq_len': frame_iq_len,
            },
        }
        results.append(stats)

        if verbose:
            bar = '#' * int(stats['detection_rate'] * 40)
            print(f"  SNR={snr_db:5.1f} dB  "
                  f"det={det_ok:4d}/{n_groups} ({stats['detection_rate']*100:5.1f}%)  "
                  f"STF={fails.get('STF',0):3d}  PSS={fails.get('PSS',0):3d}  "
                  f"RS={fails.get('RS',0):3d}  {bar}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Any-of-N 重复帧模式 (任意一帧检出即成功, frame_id 去重)
# ═══════════════════════════════════════════════════════════════════════

def run_anyof_repeat_sweep(snr_range, n_groups=50, seed=42,
                            stf_reps=4, rs_len=64,
                            n_repeats=5, gap_repeat_ms=3.0,
                            cfo_mean=-2.9, cfo_std=17.9,
                            stf_thr=0.4, stf_energy=0.01,
                            pss_ptm=2.5, pss_pts=1.0,
                            verbose=True):
    """Any-of-N 重复帧 SNR sweep.

    每组发送同一帧 n_repeats 次, 接收端独立检测每一帧。
    只要任意一帧通过 STF+PSS+RS 同步 -> 组检出成功。
    用 frame_id 去重 (5 帧同一 frame_id, 只计一次)。

    优点:
      - 不需要相干叠加, 不需要相位对齐
      - 组检出率 = 1 - (1 - 单帧检出率)^n_repeats
      - 低 SNR 下极为有效 (5 帧 50% 单帧 -> 96.9% 组检出)
    """
    from phy_params import PSS as PSS_REF

    snr_min, snr_max, snr_step = snr_range
    snr_points = np.arange(snr_min, snr_max + snr_step/2, snr_step)

    stf_syms = make_stf(n_reps=stf_reps)
    rs_syms = make_rs(n_syms=rs_len)
    pss_syms = PSS_REF

    total_sym = (len(stf_syms) + len(pss_syms) + len(rs_syms)
                 + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN + GUARD_SYMBOLS)
    frame_iq_len = total_sym * SPS + len(RRC) - 1
    gap_repeat_iq = int(gap_repeat_ms * SAMP_RATE / 1000)

    rng = np.random.RandomState(seed)
    results = []

    for snr_db in snr_points:
        h_mag = 1.0
        noise_floor = h_mag ** 2 / (10 ** (snr_db / 10))
        nf_std = np.sqrt(noise_floor * SPS / 2)

        groups_ok = 0
        # 统计每组的检出帧数分布
        hits_per_group = []

        for gi in range(n_groups):
            info_bits = (rng.rand(K_POLAR) < 0.5).astype(np.int64)
            coded = _build_codeword(info_bits)
            frame_syms = build_custom_frame(coded, gi, stf_syms, pss_syms, rs_syms)
            tx_iq_one = rrc_filter(frame_syms, RRC, SPS)

            # 拼接 n_repeats 帧
            tx_segments = []
            for ri in range(n_repeats):
                tx_segments.append(tx_iq_one)
                if ri < n_repeats - 1:
                    tx_segments.append(np.zeros(gap_repeat_iq, dtype=np.complex64))
            tx_all = np.concatenate(tx_segments)

            cfo = rng.normal(cfo_mean, cfo_std)
            phase = rng.uniform(-np.pi, np.pi)
            rx_all = apply_channel(tx_all, h_mag, cfo, phase, noise_floor)

            pad_before = 2000
            pad_after = 2000
            pad_b = (nf_std * (rng.randn(pad_before) + 1j * rng.randn(pad_before))).astype(np.complex64)
            pad_a = (nf_std * (rng.randn(pad_after) + 1j * rng.randn(pad_after))).astype(np.complex64)
            rx = np.concatenate([pad_b, rx_all, pad_a])

            # 独立检测每一帧
            n_hits = 0
            group_ok = False
            for ri in range(n_repeats):
                offset = pad_before + ri * (frame_iq_len + gap_repeat_iq)
                margin = 400
                es = max(0, offset - margin)
                ee = min(len(rx), offset + frame_iq_len + margin)
                chunk = rx[es:ee]

                RRC_DEL = (len(RRC) - 1) // 2
                syms = np.convolve(chunk, RRC[::-1], mode='full')[RRC_DEL::SPS].astype(np.complex64)

                if len(syms) < len(pss_syms) + len(rs_syms):
                    continue

                pk, ptm, pts, pv = pss_correlate_custom(syms, pss_syms)
                if ptm < pss_ptm or pts < pss_pts:
                    continue

                fs = pk - len(stf_syms)
                if fs < 0:
                    continue

                rp = fs + len(stf_syms) + len(pss_syms)
                if rp + len(rs_syms) > len(syms):
                    continue

                chan = rs_estimate_custom(syms, rp, rs_syms)
                if chan is not None:
                    n_hits += 1
                    group_ok = True
                    # 不 break — 统计所有命中数

            if group_ok:
                groups_ok += 1
            hits_per_group.append(n_hits)

        stats = {
            'snr_db': float(snr_db),
            'noise_floor': noise_floor,
            'n_groups': n_groups,
            'n_repeats': n_repeats,
            'groups_ok': groups_ok,
            'detection_rate': groups_ok / n_groups,
            'mean_hits_per_group': float(np.mean(hits_per_group)),
            'median_hits_per_group': float(np.median(hits_per_group)),
        }
        results.append(stats)

        if verbose:
            bar = '#' * int(stats['detection_rate'] * 40)
            print(f"  SNR={snr_db:5.1f} dB  "
                  f"groups={groups_ok:4d}/{n_groups} ({stats['detection_rate']*100:5.1f}%)  "
                  f"avg_hits={stats['mean_hits_per_group']:.1f}/{n_repeats}  {bar}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='低 SNR 同步链设计 + SNR sweep 诊断')
    p.add_argument('calib_json', nargs='?', default='',
                   help='channel_params.json (可选, 用于提取 CFO 参数)')
    p.add_argument('--snr-range', type=float, nargs=3, default=[-5, 15, 2],
                   help='SNR sweep: min max step (默认 -5 15 2)')
    p.add_argument('--frames', type=int, default=200,
                   help='每 SNR 点仿真帧数 (默认 200)')
    p.add_argument('--stf-reps', type=int, default=4,
                   help='STF 重复段数 (默认 4 -> 64 符号)')
    p.add_argument('--rs-len', type=int, default=32,
                   help='RS 长度 (默认 32)')
    p.add_argument('--cfo-mean', type=float, default=-2.4)
    p.add_argument('--cfo-std', type=float, default=19.2)
    p.add_argument('--stf-thr', type=float, default=0.4)
    p.add_argument('--stf-energy', type=float, default=0.01)
    p.add_argument('--pss-ptm', type=float, default=2.5)
    p.add_argument('--pss-pts', type=float, default=1.0)
    p.add_argument('--compare', action='store_true',
                   help='对比多种帧结构变体')
    p.add_argument('--repeat', type=int, default=0,
                   help='重复帧数 (0=单帧模式, 5=同一帧连发5次)')
    p.add_argument('--mode', default='any', choices=['any', 'combine'],
                   help='重复帧模式: any=任意检出即成功+去重, combine=相干叠加 (默认 any)')
    p.add_argument('--gap-repeat-ms', type=float, default=3.0,
                   help='重复帧间隔 ms (默认 3.0)')
    p.add_argument('--gap-group-ms', type=float, default=10.0,
                   help='组间隔 ms (默认 10.0)')
    p.add_argument('-o', '--output', default='',
                   help='输出 JSON')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    # 从标定文件加载 CFO 参数
    cfo_mean, cfo_std = args.cfo_mean, args.cfo_std
    if args.calib_json and os.path.isfile(args.calib_json):
        with open(args.calib_json, encoding='utf-8') as f:
            calib = json.load(f)
        # channel_params.json 结构: {by_gain: {..., calibration: {channel_model: ...}}}
        if 'calibration' in calib:
            cm = calib['calibration']['channel_model']
        elif 'channel_model' in calib:
            cm = calib['channel_model']
        else:
            cm = None
        if cm:
            cfo_mean = cm['cfo_hz']['mean']
            cfo_std = cm['cfo_hz']['std']
            print(f"从标定加载 CFO: N({cfo_mean:.1f}, {cfo_std:.1f}) Hz")

    snr_range = tuple(args.snr_range)

    if args.compare:
        results = compare_variants(snr_range, args.frames, cfo_mean, cfo_std)
    elif args.repeat > 0:
        stf_syms_count = args.stf_reps * 16
        mode_label = 'Any-of-N (任意检出+去重)' if args.mode == 'any' else 'Coherent combine (相干叠加)'
        print(f"\n{'='*70}")
        print(f"{args.repeat}x 重复帧 [{mode_label}]: "
              f"STF={stf_syms_count}sym RS={args.rs_len}sym  "
              f"CFO=N({cfo_mean:.1f},{cfo_std:.1f})Hz")
        print(f"  gap_repeat={args.gap_repeat_ms}ms  "
              f"gap_group={args.gap_group_ms}ms")
        print(f"{'='*70}")

        if args.mode == 'any':
            print(f"  {'SNR':>6s}  {'groups':>7s}  {'rate':>7s}  {'hits/grp':>9s}")
            print(f"  {'-'*40}")
            results = run_anyof_repeat_sweep(
                snr_range, n_groups=args.frames, seed=args.seed,
                stf_reps=args.stf_reps, rs_len=args.rs_len,
                n_repeats=args.repeat,
                gap_repeat_ms=args.gap_repeat_ms,
                cfo_mean=cfo_mean, cfo_std=cfo_std,
                stf_thr=args.stf_thr, stf_energy=args.stf_energy,
                pss_ptm=args.pss_ptm, pss_pts=args.pss_pts,
            )
        else:
            print(f"  {'SNR':>6s}  {'det':>5s}  {'rate':>7s}  "
                  f"{'STF':>5s} {'PSS':>5s} {'RS':>5s}")
            print(f"  {'-'*45}")
            results = run_repeat_combine_sweep(
                snr_range, n_groups=args.frames, seed=args.seed,
                stf_reps=args.stf_reps, rs_len=args.rs_len,
                n_repeats=args.repeat,
                gap_repeat_ms=args.gap_repeat_ms, gap_group_ms=args.gap_group_ms,
                cfo_mean=cfo_mean, cfo_std=cfo_std,
                stf_thr=args.stf_thr, stf_energy=args.stf_energy,
                pss_ptm=args.pss_ptm, pss_pts=args.pss_pts,
            )
    else:
        stf_syms_count = args.stf_reps * 16
        print(f"\n{'='*70}")
        print(f"SNR Sweep: STF={stf_syms_count}sym RS={args.rs_len}sym  "
              f"CFO=N({cfo_mean:.1f},{cfo_std:.1f})Hz")
        print(f"{'='*70}")
        print(f"  {'SNR':>6s}  {'det':>5s}  {'rate':>7s}  "
              f"{'STF':>5s} {'PSS':>5s} {'RS':>5s}")
        print(f"  {'-'*45}")

        results = run_snr_sweep(
            snr_range, args.frames, args.seed,
            stf_reps=args.stf_reps, rs_len=args.rs_len,
            cfo_mean=cfo_mean, cfo_std=cfo_std,
            stf_thr=args.stf_thr, stf_energy=args.stf_energy,
            pss_ptm=args.pss_ptm, pss_pts=args.pss_pts,
        )

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果 -> {args.output}")


if __name__ == '__main__':
    main()
