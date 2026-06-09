#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
channel_measure.py — 全帧已知导频信道精确测量

原理:
  TX 端发送固定已知帧 (seed 固定, 接收端可完美重建 TX 波形),
  RX 端将整帧 496 符号全部当作导频使用,
  通过全帧互相关检测 + 全帧线性相位拟合 + 全帧最小二乘信道估计,
  在 SNR << 0 dB 下仍能精确提取信道参数。

与三级同步 (STF+PSS+RS) 的区别:
  - 三级同步: 160/496 符号用作导频, 设计目标是"未知数据下盲检测"
  - 全帧测量: 496/496 符号用作导频, 设计目标是"已知数据下精确测量"
  - 处理增益: ~27 dB (496 符号) vs ~15 dB (32 符号 RS)
  - 低 SNR 可用: 全帧互相关可在 SNR < -5 dB 下检测帧位置

工作流:
  1. 从 capture 读取 RX IQ + 元数据
  2. 用相同种子重建 TX IQ (每个 frame 完全一致)
  3. 全帧互相关 -> 帧位置 (精确到样本)
  4. RRC 匹配滤波 -> 符号域
  5. 全帧线性相位拟合 -> CFO (496 点, 远优于 RS 的 32 点)
  6. 全帧 LS 信道估计 -> |h|, phase
  7. Guard 区间噪声测量 -> noise_floor (独立于信号的纯底噪)
  8. 残差噪声 -> sigma2 (包含信道估计误差)
  9. 按 gain 分组, 输出精确的 channel_params.json

用法:
  python tools/channel_measure.py capture/20260609 -o channel_params.json
  python tools/channel_measure.py capture/20260609 --gain 55 --plot
"""

import argparse, json, os, sys, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from phy_params import (SPS, STF, PSS, RS, RRC,
                        STF_LEN, PSS_LEN, RS_LEN,
                        HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN,
                        GUARD_SYMBOLS, FRAME_SYMBOLS,
                        RRC_DELAY_SAMPLES, STF_DELAY,
                        FRAME_RRC_SAMPLES)
from sender import build_frame, rrc_filter

SAMP_RATE = 1e6
TS_SYM = SPS / SAMP_RATE
TS = 1.0 / SAMP_RATE

# ═══════════════════════════════════════════════════════════════════════
# TX 波形重建 (与 loopback_capture.py 完全一致的随机种子)
# ═══════════════════════════════════════════════════════════════════════

def reconstruct_tx_waveform(frame_id, rng=None):
    """重建指定 frame_id 的 TX IQ 波形.

    使用与 loopback_capture.py 完全相同的 RandomState(42) + randint 生成比特,
    确保重建的波形与硬件发送的波形逐样本一致.

    Args:
        frame_id: 帧序号 (0, 1, 2, ...)
        rng:      np.random.RandomState(42) 实例 (复用避免重复创建)

    Returns:
        tx_iq:     (FRAME_RRC_SAMPLES,) complex64  TX IQ 样本
        tx_bits:   (PAYLOAD_LEN,) int64  发送比特
        tx_syms:   (FRAME_SYMBOLS,) complex64  符号域帧 (RRC 之前)
    """
    if rng is None:
        rng = np.random.RandomState(42)
    raw = rng.randint(0, 2, PAYLOAD_LEN).astype(np.int64)
    frame_syms = build_frame(raw, frame_id)
    tx_iq = rrc_filter(frame_syms, RRC, SPS)
    return tx_iq.astype(np.complex64), raw, frame_syms


# ═══════════════════════════════════════════════════════════════════════
# 全帧互相关帧检测 (处理增益 ~27 dB, 远优于 STF 延迟相关的 ~15 dB)
# ═══════════════════════════════════════════════════════════════════════

def full_frame_correlate(rx_iq, tx_iq_ref):
    """全帧互相关: 在 RX IQ 中搜索已知 TX 波形.

    互相关 = conv(rx, conj(tx_ref[::-1]))
    峰值位置 = 帧起始样本索引.

    Args:
        rx_iq:      (N,) complex64  接收 IQ
        tx_iq_ref:  (M,) complex64  参考 TX IQ (单帧)

    Returns:
        corr:       (N-M+1,) float32  互相关幅度
        best_pos:   int  最佳匹配位置 (样本索引)
        best_val:   float  峰值相关值
    """
    tx_rev = np.conj(tx_iq_ref[::-1])
    corr = np.abs(np.convolve(rx_iq, tx_rev, mode='valid'))
    best_pos = int(np.argmax(corr))
    best_val = float(corr[best_pos])
    return corr.astype(np.float32), best_pos, best_val


# ═══════════════════════════════════════════════════════════════════════
# 全帧 CFO 估计 (496 符号线性相位拟合, vs RS 的 32 符号)
# ═══════════════════════════════════════════════════════════════════════

def full_frame_cfo_estimate(rx_syms, tx_syms_ref, ts_sym=TS_SYM):
    """全帧线性相位拟合 -> 频偏.

    对每个符号计算 phase_diff = angle(rx * conj(tx_ref)),
    unwrap 后在全部 496 符号上做线性回归,
    精度约为 RS-only (32符号) 的 sqrt(496/32) ≈ 3.9 倍.

    Args:
        rx_syms:     (FRAME_SYMBOLS,) complex64  接收符号 (RRC 匹配后)
        tx_syms_ref: (FRAME_SYMBOLS,) complex64  参考符号
        ts_sym:      float  符号时间

    Returns:
        cfo_hz:    float  频偏 Hz
        phase_0:   float  初始相位 rad (截距)
        r_squared: float  拟合优度 (接近 1 = 纯单频偏)
    """
    # 非零符号上的相位差 (排除 Guard 零符号)
    mask = np.abs(tx_syms_ref) > 0.01
    if np.sum(mask) < 10:
        return 0.0, 0.0, 0.0

    idx = np.where(mask)[0]
    phase_diff = np.angle(rx_syms[idx] * np.conj(tx_syms_ref[idx]))
    phase_unwrapped = np.unwrap(phase_diff)

    n = idx.astype(np.float64)
    n_mean = np.mean(n)
    p_mean = np.mean(phase_unwrapped)

    num = np.sum((n - n_mean) * (phase_unwrapped - p_mean))
    den = np.sum((n - n_mean) ** 2)
    slope = num / (den + 1e-30)
    intercept = p_mean - slope * n_mean

    cfo_hz = float(slope / (2 * np.pi * ts_sym))

    # R^2
    phase_pred = slope * n + intercept
    ss_res = np.sum((phase_unwrapped - phase_pred) ** 2)
    ss_tot = np.sum((phase_unwrapped - p_mean) ** 2)
    r_sq = float(1 - ss_res / (ss_tot + 1e-30))

    return cfo_hz, float(intercept), r_sq


# ═══════════════════════════════════════════════════════════════════════
# 全帧信道估计 (最小二乘, 496 符号)
# ═══════════════════════════════════════════════════════════════════════

def full_frame_channel_estimate(rx_syms, tx_syms_ref, cfo_hz, phase_0, ts_sym=TS_SYM):
    """全帧 LS 信道估计.

    y = h * x_ref * e^(j*2pi*CFO*t + j*phi0) + noise
    y_corrected = y * e^(-j*2pi*CFO*t - j*phi0)
    h_LS = mean(y_corrected * conj(x_ref)) / mean(|x_ref|^2)

    Args:
        rx_syms, tx_syms_ref: 符号序列
        cfo_hz, phase_0: 全帧 CFO 估计结果
        ts_sym: 符号时间

    Returns:
        h:           complex  信道系数
        h_mag:       float    |h|
        h_phase:     float    angle(h)
        sigma2_data: float    数据段残差噪声方差
    """
    n_sym = np.arange(len(rx_syms))
    total_phase = 2 * np.pi * cfo_hz * n_sym * ts_sym + phase_0
    rx_corrected = rx_syms * np.exp(-1j * total_phase)

    # LS: 只在非零符号上估计
    mask = np.abs(tx_syms_ref) > 0.01
    num = np.sum(rx_corrected[mask] * np.conj(tx_syms_ref[mask]))
    den = np.sum(np.abs(tx_syms_ref[mask]) ** 2)
    h = num / (den + 1e-30)

    # 残差噪声 (数据段, 非 Guard)
    data_mask = np.zeros(len(tx_syms_ref), dtype=bool)
    guard_start = STF_LEN + PSS_LEN + RS_LEN + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN
    data_mask[:guard_start] = (np.abs(tx_syms_ref[:guard_start]) > 0.01)
    noise_data = rx_corrected[data_mask] / h - tx_syms_ref[data_mask]
    sigma2_data = max(float(np.sum(np.abs(noise_data) ** 2) / (np.sum(data_mask) - 1)), 1e-30)

    return {
        'h': h,
        'h_mag': float(abs(h)),
        'h_phase': float(np.angle(h)),
        'sigma2_data': sigma2_data,
    }


# ═══════════════════════════════════════════════════════════════════════
# Guard 区间噪声测量 (独立于信号的纯底噪)
# ═══════════════════════════════════════════════════════════════════════

def guard_noise_estimate(rx_syms, guard_start, guard_len=GUARD_SYMBOLS):
    """从 Guard 区间 (零符号) 测量独立噪声方差.

    Guard 符号在 TX 端是纯零 -> RX 端收到的就是纯噪声.
    这是最干净的底噪测量, 不依赖信道估计.
    """
    if guard_start + guard_len > len(rx_syms):
        return 0.0
    guard_seg = rx_syms[guard_start:guard_start + guard_len]
    return float(np.var(guard_seg))


# ═══════════════════════════════════════════════════════════════════════
# 前缀底噪测量 (与 polar_loopback.py 一致)
# ═══════════════════════════════════════════════════════════════════════

def prefix_noise_floor(iq, n_samples=50000):
    """从 IQ 前缀测量符号域噪声方差."""
    n = min(n_samples, len(iq))
    seg = iq[:n]
    filt = np.convolve(seg, RRC[::-1], mode='full')
    syms = filt[RRC_DELAY_SAMPLES::SPS]
    return float(np.var(syms))


# ═══════════════════════════════════════════════════════════════════════
# 主提取函数: 逐帧精确测量
# ═══════════════════════════════════════════════════════════════════════

def measure_channel(iq_path, num_frames=200, seed=42,
                   min_corr_peak=0.0, cfo_max_hz=500):
    """对单个 capture 做全帧导频信道测量.

    Args:
        iq_path:      _iq.npy 路径
        num_frames:   发送帧数
        seed:         TX 随机种子
        min_corr_peak: 最小互相关峰值 (0 = 自动)
        cfo_max_hz:   CFO 合理上限 (超过此值标记为异常)

    Returns:
        dict with:
          - per_frame: list[dict]  每帧测量结果
          - noise_floor_prefix: float
          - meta: dict
    """
    rx_iq = np.load(iq_path)
    n_total = len(rx_iq)

    # 元数据
    meta = {}
    meta_path = iq_path.replace('_iq.npy', '_meta.json')
    if os.path.isfile(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)

    # 前缀底噪
    nf_prefix = prefix_noise_floor(rx_iq)

    # TX RNG
    rng = np.random.RandomState(seed)

    per_frame = []
    guard_pos = STF_LEN + PSS_LEN + RS_LEN + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN

    for fid in range(num_frames):
        # 重建 TX 波形
        tx_iq_ref, tx_bits_ref, tx_syms_ref = reconstruct_tx_waveform(fid, rng)

        # 全帧互相关 -> 帧位置
        corr, best_pos, best_val = full_frame_correlate(rx_iq, tx_iq_ref)
        if best_val < min_corr_peak:
            continue

        # 精确提取: 在相关峰位置直接取帧长 IQ, RRC 匹配后即得对齐符号
        frame_len_iq = len(tx_iq_ref)
        if best_pos + frame_len_iq > n_total:
            continue
        frame_iq = rx_iq[best_pos:best_pos + frame_len_iq]

        # RRC 匹配滤波 (与 TX 端完全对称, 输出约 FRAME_SYMBOLS 个符号)
        filt = np.convolve(frame_iq, RRC[::-1], mode='full')
        rx_syms = filt[RRC_DELAY_SAMPLES::SPS].astype(np.complex64)

        # 取前 FRAME_SYMBOLS 个符号 (RRC mode='full' 会产生额外拖尾)
        if len(rx_syms) < FRAME_SYMBOLS:
            continue
        frame_rx_syms = rx_syms[:FRAME_SYMBOLS]

        # 全帧 CFO 估计
        cfo_hz, phase_0, r_sq = full_frame_cfo_estimate(frame_rx_syms, tx_syms_ref)

        # CFO 超限检查
        cfo_valid = abs(cfo_hz) < cfo_max_hz

        # 全帧信道估计
        chan = full_frame_channel_estimate(frame_rx_syms, tx_syms_ref, cfo_hz, phase_0)

        # Guard 噪声
        nf_guard = guard_noise_estimate(frame_rx_syms, guard_pos)

        # SNR
        snr_prefix = float(10 * np.log10(max(chan['h_mag'] ** 2 / max(nf_prefix, 1e-30), 1e-30)))
        snr_guard = float(10 * np.log10(max(chan['h_mag'] ** 2 / max(nf_guard, 1e-30), 1e-30)))
        snr_data = float(10 * np.log10(max(chan['h_mag'] ** 2 / max(chan['sigma2_data'], 1e-30), 1e-30)))

        per_frame.append({
            'frame_id': fid,
            'global_pos': best_pos,
            'corr_peak': best_val,
            'cfo_hz': cfo_hz,
            'cfo_valid': cfo_valid,
            'cfo_r_squared': r_sq,
            'phase_0_rad': phase_0,
            'h_mag': chan['h_mag'],
            'h_phase_rad': chan['h_phase'],
            'sigma2_data': chan['sigma2_data'],
            'noise_floor_guard': nf_guard,
            'snr_prefix_db': snr_prefix,
            'snr_guard_db': snr_guard,
            'snr_data_db': snr_data,
        })

    return {
        'per_frame': per_frame,
        'noise_floor_prefix': nf_prefix,
        'meta': meta,
        'n_total_frames': num_frames,
        'n_detected': len(per_frame),
    }


# ═══════════════════════════════════════════════════════════════════════
# 批量处理 + 统计汇总
# ═══════════════════════════════════════════════════════════════════════

def process_captures(prefixes, num_frames=200, **kwargs):
    """批量处理, 按 gain 分组统计."""
    results = {}
    for prefix in prefixes:
        iq_path = prefix + '_iq.npy'
        if not os.path.isfile(iq_path):
            print(f"  [skip] {prefix}")
            continue
        print(f"  measuring: {os.path.basename(prefix)} ...")
        t0 = time.time()
        m = measure_channel(iq_path, num_frames=num_frames, **kwargs)
        elapsed = time.time() - t0
        gain = m['meta'].get('gain_rx_db', 'unknown')
        print(f"    {m['n_detected']}/{num_frames} frames  [{elapsed:.1f}s]")

        key = str(gain)
        if key not in results:
            results[key] = {'frames': [], 'noise_floors_prefix': [], 'meta': m['meta']}
        results[key]['frames'].extend(m['per_frame'])
        results[key]['noise_floors_prefix'].append(m['noise_floor_prefix'])

    # 统计
    summary = {}
    for gain_str, data in results.items():
        frames = data['frames']
        n = len(frames)
        if n == 0:
            summary[gain_str] = {'n_frames': 0}
            continue

        # 过滤 CFO 有效的帧
        valid = [f for f in frames if f['cfo_valid']]
        n_valid = len(valid)
        cfos = [f['cfo_hz'] for f in valid]
        h_mags = [f['h_mag'] for f in valid]
        phases = [f['h_phase_rad'] for f in valid]
        snrs_pfx = [f['snr_prefix_db'] for f in valid]
        snrs_data = [f['snr_data_db'] for f in valid]
        r_sqs = [f['cfo_r_squared'] for f in valid]

        summary[gain_str] = {
            'n_detected': n,
            'n_cfo_valid': n_valid,
            'cfo_hz': {
                'mean': float(np.mean(cfos)) if cfos else None,
                'std': float(np.std(cfos)) if cfos else None,
                'min': float(np.min(cfos)) if cfos else None,
                'max': float(np.max(cfos)) if cfos else None,
            },
            'cfo_r_squared_mean': float(np.mean(r_sqs)) if r_sqs else None,
            'h_mag': {
                'mean': float(np.mean(h_mags)) if h_mags else None,
                'std': float(np.std(h_mags)) if h_mags else None,
            },
            'h_phase_rad': {
                'mean': float(np.mean(phases)) if phases else None,
                'std': float(np.std(phases)) if phases else None,
            },
            'snr_prefix_db': {
                'mean': float(np.mean(snrs_pfx)) if snrs_pfx else None,
                'std': float(np.std(snrs_pfx)) if snrs_pfx else None,
            },
            'snr_data_db': {
                'mean': float(np.mean(snrs_data)) if snrs_data else None,
                'std': float(np.std(snrs_data)) if snrs_data else None,
            },
            'noise_floor_prefix': float(np.mean(data['noise_floors_prefix'])),
        }

    # 生成仿真标定配置
    calib = make_calibration_config(summary, results)

    return {'by_gain': summary, 'calibration': calib, 'raw': results}


def make_calibration_config(summary, raw_results):
    """从全帧测量结果生成仿真标定配置."""
    # 找最佳参考 gain (CFO 估计最精确的, 即 r_squared 最高的)
    best_gain = None
    best_r_sq = -1
    for gain_str, stats in summary.items():
        if stats.get('cfo_r_squared_mean', 0) > best_r_sq:
            best_r_sq = stats['cfo_r_squared_mean']
            best_gain = gain_str

    if best_gain is None:
        return {'error': 'no valid measurements'}

    ref = summary[best_gain]

    config = {
        'description': 'Full-frame pilot channel measurement (496-symbol reference)',
        'method': 'full_frame_correlation + full_frame_phase_fit + full_frame_LS',
        'reference_gain_db': best_gain,
        'reference_n_frames': ref['n_cfo_valid'],
        'cfo_estimation_quality': {
            'r_squared_mean': ref['cfo_r_squared_mean'],
            'note': 'R^2 near 1.0 = pure single-tone CFO, < 0.9 = phase noise present',
        },
        'channel_model': {
            'cfo_hz': {
                'distribution': 'normal',
                'mean': ref['cfo_hz']['mean'],
                'std': max(ref['cfo_hz']['std'], 1.0),
            },
            'phase_offset_rad': {
                'distribution': 'uniform',
                'min': -np.pi,
                'max': np.pi,
            },
        },
        'per_gain': {},
    }

    for gain_str, stats in summary.items():
        entry = {
            'noise_floor_prefix': stats.get('noise_floor_prefix'),
        }
        if stats.get('n_cfo_valid', 0) > 0:
            entry.update({
                'h_mag_mean': stats['h_mag']['mean'],
                'h_mag_std': stats['h_mag']['std'],
                'snr_prefix_mean_db': stats['snr_prefix_db']['mean'],
                'snr_data_mean_db': stats['snr_data_db']['mean'],
                'cfo_hz_mean': stats['cfo_hz']['mean'],
                'cfo_hz_std': stats['cfo_hz']['std'],
                'n_frames': stats['n_cfo_valid'],
            })
        config['per_gain'][gain_str] = entry

    return config


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='全帧已知导频信道精确测量')
    p.add_argument('inputs', nargs='+',
                   help='capture 前缀 或 目录')
    p.add_argument('-o', '--output', default='channel_params.json',
                   help='输出 JSON (默认 channel_params.json)')
    p.add_argument('--num-frames', type=int, default=200,
                   help='每 capture 预期帧数')
    p.add_argument('--seed', type=int, default=42,
                   help='TX 随机种子 (默认 42, 对齐 loopback_capture.py)')
    p.add_argument('--cfo-max', type=float, default=500,
                   help='CFO 合理上限 Hz')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    # 展开输入
    import glob as _glob
    prefixes = []
    for pat in args.inputs:
        if os.path.isdir(pat):
            for f in sorted(_glob.glob(os.path.join(pat, '*_iq.npy'))):
                prefixes.append(f.replace('_iq.npy', ''))
        else:
            prefixes.append(pat.replace('_iq.npy', '').replace('_bits.npy', '').replace('_meta.json', ''))

    if not prefixes:
        print("错误: 未找到 capture")
        sys.exit(1)

    print(f"全帧导频信道测量: {len(prefixes)} 个 capture")
    for pfx in prefixes:
        print(f"  {os.path.basename(pfx)}")

    results = process_captures(prefixes, num_frames=args.num_frames,
                               seed=args.seed, cfo_max_hz=args.cfo_max)

    # 打印报告
    print(f"\n{'='*70}")
    print(f"全帧导频信道测量报告 (496-symbol reference)")
    print(f"{'='*70}")

    for gain_str in sorted(results['by_gain'].keys(),
                           key=lambda x: float(x) if x != 'unknown' else 0):
        s = results['by_gain'][gain_str]
        n = s.get('n_detected', 0)
        nv = s.get('n_cfo_valid', 0)
        print(f"\n-- gain={gain_str} dB --")
        print(f"  检出: {n} 帧  (CFO 有效: {nv})")
        if nv > 0:
            print(f"  CFO:     {s['cfo_hz']['mean']:+.2f} +/- {s['cfo_hz']['std']:.2f} Hz  "
                  f"(R^2={s['cfo_r_squared_mean']:.4f})")
            print(f"  |h|:     {s['h_mag']['mean']:.4f} +/- {s['h_mag']['std']:.4f}")
            print(f"  phase:   {s['h_phase_rad']['mean']:+.4f} +/- {s['h_phase_rad']['std']:.4f} rad")
            print(f"  SNR_pfx: {s['snr_prefix_db']['mean']:.1f} +/- {s['snr_prefix_db']['std']:.1f} dB")
            print(f"  SNR_data:{s['snr_data_db']['mean']:.1f} +/- {s['snr_data_db']['std']:.1f} dB")
            print(f"  nf_pfx:  {s['noise_floor_prefix']:.2e}")
        else:
            print(f"  nf_pfx:  {s['noise_floor_prefix']:.2e}  (无有效帧)")

    # 跨 gain CFO 一致性
    gains_with_cfo = {g: s for g, s in results['by_gain'].items()
                      if s.get('cfo_hz', {}).get('mean') is not None}
    if len(gains_with_cfo) >= 2:
        cfo_means = [s['cfo_hz']['mean'] for s in gains_with_cfo.values()]
        cfo_spread = float(np.std(cfo_means))
        print(f"\n-- 跨 gain CFO 一致性 --")
        print(f"  CFO mean 散布: {cfo_spread:.2f} Hz  "
              f"{'[OK] 不依赖增益' if cfo_spread < 20 else '[WARN]'}")
        print(f"  -> 仿真可用 CFO: N({np.mean(cfo_means):.1f}, "
              f"{np.mean([s['cfo_hz']['std'] for s in gains_with_cfo.values()]):.1f}) Hz")

    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        # 只保存统计 + 标定, 不保存逐帧 raw (太大)
        output = {
            'by_gain': results['by_gain'],
            'calibration': results['calibration'],
        }
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n信道参数 -> {args.output}")


if __name__ == '__main__':
    main()
