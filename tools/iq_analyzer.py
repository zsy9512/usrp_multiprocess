#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iq_analyzer.py — 离线 IQ 分析 + 可视化 + 增益推荐

用法:
  python tools/iq_analyzer.py capture.npy               # 文本分析
  python tools/iq_analyzer.py capture.npy --plot        # + 绘图窗口
  python tools/iq_analyzer.py capture.npy --save plot   # 保存 PNG
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phy_params import SPS, STF_DELAY, STF_THRESHOLD, STF_MIN_ENERGY
from phy_params import FRAME_SYMBOLS, FRAME_RRC_SAMPLES

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def analyze(filepath, samp_rate=1e6, do_plot=False, save_prefix=''):
    if not os.path.isfile(filepath):
        print(f"文件不存在: {filepath}"); return

    iq = np.load(filepath)
    if len(iq) == 0:
        print("空文件!"); return

    n_total = len(iq)
    dur_ms = n_total / samp_rate * 1000
    print(f"\n{'='*60}")
    print(f"IQ 文件: {os.path.basename(filepath)}")
    print(f"  样本: {n_total}  ({dur_ms:.1f} ms @ {samp_rate/1e6:.1f}Msps)")
    print(f"{'='*60}")

    # ============================================================
    # 1. 幅度统计 + 增益推荐
    # ============================================================
    mag = np.abs(iq)
    peak = np.max(mag)
    rms = np.sqrt(np.mean(mag ** 2))
    clipped_pct = np.sum(mag > 0.98) / n_total * 100 if peak > 0 else 0

    print(f"\n【幅度统计】")
    print(f"  max  = {peak:.4f}")
    print(f"  RMS  = {rms:.4f}")
    print(f"  crest = {peak/(rms+1e-30):.1f}")

    # 增益推荐
    print(f"\n【增益诊断】")
    if peak < 0.01:
        print(f"  [FAIL] 无信号 (max={peak:.4f}) — 检查 TX 是否发射、SMA 是否连接")
    elif clipped_pct > 5:
        print(f"  [WARN] 严重削峰 ({clipped_pct:.1f}% 样本饱和) — 降低 RX gain 或加衰减器")
        print(f"  -> 建议 RX gain 降低 {int(clipped_pct)} dB")
    elif clipped_pct > 1:
        print(f"  [WARN] 轻微削峰 ({clipped_pct:.1f}%) — 可适当降低 RX gain")
    elif peak < 0.15:
        print(f"  [WARN] 信号偏弱 (max={peak:.3f}) — 提高 TX gain 或 RX gain")
        boost = int(-20 * np.log10(peak / 0.5))
        print(f"  -> 建议增益提高 ~{boost} dB (目标 max≈0.5)")
    elif peak < 0.3:
        print(f"  [OK] 可用但偏弱 (max={peak:.3f})")
        boost = int(-20 * np.log10(peak / 0.5))
        print(f"  -> 可选增益 +{boost} dB")
    elif peak <= 0.85:
        print(f"  [OK] 增益合适 (max={peak:.3f})")
    else:
        print(f"  [WARN] 接近削峰 (max={peak:.3f})")

    # ============================================================
    # 2. 功率谱
    # ============================================================
    nfft = 2048
    n_seg = n_total // nfft
    if n_seg > 0:
        segs = iq[:n_seg * nfft].reshape(n_seg, nfft)
        win = np.kaiser(nfft, 8.0)
        pxx = np.mean(np.abs(np.fft.fftshift(np.fft.fft(segs * win, axis=1), axes=1)) ** 2, axis=0)
        pxx_db = 10 * np.log10(pxx + 1e-30)
        pxx_db -= np.max(pxx_db)
        freq = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / samp_rate)) / 1e3
        peak_bin = np.argmax(pxx_db) - nfft // 2

        cfo_khz = freq[nfft//2 + peak_bin]
        print(f"\n【功率谱】")
        print(f"  峰值频偏: {cfo_khz:+.1f} kHz (两台B210时钟差)")
        if abs(cfo_khz) > 3:
            print(f"  [WARN] 频偏过大 ({cfo_khz:+.1f}kHz) — 超出BPSK容忍, 考虑external_ref")
        elif abs(cfo_khz) > 0.5:
            print(f"  [WARN] 频偏 {cfo_khz:+.1f}kHz — 同步链可纠正")

    # ============================================================
    # 3. STF 延迟相关
    # ============================================================
    L = STF_DELAY
    r0 = iq[:n_total - L]; rL = iq[L:]
    prod = r0 * np.conj(rL)
    P = np.convolve(prod, np.ones(L), mode='valid')
    E = np.convolve((np.abs(rL) ** 2).astype(np.float32), np.ones(L), mode='valid')
    metric = np.abs(P) / (E + 1e-6 * L)

    # 只在 metric > 0.3 且有能量的区域找峰 (排除 0/0 假峰)
    peaks = []
    for d in range(len(metric)):
        if metric[d] > 0.3:
            # 排除无能量区域 (E≈0 产生虚假高 M)
            local_E = E[d] if d < len(E) else 0
            if local_E > STF_MIN_ENERGY * 2:  # 提高能量门槛
                peaks.append(d)

    # 去重合并
    groups = []
    last = -9999
    for d in peaks:
        if d - last > FRAME_RRC_SAMPLES // 3:
            groups.append([d])
        else:
            groups[-1].append(d)
        last = d

    n_frames = len(groups)
    print(f"\n【帧检测】")
    print(f"  STF 峰值组: {n_frames}  (候选帧数)")
    if n_frames > 0:
        gaps = np.diff([np.median(g) for g in groups])
        gap_mean = np.mean(gaps)
        frame_len = FRAME_RRC_SAMPLES  # 1012
        gap_between = gap_mean - frame_len
        print(f"  帧间距: {gap_mean:.0f} 样本 (帧 {frame_len} + 间隔 ~{gap_between:.0f})")
        if gap_between < 0:
            print(f"  [WARN] 间距异常 — 帧可能有重叠或 gap 太小")

    # ============================================================
    # 4. 帧能量分布
    # ============================================================
    frame_energies = []
    for g in groups:
        mid = int(np.median(g))
        start = max(0, mid - 20)
        end = min(n_total, mid + FRAME_RRC_SAMPLES + 20)
        frame_energies.append(float(np.mean(np.abs(iq[start:end]) ** 2)))
    if frame_energies:
        e_mean, e_std = np.mean(frame_energies), np.std(frame_energies)
        print(f"\n【帧能量】")
        print(f"  均值={e_mean:.4f}  sigma={e_std:.4f}")
        if e_std / (e_mean + 1e-30) > 0.3:
            print(f"  [WARN] 帧能量波动大 — 可能有 overflow 导致帧不完整")

    print(f"\n{'='*60}")
    print(f"总结: {n_frames} 候选帧, 频偏 {cfo_khz:+.1f}kHz, 峰值 {peak:.3f}")
    print(f"{'='*60}\n")

    # ============================================================
    # 5. 绘图
    # ============================================================
    if do_plot or save_prefix:
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1])

        # (a) 全数据原始时域 — 跨两列
        ax = fig.add_subplot(gs[0, :])
        # 长文件降采样到最多 100k 点
        step = max(1, n_total // 100000)
        t_full = np.arange(0, n_total, step) / samp_rate * 1000
        ax.plot(t_full, np.real(iq[::step]), lw=0.15, color='#1f77b4')
        ax.axhline(1, color='gray', ls='--', lw=0.5, alpha=0.5)
        ax.axhline(-1, color='gray', ls='--', lw=0.5, alpha=0.5)
        ax.axhline(peak, color='red', ls=':', lw=0.8, alpha=0.7, label=f'peak={peak:.3f}')
        ax.axhline(-peak, color='red', ls=':', lw=0.8, alpha=0.7)
        ax.set_title(f'全数据时域 ({n_total} 样本, {dur_ms:.0f} ms, 降采样 {step}x)')
        ax.set_xlabel('ms'); ax.set_ylabel('I'); ax.legend(fontsize=8)

        # (b) 功率谱
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(freq, pxx_db, lw=0.8)
        ax.axvline(0, color='gray', ls='--', lw=0.5)
        ax.set_title(f'功率谱 (峰值: {cfo_khz:+.1f} kHz)')
        ax.set_xlabel('kHz'); ax.set_ylabel('dB'); ax.set_ylim(-60, 5)

        # (c) STF 相关
        ax = fig.add_subplot(gs[1, 1])
        show_n = min(5000, len(metric))
        ax.plot(np.arange(show_n), metric[:show_n], lw=0.5)
        ax.axhline(STF_THRESHOLD, color='r', ls='--', lw=0.8, label=f'thr={STF_THRESHOLD}')
        ax.set_title(f'STF 延迟相关 (前 {show_n} 点, {n_frames} 组峰)')
        ax.set_xlabel('d (samples)'); ax.set_ylabel('M(d)'); ax.legend()

        # (d) 帧能量直方图
        ax = fig.add_subplot(gs[2, 0])
        if frame_energies:
            ax.hist(frame_energies, bins=30, edgecolor='k', alpha=0.7)
            ax.axvline(e_mean, color='r', lw=1.5, label=f'mean={e_mean:.3f}')
            ax.legend()
        ax.set_title(f'帧能量分布 ({len(frame_energies)} 帧)')
        ax.set_xlabel('|IQ|^2 mean')

        # (e) 幅值分布
        ax = fig.add_subplot(gs[2, 1])
        ax.hist(np.abs(iq[::max(1, n_total//50000)]), bins=50, edgecolor='k', alpha=0.5)
        ax.axvline(0.98, color='r', ls='--', lw=0.8, label='clip thr')
        ax.set_title('幅值分布')
        ax.set_xlabel('|IQ|'); ax.legend(fontsize=8)

        fig.suptitle(f'{os.path.basename(filepath)}  |  {n_total} samples  |  '
                     f'peak={peak:.3f}  CFO={cfo_khz:+.1f}kHz  frames={n_frames}',
                     fontsize=10)
        fig.tight_layout()

        if save_prefix:
            out = f'{save_prefix}_{os.path.basename(filepath).replace(".npy","")}.png'
            fig.savefig(out, dpi=150)
            print(f"图已保存 -> {out}")
        if do_plot:
            plt.show()
        else:
            plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('file', help='IQ .npy 文件路径')
    p.add_argument('--rate', type=float, default=1e6, help='采样率 (Hz)')
    p.add_argument('--plot', action='store_true', help='弹出绘图窗口')
    p.add_argument('--save', default='', help='保存 PNG 前缀 (如 "analysis")')
    args = p.parse_args()
    analyze(args.file, args.rate, args.plot, args.save)
