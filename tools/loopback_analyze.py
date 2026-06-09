#!/usr/bin/env python3
"""
loopback_analyze.py — 环回 IQ 逐帧精细分析 (离线调试同步链)

方案: 全量 STF 扫描 → 聚类去重 → 逐候选帧完整同步链

用法:
  python tools/loopback_analyze.py capture/loopback_sma_v2
  python tools/loopback_analyze.py capture/loopback_sma_v2 --scan
  python tools/loopback_analyze.py capture/loopback_sma_v2 --plot
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phy_params import (SPS, STF, PSS, RS, RRC, STF_LEN, PSS_LEN, RS_LEN,
                        HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN, STF_DELAY,
                        STF_THRESHOLD, STF_MIN_ENERGY, FRAME_RRC_SAMPLES,
                        crc16, crc16_check, bits_to_bytes, bytes_to_bits)
from tools.snr_metrics import (snr_symbol_domain, snr_from_sigma2,
                                evm_from_sigma2, sigma2_welch)

SAMP_RATE = 1e6
TS_SYM = SPS / SAMP_RATE
RRC_DELAY = (len(RRC) - 1) // 2


# ======================================================================
# 同步函数
# ======================================================================

def rrc_match(samples):
    f = np.convolve(samples, RRC[::-1], mode='full')
    return f[RRC_DELAY::SPS].astype(np.complex64)


def stf_detect(samples, min_energy=None):
    """返回所有 >threshold 的位置 + 粗 CFO + 度量值."""
    if min_energy is None: min_energy = STF_MIN_ENERGY
    L = STF_DELAY; N = len(samples)
    if N <= L: return [], [], [], np.array([])
    r0, rL = samples[:N - L], samples[L:]
    prod = r0 * np.conj(rL)
    ones = np.ones(L, dtype=np.float32)
    P = np.convolve(prod, ones, mode='valid')
    E = np.convolve((np.abs(rL)**2).astype(np.float32), ones, mode='valid')
    M_full = np.abs(P) / (E + 1e-6 * L)

    peaks, cfos, metrics = [], [], []
    for d in range(len(M_full)):
        if M_full[d] > STF_THRESHOLD:
            le = np.sum(np.abs(samples[d + L:d + 2 * L])**2)
            if le > min_energy:
                peaks.append(d)
                cfos.append(-np.angle(P[d]) / (2 * np.pi * L / SAMP_RATE))
                metrics.append(float(M_full[d]))
    return peaks, cfos, metrics, M_full


def stf_cluster(peaks, cfos, metrics, win=200):
    """在 win 样本窗内保留最强峰 (去重)."""
    if not peaks: return [], [], []
    arr = list(zip(peaks, cfos, metrics))
    arr.sort(key=lambda x: x[2], reverse=True)
    used = set()
    out_p, out_c, out_m = [], [], []
    for d, cfo, m in arr:
        if d in used: continue
        for dx in range(max(0, d - win), d + win + 1):
            used.add(dx)
        out_p.append(d); out_c.append(cfo); out_m.append(m)
    return out_p, out_c, out_m


def pss_find(syms):
    M = PSS_LEN
    if len(syms) < M: return -1, 0, 0, 0, np.array([])
    pss_rev = np.conj(PSS[::-1])
    c = np.abs(np.convolve(syms, pss_rev, mode='valid'))
    pk = int(np.argmax(c))
    peak_val = float(c[pk])
    ptm = peak_val / (np.mean(c) + 1e-30)
    sv = np.sort(c)[::-1]
    pts = ptm
    for v in sv[1:]:
        idx_list = np.where(np.isclose(c, v))[0]
        for idx in idx_list:
            if abs(idx - pk) > PSS_LEN // 2:
                pts = peak_val / (v + 1e-30)
                break
        if pts != ptm: break
    return pk, ptm, pts, peak_val, c.astype(np.float32)


def rs_estimate(syms, rs_pos, coarse_cfo=0.0):
    if rs_pos + RS_LEN > len(syms): return None
    rs_seg = syms[rs_pos:rs_pos + RS_LEN].copy()
    n_rs = np.arange(RS_LEN)
    if abs(coarse_cfo) > 1.0:
        pre = np.exp(-1j * 2 * np.pi * coarse_cfo * (rs_pos + n_rs) * TS_SYM)
        rs_seg *= pre
    rs_tone = rs_seg * np.conj(RS)
    rs_phase = np.unwrap(np.angle(rs_tone))
    n = np.arange(RS_LEN, dtype=np.float64)
    n_mean = np.mean(n); p_mean = np.mean(rs_phase)
    num = np.sum((n - n_mean) * (rs_phase - p_mean))
    den = np.sum((n - n_mean)**2)
    slope = num / (den + 1e-30)
    fine_cfo = slope / (2 * np.pi * TS_SYM)
    if abs(fine_cfo) > 500: return None
    total_cfo = coarse_cfo + fine_cfo
    total_comp = np.exp(-1j * 2 * np.pi * total_cfo * (rs_pos + n_rs) * TS_SYM)
    rs_corrected = syms[rs_pos:rs_pos + RS_LEN] * total_comp
    h = np.mean(rs_corrected * np.conj(RS))
    if abs(h) < 1e-6: return None
    noise = rs_corrected / h - RS
    s2 = max(float(np.sum(np.abs(noise) ** 2) / (RS_LEN - 1)), 1e-30)
    rs_corr = float(np.abs(np.sum(rs_corrected * np.conj(RS))))
    if rs_corr < RS_LEN * 0.3: return None
    return {'h': h, 'phase_est': float(np.angle(h)), 'sigma2': s2,
            'coarse_cfo': coarse_cfo, 'fine_cfo': fine_cfo,
            'total_cfo': total_cfo, 'rs_corr': rs_corr}


def bpsk_demod(syms, data_start, data_len, chan):
    if data_start + data_len > len(syms):
        return np.zeros(data_len, dtype=np.int64), np.zeros(data_len, dtype=np.complex64)
    seg = syms[data_start:data_start + data_len]
    n = np.arange(data_len)
    total_cfo = chan['total_cfo']
    cfo_comp = np.exp(-1j * 2 * np.pi * total_cfo * (data_start + n) * TS_SYM)
    y = seg * cfo_comp
    h = chan['h']
    if abs(h) > 1e-30: y = y / h
    bits = (y.real < 0).astype(np.int64)
    return bits, y


def b2i(b):
    v = 0
    for x in b: v = (v << 1) | int(x)
    return v


# ======================================================================
# 顺序扫描核心 (替换流式环形缓冲方案)
# ======================================================================

def analyze_frames_sequential(iq, tx_bits, pss_ptm_thr=3.5, pss_pts_thr=1.0,
                              stf_energy=None, verbose=True, debug_n=0):
    """全量 STF 扫描 → 聚类 → 逐候选帧同步链.

    对 IQ 分段做 STF 检测, 合并去重后逐候选位置验证 PSS+RS+解调.
    不依赖滑动窗口消费, 避免帧漏检.
    """
    n_total = len(iq)
    if stf_energy is None:
        stf_energy = STF_MIN_ENERGY

    # ── 标准 SNR 底噪: 从 IQ 开头静默期测量 (RX 先于 TX 启动) ──
    n_noise = min(50000, n_total)
    noise_iq = iq[:n_noise]
    noise_syms = rrc_match(noise_iq)
    noise_floor = float(np.var(noise_syms))
    if verbose:
        print(f"  [scan] Noise floor (IQ prefix, {n_noise} samples): "
              f"{noise_floor:.6f}  ({10*np.log10(max(noise_floor, 1e-30)):.1f} dB)")

    # ── 阶段 1: 全量 STF 检测 (分段处理以控制内存) ──
    seg_size = 1_000_000
    overlap = FRAME_RRC_SAMPLES + 5000  # 帧+gap 长度, 确保帧不会被分割
    all_peaks, all_cfos, all_metrics = [], [], []

    pos = 0
    while pos < n_total:
        end = min(pos + seg_size, n_total)
        seg = iq[pos:end]
        p, c, m, _ = stf_detect(seg, min_energy=stf_energy)
        # 转为全局位置
        all_peaks.extend([x + pos for x in p])
        all_cfos.extend(c)
        all_metrics.extend(m)
        pos += seg_size - min(overlap, seg_size // 2)
        if pos >= n_total: break

    if verbose:
        print(f"  [scan] STF raw peaks: {len(all_peaks)}")

    # ── 过滤离谱粗 CFO (同板 B210 < 2kHz) ──
    valid = [i for i, c in enumerate(all_cfos) if abs(c) < 2000]
    all_peaks  = [all_peaks[i] for i in valid]
    all_cfos   = [all_cfos[i] for i in valid]
    all_metrics = [all_metrics[i] for i in valid]
    if verbose:
        print(f"  [scan] after CFO filter: {len(all_peaks)}")

    # ── 阶段 2: 全局聚类去重 ──
    c_peaks, c_cfos, c_metrics = stf_cluster(all_peaks, all_cfos, all_metrics,
                                              win=FRAME_RRC_SAMPLES // 2)

    if verbose:
        print(f"  [scan] STF clustered: {len(c_peaks)} candidate frames")

    # ── 阶段 3: 逐候选完整同步链 ──
    frames = []
    for pi, (d, coarse_cfo) in enumerate(zip(c_peaks, c_cfos)):
        # 提取帧窗口
        margin = 400
        es = max(0, d - margin)
        ee = min(n_total, d + margin + FRAME_RRC_SAMPLES + margin)
        if pi < debug_n:
            print(f"  [dbg #{pi}] STF@{d}  M={c_metrics[pi]:.3f}  cf0={coarse_cfo:+.0f}", flush=True)

        if ee - es < FRAME_RRC_SAMPLES:
            if pi < debug_n: print(f"    -> window too small", flush=True)
            continue

        syms = rrc_match(iq[es:ee])
        if len(syms) < PSS_LEN + RS_LEN:
            if pi < debug_n: print(f"    -> syms too short ({len(syms)})", flush=True)
            continue

        pk, ptm, pts, pval, pss_corr = pss_find(syms)
        if ptm < pss_ptm_thr or pts < pss_pts_thr:
            if pi < debug_n: print(f"    -> PSS fail  ptm={ptm:.1f} pts={pts:.1f}", flush=True)
            continue

        fs = pk - STF_LEN
        if fs < 0:
            if pi < debug_n: print(f"    -> fs<0  pk={pk}", flush=True)
            continue

        rp = fs + STF_LEN + PSS_LEN
        if rp + RS_LEN + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN > len(syms):
            if pi < debug_n: print(f"    -> frame too long  rp={rp} syms={len(syms)}", flush=True)
            continue

        chan = rs_estimate(syms, rp, coarse_cfo)
        if chan is None:
            if pi < debug_n: print(f"    -> RS=FAIL  pk={pk} fs={fs} rp={rp}", flush=True)
            continue

        if pi < debug_n:
            print(f"    -> OK  pk={pk} fs={fs} rp={rp}  "
                  f"ptm={ptm:.1f} pts={pts:.1f}  rs_corr={chan['rs_corr']:.1f}  "
                  f"cf0={chan['coarse_cfo']:+.0f} cf1={chan['fine_cfo']:+.0f}  "
                  f"|h|={abs(chan['h']):.3f} σ²={chan['sigma2']:.4f}", flush=True)

        # Header
        hdr_start = rp + RS_LEN
        hdr_bits, _ = bpsk_demod(syms, hdr_start, HEADER_LEN, chan)
        hdr_ok = crc16_check(bits_to_bytes(hdr_bits[:16]), b2i(hdr_bits[16:32]))

        # Payload + CRC
        pay_start = hdr_start + HEADER_LEN
        pay_bits, pay_y = bpsk_demod(syms, pay_start,
                                     PAYLOAD_LEN + PAYLOAD_CRC_LEN, chan)
        payload = pay_bits[:PAYLOAD_LEN]
        crc_ok = crc16_check(bits_to_bytes(payload), b2i(pay_bits[PAYLOAD_LEN:]))

        # BER vs 参考 (暂存 -1, 排序后重新计算)
        ber = -1.0

        # 质量指标 — 多口径 SNR (统一于 tools/snr_metrics.py)
        hmag = abs(chan['h'])
        # prefix SNR: |h|² / 独立底噪 (与 polar_loopback.py / loopback_test.py 一致)
        nf = noise_floor if noise_floor > 0 else 0.5
        snr_prefix = snr_symbol_domain(hmag, nf)
        # RS residual SNR: |h|² / sigma2 (与 receiver.py _estimate_snr 一致)
        snr_rs = snr_from_sigma2(hmag, chan['sigma2'])
        # EVM (via snr_metrics, 10*log10 clipped sigma2)
        evm_db = evm_from_sigma2(chan['sigma2'])

        # 定时偏 (PSS 二次插值)
        if pk > 0 and pk < len(pss_corr) - 1:
            c0, c1, c2 = pss_corr[pk-1], pss_corr[pk], pss_corr[pk+1]
            denom = 2 * (c0 - 2*c1 + c2)
            toff = (c0 - c2) / (denom + 1e-30) if abs(denom) > 1e-9 else 0.0
        else:
            toff = 0.0

        frames.append({
            'idx': len(frames),
            'frame_id': b2i(hdr_bits[:16]),
            'global_pos': d,
            'ptm': ptm, 'pts': pts,
            'hdr_ok': hdr_ok, 'crc_ok': crc_ok,
            'ber': ber,
            'hmag': hmag, 'snr': snr_prefix,  # canonical symbol-domain SNR
            'snr_prefix': snr_prefix,           # |h|² / noise_floor
            'snr_rs': snr_rs,                   # |h|² / sigma2
            'evm_db': evm_db,
            'coarse_cfo': chan['coarse_cfo'],
            'fine_cfo': chan['fine_cfo'],
            'total_cfo': chan['total_cfo'],
            'phase_est': chan['phase_est'],
            'sigma2': chan['sigma2'],
            'rs_corr': chan['rs_corr'],
            'timing_offset': toff,
            'constellation': pay_y,
            'payload_bits': payload,
        })

    # ── 阶段 4: 按全局位置排序 + 去重 (同一帧可能被多个候选命中) ──
    if frames:
        frames.sort(key=lambda f: f['global_pos'])
        # 去重: 相邻帧间距 < FRAME_RRC_SAMPLES/2 的只保留 CRC 更好的
        deduped = []
        for f in frames:
            if not deduped:
                deduped.append(f)
            elif f['global_pos'] - deduped[-1]['global_pos'] < FRAME_RRC_SAMPLES // 2:
                # 重复检测: 保留 CRC OK 的那个
                if f['crc_ok'] and not deduped[-1]['crc_ok']:
                    deduped[-1] = f
                elif f['crc_ok'] == deduped[-1]['crc_ok']:
                    if f['snr'] > deduped[-1]['snr']:
                        deduped[-1] = f
            else:
                deduped.append(f)

        # 重新编号
        for i, f in enumerate(deduped):
            f['idx'] = i
        frames = deduped

        # 排序后重新计算 BER (用 Header 中的 frame_id 对齐发射序号)
        if tx_bits is not None:
            for f in frames:
                ref_start = f['frame_id'] * PAYLOAD_LEN
                ref_end = ref_start + PAYLOAD_LEN
                if ref_end <= len(tx_bits):
                    ref = tx_bits[ref_start:ref_end]
                    errs = int(np.sum(f['payload_bits'] != ref))
                    f['ber'] = errs / PAYLOAD_LEN

        # 从帧间隙估计噪声方差 (零填充区 = 纯接收机底噪)
        if len(frames) >= 2:
            gap_iqs = []
            for i in range(len(frames) - 1):
                g0 = frames[i]['global_pos'] + FRAME_RRC_SAMPLES + 100  # 帧尾 + 裕量
                g1 = frames[i + 1]['global_pos'] - 100                   # 下一帧头 - 裕量
                if g1 - g0 > 500:
                    gap_iqs.append(iq[g0:g1])
            if gap_iqs:
                gap_all = np.concatenate(gap_iqs)
                noise_var = float(np.var(gap_all))
                # 更新每帧 gap SNR (secondary metric, stored separately)
                for f in frames:
                    sig_pow = float(np.mean(np.abs(f['constellation'])**2))
                    snr_linear = max(sig_pow - noise_var, 1e-30) / max(noise_var, 1e-30)
                    f['snr_gap'] = float(10 * np.log10(snr_linear))
                if verbose:
                    print(f"  [scan] gap noise sigma2={noise_var:.2e}  "
                          f"SNR_gap(mean)={np.mean([f['snr_gap'] for f in frames]):.1f}dB")

    if verbose:
        print(f"  [scan] frames detected: {len(frames)} "
              f"(deduped from {len(frames) if frames else 0})")

    return frames


# ======================================================================
# 报告
# ======================================================================

def print_report(frames, tx_bits):
    n = len(frames)
    crc_ok = sum(1 for f in frames if f['crc_ok'])
    hdr_ok = sum(1 for f in frames if f['hdr_ok'])
    bers = [f['ber'] for f in frames if f['ber'] >= 0]
    snrs = [f['snr'] for f in frames]
    cfo_totals = [f['total_cfo'] for f in frames]
    timing_offs = [f['timing_offset'] for f in frames]

    print(f"\n{'='*70}")
    print(f"  逐帧分析报告")
    print(f"{'='*70}")
    print(f"  检测帧数:       {n}")
    print(f"  HDR CRC 通过:   {hdr_ok}/{n} ({hdr_ok/max(n,1)*100:.1f}%)")
    print(f"  Payload CRC:    {crc_ok}/{n} ({crc_ok/max(n,1)*100:.1f}%)")
    if bers:
        print(f"  BER (均值):     {np.mean(bers)*100:.2f}%  "
              f"(max={np.max(bers)*100:.2f}%)")

    if snrs:
        print(f"\n  SNR (symbol, prefix):  mean={np.mean(snrs):.1f}  std={np.std(snrs):.1f}  "
              f"min={np.min(snrs):.1f}  max={np.max(snrs):.1f} dB")
    # Show additional SNR metrics if available
    snrs_rs = [f['snr_rs'] for f in frames if 'snr_rs' in f]
    snrs_gap = [f['snr_gap'] for f in frames if 'snr_gap' in f]
    if snrs_rs:
        print(f"  SNR (RS residual):     mean={np.mean(snrs_rs):.1f}  std={np.std(snrs_rs):.1f}  "
              f"min={np.min(snrs_rs):.1f}  max={np.max(snrs_rs):.1f} dB")
    if snrs_gap:
        print(f"  SNR (gap, IQ domain):  mean={np.mean(snrs_gap):.1f}  std={np.std(snrs_gap):.1f}  "
              f"min={np.min(snrs_gap):.1f}  max={np.max(snrs_gap):.1f} dB")
    if cfo_totals:
        print(f"  CFO:  mean={np.mean(cfo_totals):+.1f}  std={np.std(cfo_totals):.1f}  "
              f"min={np.min(cfo_totals):+.1f}  max={np.max(cfo_totals):+.1f} Hz")
    if timing_offs:
        print(f"  定时偏: mean={np.mean(timing_offs):+.3f}  "
              f"std={np.std(timing_offs):.3f} sym")

    evms = [f['evm_db'] for f in frames]
    if evms:
        print(f"  EVM:   mean={np.mean(evms):.1f}  std={np.std(evms):.1f} dB")

    # 帧间距
    if len(frames) >= 2:
        gaps = np.diff([f['global_pos'] for f in frames])
        print(f"  帧间距: mean={np.mean(gaps):.0f}  std={np.std(gaps):.0f}  "
              f"expected={FRAME_RRC_SAMPLES+5000}")

    print()

    issues = []
    if n == 0:
        issues.append("[FAIL] 未检测到任何帧 — 检查增益是否合适, SMA是否连接")
    elif crc_ok / max(n, 1) < 0.9:
        if snrs and np.mean(snrs) < 15:
            issues.append(f"[WARN] SNR 偏低 ({np.mean(snrs):.1f}dB)")
        if cfo_totals and np.std(cfo_totals) > 30:
            issues.append(f"[WARN] CFO 波动大 (σ={np.std(cfo_totals):.0f}Hz) → PSS定时跳变")
        if timing_offs and np.std(timing_offs) > 0.5:
            issues.append(f"[WARN] 定时偏抖动 (σ={np.std(timing_offs):.2f}sym)")

    if not issues:
        if crc_ok / max(n, 1) >= 0.95:
            issues.append("[OK] 同步链工作正常, CRC≥95%")
        else:
            issues.append(f"[INFO] CRC={crc_ok/max(n,1)*100:.0f}%")

    print("  【诊断】")
    for iss in issues:
        print(f"    {iss}")

    failed = [f for f in frames if not f['crc_ok']]
    if failed and len(failed) <= 20:
        print(f"\n  【失败帧】(共 {len(failed)} 帧)")
        print(f"  {'idx':>4s}  {'ptm':>6s}  {'pts':>5s}  {'HDR':>4s}  "
              f"{'SNR':>6s}  {'Δf0':>6s}  {'Δf1':>6s}  {'BER':>7s}")
        for f in failed[:10]:
            print(f"  {f['idx']:4d}  {f['ptm']:5.1f}  {f['pts']:4.1f}  "
                  f"{'OK' if f['hdr_ok'] else 'XX':>4s}  "
                  f"{f['snr']:5.1f}  {f['coarse_cfo']:+5.0f}  "
                  f"{f['fine_cfo']:+5.0f}  "
                  f"{f['ber']*100:6.2f}%")

    print(f"\n{'='*70}\n")
    return frames


# ======================================================================
# 参数扫描
# ======================================================================

def scan_thresholds(iq, tx_bits, stf_energy=None):
    print(f"\n{'='*70}")
    print(f"  参数扫描: PSS 门限 vs CRC 正确率")
    print(f"{'='*70}")
    print(f"  {'ptm':>6s}  {'pts':>6s}  {'det':>5s}  {'CRC':>5s}  {'rate':>7s}")
    print(f"  {'-'*40}")

    best_rate = 0; best_params = (3.5, 1.5)
    for ptm in [2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        for pts in [1.2, 1.5, 1.8, 2.0]:
            frames = analyze_frames_sequential(
                iq, tx_bits, ptm, pts, stf_energy=stf_energy, verbose=False)
            n = len(frames)
            crc = sum(1 for f in frames if f['crc_ok'])
            rate = crc / max(n, 1) * 100
            if rate > best_rate and n >= 10:
                best_rate = rate; best_params = (ptm, pts)
            print(f"  {ptm:5.1f}  {pts:5.1f}  {n:5d}  {crc:5d}  {rate:6.1f}%")

    print(f"\n  最优: ptm={best_params[0]:.1f}  pts={best_params[1]:.1f}  "
          f"(CRC={best_rate:.1f}%)")
    return best_params


# ======================================================================
# 绘图
# ======================================================================

def plot_analysis(iq, frames, save_prefix=''):
    import matplotlib
    matplotlib.use('TkAgg')
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt

    n = len(frames)
    if n == 0:
        print("无检测帧, 跳过绘图")
        return

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 4)

    # (a) 时域 IQ 波形 (定位首帧, 展示实际信号)
    ax = fig.add_subplot(gs[0, :])
    if frames:
        center = frames[0]['global_pos']
        win_start = max(0, center - 30000)
        win_end = min(len(iq), center + FRAME_RRC_SAMPLES * 10 + 30000)
    else:
        win_start, win_end = 0, min(len(iq), 200_000)
    seg = iq[win_start:win_end]
    ax.plot(seg.real, lw=0.2, color='steelblue', alpha=0.7, label='I')
    ax.plot(seg.imag, lw=0.2, color='darkred', alpha=0.5, label='Q')
    for f in frames:
        pos = f['global_pos'] - win_start
        if 0 <= pos < len(seg):
            ax.axvline(pos, color='g', lw=0.5, alpha=0.5)
            ax.axvline(pos + FRAME_RRC_SAMPLES, color='r', lw=0.3, alpha=0.3)
    ymax = np.max(np.abs(seg)) * 1.1
    ax.set_ylim(-ymax, ymax)
    ax.set_title(f'接收 IQ 波形 (样本 {win_start}-{win_end}, {len(seg)/1000:.0f}k)  '
                 f'| 绿线=帧头 红线=帧尾  |  {n} 帧检出')
    ax.set_ylabel('幅度'); ax.legend(fontsize=7, loc='upper right')

    # (b) STF 度量 (取帧密集的中段)
    ax = fig.add_subplot(gs[1, :2])
    if len(frames) > 0:
        mid_frame = frames[len(frames)//2]
        mid_pos = mid_frame['global_pos']
        seg_start = max(0, mid_pos - 25000)
        seg_end = min(len(iq), mid_pos + 25000)
        seg = iq[seg_start:seg_end]
        _, _, _, M = stf_detect(seg)
        if len(M) > 0:
            ax.plot(M, lw=0.3)
            ax.axhline(STF_THRESHOLD, color='r', ls='--', lw=0.8,
                       label=f'STF thr={STF_THRESHOLD}')
            ax.set_title(f'STF M(d) (帧#{mid_frame["idx"]} ±25k)')
            ax.set_xlabel('d (samples)'); ax.set_ylabel('M(d)'); ax.legend()

    # (c) 每帧 SNR
    ax = fig.add_subplot(gs[1, 2])
    snrs_arr = [f['snr'] for f in frames]
    status = ['g' if f['crc_ok'] else 'r' for f in frames]
    ax.bar(range(n), snrs_arr, color=status, width=0.8)
    ax.axhline(np.mean(snrs_arr), color='b', ls='--', lw=1,
               label=f'mean={np.mean(snrs_arr):.1f}dB')
    ax.set_title(f'SNR ({sum(1 for f in frames if f["crc_ok"])}/{n} CRC OK)')
    ax.set_xlabel('Frame'); ax.set_ylabel('SNR (dB)'); ax.legend()

    # (d) CFO
    ax = fig.add_subplot(gs[1, 3])
    cfos = [f['total_cfo'] for f in frames]
    ax.bar(range(n), cfos, color=status, width=0.8)
    ax.set_title(f'CFO (σ={np.std(cfos):.1f}Hz)')
    ax.set_xlabel('Frame'); ax.set_ylabel('CFO (Hz)')

    # (d) 星座图
    ok_f = [f for f in frames if f['crc_ok']]
    bad_f = [f for f in frames if not f['crc_ok']]
    for i, (f_list, label) in enumerate([(ok_f, 'CRC OK'), (bad_f, 'CRC Fail')]):
        ax = fig.add_subplot(gs[2, i*2:i*2+2])
        colors = plt.cm.tab10(np.linspace(0, 1, min(3, len(f_list))))
        for j, f in enumerate(f_list[:3]):
            ax.scatter(f['constellation'].real, f['constellation'].imag,
                      s=2, alpha=0.5, color=colors[j], label=f'f#{f["idx"]}')
        ax.axvline(-1, color='gray', ls=':', alpha=0.5)
        ax.axvline(0, color='gray', ls=':', alpha=0.5)
        ax.axvline(1, color='gray', ls=':', alpha=0.5)
        ax.axhline(0, color='gray', ls=':', alpha=0.5)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2, 2)
        ax.set_title(f'星座图 ({label}, n={len(f_list)})')
        ax.set_xlabel('I'); ax.set_ylabel('Q')
        if len(f_list) > 0: ax.legend(fontsize=7)

    # (e) BER
    ax = fig.add_subplot(gs[3, :2])
    bers = [f['ber'] for f in frames if f['ber'] >= 0]
    if bers:
        ax.hist([b*100 for b in bers], bins=30, edgecolor='k', alpha=0.7)
        ax.axvline(np.mean(bers)*100, color='r', lw=1.5,
                   label=f'mean={np.mean(bers)*100:.2f}%')
        ax.set_title(f'BER ({len(bers)} frames)')
        ax.set_xlabel('BER (%)'); ax.set_ylabel('N'); ax.legend()

    # (f) 定时偏
    ax = fig.add_subplot(gs[3, 2:])
    toff = [f['timing_offset'] for f in frames]
    ax.bar(range(n), toff, color=status, width=0.8)
    ax.set_title(f'定时偏 (σ={np.std(toff):.3f} sym)')
    ax.set_xlabel('Frame'); ax.set_ylabel('offset (sym)')

    fig.suptitle(f'环回IQ分析  |  {n} frames  |  '
                 f'CRC={sum(1 for f in frames if f["crc_ok"])}/{n}',
                 fontsize=11)
    fig.tight_layout()

    if save_prefix:
        out = f'{save_prefix}_analysis.png'
        fig.savefig(out, dpi=150)
        print(f"图表已保存 → {out}")
    plt.show()


def plot_compare(prefix_a, prefix_b, label_a='baseline', label_b='interference'):
    """双 capture 对比: 时域波形 + 星座图 (全部符号, 不分 CRC)."""
    import matplotlib
    matplotlib.use('TkAgg')
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt

    # ── 加载 ──
    iq_a = np.load(prefix_a + '_iq.npy')
    iq_b = np.load(prefix_b + '_iq.npy')
    tx_a = tx_b = None
    if os.path.isfile(prefix_a + '_bits.npy'):
        tx_a = np.load(prefix_a + '_bits.npy')
    if os.path.isfile(prefix_b + '_bits.npy'):
        tx_b = np.load(prefix_b + '_bits.npy')

    frames_a = analyze_frames_sequential(iq_a, tx_a, verbose=False)
    frames_b = analyze_frames_sequential(iq_b, tx_b, verbose=False)

    # ── 收集全部均衡符号 (不分 CRC) ──
    all_syms_a = np.concatenate([f['constellation'] for f in frames_a]) if frames_a else np.array([])
    all_syms_b = np.concatenate([f['constellation'] for f in frames_b]) if frames_b else np.array([])

    # ==================================================================
    # 图 1: 时域 IQ 波形对比 (上下, 定位首帧)
    # ==================================================================
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    def _get_seg(iq, frames, n_frames=10):
        if frames:
            center = frames[0]['global_pos']
            w0 = max(0, center - 30000)
            w1 = min(len(iq), center + FRAME_RRC_SAMPLES * n_frames + 30000)
        else:
            w0, w1 = 0, min(len(iq), 200_000)
        return w0, w1, iq[w0:w1]

    w0_a, w1_a, seg_a = _get_seg(iq_a, frames_a)
    w0_b, w1_b, seg_b = _get_seg(iq_b, frames_b)
    t_a = (np.arange(len(seg_a)) + w0_a) / SAMP_RATE * 1000
    t_b = (np.arange(len(seg_b)) + w0_b) / SAMP_RATE * 1000

    ax1.plot(t_a, seg_a.real, lw=0.25, color='steelblue', alpha=0.8, label='I')
    ax1.plot(t_a, seg_a.imag, lw=0.25, color='darkred', alpha=0.6, label='Q')
    ylim_a = np.max(np.abs(seg_a)) * 1.1
    ax1.set_ylabel('幅度'); ax1.set_title(f'{label_a}  —  接收 IQ 波形  ({len(seg_a)/1000:.0f}k 样本)')
    ax1.set_ylim(-ylim_a, ylim_a); ax1.legend(fontsize=7, loc='upper right')

    ax2.plot(t_b, seg_b.real, lw=0.25, color='steelblue', alpha=0.8, label='I')
    ax2.plot(t_b, seg_b.imag, lw=0.25, color='darkred', alpha=0.6, label='Q')
    ylim_b = np.max(np.abs(seg_b)) * 1.1
    ax2.set_ylabel('幅度'); ax2.set_xlabel('时间 (ms)')
    ax2.set_title(f'{label_b}  —  接收 IQ 波形')
    ax2.set_ylim(-ylim_b, ylim_b); ax2.legend(fontsize=7, loc='upper right')

    fig1.suptitle(f'时域波形对比: {label_a} vs {label_b}', fontsize=12, y=1.01)
    fig1.tight_layout()
    fig1.savefig(prefix_b + '_tdomain_compare.png', dpi=150, bbox_inches='tight')
    print(f"时域对比图 → {prefix_b}_tdomain_compare.png")

    # ==================================================================
    # 图 2: 星座图对比 (左右)
    # ==================================================================
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))

    for ax, syms, label, color in [(ax3, all_syms_a, label_a, 'steelblue'),
                                    (ax4, all_syms_b, label_b, 'darkorange')]:
        if len(syms) > 0:
            n_plot = min(len(syms), 5000)
            idx = np.random.choice(len(syms), n_plot, replace=False) if len(syms) > n_plot else np.arange(len(syms))
            ax.scatter(syms[idx].real, syms[idx].imag, s=3, alpha=0.4, color=color, edgecolors='none')
        ax.axvline(-1, color='gray', ls=':', alpha=0.4)
        ax.axvline(0, color='gray', ls=':', alpha=0.4)
        ax.axvline(1, color='gray', ls=':', alpha=0.4)
        ax.axhline(0, color='gray', ls=':', alpha=0.4)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_xlabel('I'); ax.set_ylabel('Q')
        ax.set_title(f'{label}  ({len(syms)} 符号)')
        ax.set_aspect('equal')

    fig2.suptitle(f'星座图对比: {label_a} vs {label_b}', fontsize=12)
    fig2.tight_layout()
    fig2.savefig(prefix_b + '_const_compare.png', dpi=150, bbox_inches='tight')
    print(f"星座对比图 → {prefix_b}_const_compare.png")

    plt.show()


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description='环回 IQ 逐帧精细分析')
    p.add_argument('prefix', help='loopback_capture.py 输出前缀 (--compare 模式时为首个前缀)')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--save-plot', default='')
    p.add_argument('--scan', action='store_true')
    p.add_argument('--compare', default='', help='对比目标前缀 (如 capture/int_sb10)')
    p.add_argument('--ptm', type=float, default=3.5)
    p.add_argument('--pts', type=float, default=1.5)
    p.add_argument('--snr-method', default='all',
                   choices=['prefix', 'rs', 'gap', 'all'],
                   help='SNR reporting method: prefix (match real-time), '
                        'rs (RS residual), gap (inter-frame gaps), all (default)')
    p.add_argument('--stf-energy', type=float, default=0,
                   help='STF能量门限 (0=使用phy_params默认)')
    p.add_argument('--debug', type=int, default=0,
                   help='打印前 N 帧的详细调试信息')
    args = p.parse_args()

    iq_path = args.prefix + '_iq.npy'
    bits_path = args.prefix + '_bits.npy'

    if not os.path.isfile(iq_path):
        print(f"错误: 找不到 {iq_path}"); return

    print(f"加载 IQ: {iq_path}")
    iq = np.load(iq_path)
    print(f"  {len(iq)} 样本 ({len(iq)/SAMP_RATE*1000:.0f}ms)")

    mag = np.abs(iq)
    peak = np.max(mag); rms = np.sqrt(np.mean(mag**2))
    clipped_pct = np.sum(mag > 0.98) / len(iq) * 100
    print(f"  幅度: peak={peak:.4f}  RMS={rms:.4f}  clipped={clipped_pct:.1f}%", end="")
    if clipped_pct > 5: print("  [WARN] 严重削峰!")
    elif peak > 0.95: print("  [WARN] 有削峰")
    else: print("  [OK]")

    tx_bits = None
    if os.path.isfile(bits_path):
        tx_bits = np.load(bits_path)
        print(f"加载参考比特: {bits_path}")
        print(f"  {len(tx_bits)} bits ({len(tx_bits)//PAYLOAD_LEN} 帧)")

    stf_en = args.stf_energy if args.stf_energy > 0 else None

    if args.compare:
        plot_compare(args.prefix, args.compare,
                     label_a=os.path.basename(args.prefix),
                     label_b=os.path.basename(args.compare))
        return

    if args.scan:
        best = scan_thresholds(iq, tx_bits, stf_energy=stf_en)
        print(f"\n建议: --ptm {best[0]:.1f} --pts {best[1]:.1f}")

    frames = analyze_frames_sequential(iq, tx_bits, args.ptm, args.pts,
                                       stf_energy=stf_en, debug_n=args.debug)
    print_report(frames, tx_bits)

    if args.plot or args.save_plot:
        plot_analysis(iq, frames, args.save_plot or args.prefix)


if __name__ == '__main__':
    main()
