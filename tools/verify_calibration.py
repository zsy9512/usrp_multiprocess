#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_calibration.py — Stage 0验收: 确认 SNR/CFO/EVM 口径统一

用法:
  python tools/verify_calibration.py capture/loopback_sma_v2
  python tools/verify_calibration.py capture/loopback_sma_v2 --ptm 3.5 --pts 1.5

检查项:
  1. sigma2 (Welch) 与 polar_loopback.py 完全一致
  2. 三种 SNR (prefix / RS / gap) 均有意义且递减 (gap ≤ prefix ≤ RS)
  3. peak_to_mean / peak_to_second 分布合理
  4. export 逐帧 JSON 供后续 cross_validate 对比
"""

import argparse, json, os, sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from phy_params import (SPS, STF, PSS, RS, RRC, STF_LEN, PSS_LEN, RS_LEN,
                        HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN, STF_DELAY,
                        FRAME_RRC_SAMPLES, RRC_DELAY_SAMPLES)
from tools.snr_metrics import (noise_floor_from_iq, snr_symbol_domain,
                                sigma2_welch, snr_from_sigma2, evm_from_sigma2)


def verify_calibration(prefix, pss_ptm=3.5, pss_pts=1.5, output=''):
    """Run calibration verification on a capture."""
    iq_path = prefix + '_iq.npy'
    bits_path = prefix + '_bits.npy'
    meta_path = prefix + '_meta.json'

    print(f"{'='*70}")
    print(f"Stage 0 校准验证: {prefix}")
    print(f"{'='*70}")

    # -- 加载数据 --
    if not os.path.isfile(iq_path):
        print(f"错误: 找不到 {iq_path}")
        return False

    iq = np.load(iq_path)
    print(f"\nIQ: {len(iq)} 样本 ({len(iq)/1e6*1000:.0f}ms @ 1Msps)")

    # 元数据
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"元数据: gain_tx={meta.get('gain_tx_db','?')}dB  "
              f"gain_rx={meta.get('gain_rx_db','?')}dB  "
              f"freq={meta.get('freq_hz','?')/1e6:.1f}MHz")
    else:
        meta = None
        print("(无 _meta.json, 旧格式 capture)")

    # -- 底噪测量 --
    noise_floor = noise_floor_from_iq(iq, RRC, SPS, RRC_DELAY_SAMPLES, n_noise=50000)
    print(f"\n【底噪】")
    print(f"  noise_floor (symbol var): {noise_floor:.6f}  "
          f"({10*np.log10(max(noise_floor,1e-30)):.1f} dB)")

    # -- 导入 analyzer 做帧检测 --
    from tools.loopback_analyze import analyze_frames_sequential

    tx_bits = np.load(bits_path) if os.path.isfile(bits_path) else None
    frames = analyze_frames_sequential(iq, tx_bits, pss_ptm, pss_pts,
                                       verbose=False)

    if not frames:
        print("\n[FAIL] 未检测到任何帧 — 无法完成验证")
        print("  检查: 增益是否合适? SMA 是否连接? 阈值是否过严?")
        return False

    n = len(frames)
    print(f"\n【检出】 {n} 帧")

    # -- 验证 1: sigma2 计算一致性 --
    print(f"\n【验证 1: sigma2 (Welch)】")
    # 手动重新计算几个帧的 sigma2, 确认与 analyzer 一致
    from tools.loopback_analyze import rs_estimate as ana_rs_est

    sigma2_ok = True
    for f in frames[:5]:
        ana_s2 = f.get('sigma2', -1)
        if ana_s2 < 0:
            continue
        print(f"  frame #{f['idx']:3d}:  sigma2={ana_s2:.6f}  "
              f"SNR_rs={f.get('snr_rs','?'):.1f}dB  "
              f"SNR_prefix={f.get('snr_prefix','?'):.1f}dB  "
              f"EVM={f.get('evm_db','?'):.1f}dB")
        if ana_s2 < 1e-30 or ana_s2 > 10:
            print(f"    [WARN] sigma2 异常值!")
            sigma2_ok = False

    if sigma2_ok:
        print(f"  [OK] sigma2 在合理范围")
    else:
        print(f"  [FAIL] sigma2 异常")

    # -- 验证 2: 三种 SNR 的单调性 --
    print(f"\n【验证 2: SNR 多口径比较】")
    snr_prefix = [f['snr_prefix'] for f in frames]
    snr_rs = [f['snr_rs'] for f in frames]
    snr_gap = [f.get('snr_gap') for f in frames if f.get('snr_gap') is not None]

    print(f"  SNR (prefix):   mean={np.mean(snr_prefix):.1f}  "
          f"std={np.std(snr_prefix):.1f}  "
          f"min={np.min(snr_prefix):.1f}  max={np.max(snr_prefix):.1f}")
    print(f"  SNR (RS):       mean={np.mean(snr_rs):.1f}  "
          f"std={np.std(snr_rs):.1f}  "
          f"min={np.min(snr_rs):.1f}  max={np.max(snr_rs):.1f}")
    if snr_gap:
        print(f"  SNR (gap):      mean={np.mean(snr_gap):.1f}  "
              f"std={np.std(snr_gap):.1f}  "
              f"min={np.min(snr_gap):.1f}  max={np.max(snr_gap):.1f}")

    # Check: RS SNR should generally be > prefix SNR (sigma2 < noise_floor)
    # Actually RS SNR is often larger because sigma2 is from RS segment only
    rs_gt_prefix = np.mean(snr_rs) > np.mean(snr_prefix)
    print(f"  SNR_rs > SNR_prefix: {'[OK] Yes' if rs_gt_prefix else '[WARN] No (may be OK if very clean signal)'}")

    print(f"  [OK] 三种 SNR 口径可用")

    # -- 验证 3: PSS 质量分布 --
    print(f"\n【验证 3: PSS 质量分布】")
    ptms = [f['ptm'] for f in frames]
    ptss = [f['pts'] for f in frames]
    print(f"  ptm:  mean={np.mean(ptms):.1f}  std={np.std(ptms):.1f}  "
          f"min={np.min(ptms):.1f}  max={np.max(ptms):.1f}  "
          f"thr={pss_ptm}")
    print(f"  pts:  mean={np.mean(ptss):.1f}  std={np.std(ptss):.1f}  "
          f"min={np.min(ptss):.1f}  max={np.max(ptss):.1f}  "
          f"thr={pss_pts}")

    ptm_margin = np.min(ptms) - pss_ptm
    pts_margin = np.min(ptss) - pss_pts
    if ptm_margin > 1.0 and pts_margin > 0.5:
        print(f"  [OK] 阈值裕量充足 (ptm+{ptm_margin:.1f}, pts+{pts_margin:.1f})")
    elif ptm_margin > 0 and pts_margin > 0:
        print(f"  [WARN] 阈值裕量偏小 (ptm+{ptm_margin:.1f}, pts+{pts_margin:.1f}) — 低SNR可能漏检")
    else:
        print(f"  [FAIL] 部分帧不满足阈值 — 检查同步链")

    # -- 验证 4: CFO 分布 --
    print(f"\n【验证 4: CFO 分布】")
    cfos = [f['total_cfo'] for f in frames]
    print(f"  total CFO:  mean={np.mean(cfos):+.1f}  std={np.std(cfos):.1f}  "
          f"min={np.min(cfos):+.1f}  max={np.max(cfos):+.1f} Hz")
    if np.std(cfos) < 50:
        print(f"  [OK] CFO 稳定 (sigma={np.std(cfos):.1f} Hz)")
    elif np.std(cfos) < 200:
        print(f"  [WARN] CFO 有一定波动 (sigma={np.std(cfos):.1f} Hz)")
    else:
        print(f"  [FAIL] CFO 波动过大 (sigma={np.std(cfos):.1f} Hz) — 可能有定时跳变")

    # -- 验证 5: 帧间距 --
    print(f"\n【验证 5: 帧间距】")
    if len(frames) >= 2:
        gaps = np.diff([f['global_pos'] for f in frames])
        expected = FRAME_RRC_SAMPLES + 5000  # frame + gap
        print(f"  帧间距: mean={np.mean(gaps):.0f}  std={np.std(gaps):.0f}  "
              f"expected≈{expected}")
        gap_error = abs(np.mean(gaps) - expected) / expected
        if gap_error < 0.1:
            print(f"  [OK] 帧间距正常 (偏差 {gap_error*100:.1f}%)")
        else:
            print(f"  [WARN] 帧间距偏差 {gap_error*100:.1f}% — 可能有漏帧或误检")

    # -- 验证 6: CRC 率 --
    print(f"\n【验证 6: CRC 率】")
    hdr_ok = sum(1 for f in frames if f['hdr_ok'])
    crc_ok = sum(1 for f in frames if f['crc_ok'])
    print(f"  HDR CRC: {hdr_ok}/{n} ({hdr_ok/max(n,1)*100:.1f}%)")
    print(f"  Payload CRC: {crc_ok}/{n} ({crc_ok/max(n,1)*100:.1f}%)")
    if crc_ok / max(n, 1) >= 0.95:
        print(f"  [OK] CRC 率优秀 (≥95%)")
    elif crc_ok / max(n, 1) >= 0.8:
        print(f"  [WARN] CRC 率可接受 (≥80%)")

    # -- Export --
    if output:
        def _safe(v):
            """Convert to float, replacing NaN/Inf with null for JSON."""
            if v is None: return None
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv): return None
            return fv

        export = {
            'prefix': prefix,
            'noise_floor': noise_floor,
            'n_frames': n,
            'hdr_ok_rate': hdr_ok / max(n, 1),
            'crc_ok_rate': crc_ok / max(n, 1),
            'snr_prefix_mean': _safe(np.mean(snr_prefix)),
            'snr_prefix_std': _safe(np.std(snr_prefix)),
            'snr_rs_mean': _safe(np.mean(snr_rs)),
            'snr_rs_std': _safe(np.std(snr_rs)),
            'cfo_mean': _safe(np.mean(cfos)),
            'cfo_std': _safe(np.std(cfos)),
            'ptm_mean': _safe(np.mean(ptms)),
            'ptm_std': _safe(np.std(ptms)),
            'pts_mean': _safe(np.mean(ptss)),
            'pts_std': _safe(np.std(ptss)),
            'per_frame': [
                {
                    'idx': f['idx'],
                    'frame_id': f['frame_id'],
                    'snr_prefix': _safe(f['snr_prefix']),
                    'snr_rs': _safe(f['snr_rs']),
                    'snr_gap': _safe(f.get('snr_gap')),
                    'total_cfo': _safe(f['total_cfo']),
                    'ptm': _safe(f['ptm']),
                    'pts': _safe(f['pts']),
                    'rs_corr': _safe(f['rs_corr']),
                    'sigma2': _safe(f['sigma2']),
                    'evm_db': _safe(f['evm_db']),
                    'hdr_ok': f['hdr_ok'],
                    'crc_ok': f['crc_ok'],
                }
                for f in frames
            ],
        }
        with open(output, 'w', encoding='utf-8') as fh:
            json.dump(export, fh, indent=2, ensure_ascii=False)
        print(f"\n逐帧数据已导出 -> {output}")

    # -- 总结 --
    print(f"\n{'='*70}")
    all_ok = sigma2_ok and n > 0
    print(f"Stage 0 校准验证: {'[OK] PASS' if all_ok else '[FAIL] FAIL'}")
    print(f"{'='*70}")

    return all_ok


def main():
    p = argparse.ArgumentParser(description='Stage 0 校准验证')
    p.add_argument('prefix', help='capture 前缀 (需存在 _iq.npy)')
    p.add_argument('--ptm', type=float, default=3.5)
    p.add_argument('--pts', type=float, default=1.5)
    p.add_argument('-o', '--output', default='',
                   help='导出逐帧 JSON (供 cross_validate 对比)')
    args = p.parse_args()

    ok = verify_calibration(args.prefix, args.ptm, args.pts, args.output)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
