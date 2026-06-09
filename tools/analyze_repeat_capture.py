#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_repeat_capture.py — 分析 5x 重复帧 capture (any-of-N 检出 + frame_id 去重)

对应 loopback_capture.py 的 REPEAT=5 发送模式:
  每组 5 帧完全相同 (frame_id 相同), 间隔 3ms, 组间 5ms.
  接收端对每组 5 帧独立跑 STF->PSS->RS 同步,
  任意一帧通过 -> 组检出成功, 用 frame_id 去重.

用法:
  python tools/analyze_repeat_capture.py capture/low_snr_v2
  python tools/analyze_repeat_capture.py capture/low_snr_v2 --gain 15
  python tools/analyze_repeat_capture.py capture/low_snr_v2 --all -o report.json
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
                        FRAME_RRC_SAMPLES)
from sender import build_frame, rrc_filter

SAMP_RATE = 1e6
REPEAT = 5
GAP_REPEAT_IQ = int(0.003 * SAMP_RATE)  # 3000
GAP_GROUP_IQ  = int(0.005 * SAMP_RATE)  # 5000

# 复用 sync_sweep 的同步函数
from tools.sync_sweep import (make_stf, make_rs,
    stf_detect_custom, pss_correlate_custom, rs_estimate_custom)


def detect_one_frame(syms, stf_syms, pss_syms, rs_syms,
                     pss_ptm=2.5, pss_pts=1.0, stf_energy=0.01):
    """对一段符号序列跑 STF->PSS->RS, 返回 dict 或 None."""
    # STF
    stf_len = len(stf_syms)
    # 不在此做 STF 检测 — 直接从已知 IQ 位置提取符号, 只做 PSS+RS
    # (帧起始位置由 TX 时序推算, 不需要 STF 扫描)

    if len(syms) < len(pss_syms) + len(rs_syms):
        return None

    pk, ptm, pts, pv = pss_correlate_custom(syms, pss_syms)
    if ptm < pss_ptm or pts < pss_pts:
        return None

    fs = pk - stf_len
    if fs < 0:
        return None

    rp = fs + stf_len + len(pss_syms)
    if rp + len(rs_syms) > len(syms):
        return None

    chan = rs_estimate_custom(syms, rp, rs_syms)
    if chan is None:
        return None

    return {
        'ptm': ptm, 'pts': pts,
        'total_cfo': chan['total_cfo'],
        'h_mag': abs(chan['h']),
        'sigma2': chan['sigma2'],
        'rs_corr': chan['rs_corr'],
    }


def analyze_capture(prefix, num_frames=40, pss_ptm=2.5, pss_pts=1.0):
    """分析单个 capture 的 5x 重复帧检出率.

    Args:
        prefix:     capture 前缀 (如 capture/low_snr_v2/snr_gain015_r0)
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

    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)

    # 帧结构参数
    stf_syms = make_stf(4)
    rs_syms = make_rs(32)
    pss_syms = PSS_REF
    total_sym = (len(stf_syms) + len(pss_syms) + len(rs_syms)
                 + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN + GUARD_SYMBOLS)
    frame_iq_len = total_sym * SPS + len(RRC) - 1
    RRC_DEL = (len(RRC) - 1) // 2

    # TX 时序: [pad] [F0×5 + gaps] [GAP_GROUP] [F1×5 + gaps] ...
    # RX 开始采集早于 TX (~1s), 第一帧起始约在 IQ 开头后 50000 样本
    # 用全帧互相关定位第一帧 (与 channel_measure.py 相同)
    rng = np.random.RandomState(42)
    tx_iq_ref, _, _ = __reconstruct_first_frame(rng, stf_syms, pss_syms, rs_syms)

    # 全帧互相关找第一帧
    tx_rev = np.conj(tx_iq_ref[::-1])
    corr = np.abs(np.convolve(iq[:min(200000, n_total)], tx_rev, mode='valid'))
    first_offset = int(np.argmax(corr))
    print(f"    第一帧 @ IQ[{first_offset}], corr_peak={corr[first_offset]:.1f}")

    groups_ok = 0
    hits_per_group = []
    total_groups = 0

    for gi in range(num_frames):
        group_offset = first_offset + gi * (
            REPEAT * frame_iq_len + (REPEAT - 1) * GAP_REPEAT_IQ + GAP_GROUP_IQ)
        n_hits = 0

        for ri in range(REPEAT):
            offset = group_offset + ri * (frame_iq_len + GAP_REPEAT_IQ)
            if offset + frame_iq_len > n_total:
                break

            # 提取 + RRC 匹配
            margin = 400
            es = max(0, offset - margin)
            ee = min(n_total, offset + frame_iq_len + margin)
            chunk = iq[es:ee]
            syms = np.convolve(chunk, RRC[::-1], mode='full')[RRC_DEL::SPS].astype(np.complex64)

            if len(syms) < len(pss_syms) + len(rs_syms):
                continue

            result = detect_one_frame(syms, stf_syms, pss_syms, rs_syms,
                                      pss_ptm, pss_pts)
            if result is not None:
                n_hits += 1

        if n_hits > 0:
            groups_ok += 1
        hits_per_group.append(n_hits)
        total_groups += 1

    return {
        'total_groups': total_groups,
        'groups_ok': groups_ok,
        'detection_rate': groups_ok / max(total_groups, 1),
        'mean_hits': float(np.mean(hits_per_group)) if hits_per_group else 0,
        'hits_distribution': {i: hits_per_group.count(i) for i in range(REPEAT + 1)},
    }


def __reconstruct_first_frame(rng, stf_syms, pss_syms, rs_syms):
    """重建 frame_id=0 的 TX IQ (用于全帧互相关定位)."""
    from tools.sync_sweep import build_custom_frame
    raw = rng.randint(0, 2, PAYLOAD_LEN).astype(np.int64)
    frame_syms = build_custom_frame(raw, 0, stf_syms, pss_syms, rs_syms)
    tx_iq = rrc_filter(frame_syms, RRC, SPS)
    return tx_iq.astype(np.complex64), raw, frame_syms


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
        gain_str = tag.split('_')[1].replace('gain', '')
        print(f"\n-- {tag} (gain={int(gain_str)} dB) --")
        r = analyze_capture(pfx, num_frames=args.num_frames,
                            pss_ptm=args.pss_ptm, pss_pts=args.pss_pts)
        if 'error' in r:
            print(f"  错误: {r['error']}")
            continue

        print(f"  组检出: {r['groups_ok']}/{r['total_groups']} "
              f"({r['detection_rate']*100:.1f}%)")
        print(f"  平均命中: {r['mean_hits']:.1f}/{REPEAT} 帧/组")
        hits_str = ' '.join(f'{k}:{v}' for k, v in sorted(r['hits_distribution'].items()))
        print(f"  命中分布: {hits_str}")
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
