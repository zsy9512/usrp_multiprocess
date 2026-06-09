#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
polar_offline_decode.py — 离线极化译码 + BER (从 capture 读取)

帧结构 (v2):
  STF(128) + PSS(64) + RS(64) + Header(32) + Payload(256) + CRC(16) + Guard(32) = 592 sym

工作流:
  1. 全帧互相关定位第一帧 → 组时序推算
  2. 逐帧 STF 扫描 → PSS → RS (长 64 符号, 处理增益 3dB 高于 32)
  3. RS 信道估计 → LLR 软解调 Payload
  4. Hard Inverse Polar 变换 → 信息比特 → BER vs TX info

用法:
  python tools/polar_offline_decode.py capture/ebn0_tx60/snr_gain025_r0
  python tools/polar_offline_decode.py capture/ebn0_tx60 --gain 25
"""
import argparse, json, math, os, sys, glob, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from phy_params import (SPS, PSS as PSS_REF, RRC,
                        PSS_LEN, HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN,
                        GUARD_SYMBOLS, RRC_DELAY_SAMPLES)
from tools.sync_sweep import (pss_correlate_custom, rs_estimate_custom,
                               make_stf, make_rs)

SAMP_RATE = 1e6
TS_SYM = SPS / SAMP_RATE
N_POLAR = 256
K_POLAR = 128
REPEAT = 5

FROZEN_PATH = os.path.join(BASE, 'deploy', 'matrices', 'A.npy')
FROZEN_MASK = np.load(FROZEN_PATH).squeeze()
LLR_CLIP = 20.0


def _polar_encode(u):
    cw = u.copy().ravel()
    for stage in range(1, int(math.log2(N_POLAR)) + 1):
        sep = N_POLAR // (1 << stage)
        for j in range(N_POLAR):
            if (j // sep) % 2 == 0:
                cw[j] = (cw[j] + cw[j + sep]) % 2
    return cw


def _polar_hard_inverse(llr):
    hard_bits = (llr < 0).astype(np.int64)
    u_hat = _polar_encode(hard_bits)
    return u_hat[FROZEN_MASK.astype(bool)]


def _local_peak(corr, center, radius=16):
    if len(corr) == 0:
        return -1, 0.0
    lo, hi = max(0, int(center)-radius), min(len(corr), int(center)+radius+1)
    if hi <= lo:
        return -1, 0.0
    rel = int(np.argmax(corr[lo:hi]))
    return lo + rel, float(corr[lo+rel])


def _find_first_repeat(corr, stride, n_repeats=5):
    if len(corr) == 0:
        return -1, {}
    best_pos = int(np.argmax(corr))
    best_val = float(corr[best_pos])
    pos = best_pos
    ri = 0
    for _ in range(n_repeats - 1):
        expected = pos - stride
        if expected < 0:
            break
        prev_pos, prev_val = _local_peak(corr, expected)
        if prev_pos < 0 or prev_val < best_val * 0.35:
            break
        pos = prev_pos
        ri += 1
    return int(pos), {'peak': best_val, 'repeat_index': ri}


def _bpsk_demod_llr(symbols, data_start, data_len, h, total_cfo, sigma2):
    if data_start + data_len > len(symbols):
        return np.zeros(data_len, dtype=np.float32)
    seg = symbols[data_start:data_start + data_len]
    n = np.arange(data_len)
    cfo_comp = np.exp(-1j * 2 * np.pi * total_cfo * (data_start + n) * TS_SYM)
    y = seg * cfo_comp
    if abs(h) > 1e-30:
        y = y / h
    sigma2_out = max(float(sigma2), 1e-6)
    llr = np.clip(4.0 * y.real / sigma2_out, -LLR_CLIP, LLR_CLIP).astype(np.float32)
    return llr


def decode_capture(prefix, num_frames=100, sgnn_model=None, sgnn_graph=None,
                   sgnn_device='cpu'):
    """离线解码一个 capture.

    Returns dict with per_frame BER and summary.
    """
    iq_path = prefix + '_iq.npy'
    meta_path = prefix + '_meta.json'
    info_path = prefix + '_info.npy'

    if not os.path.isfile(iq_path):
        return {'error': f'no file: {iq_path}'}

    iq = np.load(iq_path)
    n_total = len(iq)
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)

    # 帧结构参数 (从 meta 或默认 v2)
    stf_reps = meta.get('stf_syms', 128) // 16
    rs_len = meta.get('rs_syms', 64)
    gap_group_ms = meta.get('frame_gap_ms', 30.0)
    gap_repeat_ms = meta.get('gap_repeat_ms', 5.0)
    gap_group_iq = int(gap_group_ms * SAMP_RATE / 1000)
    gap_repeat_iq = int(gap_repeat_ms * SAMP_RATE / 1000)

    stf_syms = make_stf(stf_reps)
    rs_syms = make_rs(rs_len)
    pss_syms = PSS_REF
    total_sym = (len(stf_syms) + len(pss_syms) + len(rs_syms)
                 + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN + GUARD_SYMBOLS)
    frame_iq_len = total_sym * SPS + len(RRC) - 1

    # 加载 TX 信息比特 + TX 编码比特
    tx_info = None
    tx_coded_bits = None
    if os.path.isfile(info_path):
        tx_info = np.load(info_path).reshape(-1, K_POLAR)
        num_frames = min(num_frames, len(tx_info))
    bits_path_iq = prefix + '_bits.npy'
    if os.path.isfile(bits_path_iq):
        tx_coded_bits = np.load(bits_path_iq).reshape(-1, N_POLAR)
        num_frames = min(num_frames, len(tx_coded_bits))

    # 全帧互相关找 frame0 (用第一帧的编码比特重建)
    rng = np.random.RandomState(42)
    info0 = (rng.rand(K_POLAR) < 0.5).astype(np.int64)
    u0 = np.zeros(N_POLAR, dtype=np.int64)
    u0[FROZEN_MASK.astype(bool)] = info0
    coded0 = _polar_encode(u0)

    # 重建 frame0 TX 波形
    def _reconstruct_frame_v2(info, fid):
        u = np.zeros(N_POLAR, dtype=np.int64)
        u[FROZEN_MASK.astype(bool)] = info
        coded = _polar_encode(u)
        from tools.loopback_capture_v2 import build_frame_v2, rrc_pulse
        frame_syms = build_frame_v2(coded, fid, stf_syms, pss_syms, rs_syms)
        return rrc_pulse(frame_syms, RRC, SPS).astype(np.complex64)

    tx_ref0 = _reconstruct_frame_v2(info0, 0)

    tx_rev = np.conj(tx_ref0[::-1])
    corr = np.abs(np.convolve(iq[:min(3000000, n_total)], tx_rev, mode='valid'))
    repeat_stride = frame_iq_len + gap_repeat_iq
    first_offset, first_info = _find_first_repeat(corr, repeat_stride)
    if first_offset < 0:
        return {'error': 'first frame not found'}
    print(f"  第一帧 @ IQ[{first_offset}], corr_peak={first_info['peak']:.1f} "
          f"(回溯 {first_info['repeat_index']} repeats)")

    per_frame = []
    ber_errs, ber_total = 0, 0
    bpsk_errs, bpsk_total = 0, 0
    sgnn_errs, sgnn_total = 0, 0
    mean_llr_vals = []
    first_decoded = True
    rng2 = np.random.RandomState(42)
    for gi in range(num_frames):
        group_offset = first_offset + gi * (
            REPEAT * frame_iq_len + (REPEAT - 1) * gap_repeat_iq + gap_group_iq)
        decoded = False
        for ri in range(REPEAT):
            offset = group_offset + ri * repeat_stride
            if offset + frame_iq_len > n_total:
                break
            margin = 400
            es = max(0, offset - margin)
            ee = min(n_total, offset + frame_iq_len + margin)
            chunk = iq[es:ee]
            syms = np.convolve(chunk, RRC[::-1], mode='full')[RRC_DELAY_SAMPLES::SPS].astype(np.complex64)

            if len(syms) < len(pss_syms) + len(rs_syms):
                continue

            # PSS 位置约束: 只在预期位置附近搜索 (避免 STF 假峰)
            frame_sym_est = int((margin - RRC_DELAY_SAMPLES) / SPS)
            expected_pss = frame_sym_est + len(stf_syms)
            search_win = 40  # ±40 符号
            pss_lo = max(0, expected_pss - search_win)
            pss_hi = min(len(syms) - len(pss_syms), expected_pss + search_win)
            if pss_hi <= pss_lo:
                continue
            pss_rev = np.conj(pss_syms[::-1])
            pss_corr = np.abs(np.convolve(syms[pss_lo:pss_hi + len(pss_syms)],
                                          pss_rev, mode='valid'))
            pk_local = int(np.argmax(pss_corr))
            pk = pss_lo + pk_local
            pval = float(pss_corr[pk_local])
            ptm = pval / (np.mean(pss_corr) + 1e-30)
            sv = np.sort(pss_corr)[::-1]
            pts = ptm
            for v in sv[1:]:
                for idx in np.where(np.isclose(pss_corr, v))[0]:
                    if abs(idx - pk_local) > len(pss_syms) // 2:
                        pts = pval / (v + 1e-30)
                        break
                if pts != ptm:
                    break
            if ptm < 2.5 or pts < 1.0:
                continue
            fs = pk - len(stf_syms)
            if fs < 0:
                continue
            rp = fs + len(stf_syms) + len(pss_syms)
            if rp + len(rs_syms) + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN > len(syms):
                continue

            chan = rs_estimate_custom(syms, rp, rs_syms)
            used_simple_chan = False
            if chan is None:
                # 自适应重试: RS 自相关算 |h|, CFO 写死 (真实 < 5 Hz, 忽略)
                rs_seg = syms[rp:rp + len(rs_syms)]
                rs_tone = rs_seg * np.conj(rs_syms)
                rs_corr_raw = float(np.abs(np.sum(rs_tone)))
                h_est = rs_corr_raw / len(rs_syms)
                if h_est < 1e-6:
                    continue
                used_simple_chan = True
                total_cfo = 0.0  # 真实 CFO < 5 Hz, 256 符号内相位漂移 < 0.02 rad
                # 信道估计 (无 CFO 补偿)
                h_c = np.mean(syms[rp:rp + rs_len] * np.conj(rs_syms))
                if abs(h_c) > 1e-6:
                    h = h_c
                else:
                    h = complex(h_est, 0)
                # 残差噪声
                noise = syms[rp:rp + rs_len] / h - rs_syms
                sigma2 = max(float(np.sum(np.abs(noise)**2) / max(rs_len - 1, 1)), 1e-6)
            else:
                total_cfo = chan['total_cfo']
                h = chan['h']
                sigma2 = chan['sigma2']

            pay_start = rp + len(rs_syms) + HEADER_LEN
            if pay_start + N_POLAR > len(syms):
                continue

            llr = _bpsk_demod_llr(syms, pay_start, N_POLAR, h, total_cfo, sigma2)
            info_hat = _polar_hard_inverse(llr)
            decoded = True

            # SGNN 译码
            info_sgnn = None
            if sgnn_model is not None and sgnn_graph is not None:
                import torch
                v = np.zeros((1, sgnn_graph['N_hat'], 1), dtype=np.float32)
                v[0, -N_POLAR:, 0] = llr
                vt = torch.from_numpy(v).to(sgnn_device)
                ft = sgnn_graph['template_f'].unsqueeze(0)
                with torch.no_grad():
                    out = sgnn_model(vt, ft,
                                     sgnn_graph['edge_index'],
                                     sgnn_graph['edge_index_rev'])
                sgnn_bits = (out[-1][0, -N_POLAR:, 0].cpu().numpy() < 0).astype(np.int64)
                info_sgnn = _polar_encode(sgnn_bits)[FROZEN_MASK.astype(bool)]

            # BPSK 硬判 BER + SGNN BER
            bpsk_hard = (llr < 0).astype(np.int64)
            mean_llr_vals.append(float(np.mean(np.abs(llr))))
            if tx_coded_bits is not None and gi < len(tx_coded_bits):
                ref_coded = tx_coded_bits[gi].astype(np.int64)
                bpsk_errs += int(np.sum(bpsk_hard != ref_coded))
                bpsk_total += N_POLAR
            if info_sgnn is not None and tx_info is not None and gi < len(tx_info):
                ref_info = tx_info[gi].astype(np.int64)
                sgnn_errs += int(np.sum(info_sgnn != ref_info))
                sgnn_total += K_POLAR

            # 第一帧诊断
            if first_decoded:
                first_decoded = False
                print(f"  [诊断 帧{gi}] PSS: pk={pk} fs={fs} rp={rp} pay_start={pay_start} "
                      f"ptm={ptm:.1f} pts={pts:.1f}")
                print(f"  [诊断 帧{gi}] RS: |h|={abs(h):.6f} angle(h)={np.angle(h):+.3f}rad "
                      f"sigma2={sigma2:.2e} cfo={total_cfo:+.1f}Hz "
                      f"{'[简化chan!]' if used_simple_chan else ''}")
                # 直接 BPSK 硬判 (不用 LLR)
                y_raw = syms[pay_start:pay_start + N_POLAR]
                n_arr = np.arange(N_POLAR)
                cfo_comp = np.exp(-1j * 2 * np.pi * total_cfo * (pay_start + n_arr) * TS_SYM)
                y_eq = y_raw * cfo_comp
                # 不除以 h, 只看原始实部符号
                raw_bits = (y_eq.real < 0).astype(np.int64)
                if tx_coded_bits is not None and gi < len(tx_coded_bits):
                    ref10 = tx_coded_bits[gi][:10].astype(np.int64)
                    print(f"  [诊断 帧{gi}] TX前10: {ref10}")
                    print(f"  [诊断 帧{gi}] BPSK前10(÷h): {bpsk_hard[:10]}")
                    print(f"  [诊断 帧{gi}] BPSK前10(裸): {raw_bits[:10]}")
            break

        frame_result = {'frame_id': gi, 'decoded': decoded}
        if decoded and tx_info is not None and gi < len(tx_info):
            ref = tx_info[gi].astype(np.int64)
            errs = int(np.sum(info_hat != ref))
            frame_result['ber'] = float(errs / K_POLAR)
            frame_result['info_errs'] = errs
            ber_errs += errs
            ber_total += K_POLAR
        per_frame.append(frame_result)

    n_decoded = sum(1 for f in per_frame if f['decoded'])
    bpsk_ber = bpsk_errs / max(bpsk_total, 1)
    sgnn_ber = sgnn_errs / max(sgnn_total, 1) if sgnn_total > 0 else 0
    mean_abs_llr = float(np.mean(mean_llr_vals)) if mean_llr_vals else 0
    summary = {
        'n_frames': num_frames,
        'n_decoded': n_decoded,
        'decode_rate': n_decoded / max(num_frames, 1),
        'polar_ber': ber_errs / max(ber_total, 1),
        'bpsk_ber': bpsk_ber,
        'sgnn_ber': sgnn_ber,
        'info_errs_total': ber_errs,
        'info_bits_total': ber_total,
        'bpsk_errs_total': bpsk_errs,
        'bpsk_bits_total': bpsk_total,
        'sgnn_errs_total': sgnn_errs,
        'sgnn_bits_total': sgnn_total,
        'mean_abs_llr': mean_abs_llr,
    }

    sgnn_str = f" SGNN_BER={sgnn_ber*100:.2f}%" if sgnn_total > 0 else ""
    print(f"  译码: {n_decoded}/{num_frames} ({summary['decode_rate']*100:.1f}%)  "
          f"BPSK_BER={bpsk_ber*100:.2f}%  Polar_BER={summary['polar_ber']*100:.2f}%"
          f"{sgnn_str}  mean|LLR|={mean_abs_llr:.2f}")
    return {'summary': summary, 'per_frame': per_frame}


def main():
    p = argparse.ArgumentParser(description='离线极化译码 + BER')
    p.add_argument('inputs', nargs='+', help='capture 前缀 或 目录')
    p.add_argument('--gain', type=int, default=0, help='只分析指定 gain')
    p.add_argument('--num-frames', type=int, default=100)
    p.add_argument('--sgnn', action='store_true', help='启用 SGNN 译码 (需要 torch)')
    p.add_argument('--device', default='cpu', help='SGNN 设备 (cpu/cuda)')
    p.add_argument('-o', '--output', default='', help='输出 JSON')
    args = p.parse_args()

    # SGNN 模型加载
    sgnn_model = None
    sgnn_graph = None
    if args.sgnn:
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location(
            'common', os.path.join(BASE, 'deploy', 'common.py'))
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        ckpt = os.path.join(BASE, 'deploy', 'checkpoint', 'polar_GNN_20_iter_0_epoches_13.pt')
        pcm  = os.path.join(BASE, 'deploy', 'matrices', 'pcm.npy')
        sgnn_model, cfg = common.load_model(ckpt, device=args.device)
        sgnn_graph = common.build_graph(pcm, device=args.device)
        print(f"SGNN: nstate={cfg['nstate']} device={args.device}")

    prefixes = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            for f in sorted(glob.glob(os.path.join(inp, '*_iq.npy'))):
                pfx = f.replace('_iq.npy', '')
                if args.gain > 0 and f'gain{args.gain:03d}' not in pfx:
                    continue
                prefixes.append(pfx)
        else:
            prefixes.append(inp.replace('_iq.npy', ''))

    if not prefixes:
        print("未找到 capture")
        sys.exit(1)

    print(f"离线极化译码: {len(prefixes)} captures")
    all_results = {}
    for pfx in prefixes:
        tag = os.path.basename(pfx)
        print(f"\n-- {tag} --")
        r = decode_capture(pfx, num_frames=args.num_frames,
                            sgnn_model=sgnn_model, sgnn_graph=sgnn_graph,
                            sgnn_device=args.device)
        if 'error' in r:
            print(f"  错误: {r['error']}")
            continue
        all_results[tag] = r

    if all_results:
        has_sgnn = any(r['summary'].get('sgnn_bits_total', 0) > 0 for r in all_results.values())
        print(f"\n{'='*60}")
        print(f"  汇总")
        if has_sgnn:
            print(f"  {'capture':>25s}  {'decode':>7s}  {'BPSK':>7s}  {'Hard':>7s}  {'SGNN':>7s}  {'|LLR|':>6s}")
        else:
            print(f"  {'capture':>25s}  {'decode':>7s}  {'BPSK_BER':>8s}  {'Polar_BER':>8s}  {'mean|LLR|':>10s}")
        print(f"  {'-'*60}")
        for tag, r in sorted(all_results.items()):
            s = r['summary']
            if has_sgnn:
                print(f"  {tag:>25s}  {s['n_decoded']:4d}/{s['n_frames']:<4d}  "
                      f"{s['bpsk_ber']*100:6.2f}%  {s['polar_ber']*100:6.2f}%  "
                      f"{s['sgnn_ber']*100:6.2f}%  {s['mean_abs_llr']:5.2f}")
            else:
                print(f"  {tag:>25s}  {s['n_decoded']:4d}/{s['n_frames']:<4d}  "
                      f"{s['bpsk_ber']*100:7.2f}%  {s['polar_ber']*100:7.2f}%  "
                      f"{s['mean_abs_llr']:9.2f}")
        print(f"{'='*60}")

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n报告 -> {args.output}")


if __name__ == '__main__':
    main()
