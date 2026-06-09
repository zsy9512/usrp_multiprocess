#!/usr/bin/env python3
"""polar_loopback.py — 极化码 USRP 环回 (收敛版 v2)

  信息比特(128) → Polar编码 → BPSK → 同步 → LLR软解调 → Hard Inverse → BER
  SGNN 评估通过 --dump-llr 离线完成。

  同步链 (三级, 参考 receiver.py):
    ① STF 延迟相关 → 粗包检测 + 粗 CFO (峰值聚类去重)
    ② PSS 互相关   → 精定时 + 峰值质量 (peak_to_mean, peak_to_second)
    ③ RS 线性相位拟合 → 细 CFO + 公共相位 + 信道幅度 + 噪声方差

  帧结构 (符号域):
    STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)
    Payload 承载 N=256 极化编码比特；CRC 仅传输占位，RX 侧不验证。
"""
import sys, os, time, threading, argparse, math
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory, Process, Event, Value
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phy_params import SPS, STF, PSS, RS, RRC, STF_LEN, PSS_LEN, RS_LEN
from phy_params import HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN, STF_DELAY
from phy_params import STF_THRESHOLD, STF_MIN_ENERGY
from phy_params import MIN_WIN_SAMPLES
from phy_params import crc16, crc16_check, bits_to_bytes, bytes_to_bits

RING_CAP = 2_000_000
SAMP_RATE = 1e6
RRC_DELAY = (len(RRC) - 1) // 2    # 10 samples
FRAME_RRC_SAMPLES = 496 * SPS + len(RRC) - 1  # 1012 samples

# ═══════════════════════════════════════════════════════════════════════
# 极化码常量与函数 (纯 numpy, 不依赖 deploy/common 的顶层 torch import)
# ═══════════════════════════════════════════════════════════════════════
N = 256          # 码长
K = 128          # 信息比特数
BASE = os.path.dirname(os.path.abspath(__file__))
FROZEN_PATH = os.path.join(BASE, 'deploy', 'matrices', 'A.npy')
FROZEN_MASK = np.load(FROZEN_PATH).squeeze()  # (256,)  {0,1}, 1=information
LLR_CLIP = 20.0  # LLR 裁剪范围 [-20, 20]


def _polar_encode(u):
    """Arikan polar transform (自逆). u: (N,) {0,1} → cw: (N,) {0,1}."""
    cw = u.copy().ravel()
    for stage in range(1, int(math.log2(N)) + 1):
        sep = N // (1 << stage)
        for j in range(N):
            if (j // sep) % 2 == 0:
                cw[j] = (cw[j] + cw[j + sep]) % 2
    return cw


def _build_codeword(info_bits):
    """K=128 信息比特 → 放置到非冻结位 → Polar 编码 → N=256 码字."""
    u = np.zeros(N, dtype=np.int64)
    u[FROZEN_MASK.astype(bool)] = info_bits.ravel()
    return _polar_encode(u)


def _polar_hard_inverse(llr):
    """硬判逆 Polar 变换 (baseline, 非纠错译码).

    sign(LLR) → polar_transform → 提取信息位.
    这不会利用冻结位约束做 SC/SCL 纠错。
    Polar 纠错能力主要由 SGNN 路径体现。
    """
    hard_bits = (llr < 0).astype(np.int64)
    u_hat = _polar_encode(hard_bits)  # Arikan self-inverse property
    return u_hat[FROZEN_MASK.astype(bool)]


# ===================================================================
# 子进程: PHY 处理 (从共享内存读, 随便多慢都不影响 USRP)
# ===================================================================

def _proc_worker(shm_name, wr_count, has_data, running, num_frames,
                 tx_ts_shm_name, tx_info_shm_name, samp_rate,
                 dump_shm_name=None):
    shm = shared_memory.SharedMemory(name=shm_name)
    ring = np.ndarray(RING_CAP, dtype=np.complex64, buffer=shm.buf)
    tx_ts_shm = shared_memory.SharedMemory(name=tx_ts_shm_name) if tx_ts_shm_name else None
    tx_ts = np.ndarray(num_frames, dtype=np.uint64, buffer=tx_ts_shm.buf) if tx_ts_shm else None
    tx_info_shm = shared_memory.SharedMemory(name=tx_info_shm_name) if tx_info_shm_name else None
    tx_info = np.ndarray((num_frames, K), dtype=np.int8, buffer=tx_info_shm.buf) if tx_info_shm else None
    ts_sym = SPS / samp_rate

    # ── LLR dump 共享内存 ──
    dump_shm = None
    dump_data = None
    dump_count = 0
    if dump_shm_name is not None:
        dump_shm = shared_memory.SharedMemory(name=dump_shm_name)
        # 每帧: 4B frame_id (uint32 LE) + 1024B LLR (256 float32) + 128B tx_info (int8)
        dump_row_bytes = 4 + N * 4 + K
        dump_data = np.ndarray((num_frames, dump_row_bytes), dtype=np.uint8,
                               buffer=dump_shm.buf)

    # ------------------------------------------------------------------
    # ① STF 延迟相关 + 粗 CFO + 峰值聚类
    # ------------------------------------------------------------------
    def _stf_detect(samples):
        """返回 clustered_peaks 和对应的 coarse_cfo.

        聚类: 128-sample 窗口内只保留最高 M 值峰, 避免同一 STF 产生多个候选.
        """
        L = STF_DELAY; N = len(samples)
        if N <= L: return [], []
        r0, rL = samples[:N - L], samples[L:]
        prod = r0 * np.conj(rL)
        ones = np.ones(L, dtype=np.float32)
        P = np.convolve(prod, ones, mode='valid')
        E = np.convolve((np.abs(rL) ** 2).astype(np.float32), ones, mode='valid')
        M = np.abs(P) / (E + 1e-6 * L)

        # 大于门限 + 能量足够
        raw = []
        for d in range(len(M)):
            if M[d] > STF_THRESHOLD:
                le = np.sum(np.abs(samples[d + L:d + 2 * L]) ** 2)
                if le > STF_MIN_ENERGY:
                    raw.append((d, M[d], P[d]))
        if not raw: return [], []

        # 按 M 排序, 128-sample 窗内去重保留最强
        raw.sort(key=lambda x: x[1], reverse=True)
        peaks, cfos, used = [], [], set()
        for d, _, p in raw:
            if d in used: continue
            for dx in range(max(0, d - 128), min(len(M), d + 128)):
                used.add(dx)
            peaks.append(d)
            phase = np.angle(p)
            cfos.append(-phase / (2 * np.pi * L / samp_rate))
        return peaks, cfos

    # ------------------------------------------------------------------
    # ② PSS 互相关 + 质量
    # ------------------------------------------------------------------
    def _pss_find(syms):
        """PSS 交叉相关: 返回 peak_idx, peak_to_mean, peak_to_second, peak_val."""
        M = PSS_LEN
        if len(syms) < M: return -1, 0, 0, 0
        pss_rev = np.conj(PSS[::-1])
        c = np.abs(np.convolve(syms, pss_rev, mode='valid'))
        pk = int(np.argmax(c))
        peak_val = float(c[pk])
        ptm = peak_val / (np.mean(c) + 1e-30)

        # peak_to_second: 搜索远离主峰 ±PSS_LEN/2 外的次大峰
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
            if found: break
        return pk, ptm, pts, peak_val

    # ------------------------------------------------------------------
    # ③ RS 细 CFO + 信道/相位/噪声估计 (粗 CFO 预补偿)
    # ------------------------------------------------------------------
    def _rs_estimate(symbols, rs_pos, coarse_cfo=0.0):
        if rs_pos + RS_LEN > len(symbols): return None

        rs_seg = symbols[rs_pos:rs_pos + RS_LEN].copy()
        n_rs = np.arange(RS_LEN)

        # 粗 CFO 预补偿 → 残余小频偏上用线性拟合
        if abs(coarse_cfo) > 0.0:
            pre_comp = np.exp(-1j * 2 * np.pi * coarse_cfo * (rs_pos + n_rs) * ts_sym)
            rs_seg = rs_seg * pre_comp

        # 细 CFO: unwrap 相位 → 线性回归斜率
        rs_tone = rs_seg * np.conj(RS)
        rs_corr = float(np.abs(np.sum(rs_tone)))
        rs_phase = np.unwrap(np.angle(rs_tone))
        n = np.arange(RS_LEN, dtype=np.float64)
        n_mean = np.mean(n); p_mean = np.mean(rs_phase)
        num = np.sum((n - n_mean) * (rs_phase - p_mean))
        den = np.sum((n - n_mean) ** 2)
        slope = num / (den + 1e-30)
        fine_cfo = slope / (2 * np.pi * ts_sym)

        # 细 CFO 超限 → PSS 定时错误, 拒收
        if abs(fine_cfo) > 500: return None

        # 总 CFO 补偿 + 信道估计
        total_cfo = coarse_cfo + fine_cfo
        total_comp = np.exp(-1j * 2 * np.pi * total_cfo * (rs_pos + n_rs) * ts_sym)
        rs_corrected = symbols[rs_pos:rs_pos + RS_LEN] * total_comp

        h = np.mean(rs_corrected * np.conj(RS))
        if abs(h) < 1e-6: return None

        # Welch 校正噪声方差
        noise = rs_corrected / h - RS
        s2 = max(float(np.sum(np.abs(noise) ** 2) / (RS_LEN - 1)), 1e-30)

        # RS 相关质量门限: 平均每符号相关性 > 0.3
        if rs_corr < RS_LEN * 0.3: return None

        return {'h': h, 'phase_est': float(np.angle(h)), 'sigma2': s2,
                'coarse_cfo': coarse_cfo, 'fine_cfo': fine_cfo,
                'total_cfo': total_cfo, 'rs_corr': rs_corr}

    # ------------------------------------------------------------------
    # BPSK 软解调 (LLR) + 硬判决
    # ------------------------------------------------------------------
    def _bpsk_demod_llr(symbols, data_start, data_len, chan):
        """BPSK 软解调 → LLR.

        LLR = 4 * Re(y_eq) / sigma2,  y_eq = symbols * exp(-j·2π·Δf·t) / h

        RS 估计的 sigma2 是复残差方差 → 实部方差 = sigma2/2.
        LLR = 2*y/(sigma2/2) = 4*y/sigma2.

        sigma2 用 max(sigma2, 1e-6) 做下限保护 (不用 sigma2_clip).
        LLR 再 clip 到 [-LLR_CLIP, LLR_CLIP].
        """
        if data_start + data_len > len(symbols):
            return np.zeros(data_len, dtype=np.float32)
        seg = symbols[data_start:data_start + data_len]
        n = np.arange(data_len)
        total_cfo = chan['total_cfo']
        cfo_comp = np.exp(-1j * 2 * np.pi * total_cfo * (data_start + n) * ts_sym)
        y = seg * cfo_comp
        h = chan['h']
        if abs(h) > 1e-30:
            y = y / h
        sigma2 = max(float(chan['sigma2']), 1e-6)
        llr = 4.0 * y.real / sigma2
        return np.clip(llr, -LLR_CLIP, LLR_CLIP).astype(np.float32)

    def _bpsk_demod_hard(symbols, data_start, data_len, chan):
        """BPSK 硬判决 (仅用于 Header 和 coded bits)."""
        llr = _bpsk_demod_llr(symbols, data_start, data_len, chan)
        return (llr < 0).astype(np.int64)

    def _b2i(b):
        v = 0
        for x in b: v = (v << 1) | int(x)
        return v

    # ------------------------------------------------------------------
    # 主循环: 缓冲 → 检测 → 同步 → 解调 → 消费
    # ------------------------------------------------------------------
    buf = np.zeros(1_000_000, dtype=np.complex64); buf_len = 0
    rd = 0; total = 0; hdr_ok_cnt = 0
    false_alarms = 0
    info_errs = 0; info_total = 0
    coded_errs = 0; coded_total = 0
    total_lat_us = 0
    noise_floor = None  # 标准 SNR 底噪: 启动期独立测量

    while running.value:
        has_data.wait(timeout=0.5); has_data.clear()
        wc = wr_count.value; avail = wc - rd
        if avail <= 0: continue

        # ---- 标准 SNR 底噪测量（从 ring buffer 直接读，避开 buf 的 5000-cap 窗口） ----
        if noise_floor is None and wc >= 50000:
            start = (wc - 50000) % RING_CAP
            end = wc % RING_CAP
            if end > start:
                noise_iq = ring[start:end].copy()
            else:
                noise_iq = np.concatenate([ring[start:], ring[:end]])
            noise_syms = _rrc_match(noise_iq)
            noise_floor = float(np.var(noise_syms))
            print(f"  [polar_loopback] Noise floor (symbol-level): {noise_floor:.6f}  "
                  f"({10*np.log10(max(noise_floor, 1e-30)):.1f} dB)", flush=True)

        if avail > RING_CAP: rd = wc - RING_CAP

        CHUNK = 4096
        while avail > 0:
            take = min(avail, CHUNK)
            pos = rd % RING_CAP; end = (pos + take) % RING_CAP
            if end > pos: c = ring[pos:pos + take].copy()
            else:
                c = ring[pos:].copy()
                if end > 0: c = np.concatenate([c, ring[:end]])
            rd += take; avail -= take

            n = len(c)
            if n > len(buf) - buf_len:
                d2 = buf_len - len(buf) // 2
                if d2 > 0: buf[:buf_len - d2] = buf[d2:buf_len]; buf_len -= d2
            s = min(n, len(buf) - buf_len)
            buf[buf_len:buf_len + s] = c[:s]; buf_len += s

            while buf_len >= MIN_WIN_SAMPLES:
                ws = max(0, buf_len - 5000)
                r = buf[ws:buf_len]
                peaks, cfos = _stf_detect(r)

                if not peaks:
                    buf[:buf_len - ws] = buf[ws:buf_len]; buf_len -= ws
                    break

                found = False
                for pi, d in enumerate(peaks[:8]):      # 最多尝试 8 个聚类峰
                    coarse_cfo = cfos[pi]
                    if abs(coarse_cfo) > 2000.0:
                        continue
                    coarse = ws + d

                    # 提取窗口: STF 前 200 + 整帧 + 后 200 样本裕量
                    es = max(0, coarse - 200)
                    ee = min(buf_len, coarse + 200 + FRAME_RRC_SAMPLES + 200)
                    syms = _rrc_match(buf[es:ee])
                    if len(syms) < PSS_LEN + RS_LEN: continue

                    pk, ptm, pts, pval = _pss_find(syms)
                    # PSS 质量门限 (略保守, 适配简化信道估计)
                    if ptm < 3.5 or pts < 1.5: continue

                    fs = pk - STF_LEN                    # 帧起始符号索引
                    if fs < 0: continue

                    rp = fs + STF_LEN + PSS_LEN          # RS 起始符号索引
                    if rp + RS_LEN + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN > len(syms):
                        continue

                    # ③ RS 信道估计 (含粗+细 CFO)
                    chan = _rs_estimate(syms, rp, coarse_cfo)
                    if chan is None: continue

                    sigma2_clip = min(chan['sigma2'], 0.5)
                    # 标准 SNR: 信号功率 / 独立底噪
                    hmag = abs(chan['h'])
                    nf = noise_floor if noise_floor is not None else 0.5  # fallback
                    snr_db = 10 * np.log10(max(hmag**2 / max(nf, 1e-30), 1e-30))
                    # EVM: 均衡残差 (反映定时/CFO 等所有损伤)
                    evm_db = 10 * np.log10(max(sigma2_clip, 1e-30))

                    # ④ 解调 Header (硬判, 取 frame_id)
                    hdr_start = rp + RS_LEN
                    hdr_bits = _bpsk_demod_hard(syms, hdr_start, HEADER_LEN, chan)
                    hdr_ok = crc16_check(bits_to_bytes(hdr_bits[:16]),
                                         _b2i(hdr_bits[16:32]))

                    # ⑤ 解调 Payload: 硬判 coded bits + LLR 软信息
                    pay_start = hdr_start + HEADER_LEN
                    pay_hard = _bpsk_demod_hard(syms, pay_start, PAYLOAD_LEN, chan)
                    pay_llr = _bpsk_demod_llr(syms, pay_start, PAYLOAD_LEN, chan)

                    # ⑥ 极化逆变换 → 信息比特估计
                    info_hat = _polar_hard_inverse(pay_llr)

                    # ⑦ 统计 (CRC 不参与成败判据)
                    total += 1
                    if hdr_ok:
                        hdr_ok_cnt += 1

                    fid = _b2i(hdr_bits[:16])
                    if tx_info is not None and fid < len(tx_info) and np.any(tx_info[fid]):
                        ref_info = tx_info[fid].astype(np.int64)
                        # 信息比特 BER (K=128)
                        info_errs += int(np.sum(info_hat != ref_info))
                        info_total += K
                        # 编码比特 BER (N=256, 物理层调试)
                        ref_cw = _build_codeword(ref_info)
                        coded_errs += int(np.sum(pay_hard != ref_cw))
                        coded_total += N

                    # accumulate latency every frame
                    if tx_ts is not None and fid < len(tx_ts) and tx_ts[fid] > 0:
                        total_lat_us += int((time.time_ns() - tx_ts[fid]) / 1000)

                    # \u2500\u2500 LLR dump \u2500\u2500
                    if dump_data is not None and dump_count < num_frames:
                        row = np.zeros(4 + N * 4 + K, dtype=np.uint8)
                        row[0:4] = np.array([fid], dtype=np.uint32).view(np.uint8)
                        row[4:4 + N * 4] = pay_llr.astype(np.float32).view(np.uint8)
                        row[4 + N * 4:4 + N * 4 + K] = info_hat.astype(np.uint8)
                        dump_data[dump_count] = row
                        dump_count += 1

                    if total <= 5 or total % 100 == 0:
                        avg_lat = total_lat_us // max(total, 1)
                        iber = f"iBER={info_errs/max(info_total,1)*100:.2f}%" if info_total > 0 else "---"
                        cber = f"cBER={coded_errs/max(coded_total,1)*100:.2f}%" if coded_total > 0 else "---"
                        print(
                            f"  frame={total:5d}  "
                            f"ptm={ptm:.1f}  pts={pts:.1f}  "
                            f"\u0394f0={chan['coarse_cfo']:+.0f}  "
                            f"\u0394f1={chan['fine_cfo']:+.0f}  "
                            f"|h|={hmag:.3f}  SNR={snr_db:.1f}dB  "
                            f"EVM={evm_db:.1f}dB  "
                            f"avglat={avg_lat}us  "
                            f"HDR={'OK' if hdr_ok else 'XX'}  "
                            f"{iber}  {cber}",
                            flush=True)

                    # ⑦ 消费窗口: 帧起始 + 帧样本数 + 50 样本裕量
                    consume_end = es + fs * SPS + FRAME_RRC_SAMPLES + 50
                    if consume_end > buf_len: consume_end = buf_len
                    if consume_end < buf_len:
                        buf[:buf_len - consume_end] = buf[consume_end:buf_len]
                    buf_len -= min(consume_end, buf_len)
                    found = True
                    break

                if not found:
                    false_alarms += len(peaks[:8])
                    buf[:buf_len - ws] = buf[ws:buf_len]; buf_len -= ws
                    break

    print(f"\n--- \u7ed3\u679c ---")
    print(f"  frames={total}  HDR={hdr_ok_cnt}  false_alarms={false_alarms}",
          flush=True)
    if info_total > 0:
        print(f"  Info Hard Inverse BER={info_errs/info_total*100:.2f}%  "
              f"({info_errs}/{info_total})  [128 info bits/frame]",
              flush=True)
    if coded_total > 0:
        print(f"  Coded Hard BER={coded_errs/coded_total*100:.2f}%  "
              f"({coded_errs}/{coded_total})  [256 coded bits/frame]",
              flush=True)
    shm.close()
    if tx_ts_shm: tx_ts_shm.close()
    if tx_info_shm: tx_info_shm.close()
    if dump_shm: dump_shm.close()


# ===================================================================
# RRC 匹配滤波 (模块级函数, 供子进程和 TX 侧引用)
# ===================================================================

def _rrc_match(samples):
    f = np.convolve(samples, RRC[::-1], mode='full')
    return f[RRC_DELAY::SPS].astype(np.complex64)


# ===================================================================
# 主进程: USRP + TX线程 + RX线程
# ===================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--serial', default='320F33F')
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--rate', type=float, default=SAMP_RATE)
    p.add_argument('--gain-tx', type=float, default=65)
    p.add_argument('--gain-rx', type=float, default=64)
    p.add_argument('--rx-channel', type=int, default=0,
                   help='RX 通道号 (0=A板 1=B板, 默认0)')
    p.add_argument('--rx-antenna', default='RX2',
                   help='RX 天线端口 (默认RX2, channel 0/2用TX/RX)')
    p.add_argument('--num-frames', type=int, default=1000)
    p.add_argument('--frame-gap-ms', type=float, default=5.0)
    p.add_argument('--dump-llr', default='', help='保存 LLR 到 .npy (离线 SGNN 评估)')

    args = p.parse_args()

    import uhd
    dev = f'serial={args.serial}' if args.serial else ''
    usrp = uhd.usrp.MultiUSRP(dev)
    usrp.set_tx_freq(uhd.types.TuneRequest(args.freq)); usrp.set_tx_gain(args.gain_tx)
    usrp.set_tx_rate(args.rate); usrp.set_tx_bandwidth(args.rate)
    usrp.set_tx_antenna("TX/RX")
    usrp.set_rx_freq(uhd.types.TuneRequest(args.freq), args.rx_channel)
    usrp.set_rx_gain(args.gain_rx, args.rx_channel)
    usrp.set_rx_rate(args.rate, args.rx_channel)
    usrp.set_rx_bandwidth(args.rate, args.rx_channel)
    usrp.set_rx_antenna(args.rx_antenna, args.rx_channel)
    usrp.set_clock_source("internal"); usrp.set_time_source("internal")
    ns = time.time_ns()
    usrp.set_time_now(uhd.types.TimeSpec(ns // 1_000_000_000,
                                         (ns % 1_000_000_000) / 1e9))

    tx_s = uhd.usrp.StreamArgs('fc32', 'sc16'); tx_s.channels = [0]
    tx = usrp.get_tx_stream(tx_s)
    rx_s = uhd.usrp.StreamArgs('fc32', 'sc16'); rx_s.channels = [args.rx_channel]
    rx = usrp.get_rx_stream(rx_s)
    rx.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))

    # ── 创建共享内存 + 子进程 ──
    ctx = mp.get_context('spawn')
    shm = shared_memory.SharedMemory(create=True, size=RING_CAP * 8)
    ring = np.ndarray(RING_CAP, dtype=np.complex64, buffer=shm.buf); ring[:] = 0j
    wr_count = ctx.Value('Q', 0); has_data = ctx.Event()
    running = ctx.Value('i', 1)

    tx_ts_shm = shared_memory.SharedMemory(create=True, size=args.num_frames * 8)
    tx_ts = np.ndarray(args.num_frames, dtype=np.uint64, buffer=tx_ts_shm.buf)
    tx_ts[:] = 0

    tx_info_shm = shared_memory.SharedMemory(create=True, size=args.num_frames * K)
    tx_info_arr = np.ndarray((args.num_frames, K), dtype=np.int8, buffer=tx_info_shm.buf)
    tx_info_arr[:] = 0

    # ── LLR dump 共享内存 ──
    dump_shm = None
    dump_shm_name = None
    if args.dump_llr:
        dump_row_bytes = 4 + N * 4 + K
        dump_shm = shared_memory.SharedMemory(create=True,
            size=args.num_frames * dump_row_bytes)
        dump_shm_name = dump_shm.name

    proc = ctx.Process(target=_proc_worker,
                       args=(shm.name, wr_count, has_data, running, args.num_frames,
                             tx_ts_shm.name, tx_info_shm.name, args.rate,
                             dump_shm_name),
                       daemon=True)
    proc.start()
    print(f"[polar_loopback] 处理子进程 PID={proc.pid}")

    # ── RX 收样线程 ──
    def rx_thread():
        md = uhd.types.RXMetadata()
        b = np.zeros((1, 4096), dtype=np.complex64); w = 0
        while running.value:
            n = rx.recv(b, md, timeout=0.2)
            if n == 0: continue
            if md.error_code == uhd.types.RXMetadataErrorCode.overflow: continue
            data = b[0, :n]; end = w + n
            if end <= RING_CAP: ring[w:end] = data
            else:
                n1 = RING_CAP - w; ring[w:] = data[:n1]; ring[:n - n1] = data[n1:]
            w = end % RING_CAP; wr_count.value += n; has_data.set()

    threading.Thread(target=rx_thread, daemon=True).start()
    time.sleep(1)

    # ── TX 线程 ──
    gap = max(16, int(args.frame_gap_ms * args.rate / 1000))
    tx_done = threading.Event()

    def tx_thread():
        from sender import build_frame, rrc_filter
        md = uhd.types.TXMetadata(); md.start_of_burst = True
        rng_state = 42

        def next_bit():
            nonlocal rng_state
            rng_state = (rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
            return (rng_state >> 31) & 1

        for f in range(args.num_frames):
            tx_ts[f] = time.time_ns()
            # K=128 信息比特 → 极化编码 → N=256 码字
            info_bits = np.fromiter((next_bit() for _ in range(K)),
                                    dtype=np.int64, count=K)
            tx_info_arr[f] = info_bits.astype(np.int8)
            coded_bits = _build_codeword(info_bits)
            iq = rrc_filter(build_frame(coded_bits, f), RRC, SPS)
            tx.send(iq.astype(np.complex64), md); md.start_of_burst = False
            if gap > 0:
                gm = uhd.types.TXMetadata()
                gm.start_of_burst = gm.end_of_burst = False
                tx.send(np.zeros(gap, dtype=np.complex64), gm)
        eob = uhd.types.TXMetadata(); eob.end_of_burst = True
        tx.send(np.zeros(1, dtype=np.complex64), eob)
        tx_done.set()

    threading.Thread(target=tx_thread, daemon=True).start()
    print(f"[polar_loopback] {args.num_frames} frames  gap={args.frame_gap_ms}ms  "
          f"Polar(N=256,K=128) BPSK  "
          f"PSS_thr=(ptm=3.5,pts=1.5)  RS_corr>0.3  "
          f"dump_llr={'on' if args.dump_llr else 'off'}", flush=True)

    # 等 TX 发完 + 子进程处理完
    tx_done.wait()
    time.sleep(3)
    running.value = 0; proc.join(timeout=5)
    if proc.is_alive(): proc.terminate()

    # ── 保存 LLR dump ──
    if args.dump_llr and dump_shm is not None:
        dump_row_bytes = 4 + N * 4 + K
        dump_arr = np.ndarray((args.num_frames, dump_row_bytes), dtype=np.uint8,
                              buffer=dump_shm.buf)
        save_path = args.dump_llr
        np.save(save_path, dump_arr)
        print(f"[polar_loopback] LLR dump saved: {save_path}  "
              f"({args.num_frames} rows, each {dump_row_bytes}B)", flush=True)

    shm.close(); shm.unlink()
    tx_ts_shm.close(); tx_ts_shm.unlink()
    tx_info_shm.close(); tx_info_shm.unlink()
    if dump_shm:
        dump_shm.close(); dump_shm.unlink()


if __name__ == '__main__':
    mp.freeze_support()
    main()
