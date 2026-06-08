#!/usr/bin/env python3
"""loopback_test.py — 单 B210 环回 (USRP线程+处理后子进程, 零overflow)

同步链 (三级, 参考 receiver.py):
  ① STF 延迟相关 → 粗包检测 + 粗 CFO (峰值聚类去重)
  ② PSS 互相关   → 精定时 + 峰值质量 (peak_to_mean, peak_to_second)
  ③ RS 线性相位拟合 → 细 CFO + 公共相位 + 信道幅度 + 噪声方差

帧结构 (符号域):
  STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)
"""
import sys, os, time, threading, argparse
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory, Process, Event, Value
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phy_params import SPS, STF, PSS, RS, RRC, STF_LEN, PSS_LEN, RS_LEN
from phy_params import HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN, STF_DELAY
from phy_params import STF_THRESHOLD, STF_MIN_ENERGY
from phy_params import crc16, crc16_check, bits_to_bytes, bytes_to_bits

RING_CAP = 2_000_000
SAMP_RATE = 1e6
TS_SYM = SPS / SAMP_RATE           # 符号周期 (s) = 2e-6
RRC_DELAY = (len(RRC) - 1) // 2    # 10 samples
FRAME_RRC_SAMPLES = 496 * SPS + len(RRC) - 1  # 1012 samples


# ===================================================================
# 子进程: PHY 处理 (从共享内存读, 随便多慢都不影响 USRP)
# ===================================================================

def _proc_worker(shm_name, wr_count, has_data, running, num_frames, tx_ts_shm_name):
    shm = shared_memory.SharedMemory(name=shm_name)
    ring = np.ndarray(RING_CAP, dtype=np.complex64, buffer=shm.buf)
    tx_ts_shm = shared_memory.SharedMemory(name=tx_ts_shm_name) if tx_ts_shm_name else None
    tx_ts = np.ndarray(num_frames, dtype=np.uint64, buffer=tx_ts_shm.buf) if tx_ts_shm else None

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
            cfos.append(-phase / (2 * np.pi * L / SAMP_RATE))
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
        if abs(coarse_cfo) > 1.0:
            pre_comp = np.exp(-1j * 2 * np.pi * coarse_cfo * (rs_pos + n_rs) * TS_SYM)
            rs_seg = rs_seg * pre_comp

        # 细 CFO: unwrap 相位 → 线性回归斜率
        rs_tone = rs_seg * np.conj(RS)
        rs_phase = np.unwrap(np.angle(rs_tone))
        n = np.arange(RS_LEN, dtype=np.float64)
        n_mean = np.mean(n); p_mean = np.mean(rs_phase)
        num = np.sum((n - n_mean) * (rs_phase - p_mean))
        den = np.sum((n - n_mean) ** 2)
        slope = num / (den + 1e-30)
        fine_cfo = slope / (2 * np.pi * TS_SYM)

        # 细 CFO 超限 → PSS 定时错误, 拒收
        if abs(fine_cfo) > 500: return None

        # 总 CFO 补偿 + 信道估计
        total_cfo = coarse_cfo + fine_cfo
        total_comp = np.exp(-1j * 2 * np.pi * total_cfo * (rs_pos + n_rs) * TS_SYM)
        rs_corrected = symbols[rs_pos:rs_pos + RS_LEN] * total_comp

        h = np.mean(rs_corrected * np.conj(RS))
        if abs(h) < 1e-6: return None

        # Welch 校正噪声方差
        noise = rs_corrected / h - RS
        s2 = max(float(np.var(noise)) * RS_LEN / (RS_LEN - 1), 1e-30)

        # RS 相关质量门限: 平均每符号相关性 > 0.3
        rs_corr = float(np.abs(np.sum(rs_corrected * np.conj(RS))))
        if rs_corr < RS_LEN * 0.3: return None

        return {'h': h, 'phase_est': float(np.angle(h)), 'sigma2': s2,
                'coarse_cfo': coarse_cfo, 'fine_cfo': fine_cfo,
                'total_cfo': total_cfo, 'rs_corr': rs_corr}

    # ------------------------------------------------------------------
    # BPSK 解调 (CFO + 相位 + 信道全补偿后硬判决)
    # ------------------------------------------------------------------
    def _bpsk_demod(symbols, data_start, data_len, chan):
        if data_start + data_len > len(symbols):
            return np.zeros(data_len, dtype=np.int64)
        seg = symbols[data_start:data_start + data_len]
        n = np.arange(data_len)
        total_cfo = chan['total_cfo']
        cfo_comp = np.exp(-1j * 2 * np.pi * total_cfo * (data_start + n) * TS_SYM)
        y = seg * cfo_comp
        h = chan['h']
        if abs(h) > 1e-30: y = y / h
        return (y.real < 0).astype(np.int64)

    def _b2i(b):
        v = 0
        for x in b: v = (v << 1) | int(x)
        return v

    # ------------------------------------------------------------------
    # 主循环: 缓冲 → 检测 → 同步 → 解调 → 消费
    # ------------------------------------------------------------------
    buf = np.zeros(1_000_000, dtype=np.complex64); buf_len = 0
    rd = 0; total = 0; hdr_ok_cnt = 0; crc_ok_cnt = 0
    false_alarms = 0

    while running.value:
        has_data.wait(timeout=0.5); has_data.clear()
        wc = wr_count.value; avail = wc - rd
        if avail <= 0: continue
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

            while buf_len >= 2000:
                ws = max(0, buf_len - 5000)
                r = buf[ws:buf_len]
                peaks, cfos = _stf_detect(r)

                if not peaks:
                    buf[:buf_len - ws] = buf[ws:buf_len]; buf_len -= ws
                    break

                found = False
                for pi, d in enumerate(peaks[:8]):      # 最多尝试 8 个聚类峰
                    coarse_cfo = cfos[pi]
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

                    # ④ 解调 Header
                    hdr_start = rp + RS_LEN
                    hdr = _bpsk_demod(syms, hdr_start, HEADER_LEN, chan)
                    hdr_ok = crc16_check(bits_to_bytes(hdr[:16]), _b2i(hdr[16:32]))

                    # ⑤ 解调 Payload + CRC
                    pay_start = hdr_start + HEADER_LEN
                    pay = _bpsk_demod(syms, pay_start, PAYLOAD_LEN + PAYLOAD_CRC_LEN, chan)
                    payload_bits = pay[:PAYLOAD_LEN]
                    crc_ok = crc16_check(bits_to_bytes(payload_bits),
                                         _b2i(pay[PAYLOAD_LEN:]))

                    # ⑥ 统计
                    total += 1
                    if hdr_ok: hdr_ok_cnt += 1
                    if crc_ok: crc_ok_cnt += 1

                    if total <= 5 or total % 100 == 0:
                        hmag = abs(chan['h'])
                        snr = 10 * np.log10(max(hmag ** 2 / sigma2_clip, 1e-30))
                        fid = _b2i(hdr[:16])
                        lat_us = -1
                        if tx_ts is not None and fid < len(tx_ts) and tx_ts[fid] > 0:
                            lat_us = int((time.time_ns() - tx_ts[fid]) / 1000)
                        print(
                            f"  frame={total:5d}  "
                            f"ptm={ptm:.1f}  pts={pts:.1f}  "
                            f"\u0394f0={chan['coarse_cfo']:+.0f}  "
                            f"\u0394f1={chan['fine_cfo']:+.0f}  "
                            f"|h|={hmag:.3f}  SNR={snr:.1f}dB  "
                            f"lat={lat_us}us  "
                            f"HDR={'OK' if hdr_ok else 'XX'}  "
                            f"CRC={'OK' if crc_ok else 'XX'}",
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
    print(f"  frames={total}  CRC={crc_ok_cnt}/{total} "
          f"({crc_ok_cnt/max(total,1)*100:.1f}%)  "
          f"HDR={hdr_ok_cnt}  false_alarms={false_alarms}",
          flush=True)
    shm.close()
    if tx_ts_shm: tx_ts_shm.close()


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
    p.add_argument('--gain-tx', type=float, default=65)
    p.add_argument('--gain-rx', type=float, default=64)
    p.add_argument('--rx-channel', type=int, default=0,
                   help='RX 通道号 (0=A板 1=B板, 默认0)')
    p.add_argument('--rx-antenna', default='RX2',
                   help='RX 天线端口 (默认RX2, channel 0/2用TX/RX)')
    p.add_argument('--num-frames', type=int, default=1000)
    p.add_argument('--frame-gap-ms', type=float, default=5.0)

    args = p.parse_args()

    import uhd
    dev = f'serial={args.serial}' if args.serial else ''
    usrp = uhd.usrp.MultiUSRP(dev)
    usrp.set_tx_freq(uhd.types.TuneRequest(args.freq)); usrp.set_tx_gain(args.gain_tx)
    usrp.set_tx_rate(SAMP_RATE); usrp.set_tx_bandwidth(SAMP_RATE)
    usrp.set_tx_antenna("TX/RX")
    usrp.set_rx_freq(uhd.types.TuneRequest(args.freq), args.rx_channel)
    usrp.set_rx_gain(args.gain_rx, args.rx_channel)
    usrp.set_rx_rate(SAMP_RATE, args.rx_channel)
    usrp.set_rx_bandwidth(SAMP_RATE, args.rx_channel)
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

    proc = ctx.Process(target=_proc_worker,
                       args=(shm.name, wr_count, has_data, running, args.num_frames,
                             tx_ts_shm.name),
                       daemon=True)
    proc.start()
    print(f"[loopback] 处理子进程 PID={proc.pid}")

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
    gap = max(16, int(args.frame_gap_ms * SAMP_RATE / 1000))
    tx_done = threading.Event()

    def tx_thread():
        from sender import build_frame, rrc_filter
        md = uhd.types.TXMetadata(); md.start_of_burst = True
        for f in range(args.num_frames):
            tx_ts[f] = time.time_ns()
            raw = np.random.randint(0, 2, PAYLOAD_LEN).astype(np.int64)
            iq = rrc_filter(build_frame(raw, f), RRC, SPS)
            tx.send(iq.astype(np.complex64), md); md.start_of_burst = False
            if gap > 0:
                gm = uhd.types.TXMetadata()
                gm.start_of_burst = gm.end_of_burst = False
                tx.send(np.zeros(gap, dtype=np.complex64), gm)
        eob = uhd.types.TXMetadata(); eob.end_of_burst = True
        tx.send(np.zeros(1, dtype=np.complex64), eob)
        tx_done.set()

    threading.Thread(target=tx_thread, daemon=True).start()
    print(f"[loopback] {args.num_frames} frames  gap={args.frame_gap_ms}ms  BPSK  "
          f"PSS_thr=(ptm=3.5,pts=1.5)  RS_corr>0.3", flush=True)

    # 等 TX 发完 + 子进程处理完
    tx_done.wait()
    time.sleep(3)
    running.value = 0; proc.join(timeout=5)
    if proc.is_alive(): proc.terminate()
    shm.close(); shm.unlink()
    tx_ts_shm.close(); tx_ts_shm.unlink()


if __name__ == '__main__':
    mp.freeze_support()
    main()
