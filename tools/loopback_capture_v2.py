#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loopback_capture_v2.py — 长前导 + 极化码 USRP 采集 (离线分析用)

与 loopback_capture.py 的区别:
  - STF=128 (8x16), RS=64 (原 64+32)
  - Payload 为极化编码比特 (N=256), 信息比特 (K=128) 随机生成
  - 帧间间隔 (不同 frame_id) = 30 ms, 重复帧间隔 = 3 ms
  - 保存 TX 信息比特到 _info.npy (用于离线 BER 计算)

帧结构 (符号域, 共 592 symbols):
  STF(128) + PSS(64) + RS(64) + Header(32) + Payload(256) + CRC(16) + Guard(32)

用法:
  python tools/loopback_capture_v2.py --serial 320F33F --gain-tx 60 --gain-rx 30 -o capture/test
"""
import argparse, json, os, sys, time, threading, queue, math
from datetime import datetime, timezone
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phy_params import SPS, RRC, PSS_LEN
from phy_params import HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN, GUARD_SYMBOLS
from phy_params import crc16, bits_to_bytes, bytes_to_bits

SAMP_RATE = 1e6

# ── 极化码常量 ──────────────────────────────────────────────
N_POLAR = 256
K_POLAR = 128
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_PATH = os.path.join(BASE_DIR, 'deploy', 'matrices', 'A.npy')
FROZEN_MASK = np.load(FROZEN_PATH).squeeze()  # (256,)


def _polar_encode(u):
    cw = u.copy().ravel()
    for stage in range(1, int(math.log2(N_POLAR)) + 1):
        sep = N_POLAR // (1 << stage)
        for j in range(N_POLAR):
            if (j // sep) % 2 == 0:
                cw[j] = (cw[j] + cw[j + sep]) % 2
    return cw


def _build_codeword(info_bits):
    u = np.zeros(N_POLAR, dtype=np.int64)
    u[FROZEN_MASK.astype(bool)] = info_bits.ravel()
    return _polar_encode(u)


# ── 可变帧结构辅助 ─────────────────────────────────────────
def make_stf(n_reps=8, base_len=16):
    """STF: n_reps × base_len 重复 BPSK (默认 8×16=128)."""
    rng = np.random.RandomState(7)
    base = 2 * rng.randint(0, 2, base_len) - 1
    return np.tile(base, n_reps).astype(np.complex64)


def make_rs(n_syms=64):
    """RS: 固定 BPSK 导频 (默认 64)."""
    rng = np.random.RandomState(13)
    return (2 * rng.randint(0, 2, n_syms) - 1).astype(np.complex64)


def build_frame_v2(data_bits, frame_id, stf_syms, pss_syms, rs_syms):
    """构建 v2 帧 (长前导版本)."""
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


def rrc_pulse(symbols, rrc, sps):
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    return np.convolve(up, rrc, mode='full').astype(np.complex64)


def main():
    p = argparse.ArgumentParser(description='B210 自发自收 IQ 采集 (长前导 + 极化码)')
    p.add_argument('--serial', default='320F33F')
    p.add_argument('--rx-channel', type=int, default=1)
    p.add_argument('--rx-antenna', default='RX2')
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--gain-tx', type=float, default=60)
    p.add_argument('--gain-rx', type=float, default=30)
    p.add_argument('--num-frames', type=int, default=100,
                   help='唯一帧数 (组数)')
    p.add_argument('--frame-gap-ms', type=float, default=30.0,
                   help='不同 frame_id 间隔 ms')
    p.add_argument('-o', '--output', default='capture_v2/test')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.',
                exist_ok=True)

    import uhd
    dev = f'serial={args.serial}' if args.serial else ''
    usrp = uhd.usrp.MultiUSRP(dev)
    usrp.set_tx_freq(uhd.types.TuneRequest(args.freq))
    usrp.set_tx_gain(args.gain_tx)
    usrp.set_tx_rate(SAMP_RATE)
    usrp.set_tx_bandwidth(SAMP_RATE)
    usrp.set_tx_antenna("TX/RX")
    usrp.set_rx_freq(uhd.types.TuneRequest(args.freq), args.rx_channel)
    usrp.set_rx_gain(args.gain_rx, args.rx_channel)
    usrp.set_rx_rate(SAMP_RATE, args.rx_channel)
    usrp.set_rx_bandwidth(SAMP_RATE, args.rx_channel)
    usrp.set_rx_antenna(args.rx_antenna, args.rx_channel)
    usrp.set_clock_source("internal")
    usrp.set_time_source("internal")
    ns = time.time_ns()
    usrp.set_time_now(uhd.types.TimeSpec(ns // 1_000_000_000,
                                         (ns % 1_000_000_000) / 1e9))

    tx_s = uhd.usrp.StreamArgs('fc32', 'sc16'); tx_s.channels = [0]
    tx_stream = usrp.get_tx_stream(tx_s)
    rx_s = uhd.usrp.StreamArgs('fc32', 'sc16')
    rx_s.channels = [args.rx_channel]
    rx_stream = usrp.get_rx_stream(rx_s)
    rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))

    # 共享状态
    running = True
    rx_buf = queue.Queue()
    overflow_count = [0]
    tx_info_list = []  # 保存每帧的 128bit 信息位

    # ── 参考序列 ──
    from phy_params import PSS as PSS_REF
    stf_syms = make_stf(8)      # 128 symbols
    rs_syms = make_rs(64)       # 64 symbols
    pss_syms = PSS_REF          # 64 symbols (Zadoff-Chu)

    total_sym = (len(stf_syms) + len(pss_syms) + len(rs_syms)
                 + HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN + GUARD_SYMBOLS)
    frame_iq_len = total_sym * SPS + len(RRC) - 1

    # ── RX 收样线程 ──
    def rx_thread():
        md = uhd.types.RXMetadata()
        buf = np.zeros((1, 8192), dtype=np.complex64)
        while running:
            n = rx_stream.recv(buf, md, timeout=0.2)
            if n == 0: continue
            if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
                overflow_count[0] += 1
                continue
            rx_buf.put(buf[0, :n].copy())

    import threading as th
    th.Thread(target=rx_thread, daemon=True).start()
    time.sleep(1)

    # ── TX 线程 ──
    REPEAT = 5
    GAP_REPEAT = int(0.005 * SAMP_RATE)   # 5 ms (same-ID repeat gap)
    GAP_GROUP  = int(args.frame_gap_ms * SAMP_RATE / 1000)
    tx_done = threading.Event()

    def tx_thread():
        md_tx = uhd.types.TXMetadata(); md_tx.start_of_burst = True
        gm = uhd.types.TXMetadata(); gm.start_of_burst = gm.end_of_burst = False
        rng = np.random.RandomState(42)
        for f in range(args.num_frames):
            # 随机信息比特 → Polar 编码 → 成帧
            info = (rng.rand(K_POLAR) < 0.5).astype(np.int64)
            coded = _build_codeword(info)
            tx_info_list.append(info)
            frame_syms = build_frame_v2(coded, f, stf_syms, pss_syms, rs_syms)
            iq = rrc_pulse(frame_syms, RRC, SPS)

            for r in range(REPEAT):
                tx_stream.send(iq.astype(np.complex64), md_tx)
                md_tx.start_of_burst = False
                if r < REPEAT - 1:
                    tx_stream.send(np.zeros(GAP_REPEAT, dtype=np.complex64), gm)
            if GAP_GROUP > 0:
                tx_stream.send(np.zeros(GAP_GROUP, dtype=np.complex64), gm)
        eob = uhd.types.TXMetadata(); eob.end_of_burst = True
        tx_stream.send(np.zeros(1, dtype=np.complex64), eob)
        tx_done.set()

    th.Thread(target=tx_thread, daemon=True).start()

    # ── 等待 TX 完成 = 采集 ──
    stf_sym_count = len(stf_syms)
    rs_sym_count = len(rs_syms)
    print(f"[capture] TX {args.num_frames} frames ×{REPEAT} repeats "
          f"STF={stf_sym_count} RS={rs_sym_count} @ {args.freq/1e6:.1f}MHz  "
          f"TXgain={args.gain_tx}  RXgain={args.gain_rx}  "
          f"gap={args.frame_gap_ms}ms")
    tx_done.wait()
    time.sleep(2)
    running = False
    time.sleep(0.5)

    # ── 收集 RX IQ ──
    rx_chunks = []
    while not rx_buf.empty():
        rx_chunks.append(rx_buf.get_nowait())
    rx_iq = np.concatenate(rx_chunks) if rx_chunks else np.array([], dtype=np.complex64)

    # ── 保存 ──
    iq_path = args.output + '_iq.npy'
    bits_path = args.output + '_bits.npy'
    info_path = args.output + '_info.npy'
    meta_path = args.output + '_meta.json'
    np.save(iq_path, rx_iq)

    coded_bits = np.concatenate(
        [_build_codeword(info).astype(np.int64) for info in tx_info_list])
    np.save(bits_path, coded_bits)

    tx_info_all = np.concatenate(tx_info_list) if tx_info_list else np.zeros(0, dtype=np.int64)
    np.save(info_path, tx_info_all)

    meta = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'version': 'v2_long_preamble',
        'freq_hz': args.freq,
        'gain_tx_db': args.gain_tx,
        'gain_rx_db': args.gain_rx,
        'rx_channel': args.rx_channel,
        'rx_antenna': args.rx_antenna,
        'num_frames': args.num_frames,
        'frame_gap_ms': args.frame_gap_ms,
        'gap_repeat_ms': 5.0,
        'samp_rate': SAMP_RATE,
        'sps': SPS,
        'stf_syms': stf_sym_count,
        'rs_syms': rs_sym_count,
        'pss_syms': PSS_LEN,
        'total_syms': total_sym,
        'frame_iq_len': frame_iq_len,
        'serial': args.serial,
        'payload_len': PAYLOAD_LEN,
        'n_polar': N_POLAR,
        'k_polar': K_POLAR,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  元数据: {meta_path}")

    # 信号质量诊断
    mag = np.abs(rx_iq)
    peak = np.max(mag) if len(mag) > 0 else 0
    rms = np.sqrt(np.mean(mag**2)) if len(mag) > 0 else 0
    dur_ms = len(rx_iq) / SAMP_RATE * 1000

    print(f"\n[capture] 完成:")
    print(f"  RX IQ: {len(rx_iq)} 样本 ({dur_ms:.0f}ms)  ->  {iq_path}")
    print(f"  TX coded bits: {len(coded_bits)} ({args.num_frames} unique ×{N_POLAR})  ->  {bits_path}")
    print(f"  TX info bits: {len(tx_info_all)} ({args.num_frames} ×{K_POLAR})  ->  {info_path}")
    print(f"  幅度: peak={peak:.4f}  RMS={rms:.4f}  crest={peak/(rms+1e-30):.1f}")
    print(f"  overflow={overflow_count[0]}")
    if peak > 0.95:
        print(f"  [WARN] 削峰!")
    elif peak < 0.05:
        print(f"  [WARN] 信号极弱!")
    elif peak < 0.1:
        print(f"  [WARN] 信号偏弱 (peak={peak:.3f})")
    else:
        print(f"  [OK] 信号幅度正常")

    rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    usrp = None


if __name__ == '__main__':
    main()
