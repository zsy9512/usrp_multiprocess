#!/usr/bin/env python3
"""
loopback_capture.py — B210 自发自收 IQ 采集 (用于离线调试同步链)

与 loopback_test.py 相同硬件配置, 但额外:
  - 保存 RX IQ 到 .npy
  - 保存 TX 参考比特到 _bits.npy (每帧 PAYLOAD_LEN bits)
  - 自动计算并打印信号幅度诊断
  - 支持切换接收子板 (--rx-subdev)

用法:
  python tools/loopback_capture.py --serial 320F33F -o capture/baseline
  python tools/loopback_capture.py --serial 320F33F --rx-subdev B:A -o capture/subb_b
"""
import argparse, json, os, sys, time, threading, queue
from datetime import datetime, timezone
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phy_params import SPS, RRC, STF_LEN, PSS_LEN, RS_LEN
from phy_params import HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN
from phy_params import FRAME_RRC_SAMPLES

SAMP_RATE = 1e6


def main():
    p = argparse.ArgumentParser(description='B210 自发自收 IQ 采集')
    p.add_argument('--serial', default='320F33F')
    p.add_argument('--rx-channel', type=int, default=1,
                   help='RX 通道号 (0=A:TX/RX 1=A:RX2 2=B:TX/RX 3=B:RX2, 默认1)')
    p.add_argument('--rx-antenna', default='RX2',
                   help='RX 天线端口 (默认RX2, channel 0/2用TX/RX)')
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--gain-tx', type=float, default=65)
    p.add_argument('--gain-rx', type=float, default=64)
    p.add_argument('--num-frames', type=int, default=200)
    p.add_argument('--frame-gap-ms', type=float, default=5.0)
    p.add_argument('-o', '--output', default='loopback_capture',
                   help='输出文件前缀 (生成 _iq.npy 和 _bits.npy)')
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
    tx_bits_list = []   # 保存每帧的随机比特

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
    from sender import build_frame, rrc_filter

    gap = max(16, int(args.frame_gap_ms * SAMP_RATE / 1000))
    tx_done = threading.Event()

    def tx_thread():
        md_tx = uhd.types.TXMetadata(); md_tx.start_of_burst = True
        # 固定种子保证可复现
        rng = np.random.RandomState(42)
        for f in range(args.num_frames):
            raw = rng.randint(0, 2, PAYLOAD_LEN).astype(np.int64)
            tx_bits_list.append(raw)
            iq = rrc_filter(build_frame(raw, f), RRC, SPS)
            tx_stream.send(iq.astype(np.complex64), md_tx)
            md_tx.start_of_burst = False
            if gap > 0:
                gm = uhd.types.TXMetadata()
                gm.start_of_burst = gm.end_of_burst = False
                tx_stream.send(np.zeros(gap, dtype=np.complex64), gm)
        eob = uhd.types.TXMetadata(); eob.end_of_burst = True
        tx_stream.send(np.zeros(1, dtype=np.complex64), eob)
        tx_done.set()

    th.Thread(target=tx_thread, daemon=True).start()

    # ── 等待 TX 完成 + 额外 2s 收尾 ──
    print(f"[capture] TX {args.num_frames} frames @ {args.freq/1e6:.1f}MHz  "
          f"TXgain={args.gain_tx}  RXgain={args.gain_rx}  gap={args.frame_gap_ms}ms")
    tx_done.wait()
    time.sleep(2)
    running = False
    time.sleep(0.5)

    # ── 收集 RX IQ ──
    rx_chunks = []
    while not rx_buf.empty():
        rx_chunks.append(rx_buf.get_nowait())
    rx_iq = np.concatenate(rx_chunks) if rx_chunks else np.array([], dtype=np.complex64)
    tx_bits = np.concatenate(tx_bits_list) if tx_bits_list else np.array([], dtype=np.int64)

    # ── 保存 ──
    iq_path = args.output + '_iq.npy'
    bits_path = args.output + '_bits.npy'
    meta_path = args.output + '_meta.json'
    np.save(iq_path, rx_iq)
    np.save(bits_path, tx_bits)

    # 元数据 JSON
    from phy_params import STF_THRESHOLD, STF_MIN_ENERGY
    meta = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'freq_hz': args.freq,
        'gain_tx_db': args.gain_tx,
        'gain_rx_db': args.gain_rx,
        'rx_channel': args.rx_channel,
        'rx_antenna': args.rx_antenna,
        'num_frames': args.num_frames,
        'frame_gap_ms': args.frame_gap_ms,
        'samp_rate': 1e6,
        'sps': SPS,
        'serial': args.serial,
        'stf_threshold': STF_THRESHOLD,
        'stf_min_energy': STF_MIN_ENERGY,
        'pss_ptm': 3.5,
        'pss_pts': 1.5,
        'rs_corr_thr': 0.3,
        'payload_len': PAYLOAD_LEN,
        'header_len': HEADER_LEN,
        'stf_len': STF_LEN,
        'pss_len': PSS_LEN,
        'rs_len': RS_LEN,
        'frame_rrc_samples': FRAME_RRC_SAMPLES,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  元数据: {meta_path}")

    # ── 信号质量诊断 ──
    mag = np.abs(rx_iq)
    peak = np.max(mag) if len(mag) > 0 else 0
    rms = np.sqrt(np.mean(mag**2)) if len(mag) > 0 else 0
    dur_ms = len(rx_iq) / SAMP_RATE * 1000

    print(f"\n[capture] 完成:")
    print(f"  RX IQ: {len(rx_iq)} 样本 ({dur_ms:.0f}ms)  →  {iq_path}")
    print(f"  TX bits: {len(tx_bits)} bits ({args.num_frames}×{PAYLOAD_LEN})  →  {bits_path}")
    print(f"  幅度: peak={peak:.4f}  RMS={rms:.4f}  crest={peak/(rms+1e-30):.1f}")
    print(f"  overflow={overflow_count[0]}")
    if peak > 0.95:
        print(f"  ⚠ 削峰! 建议降低 gain-rx 或 gain-tx")
    elif peak < 0.05:
        print(f"  ⚠ 信号极弱! 检查 SMA 连接或提高增益")
    elif peak < 0.1:
        print(f"  ⚠ 信号偏弱 (peak={peak:.3f}), 建议提高增益")
    else:
        print(f"  ✅ 信号幅度正常")

    rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    usrp = None


if __name__ == '__main__':
    main()
