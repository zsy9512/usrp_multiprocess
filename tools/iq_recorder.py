#!/usr/bin/env python3
"""
iq_recorder.py — B210 IQ 录制工具

连续接收并保存原始 IQ 到 .npy 文件，用于离线分析。

用法:
  python tools/iq_recorder.py --serial 320F33F --freq 915e6 --gain 20 --duration 5 -o capture.npy
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--serial', default='')
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--gain', type=float, default=20)
    p.add_argument('--rate', type=float, default=1e6)
    p.add_argument('--duration', type=float, default=5.0)
    p.add_argument('--subdev', default='A:A')
    p.add_argument('-o', '--output', default='iq_capture.npy')
    args = p.parse_args()

    print(f"录制: {args.freq/1e6:.3f}MHz  gain={args.gain}dB  {args.duration}s → {args.output}")

    import uhd
    dev_args = f'serial={args.serial}' if args.serial else ''
    usrp = uhd.usrp.MultiUSRP(dev_args)
    if args.subdev:
        try: usrp.set_rx_subdev_spec(args.subdev)
        except: pass
    usrp.set_rx_freq(uhd.types.TuneRequest(args.freq))
    usrp.set_rx_gain(args.gain)
    usrp.set_rx_rate(args.rate)
    usrp.set_rx_bandwidth(args.rate)
    usrp.set_rx_antenna("RX2")
    usrp.set_clock_source("internal")
    usrp.set_time_source("internal")
    pc_ns = time.time_ns()
    tspec = uhd.types.TimeSpec(pc_ns // 1_000_000_000, (pc_ns % 1_000_000_000) / 1e9)
    usrp.set_time_now(tspec)

    stream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
    stream_args.channels = [0]
    rx_stream = usrp.get_rx_stream(stream_args)
    rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))

    md = uhd.types.RXMetadata()
    buf = np.zeros((1, 8192), dtype=np.complex64)
    all_iq = []
    n_samples = 0
    overflow = 0
    t_start = time.time()

    print("录制中...")
    while time.time() - t_start < args.duration:
        ns = rx_stream.recv(buf, md, timeout=0.5)
        if ns == 0: continue
        if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
            overflow += 1
            continue
        chunk = buf[0, :ns].copy()
        all_iq.append(chunk)
        n_samples += ns
        elapsed = time.time() - t_start
        print(f"\r  {n_samples} samples  {elapsed:.1f}s  overflow={overflow}  ", end='', flush=True)

    print()
    rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    usrp = None

    iq = np.concatenate(all_iq) if all_iq else np.array([], dtype=np.complex64)
    np.save(args.output, iq)
    peak = np.max(np.abs(iq)) if len(iq) > 0 else 0

    print(f"完成: {len(iq)} 样本 ({len(iq)/args.rate:.3f}s)")
    print(f"  幅度: max={peak:.3f}  rms={np.sqrt(np.mean(np.abs(iq)**2)):.3f}" if len(iq) > 0 else "  空数据!")
    print(f"  overflow={overflow}")
    print(f"  保存 → {args.output}")


if __name__ == '__main__':
    main()
