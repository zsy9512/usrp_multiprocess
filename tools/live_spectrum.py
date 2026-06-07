#!/usr/bin/env python3
"""
live_spectrum.py — 实时频谱仪 (共享内存多进程, 零 overflow)

子进程: UHD recv() → shared_memory ring buffer (零拷贝)
主进程: ring 读 → FFT → 绘图

用法:
  python tools/live_spectrum.py --serial 320F33F --freq 915e6 --gain 40
"""
import os, sys, time, argparse
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory, Process, Event, Value
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

RING_CAP = 50_000  # 50ms 缓冲 (频谱仪不需要长历史)


def _recv_proc(shm_name, serial, freq, gain, rate, nfft,
               wr_count, need_fft, running, ovf_count):
    import uhd

    dev_args = f'serial={serial}' if serial else ''
    usrp = uhd.usrp.MultiUSRP(dev_args)
    usrp.set_rx_freq(uhd.types.TuneRequest(freq))
    usrp.set_rx_gain(gain)
    usrp.set_rx_rate(rate)
    usrp.set_rx_bandwidth(rate)
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

    shm = shared_memory.SharedMemory(name=shm_name)
    ring = np.ndarray(RING_CAP, dtype=np.complex64, buffer=shm.buf)

    md = uhd.types.RXMetadata()
    buf = np.zeros((1, 4096), dtype=np.complex64)
    w = 0
    acc = np.zeros(nfft, dtype=np.complex64)
    acc_pos = 0

    while running.value:
        ns = rx_stream.recv(buf, md, timeout=0.2)
        if ns == 0: continue
        if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
            ovf_count.value += 1; acc_pos = 0; continue

        data = buf[0, :ns]
        end = w + ns
        if end <= RING_CAP: ring[w:end] = data
        else:
            n1 = RING_CAP - w; ring[w:] = data[:n1]; ring[:ns - n1] = data[n1:]
        w = end % RING_CAP
        wr_count.value += ns

        remaining = nfft - acc_pos; take = min(ns, remaining)
        acc[acc_pos:acc_pos + take] = data[:take]; acc_pos += take
        if acc_pos >= nfft:
            need_fft.set(); acc_pos = 0
            if take < ns:
                leftover = ns - take
                if leftover <= nfft: acc[:leftover] = data[take:]; acc_pos = leftover

    rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    shm.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--serial', default='')
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--gain', type=float, default=40)
    p.add_argument('--rate', type=float, default=1e6)
    p.add_argument('--nfft', type=int, default=2048)
    args = p.parse_args()

    serial = args.serial
    if 'serial=' in serial: serial = serial.split('serial=')[1].split(',')[0]

    print(f"频谱仪 @ {args.freq/1e6:.3f}MHz  gain={args.gain}dB")
    ctx = mp.get_context('spawn')
    shm = shared_memory.SharedMemory(create=True, size=RING_CAP * 8)
    ring = np.ndarray(RING_CAP, dtype=np.complex64, buffer=shm.buf); ring[:] = 0j
    wr_count = ctx.Value('Q', 0); need_fft = ctx.Event()
    running = ctx.Value('i', 1); ovf_count = ctx.Value('i', 0)

    proc = ctx.Process(target=_recv_proc,
                        args=(shm.name, serial, args.freq, args.gain, args.rate,
                              args.nfft, wr_count, need_fft, running, ovf_count),
                        daemon=True)
    proc.start()
    print(f"  收样 PID={proc.pid}  Ctrl+C 退出")

    freq_axis = np.fft.fftshift(np.fft.fftfreq(args.nfft, 1 / args.rate)) / 1e3
    win = np.kaiser(args.nfft, 8.0)
    acc = np.zeros(args.nfft, dtype=np.complex64); acc_pos = 0; rd_count = 0
    n_history = 30; spec_hist = np.full((n_history, args.nfft), -80.0)

    fig, (ax_time, ax_spec) = plt.subplots(2, 1, figsize=(12, 8))
    plt.ion(); fig.show()
    t_line, = ax_time.plot(np.arange(args.nfft), np.zeros(args.nfft), lw=0.5)
    ax_time.set_ylabel('I'); ax_time.set_title(f'{args.freq/1e6:.3f} MHz  gain={args.gain}dB')
    ax_time.set_ylim(-1.5, 1.5)
    spec_img = ax_spec.imshow(spec_hist, aspect='auto', origin='lower',
                               extent=[freq_axis[0], freq_axis[-1], 0, n_history],
                               vmin=-60, vmax=0, cmap='inferno')
    ax_spec.set_xlabel('Freq (kHz)'); ax_spec.set_ylabel('Frame')
    fig.tight_layout()

    hist_idx = 0; fft_count = 0; ovf_last = 0
    try:
        while proc.is_alive():
            need_fft.wait(timeout=0.3); need_fft.clear()
            wc = wr_count.value; avail = wc - rd_count
            if avail <= 0: plt.pause(0.1); continue
            if avail > RING_CAP // 2:
                rd_count = wc - RING_CAP // 2  # 落后太多, 跳到最新

            while avail > 0 and acc_pos < args.nfft:
                pos = rd_count % RING_CAP
                take = min(avail, args.nfft - acc_pos, RING_CAP - pos)
                acc[acc_pos:acc_pos + take] = ring[pos:pos + take]
                acc_pos += take; rd_count += take; avail = wc - rd_count

            if acc_pos >= args.nfft:
                t_line.set_ydata(np.real(acc))
                peak = np.max(np.abs(acc))
                ax_time.set_ylim(-max(peak * 1.2, 0.01), max(peak * 1.2, 0.01))
                pxx = np.abs(np.fft.fftshift(np.fft.fft(acc * win))) ** 2
                pxx_db = 10 * np.log10(pxx + 1e-30); pxx_db -= np.max(pxx_db)
                spec_hist[hist_idx % n_history] = pxx_db
                spec_img.set_data(spec_hist); hist_idx += 1; fft_count += 1; acc_pos = 0
                if fft_count % 10 == 0:
                    plt.pause(0.03)
                    if ovf_count.value != ovf_last:
                        print(f"  overflow={ovf_count.value}", flush=True)
                        ovf_last = ovf_count.value
                else:
                    fig.canvas.flush_events()
    except KeyboardInterrupt: pass
    finally:
        running.value = 0; proc.join(timeout=2)
        if proc.is_alive(): proc.terminate()
        shm.close(); shm.unlink(); plt.close()
        print(f"\n{fft_count} 帧  overflow={ovf_count.value}")


if __name__ == '__main__':
    mp.freeze_support()
    main()
