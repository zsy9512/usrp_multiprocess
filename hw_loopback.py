#!/usr/bin/env python3
"""hw_loopback.py — 双 B210 空口测试 (TX/RX 线程, 不同 serial)"""
import sys, os, time, threading, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sender import BpskPhySender
from receiver import BpskPhyReceiver

results = {'stats': None}

def rx_thread_func(kwargs):
    rx = BpskPhyReceiver(samp_rate=1e6,
                         stf_threshold=kwargs['stf_threshold'],
                         pss_pts=kwargs['pss_pts'],
                         rs_corr_thr=kwargs['rs_corr_thr'])
    rx.start(mode='hardware', freq=kwargs['freq'], gain=kwargs['gain_rx'],
             usrp_args=f"serial={kwargs['serial_rx']}",
             sync_mode=kwargs['sync_mode'])
    results['stats'] = rx.get_stats()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--serial-tx', default='320F2BD')
    p.add_argument('--serial-rx', default='320F33F')
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--gain-tx', type=float, default=30)
    p.add_argument('--gain-rx', type=float, default=30)
    p.add_argument('--num-frames', type=int, default=50)
    p.add_argument('--frame-gap-ms', type=float, default=2.0)
    p.add_argument('--sync-mode', default='host')
    p.add_argument('--stf-threshold', type=float, default=0.35)
    p.add_argument('--pss-pts', type=float, default=1.5)
    p.add_argument('--rs-corr-thr', type=float, default=0.25)
    args = p.parse_args()

    print(f"双 B210: TX={args.serial_tx} → RX={args.serial_rx} @ {args.freq/1e6:.3f}MHz")

    # 先启 RX 线程
    kw = vars(args)
    t = threading.Thread(target=rx_thread_func, args=(kw,), daemon=True)
    t.start()
    time.sleep(3)

    # 再发 TX
    print(f'[test] TX: {args.num_frames} frames')
    tx = BpskPhySender(samp_rate=1e6)
    tx.start(mode='hardware', freq=args.freq, gain=args.gain_tx,
             usrp_args=f'serial={args.serial_tx}',
             sync_mode=args.sync_mode,
             num_frames=args.num_frames,
             frame_gap_ms=args.frame_gap_ms,
             tx_delay_s=1.0)
    print(f'[test] TX done, flushing RX...')

    t.join(timeout=args.num_frames * args.frame_gap_ms / 1000 + 8)

    s = results['stats']
    if s:
        crc_rate = s['crc_pass'] / max(s['total_frames'], 1) * 100
        print(f'\n  det={s["total_frames"]}  CRC={s["crc_pass"]}/{s["total_frames"]} ({crc_rate:.1f}%)')
        print(f'  false_alarms={s["false_alarms"]}  overflow={s["overflow"]}')
        print(f'  header_ok={s["header_crc_pass"]}')
    else:
        print('[test] RX: no results')

if __name__ == '__main__':
    main()
