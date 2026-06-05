#!/usr/bin/env python3
"""
interferer.py — 干扰器 (仿真 + USRP 硬件)

功能:
  仿真: 在 IQ 文件中添加突发噪声/同频干扰
  硬件: 用第三台 USRP 发送同频突发噪声

用法:
  仿真干扰: python interferer.py tx_iq.npy rx_with_interference.npy --snr-db 10 --burst-prob 0.1
  硬件干扰: python interferer.py --mode hardware --freq 915e6 --gain 50
"""

import argparse, os, sys, time
import numpy as np

GAP_SAMPLES = 100000  # 与 sender.py 一致的帧间隔


class Interferer:
    """干扰器: AWGN / 突发噪声 / 同频BPSK."""

    @staticmethod
    def process(tx_signal, snr_db=15.0, freq_offset=0.0, burst_prob=0.0,
                sigma_b=1.0, samp_rate=1e6):
        """在信号上叠加干扰.

        Args:
            tx_signal: 原始发送信号
            snr_db: AWGN 信噪比
            freq_offset: 频偏 (Hz)
            burst_prob: 每符号突发概率 (0=无突发)
            sigma_b: 突发噪声标准差
            samp_rate: 采样率
        Returns:
            rx_signal: 加干扰后的信号
        """
        rx = tx_signal.copy().astype(np.complex64)

        # 频偏
        if abs(freq_offset) > 0:
            t = np.arange(len(rx), dtype=np.float64) / samp_rate
            rx *= np.exp(1j * 2 * np.pi * freq_offset * t)

        # AWGN
        if snr_db < 100:
            sig_power = np.mean(np.abs(rx)**2)
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise = np.sqrt(noise_power / 2) * \
                    (np.random.randn(len(rx)) + 1j * np.random.randn(len(rx)))
            rx += noise.astype(np.complex64)

        # 突发噪声 (匹配训练时的 burst 模型)
        if burst_prob > 0 and sigma_b > 0:
            burst_mask = np.random.rand(len(rx)) < burst_prob
            burst_noise = sigma_b * \
                (np.random.randn(len(rx)) + 1j * np.random.randn(len(rx)))
            rx += (burst_mask * burst_noise).astype(np.complex64)

        return rx

    @staticmethod
    def generate_burst_noise(duration_samples, sigma_b=1.0, duty_cycle=0.3):
        """生成突发噪声序列 (用于硬件干扰)."""
        noise = np.zeros(duration_samples, dtype=np.complex64)
        burst_len = int(duration_samples * duty_cycle)
        burst_start = np.random.randint(0, duration_samples - burst_len)
        burst = sigma_b * (np.random.randn(burst_len) + 1j * np.random.randn(burst_len))
        noise[burst_start:burst_start + burst_len] = burst.astype(np.complex64)
        return noise


def hardware_interferer(freq, gain, rate, usrp_args, sigma_b, burst_prob):
    """用 USRP 发送同频突发噪声."""
    import uhd
    usrp = uhd.usrp.MultiUSRP(usrp_args)
    usrp.set_tx_freq(uhd.types.TuneRequest(freq))
    usrp.set_tx_gain(gain)
    usrp.set_tx_rate(rate)
    usrp.set_clock_source('internal')
    usrp.set_time_source('internal')

    tx_stream = usrp.get_tx_stream(uhd.usrp.StreamArgs('fc32', 'sc16'))
    md = uhd.types.TXMetadata()
    md.start_of_burst = True

    print(f"[interferer] USRP TX: {freq/1e6:.1f}MHz, gain={gain}dB, "
          f"σ_b={sigma_b}, burst_prob={burst_prob}")

    try:
        while True:
            # 突发噪声 + 静默间隔
            burst = Interferer.generate_burst_noise(rate // 10, sigma_b)  # 100ms
            md.end_of_burst = False
            tx_stream.send(burst.astype(np.complex64), md)
            md.start_of_burst = False

            # 静默
            silence = np.zeros(rate // 10, dtype=np.complex64)  # 100ms
            tx_stream.send(silence, md)

    except KeyboardInterrupt:
        md.end_of_burst = True
        tx_stream.send(np.zeros(1, dtype=np.complex64), md)
        print("[interferer] 已停止")


def main():
    p = argparse.ArgumentParser(description='干扰器')
    p.add_argument('input', nargs='?', default='', help='输入IQ文件')
    p.add_argument('output', nargs='?', default='', help='输出IQ文件')
    p.add_argument('--mode', default='sim', choices=['sim', 'hardware'])
    p.add_argument('--snr-db', type=float, default=15.0)
    p.add_argument('--freq-offset', type=float, default=0.0)
    p.add_argument('--burst-prob', type=float, default=0.0, help='突发概率')
    p.add_argument('--sigma-b', type=float, default=1.0, help='突发噪声强度')
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--gain', type=float, default=50)
    p.add_argument('--rate', type=float, default=1e6)
    p.add_argument('--usrp-args', default='')
    args = p.parse_args()

    if args.mode == 'hardware':
        hardware_interferer(args.freq, args.gain, args.rate,
                            args.usrp_args, args.sigma_b, args.burst_prob)
    else:
        if not args.input or not args.output:
            print("仿真模式需要指定输入和输出文件")
            return
        tx = np.load(args.input)
        rx = Interferer.process(tx, snr_db=args.snr_db,
                                freq_offset=args.freq_offset,
                                burst_prob=args.burst_prob,
                                sigma_b=args.sigma_b,
                                samp_rate=args.rate)
        np.save(args.output, rx)
        print(f"[interferer] 已保存 {len(rx)} 样本 → {args.output}")


if __name__ == '__main__':
    main()
