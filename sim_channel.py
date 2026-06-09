#!/usr/bin/env python3
"""
sim_channel.py — 独立仿真信道工具

在 sender 和 receiver 之间充当"射频信道".
读取 sender 输出的 .npy 文件, 施加信道效应后输出供 receiver 消费.

信道模型:
  y = H * (x * e^(j2piDeltaf·t) * e^(jphi)) + w

  其中:
    H       = 多径信道 (可选的 FIR)
    Deltaf      = 载波频偏 (Hz)
    phi       = 初始相位偏移 (rad)
    w       = AWGN (由 SNR 控制)

用法:
  # 基础: 仅 AWGN
  python sim_channel.py tx_iq.npy rx_iq.npy --snr-db 10

  # 完整: AWGN + 频偏 + 相偏 + 多径
  python sim_channel.py tx_iq.npy rx_iq.npy --snr-db 10 --freq-offset 2000 \
         --phase-offset 0.5 --multipath "1.0,0.3@3,0.1@7"

SNR 说明:
  --snr-db 定义信号级 SNR:
    SNR_dB = 10*log10(信号功率 / 噪声功率)
"""

import argparse
import os
import sys

import numpy as np


class SimChannel:
    """仿真信道: AWGN + 频偏 + 相偏 + 多径."""

    def __init__(self, samp_rate: float = 1e6):
        self.samp_rate = samp_rate

    def process(self, tx_signal: np.ndarray, snr_db: float = 15.0,
                freq_offset: float = 0.0, phase_offset: float = 0.0,
                multipath: str = "") -> np.ndarray:
        """施加信道效应.

        Args:
            tx_signal: 发送基带信号 (complex64)
            snr_db: 信噪比 (dB)
            freq_offset: 频偏 (Hz)
            phase_offset: 初始相偏 (rad)
            multipath: 多径配置, 格式 "gain0,gain1@delay1,gain2@delay2,..."
        Returns:
            接收基带信号
        """
        rx = tx_signal.copy().astype(np.complex64)

        # 频偏
        if abs(freq_offset) > 0:
            t = np.arange(len(rx), dtype=np.float64) / self.samp_rate
            rx = rx * np.exp(1j * 2 * np.pi * freq_offset * t)

        # 相偏
        if abs(phase_offset) > 0:
            rx = rx * np.exp(1j * phase_offset)

        # 多径
        if multipath:
            taps = self._parse_multipath(multipath)
            if len(taps) > 1:
                rx = np.convolve(rx, taps, mode='full')[:len(rx)]

        # AWGN
        if snr_db < 100:
            signal_power = np.mean(np.abs(rx) ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = (np.sqrt(noise_power / 2)
                     * (np.random.randn(len(rx))
                        + 1j * np.random.randn(len(rx))))
            rx = rx + noise.astype(np.complex64)

        return rx.astype(np.complex64)

    @staticmethod
    def _parse_multipath(cfg: str) -> np.ndarray:
        """解析多径配置字符串 -> 信道冲激响应."""
        parts = cfg.split(',')
        max_delay = 0
        tap_info = []
        for part in parts:
            if '@' in part:
                gain_str, delay_str = part.split('@')
                gain = float(gain_str)
                delay = int(delay_str)
            else:
                gain = float(part)
                delay = 0
            tap_info.append((gain, delay))
            max_delay = max(max_delay, delay)

        h = np.zeros(max_delay + 1, dtype=np.complex64)
        for gain, delay in tap_info:
            h[delay] = gain

        h /= np.sqrt(np.sum(np.abs(h) ** 2))
        return h


# ======================================================================
# CLI
# ======================================================================
def main():
    p = argparse.ArgumentParser(
        description='仿真信道: AWGN + 频偏 + 相偏 + 多径')
    p.add_argument('input', help='输入 .npy 文件 (发送 IQ)')
    p.add_argument('output', help='输出 .npy 文件 (接收 IQ)')
    p.add_argument('--snr-db', type=float, default=15.0, help='信噪比 dB')
    p.add_argument('--freq-offset', type=float, default=0.0,
                   help='频偏 Hz (例如 2000 = 2kHz)')
    p.add_argument('--phase-offset', type=float, default=0.0,
                   help='初始相偏 rad')
    p.add_argument('--multipath', type=str, default='',
                   help='多径参数 "g0,g1@d1,g2@d2,..."')
    p.add_argument('--rate', type=float, default=1e6, help='采样率 Hz')
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"[sim_channel] 错误: 输入文件不存在 -> {args.input}")
        sys.exit(1)

    print(f"[sim_channel] ")
    print(f"  input:    {args.input}")
    print(f"  output:   {args.output}")
    print(f"  SNR:      {args.snr_db} dB")
    print(f"  频偏:     {args.freq_offset} Hz")
    print(f"  相偏:     {args.phase_offset} rad")
    if args.multipath:
        print(f"  多径:     {args.multipath}")

    # Support both .npy and raw binary (interleaved float32 I/Q)
    if args.input.endswith('.npy'):
        tx = np.load(args.input)
    else:
        tx = np.fromfile(args.input, dtype=np.complex64)
    print(f"  输入:     {len(tx)} 复样本 ({len(tx) / args.rate * 1000:.1f} ms)")

    ch = SimChannel(samp_rate=args.rate)
    rx = ch.process(tx, snr_db=args.snr_db, freq_offset=args.freq_offset,
                    phase_offset=args.phase_offset, multipath=args.multipath)

    # Output as raw binary (interleaved float32 I/Q) for C++ rx
    rx.astype(np.complex64).tofile(args.output)
    print(f"  输出:     {len(rx)} 复样本 -> {args.output}")
    print(f"[sim_channel] 完成")


if __name__ == '__main__':
    main()
