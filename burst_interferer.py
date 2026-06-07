#!/usr/bin/env python3
"""
burst_interferer.py — 突发干扰源 (B210 UHD 直控)

信道模型:  y_i = s_i + n_i + rho_i * omega_i
  n_i   ~ N(0, sigma^2)          — 背景高斯噪声 (AWGN)
  omega_i ~ N(0, sigma_b^2)      — 突发干扰分量, sigma_b^2 >> sigma^2
  rho_i ~ Bernoulli(p_b)         — 突发指示变量

发射策略:
  rho_i=0 (非突发): 极低幅度 CW 正弦波 (平稳, 幅度 ~sigma_bg)
  rho_i=1 (突发):   复高斯噪声 N(0,sigma_b) + j*N(0,sigma_b)

用法:
  python burst_interferer.py --freq 915e6 --sigma-b 3.0 --p-b 0.05
  python burst_interferer.py --serial MyB210_02 --gain 60 --duration 30
"""
from __future__ import annotations

import argparse, os, sys, time
import numpy as np

SPS = 2                # 每符号采样数
SAMP_RATE = 1e6        # 默认采样率

# ======================================================================
# BurstInterferer
# ======================================================================

class BurstInterferer:
    """突发干扰源: 逐符号 Bernoulli 判定, UHD 连续发射."""

    def __init__(self, samp_rate: float = SAMP_RATE, sps: int = SPS,
                 p_b: float = 0.05, sigma_b: float = 2.0,
                 sigma_bg: float = 0.001):
        self.samp_rate = samp_rate
        self.sps = sps
        self.p_b = p_b
        self.sigma_b = sigma_b
        self.sigma_bg = sigma_bg
        self.running = False
        self.usrp = None
        self.tx_stream = None

    # ------------------------------------------------------------------
    def _generate_chunk(self, n_samples: int) -> tuple[np.ndarray, int]:
        """生成一段带突发干扰的基带样本.

        每 sps 个样本为一个符号周期, 独立 Bernoulli(p_b) 判定.

        Returns:
            (chunk, burst_sym_count)
        """
        nsym = n_samples // self.sps
        rho = np.random.rand(nsym) < self.p_b
        n_burst = int(np.sum(rho))

        # 非突发: 极低幅度 CW 正弦波 (平稳信号)
        t = np.arange(n_samples, dtype=np.float32) / self.samp_rate
        bg_signal = self.sigma_bg * np.exp(1j * 2 * np.pi * 10e3 * t)

        # 突发: 复高斯噪声
        burst_noise = (np.random.randn(n_samples).astype(np.float32)
                       + 1j * np.random.randn(n_samples).astype(np.float32))
        burst_noise *= self.sigma_b

        # 逐符号组装: 突发符号覆盖为噪声
        chunk = bg_signal.astype(np.complex64).copy()
        for i in range(nsym):
            if rho[i]:
                s0 = i * self.sps
                s1 = s0 + self.sps
                chunk[s0:s1] = burst_noise[s0:s1]

        return chunk, n_burst

    # ------------------------------------------------------------------
    def start(self, freq: float = 915e6, gain: float = 50,
              serial: str = 'MyB210_01', duration: float = 0.0):
        """启动突发干扰发射.

        Args:
            freq:     中心频率 (Hz)
            gain:     TX 增益 (dB)
            serial:   USRP 序列号
            duration: 总时长 (s), 0=无限
        """
        self.running = True

        import uhd
        dev = f'serial={serial}' if serial else ''
        self.usrp = uhd.usrp.MultiUSRP(dev)
        self.usrp.set_tx_freq(uhd.types.TuneRequest(freq))
        self.usrp.set_tx_gain(gain)
        self.usrp.set_tx_rate(self.samp_rate)
        self.usrp.set_tx_bandwidth(self.samp_rate)
        self.usrp.set_tx_antenna("TX/RX")
        self.usrp.set_clock_source("internal")
        self.usrp.set_time_source("internal")

        ns = time.time_ns()
        self.usrp.set_time_now(uhd.types.TimeSpec(
            ns // 1_000_000_000, (ns % 1_000_000_000) / 1e9))

        stream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
        stream_args.channels = [0]
        self.tx_stream = self.usrp.get_tx_stream(stream_args)

        CHUNK = 4096
        tx_md = uhd.types.TXMetadata()
        tx_md.start_of_burst = True

        sym_period = self.sps / self.samp_rate
        burst_sym_count = 0
        total_sym_count = 0

        print(f"[burst_interferer] USRP={serial}  freq={freq/1e6:.3f}MHz  "
              f"gain={gain:.0f}dB  rate={self.samp_rate/1e6:.1f}Msps",
              flush=True)
        print(f"[burst_interferer] p_b={self.p_b}  sigma_b={self.sigma_b}  "
              f"sigma_bg={self.sigma_bg}  sym_period={sym_period*1e6:.0f}us",
              flush=True)
        print(f"[burst_interferer] 发射中... (Ctrl+C 停止)", flush=True)

        t_start = time.time()
        try:
            while self.running:
                chunk, n_burst = self._generate_chunk(CHUNK)
                self.tx_stream.send(chunk, tx_md)
                tx_md.start_of_burst = False

                total_sym_count += CHUNK // self.sps
                burst_sym_count += n_burst

                if duration > 0 and (time.time() - t_start) >= duration:
                    break

        except KeyboardInterrupt:
            print("\n[burst_interferer] 用户中断", flush=True)

        finally:
            # 发送 EOB
            eob_md = uhd.types.TXMetadata()
            eob_md.end_of_burst = True
            try:
                self.tx_stream.send(np.zeros(1, dtype=np.complex64), eob_md)
            except Exception:
                pass

            elapsed = time.time() - t_start
            actual_p = burst_sym_count / max(total_sym_count, 1) * 100
            print(f"\n--- 结果 ---", flush=True)
            print(f"  发送时长: {elapsed:.1f}s", flush=True)
            print(f"  总符号数: {total_sym_count}", flush=True)
            print(f"  突发符号: {burst_sym_count} ({actual_p:.1f}%)", flush=True)
            print(f"  sigma_b={self.sigma_b}  sigma_bg={self.sigma_bg}", flush=True)

    def stop(self):
        self.running = False


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description='突发干扰源 (B210 UHD 直控)')
    p.add_argument('--serial', default='320F2BD', help='USRP 序列号')
    p.add_argument('--freq', type=float, default=915e6, help='中心频率 (Hz)')
    p.add_argument('--gain', type=float, default=50, help='TX 增益 (dB)')
    p.add_argument('--rate', type=float, default=SAMP_RATE, help='采样率 (Hz)')
    p.add_argument('--sps', type=int, default=SPS, help='每符号采样数')
    p.add_argument('--p-b', type=float, default=0.05, help='突发概率')
    p.add_argument('--sigma-b', type=float, default=2.0,
                   help='突发噪声标准差')
    p.add_argument('--sigma-bg', type=float, default=0.001,
                   help='背景平稳信号幅度')
    p.add_argument('--duration', type=float, default=0.0,
                   help='总时长 (s), 0=无限')
    args = p.parse_args()

    interferer = BurstInterferer(
        samp_rate=args.rate, sps=args.sps,
        p_b=args.p_b, sigma_b=args.sigma_b, sigma_bg=args.sigma_bg)

    interferer.start(
        freq=args.freq, gain=args.gain,
        serial=args.serial, duration=args.duration)


if __name__ == '__main__':
    main()
