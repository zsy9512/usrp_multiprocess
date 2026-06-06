#!/usr/bin/env python3
"""
sender.py — BPSK PHY 发送端 (完全自包含)

帧结构 (352 符号 = 704 样本 @ sps=2):
  STF(48) + PSS(32) + RS(16) + Data(256)

  STF  = 3×16 重复 BPSK   → 延时相关粗捕获 (免疫频偏)
  PSS  = Zadoff-Chu u=25  → 精定时同步
  RS   = 已知 BPSK 导频   → 细频偏估计
  Data = 载荷 (当前: 随机比特, 预留上层极化码接口)

模式:
  --mode hardware   → USRP B210 实时发送
  --mode sim        → 输出 IQ 到文件 (供 sim_channel.py + receiver.py 仿真)

用法:
  # 硬件模式
  python sender.py --mode hardware --freq 2.45e9

  # 仿真模式 (保存 IQ 到文件)
  python sender.py --mode sim --sim-file tx_iq.npy --num-frames 2000

上层接口预留:
  get_info_bits(k) 回调, 当前为 np.random.randint(0, 2, 128)
  替换为 polar_encode() 即可接入极化码
"""

import argparse
import os
import sys
import time
from typing import Optional, Callable

import numpy as np

# ======================================================================
#  全部 PHY 函数内嵌 (零外部依赖, 完全自包含)
# ======================================================================

# ── 帧参数 ──
STF_REP  = 16
STF_NUM  = 3
PSS_LEN  = 32
RS_LEN   = 16
DATA_LEN = 256
FRAME_LEN = STF_REP * STF_NUM + PSS_LEN + RS_LEN + DATA_LEN  # 352
GUARD_SYMBOLS = 32  # 帧间保护间隔, 消除 RRC 泄漏
CODEWORD_LEN = 256
# USRP 突发参数 (参考原始 multiprocess)
GAP_SAMPLES = 100000  # 帧间100ms零填充, 消除UUU, 不需Python sleep
BURST_COPIES = 1
REPEAT_COUNT = 5  # 每帧重复次数 (提高接收成功率)


def _gen_stf() -> np.ndarray:
    rng = np.random.RandomState(7)
    base = 2 * rng.randint(0, 2, STF_REP) - 1
    return np.tile(base, STF_NUM).astype(np.complex64)


def _gen_pss() -> np.ndarray:
    n = np.arange(PSS_LEN)
    zc = np.exp(-1j * np.pi * 25 * n * (n + 1) / PSS_LEN)
    return zc.astype(np.complex64)


def _gen_rs() -> np.ndarray:
    rng = np.random.RandomState(13)
    return (2 * rng.randint(0, 2, RS_LEN) - 1).astype(np.complex64)


def _design_rrc(sps: int, rolloff: float = 0.35, num_sym: int = 10) -> np.ndarray:
    n_taps = num_sym * sps
    t = np.arange(-num_sym / 2, num_sym / 2, 1 / sps)
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 + rolloff * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * rolloff)) < 1e-12:
            h[i] = (rolloff / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff)))
        else:
            pi_t = np.pi * ti
            num = np.sin(pi_t * (1 - rolloff)) + 4 * rolloff * ti * np.cos(pi_t * (1 + rolloff))
            den = pi_t * (1 - (4 * rolloff * ti) ** 2)
            h[i] = num / den
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def _bpsk_mod(bits: np.ndarray) -> np.ndarray:
    return (1.0 - 2.0 * bits).astype(np.float32)


def _rrc_filter(symbols: np.ndarray, rrc: np.ndarray, sps: int) -> np.ndarray:
    """RRC 脉冲成形: 上采样 → 滤波 (mode='full', 保留全部样本)."""
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    return np.convolve(up, rrc, mode='full').astype(np.complex64)


# ======================================================================
#  帧序列 (程序生命周期内只生成一次)
# ======================================================================
STF_SYMS = _gen_stf()       # (48,) 符号级 BPSK
PSS      = _gen_pss()       # (32,)
RS       = _gen_rs()        # (16,)
RRC      = _design_rrc(2)   # (20*2+1,)


# ======================================================================
#  占位: 上层信息比特接口
# ======================================================================
def default_bit_source(n: int = CODEWORD_LEN) -> np.ndarray:
    """默认: 随机码字 (256 bits, 模拟极化码输出).

    替换为 polar_encode(info_bits, frozen_mask) 即可接入极化码.
    """
    return np.random.randint(0, 2, n).astype(np.int64)

def fixed_bit_source(n: int = CODEWORD_LEN) -> np.ndarray:
    """固定已知序列: 交替 0xAA, 0x55 模式, 用于物理层调试."""
    pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    return np.tile(pattern, n // 8 + 1)[:n]


# ======================================================================
#  Sender 核心类
# ======================================================================
class BpskPhySender:
    """BPSK PHY 发送端.

    用法:
        sender = BpskPhySender(samp_rate=1e6, sps=2)
        sender.start(mode='hardware', freq=2.45e9, gain=50,
                     frame_interval=0.002)
    """

    def __init__(self, samp_rate: float = 1e6, sps: int = 2,
                 bit_source: Optional[Callable] = None):
        self.samp_rate = samp_rate
        self.sps = sps
        self.bit_source = bit_source or default_bit_source
        self.running = False
        self.usrp = None
        self.tx_stream = None

    # ── 公共接口 ──

    def start(self, mode: str = 'sim', freq: float = 2.45e9, gain: float = 60,
              frame_interval: float = 0.001, num_frames: int = 0,
              sim_file: str = 'tx_iq.npy', usrp_args: str = '',
              save_bits: bool = False, repeat: int = 1, fixed_seq: bool = False):
        """启动发送.

        Args:
            mode: 'hardware' | 'sim'
            freq: 中心频率 (Hz), 仅 hardware 模式
            gain: 发射增益 (dB), 仅 hardware 模式
            frame_interval: 帧间间隔 (秒), 0=最快
            num_frames: 发送帧数, 0=无限
            sim_file: 仿真模式输出文件 (.npy)
            usrp_args: UHD 设备参数, 如 "name=MyB210"
        """
        self.running = True
        frame_count = 0

        if mode == 'hardware':
            self._init_usrp(freq, gain, usrp_args)

        # 仿真模式: 预分配全部帧
        tx_buf = [] if mode == 'sim' else None
        all_bits = [] if save_bits else None

        print(f"[sender] mode={mode}  rate={self.samp_rate/1e6:.1f}Msps  "
              f"sps={self.sps}  interval={frame_interval*1000:.2f}ms")
        payload_syms = PSS_LEN + RS_LEN + DATA_LEN + GUARD_SYMBOLS
        payload_samples = payload_syms * self.sps + len(RRC) - 1
        print(f"[sender] frame={payload_syms}sym ({payload_samples}samples)  "
              f"air_time={payload_samples/self.samp_rate*1000:.3f}ms")

        if fixed_seq:
            self.bit_source = fixed_bit_source
            print("[sender] 使用固定测试序列 (0xAA模式)")

        t_start = time.time()
        gap_zeros = np.zeros(GAP_SAMPLES, dtype=np.complex64)

        # 硬件模式: 创建一次 TXMetadata, 持续流式发送
        if mode == 'hardware':
            import uhd
            tx_md = uhd.types.TXMetadata()
            tx_md.start_of_burst = True
            tx_md.end_of_burst = False

        try:
            while self.running:
                # ── 1. 获取码字比特 (256 bits, 上层极化码接口) ──
                codeword = self.bit_source(CODEWORD_LEN)

                # ── 2. BPSK 调制 → 256 符号 ──
                data_syms = _bpsk_mod(codeword)
                if len(data_syms) < DATA_LEN:
                    pad = np.zeros(DATA_LEN - len(data_syms), dtype=np.float32)
                    data_syms = np.concatenate([data_syms, pad])
                elif len(data_syms) > DATA_LEN:
                    data_syms = data_syms[:DATA_LEN]

                # ── 3. 帧: PSS+RS+Data+Guard 经 RRC 脉冲成形 ──
                frame_syms = np.concatenate([PSS, RS, data_syms,
                                              np.zeros(GUARD_SYMBOLS, dtype=np.complex64)])
                tx_signal = _rrc_filter(frame_syms, RRC, self.sps).astype(np.complex64)

                # ── 4. 重复发送当前帧 repeat 次 ──
                for ri in range(repeat):
                    if mode == 'hardware':
                        self.tx_stream.send(tx_signal, tx_md)
                        tx_md.start_of_burst = False
                        self.tx_stream.send(gap_zeros, tx_md)
                    else:
                        tx_buf.append(tx_signal)
                    if save_bits:
                        all_bits.append(codeword)

                frame_count += 1
                if num_frames > 0 and frame_count >= num_frames:
                    break
                # 硬件模式: gap 控制帧率, 无需 sleep
                if mode != 'hardware' and frame_interval > 0:
                    time.sleep(frame_interval)

            # ── 关闭发射链 ──
            if mode == 'hardware':
                tx_md.end_of_burst = True
                self.tx_stream.send(np.zeros(1, dtype=np.complex64), tx_md)

        except KeyboardInterrupt:
            print("\n[sender] 用户中断")

        finally:
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[sender] 完成: {frame_count} 帧, {elapsed:.1f}s, {fps:.1f} fps")

            if mode == 'hardware':
                self._close_usrp()
            else:
                # 保存 IQ 文件
                tx_iq = np.concatenate(tx_buf)
                np.save(sim_file, tx_iq)
                print(f"[sender] 已保存 {len(tx_iq)} 样本 → {sim_file}")
                # 保存发送比特 (用于 BER 计算)
                if save_bits and all_bits:
                    bits_file = sim_file.replace('.npy', '_bits.npy')
                    np.save(bits_file, np.concatenate(all_bits))
                    print(f"[sender] 已保存 {len(np.concatenate(all_bits))} 比特 → {bits_file}")

    def stop(self):
        self.running = False

    # ── UHD 硬件接口 ──

    def _init_usrp(self, freq: float, gain: float, usrp_args: str):
        import uhd
        self.usrp = uhd.usrp.MultiUSRP(usrp_args)
        self.usrp.set_tx_freq(uhd.types.TuneRequest(freq))
        self.usrp.set_tx_gain(gain)
        self.usrp.set_tx_rate(self.samp_rate)
        # 时钟初始化 (PC 纳秒时间)
        pc_ns = time.time_ns()
        tspec = uhd.types.TimeSpec(pc_ns // 1_000_000_000,
                                   (pc_ns % 1_000_000_000) / 1e9)
        self.usrp.set_time_now(tspec)
        self.usrp.set_clock_source('internal')
        self.usrp.set_time_source('internal')
        # 创建发送流
        args = uhd.usrp.StreamArgs('fc32', 'sc16')
        args.channels = [0]
        self.tx_stream = self.usrp.get_tx_stream(args)
        print(f"[sender] USRP TX: {freq/1e6:.1f}MHz, gain={gain}dB")

    def _send_usrp_burst(self, burst_signals, is_first, is_last):
        """发送 USRP 突发.

        Args:
            burst_signals: list of ndarray, 每个元素是一帧+RRC的完整信号
            is_first: 是否是整个传输的第一个 burst (设置 start_of_burst)
            is_last:  是否是整个传输的最后一个 burst (设置 end_of_burst)
        """
        import uhd
        md = uhd.types.TXMetadata()
        md.start_of_burst = is_first
        md.end_of_burst = False

        gap = np.zeros(GAP_SAMPLES, dtype=np.complex64)

        for i, sig in enumerate(burst_signals):
            # 发送帧
            md.end_of_burst = False
            self.tx_stream.send(sig.astype(np.complex64), md, timeout=0.1)
            md.start_of_burst = False

            # 发送间隙 (维持 USRP 发射链, 避免瞬态)
            is_last_sig = (i == len(burst_signals) - 1) and is_last
            md.end_of_burst = is_last_sig
            self.tx_stream.send(gap, md, timeout=0.1)

    def _close_usrp(self):
        import uhd
        if self.tx_stream is not None:
            md = uhd.types.TXMetadata()
            md.end_of_burst = True
            try:
                self.tx_stream.send(np.zeros(1, dtype=np.complex64), md)
            except Exception:
                pass
        if self.usrp is not None:
            self.usrp = None
            self.tx_stream = None
        print("[sender] USRP 已关闭")


# ======================================================================
#  CLI
# ======================================================================
def main():
    p = argparse.ArgumentParser(description='BPSK PHY 发送端 (自包含)')
    p.add_argument('--mode', default='sim', choices=['hardware', 'sim'])
    p.add_argument('--freq', type=float, default=915e6, help='中心频率 Hz')
    p.add_argument('--gain', type=float, default=60, help='发射增益 dB')
    p.add_argument('--rate', type=float, default=1e6, help='采样率 Hz')
    p.add_argument('--sps', type=int, default=2, help='每符号采样数')
    p.add_argument('--interval', type=float, default=0.002, help='帧间隔 s')
    p.add_argument('--num-frames', type=int, default=0, help='帧数 (0=无限)')
    p.add_argument('--sim-file', default='tx_iq.npy', help='仿真输出文件')
    p.add_argument('--save-bits', action='store_true', help='保存发送比特用于 BER')
    p.add_argument('--repeat', type=int, default=5, help='每帧重复次数')
    p.add_argument('--fixed-seq', action='store_true', help='使用固定测试序列(0xAA)')
    p.add_argument('--usrp-args', default='', help='UHD 参数')
    args = p.parse_args()

    sender = BpskPhySender(samp_rate=args.rate, sps=args.sps)
    sender.start(
        mode=args.mode,
        freq=args.freq,
        gain=args.gain,
        frame_interval=args.interval,
        num_frames=args.num_frames,
        sim_file=args.sim_file,
        save_bits=args.save_bits,
        repeat=args.repeat,
        fixed_seq=args.fixed_seq,
        usrp_args=args.usrp_args,
    )


if __name__ == '__main__':
    main()
