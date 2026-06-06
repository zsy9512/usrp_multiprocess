#!/usr/bin/env python3
"""
sender.py — BPSK PHY 发送端 (完整帧结构)

帧结构 (符号域):
  STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)

  STF   = 4×16 重复 BPSK  → 粗检测 + 粗 CFO
  PSS   = Zadoff-Chu u=25 → 精定时
  RS    = 已知 BPSK 导频  → 细 CFO + 相位 + 信道估计
  Header= 预留 + CRC16    → 帧控制
  Payload = 数据比特
  CRC   = Payload CRC16   → 帧正确性
  Guard = 零符号          → 滤波尾巴

流水线:
  Info bits → Polar编码 → BPSK → 成帧 → RRC → 发送/保存

用法:
  仿真: python sender.py --mode sim --num-frames 200 --sim-file tx_iq.npy
  硬件: python sender.py --mode hardware --freq 915e6 --gain 30
"""
from __future__ import annotations

import argparse, os, sys, time
from typing import Callable, Optional
import numpy as np

from phy_params import (
    SPS, STF, PSS, RS, RRC, STF_LEN, PSS_LEN, RS_LEN,
    HEADER_LEN, PAYLOAD_LEN, PAYLOAD_CRC_LEN, GUARD_SYMBOLS,
    FRAME_SYMBOLS, INFO_BITS,
    crc16, bits_to_bytes, bytes_to_bits,
)

# ======================================================================
# Polar 编码 (自包含, 无 torch 依赖)
# ======================================================================

def _polar_encode(u: np.ndarray) -> np.ndarray:
    """Arikan polar transform.  u: (N,) {0,1} → cw: (N,) {0,1}."""
    N = u.shape[0]
    cw = u.copy().ravel()
    n_stages = int(np.log2(N))
    for stage in range(1, n_stages + 1):
        sep = N // (1 << stage)
        for j in range(N):
            if (j // sep) % 2 == 0:
                cw[j] = (cw[j] + cw[j + sep]) % 2
    return cw


def _build_codeword(info_bits: np.ndarray, frozen_mask: np.ndarray) -> np.ndarray:
    N = frozen_mask.shape[0]
    u = np.zeros(N, dtype=np.int64)
    u[frozen_mask.astype(bool)] = info_bits.ravel()
    return _polar_encode(u)


# ======================================================================
# 帧打包
# ======================================================================

def _bpsk(bits: np.ndarray) -> np.ndarray:
    """{0,1} → {+1,-1} BPSK."""
    return (1.0 - 2.0 * bits).astype(np.float32)


def build_frame(data_bits: np.ndarray) -> np.ndarray:
    """构建一帧的基带符号 (符号域).

    Args:
        data_bits: (PAYLOAD_LEN,) {0,1} 数据比特
    Returns:
        (FRAME_SYMBOLS,) complex64
    """
    assert len(data_bits) == PAYLOAD_LEN, f"data_bits len={len(data_bits)} != {PAYLOAD_LEN}"

    # --- Payload CRC ---
    payload_bytes = bits_to_bytes(data_bits)
    payload_crc = crc16(payload_bytes)
    crc_bits = bytes_to_bits(
        np.array([(payload_crc >> 8) & 0xFF, payload_crc & 0xFF], dtype=np.uint8), 16)

    # --- Header (预留 + CRC16) ---
    # Header 前 16 bit 为预留字段, 后 16 bit 为 Header CRC
    header_reserved = np.zeros(16, dtype=np.int64)
    header_bytes = bits_to_bytes(header_reserved)
    header_crc = crc16(header_bytes)
    header_crc_bits = bytes_to_bits(
        np.array([(header_crc >> 8) & 0xFF, header_crc & 0xFF], dtype=np.uint8), 16)
    header_bits = np.concatenate([header_reserved, header_crc_bits])

    # --- 符号域帧 ---
    stf_syms  = STF.astype(np.complex64)
    pss_syms  = PSS.astype(np.complex64)
    rs_syms   = RS.astype(np.complex64)
    hdr_syms  = _bpsk(header_bits).astype(np.complex64)
    data_syms = _bpsk(data_bits).astype(np.complex64)
    crc_syms  = _bpsk(crc_bits).astype(np.complex64)
    guard     = np.zeros(GUARD_SYMBOLS, dtype=np.complex64)

    frame = np.concatenate([stf_syms, pss_syms, rs_syms,
                            hdr_syms, data_syms, crc_syms, guard])
    return frame


def rrc_filter(symbols: np.ndarray, rrc: np.ndarray, sps: int) -> np.ndarray:
    """RRC 脉冲成形: 上采样 → 滤波."""
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    return np.convolve(up, rrc, mode='full').astype(np.complex64)


# ======================================================================
# 默认比特源
# ======================================================================

FROZEN_MASK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'deploy', 'matrices', 'A.npy')

_FROZEN = None
if os.path.isfile(FROZEN_MASK_PATH):
    _FROZEN = np.load(FROZEN_MASK_PATH).squeeze()
    _K = int(_FROZEN.sum())
else:
    _K = INFO_BITS


def default_bit_source(n: int = PAYLOAD_LEN) -> np.ndarray:
    """默认: 随机信息比特 → Polar 编码 → 码字.

    若无 frozen mask, 直接返回随机比特.
    """
    if _FROZEN is not None:
        info = np.random.randint(0, 2, _K).astype(np.int64)
        return _build_codeword(info, _FROZEN)
    else:
        return np.random.randint(0, 2, n).astype(np.int64)


def fixed_bit_source(n: int = PAYLOAD_LEN) -> np.ndarray:
    """固定已知序列: 交替 0xAA, 0x55 模式, 用于物理层调试."""
    pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    return np.tile(pattern, n // 8 + 1)[:n]


# ======================================================================
# Sender 类
# ======================================================================

class BpskPhySender:
    """BPSK PHY 发送端 (完整帧结构)."""

    def __init__(self, samp_rate: float = 1e6, sps: int = SPS,
                 bit_source: Optional[Callable] = None):
        self.samp_rate = samp_rate
        self.sps = sps
        self.bit_source = bit_source or default_bit_source
        self.running = False
        self.usrp = None
        self.tx_stream = None

    def start(self, mode: str = 'sim', freq: float = 915e6, gain: float = 60,
              frame_interval: float = 0.002, num_frames: int = 0,
              sim_file: str = 'tx_iq.npy', usrp_args: str = '',
              save_bits: bool = False, repeat: int = 1, fixed_seq: bool = False,
              tx_delay_s: float = 1.0):
        """启动发送.

        Args:
            mode: 'hardware' | 'sim'
            freq: 中心频率 (Hz)
            gain: TX 增益 (dB)
            frame_interval: 帧间隔 (s), 0=最快
            num_frames: 帧数, 0=无限
            sim_file: 仿真输出 .npy
            save_bits: 保存发送比特
            repeat: 每帧重复次数
            fixed_seq: 使用固定测试序列
            tx_delay_s: 硬件模式发射前延迟 (等待 RX 就绪)
        """
        self.running = True
        frame_count = 0
        iq_list = [] if mode == 'sim' else None
        bits_list = [] if save_bits else None

        if fixed_seq:
            self.bit_source = fixed_bit_source
            print("[sender] 使用固定测试序列 (0xAA模式)")

        # --- 预计算 ---
        rrc = RRC
        frame_samples = FRAME_SYMBOLS * self.sps + len(rrc) - 1
        air_time_ms = frame_samples / self.samp_rate * 1000

        print(f"[sender] mode={mode}  rate={self.samp_rate/1e6:.1f}Msps  "
              f"sps={self.sps}  interval={frame_interval*1000:.2f}ms")
        print(f"[sender] frame={FRAME_SYMBOLS}sym → {frame_samples}samples  "
              f"air_time={air_time_ms:.3f}ms")

        # --- 硬件初始化 ---
        if mode == 'hardware':
            import uhd
            self._init_usrp(freq, gain, usrp_args)
            print(f"[sender] 等待 {tx_delay_s}s 后发射...")
            time.sleep(tx_delay_s)
            tx_md = uhd.types.TXMetadata()
            tx_md.start_of_burst = True
            tx_md.end_of_burst = False

        t_start = time.time()
        try:
            while self.running:
                # --- 1. 生成数据比特 ---
                data_bits = self.bit_source(PAYLOAD_LEN)

                # --- 2. 构建帧 (符号域) ---
                frame_syms = build_frame(data_bits)

                # --- 3. RRC 脉冲成形 ---
                tx_signal = rrc_filter(frame_syms, rrc, self.sps)

                # --- 4. 重复发送 (repeat 次) ---
                for ri in range(repeat):
                    if mode == 'hardware':
                        self.tx_stream.send(tx_signal.astype(np.complex64), tx_md)
                        tx_md.start_of_burst = False
                        # 帧间间隔
                        gap_len = max(16, int(frame_interval * self.samp_rate))
                        gap = np.zeros(gap_len, dtype=np.complex64)
                        tx_gap_md = uhd.types.TXMetadata()
                        tx_gap_md.start_of_burst = False
                        tx_gap_md.end_of_burst = False
                        self.tx_stream.send(gap, tx_gap_md)
                    else:
                        iq_list.append(tx_signal)
                        if save_bits:
                            bits_list.append(data_bits)

                frame_count += 1
                if num_frames > 0 and frame_count >= num_frames:
                    break
                if mode != 'hardware' and frame_interval > 0:
                    time.sleep(frame_interval)

            # --- 关闭发射链 ---
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
                tx_iq = np.concatenate(iq_list) if iq_list else np.array([], dtype=np.complex64)
                np.save(sim_file, tx_iq)
                print(f"[sender] 已保存 {len(tx_iq)} 样本 → {sim_file}")
                if save_bits and bits_list:
                    bits_file = sim_file.replace('.npy', '_bits.npy')
                    np.save(bits_file, np.concatenate(bits_list))
                    print(f"[sender] 已保存发送比特 → {bits_file}")

    def stop(self):
        self.running = False

    def _init_usrp(self, freq: float, gain: float, usrp_args: str = ''):
        import uhd
        self.usrp = uhd.usrp.MultiUSRP(usrp_args)
        actual_freq = self.usrp.set_tx_freq(uhd.types.TuneRequest(freq))
        actual_gain = self.usrp.set_tx_gain(gain)
        actual_rate = self.usrp.set_tx_rate(self.samp_rate)
        actual_bw   = self.usrp.set_tx_bandwidth(self.samp_rate)
        self.usrp.set_tx_antenna("TX/RX")
        self.usrp.set_clock_source("internal")
        self.usrp.set_time_source("internal")
        pc_ns = time.time_ns()
        tspec = uhd.types.TimeSpec(pc_ns // 1_000_000_000,
                                   (pc_ns % 1_000_000_000) / 1e9)
        self.usrp.set_time_now(tspec)
        args = uhd.usrp.StreamArgs('fc32', 'sc16')
        args.channels = [0]
        self.tx_stream = self.usrp.get_tx_stream(args)
        print(f"[sender] USRP TX: freq={actual_freq:.6e}Hz  "
              f"gain={actual_gain:.1f}dB  rate={actual_rate:.6e}  "
              f"bw={actual_bw:.6e}")

    def _close_usrp(self):
        import uhd
        if self.tx_stream is not None:
            md = uhd.types.TXMetadata()
            md.end_of_burst = True
            try:
                self.tx_stream.send(np.zeros(1, dtype=np.complex64), md)
            except Exception:
                pass
        self.usrp = None
        self.tx_stream = None
        print("[sender] USRP 已关闭")


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description='BPSK PHY 发送端')
    p.add_argument('--mode', default='sim', choices=['hardware', 'sim'])
    p.add_argument('--freq', type=float, default=915e6, help='中心频率 Hz')
    p.add_argument('--gain', type=float, default=60, help='发射增益 dB')
    p.add_argument('--rate', type=float, default=1e6, help='采样率 Hz')
    p.add_argument('--sps', type=int, default=SPS, help='每符号采样数')
    p.add_argument('--interval', type=float, default=0.002, help='帧间隔 s')
    p.add_argument('--num-frames', type=int, default=0, help='帧数 (0=无限)')
    p.add_argument('--sim-file', default='tx_iq.npy', help='仿真输出文件')
    p.add_argument('--save-bits', action='store_true', help='保存发送比特')
    p.add_argument('--repeat', type=int, default=1, help='每帧重复次数')
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
