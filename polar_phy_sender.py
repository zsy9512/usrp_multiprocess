#!/usr/bin/env python3
"""
polar_phy_sender.py — 极化码 + BPSK PHY 融合发送端

链路:
  Info bits (K=128) → Polar编码 (N=256) → BPSK调制 → 成帧 → USRP/文件

用法:
  仿真: python polar_phy_sender.py --mode sim --num-frames 500 --sim-file test.npy
  硬件: python polar_phy_sender.py --mode hardware --freq 915e6 --gain 70
"""

import argparse, os, sys, time
import numpy as np

# ── 导入 PHY TX 组件 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sender import (BpskPhySender, PSS, RS, RRC, GUARD_SYMBOLS,
                    _rrc_filter, _bpsk_mod, _design_rrc)

# ── 内置极化码编码 (无需 torch) ──
def _polar_encode(u):
    N = u.shape[0]
    cw = u.copy().ravel()
    for stage in range(1, int(np.log2(N)) + 1):
        sep = N // (1 << stage)
        for j in range(N):
            if (j // sep) % 2 == 0:
                cw[j] = (cw[j] + cw[j + sep]) % 2
    return cw

def _build_codeword(info_bits, frozen_mask):
    N = frozen_mask.shape[0]
    u = np.zeros(N, dtype=np.int64)
    u[frozen_mask.astype(bool)] = info_bits.ravel()
    return _polar_encode(u)

# 加载冻结比特掩膜
MATRICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deploy', 'matrices')
A_PATH = os.path.join(MATRICES_DIR, 'A.npy')
if not os.path.isfile(A_PATH):
    raise FileNotFoundError(f"冻结比特掩膜未找到: {A_PATH}")
FROZEN_MASK = np.load(A_PATH).squeeze()  # (N,) 1=信息位, 0=冻结位
K = int(FROZEN_MASK.sum())  # 128
N = FROZEN_MASK.shape[0]    # 256

class PolarPhySender:
    """极化码 + PHY 融合发送端."""

    def __init__(self):
        self.phy = BpskPhySender(samp_rate=1e6, sps=2)
        self.phy.bit_source = self._polar_bit_source

    def _polar_bit_source(self, n=256):
        """上层接口: 随机信息比特 → Polar编码 → 256码字."""
        info_bits = np.random.randint(0, 2, K).astype(np.int64)
        codeword = _build_codeword(info_bits, FROZEN_MASK)
        return codeword

    def start(self, mode='sim', freq=915e6, gain=70, rate=1e6,
              interval=0, num_frames=100, sim_file='tx_polar.npy',
              usrp_args='', save_bits=False):
        # 保存信息比特 (用于BER)
        self._info_bits_all = [] if save_bits else None
        self._save_bits = save_bits

        # 包装 start 以保存信息比特
        original_source = self.phy.bit_source
        def wrapped_source(n=256):
            info = np.random.randint(0, 2, K).astype(np.int64)
            if self._save_bits:
                self._info_bits_all.append(info.copy())
            cw = _build_codeword(info, FROZEN_MASK)
            return cw
        self.phy.bit_source = wrapped_source

        self.phy.start(mode=mode, freq=freq, gain=gain,
                       frame_interval=interval, num_frames=num_frames,
                       sim_file=sim_file, usrp_args=usrp_args,
                       save_bits=save_bits)

        # 保存信息比特到文件
        if save_bits and self._info_bits_all:
            bits_file = sim_file.replace('.npy', '_info_bits.npy')
            np.save(bits_file, np.concatenate(self._info_bits_all))
            print(f"[sender] 已保存 {len(np.concatenate(self._info_bits_all))} "
                  f"信息比特 → {bits_file}")


def main():
    p = argparse.ArgumentParser(description='极化码-PHY 融合发送端')
    p.add_argument('--mode', default='sim', choices=['hardware', 'sim'])
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--gain', type=float, default=70)
    p.add_argument('--rate', type=float, default=1e6)
    p.add_argument('--interval', type=float, default=0, help='帧间隔(0=gap控制)')
    p.add_argument('--num-frames', type=int, default=100)
    p.add_argument('--sim-file', default='tx_polar.npy')
    p.add_argument('--save-bits', action='store_true', help='保存信息比特')
    p.add_argument('--usrp-args', default='')
    args = p.parse_args()

    sender = PolarPhySender()
    sender.start(mode=args.mode, freq=args.freq, gain=args.gain,
                 rate=args.rate, interval=args.interval,
                 num_frames=args.num_frames, sim_file=args.sim_file,
                 usrp_args=args.usrp_args, save_bits=args.save_bits)

if __name__ == '__main__':
    main()
