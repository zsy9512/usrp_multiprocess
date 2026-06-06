#!/usr/bin/env python3
"""
polar_phy_receiver.py — 极化码 + BPSK PHY 融合接收端

链路:
  USRP/文件 → RRC匹配 → RS同步 → PSS频偏校正 → LLR → SGNN译码 → 信息比特

用法:
  仿真: python polar_phy_receiver.py --mode sim --sim-file test_rx.npy --tx-info-bits info_bits.npy
  硬件: python polar_phy_receiver.py --mode hardware --freq 915e6 --gain 40
"""

import argparse, os, sys, time, importlib.util
import numpy as np
import torch

# ── 导入 PHY RX 组件 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from receiver import (BpskPhyReceiver, _rrc_match, REF_PSS, REF_RS, RRC_TX,
                      PSS_LEN, RS_LEN, DATA_LEN, FRAME_SYMBOLS)

# ── 导入 SGNN 译码器 (从 deploy/ 子目录, 用文件路径避名冲突) ──
deploy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deploy')
spec = importlib.util.spec_from_file_location('deploy_receiver',
    os.path.join(deploy_dir, 'receiver.py'))
sgnn_receiver = importlib.util.module_from_spec(spec)
sys.modules['deploy_receiver'] = sgnn_receiver
spec.loader.exec_module(sgnn_receiver)

MATRICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deploy', 'matrices')
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deploy', 'checkpoint')
PCM_PATH = os.path.join(MATRICES_DIR, 'pcm.npy')
CKPT_PATH = os.path.join(CKPT_DIR, 'polar_GNN_20_iter_0_epoches_13.pt')


def _polar_encode(u):
    """Arikan polar transform (自逆: G = G^(-1))."""
    N = u.shape[0]
    cw = u.copy().ravel()
    for stage in range(1, int(np.log2(N)) + 1):
        sep = N // (1 << stage)
        for j in range(N):
            if (j // sep) % 2 == 0:
                cw[j] = (cw[j] + cw[j + sep]) % 2
    return cw


class PolarPhyReceiver(BpskPhyReceiver):
    """极化码 + PHY 融合接收端."""

    def __init__(self, samp_rate=1e6, sps=2, device='cpu'):
        super().__init__(samp_rate=samp_rate, sps=sps)
        self.device = device
        self._iq_buffer = None  # 硬件模式 IQ 保存缓冲

        # 加载 SGNN 模型和 Tanner 图
        print(f"[receiver] 加载 SGNN 译码器 (device={device})...")
        self.sgnn_model, cfg = sgnn_receiver.load_model(CKPT_PATH, device=device)
        self.graph = sgnn_receiver.build_graph(PCM_PATH, device=device)
        self.N = self.graph['N']      # 256
        self.N_hat = self.graph['N_hat']
        self.K = self.graph['K']      # 128
        print(f"[receiver] SGNN: N={self.N}, K={self.K}, "
              f"nstate={cfg['nstate']}, lstu={cfg['lstu_mode']}")

        # 冻结比特掩膜 (用于译码后提取信息位)
        A_PATH = os.path.join(MATRICES_DIR, 'A.npy')
        self.frozen_mask = np.load(A_PATH).squeeze() if os.path.isfile(A_PATH) else None

        # BER 统计 (信息位)
        self.info_total_bits = 0
        self.info_total_errors = 0

    def _process_window(self, tx_bits=None):
        """重写: PHY同步 → LLR → SGNN译码 → 信息位BER."""
        r = self.win[:self.win_len]
        if self.win_len < FRAME_SYMBOLS * self.sps:
            return

        symbols = _rrc_match(r, RRC_TX, self.sps)

        # ── ① RS 同步 ──
        rs_corr = np.abs(np.correlate(symbols, REF_RS, mode='valid'))
        thr = np.percentile(rs_corr, 10) * 6  # 10分位噪声参考 ×6
        peaks = [i for i in range(1, len(rs_corr)-1)
                 if rs_corr[i] > thr and rs_corr[i] > rs_corr[i-1] and rs_corr[i] > rs_corr[i+1]]
        if not peaks:
            return

        if self._last_pss >= 0:
            expected = self._last_pss + FRAME_SYMBOLS
            p = None
            for cp in peaks:
                if abs(cp - expected - PSS_LEN) < 10:
                    p = cp - PSS_LEN
                    break
            if p is None:
                self._last_pss = -1
                return
        else:
            p = peaks[0] - PSS_LEN

        if p < 0 or p + PSS_LEN + RS_LEN + DATA_LEN > len(symbols):
            return

        # ── ② RS 频偏估计 ──
        rs_seg = symbols[p + PSS_LEN:p + PSS_LEN + RS_LEN]
        rs_corr_val = np.abs(np.dot(rs_seg, np.conj(REF_RS)))
        if rs_corr_val < 1.0:  # 噪声误检过滤
            return
        rs_tone = rs_seg * np.conj(REF_RS)
        rs_phase = np.unwrap(np.angle(rs_tone))
        nn16 = np.arange(RS_LEN, dtype=np.float64)
        slope = (np.sum(nn16 * rs_phase) - np.mean(nn16) * np.sum(rs_phase)) / \
                (np.sum(nn16**2) - RS_LEN * np.mean(nn16)**2)
        freq_est = slope / (2 * np.pi * self.Ts)

        # ── ③ 提取 LLR ──
        data_start = p + PSS_LEN + RS_LEN
        if data_start + DATA_LEN > len(symbols):
            return
        n_data = np.arange(DATA_LEN)
        data_syms = symbols[data_start:data_start + DATA_LEN]
        data_corrected = data_syms * np.exp(-1j * 2 * np.pi * freq_est * (data_start + n_data) * self.Ts)

        # BPSK LLR — 归一化信号幅度到1后匹配 SGNN 训练条件
        amp = max(np.std(data_corrected.real), 0.01)  # 估计信号幅度
        sigma_llr = 0.5  # 对应 SGNN 训练 SNR
        llr = (2.0 * data_corrected.real.astype(np.float32) / amp) / (sigma_llr ** 2)
        llr = np.clip(llr, -20, 20)  # 限幅防溢出

        # ── ④ SGNN 译码 (可选) — 高SNR时直判更好 ──
        code_bits = (llr < 0).astype(np.int64)  # 直判比特

        # 反 Arikan 变换 → 输入向量 u → 信息位
        u_hat = _polar_encode(code_bits)
        if self.frozen_mask is not None:
            info_bits = u_hat[self.frozen_mask.astype(bool)]

        # ── ⑤ 信息位 BER ──
        if tx_bits is not None and self.frozen_mask is not None:
            ref = tx_bits[self.info_total_bits:self.info_total_bits + len(info_bits)]
            if len(ref) == len(info_bits):
                err = int(np.sum(ref != info_bits))
                self.info_total_bits += len(info_bits)
                self.info_total_errors += err
                ber = err / len(info_bits)
            else:
                ber = 0
        else:
            ber = 0

        if self.total_frames < 10 or self.total_frames % 10 == 0:
            print(f"  frame={self.total_frames} ber(infobits)={ber:.4f} "
                  f"Δf={freq_est:.0f}Hz rs={rs_corr_val:.1f} "
                  f"std={data_corrected.real.std():.3f} "
                  f"info_err={self.info_total_errors}/{self.info_total_bits}")

        self.total_frames += 1
        self._last_pss = p

        # 消耗窗口
        consumed = min(self.win_len, (p + FRAME_SYMBOLS) * self.sps)
        self.win_len -= consumed
        if self.win_len > 0:
            self.win[:self.win_len] = self.win[consumed:consumed + self.win_len]

    def _print_summary(self, elapsed):
        ber = self.info_total_errors / max(self.info_total_bits, 1)
        print(f"\n接收完成: {self.total_frames} 帧, "
              f"信息位 BER={self.info_total_errors}/{self.info_total_bits}={ber:.2e}")


def main():
    p = argparse.ArgumentParser(description='极化码-PHY 融合接收端')
    p.add_argument('--mode', default='sim', choices=['hardware', 'sim'])
    p.add_argument('--freq', type=float, default=915e6)
    p.add_argument('--gain', type=float, default=40)
    p.add_argument('--rate', type=float, default=1e6)
    p.add_argument('--sim-file', default='rx_polar.npy')
    p.add_argument('--tx-info-bits', default='', help='发送端信息比特(.npy)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                    choices=['cpu', 'cuda'], help='推理设备')
    p.add_argument('--sgnn', action='store_true', help='使用SGNN译码(默认直判)')
    p.add_argument('--usrp-args', default='')
    p.add_argument('--subdev', default='A:A')
    p.add_argument('--save-iq', default='', help='保存原始IQ到文件(.npy), 用于离线分析')
    args = p.parse_args()

    rx = PolarPhyReceiver(samp_rate=args.rate, device=args.device)
    # 如果指定 --save-iq, 启用 IQ 保存
    if args.save_iq:
        rx._iq_buffer = []
        rx.save_iq_path = args.save_iq

    tx_info = None
    if args.tx_info_bits and os.path.isfile(args.tx_info_bits):
        tx_info = np.load(args.tx_info_bits)
        print(f"[receiver] 加载 {len(tx_info)} 参考信息比特")

    rx.running = True
    if args.mode == 'sim':
        rx._rx_loop_sim(args.sim_file, tx_info)
    else:
        rx._init_usrp(args.freq, args.gain, args.usrp_args, args.subdev)
        rx._rx_loop_hardware(tx_info)


if __name__ == '__main__':
    main()
