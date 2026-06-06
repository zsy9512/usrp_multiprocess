#!/usr/bin/env python3
"""
polar_phy_receiver.py — 极化码 + BPSK PHY 融合接收端

链路:
  USRP/文件 → STF粗同步 → PSS精定时 → RS细CFO+信道估计 → LLR解调 → SGNN译码 → 信息比特

帧结构 (与 receiver.py 一致):
  STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)

用法:
  仿真: python polar_phy_receiver.py --mode sim --sim-file test_rx.npy --tx-info-bits info_bits.npy
  硬件: python polar_phy_receiver.py --mode hardware --freq 915e6 --gain 40
"""

import argparse, os, sys, time, importlib.util
import numpy as np
import torch

# ── 导入 PHY RX 组件 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from receiver import (
    BpskPhyReceiver,
    _rrc_match_conj, _stf_delay_correlation, _compute_coarse_cfo,
    _pss_correlation, _rs_fine_cfo, _rs_channel_estimate,
    _demod_llr, _verify_header, _verify_payload_crc,
)
from phy_params import (
    SPS, TS, STF, PSS, RS, RRC,
    STF_LEN, PSS_LEN, RS_LEN, HEADER_LEN,
    PAYLOAD_LEN, PAYLOAD_CRC_LEN, GUARD_SYMBOLS,
    FRAME_SYMBOLS, STF_DELAY,
    STF_THRESHOLD, PSS_PEAK_TO_MEAN_THR, PSS_PEAK_TO_SECOND_THR,
    FRAME_RRC_SAMPLES, PSS_SEARCH_WIN_SAMPLES, MIN_WIN_SAMPLES,
)

# ── 导入 SGNN 译码器 ──
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


def _polar_transform(u):
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
        self._iq_buffer = None

        # 加载 SGNN 模型和 Tanner 图
        print(f"[receiver] 加载 SGNN 译码器 (device={device})...")
        self.sgnn_model, cfg = sgnn_receiver.load_model(CKPT_PATH, device=device)
        self.graph = sgnn_receiver.build_graph(PCM_PATH, device=device)
        self.N_sgnn = self.graph['N']
        self.N_hat = self.graph['N_hat']
        self.K_sgnn = self.graph['K']
        print(f"[receiver] SGNN: N={self.N_sgnn}, K={self.K_sgnn}, "
              f"nstate={cfg['nstate']}, lstu={cfg['lstu_mode']}")

        # 冻结比特掩膜
        A_PATH = os.path.join(MATRICES_DIR, 'A.npy')
        self.frozen_mask = np.load(A_PATH).squeeze() if os.path.isfile(A_PATH) else None

        # BER 统计 (信息位)
        self.info_total_bits = 0
        self.info_total_errors = 0

    def _detect_and_demod(self):
        """重写: 三级同步 → LLR → SGNN译码 → 提取信息位 → BER."""
        r = self.buf[:self.buf_len]

        while self.running and self.buf_len >= MIN_WIN_SAMPLES:
            self.detection_attempts += 1

            # ===== 阶段 1: STF 延迟相关 =====
            metric, P = _stf_delay_correlation(r)
            if len(metric) == 0:
                break

            candidates = []
            for d in range(len(metric)):
                if metric[d] > STF_THRESHOLD:
                    local_E = np.sum(np.abs(r[d + STF_DELAY:d + 2 * STF_DELAY]) ** 2)
                    if local_E > 0.1 * STF_DELAY:
                        candidates.append(d)

            if not candidates:
                self._advance_window(128 * SPS)  # 推进 ~1 个 PSS 窗口
                self._expected_frame_start = -1
                continue

            frame_found = False
            for candidate_d in candidates[:32]:
                coarse_sample_pos = candidate_d
                coarse_cfo = (_compute_coarse_cfo(P[candidate_d], STF_DELAY, self.samp_rate)
                              if candidate_d < len(P) else 0.0)

                # ===== 阶段 2: PSS 精定时 =====
                margin = PSS_SEARCH_WIN_SAMPLES
                extract_start = max(0, coarse_sample_pos - margin)
                extract_end = min(self.buf_len,
                                  coarse_sample_pos + margin + FRAME_RRC_SAMPLES)
                chunk = r[extract_start:extract_end]

                symbols = _rrc_match_conj(chunk, RRC)
                if len(symbols) < PSS_LEN + RS_LEN:
                    continue

                _, pss_peak, ptm, pts = _pss_correlation(symbols)
                if ptm < PSS_PEAK_TO_MEAN_THR or pts < PSS_PEAK_TO_SECOND_THR:
                    continue

                pss_start = pss_peak
                frame_sym_start = pss_start - STF_LEN
                if frame_sym_start < 0:
                    continue

                rrc_delay = (len(RRC) - 1) // 2
                frame_sample_start = (extract_start
                                      + frame_sym_start * self.sps
                                      - rrc_delay)
                if frame_sample_start < 0:
                    continue

                # ===== 阶段 3: RS 细 CFO + 信道估计 =====
                rs_sym_start = frame_sym_start + STF_LEN + PSS_LEN
                fine_cfo, rs_corr = _rs_fine_cfo(symbols, rs_sym_start)
                if rs_corr < RS_LEN * 0.3:
                    continue

                h, phase_est, sigma2 = _rs_channel_estimate(
                    symbols, rs_sym_start, fine_cfo)
                total_cfo = coarse_cfo + fine_cfo

                # ===== 阶段 4: Payload LLR → SGNN 译码 =====
                pay_start = frame_sym_start + STF_LEN + PSS_LEN + RS_LEN + HEADER_LEN
                pay_llr = _demod_llr(symbols, pay_start,
                                     PAYLOAD_LEN + PAYLOAD_CRC_LEN,
                                     h, phase_est, fine_cfo, sigma2)

                # 区分 Payload 和 CRC
                payload_llr = pay_llr[:PAYLOAD_LEN]

                # SGNN 译码 (可选)
                code_bits = (payload_llr < 0).astype(np.int64)
                u_hat = _polar_transform(code_bits)

                if self.frozen_mask is not None:
                    info_bits = u_hat[self.frozen_mask.astype(bool)]

                # CRC 校验 (直判比特用于 CRC)
                crc_llr = pay_llr[PAYLOAD_LEN:PAYLOAD_LEN + PAYLOAD_CRC_LEN]
                crc_bits = (crc_llr < 0).astype(np.int64)
                pay_crc_ok = _verify_payload_crc(code_bits, crc_bits)

                # ===== 阶段 5: 信息位 BER =====
                self.total_frames += 1
                if pay_crc_ok:
                    self.crc_pass += 1

                info_ber = 0.0
                if (self.tx_ref_bits is not None
                        and self.frozen_mask is not None):
                    start_idx = self.info_total_bits
                    end_idx = start_idx + len(info_bits)
                    if end_idx <= len(self.tx_ref_bits):
                        ref = self.tx_ref_bits[start_idx:end_idx]
                        err = int(np.sum(info_bits != ref))
                        self.info_total_bits += len(info_bits)
                        self.info_total_errors += err
                        info_ber = err / len(info_bits)

                # 也需要统计 Payload 级别的比特
                if self.tx_ref_bits is not None:
                    pl_start = self.total_bits
                    pl_end = pl_start + PAYLOAD_LEN
                    if pl_end <= len(self.tx_ref_bits):
                        ref_pl = self.tx_ref_bits[pl_start:pl_end]
                        errs_pl = int(np.sum(code_bits != ref_pl))
                        self.total_bits += PAYLOAD_LEN
                        self.total_errors += errs_pl

                if self.total_frames <= 5 or self.total_frames % 10 == 0:
                    print(f"  frame={self.total_frames} "
                          f"ber(infobits)={info_ber:.4f} "
                          f"Δf={total_cfo:.0f}Hz rs={rs_corr:.1f} "
                          f"σ²={sigma2:.4f} "
                          f"CRC={'OK' if pay_crc_ok else 'XX'} "
                          f"info_err={self.info_total_errors}/{self.info_total_bits}",
                          flush=True)

                # ===== 阶段 6: 消费窗口 =====
                consume_end = frame_sample_start + FRAME_RRC_SAMPLES
                if consume_end > self.buf_len:
                    consume_end = self.buf_len
                self._consume(consume_end)
                self._expected_frame_start = -1
                frame_found = True
                break

            if not frame_found:
                self.false_alarms += len(candidates[:32])
                self._advance_window(128 * SPS)
                self._expected_frame_start = -1

    def _process_window(self, tx_bits=None):
        """兼容旧接口."""
        self._detect_and_demod()

    def _print_summary(self, elapsed):
        super()._print_summary(elapsed)
        if self.info_total_bits > 0:
            ber = self.info_total_errors / self.info_total_bits
            print(f"  信息位 BER={self.info_total_errors}/{self.info_total_bits}={ber:.2e}")


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
    p.add_argument('--save-iq', default='', help='保存原始IQ到文件(.npy)')
    args = p.parse_args()

    rx = PolarPhyReceiver(samp_rate=args.rate, device=args.device)
    if args.save_iq:
        rx._iq_buffer = []
        rx._save_iq_path = args.save_iq

    tx_info = None
    if args.tx_info_bits and os.path.isfile(args.tx_info_bits):
        tx_info = np.load(args.tx_info_bits)
        print(f"[receiver] 加载 {len(tx_info)} 参考信息比特")

    rx.running = True
    if args.mode == 'sim':
        rx._rx_loop_sim(args.sim_file)
        if tx_info is not None:
            rx.tx_ref_bits = tx_info
    else:
        rx._init_usrp(args.freq, args.gain, args.usrp_args, args.subdev)
        rx._rx_loop_hardware()


if __name__ == '__main__':
    main()
