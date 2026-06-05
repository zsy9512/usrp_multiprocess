#!/usr/bin/env python3
"""
simulate.py  —  Polar-SGNN closed-loop simulation  (fully self-contained)

Runs the full pipeline in software: encode → channel → decode → BER/FER.
Use this for performance evaluation before deploying to USRP hardware.

No external dependencies beyond numpy, torch + stdlib.

Usage:
  # Sweep Eb/N0 from 1 to 5 dB, pure AWGN
  python simulate.py --checkpoint model.pt --ebn0-range 1.0,5.0,0.5 --frames 5000

  # Single SNR point with burst noise
  python simulate.py --checkpoint model.pt --ebn0 3.0 --sigma-b 2.0 --burst-prob 0.1 --frames 10000

  # Full sweep with burst noise, GPU
  python simulate.py --checkpoint model.pt --ebn0-range 1.0,5.0,0.5 --sigma-b 2.0 --burst-prob 0.1 --device cuda
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ======================================================================
#  1.  Polar encoding
# ======================================================================

def polar_encode(u: np.ndarray) -> np.ndarray:
    N = u.shape[0]
    cw = u.copy().ravel()
    n_stages = int(math.log2(N))
    for stage in range(1, n_stages + 1):
        sep = N // (1 << stage)
        for j in range(N):
            if (j // sep) % 2 == 0:
                cw[j] = (cw[j] + cw[j + sep]) % 2
    return cw


def build_codeword(info_bits: np.ndarray, frozen_mask: np.ndarray) -> np.ndarray:
    N = frozen_mask.shape[0]
    u = np.zeros(N, dtype=np.int64)
    u[frozen_mask.astype(bool)] = info_bits.ravel()
    return polar_encode(u)


def load_frozen_mask(path: str) -> np.ndarray:
    return np.load(path).squeeze()


# ======================================================================
#  2.  BPSK
# ======================================================================

def bpsk_modulate(cw: np.ndarray) -> np.ndarray:
    return (1.0 - 2.0 * cw).astype(np.float32)


def bpsk_demodulate_llr(y: np.ndarray, sigma: float) -> np.ndarray:
    return (2.0 * y) / (sigma ** 2)


# ======================================================================
#  3.  Scatter sum
# ======================================================================

def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = src.new_zeros((dim_size, src.shape[-1]))
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    return out


# ======================================================================
#  4.  BPConv
# ======================================================================

class BPConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_hidden,
                 activation=nn.Tanh, bias=False, use_cat=False):
        super().__init__()
        self.out_channels = out_channels
        self.use_cat = use_cat
        self.lin_msg = nn.Sequential(
            nn.Linear(in_channels, n_hidden, bias=bias),
            activation(),
            nn.Linear(n_hidden, out_channels, bias=bias),
        )
        if use_cat:
            self.lin_emb_cat = nn.Sequential(
                nn.Linear(out_channels * 2, n_hidden, bias=bias),
                activation(),
                nn.Linear(n_hidden, out_channels, bias=bias),
            )
        else:
            self.lin_emb = nn.Sequential(
                nn.Linear(out_channels, n_hidden, bias=bias),
                activation(),
                nn.Linear(n_hidden, out_channels, bias=bias),
            )

    def message(self, x_i, x_j):
        return self.lin_msg(torch.cat([x_i, x_j], dim=-1))

    def forward(self, x, edge_index):
        if isinstance(x, (tuple, list)):
            h_from, h_to = x
        else:
            h_from = h_to = x
        h_i = h_from[:, edge_index[0], :]
        h_j = h_to[:, edge_index[1], :]
        msg = self.message(h_i, h_j)
        B, E, C_out = msg.shape
        out = msg.new_zeros((B, h_to.shape[1], C_out))
        for b in range(B):
            out[b] = scatter_sum(msg[b], edge_index[1], h_to.shape[1])
        if self.use_cat:
            out = self.lin_emb_cat(torch.cat([out, h_to], dim=-1))
        else:
            out = self.lin_emb(out)
        return out


# ======================================================================
#  5.  LSTU variants
# ======================================================================

class PassThrough(nn.Module):
    def __init__(self, in_channels): super().__init__()
    def forward(self, x, h): return x

class ResNet(nn.Module):
    def __init__(self, in_channels): super().__init__()
    def forward(self, x, h): return h + x

class Gate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.s = nn.Parameter(torch.empty(in_channels).uniform_(2.2, 5.0))
    def forward(self, x, h): return torch.sigmoid(self.s) * h + x

class Gate2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.s_h = nn.Parameter(torch.empty(in_channels).uniform_(2.2, 5.0))
        self.s_x = nn.Parameter(torch.empty(in_channels).uniform_(-0.5, 0.5))
    def forward(self, x, h):
        return torch.sigmoid(self.s_h) * h + torch.sigmoid(self.s_x) * x

class LSTU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_gate = nn.Linear(in_channels, in_channels, bias=True)
        nn.init.uniform_(self.lin_gate.bias, 2.2, 5.0)
    def forward(self, x, h): return torch.sigmoid(self.lin_gate(x)) * h + x

class LSTU2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_gate_h = nn.Linear(in_channels, in_channels, bias=True)
        self.lin_gate_x = nn.Linear(in_channels, in_channels, bias=True)
        nn.init.uniform_(self.lin_gate_h.bias, 2.2, 5.0)
        nn.init.normal_(self.lin_gate_h.weight, std=0.01)
        nn.init.uniform_(self.lin_gate_x.bias, -0.5, 0.5)
        nn.init.normal_(self.lin_gate_x.weight, std=0.01)
    def forward(self, x, h):
        return torch.sigmoid(self.lin_gate_h(x)) * h + torch.sigmoid(self.lin_gate_x(x)) * x

_LSTU_MAP = {
    "baseline": PassThrough, "fixed": ResNet,
    "ssm": LSTU, "learnable": Gate,
    "ssm2": LSTU2, "learnable2": Gate2,
}


# ======================================================================
#  6.  SGNN
# ======================================================================

class SGNN(nn.Module):
    def __init__(self, ninv, nstate=32, nhid=64, nstep=20,
                 lstu_mode="ssm", activation="tanh", use_cat=True):
        super().__init__()
        self.nstep = nstep
        _act = {"tanh": nn.Tanh, "relu": nn.ReLU,
                "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        self.lin_v = nn.Linear(ninv, nstate, bias=True)
        ck = {"n_hidden": nhid, "activation": _act, "use_cat": use_cat}
        self.propagatorv2f = BPConv(nstate * 2, nstate, **ck)
        self.propagatorf2v = BPConv(nstate * 2, nstate, **ck)
        Lstu = _LSTU_MAP[lstu_mode]
        self.lstu_v2f = Lstu(nstate)
        self.lstu_f2v = Lstu(nstate)
        self.decoder = nn.Linear(nstate, 1, bias=False)

    def forward(self, v, f, edge_index, edge_index_rev, nstep=None):
        if nstep is None:
            nstep = self.nstep
        B = v.shape[0]
        hv = self.lin_v(v)
        hf = torch.zeros(B, f.shape[1], hv.shape[2], device=v.device)
        llr_hat = []
        for _ in range(nstep):
            xf = self.propagatorv2f((hv, hf), edge_index)
            hf = self.lstu_v2f(xf, hf)
            xv = self.propagatorf2v((hf, hv), edge_index_rev)
            hv = self.lstu_f2v(xv, hv)
            llr_hat.append(self.decoder(hv))
        return llr_hat


# ======================================================================
#  7.  Model loading
# ======================================================================

def _infer_from_ckpt(state_dict: dict) -> dict:
    w = state_dict["lin_v.weight"]
    nstate, ninv = w.shape
    nhid = state_dict["propagatorv2f.lin_msg.0.weight"].shape[0]
    # use_cat detection: checkpoint uses 'lin_emb_cat' for cat mode
    has_cat = "propagatorv2f.lin_emb_cat.0.weight" in state_dict
    has_simple = "propagatorv2f.lin_emb.0.weight" in state_dict
    use_cat = has_cat or not has_simple
    if "lstu_v2f.lin_gate_h.bias" in state_dict:
        lstu_mode = "ssm2"
    elif "lstu_v2f.lin_gate.bias" in state_dict:
        lstu_mode = "ssm"
    elif "lstu_v2f.s_h" in state_dict:
        lstu_mode = "learnable2"
    elif "lstu_v2f.s" in state_dict:
        lstu_mode = "learnable"
    else:
        lstu_mode = "baseline"
    return {"ninv": ninv, "nstate": nstate, "nhid": nhid,
            "use_cat": use_cat, "lstu_mode": lstu_mode}


def load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[SGNN, dict]:
    ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
    sd = ckpt.get("model_state_dict", ckpt)
    cfg = _infer_from_ckpt(sd)
    model = SGNN(ninv=cfg["ninv"], nstate=cfg["nstate"], nhid=cfg["nhid"],
                 lstu_mode=cfg["lstu_mode"], use_cat=cfg["use_cat"])
    model_keys = set(model.state_dict().keys())
    sd_f = {k: v for k, v in sd.items() if k in model_keys}
    missing = model_keys - set(sd_f.keys())
    if missing:
        print(f"[load] warning: missing keys: {missing}")
    model.load_state_dict(sd_f, strict=False)
    model.to(device)
    model.eval()
    return model, cfg


# ======================================================================
#  8.  Graph structure
# ======================================================================

def build_graph(pcm_path: str, device: str = "cpu") -> dict:
    h = np.load(pcm_path)
    N = 256
    N_hat = h.shape[1]
    M = h.shape[0]
    ht = np.where(h.T == 1)
    ev2f = np.array([ht[0], ht[1]])
    edge_index = torch.tensor(ev2f, dtype=torch.long, device=device)
    edge_index_rev = torch.stack([edge_index[1], edge_index[0]])
    template_f = torch.rand(M, 1, device=device)
    return {"edge_index": edge_index, "edge_index_rev": edge_index_rev,
            "template_f": template_f,
            "N": N, "N_hat": N_hat, "M": M}


# ======================================================================
#  9.  Channel
# ======================================================================

def awgn_channel(syms: np.ndarray, ebn0_db: float, K: int, N: int) -> Tuple[np.ndarray, float]:
    """AWGN channel: returns (noisy_symbols, sigma)."""
    snr_db = ebn0_db + 10.0 * math.log10(K / N)
    sigma = (1.0 / math.sqrt(10.0 ** (snr_db / 10.0))) / math.sqrt(2.0)
    noise = sigma * np.random.randn(N).astype(np.float32)
    return syms + noise, sigma


def burst_channel(syms: np.ndarray, sigma: float, sigma_b: float,
                  burst_prob: float) -> np.ndarray:
    """Add burst noise on top of AWGN symbols."""
    if sigma_b <= 1e-20:
        return syms
    y = syms.copy()
    for i in range(len(y)):
        if np.random.rand() < burst_prob:
            y[i] += np.random.randn() * sigma_b
    return y


# ======================================================================
#  10.  Main simulation
# ======================================================================

DEFAULT_MATRICES = Path(__file__).resolve().parent / "matrices"
DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoint" / "model.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Polar-SGNN closed-loop simulation")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--matrices-dir", default=str(DEFAULT_MATRICES),
                        help="Directory containing pcm.npy and A.npy")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    # SNR config
    parser.add_argument("--ebn0", type=float, default=None,
                        help="Single Eb/N0 (dB). Overrides --ebn0-range")
    parser.add_argument("--ebn0-range", default="1.0,5.0,0.5",
                        help="Eb/N0 sweep: start,end,step (dB)")
    # Burst noise
    parser.add_argument("--sigma-b", type=float, default=0.0,
                        help="Burst noise std-dev (0 = AWGN only)")
    parser.add_argument("--burst-prob", type=float, default=0.1,
                        help="Per-symbol burst probability")
    # Simulation
    parser.add_argument("--frames", type=int, default=2000,
                        help="Frames per SNR point")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for GPU inference (0 = auto)")
    parser.add_argument("--max-frame-errors", type=int, default=500,
                        help="Stop early after N frame errors per SNR point")
    args = parser.parse_args()

    # ----  parse SNR range  ----
    if args.ebn0 is not None:
        ebn0_list = [args.ebn0]
    else:
        parts = [float(x.strip()) for x in args.ebn0_range.split(",")]
        if len(parts) != 3:
            raise ValueError("--ebn0-range requires start,end,step")
        start, end, step = parts
        ebn0_list = list(np.arange(start, end + step * 0.5, step))

    # ----  load model  ----
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    print(f"[sim] loading {args.checkpoint} ...")
    model, cfg = load_model(args.checkpoint, device=args.device)
    print(f"[sim] arch: ninv={cfg['ninv']} nstate={cfg['nstate']} "
          f"nhid={cfg['nhid']} lstu={cfg['lstu_mode']} use_cat={cfg['use_cat']}")

    # ----  load matrices  ----
    pcm_path = os.path.join(args.matrices_dir, "pcm.npy")
    a_path = os.path.join(args.matrices_dir, "A.npy")
    if not os.path.isfile(pcm_path):
        raise FileNotFoundError(f"PCM not found: {pcm_path}")
    if not os.path.isfile(a_path):
        raise FileNotFoundError(f"Frozen mask not found: {a_path}")
    graph = build_graph(pcm_path, device=args.device)
    frozen = load_frozen_mask(a_path)
    N = graph["N"]
    N_hat = graph["N_hat"]
    K_info = int(frozen.sum())
    edge_index = graph["edge_index"]
    edge_index_rev = graph["edge_index_rev"]
    template_f = graph["template_f"]

    print(f"[sim] N={N}  K_info={K_info}  N_hat={N_hat}  "
          f"sigma_b={args.sigma_b}  burst_prob={args.burst_prob}")

    # ----  batch inference helper  ----
    batch_size = args.batch_size if args.batch_size > 0 else args.frames

    def decode_batch(llr_batch: np.ndarray) -> np.ndarray:
        """llr_batch: (B, N)  →  hard_bits: (B, N)"""
        B = llr_batch.shape[0]
        v_feat = np.zeros((B, N_hat, 1), dtype=np.float32)
        v_feat[:, -N:, 0] = llr_batch
        v_t = torch.from_numpy(v_feat).to(args.device)
        f_t = template_f.unsqueeze(0).expand(B, -1, -1)
        with torch.no_grad():
            out = model(v_t, f_t, edge_index, edge_index_rev)
            code_llr = out[-1][:, -N:, 0].cpu().numpy()
        return (code_llr < 0).astype(np.int64)

    # ----  sweep  ----
    print(f"\n{'Eb/N0':>8s}  {'BER':>10s}  {'FER':>10s}  "
          f"{'BitErrs':>10s}  {'FrmErrs':>10s}  {'Frames':>10s}  {'Time':>8s}")
    print("-" * 75)

    t_total = time.time()
    for ebn0 in ebn0_list:
        snr_db = ebn0 + 10.0 * math.log10(K_info / N)
        sigma_a = (1.0 / math.sqrt(10.0 ** (snr_db / 10.0))) / math.sqrt(2.0)

        bit_errs = 0
        frm_errs = 0
        frames_done = 0
        t0 = time.time()

        llr_buf = []
        cw_buf = []

        while frames_done < args.frames:
            # generate a batch
            B_cur = min(batch_size, args.frames - frames_done)
            llr_batch = np.zeros((B_cur, N), dtype=np.float32)
            cw_batch = np.zeros((B_cur, N), dtype=np.int64)

            for b in range(B_cur):
                info = np.random.randint(0, 2, K_info).astype(np.int64)
                cw = build_codeword(info, frozen)
                syms = bpsk_modulate(cw)

                # AWGN
                y, _sigma = awgn_channel(syms, ebn0, K_info, N)
                # burst
                y = burst_channel(y, sigma_a, args.sigma_b, args.burst_prob)
                # LLR
                # Effective noise: sigma_eff^2 = sigma_a^2 + burst_prob * sigma_b^2
                sigma_eff2 = sigma_a ** 2 + args.burst_prob * (args.sigma_b ** 2)
                sigma_eff = math.sqrt(max(sigma_eff2, 1e-30))
                llr = bpsk_demodulate_llr(y, sigma_eff)

                llr_batch[b] = llr
                cw_batch[b] = cw

            # decode batch
            hard = decode_batch(llr_batch)

            # count errors
            for b in range(B_cur):
                errs = (hard[b] != cw_batch[b]).sum()
                bit_errs += int(errs)
                if errs > 0:
                    frm_errs += 1

            frames_done += B_cur
            if frm_errs >= args.max_frame_errors:
                break

        t1 = time.time()
        total_bits = frames_done * N
        ber = bit_errs / total_bits if total_bits > 0 else 0
        fer = frm_errs / frames_done if frames_done > 0 else 0

        print(f"{ebn0:8.2f}  {ber:10.2e}  {fer:10.4f}  "
              f"{bit_errs:10d}  {frm_errs:10d}  {frames_done:10d}  "
              f"{t1 - t0:7.1f}s", flush=True)

    t_total = time.time() - t_total
    print(f"\n[sim] done.  total time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
