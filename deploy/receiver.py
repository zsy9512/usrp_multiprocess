#!/usr/bin/env python3
"""
receiver.py  —  Polar-code SGNN receiver  (fully self-contained)

No external dependencies beyond numpy, torch + stdlib.
Copy this single file (plus matrices/ and checkpoint/) to any machine.

Pipeline:
  1. Receive interleaved I/Q float32 symbols via UDP from USRP RX
  2. BPSK demodulate → LLR
  3. Build Tanner-graph features → SGNN decode
  4. Output hard-decisions

Usage:
  python receiver.py --rx-port 5001 --checkpoint model.pt --sigma 0.5
  python receiver.py --rx-port 5001 --checkpoint model.pt --device cuda --print-every 50
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ======================================================================
#  1.  Scatter sum  (no torch_scatter needed)
# ======================================================================

def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = src.new_zeros((dim_size, src.shape[-1]))
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    return out


# ======================================================================
#  2.  BPConv  —  message propagation layer
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
#  3.  LSTU variants  —  state-transition units
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
#  4.  SGNN  model
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
#  5.  Model loading
# ======================================================================

def _infer_from_ckpt(state_dict: dict) -> dict:
    w = state_dict["lin_v.weight"]
    nstate, ninv = w.shape
    nhid = state_dict["propagatorv2f.lin_msg.0.weight"].shape[0]
    use_cat = ("propagatorv2f.lin_emb_cat.0.weight" in state_dict or
               "propagatorv2f.lin_emb.0.weight" not in state_dict)
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
        print(f"[load_model] warning: missing keys: {missing}")
    model.load_state_dict(sd_f, strict=False)
    model.to(device)
    model.eval()
    return model, cfg


# ======================================================================
#  6.  Graph structure
# ======================================================================

def build_graph(pcm_path: str, device: str = "cpu") -> dict:
    h = np.load(pcm_path)
    N = 256
    N_hat = h.shape[1]
    M = h.shape[0]
    K = N_hat - M
    ht = np.where(h.T == 1)
    ev2f = np.array([ht[0], ht[1]])
    edge_index = torch.tensor(ev2f, dtype=torch.long, device=device)
    edge_index_rev = torch.stack([edge_index[1], edge_index[0]])
    template_f = torch.rand(M, 1, device=device)
    return {"edge_index": edge_index, "edge_index_rev": edge_index_rev,
            "template_f": template_f,
            "N": N, "N_hat": N_hat, "M": M, "K": K, "nv": N_hat, "nf": M}


# ======================================================================
#  7.  UDP deserialization + LLR
# ======================================================================

def bytes_to_symbols(data: bytes, num_symbols: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.float32)
    expected = 2 * num_symbols
    if len(arr) < expected:
        raise ValueError(f"Expected {expected} floats, got {len(arr)}")
    arr = arr[:expected]
    return arr[0::2] + 1j * arr[1::2]


def bpsk_demodulate_llr(y: np.ndarray, sigma: float) -> np.ndarray:
    return (2.0 * y) / (sigma ** 2)


# ======================================================================
#  8.  Main
# ======================================================================

DEFAULT_MATRICES = Path(__file__).resolve().parent / "matrices"
DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoint" / "model.pt"


def main():
    parser = argparse.ArgumentParser(description="Polar-code SGNN UDP receiver")
    parser.add_argument("--rx-ip", default="0.0.0.0")
    parser.add_argument("--rx-port", type=int, default=5001)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--matrices-dir", default=str(DEFAULT_MATRICES))
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Noise std-dev for LLR (0 = auto-estimate)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--print-every", type=int, default=100)
    args = parser.parse_args()

    # ----  load model  ----
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    print(f"[receiver] loading {args.checkpoint} ...")
    model, cfg = load_model(args.checkpoint, device=args.device)
    print(f"[receiver] arch: ninv={cfg['ninv']} nstate={cfg['nstate']} "
          f"nhid={cfg['nhid']} lstu={cfg['lstu_mode']} use_cat={cfg['use_cat']}")

    # ----  build graph  ----
    pcm_path = os.path.join(args.matrices_dir, "pcm.npy")
    if not os.path.isfile(pcm_path):
        raise FileNotFoundError(f"PCM not found: {pcm_path}")
    graph = build_graph(pcm_path, device=args.device)
    N = graph["N"]
    N_hat = graph["N_hat"]
    edge_index = graph["edge_index"]
    edge_index_rev = graph["edge_index_rev"]
    template_f = graph["template_f"]
    FRAME_BYTES = N * 2 * 4

    # ----  UDP  ----
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.rx_ip, args.rx_port))
    sock.settimeout(2.0)
    print(f"[receiver] listening {args.rx_ip}:{args.rx_port}  "
          f"({FRAME_BYTES} bytes/frame)")

    frame_idx = 0
    t_start = time.time()
    try:
        while True:
            try:
                data, _addr = sock.recvfrom(65536)
            except socket.timeout:
                print("[receiver] timeout — waiting ...")
                continue

            syms = bytes_to_symbols(data, N)
            y = syms.real.astype(np.float32)
            sigma = args.sigma if args.sigma > 0 else float(np.std(y)) * 0.5
            llr = bpsk_demodulate_llr(y, sigma)

            v_feat = np.zeros((N_hat, 1), dtype=np.float32)
            v_feat[-N:, 0] = llr
            v_t = torch.from_numpy(v_feat).unsqueeze(0).to(args.device)
            f_t = template_f.unsqueeze(0)

            with torch.no_grad():
                out = model(v_t, f_t, edge_index, edge_index_rev)
                code_llr = out[-1][0, -N:, 0].cpu().numpy()
            hard_bits = (code_llr < 0).astype(np.int64)

            frame_idx += 1
            if args.print_every > 0 and frame_idx % args.print_every == 0:
                elapsed = time.time() - t_start
                fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"[receiver] frame {frame_idx:6d}  fps={fps:.1f}  "
                      f"sigma={sigma:.4f}  ones={hard_bits.sum()}/{N}",
                      flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    elapsed = time.time() - t_start
    fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"[receiver] done.  {frame_idx} frames in {elapsed:.1f}s  ({fps:.1f} fps)")


if __name__ == "__main__":
    main()
