"""
common.py — Self-contained utilities for USRP-based polar code deployment.

Copy this file together with sender.py / receiver.py / interferer.py
to any machine.  Dependencies: numpy, torch.  torch-scatter is NOT required.

Paths to matrices/ and model checkpoint are configurable via CLI or env vars.
"""
from __future__ import annotations

import math
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# ======================================================================
# 1.  Polar encoding  (Arikan recursive butterfly, matches polar.py)
# ======================================================================

def polar_encode(u: np.ndarray) -> np.ndarray:
    """Arikan polar transform.  u: (N,)  {0,1}  →  cw: (N,)  {0,1}."""
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
    """Insert K info bits into unfrozen positions, then polar-encode.

    info_bits  (K,)
    frozen_mask (N,)  1 = information bit, 0 = frozen
    → codeword  (N,)
    """
    N = frozen_mask.shape[0]
    u = np.zeros(N, dtype=np.int64)
    u[frozen_mask.astype(bool)] = info_bits.ravel()
    return polar_encode(u)


# ======================================================================
# 2.  BPSK  modulation / demodulation
# ======================================================================

def bpsk_modulate(cw: np.ndarray) -> np.ndarray:
    """{0,1} → {+1,-1}."""
    return (1.0 - 2.0 * cw).astype(np.float32)


def bpsk_demodulate_llr(y: np.ndarray, sigma: float) -> np.ndarray:
    """LLR = 2*y / sigma^2   (AWGN)."""
    return (2.0 * y) / (sigma ** 2)


# ======================================================================
# 3.  UDP serialization  (interleaved I/Q float32)
# ======================================================================

def symbols_to_bytes(symbols: np.ndarray) -> bytes:
    """Pack complex (or real) symbols → interleaved I/Q float32 bytes."""
    if np.iscomplexobj(symbols):
        interleaved = np.empty(2 * len(symbols), dtype=np.float32)
        interleaved[0::2] = symbols.real.astype(np.float32)
        interleaved[1::2] = symbols.imag.astype(np.float32)
    else:
        interleaved = np.zeros(2 * len(symbols), dtype=np.float32)
        interleaved[0::2] = symbols.astype(np.float32)
    return interleaved.tobytes()


def bytes_to_symbols(data: bytes, num_symbols: int) -> np.ndarray:
    """Unpack interleaved I/Q float32 bytes → complex64 symbols."""
    arr = np.frombuffer(data, dtype=np.float32)
    expected = 2 * num_symbols
    if len(arr) < expected:
        raise ValueError(
            f"Expected {expected} floats for {num_symbols} symbols, got {len(arr)}"
        )
    arr = arr[:expected]
    return arr[0::2] + 1j * arr[1::2]


# ======================================================================
# 4.  Scatter sum  (replaces torch_scatter to keep deployment lean)
# ======================================================================

def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum src values into output according to index (like torch_scatter).

    src:   (E, C)   edge messages
    index: (E,)     target node id for each edge
    → out: (dim_size, C)
    """
    out = src.new_zeros((dim_size, src.shape[-1]))
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, index_expanded, src)
    return out


# ======================================================================
# 5.  BPConv  —  message-propagation layer  (self-contained copy)
# ======================================================================

class BPConv(nn.Module):
    """Message propagation on Tanner graph (B matrix).

    * message:  MLP([h_i || h_j])
    * aggregate: scatter_sum at target nodes
    * fuse:  opt. concat with target state → MLP
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_hidden: int,
        activation=nn.Tanh,
        bias: bool = False,
        use_cat: bool = False,
    ):
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

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.lin_msg(torch.cat([x_i, x_j], dim=-1))

    def forward(self, x, edge_index: Tensor) -> Tensor:
        """x = (h_from, h_to)  or just h_from."""
        if isinstance(x, (tuple, list)):
            h_from, h_to = x
        else:
            h_from = h_to = x

        # Gather source / target states along edge dimension
        h_i = h_from[:, edge_index[0], :]          # (B, E, C)
        h_j = h_to[:, edge_index[1], :]            # (B, E, C)

        msg = self.message(h_i, h_j)               # (B, E, C_out)

        # Sum-aggregate per target node  (support batch dim)
        B, E, C_out = msg.shape
        dim_size = h_to.shape[1]
        out = msg.new_zeros((B, dim_size, C_out))
        for b in range(B):
            out[b] = scatter_sum(msg[b], edge_index[1], dim_size)

        if self.use_cat:
            out = self.lin_emb_cat(torch.cat([out, h_to], dim=-1))
        else:
            out = self.lin_emb(out)
        return out


# ======================================================================
# 6.  LSTU variants  —  A-matrix  (self-contained copy)
# ======================================================================

class PassThrough(nn.Module):
    """h_new = x  (no memory)"""
    def __init__(self, in_channels: int):
        super().__init__()
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return x


class ResNet(nn.Module):
    """h_new = h + x  (residual)"""
    def __init__(self, in_channels: int):
        super().__init__()
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return h + x


class Gate(nn.Module):
    """h_new = σ(s) ⊙ h + x"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.s = nn.Parameter(torch.empty(in_channels).uniform_(2.2, 5.0))
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return torch.sigmoid(self.s) * h + x


class Gate2(nn.Module):
    """h_new = σ(s_h) ⊙ h + σ(s_x) ⊙ x"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.s_h = nn.Parameter(torch.empty(in_channels).uniform_(2.2, 5.0))
        self.s_x = nn.Parameter(torch.empty(in_channels).uniform_(-0.5, 0.5))
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return torch.sigmoid(self.s_h) * h + torch.sigmoid(self.s_x) * x


class LSTU(nn.Module):
    """h_new = σ(W·x + b) ⊙ h + x"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.lin_gate = nn.Linear(in_channels, in_channels, bias=True)
        nn.init.uniform_(self.lin_gate.bias, 2.2, 5.0)
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return torch.sigmoid(self.lin_gate(x)) * h + x


class LSTU2(nn.Module):
    """h_new = σ(W_h·x + b_h) ⊙ h + σ(W_x·x + b_x) ⊙ x"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.lin_gate_h = nn.Linear(in_channels, in_channels, bias=True)
        self.lin_gate_x = nn.Linear(in_channels, in_channels, bias=True)
        nn.init.uniform_(self.lin_gate_h.bias, 2.2, 5.0)
        nn.init.normal_(self.lin_gate_h.weight, std=0.01)
        nn.init.uniform_(self.lin_gate_x.bias, -0.5, 0.5)
        nn.init.normal_(self.lin_gate_x.weight, std=0.01)
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        g_h = torch.sigmoid(self.lin_gate_h(x))
        g_x = torch.sigmoid(self.lin_gate_x(x))
        return g_h * h + g_x * x


_LSTU_MAP = {
    "baseline":   PassThrough,
    "fixed":      ResNet,
    "ssm":        LSTU,
    "learnable":  Gate,
    "ssm2":       LSTU2,
    "learnable2": Gate2,
}


# ======================================================================
# 7.  SGNN  model  (self-contained)
# ======================================================================

class SGNN(nn.Module):
    """Sparse Graph Neural Network for polar decoding."""

    def __init__(
        self,
        ninv: int,
        nstate: int = 32,
        nhid: int = 64,
        nstep: int = 20,
        lstu_mode: str = "ssm",
        activation: str = "tanh",
        use_cat: bool = True,
    ):
        super().__init__()
        self.nstep = nstep

        _act = {"tanh": nn.Tanh, "relu": nn.ReLU,
                "gelu": nn.GELU, "silu": nn.SiLU}[activation]

        self.lin_v = nn.Linear(ninv, nstate, bias=True)

        conv_kw = {"n_hidden": nhid, "activation": _act, "use_cat": use_cat}
        self.propagatorv2f = BPConv(nstate * 2, nstate, **conv_kw)
        self.propagatorf2v = BPConv(nstate * 2, nstate, **conv_kw)

        Lstu = _LSTU_MAP[lstu_mode]
        self.lstu_v2f = Lstu(nstate)
        self.lstu_f2v = Lstu(nstate)

        self.decoder = nn.Linear(nstate, 1, bias=False)

    def forward(
        self,
        v: Tensor,
        f: Tensor,
        edge_index: Tensor,
        edge_index_rev: Tensor,
        nstep: int | None = None,
    ) -> List[Tensor]:
        if nstep is None:
            nstep = self.nstep

        B = v.shape[0]
        hv = self.lin_v(v)                          # (B, Nv, C)
        hf = torch.zeros(B, f.shape[1], hv.shape[2], device=v.device)

        llr_hat: List[Tensor] = []
        for _ in range(nstep):
            # V → F
            xf = self.propagatorv2f((hv, hf), edge_index)
            hf = self.lstu_v2f(xf, hf)
            # F → V
            xv = self.propagatorf2v((hf, hv), edge_index_rev)
            hv = self.lstu_f2v(xv, hv)
            # decode
            llr_hat.append(self.decoder(hv))

        return llr_hat


# ======================================================================
# 8.  Model loading  (architecture auto-inference from checkpoint)
# ======================================================================

def _infer_from_ckpt(state_dict: dict) -> dict:
    """Infer model hyper-params from raw state_dict keys."""
    w = state_dict["lin_v.weight"]             # [nstate, ninv]
    nstate, ninv = w.shape
    nhid = state_dict["propagatorv2f.lin_msg.0.weight"].shape[0]
    use_cat = "propagatorv2f.lin_emb.0.weight" in state_dict
    # or "propagatorv2f.lin_emb_cat.0.weight" — both BPConv variants
    if not use_cat:
        use_cat = "propagatorv2f.lin_emb_cat.0.weight" in state_dict

    if "lstu_v2f.lin_gate_h.bias" in state_dict:
        lstu_mode = "ssm2"
    elif "lstu_v2f.lin_gate.bias" in state_dict:
        lstu_mode = "ssm"
    elif "lstu_v2f.s_h" in state_dict:
        lstu_mode = "learnable2"
    elif "lstu_v2f.s" in state_dict:
        lstu_mode = "learnable"
    else:
        lstu_mode = "baseline"   # PassThrough or ResNet

    return {
        "ninv": ninv,
        "nstate": nstate,
        "nhid": nhid,
        "use_cat": use_cat,
        "lstu_mode": lstu_mode,
    }


def load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[SGNN, dict]:
    """Load a trained SGNN from checkpoint.

    Returns (model, config_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
    sd = ckpt.get("model_state_dict", ckpt)
    cfg = _infer_from_ckpt(sd)

    model = SGNN(
        ninv=cfg["ninv"],
        nstate=cfg["nstate"],
        nhid=cfg["nhid"],
        lstu_mode=cfg["lstu_mode"],
        use_cat=cfg["use_cat"],
    )
    # Checkpoint may contain extra keys from thop profiler / buffers;
    # filter to only weights that exist in the model.
    model_keys = set(model.state_dict().keys())
    sd_filtered = {k: v for k, v in sd.items() if k in model_keys}
    missing = model_keys - set(sd_filtered.keys())
    if missing:
        print(f"[load_model] warning: missing keys (will be random): {missing}")
    model.load_state_dict(sd_filtered, strict=False)
    model.to(device)
    model.eval()
    return model, cfg


# ======================================================================
# 9.  Tanner-graph structure from parity-check matrix
# ======================================================================

def build_graph(pcm_path: str, device: str = "cpu") -> dict:
    """Build edge_index, etc. from a .npy parity-check matrix.

    Returns dict with keys:
      edge_index, edge_index_rev, template_f,
      N, N_hat, M, K, nv, nf
    """
    h = np.load(pcm_path)                # (M, N_hat)
    N = 256
    N_hat = h.shape[1]
    M = h.shape[0]
    K = N_hat - M

    # edge list: where H^T == 1
    ht = np.where(h.T == 1)
    ev2f = np.array([ht[0], ht[1]])
    edge_index = torch.tensor(ev2f, dtype=torch.long, device=device)
    edge_index_rev = torch.stack([edge_index[1], edge_index[0]])

    template_f = torch.rand(M, 1, device=device)

    return {
        "edge_index": edge_index,
        "edge_index_rev": edge_index_rev,
        "template_f": template_f,
        "N": N, "N_hat": N_hat, "M": M, "K": K,
        "nv": N_hat, "nf": M,
    }


def load_frozen_mask(a_path: str) -> np.ndarray:
    """Load frozen-bit mask A.npy → (N,)  {0,1},  1=information."""
    return np.load(a_path).squeeze()
