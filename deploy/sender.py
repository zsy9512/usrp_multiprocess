#!/usr/bin/env python3
"""
sender.py  —  Polar-code transmitter  (fully self-contained)

No external dependencies beyond numpy + stdlib.
Copy this single file to any machine and run.

Pipeline:
  1. Randomly generate K=128 information bits
  2. Insert into unfrozen positions, polar-encode → N=256 codeword
  3. BPSK modulate → 256 symbols (0→+1, 1→-1)
  4. Pack as interleaved I/Q float32 → send via UDP to USRP TX port

Usage:
  python sender.py --tx-ip 192.168.10.2 --tx-port 5000
  python sender.py --tx-ip 127.0.0.1 --tx-port 5000 --matrices-dir ./matrices --frame-count 1000
"""
from __future__ import annotations

import argparse
import math
import os
import socket
import time
from pathlib import Path

import numpy as np

# ======================================================================
#  Polar encoding  (Arikan recursive butterfly)
# ======================================================================

def polar_encode(u: np.ndarray) -> np.ndarray:
    """Arikan polar transform.  u: (N,) {0,1} → cw: (N,) {0,1}."""
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
    frozen_mask (N,)  1 = information, 0 = frozen
    → codeword  (N,)
    """
    N = frozen_mask.shape[0]
    u = np.zeros(N, dtype=np.int64)
    u[frozen_mask.astype(bool)] = info_bits.ravel()
    return polar_encode(u)


def load_frozen_mask(path: str) -> np.ndarray:
    """Load A.npy → (N,)  {0,1}, 1=information."""
    return np.load(path).squeeze()


# ======================================================================
#  BPSK modulation
# ======================================================================

def bpsk_modulate(cw: np.ndarray) -> np.ndarray:
    """{0,1} → {+1,-1}."""
    return (1.0 - 2.0 * cw).astype(np.float32)


# ======================================================================
#  UDP serialization  (interleaved I/Q float32)
# ======================================================================

def symbols_to_bytes(symbols: np.ndarray) -> bytes:
    """Pack real or complex symbols → interleaved I/Q float32 bytes."""
    if np.iscomplexobj(symbols):
        interleaved = np.empty(2 * len(symbols), dtype=np.float32)
        interleaved[0::2] = symbols.real.astype(np.float32)
        interleaved[1::2] = symbols.imag.astype(np.float32)
    else:
        interleaved = np.zeros(2 * len(symbols), dtype=np.float32)
        interleaved[0::2] = symbols.astype(np.float32)
    return interleaved.tobytes()


# ======================================================================
#  Main
# ======================================================================

DEFAULT_MATRICES = Path(__file__).resolve().parent / "matrices"


def main():
    parser = argparse.ArgumentParser(description="Polar-code UDP transmitter")
    parser.add_argument("--tx-ip", default="127.0.0.1", help="USRP TX IP")
    parser.add_argument("--tx-port", type=int, default=5000, help="USRP TX UDP port")
    parser.add_argument("--interval", type=float, default=0.001,
                        help="Inter-frame interval (s). 0 = send as fast as possible")
    parser.add_argument("--matrices-dir", default=str(DEFAULT_MATRICES),
                        help="Directory containing A.npy (frozen-bit mask)")
    parser.add_argument("--frame-count", type=int, default=0,
                        help="Stop after N frames (0 = infinite)")
    args = parser.parse_args()

    # ----  load frozen-bit mask  ----
    a_path = os.path.join(args.matrices_dir, "A.npy")
    if not os.path.isfile(a_path):
        raise FileNotFoundError(f"Frozen mask not found: {a_path}")
    frozen_mask = load_frozen_mask(a_path)
    N = frozen_mask.shape[0]
    K = int(frozen_mask.sum())
    FRAME_BYTES = N * 2 * 4                         # 256 symbols × I/Q × float32

    print(f"[sender] N={N}  K={K}  mask={a_path}")

    # ----  UDP socket  ----
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx_addr = (args.tx_ip, args.tx_port)
    print(f"[sender] → {tx_addr[0]}:{tx_addr[1]}  ({FRAME_BYTES} bytes/frame)")

    # ----  main loop  ----
    frame_idx = 0
    t_start = time.time()
    try:
        while True:
            info = np.random.randint(0, 2, K).astype(np.int64)
            cw = build_codeword(info, frozen_mask)
            syms = bpsk_modulate(cw)
            data = symbols_to_bytes(syms)
            sock.sendto(data, tx_addr)

            frame_idx += 1
            if args.frame_count > 0 and frame_idx >= args.frame_count:
                break
            if args.interval > 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    elapsed = time.time() - t_start
    fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"[sender] done.  {frame_idx} frames in {elapsed:.1f}s  ({fps:.1f} fps)")


if __name__ == "__main__":
    main()
