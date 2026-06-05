#!/usr/bin/env python3
"""
interferer.py  —  Burst-interference generator  (fully self-contained)

No external dependencies beyond numpy + stdlib.
Generates complex AWGN bursts and sends them via UDP to a separate USRP.

Modes:
  burst      — per-symbol probabilistic noise (burst_prob per symbol)
  continuous — full-frame Gaussian noise (sigma_b on every symbol)

Usage:
  python interferer.py --tx-ip 192.168.10.3 --tx-port 5002 --sigma-b 2.0 --burst-prob 0.1
  python interferer.py --mode continuous --sigma-b 1.0 --frame-length 256
"""
from __future__ import annotations

import argparse
import socket
import time

import numpy as np

# ======================================================================
#  UDP serialization
# ======================================================================

def symbols_to_bytes(symbols: np.ndarray) -> bytes:
    """Pack complex symbols → interleaved I/Q float32 bytes."""
    interleaved = np.empty(2 * len(symbols), dtype=np.float32)
    interleaved[0::2] = symbols.real.astype(np.float32)
    interleaved[1::2] = symbols.imag.astype(np.float32)
    return interleaved.tobytes()


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Burst-interference UDP transmitter")
    parser.add_argument("--tx-ip", default="127.0.0.1", help="Interference USRP TX IP")
    parser.add_argument("--tx-port", type=int, default=5002,
                        help="Interference USRP TX UDP port")
    parser.add_argument("--sigma-b", type=float, default=2.0,
                        help="Burst noise standard deviation")
    parser.add_argument("--burst-prob", type=float, default=0.1,
                        help="Per-symbol burst probability (burst mode only)")
    parser.add_argument("--frame-length", type=int, default=256,
                        help="Symbols per frame (should match sender N)")
    parser.add_argument("--interval", type=float, default=0.001,
                        help="Inter-frame interval (s). 0 = as fast as possible")
    parser.add_argument("--frame-count", type=int, default=0,
                        help="Stop after N frames (0 = infinite)")
    parser.add_argument("--mode", choices=["burst", "continuous"], default="burst",
                        help="burst = per-symbol probabilistic, "
                             "continuous = full-frame AWGN")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx_addr = (args.tx_ip, args.tx_port)
    N = args.frame_length
    FRAME_BYTES = N * 2 * 4
    print(f"[interferer] → {tx_addr[0]}:{tx_addr[1]}  mode={args.mode}  "
          f"sigma_b={args.sigma_b}  burst_prob={args.burst_prob}  N={N}")

    frame_idx = 0
    t_start = time.time()
    try:
        while True:
            if args.mode == "burst":
                noise_i = np.zeros(N, dtype=np.float32)
                noise_q = np.zeros(N, dtype=np.float32)
                for i in range(N):
                    if np.random.rand() < args.burst_prob:
                        noise_i[i] = np.random.randn() * args.sigma_b
                        noise_q[i] = np.random.randn() * args.sigma_b
                syms = noise_i + 1j * noise_q
            else:
                noise_i = np.random.randn(N).astype(np.float32) * args.sigma_b
                noise_q = np.random.randn(N).astype(np.float32) * args.sigma_b
                syms = noise_i + 1j * noise_q

            sock.sendto(symbols_to_bytes(syms), tx_addr)

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
    print(f"[interferer] done.  {frame_idx} frames in {elapsed:.1f}s  ({fps:.1f} fps)")


if __name__ == "__main__":
    main()
