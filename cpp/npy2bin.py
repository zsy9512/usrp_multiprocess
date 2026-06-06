#!/usr/bin/env python3
"""
npy2bin.py — 将 .npy (complex64) 转换为 C++ 接收端可读的 .bin

用法:
  python npy2bin.py tx_iq.npy tx_iq.bin
"""
import sys, numpy as np
data = np.load(sys.argv[1])
data.astype(np.complex64).tofile(sys.argv[2])
print(f"{sys.argv[1]} ({len(data)} samples) → {sys.argv[2]}")
