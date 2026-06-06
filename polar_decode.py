#!/usr/bin/env python3
"""
polar_decode.py — 极化码硬判译码器 (IPC 管道)

  stdin:  [4B frame_id(BE)][1024B LLR(256 float32)][1B crc_ok]
  stderr: 帧号 / BER / CRC 状态

用法:
  ./rx rx_iq.bin | python polar_decode.py --ref info_ref.npy
  ./rx --crc-filter rx_iq.bin | python polar_decode.py
"""
import argparse, os, sys, struct, numpy as np

A_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'deploy', 'matrices', 'A.npy')
FROZEN = np.load(A_PATH).squeeze() if os.path.isfile(A_PATH) else None
K = int(FROZEN.sum()) if FROZEN is not None else 128
N = 256
FRAME_BYTES = 4 + N * 4 + 1  # 1029


def polar_transform(u):
    cw = u.copy().ravel()
    for s in range(1, int(np.log2(N)) + 1):
        sp = N // (1 << s)
        for j in range(N):
            if (j // sp) % 2 == 0:
                cw[j] = (cw[j] + cw[j + sp]) % 2
    return cw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', default='', help='参考信息位 .npy (帧id对齐)')
    args = p.parse_args()

    ref_info = np.load(args.ref) if args.ref and os.path.isfile(args.ref) else None
    if ref_info is not None:
        print(f"[decode] 参考: {len(ref_info)} bits ({len(ref_info)//K} frames)", file=sys.stderr)

    total_frames = 0
    total_crc_ok = 0
    total_bit_errs = 0
    total_bits = 0

    while True:
        raw = sys.stdin.buffer.read(FRAME_BYTES)
        if len(raw) < FRAME_BYTES:
            break
        total_frames += 1

        frame_id = struct.unpack('>H', raw[0:2])[0]
        llr = np.frombuffer(raw[4:4 + N * 4], dtype=np.float32)
        crc_ok = raw[-1] == 1
        if crc_ok:
            total_crc_ok += 1

        # 硬判 → polar 变换 → info bits
        hard = (llr < 0).astype(np.int64)
        info_hat = polar_transform(hard)[FROZEN.astype(bool)] if FROZEN is not None else polar_transform(hard)[:K]

        ber = -1.0
        if ref_info is not None:
            s = frame_id * K
            if s + K <= len(ref_info):
                errs = int(np.sum(info_hat != ref_info[s:s + K]))
                total_bit_errs += errs
                total_bits += K
                ber = errs / K

        if total_frames <= 20 or total_frames % 100 == 0:
            status = f"BER={ber:.4f}" if ber >= 0 else "---"
            crc_s = "OK" if crc_ok else "XX"
            print(f"  frame={frame_id:5d}  CRC={crc_s}  {status}",
                  file=sys.stderr, flush=True)

    print(f"\n[decode] {total_frames} frames, CRC OK={total_crc_ok}, "
          f"BER={total_bit_errs}/{total_bits}",
          file=sys.stderr)


if __name__ == '__main__':
    main()
