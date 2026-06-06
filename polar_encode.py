#!/usr/bin/env python3
"""
polar_encode.py — 极化码编码器 (stdin/stdout 管道)

  随机生成 128bit 信息位 → Arikan 极化编码 → 256bit 码字 → stdout

用法:
  python polar_encode.py --frames 100 | ./tx -o tx_iq.bin
  python polar_encode.py --frames 100 --save-info info.npy | ./tx -o tx_iq.bin
"""
import argparse, os, sys, struct, numpy as np

# ── 加载冻结比特掩膜 ──
MATRICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deploy', 'matrices')
A_PATH = os.path.join(MATRICES_DIR, 'A.npy')
if not os.path.isfile(A_PATH):
    print(f"[encode] 错误: A.npy 未找到: {A_PATH}", file=sys.stderr); sys.exit(1)
FROZEN = np.load(A_PATH).squeeze()
K = int(FROZEN.sum())  # 128
N = FROZEN.shape[0]    # 256


def polar_encode(u):
    """Arikan polar transform (自逆)."""
    cw = u.copy().ravel()
    for stage in range(1, int(np.log2(N)) + 1):
        sep = N // (1 << stage)
        for j in range(N):
            if (j // sep) % 2 == 0:
                cw[j] = (cw[j] + cw[j + sep]) % 2
    return cw


def build_codeword(info_bits):
    u = np.zeros(N, dtype=np.int64)
    u[FROZEN.astype(bool)] = info_bits.ravel()
    return polar_encode(u)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, default=10)
    p.add_argument('--save-info', default='')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    if args.seed:
        np.random.seed(args.seed)

    info_all = [] if args.save_info else None

    for fi in range(args.frames):
        info = np.random.randint(0, 2, K).astype(np.int64)
        cw = build_codeword(info)
        # [4B frame_id (uint16 BE)][32B codeword]
        header = struct.pack('>H', fi) + b'\x00\x00'
        buf = np.packbits(cw.astype(np.uint8)).tobytes()
        sys.stdout.buffer.write(header + buf)
        if info_all is not None:
            info_all.append(info)

    sys.stdout.buffer.flush()

    if info_all:
        np.save(args.save_info, np.concatenate(info_all))
        print(f"[encode] {args.frames} frames, info bits → {args.save_info}", file=sys.stderr)


if __name__ == '__main__':
    main()
