#!/usr/bin/env python3
"""
polar_decode.py — 极化码 SGNN 译码器

  stdin:  256 float32 LLR per frame
  stderr: 帧号 / BER（硬判 + SGNN）

用法:
  ./rx rx_iq.bin | python polar_decode.py --ref info_ref.npy
"""
import argparse, os, sys, numpy as np, torch

BASE = os.path.dirname(os.path.abspath(__file__))
A_PATH = os.path.join(BASE, 'deploy', 'matrices', 'A.npy')
PCM   = os.path.join(BASE, 'deploy', 'matrices', 'pcm.npy')
CKPT  = os.path.join(BASE, 'deploy', 'checkpoint', 'polar_GNN_20_iter_0_epoches_13.pt')

import importlib.util
spec = importlib.util.spec_from_file_location('common', os.path.join(BASE, 'deploy', 'common.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

FROZEN = np.load(A_PATH).squeeze() if os.path.isfile(A_PATH) else None
K = int(FROZEN.sum()) if FROZEN is not None else 128
N = 256


def polar_transform(u):
    cw = u.copy().ravel()
    for stage in range(1, int(np.log2(N)) + 1):
        sep = N // (1 << stage)
        for j in range(N):
            if (j // sep) % 2 == 0:
                cw[j] = (cw[j] + cw[j + sep]) % 2
    return cw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', default='')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    ref_info = np.load(args.ref) if args.ref and os.path.isfile(args.ref) else None

    print(f"[decode] 加载 SGNN ({args.device})...", file=sys.stderr)
    model, cfg = common.load_model(CKPT, device=args.device)
    graph = common.build_graph(PCM, device=args.device)
    ei, eir, tf = graph['edge_index'], graph['edge_index_rev'], graph['template_f']
    N_hat = graph['N_hat']
    print(f"[decode] N={N} K={K} nstate={cfg['nstate']}", file=sys.stderr)

    frame_idx = 0
    total_hard_errs = 0
    total_sgnn_errs = 0
    total_bits = 0
    total_frames = 0

    while True:
        raw = sys.stdin.buffer.read(N * 4)
        if len(raw) < N * 4:
            break
        total_frames += 1
        llr = np.frombuffer(raw, dtype=np.float32).copy()

        # 硬判
        hard = (llr < 0).astype(np.int64)
        u_hard = polar_transform(hard)
        info_hard = u_hard[FROZEN.astype(bool)] if FROZEN is not None else u_hard[:K]

        # SGNN
        v_feat = np.zeros((1, N_hat, 1), dtype=np.float32)
        v_feat[0, -N:, 0] = llr
        v_t = torch.from_numpy(v_feat).to(args.device)
        f_t = tf.unsqueeze(0)
        with torch.no_grad():
            out = model(v_t, f_t, ei, eir)
            code_llr = out[-1][0, -N:, 0].cpu().numpy()
        sgnn_bits = (code_llr < 0).astype(np.int64)
        u_sgnn = polar_transform(sgnn_bits)
        info_sgnn = u_sgnn[FROZEN.astype(bool)] if FROZEN is not None else u_sgnn[:K]

        ber_hard = -1.0
        ber_sgnn = -1.0
        if ref_info is not None:
            s = frame_idx * K
            if s + K <= len(ref_info):
                ref = ref_info[s:s+K]
                total_hard_errs += int(np.sum(info_hard != ref))
                ber_hard = int(np.sum(info_hard != ref)) / K
                total_sgnn_errs += int(np.sum(info_sgnn != ref))
                ber_sgnn = int(np.sum(info_sgnn != ref)) / K
                total_bits += K

        if total_frames <= 10 or total_frames % 20 == 0:
            print(f"  frame={total_frames:5d}  hard={ber_hard:.4f}  sgnn={ber_sgnn:.4f}",
                  file=sys.stderr, flush=True)

        frame_idx += 1

    print(f"\n[decode] {total_frames} frames  "
          f"hard_BER={total_hard_errs}/{total_bits}  "
          f"sgnn_BER={total_sgnn_errs}/{total_bits}",
          file=sys.stderr)


if __name__ == '__main__':
    main()
