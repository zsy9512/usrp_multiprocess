#!/usr/bin/env python3
"""
polar_decode.py — 极化码译码器 (IPC 管道, 支持硬判 + SGNN)

  stdin:  [4B frame_id(BE)][1024B LLR(256 float32)][1B crc_ok]
  stderr: 帧号 / hard_BER / sgnn_BER / CRC

用法:
  ./rx rx_iq.bin | python polar_decode.py --ref info.npy
  ./rx rx_iq.bin | python polar_decode.py --ref info.npy --sgnn
"""
import argparse, os, sys, struct, importlib.util
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
A_PATH = os.path.join(BASE, 'deploy', 'matrices', 'A.npy')
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
    p.add_argument('--ref', default='', help='参考信息位 .npy')
    p.add_argument('--sgnn', action='store_true', help='使用 SGNN 译码 (需 torch)')
    p.add_argument('--device', default='cpu', help='SGNN 设备 cpu/cuda')
    args = p.parse_args()

    # -- SGNN 初始化 --
    model = graph = None
    if args.sgnn:
        import torch
        spec = importlib.util.spec_from_file_location('common',
            os.path.join(BASE, 'deploy', 'common.py'))
        common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
        ckpt = os.path.join(BASE, 'deploy', 'checkpoint',
                           'polar_GNN_20_iter_0_epoches_13.pt')
        pcm  = os.path.join(BASE, 'deploy', 'matrices', 'pcm.npy')
        print(f"[decode] 加载 SGNN ({args.device})...", file=sys.stderr)
        model, cfg = common.load_model(ckpt, device=args.device)
        graph = common.build_graph(pcm, device=args.device)
        print(f"[decode] SGNN: nstate={cfg['nstate']} N_hat={graph['N_hat']}",
              file=sys.stderr)

    ref_info = np.load(args.ref) if args.ref and os.path.isfile(args.ref) else None
    if ref_info is not None:
        print(f"[decode] 参考: {len(ref_info)} bits ({len(ref_info)//K} frames)", file=sys.stderr)

    total_frames = 0; total_crc_ok = 0
    hard_errs = 0; sgnn_errs = 0; total_bits = 0

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

        # 硬判
        hard = (llr < 0).astype(np.int64)
        info_hard = polar_transform(hard)[FROZEN.astype(bool)] if FROZEN is not None else polar_transform(hard)[:K]

        # SGNN
        info_sgnn = None
        if model is not None:
            v_feat = np.zeros((1, graph['N_hat'], 1), dtype=np.float32)
            v_feat[0, -N:, 0] = llr
            v_t = torch.from_numpy(v_feat).to(args.device)
            f_t = graph['template_f'].unsqueeze(0)
            with torch.no_grad():
                out = model(v_t, f_t, graph['edge_index'], graph['edge_index_rev'])
            sgnn_bits = (out[-1][0, -N:, 0].cpu().numpy() < 0).astype(np.int64)
            info_sgnn = polar_transform(sgnn_bits)[FROZEN.astype(bool)] if FROZEN is not None else polar_transform(sgnn_bits)[:K]

        # BER
        ber_h = ber_s = -1.0
        if ref_info is not None:
            s = frame_id * K
            if s + K <= len(ref_info):
                ref = ref_info[s:s + K]
                hard_errs += int(np.sum(info_hard != ref))
                ber_h = int(np.sum(info_hard != ref)) / K
                if info_sgnn is not None:
                    sgnn_errs += int(np.sum(info_sgnn != ref))
                    ber_s = int(np.sum(info_sgnn != ref)) / K
                total_bits += K

        if total_frames <= 20 or total_frames % 100 == 0:
            h_s = f"hard={ber_h:.4f}" if ber_h >= 0 else "---"
            s_s = f"sgnn={ber_s:.4f}" if ber_s >= 0 else ""
            crc_s = "OK" if crc_ok else "XX"
            print(f"  frame={frame_id:5d}  CRC={crc_s}  {h_s}  {s_s}",
                  file=sys.stderr, flush=True)

    print(f"\n[decode] {total_frames} frames, CRC OK={total_crc_ok}, "
          f"hard_BER={hard_errs}/{total_bits}  sgnn_BER={sgnn_errs}/{total_bits}",
          file=sys.stderr)


if __name__ == '__main__':
    main()
