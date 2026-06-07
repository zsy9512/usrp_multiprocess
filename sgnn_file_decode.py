#!/usr/bin/env python3
"""SGNN decoder from file (for conda env with torch)"""
import argparse, os, sys, struct, importlib.util, numpy as np, torch

BASE = os.path.dirname(os.path.abspath(__file__))
A_PATH = os.path.join(BASE, 'deploy', 'matrices', 'A.npy')
FROZEN = np.load(A_PATH).squeeze() if os.path.isfile(A_PATH) else None
K = int(FROZEN.sum()) if FROZEN is not None else 128; N = 256
FRAME_BYTES = 4 + N * 4 + 1

def polar_transform(u):
    cw = u.copy().ravel()
    for s in range(1, int(np.log2(N)) + 1):
        sp = N // (1 << s)
        for j in range(N):
            if (j // sp) % 2 == 0: cw[j] = (cw[j] + cw[j + sp]) % 2
    return cw

def main():
    p = argparse.ArgumentParser()
    p.add_argument('llr_file', help='LLR binary from rx.exe')
    p.add_argument('--ref', default='')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    # Load SGNN
    spec = importlib.util.spec_from_file_location('common', os.path.join(BASE, 'deploy', 'common.py'))
    common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    ckpt = os.path.join(BASE, 'deploy', 'checkpoint', 'polar_GNN_20_iter_0_epoches_13.pt')
    pcm  = os.path.join(BASE, 'deploy', 'matrices', 'pcm.npy')
    print(f"[decode] loading SGNN ({args.device})...", file=sys.stderr)
    model, cfg = common.load_model(ckpt, device=args.device)
    graph = common.build_graph(pcm, device=args.device)
    print(f"[decode] SGNN: nstate={cfg['nstate']} N_hat={graph['N_hat']}", file=sys.stderr)

    ref_info = np.load(args.ref) if args.ref and os.path.isfile(args.ref) else None
    if ref_info is not None:
        print(f"[decode] ref: {len(ref_info)} bits ({len(ref_info)//K} frames)", file=sys.stderr)

    data = open(args.llr_file, 'rb').read()
    nframes = len(data) // FRAME_BYTES
    print(f"[decode] {nframes} frames in file", file=sys.stderr)

    total = 0; crc_ok = 0; hard_errs = 0; sgnn_errs = 0; bits = 0
    for fi in range(nframes):
        off = fi * FRAME_BYTES
        frame_id = struct.unpack('>H', data[off:off+2])[0]
        llr = np.frombuffer(data[off+4:off+4+N*4], dtype=np.float32)
        crc_flag = data[off + FRAME_BYTES - 1] == 1
        total += 1
        if crc_flag: crc_ok += 1

        # hard
        hard = (llr < 0).astype(np.int64)
        info_h = polar_transform(hard)[FROZEN.astype(bool)]

        # SGNN
        v = np.zeros((1, graph['N_hat'], 1), dtype=np.float32)
        v[0, -N:, 0] = llr
        vt = torch.from_numpy(v).to(args.device)
        ft = graph['template_f'].unsqueeze(0)
        with torch.no_grad():
            out = model(vt, ft, graph['edge_index'], graph['edge_index_rev'])
        sgnn_bits = (out[-1][0, -N:, 0].cpu().numpy() < 0).astype(np.int64)
        info_s = polar_transform(sgnn_bits)[FROZEN.astype(bool)]

        ber_h = ber_s = -1.0
        if ref_info is not None:
            s = frame_id * K
            if s + K <= len(ref_info):
                ref = ref_info[s:s+K]
                hard_errs += int(np.sum(info_h != ref)); ber_h = int(np.sum(info_h != ref)) / K
                sgnn_errs += int(np.sum(info_s != ref)); ber_s = int(np.sum(info_s != ref)) / K
                bits += K

        if total <= 20 or total % 100 == 0:
            print(f"  frame={frame_id:5d} CRC={'OK' if crc_flag else 'XX'} hard={ber_h:.4f} sgnn={ber_s:.4f}", file=sys.stderr, flush=True)

    print(f"\n[decode] {total} frames CRC_OK={crc_ok} hard={hard_errs}/{bits} sgnn={sgnn_errs}/{bits}", file=sys.stderr)

if __name__ == '__main__':
    main()
