#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_capture.py — 批量 SNR sweep 采集 (一键跑完所有增益档位)

用法:
  python tools/batch_capture.py                          # 默认 v1 全扫
  python tools/batch_capture.py --v2                     # v2 长前导 + 30ms 帧间隔
  python tools/batch_capture.py --gains 64 55 48         # 指定增益
  python tools/batch_capture.py --runs 3                 # 每档重复3次
  python tools/batch_capture.py --dry-run                # 只打印命令, 不执行

v1 vs v2:
  v1: STF=64 RS=32 gap=5ms   (loopback_capture.py)
  v2: STF=128 RS=64 gap=30ms (loopback_capture_v2.py, 极化编码)
"""

import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone

# -- 默认配置 --
SERIAL       = "320F33F"
FREQ         = 915e6
GAIN_TX      = 60
RX_CHANNEL   = 1
RX_ANTENNA   = "RX2"
NUM_FRAMES   = 200
FRAME_GAP_MS = 5.0           # v1 default, v2 overrides to 30ms

# RX 增益扫参 (TX=60 固定, RX=21~40 覆盖 Eb/N0 0-25 dB)
DEFAULT_GAINS = [21, 23, 25,27, 30,40]
DEFAULT_RUNS  = 1          # 每档重复次数 (≥3 推荐做统计)


def run_cmd(cmd, dry_run=False):
    """打印并(可选)执行命令。返回 (returncode, stdout, stderr)."""
    print(f"\n  $ {' '.join(cmd)}")
    if dry_run:
        return 0, "", ""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        for line in r.stdout.strip().splitlines():
            print(f"    {line}")
    if r.returncode != 0 and r.stderr:
        print(f"  [stderr] {r.stderr[:500]}")
    return r.returncode, r.stdout, r.stderr


def main():
    p = argparse.ArgumentParser(description="批量 SNR sweep IQ 采集")
    p.add_argument("--serial", default=SERIAL)
    p.add_argument("--freq", type=float, default=FREQ)
    p.add_argument("--gain-tx", type=float, default=GAIN_TX)
    p.add_argument("--rx-channel", type=int, default=RX_CHANNEL,
                   help=f"RX 通道号 (0=A板TX/RX 1=A板RX2 2=B板TX/RX 3=B板RX2, 默认 {RX_CHANNEL})")
    p.add_argument("--rx-antenna", default=RX_ANTENNA,
                   help=f"RX 天线端口 (默认 {RX_ANTENNA})")
    p.add_argument("--gains", type=int, nargs="+", default=DEFAULT_GAINS,
                   help=f"RX 增益列表 (默认 {DEFAULT_GAINS})")
    p.add_argument('--v2', action='store_true',
                   help='使用 v2 capture (STF=128 RS=64 gap=30ms 极化编码)')
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                   help=f"每档重复次数 (默认 {DEFAULT_RUNS})")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    p.add_argument("--frame-gap-ms", type=float, default=FRAME_GAP_MS)
    p.add_argument("--outdir", default="",
                   help="输出目录 (默认 capture/YYYYMMDD/)")
    p.add_argument("--dry-run", action="store_true", help="只打印命令, 不实际执行")
    p.add_argument("--skip-analyze", action="store_true", help="跳过离线分析, 只采集")
    args = p.parse_args()

    # -- 输出目录 --
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    outdir = args.outdir or f"capture/{today}"
    os.makedirs(outdir, exist_ok=True)
    print(f"输出目录: {outdir}/")

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.v2:
        CAPTURE_SCRIPT = os.path.join(BASE, "tools", "loopback_capture_v2.py")
        frame_gap = args.frame_gap_ms if args.frame_gap_ms != 5.0 else 30.0
    else:
        CAPTURE_SCRIPT = os.path.join(BASE, "tools", "loopback_capture.py")
        frame_gap = args.frame_gap_ms
    ANALYZE_SCRIPT = os.path.join(BASE, "tools", "analyze_repeat_capture.py")

    results = []
    t_start = time.time()

    for gain in args.gains:
        for run_idx in range(args.runs):
            tag = f"snr_gain{gain:03d}_r{run_idx}"
            prefix = os.path.join(outdir, tag)
            print(f"\n{'='*60}")
            print(f"  [{tag}]  RX gain={gain} dB  (run {run_idx+1}/{args.runs})")
            print(f"{'='*60}")

            # -- 步骤 1: 采集 IQ --
            cmd = [
                sys.executable, CAPTURE_SCRIPT,
                "--serial", args.serial,
                "--freq", str(args.freq),
                "--gain-tx", str(args.gain_tx),
                "--gain-rx", str(gain),
                "--rx-channel", str(args.rx_channel),
                "--rx-antenna", args.rx_antenna,
                "--num-frames", str(args.num_frames),
                "--frame-gap-ms", str(frame_gap),
                "-o", prefix,
            ]
            rc, stdout, stderr = run_cmd(cmd, dry_run=args.dry_run)
            if rc != 0 and not args.dry_run:
                # UHD 在 Windows 退出时常有非零返回码 (INFO 日志写 stderr),
                # 不影响文件生成, 仅警告不阻断
                print(f"  [info] loopback_capture.py rc={rc} (UHD cleanup noise, ignoring)")

            # 检查输出文件 (比返回码更可靠)
            iq_path = prefix + "_iq.npy"
            if not args.dry_run and not os.path.isfile(iq_path):
                print(f"  [FAIL] 未生成 {iq_path}")
                results.append({"tag": tag, "gain_rx": gain, "run": run_idx,
                                "capture_ok": False, "error": "no output file"})
                continue

            # 采集成功确认 (文件存在)
            iq_size_mb = os.path.getsize(iq_path) / 1e6 if not args.dry_run else 0
            print(f"  [OK] {tag}_iq.npy  ({iq_size_mb:.1f} MB)")

            # -- 步骤 2: 分析 (any-of-5 检出, 适配 5x 重复帧) --
            if not args.skip_analyze:
                stats_json = prefix + "_stats.json"
                cmd2 = [
                    sys.executable, ANALYZE_SCRIPT,
                    os.path.dirname(prefix),  # input dir
                    "--gain", str(gain),
                    "--num-frames", str(args.num_frames),
                    "-o", stats_json,
                ]
                rc2, stdout2, stderr2 = run_cmd(cmd2, dry_run=args.dry_run)

                if not args.dry_run and os.path.isfile(stats_json):
                    with open(stats_json, encoding='utf-8') as f:
                        stats = json.load(f)
                    gain_str = str(gain)
                    gs = stats.get(gain_str, {})
                    results.append({
                        "tag": tag, "gain_rx": gain, "run": run_idx,
                        "capture_ok": True,
                        "groups_ok": gs.get("groups_ok", 0),
                        "total_groups": gs.get("total_groups", 0),
                        "detection_rate": gs.get("detection_rate", 0),
                        "mean_hits": gs.get("mean_hits", 0),
                    })
                    dr = gs.get('detection_rate', 0)
                    hits = gs.get('mean_hits', 0)
                    print(f"    groups={gs.get('groups_ok',0)}/{gs.get('total_groups',0)} "
                          f"({dr*100:.0f}%)  hits={hits:.1f}/5")
                else:
                    results.append({"tag": tag, "gain_rx": gain, "run": run_idx,
                                    "capture_ok": True, "analyze_ok": False})
            else:
                results.append({"tag": tag, "gain_rx": gain, "run": run_idx,
                                "capture_ok": True})

    # -- 汇总 --
    elapsed = time.time() - t_start
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "serial": args.serial,
        "freq_hz": args.freq,
        "gain_tx_db": args.gain_tx,
        "rx_channel": args.rx_channel,
        "rx_antenna": args.rx_antenna,
        "num_frames": args.num_frames,
        "frame_gap_ms": args.frame_gap_ms,
        "runs_per_gain": args.runs,
        "elapsed_s": elapsed,
        "results": results,
    }
    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"完成! 耗时 {elapsed:.0f}s  汇总 -> {summary_path}")
    print(f"{'='*60}")

    # Summary table
    if results:
        header = f"  {'gain':>5s}  {'groups':>8s}  {'rate':>7s}  {'hits':>6s}"
        print(f"\n{header}")
        print(f"  {'-'*35}")
        for r in results:
            go = r.get("groups_ok", 0)
            tg = r.get("total_groups", 0)
            dr = r.get("detection_rate", 0)
            mh = r.get("mean_hits", 0)
            go_str = f'{go}/{tg}' if tg > 0 else 'N/A'
            dr_str = f'{dr*100:.0f}%' if tg > 0 else 'N/A'
            mh_str = f'{mh:.1f}/5' if tg > 0 else 'N/A'
            print(f"  {r['gain_rx']:5d}  {go_str:>8s}  {dr_str:>7s}  {mh_str:>5s}")


if __name__ == "__main__":
    main()
