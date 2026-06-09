#!/usr/bin/env python3
"""
batch_capture.py — 批量 SNR sweep 采集 (一键跑完所有增益档位)

用法:
  python tools/batch_capture.py                          # 默认全扫
  python tools/batch_capture.py --gains 64 55 48         # 指定增益
  python tools/batch_capture.py --runs 3                 # 每档重复3次
  python tools/batch_capture.py --dry-run                # 只打印命令, 不执行

输出目录:
  capture/YYYYMMDD/
    snr_gain064_r0_iq.npy   snr_gain064_r0_bits.npy   snr_gain064_r0_meta.json
    snr_gain064_r0_stats.json
    snr_gain055_r0_iq.npy   ...
    ...
    summary.json                                          # 汇总所有采集结果

依赖:
  tools/loopback_capture.py  (采集 IQ)
  tools/loopback_analyze.py  (离线分析)
  tools/verify_calibration.py (口径验证)
"""

import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone

# ── 默认配置 (对齐 polar_loopback.py / loopback_test.py) ──
SERIAL       = "320F33F"
FREQ         = 915e6
GAIN_TX      = 65
RX_CHANNEL   = 1          # 0=A板TX/RX, 1=A板RX2, 2=B板TX/RX, 3=B板RX2
RX_ANTENNA   = "RX2"
NUM_FRAMES   = 200
FRAME_GAP_MS = 5.0

# RX 增益扫参 (近似 SNR: 64→~30dB, 55→~20dB, 48→~15dB, 42→~10dB, 38→~5dB, 35→~0dB)
DEFAULT_GAINS = [64, 55, 48, 42, 38, 35]
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
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                   help=f"每档重复次数 (默认 {DEFAULT_RUNS})")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    p.add_argument("--frame-gap-ms", type=float, default=FRAME_GAP_MS)
    p.add_argument("--outdir", default="",
                   help="输出目录 (默认 capture/YYYYMMDD/)")
    p.add_argument("--dry-run", action="store_true", help="只打印命令, 不实际执行")
    p.add_argument("--skip-analyze", action="store_true", help="跳过离线分析, 只采集")
    args = p.parse_args()

    # ── 输出目录 ──
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    outdir = args.outdir or f"capture/{today}"
    os.makedirs(outdir, exist_ok=True)
    print(f"输出目录: {outdir}/")

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CAPTURE_SCRIPT = os.path.join(BASE, "tools", "loopback_capture.py")
    ANALYZE_SCRIPT = os.path.join(BASE, "tools", "loopback_analyze.py")
    VERIFY_SCRIPT  = os.path.join(BASE, "tools", "verify_calibration.py")

    results = []
    t_start = time.time()

    for gain in args.gains:
        for run_idx in range(args.runs):
            tag = f"snr_gain{gain:03d}_r{run_idx}"
            prefix = os.path.join(outdir, tag)
            print(f"\n{'='*60}")
            print(f"  [{tag}]  RX gain={gain} dB  (run {run_idx+1}/{args.runs})")
            print(f"{'='*60}")

            # ── 步骤 1: 采集 IQ ──
            cmd = [
                sys.executable, CAPTURE_SCRIPT,
                "--serial", args.serial,
                "--freq", str(args.freq),
                "--gain-tx", str(args.gain_tx),
                "--gain-rx", str(gain),
                "--rx-channel", str(args.rx_channel),
                "--rx-antenna", args.rx_antenna,
                "--num-frames", str(args.num_frames),
                "--frame-gap-ms", str(args.frame_gap_ms),
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
                print(f"  ❌ 未生成 {iq_path}")
                results.append({"tag": tag, "gain_rx": gain, "run": run_idx,
                                "capture_ok": False, "error": "no output file"})
                continue

            # 采集成功确认 (文件存在)
            iq_size_mb = os.path.getsize(iq_path) / 1e6 if not args.dry_run else 0
            print(f"  ✅ {tag}_iq.npy  ({iq_size_mb:.1f} MB)")

            # ── 步骤 2: 离线分析 ──
            if not args.skip_analyze:
                stats_json = prefix + "_stats.json"
                cmd2 = [
                    sys.executable, VERIFY_SCRIPT,
                    prefix,
                    "--ptm", "3.5", "--pts", "1.5",
                    "-o", stats_json,
                ]
                rc2, stdout2, stderr2 = run_cmd(cmd2, dry_run=args.dry_run)

                # 解析 stats 提取关键指标 (以文件存在为准, 不依赖返回码)
                if not args.dry_run and os.path.isfile(stats_json):
                    with open(stats_json) as f:
                        stats = json.load(f)
                    results.append({
                        "tag": tag, "gain_rx": gain, "run": run_idx,
                        "capture_ok": True,
                        "n_frames": stats.get("n_frames", 0),
                        "snr_prefix_mean": stats.get("snr_prefix_mean"),
                        "snr_rs_mean": stats.get("snr_rs_mean"),
                        "cfo_mean": stats.get("cfo_mean"),
                        "cfo_std": stats.get("cfo_std"),
                        "crc_ok_rate": stats.get("crc_ok_rate"),
                        "hdr_ok_rate": stats.get("hdr_ok_rate"),
                    })
                    print(f"    检出={stats.get('n_frames','?')}  "
                          f"SNR={stats.get('snr_prefix_mean','?'):.1f}dB  "
                          f"CRC={stats.get('crc_ok_rate',0)*100:.0f}%")
                else:
                    results.append({"tag": tag, "gain_rx": gain, "run": run_idx,
                                    "capture_ok": True, "analyze_ok": False})
            else:
                results.append({"tag": tag, "gain_rx": gain, "run": run_idx,
                                "capture_ok": True})

    # ── 汇总 ──
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
    print(f"完成! 耗时 {elapsed:.0f}s  汇总 → {summary_path}")
    print(f"{'='*60}")

    # 简洁报表
    if results:
        print(f"\n  {'gain':>5s}  {'run':>3s}  {'帧数':>5s}  {'SNR(dB)':>8s}  {'CRC%':>6s}  {'CFO(Hz)':>10s}")
        print(f"  {'-'*45}")
        for r in results:
            nf = r.get("n_frames", "?")
            snr = f'{r["snr_prefix_mean"]:.1f}' if r.get("snr_prefix_mean") is not None else "?"
            crc = f'{r.get("crc_ok_rate",0)*100:.0f}' if r.get("crc_ok_rate") is not None else "?"
            cfo = f'{r.get("cfo_mean",0):+.0f}±{r.get("cfo_std",0):.0f}' if r.get("cfo_mean") is not None else "?"
            print(f"  {r['gain_rx']:5d}  {r['run']:3d}  {str(nf):>5s}  {snr:>8s}  {crc:>6s}  {cfo:>10s}")


if __name__ == "__main__":
    main()
