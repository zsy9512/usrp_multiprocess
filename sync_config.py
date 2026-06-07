"""
sync_config.py — TX/RX 统一时钟配置 (B210)

两种模式:
  host:         USRP internal 时钟 + set_time_now(time.time_ns())
                 适合 free-running 连续流 + 空口自同步
  external_ref: 外部 10MHz REF IN 锁频 + 内部时间寄存器
                 必须用 SMA 将两台 B210 的 REF IN 接到同一 10MHz 源

TX 和 RX 各自独立调用，互不依赖。
"""
from __future__ import annotations

import time
import numpy as np

# uhd 仅在硬件模式时由调用方导入；此处提供工具函数由调用方传入 uhd 模块
_UHD = None


def _uhd():
    global _UHD
    if _UHD is None:
        import uhd as _m
        _UHD = _m
    return _UHD


def configure_clock_and_time(usrp, mode: str = 'host', settle_s: float = 1.0):
    """配置 B210 时钟和时间源。

    Args:
        usrp:   uhd.usrp.MultiUSRP 实例
        mode:   'host' | 'external_ref'
        settle_s: 外部参考锁定时长 (s)

    Returns:
        无。不返回启动时间 —— 当前链路默认连续流，不依赖 timed start。
    """
    uhd = _uhd()
    if mode == 'host':
        usrp.set_clock_source("internal")
        usrp.set_time_source("internal")
        # 对齐 PC 纳秒时间 (保留为后续 --timed-start 调试用)
        pc_ns = time.time_ns()
        tspec = uhd.types.TimeSpec(pc_ns // 1_000_000_000,
                                    (pc_ns % 1_000_000_000) / 1e9)
        usrp.set_time_now(tspec)
        return

    elif mode == 'external_ref':
        usrp.set_clock_source("external")
        usrp.set_time_source("internal")   # 无 PPS，时间寄存器自由运行
        time.sleep(settle_s)
        try:
            ref_locked = usrp.get_mboard_sensor("ref_locked").to_bool()
        except Exception:
            ref_locked = False
        if not ref_locked:
            print("[sync] 警告: ref_locked=False，外部10MHz未锁定！检查REF IN连接。")
        else:
            print("[sync] ref_locked=True，外部10MHz已锁定。")
        # set_time_now 对齐 PC — 使得 TX/RX 即便 timed start 不可靠，
        # 至少时间寄存器有意义，方便调试
        pc_ns = time.time_ns()
        tspec = uhd.types.TimeSpec(pc_ns // 1_000_000_000,
                                    (pc_ns % 1_000_000_000) / 1e9)
        usrp.set_time_now(tspec)
        return

    else:
        raise ValueError(f"未知 sync_mode: {mode}，可选 'host' | 'external_ref'")


def make_timed_tx_metadata(time_spec, start_of_burst: bool = True,
                           end_of_burst: bool = False):
    """创建带时间戳的 TXMetadata (调试用, 需 `--timed-start`).

    Args:
        time_spec: uhd.types.TimeSpec
    Returns:
        uhd.types.TXMetadata
    """
    md = _uhd().types.TXMetadata()
    md.has_time_spec = True
    md.time_spec = time_spec
    md.start_of_burst = start_of_burst
    md.end_of_burst = end_of_burst
    return md


def make_timed_stream_cmd(time_spec, mode: str = 'continuous'):
    """创建带时间戳的 StreamCMD (调试用).

    Args:
        time_spec: uhd.types.TimeSpec
        mode:     'continuous' | 'num_samps'
    Returns:
        uhd.types.StreamCMD
    """
    uhd = _uhd()
    if mode == 'continuous':
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    elif mode == 'num_samps':
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    else:
        raise ValueError(f"未知 StreamMode: {mode}")
    cmd.stream_now = False
    cmd.time_spec = time_spec
    return cmd
