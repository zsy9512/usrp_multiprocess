#!/usr/bin/env python3
"""
分析工具：analyze_rxdata.py
--------------------------
用于分析usrp_scope.py记录的接收数据（如rxdata.bin），支持：
1. 原始时域和频谱显示
2. 帧定位、同步、解调与BER/星座图展示
3. 命令行参数自定义

用法示例：
python analyze_rxdata.py --file rxdata.bin --rate 1e6 --sps 2

后续将逐步完善各功能。
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random


# 导入DQPSK系统
from dqpsk_system import USRP_DQPSK_System

def parse_args():
    parser = argparse.ArgumentParser(description="USRP DQPSK接收数据分析工具")
    parser.add_argument('--file', type=str, required=True, help='接收数据文件（如rxdata.bin）')
    parser.add_argument('--rate', type=float, default=1e6, help='采样率 (Hz)')
    parser.add_argument('--sps', type=int, default=2, help='每符号采样点数')
    parser.add_argument('--roll_off', type=float, default=0.35, help='RRC滚降系数')
    parser.add_argument('--center_freq', type=float, default=915e6, help='中心频率 (Hz)，默认915e6')
    parser.add_argument('--show', action='store_true', help='显示原始时域和频谱')
    parser.add_argument('--analyze', action='store_true', help='执行帧同步与解调分析')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"分析文件: {args.file}\n采样率: {args.rate} Hz, sps: {args.sps}, roll_off: {args.roll_off}, center_freq: {args.center_freq}")

    # 读取数据文件（假定为complex64二进制）
    try:
        rxdata = np.fromfile(args.file, dtype=np.complex64)
    except Exception as e:
        print(f"读取数据文件失败: {e}")
        return
    print(f"数据长度: {len(rxdata)} 样本 ({len(rxdata)/args.sps:.1f} 符号)")

    if args.show:
        # 仅显示时域波形
        plt.figure(figsize=(12,4))
        plt.plot(np.real(rxdata), label='I')
        plt.plot(np.imag(rxdata), label='Q', alpha=0.7)
        plt.title('Received Signal (Time Domain)')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    if args.analyze:
        # 1. 周期分割
        dqpsk = USRP_DQPSK_System(
            mode="simulation",
            center_freq=args.center_freq,
            samp_rate=args.rate,
            sps=args.sps,
            roll_off=args.roll_off,
            verbose=False
        )
        env = np.abs(rxdata)
        win_size = 2000
        env_smooth = np.convolve(env, np.ones(win_size)/win_size, mode='same')
        threshold = np.max(env_smooth) * 0.1
        low_regions = (env_smooth < threshold).astype(int)
        edges = np.where(np.diff(low_regions) == 1)[0]
        # 可视化周期分割
        plt.figure(figsize=(12,3))
        plt.plot(env_smooth, label='Envelope (Smoothed)')
        for idx in edges:
            plt.axvline(idx, color='r', alpha=0.5)
        plt.title('Smoothed Envelope and Detected Period Gaps')
        plt.xlabel('Sample Index')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 2. 周期筛选与随机挑段
        periods = []
        shift = 100000
        if len(edges) > 1:
            for i in range(len(edges) - 1):
                start = max(0, edges[i] - shift)
                end = max(0, edges[i + 1] - shift)
                periods.append((start, end))
        else:
            periods.append((0, min(len(rxdata), int(args.rate))))
        energy_list = [np.sum(np.abs(rxdata[start:end])**2) for (start, end) in periods]
        energy_threshold = np.max(energy_list) * 0.2
        valid_periods = [(start, end) for (start, end) in periods if np.sum(np.abs(rxdata[start:end])**2) > energy_threshold]
        if not valid_periods:
            print("[ERROR] No valid periods detected")
            return
        # 随机挑一段
        selected_period = random.choice(valid_periods)
        start_idx, end_idx = selected_period
        selected_segment = rxdata[start_idx:end_idx]
        print(f"[INFO] Selected segment: {start_idx} to {end_idx} (length {len(selected_segment)} samples)")

        # 进一步包络能量检测找到有数据的部分
        envelope = np.abs(selected_segment)
        env_smooth = np.convolve(envelope, np.ones(5000)/5000, mode='same')
        threshold = np.max(env_smooth) * 0.3
        above = (env_smooth > threshold).astype(int)
        frame_starts = np.where(np.diff(above) == 1)[0]
        frame_ends = np.where(np.diff(above) == -1)[0]
        # 只保留成对的起止点
        min_len = min(len(frame_starts), len(frame_ends))
        frame_starts = frame_starts[:min_len]
        frame_ends = frame_ends[:min_len]
        print(f"[INFO] Detected {len(frame_starts)} active signal regions")

        # 合并活跃区域为一个大的数据区域
        if len(frame_starts) > 0 and len(frame_ends) > 0:
            merged_start = frame_starts[0]
            merged_end = frame_ends[-1]
            merged_segment = selected_segment[merged_start:merged_end]
            print(f"[INFO] Merged active regions into: {merged_start}-{merged_end} ({len(merged_segment)} samples)")
        else:
            print("[INFO] No active regions, skipping.")
            return

        # 可视化包络和活跃区域（合并前）
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(envelope, alpha=0.7, label='Raw Envelope')
        plt.plot(env_smooth, 'r-', linewidth=2, label='Smoothed Envelope')
        plt.axhline(threshold, color='g', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.title('Envelope Detection in Selected Segment')
        plt.xlabel('Sample Index')
        plt.ylabel('Envelope Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(envelope, alpha=0.7, label='Raw Envelope')
        for start, end in zip(frame_starts, frame_ends):
            plt.axvspan(start, end, alpha=0.3, color='red', label='Active Region' if start == frame_starts[0] else "")
        plt.axvspan(merged_start, merged_end, alpha=0.2, color='blue', label='Merged Region')
        plt.title(f'Active Regions Detected ({len(frame_starts)} regions) and Merged')
        plt.xlabel('Sample Index')
        plt.ylabel('Envelope Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 3. 在合并的数据区域内滑动窗口检测
        detected_starts = []
        frame_len_samples = 1536  # 768 * 2 (sps)
        print(f"[Merged Region] Analyzing: {merged_start}-{merged_end} ({len(merged_segment)} samples)")

        # 滑动窗口参数
        win_len = 1800
        overlap = 500
        step = win_len - overlap
        num_windows = max(1, (len(merged_segment) - win_len) // step + 1)

        for w in range(num_windows):
            win_start_rel = w * step
            win_end_rel = win_start_rel + win_len
            if win_end_rel > len(merged_segment):
                win_end_rel = len(merged_segment)
                win_start_rel = max(0, win_end_rel - win_len)
            window = merged_segment[win_start_rel:win_end_rel]

            # 匹配滤波和下采样
            mf = np.convolve(window, dqpsk.rrc_filter, mode='full')
            #rrc_delay = (len(dqpsk.rrc_filter) - 1) // 2
            #mf = mf[rrc_delay:rrc_delay + len(window)]
            symbols = mf[::dqpsk.sps]

            # 使用 _enhanced_pss_sync 检测帧起始
            timing_offset = dqpsk._enhanced_pss_sync(symbols)

            if timing_offset >= 0 and timing_offset + 128 <= len(symbols):
                detected_start = merged_start + win_start_rel + timing_offset * dqpsk.sps
                # 检查帧完整性（帧不能超过窗口长度）
                if detected_start + frame_len_samples <= merged_start + win_end_rel:
                    # 帧间隔约束，去重
                    if len(detected_starts) == 0 or detected_start - detected_starts[-1] > frame_len_samples * 0.8:
                        detected_starts.append(detected_start)

                        # 提取帧并解调
                        frame_start = timing_offset
                        frame_end = frame_start + dqpsk.preamble_len + dqpsk.data_symbols
                        frame_syms = symbols[frame_start:frame_end]
                        print("frame_syms",len(frame_syms))
                        coarse_freq = dqpsk._enhanced_sss_sync( frame_syms, 0)
                        fine_freq = dqpsk._enhanced_rs_sync( frame_syms, 0, coarse_freq)
                        total_freq = coarse_freq + fine_freq

                        # 频偏校正
                        n = np.arange(len(frame_syms))
                        frame_syms_corr = frame_syms * np.exp(-1j * 2 * np.pi * total_freq * n * dqpsk.Ts)
                        # 数据提取
                        data_start =  dqpsk.preamble_len
                        data_symbols = frame_syms_corr[data_start:]
                        print("data_symbols",len(data_symbols))        
                        # 相位同步
                        costas = dqpsk._init_costas_loop(loop_bw=0.001)
                        synchronized_symbols = costas.process(data_symbols)
                        # 差分解码
                        demod_symbols = dqpsk.differential_decode(synchronized_symbols)
                        print("demod_symbols",len(demod_symbols))
                        # BER 计算
                        ref_frame, ref_bits = dqpsk.generate_frame(return_bits=True)
                        rx_bits = dqpsk._symbols_to_bits(demod_symbols)
                        ber = dqpsk._calculate_ber(ref_bits, rx_bits)

                        print(f"[Window {w} in Merged {merged_start}-{merged_end}] Detected frame at {detected_start}, Freq offset: {total_freq:.2f} Hz, BER: {ber:.6f}")

                        # 保存当前窗口数据整体
                        np.save(f'window_{w}_{int(detected_start)}.npy', window)

                        # 展示每个窗口的详细结果
                        plt.figure(figsize=(12, 8))

                        # IQ 时域数据（标注峰值）
                        plt.subplot(2, 2, 1)
                        plt.plot(np.real(window), label='I')
                        plt.plot(np.imag(window), label='Q')
                        peak_sample = timing_offset * dqpsk.sps
                        if peak_sample < len(window):
                            plt.axvline(peak_sample, color='r', linestyle='--', label='Detected Peak')
                        plt.title(f'Window {w} IQ Time Domain')
                        plt.xlabel('Sample Index')
                        plt.ylabel('Amplitude')
                        plt.legend()

                        # 总估计频偏
                        plt.subplot(2, 2, 2)
                        plt.bar(['Coarse', 'Fine', 'Total'], [coarse_freq, fine_freq, total_freq])
                        plt.title(f'Freq Offsets: Total {total_freq:.1f} Hz')

                        # 最终恒定星座图
                        plt.subplot(2, 2, 3)
                        plt.scatter(np.real(demod_symbols), np.imag(demod_symbols))
                        plt.title('Demodulated Constellation')
                        plt.xlabel('I')
                        plt.ylabel('Q')
                        plt.axis('equal')

                        # BER
                        plt.subplot(2, 2, 4)
                        plt.text(0.5, 0.5, f'BER: {ber:.6f}', fontsize=20, ha='center', va='center')
                        plt.title('BER')
                        plt.axis('off')

                        plt.tight_layout()
                        plt.show()
                else:
                    print(f"[Window {w}] Frame exceeds window length, discarded.")        # 5. 输出和统计
        print(f"Total detected frames: {len(detected_starts)}")
        if detected_starts:
            print(f"Detected starts: {detected_starts}")

if __name__ == "__main__":
    main()
