#!/usr/bin/env python3
"""
测试脚本：test_frame_detection.py
-------------------------------
按照用户思路测试帧检测：
1. 生成768符号帧，前128为帧头（32 PSS + 32 SSS + 64 RS）
2. 前后加随机噪声，记录真实位置
3. 模拟信道，检测帧位置并对比
"""
import numpy as np
import matplotlib.pyplot as plt
from dqpsk_system import USRP_DQPSK_System



def estimate_freq_offset(signal, ref, Ts):
    # 改进的频偏估计
    corr = np.sum(signal * np.conj(ref))
    phase_diff = np.angle(corr)
    N = len(ref)
    time_span = (N - 1) * Ts
    return phase_diff / (2 * np.pi * time_span)


def test_frame_detection_continuous():
    dqpsk = USRP_DQPSK_System(mode="simulation", samp_rate=1e6, sps=2, roll_off=0.35, verbose=False)
    frame, _ = dqpsk.generate_frame(return_bits=True)
    frame_header = frame[:128]
    n_frames = 10
    tx_signal = np.concatenate([dqpsk.prepare_tx_signal(frame) for _ in range(n_frames)])
    pre_noise = 0.01 * (np.random.randn(100) + 1j * np.random.randn(100))
    post_noise = 0.01 * (np.random.randn(100) + 1j * np.random.randn(100))
    tx_signal = np.concatenate([pre_noise, tx_signal, post_noise])
    frame_len_samples = len(dqpsk.prepare_tx_signal(frame))
    true_starts = [100 + i * frame_len_samples for i in range(n_frames)]
    freq_offset = -800.0
    noise_level = 0.01
    n = np.arange(len(tx_signal))
    rx_signal = tx_signal * np.exp(1j * 2 * np.pi * freq_offset * n * dqpsk.Ts)
    rx_signal += noise_level * (np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal)))

    win_len = 2000
    overlap = 200
    step = win_len - overlap
    num_windows = (len(rx_signal) - win_len) // step + 1
    detected_starts = []
    window_starts = []
    window_ends = []
    for w in range(num_windows):
        win_start = w * step
        win_end = win_start + win_len
        window_starts.append(win_start)
        window_ends.append(win_end)
        window = rx_signal[win_start:win_end]
        
        # 匹配滤波和下采样到符号
        mf = np.convolve(window, dqpsk.rrc_filter, mode='full')
        #rrc_delay = (len(dqpsk.rrc_filter) - 1) // 2
        #mf = mf[rrc_delay:]
        symbols = mf[::dqpsk.sps]
        
        # 使用整个帧头（128符号）进行相关检测
        #corr = np.abs(np.correlate(symbols, frame_header, mode='valid'))
        #timing_offset = np.argmax(corr)
        timing_offset = dqpsk._enhanced_pss_sync(symbols)
        # 检查是否有效检测（timing_offset 应为帧头起始符号索引）
        if timing_offset >= 0 and timing_offset + 128 <= len(symbols):
            detected_start = win_start + timing_offset * dqpsk.sps
            # 帧间隔约束，去重
            if len(detected_starts) == 0 or detected_start - detected_starts[-1] > frame_len_samples * 0.8:
                detected_starts.append(detected_start)
        coarse_freq =dqpsk._enhanced_sss_sync(symbols, timing_offset )
        print(f"Window {w}: Detected timing offset {timing_offset},Detecd start {detected_start}, Coarse freq offset {coarse_freq:.1f} Hz")
    print(f"True frame starts: {true_starts}")
    print(f"Detected frame starts: {detected_starts}")
    # 计算每个检测帧与最近真值帧的误差
    detection_errors = []
    for d in detected_starts:
        closest_t = min(true_starts, key=lambda t: abs(d - t))
        error = d - closest_t
        detection_errors.append(error)
    print(f"Detection errors: {detection_errors}")

    # 可视化
    plt.figure(figsize=(12,5))
    plt.plot(np.abs(rx_signal), label='RX Signal Magnitude')
    for t in true_starts:
        plt.axvline(t, color='g', linestyle='--', label='True Start' if t==true_starts[0] else None)
    for d in detected_starts:
        plt.axvline(d, color='r', linestyle='-', label='Detected Start' if d==detected_starts[0] else None)
    # 标注滑动窗口的起点和终点
    for ws, we in zip(window_starts, window_ends):
        plt.axvline(ws, color='b', linestyle=':', alpha=0.7, label='Window Start' if ws == window_starts[0] else None)
        plt.axvline(we, color='b', linestyle=':', alpha=0.7, label='Window End' if we == window_ends[0] else None)
    plt.title('Frame Start Detection (Green: True, Red: Detected, Blue: Windows)')
    plt.xlabel('Sample Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_frame_detection_continuous()