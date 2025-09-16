#!/usr/bin/env python3
"""
简单的DQPSK仿真测试文件 - 测试_process_frame_packet性能
按照transmit_and_receive的核心思想，简化实现
"""

import numpy as np
import time
from dqpsk_system import USRP_DQPSK_System

def create_frame_packet(rx_signal, tx_bits, frame_id=0):
    """创建帧数据包，模拟SimulationIPC的格式"""
    return {
        'frame_id': frame_id,
        'rx_signal': rx_signal,
        'tx_signal': None,  # 测试中不需要
        'tx_bits': tx_bits,
        'timestamp': time.time(),
        'metadata': {}
    }

def apply_channel_effects(tx_signal, snr_db, freq_offset, phase_offset, samp_rate):
    """应用信道效应：频偏、相偏、AWGN噪声"""
    n = np.arange(len(tx_signal))

    # 应用频率偏移
    tx_signal = tx_signal * np.exp(1j * 2 * np.pi * freq_offset * n / samp_rate)

    # 应用相位偏移
    tx_signal = tx_signal * np.exp(1j * phase_offset)

    # 添加AWGN噪声
    signal_power = np.mean(np.abs(tx_signal)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
    rx_signal = tx_signal + noise

    return rx_signal

def test_process_frame_packet_performance():
    """测试_process_frame_packet的性能"""
    print("=== DQPSK _process_frame_packet 性能测试 ===\n")

    # 创建DQPSK系统实例
    dqpsk_system = USRP_DQPSK_System(
        mode="simulation",
        samp_rate=1e6,
        sps=2,
        verbose=True
    )

    # 测试参数
    test_configs = [
        {"snr_db": 20, "freq_offset": 500, "phase_offset": 0.0, "name": "高SNR，低频偏"},
        {"snr_db": 15, "freq_offset": 1000, "phase_offset": np.pi/6, "name": "中等SNR，中等频偏"},
        {"snr_db": 10, "freq_offset": 2000, "phase_offset": np.pi/3, "name": "低SNR，高频偏"},
    ]

    n_frames = 5  # 每个配置测试的帧数
    total_results = []

    for config in test_configs:
        print(f"测试配置: {config['name']}")
        print(f"  SNR: {config['snr_db']} dB")
        print(f"  频偏: {config['freq_offset']} Hz")
        print(f"  相偏: {config['phase_offset']:.3f} rad")
        print(f"  测试帧数: {n_frames}")

        ber_results = []
        processing_times = []

        for frame_idx in range(n_frames):
            # 1. 生成帧数据
            frame, tx_bits = dqpsk_system.generate_frame(return_bits=True)
            tx_signal = dqpsk_system.prepare_tx_signal(frame)

            # 2. 应用信道效应
            rx_signal = apply_channel_effects(
                tx_signal,
                config['snr_db'],
                config['freq_offset'],
                config['phase_offset'],
                dqpsk_system.samp_rate
            )

            # 3. 创建帧数据包
            frame_packet = create_frame_packet(rx_signal, tx_bits, frame_idx)

            # 4. 测试_process_frame_packet性能
            start_time = time.time()

            # 这里我们需要模拟ProcessingProgram的调用
            # 由于_process_frame_packet是私有方法，我们需要创建一个测试实例
            success = test_single_frame(dqpsk_system, frame_packet)

            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)

            if isinstance(success, (int, float)):
                ber_results.append(success)
                print(f"  帧 {frame_idx}: BER={success:.2e}")
            else:
                print(f"  帧 {frame_idx}: 处理失败")


    # 计算统计结果
    avg_processing_time = np.mean(processing_times)
    std_processing_time = np.std(processing_times)
    avg_ber = np.mean(ber_results) if ber_results else float('nan')
    print(f"  平均处理时长: {avg_processing_time:.4f}s ± {std_processing_time:.4f}s")
    print(f"  平均BER: {avg_ber:.4e}")
    print()

    total_results.append({
            'config': config,
            'avg_time': avg_processing_time,
            'std_time': std_processing_time,
            'ber_results': ber_results
        })

    # 输出总体统计
    print("=== 总体统计 ===")
    for result in total_results:
        config = result['config']
        avg_ber = np.mean(result['ber_results']) if result['ber_results'] else float('nan')
        print(f"配置: {config['name']}, 平均BER: {avg_ber:.4e}, 平均处理时长: {result['avg_time']:.4f}s ± {result['std_time']:.4f}s")

def test_single_frame(dqpsk_system, frame_packet):
    """测试单帧处理 - 模拟_process_frame_packet的核心逻辑"""
    try:
        frame_id = frame_packet.get('frame_id', 'unknown')
        rx_signal = frame_packet.get('rx_signal')
        tx_bits = frame_packet.get('tx_bits')

        if rx_signal is None or len(rx_signal) == 0:
            print(f"帧 {frame_id}: 接收信号为空")
            return False

        # 1. 匹配滤波
        filtered = np.convolve(rx_signal, dqpsk_system.rrc_filter, mode='full')
        if len(filtered) < 2000:  # 确保有足够的数据
            print(f"帧 {frame_id}: 滤波后数据不足")
            return False

        # 下采样得到符号
        rx_symbols = filtered[::dqpsk_system.sps]

        # 2. PSS同步
        timing_offset = dqpsk_system._enhanced_pss_sync(rx_symbols)

        # 3. SSS同步
        coarse_freq = dqpsk_system._enhanced_sss_sync(rx_symbols, timing_offset)

        # 4. RS同步
        fine_freq = dqpsk_system._enhanced_rs_sync(rx_symbols, timing_offset, coarse_freq)
        total_freq = coarse_freq + fine_freq

        # 5. 频率校正
        n = np.arange(len(rx_symbols))
        Ts = 1.0 / dqpsk_system.symbol_rate
        phase_correction = np.exp(-1j * 2 * np.pi * total_freq * n * Ts)
        rx_symbols_corr = rx_symbols * phase_correction

        # 6. 数据提取
        data_start = timing_offset + dqpsk_system.preamble_len
        data_end = data_start + dqpsk_system.data_symbols

        # 增加详细调试信息
        if data_start >= len(rx_symbols_corr) or data_end > len(rx_symbols_corr) or data_end - data_start < 100:
            print(f"帧 {frame_id}: 数据提取范围无效: start={data_start}, end={data_end}, total={len(rx_symbols_corr)}, "
                  f"timing_offset={timing_offset}, preamble_len={dqpsk_system.preamble_len}, data_symbols={dqpsk_system.data_symbols}")
            return False

        data_symbols = rx_symbols_corr[data_start:data_end]

        # 7. Costas环相位同步
        costas = dqpsk_system._init_costas_loop(loop_bw=0.002)
        synchronized_symbols = costas.process(data_symbols)

        # 8. 差分解码
        demod_symbols = dqpsk_system.differential_decode(synchronized_symbols)

        # 9. 符号到比特转换
        recv_bits = dqpsk_system._symbols_to_bits(demod_symbols)

        # 10. 计算BER
        if tx_bits is not None and len(tx_bits) == len(recv_bits):
            errors = np.sum(tx_bits != recv_bits)
            ber = errors / len(tx_bits)
            return ber
        else:
            print(f"帧 {frame_id}: BER计算失败，比特长度不匹配")
            return False

    except Exception as e:
        print(f"帧处理错误: {str(e)}")
        return False

if __name__ == "__main__":
    test_process_frame_packet_performance()