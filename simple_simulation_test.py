#!/usr/bin/env python3
"""
简单的DQPSK仿真测试文件 - 测试_process_frame_packet性能
按照transmit_and_receive的核心思想，简化实现
"""

import numpy as np
import time
import matplotlib.pyplot as plt
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
        #{"snr_db": 20, "freq_offset": 500, "phase_offset": 0.0, "name": "高SNR，低频偏"},
        #{"snr_db": 15, "freq_offset": 1000, "phase_offset": np.pi/4, "name": "中等SNR，中等频偏"},
        {"snr_db": 10, "freq_offset": 2000, "phase_offset": np.pi/3, "name": "低SNR，高频偏"},
    ]

    n_frames = 1  # 每个配置测试的帧数
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
            # === 新增：随机插入噪声样本到两端 ===
            total_len = 2000
            frame_len = len(rx_signal)
            pad_len = total_len - frame_len
            if pad_len > 0:
                pre_len = 112
                post_len = pad_len - pre_len
                signal_power = np.mean(np.abs(rx_signal)**2)
                noise_power = signal_power / (10**(config['snr_db']/10))
                pre_noise = np.sqrt(noise_power/2) * (np.random.randn(pre_len) + 1j * np.random.randn(pre_len))
                post_noise = np.sqrt(noise_power/2) * (np.random.randn(post_len) + 1j * np.random.randn(post_len))
                rx_signal = np.concatenate([pre_noise, rx_signal, post_noise])
                # 记录真实帧头位置
                true_frame_start = pre_len

            else:
                true_frame_start = 0

            # 3. 创建帧数据包
            frame_packet = create_frame_packet(rx_signal, tx_bits, frame_idx)
            frame_packet['true_frame_start'] = true_frame_start  # 传递真实帧头位置

            print(f"  帧 {frame_idx}: 生成帧长度={len(frame)}, true_frame_start={true_frame_start}")
            # 4. 测试_process_frame_packet性能
            start_time = time.time()

            # 这里我们需要模拟ProcessingProgram的调用
            # 由于_process_frame_packet是私有方法，我们需要创建一个测试实例
            success = test_single_frame(dqpsk_system, frame_packet)

            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)

            if isinstance(success, (int, float)) and success >= 0:
                ber_results.append(success)
                print(f"  帧 {frame_idx}: BER={success:.2e}")
            else:
                print(f"  帧 {frame_idx}: 处理失败")
                ber_results.append(float('nan'))  # 记录失败帧



        # 计算统计结果
        avg_processing_time = np.mean(processing_times)
        std_processing_time = np.std(processing_times)
        
        # 过滤掉 nan 值计算平均BER
        valid_ber = [b for b in ber_results if not np.isnan(b)]
        avg_ber = np.mean(valid_ber) if valid_ber else float('nan')
        
        print(f"  平均处理时长: {avg_processing_time:.4f}s ± {std_processing_time:.4f}s")
        print(f"  平均BER: {avg_ber:.4e} (成功帧数: {len(valid_ber)}/{len(ber_results)})")
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
        valid_ber = [b for b in result['ber_results'] if not np.isnan(b)]
        avg_ber = np.mean(valid_ber) if valid_ber else float('nan')
        print(f"配置: {config['name']}, 平均BER: {avg_ber:.4e}, 平均处理时长: {result['avg_time']:.4f}s ± {result['std_time']:.4f}s")
def test_from_saved_window(file_path):
    """从保存的窗口数据文件进行同步和解调测试（数据已下采样为符号）"""
    print(f"=== 从文件 {file_path} 加载窗口数据进行测试 ===")

    # 加载窗口数据（已下采样为符号）
    try:
        rx_symbols = np.load(file_path)
        print(f"加载窗口数据: 长度={len(rx_symbols)} 符号")
    except Exception as e:
        print(f"加载文件失败: {e}")
        return

    # 创建DQPSK系统实例
    dqpsk_system = USRP_DQPSK_System(
        mode="simulation",
        samp_rate=1e6,
        sps=2,
        verbose=True
    )

    # 模拟帧数据包（没有tx_bits，因为是真实数据）
    frame_packet = {
        'frame_id': 0,
        'rx_symbols': rx_symbols,  # 直接使用已下采样的符号
        'tx_signal': None,
        'tx_bits': None,
        'timestamp': time.time(),
        'metadata': {}
    }

    # 调用修改后的test_single_frame进行处理
    result = test_single_frame_from_symbols(dqpsk_system, frame_packet)
    if isinstance(result, bool) and result:
        print("处理成功")
    else:
        print("处理失败")
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
        
        # 添加调试信息
        print(f"帧 {frame_id}: rx_signal长度={len(rx_signal)}, filtered长度={len(filtered)}, rrc_filter长度={len(dqpsk_system.rrc_filter)}")
        
        rx_symbols = filtered[::dqpsk_system.sps]
        print(f"帧 {frame_id}: 下采样后符号长度={len(rx_symbols)},sps={dqpsk_system.sps}    ")

        # 3. PSS同步
        timing_offset = dqpsk_system._enhanced_pss_sync(rx_symbols)
        print(f"帧 {frame_id}: PSS同步偏移={timing_offset}")

        true_frame_start = frame_packet.get('true_frame_start', None)
        # 输出帧同步偏移误差
        if true_frame_start is not None:
            # 计算真实帧头在rx_symbols中的索引
            filter_delay = len(dqpsk_system.rrc_filter) // 2
            expected_offset = (true_frame_start + filter_delay) // dqpsk_system.sps
            sync_error = timing_offset - expected_offset
            print(f"帧 {frame_id}: 帧同步偏移误差 = {sync_error} (检测={timing_offset}, 理论={expected_offset})")
        # 4. SSS同步
        coarse_freq = dqpsk_system._enhanced_sss_sync(rx_symbols, timing_offset)
        print(f"帧 {frame_id}: SSS粗频估计={coarse_freq:.2f} Hz")

        # 5. RS同步
        fine_freq = dqpsk_system._enhanced_rs_sync(rx_symbols, timing_offset, coarse_freq)
        total_freq = coarse_freq + fine_freq
        print(f"帧 {frame_id}: RS细频估计={fine_freq:.2f} Hz, 总频偏={total_freq:.2f} Hz")

        # 6. 频率校正
        n = np.arange(len(rx_symbols))
        Ts = 1.0 / dqpsk_system.symbol_rate
        phase_correction = np.exp(-1j * 2 * np.pi * total_freq * n * Ts)
        rx_symbols_corr = rx_symbols * phase_correction
        print(f"帧 {frame_id}: 频率校正后，符号能量={np.mean(np.abs(rx_symbols_corr)**2):.4f}")

        # 7. 数据提取 - 尝试调整起始位置以优化符号定时
        # 原始计算
        data_start = timing_offset + dqpsk_system.preamble_len
        data_end = data_start + dqpsk_system.data_symbols
        data_symbols = rx_symbols_corr[data_start:data_end]
                   # 6. 相位同步
        costas = dqpsk_system._init_costas_loop(loop_bw=0.001)
        synchronized_symbols = costas.process(data_symbols)

        demod_symbols = dqpsk_system.differential_decode(synchronized_symbols)
        print(f"帧 {frame_id}: 差分解码后，符号数={len(demod_symbols)}, 星座分布检查...")
        
        # 检查星座分布
        constellation_points = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        distances = []
        for sym in demod_symbols[:min(50, len(demod_symbols))]:  # 检查前50个符号
            dist = np.min(np.abs(sym - constellation_points))
            distances.append(dist)
        avg_distance = np.mean(distances)
        print(f"帧 {frame_id}: 平均到星座点距离={avg_distance:.4f}")

        # 10. 符号到比特转换
        recv_bits = dqpsk_system._symbols_to_bits(demod_symbols)
        print(f"帧 {frame_id}: 解码比特长度={len(recv_bits)}")

        # 11. 计算BER
        if tx_bits is not None and len(tx_bits) == len(recv_bits):
            errors = np.sum(tx_bits != recv_bits)
            ber = errors / len(tx_bits)
            print(f"帧 {frame_id}: BER计算成功, errors={errors}, total_bits={len(tx_bits)}, ber={ber:.2e}")
            
            # 额外检查：比较前20个比特
            if len(tx_bits) >= 20:
                print(f"帧 {frame_id}: 发送比特前20: {tx_bits[:20]}")
                print(f"帧 {frame_id}: 接收比特前20: {recv_bits[:20]}")
                bit_errors = np.sum(tx_bits[:20] != recv_bits[:20])
                print(f"帧 {frame_id}: 前20比特错误数: {bit_errors}/20")
            
            # 可视化
            plt.figure(figsize=(12, 8))

            # IQ 时域数据（标注峰值）
            plt.subplot(2, 2, 1)
            plt.plot(np.real(rx_signal), label='I')
            plt.plot(np.imag(rx_signal), label='Q')
            peak_sample = timing_offset * dqpsk_system.sps
            if peak_sample < len(rx_signal):
                plt.axvline(peak_sample, color='r', linestyle='--', label='Detected Peak')
            plt.title('IQ Time Domain')
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
            
            return ber
        else:
            print(f"帧 {frame_id}: BER计算失败，比特长度不匹配 ({len(tx_bits) if tx_bits is not None else 0} vs {len(recv_bits)})")
            # 即使没有BER，也进行可视化
            plt.figure(figsize=(12, 8))

            # IQ 时域数据（标注峰值）
            plt.subplot(2, 2, 1)
            plt.plot(np.real(rx_signal), label='I')
            plt.plot(np.imag(rx_signal), label='Q')
            peak_sample = timing_offset * dqpsk_system.sps
            if peak_sample < len(rx_signal):
                plt.axvline(peak_sample, color='r', linestyle='--', label='Detected Peak')
            plt.title('IQ Time Domain')
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
            plt.text(0.5, 0.5, 'BER: N/A', fontsize=20, ha='center', va='center')
            plt.title('BER')
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            return True

    except Exception as e:
        print(f"帧处理错误: {str(e)}")
        return False

def test_single_frame_from_symbols(dqpsk_system, frame_packet):
    """测试单帧处理 - 直接从符号开始（跳过匹配滤波和下采样）"""
    try:
        frame_id = frame_packet.get('frame_id', 'unknown')
        rx_symbols = frame_packet.get('rx_symbols')
        tx_bits = frame_packet.get('tx_bits')

        if rx_symbols is None or len(rx_symbols) == 0:
            print(f"帧 {frame_id}: 接收符号为空")
            return False
        
        print(f"帧 {frame_id}: 符号长度={len(rx_symbols)}")

        # PSS同步
        timing_offset = dqpsk_system._enhanced_pss_sync(rx_symbols)
        print(f"帧 {frame_id}: PSS同步偏移={timing_offset}")

        # SSS同步
        coarse_freq = dqpsk_system._enhanced_sss_sync(rx_symbols, timing_offset)
        print(f"帧 {frame_id}: SSS粗频估计={coarse_freq:.2f} Hz")

        # RS同步
        fine_freq = dqpsk_system._enhanced_rs_sync(rx_symbols, timing_offset, coarse_freq)
        total_freq = coarse_freq + fine_freq
        print(f"帧 {frame_id}: RS细频估计={fine_freq:.2f} Hz, 总频偏={total_freq:.2f} Hz")

        # 频率校正
        n = np.arange(len(rx_symbols))
        Ts = 1.0 / dqpsk_system.symbol_rate
        phase_correction = np.exp(-1j * 2 * np.pi * total_freq * n * Ts)
        rx_symbols_corr = rx_symbols * phase_correction
        print(f"帧 {frame_id}: 频率校正后，符号能量={np.mean(np.abs(rx_symbols_corr)**2):.4f}")

        # 数据提取
        data_start = timing_offset + dqpsk_system.preamble_len
        data_end = data_start + dqpsk_system.data_symbols
        data_symbols = rx_symbols_corr[data_start:data_end]
                   
        # 相位同步
        costas = dqpsk_system._init_costas_loop(loop_bw=0.001)
        synchronized_symbols = costas.process(data_symbols)

        demod_symbols = dqpsk_system.differential_decode(synchronized_symbols)
        print(f"帧 {frame_id}: 差分解码后，符号数={len(demod_symbols)}, 星座分布检查...")
        
        # 检查星座分布
        constellation_points = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        distances = []
        for sym in demod_symbols[:min(50, len(demod_symbols))]:  # 检查前50个符号
            dist = np.min(np.abs(sym - constellation_points))
            distances.append(dist)
        avg_distance = np.mean(distances)
        print(f"帧 {frame_id}: 平均到星座点距离={avg_distance:.4f}")

        # 符号到比特转换
        recv_bits = dqpsk_system._symbols_to_bits(demod_symbols)
        print(f"帧 {frame_id}: 解码比特长度={len(recv_bits)}")

        # 计算BER
        if tx_bits is not None and len(tx_bits) == len(recv_bits):
            errors = np.sum(tx_bits != recv_bits)
            ber = errors / len(tx_bits)
            print(f"帧 {frame_id}: BER计算成功, errors={errors}, total_bits={len(tx_bits)}, ber={ber:.2e}")
            
            # 额外检查：比较前20个比特
            if len(tx_bits) >= 20:
                print(f"帧 {frame_id}: 发送比特前20: {tx_bits[:20]}")
                print(f"帧 {frame_id}: 接收比特前20: {recv_bits[:20]}")
                bit_errors = np.sum(tx_bits[:20] != recv_bits[:20])
                print(f"帧 {frame_id}: 前20比特错误数: {bit_errors}/20")
            
            # 可视化
            plt.figure(figsize=(12, 8))

            # IQ 时域数据（标注峰值）
            plt.subplot(2, 2, 1)
            plt.plot(np.real(rx_symbols), label='I')
            plt.plot(np.imag(rx_symbols), label='Q')
            if timing_offset < len(rx_symbols):
                plt.axvline(timing_offset, color='r', linestyle='--', label='Detected Peak')
            plt.title('IQ Time Domain')
            plt.xlabel('Symbol Index')
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
            
            return ber
        else:
            print(f"帧 {frame_id}: BER计算失败，比特长度不匹配 ({len(tx_bits) if tx_bits is not None else 0} vs {len(recv_bits)})")
            # 即使没有BER，也进行可视化
            plt.figure(figsize=(12, 8))

            # IQ 时域数据（标注峰值）
            plt.subplot(2, 2, 1)
            plt.plot(np.real(rx_symbols), label='I')
            plt.plot(np.imag(rx_symbols), label='Q')
            if timing_offset < len(rx_symbols):
                plt.axvline(timing_offset, color='r', linestyle='--', label='Detected Peak')
            plt.title('IQ Time Domain')
            plt.xlabel('Symbol Index')
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
            plt.text(0.5, 0.5, 'BER: N/A', fontsize=20, ha='center', va='center')
            plt.title('BER')
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            return True

    except Exception as e:
        print(f"帧处理错误: {str(e)}")
        return False

if __name__ == "__main__":
    # 运行性能测试
    #test_process_frame_packet_performance()
    test_from_saved_window('window_0_81743.npy')

    # 示例：从保存的窗口数据文件进行同步和解调
    # 替换为实际的文件路径，例如 'window_0_81589.npy'
    # test_from_saved_window('window_0_81589.npy')

