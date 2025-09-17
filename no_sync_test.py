import numpy as np
import sys
import os

# 添加路径以导入dqpsk_system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from usrp_multiprocess.dqpsk_system import USRP_DQPSK_System

def test_no_sync_simulation():
    """测试仿真但跳过同步步骤，直接使用已知参数"""
    print("=== 无同步仿真测试（使用已知参数） ===")

    # 创建DQPSK系统
    dqpsk = USRP_DQPSK_System()

    # 生成测试数据
    np.random.seed(42)
    data_bits = np.random.randint(0, 2, 1280)  # 1280比特 = 640符号
    print(f"原始比特长度: {len(data_bits)}")

    # 1. 比特到符号映射
    data_symbols = dqpsk._bits_to_symbols(data_bits)
    print(f"数据符号长度: {len(data_symbols)}")

    # 2. 差分编码
    encoded_symbols = dqpsk.differential_encode(data_symbols)
    print(f"编码符号长度: {len(encoded_symbols)}")

    # 3. 上采样和滤波（发射端）
    tx_signal = dqpsk.prepare_tx_signal(encoded_symbols)
    print(f"发射信号长度: {len(tx_signal)}")

    # 4. 添加信道效应（噪声+频偏）
    # 添加AWGN噪声
    snr_db = 20  # 高SNR
    snr_linear = 10**(snr_db/10)
    noise_power = np.var(tx_signal) / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))
    rx_signal = tx_signal + noise

    # 添加频偏
    freq_offset = 500.0  # 500 Hz
    Ts = 1.0 / dqpsk.samp_rate
    n = np.arange(len(rx_signal))
    phase_offset = np.exp(1j * 2 * np.pi * freq_offset * n * Ts)
    rx_signal = rx_signal * phase_offset

    print(f"添加噪声: SNR={snr_db}dB")
    print(f"添加频偏: {freq_offset}Hz")

    # 5. 匹配滤波
    filtered = np.convolve(rx_signal, dqpsk.rrc_filter, mode='full')
    filter_delay = len(dqpsk.rrc_filter) // 2
    print(f"滤波后长度: {len(filtered)}, 滤波器延迟: {filter_delay}")

    # 6. 下采样（使用已知参数）
    rx_symbols = filtered[filter_delay::dqpsk.sps]
    print(f"下采样后符号长度: {len(rx_symbols)}")

    # 7. 跳过PSS同步，直接使用已知的数据起始位置
    # 在理想情况下，数据应该从索引0开始（假设没有前导）
    data_start = 0  # 跳过同步，直接从开头提取
    data_end = data_start + dqpsk.data_symbols
    data_symbols_extracted = rx_symbols[data_start:data_end]
    print(f"提取数据符号长度: {len(data_symbols_extracted)}")

    # 8. 频率校正（使用已知频偏）
    n_sym = np.arange(len(data_symbols_extracted))
    freq_correction = np.exp(-1j * 2 * np.pi * freq_offset * n_sym * Ts * dqpsk.sps)
    corrected_symbols = data_symbols_extracted * freq_correction
    print(f"频率校正后符号能量: {np.mean(np.abs(corrected_symbols)):.4f}")

    # 9. 跳过Costas环，直接使用频偏校正后的符号
    synchronized_symbols = corrected_symbols.copy()
    print(f"跳过Costas环，使用纯频偏校正")

    # 10. 差分解码
    demod_symbols = dqpsk.differential_decode(synchronized_symbols)
    print(f"差分解码后符号数: {len(demod_symbols)}")

    # 11. 符号到比特转换
    decoded_bits = dqpsk._symbols_to_bits(demod_symbols)
    print(f"解码比特长度: {len(decoded_bits)}")

    # 12. 计算BER
    min_len = min(len(data_bits), len(decoded_bits))
    data_bits = data_bits[:min_len]
    decoded_bits = decoded_bits[:min_len]

    errors = np.sum(data_bits != decoded_bits)
    ber = errors / len(data_bits)
    print(f"BER: {ber} (错误数: {errors}/{len(data_bits)})")

    # 检查星座分布
    constellation_points = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
    distances = []
    for sym in demod_symbols[:min(50, len(demod_symbols))]:
        dist = np.min(np.abs(sym - constellation_points))
        distances.append(dist)
    avg_distance = np.mean(distances)
    print(f"平均到星座点距离: {avg_distance:.6f}")

    return ber

if __name__ == "__main__":
    print("开始无同步仿真测试...\n")

    ber = test_no_sync_simulation()

    print("\n=== 测试结果 ===")
    if ber == 0.0:
        print("无同步测试BER为0 - 同步算法是问题根源")
    else:
        print(f"无同步测试BER仍为{ber} - 问题不在于同步")