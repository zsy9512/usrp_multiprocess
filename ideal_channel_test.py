import numpy as np
import sys
import os

# 添加路径以导入dqpsk_system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from usrp_multiprocess.dqpsk_system import USRP_DQPSK_System

def test_ideal_channel():
    """测试理想信道条件下的BER（无噪声、无频偏、无定时偏移）"""
    print("=== 理想信道测试（无同步处理） ===")

    # 创建DQPSK系统
    dqpsk = USRP_DQPSK_System()

    # 生成随机数据比特
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

    # 4. 理想信道（无噪声、无频偏）
    rx_signal = tx_signal.copy()

    # 5. 匹配滤波（接收端）- 使用full模式避免边界效应
    filtered = np.convolve(rx_signal, dqpsk.rrc_filter[::-1], mode='full')
    print(f"滤波后信号长度: {len(filtered)}")

    # 6. 下采样（理想定时）- 需要适当的偏移以对齐
    # 由于卷积的延迟，我们需要找到正确的采样点
    filter_delay = len(dqpsk.rrc_filter) // 2
    rx_symbols = filtered[filter_delay::dqpsk.sps]
    print(f"下采样后符号长度: {len(rx_symbols)}")

    # 确保长度匹配
    min_len = min(len(data_symbols), len(rx_symbols))
    data_symbols = data_symbols[:min_len]
    rx_symbols = rx_symbols[:min_len]

    # 7. 直接差分解码（跳过所有同步）
    demod_symbols = dqpsk.differential_decode(rx_symbols)
    print(f"差分解码后符号长度: {len(demod_symbols)}")

    # 8. 符号到比特转换
    decoded_bits = dqpsk._symbols_to_bits(demod_symbols)
    print(f"解码比特长度: {len(decoded_bits)}")

    # 9. 计算BER
    errors = np.sum(data_bits != decoded_bits)
    ber = errors / len(data_bits)
    print(f"BER: {ber} (错误数: {errors}/{len(data_bits)})")

    # 检查符号是否匹配
    symbol_match = np.allclose(data_symbols, demod_symbols, atol=1e-6)
    print(f"符号完全匹配: {symbol_match}")

    # 检查星座分布
    constellation_points = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
    distances = []
    for sym in demod_symbols[:min(50, len(demod_symbols))]:
        dist = np.min(np.abs(sym - constellation_points))
        distances.append(dist)
    avg_distance = np.mean(distances)
    print(f"平均到星座点距离: {avg_distance:.6f}")

    return ber

def test_with_known_offset():
    """测试已知频偏的情况"""
    print("\n=== 已知频偏测试 ===")

    dqpsk = USRP_DQPSK_System()
    np.random.seed(42)

    # 生成数据
    data_bits = np.random.randint(0, 2, 1280)
    data_symbols = dqpsk._bits_to_symbols(data_bits)
    encoded_symbols = dqpsk.differential_encode(data_symbols)
    tx_signal = dqpsk.prepare_tx_signal(encoded_symbols)

    # 添加已知频偏
    freq_offset = 1000.0  # 1000 Hz
    Ts = 1.0 / dqpsk.samp_rate
    n = np.arange(len(tx_signal))
    phase_offset = np.exp(1j * 2 * np.pi * freq_offset * n * Ts)
    rx_signal = tx_signal * phase_offset

    print(f"添加频偏: {freq_offset} Hz")

    # 匹配滤波
    filtered = np.convolve(rx_signal, dqpsk.rrc_filter[::-1], mode='valid')
    rx_symbols = filtered[::dqpsk.sps]

    # 手动频率校正
    n_sym = np.arange(len(rx_symbols))
    freq_correction = np.exp(-1j * 2 * np.pi * freq_offset * n_sym * Ts * dqpsk.sps)
    rx_corrected = rx_symbols * freq_correction

    # 差分解码
    demod_symbols = dqpsk.differential_decode(rx_corrected)
    decoded_bits = dqpsk._symbols_to_bits(demod_symbols)

    # 计算BER
    errors = np.sum(data_bits != decoded_bits)
    ber = errors / len(data_bits)
    print(f"BER: {ber} (错误数: {errors}/{len(data_bits)})")

    return ber

if __name__ == "__main__":
    print("开始理想信道DQPSK测试...\n")

    # 测试1: 理想信道
    ideal_ber = test_ideal_channel()

    # 测试2: 已知频偏
    offset_ber = test_with_known_offset()

    print("\n=== 测试总结 ===")
    print(f"理想信道BER: {ideal_ber}")
    print(f"已知频偏BER: {offset_ber}")

    if ideal_ber == 0.0:
        print("理想信道测试通过 - 基本信号处理正确")
    else:
        print("理想信道测试失败 - 存在基本信号处理问题")

    if offset_ber == 0.0:
        print("频偏校正测试通过")
    else:
        print("频偏校正测试失败")