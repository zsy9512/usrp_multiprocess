import numpy as np
import sys
import os

# 添加路径以导入dqpsk_system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from usrp_multiprocess.dqpsk_system import USRP_DQPSK_System

def test_differential_decode_detailed():
    """详细测试差分解码的每个步骤"""
    print("=== 差分解码详细测试 ===")

    dqpsk = USRP_DQPSK_System()
    np.random.seed(42)

    # 生成简单测试数据
    data_bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])  # 4个符号
    print(f"测试比特: {data_bits}")

    # 比特到符号
    data_symbols = dqpsk._bits_to_symbols(data_bits)
    print(f"数据符号: {data_symbols}")

    # 差分编码
    encoded_symbols = dqpsk.differential_encode(data_symbols)
    print(f"差分编码后: {encoded_symbols}")

    # 模拟接收（添加小噪声）
    noise_level = 0.01
    noise = noise_level * (np.random.randn(len(encoded_symbols)) + 1j*np.random.randn(len(encoded_symbols)))
    received_symbols = encoded_symbols + noise
    print(f"接收符号（加噪声）: {received_symbols}")

    # 手动差分解码
    print("\n=== 手动差分解码过程 ===")

    decoded_manual = []
    # 第一步：使用参考符号
    ref_symbol = dqpsk.diff_ref_symbol
    first_decoded = received_symbols[0] * np.conj(ref_symbol)
    decoded_manual.append(first_decoded)
    print(f"第一符号解码: {received_symbols[0]} * conj({ref_symbol}) = {first_decoded}")

    # 后续步骤：差分解码
    for i in range(1, len(received_symbols)):
        diff_decoded = received_symbols[i] * np.conj(received_symbols[i-1])
        decoded_manual.append(diff_decoded)
        print(f"第{i+1}符号解码: {received_symbols[i]} * conj({received_symbols[i-1]}) = {diff_decoded}")

    decoded_manual = np.array(decoded_manual)
    print(f"手动解码结果: {decoded_manual}")

    # 使用系统函数解码
    decoded_system = dqpsk.differential_decode(received_symbols)
    print(f"系统解码结果: {decoded_system}")

    # 比较
    print(f"手动vs系统匹配: {np.allclose(decoded_manual, decoded_system)}")

    # 符号判决
    manual_bits = dqpsk._symbols_to_bits(decoded_manual)
    system_bits = dqpsk._symbols_to_bits(decoded_system)

    print(f"手动解码比特: {manual_bits}")
    print(f"系统解码比特: {system_bits}")
    print(f"原始比特:     {data_bits}")

    # 计算错误
    manual_errors = np.sum(data_bits != manual_bits)
    system_errors = np.sum(data_bits != system_bits)

    print(f"手动解码错误: {manual_errors}/{len(data_bits)}")
    print(f"系统解码错误: {system_errors}/{len(data_bits)}")

    return manual_errors == 0 and system_errors == 0

def test_constellation_decoding():
    """测试星座判决的准确性"""
    print("\n=== 星座判决测试 ===")

    dqpsk = USRP_DQPSK_System()

    # 测试理想星座点
    constellation_points = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
    expected_bits = [[0,0], [0,1], [1,0], [1,1]]

    print("理想星座点判决:")
    all_correct = True
    for i, point in enumerate(constellation_points):
        decoded_bits = dqpsk._symbols_to_bits(np.array([point]))
        expected = expected_bits[i]
        correct = np.array_equal(decoded_bits, expected)
        print(f"{point:.4f} -> {decoded_bits} (期望: {expected}) {'✓' if correct else '✗'}")
        if not correct:
            all_correct = False

    # 测试带噪声的点
    print("\n带噪声的星座点判决:")
    noise_level = 0.1
    for i, point in enumerate(constellation_points):
        noisy_point = point + noise_level * (np.random.randn() + 1j*np.random.randn())
        decoded_bits = dqpsk._symbols_to_bits(np.array([noisy_point]))
        expected = expected_bits[i]
        correct = np.array_equal(decoded_bits, expected)
        distance = np.abs(noisy_point - point)
        print(f"{noisy_point:.4f} -> {decoded_bits} (期望: {expected}, 距离: {distance:.3f}) {'✓' if correct else '✗'}")

    return all_correct

if __name__ == "__main__":
    print("开始差分解码详细测试...\n")

    test1 = test_differential_decode_detailed()
    test2 = test_constellation_decoding()

    print("\n=== 总结 ===")
    print(f"差分解码测试: {'通过' if test1 else '失败'}")
    print(f"星座判决测试: {'通过' if test2 else '失败'}")

    if test1 and test2:
        print("所有组件测试通过")
    else:
        print("发现组件问题")