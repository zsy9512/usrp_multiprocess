"""
phy_params.py — PHY 层统一参数 + CRC16

所有帧结构、参考序列、滤波器参数集中于此。
sender / receiver / test 共用。
"""
from __future__ import annotations

import numpy as np

# ======================================================================
# 帧结构参数 (符号域)
# ======================================================================
SPS = 2                     # 每符号采样数
ROLLOFF = 0.35              # RRC 滚降系数
RRC_NUM_SYM = 10            # RRC 滤波器单边符号长度

# STF: 4×16 重复 BPSK -> 延迟相关粗捕获 (免疫大频偏)
STF_REP = 16                # 每段长度 (延迟相关间距 L)
STF_NUM = 4                 # 重复段数
STF_LEN = STF_REP * STF_NUM # 64 符号

# PSS: Zadoff-Chu -> 精定时 + 相关确认
PSS_LEN = 64
PSS_U = 25

# RS: 已知 BPSK -> 细 CFO + 公共相位 + 信道/噪声估计
RS_LEN = 32

# Header: 预留信息 (后续 MCS/length/ID) + CRC16
HEADER_LEN = 32

# Payload
PAYLOAD_LEN = 256

# Payload CRC
PAYLOAD_CRC_LEN = 16

# Guard (滤波尾巴 + 帧间隔)
GUARD_SYMBOLS = 32

# 帧总符号数
FRAME_SYMBOLS = (STF_LEN + PSS_LEN + RS_LEN +
                 HEADER_LEN + PAYLOAD_LEN + PAYLOAD_CRC_LEN +
                 GUARD_SYMBOLS)   # = 64+64+32+32+256+16+32 = 496

# RRC 延迟 (样本域)
RRC_DELAY_SAMPLES = RRC_NUM_SYM * SPS  # 20 samples

# STF 延迟相关间距 (样本域)
STF_DELAY = STF_REP * SPS  # 32 samples

# 符号采样时间 (1 Msps)
TS = 1.0 / 1e6

# 数据段信息比特数 (预留给 polar 码)
INFO_BITS = 128

# ======================================================================
# SNR / Sigma2 测量方法 (统一口径, 2026-06)
# ======================================================================

# 规范 SNR 定义: SNR_symbol = 10*log10(|h|^2 / noise_floor)
#   noise_floor = var( RRC_matched(first_N_IQ_samples) )  — 符号域独立底噪
# 参考: tools/snr_metrics.py (统一实现)
SNR_METHOD = "symbol_domain"        # canonical SNR type
NOISE_FLOOR_WINDOW = 50000          # IQ samples for noise floor measurement

# Sigma2 (RS 残差方差) 计算方法
#   "welch":  s2 = max(sum(|noise|^2) / (RS_LEN - 1), 1e-30)
#   等效于 np.var(noise) * RS_LEN/(RS_LEN-1)  if noise is zero-mean
SIGMA2_METHOD = "welch"

# ======================================================================
# 同步检测参数
# ======================================================================

# STF 检测门限 (归一化度量 M(d) = |P(d)| / E(d))
STF_THRESHOLD = 0.4

# STF 最小能量门限 (样本域, 避免 Guard 区虚假峰)
STF_MIN_ENERGY = 0.02 * STF_DELAY  # 0.64, 适配较宽信号动态范围 (原 3.2 过于严格)

# PSS 峰值质量门限
#   Design values:  ptm=4.0  pts=1.5  (高 SNR 严格)
#   Operational defaults:  ptm=3.5  pts=1.5  (低 SNR 放宽, 与 loopback_test/polar_loopback 一致)
PSS_PEAK_TO_MEAN_THR = 4.0
PSS_PEAK_TO_SECOND_THR = 1.5

# PSS 搜索窗口 (样本域, 围绕 STF 粗位置)
PSS_SEARCH_WIN_SAMPLES = STF_LEN * SPS  # 128

# 检测推进步长
ADVANCE_SAMPLES = (PSS_LEN + RS_LEN + HEADER_LEN) * SPS  # ~224

# 帧样本总长 (含 RRC 卷积溢出: mode='full' -> len + len(rrc) - 1)
RRC_OUT_EXTRA = RRC_NUM_SYM * SPS  # = 20
FRAME_RRC_SAMPLES = FRAME_SYMBOLS * SPS + RRC_OUT_EXTRA  # = 1012

# 最小处理窗口
MIN_WIN_SAMPLES = FRAME_RRC_SAMPLES + PSS_SEARCH_WIN_SAMPLES

# ======================================================================
# 参考序列生成 (固定种子, 确保 sender/receiver 一致)
# ======================================================================

_RNG_STF = np.random.RandomState(7)
_RNG_RS  = np.random.RandomState(13)


def gen_stf() -> np.ndarray:
    """STF: 4×16 重复 BPSK (+1,-1)."""
    base = 2 * _RNG_STF.randint(0, 2, STF_REP) - 1
    return np.tile(base, STF_NUM).astype(np.complex64)


def gen_pss() -> np.ndarray:
    """PSS: Zadoff-Chu u=25, 长度 64."""
    n = np.arange(PSS_LEN)
    zc = np.exp(-1j * np.pi * PSS_U * n * (n + 1) / PSS_LEN)
    return zc.astype(np.complex64)


def gen_rs() -> np.ndarray:
    """RS: 固定 BPSK 导频."""
    return (2 * _RNG_RS.randint(0, 2, RS_LEN) - 1).astype(np.complex64)


# ======================================================================
# RRC 脉冲成形滤波器
# ======================================================================

def design_rrc(sps: int = SPS, rolloff: float = ROLLOFF,
               num_sym: int = RRC_NUM_SYM) -> np.ndarray:
    """设计 RRC 滤波器.

    Returns:
        (num_sym*2*sps+1,) float32, 单位能量
    """
    half_sym = num_sym / 2
    t = np.arange(-half_sym, half_sym + 1e-12, 1 / sps)
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 + rolloff * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * rolloff)) < 1e-12:
            h[i] = (rolloff / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff)))
        else:
            pi_t = np.pi * ti
            num = (np.sin(pi_t * (1 - rolloff))
                   + 4 * rolloff * ti * np.cos(pi_t * (1 + rolloff)))
            den = pi_t * (1 - (4 * rolloff * ti) ** 2)
            h[i] = num / den
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


# ======================================================================
# CRC16-IBM (x^16 + x^15 + x^2 + 1)
# ======================================================================

def _crc16_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x8005) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table[i] = crc
    return table


_CRC16_TABLE = _crc16_table()


def crc16(data: np.ndarray) -> int:
    """计算 CRC16-IBM.

    Args:
        data: uint8 数组
    Returns:
        uint16 CRC
    """
    crc = 0x0000
    for byte in data:
        idx = (crc >> 8) ^ int(byte)
        crc = ((crc << 8) ^ int(_CRC16_TABLE[idx & 0xFF])) & 0xFFFF
    return crc


def crc16_check(data: np.ndarray, expected_crc: int) -> bool:
    return crc16(data) == expected_crc


def bits_to_bytes(bits: np.ndarray) -> np.ndarray:
    """{0,1} bits -> uint8 bytes (MSB first)."""
    n_bytes = (len(bits) + 7) // 8
    padded = np.zeros(n_bytes * 8, dtype=np.uint8)
    padded[:len(bits)] = bits
    return np.packbits(padded)


def bytes_to_bits(data: np.ndarray, n_bits: int) -> np.ndarray:
    """uint8 bytes -> {0,1} bits (MSB first)."""
    bits = np.unpackbits(data).astype(np.int64)
    return bits[:n_bits]


# ======================================================================
# 预生成全局参考序列 (模块加载时生成, 供 sender/receiver 直接引用)
# ======================================================================
STF  = gen_stf()
PSS  = gen_pss()
RS   = gen_rs()
RRC  = design_rrc()
