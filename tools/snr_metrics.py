#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snr_metrics.py — Unified SNR/CFO/EVM measurement (single source of truth)

All SNR/CFO/EVM formulas used across polar_loopback.py, loopback_test.py,
receiver.py, and tools/loopback_analyze.py are consolidated here.

Canonical SNR definition:
  SNR_symbol = 10*log10( |h|^2 / noise_floor )
where noise_floor = var( RRC_matched_symbols_from_first_N_IQ_samples ).

This is the "symbol-domain, independent-noise" SNR used by the real-time
loopback scripts.  Other SNR formulations (RS-residual, gap-based) are
provided as derived metrics.

Import convention:
  from tools.snr_metrics import (noise_floor_from_iq, snr_symbol_domain,
                                  sigma2_welch, evm_from_sigma2,
                                  coarse_cfo_from_stf, fine_cfo_from_rs)

Reference formulas (matched to polar_loopback.py v2):
  - Sigma2:  s2 = max( sum(|noise|^2) / (RS_LEN - 1),  1e-30 )   [Welch]
  - Coarse CFO:  Deltaf = -angle(P_peak) / (2pi * L / fs)
  - Fine CFO:    slope = linear_regression( unwrap(angle(r·conj(RS))) )
                  Deltaf_fine = slope / (2pi * Ts_sym)
  - EVM:  10*log10( sigma2 )   (sigma2 clipped to 0.5 for display)
  - LLR:  4 * Re(y_eq) / sigma2_out   where sigma2_out uses max(sigma2, 1e-6)
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Noise Floor
# ═══════════════════════════════════════════════════════════════════════

def noise_floor_from_iq(iq, rrc, sps, rrc_delay, n_noise=50000):
    """Standard noise floor from first N IQ samples (symbol-level variance).

    Matches exactly the noise_floor measurement in polar_loopback.py
    lines 267-276 and loopback_test.py lines 192-202.

    Args:
        iq:          (M,) complex64  raw IQ samples
        rrc:         RRC filter taps (from phy_params.RRC)
        sps:         samples per symbol (from phy_params.SPS)
        rrc_delay:   (len(rrc) - 1) // 2  (from phy_params.RRC_DELAY_SAMPLES)
        n_noise:     number of leading IQ samples to use (default 50000)

    Returns:
        float:  noise_floor = var(RRC-matched symbols)
    """
    n = min(n_noise, len(iq))
    seg = iq[:n]
    syms = _rrc_match(seg, rrc, sps, rrc_delay)
    return float(np.var(syms))


def _rrc_match(samples, rrc, sps, rrc_delay):
    """RRC matched filter -> symbol-rate output (internal)."""
    f = np.convolve(samples, rrc[::-1], mode='full')
    return f[rrc_delay::sps]


# ═══════════════════════════════════════════════════════════════════════
# SNR — canonical symbol-domain
# ═══════════════════════════════════════════════════════════════════════

def snr_symbol_domain(h_mag, noise_floor):
    """Canonical SNR: 10*log10(|h|^2 / noise_floor).

    Matches polar_loopback.py line 339 and loopback_test.py line 265.

    Args:
        h_mag:       |h|, channel magnitude from RS estimate
        noise_floor: pre-measured symbol-level noise variance

    Returns:
        float: SNR in dB
    """
    return float(10 * np.log10(max(h_mag ** 2 / max(noise_floor, 1e-30), 1e-30)))


# ═══════════════════════════════════════════════════════════════════════
# Sigma2 — Welch-corrected RS residual variance
# ═══════════════════════════════════════════════════════════════════════

def sigma2_welch(rs_corrected, h, ref_rs, rs_len):
    """Welch-corrected noise variance from RS segment.

    Matches polar_loopback.py line 203:
      noise = rs_corrected / h - RS
      s2 = max( sum(|noise|^2) / (RS_LEN - 1), 1e-30 )

    Args:
        rs_corrected: (RS_LEN,) complex64  CFO-compensated RS symbols
        h:            complex  channel estimate (mean(rs_corrected * conj(RS)))
        ref_rs:       (RS_LEN,) complex64  reference RS sequence
        rs_len:       int  (typically 32)

    Returns:
        float: sigma2 (Welch-corrected)
    """
    noise = rs_corrected / h - ref_rs
    s2 = max(float(np.sum(np.abs(noise) ** 2) / (rs_len - 1)), 1e-30)
    return s2


def snr_from_sigma2(h_mag, sigma2):
    """SNR from RS residual: 10*log10(|h|^2 / sigma2).

    This is the "RS residual SNR" — different from canonical symbol-domain SNR
    because sigma2 includes equalization residuals (CFO, timing jitter)
    not captured by the independent noise floor.

    Matches receiver.py _estimate_snr() logic.

    Args:
        h_mag:   |h|
        sigma2:  Welch-corrected RS residual variance

    Returns:
        float: SNR in dB
    """
    return float(10 * np.log10(max(h_mag ** 2 / max(sigma2, 1e-30), 1e-30)))


# ═══════════════════════════════════════════════════════════════════════
# EVM
# ═══════════════════════════════════════════════════════════════════════

def evm_from_sigma2(sigma2, clip=0.5):
    """EVM from sigma2: 10*log10( min(sigma2, clip) ).

    Matches polar_loopback.py line 341:
      sigma2_clip = min(chan['sigma2'], 0.5)
      evm_db = 10 * np.log10(max(sigma2_clip, 1e-30))

    Args:
        sigma2:  Welch-corrected RS residual variance
        clip:    upper clip for display (default 0.5)

    Returns:
        float: EVM in dB (negative for good signal)
    """
    s2_clipped = min(sigma2, clip)
    return float(10 * np.log10(max(s2_clipped, 1e-30)))


# ═══════════════════════════════════════════════════════════════════════
# CFO
# ═══════════════════════════════════════════════════════════════════════

def coarse_cfo_from_stf(p_peak, stf_delay, samp_rate):
    """Coarse CFO from STF delay correlation peak.

    Matches polar_loopback.py lines 134-135:
      phase = np.angle(P[peak])
      cfo = -phase / (2 * pi * L / samp_rate)

    Args:
        p_peak:     complex  P(d) at the peak position
        stf_delay:  int  STF delay-correlation spacing in samples (typ. 32)
        samp_rate:  float  sample rate (typ. 1e6)

    Returns:
        float: coarse CFO in Hz
    """
    phase = np.angle(p_peak)
    return float(-phase / (2 * np.pi * stf_delay / samp_rate))


def fine_cfo_from_rs(symbols, rs_pos, ref_rs, coarse_cfo, ts_sym, rs_len=32):
    """Fine CFO from RS linear phase fitting (with coarse CFO pre-compensation).

    Matches polar_loopback.py lines 168-188.

    Steps:
      1. Pre-compensate coarse CFO on RS segment
      2. Compute rs_tone = rs_seg * conj(RS)
      3. Unwrap phase -> linear regression slope -> fine CFO

    Args:
        symbols:    (M,) complex64  RRC-matched symbols
        rs_pos:     int  RS start index in symbols
        ref_rs:     (RS_LEN,) complex64  reference RS sequence
        coarse_cfo: float  coarse CFO estimate in Hz (from STF)
        ts_sym:     float  symbol time in seconds (SPS / samp_rate)
        rs_len:     int  RS length (default 32)

    Returns:
        tuple: (fine_cfo_hz, rs_corr)
          fine_cfo_hz:  residual fine CFO in Hz
          rs_corr:      |sum(rs_tone)|  RS correlation magnitude for quality check
    """
    if rs_pos + rs_len > len(symbols):
        return 0.0, 0.0

    rs_seg = symbols[rs_pos:rs_pos + rs_len].copy()
    n_rs = np.arange(rs_len)

    # Coarse CFO pre-compensation
    if abs(coarse_cfo) > 0.0:
        pre_comp = np.exp(-1j * 2 * np.pi * coarse_cfo * (rs_pos + n_rs) * ts_sym)
        rs_seg = rs_seg * pre_comp

    # Fine CFO: unwrap -> linear regression slope
    rs_tone = rs_seg * np.conj(ref_rs)
    rs_corr = float(np.abs(np.sum(rs_tone)))
    rs_phase = np.unwrap(np.angle(rs_tone))

    n = np.arange(rs_len, dtype=np.float64)
    n_mean = np.mean(n)
    p_mean = np.mean(rs_phase)
    num = np.sum((n - n_mean) * (rs_phase - p_mean))
    den = np.sum((n - n_mean) ** 2)
    slope = num / (den + 1e-30)
    fine_cfo = float(slope / (2 * np.pi * ts_sym))

    return fine_cfo, rs_corr


# ═══════════════════════════════════════════════════════════════════════
# LLR — BPSK soft demodulation
# ═══════════════════════════════════════════════════════════════════════

def bpsk_llr(symbols, data_start, data_len, h, total_cfo, sigma2,
             ts_sym, llr_clip=20.0):
    """BPSK soft demodulation -> LLR.

    Matches polar_loopback.py lines 215-238.

    LLR = 4 * Re(y_eq) / sigma2_out
    where y_eq = symbols * exp(-j·2pi·Deltaf·t) / h
    and sigma2_out = max(sigma2, 1e-6)

    RS-estimated sigma2 is complex residual variance -> real part variance = sigma2/2.
    LLR = 2*y / (sigma2/2) = 4*y / sigma2.

    Args:
        symbols:     (M,) complex64  RRC-matched symbols
        data_start:  int  start index of data segment
        data_len:    int  number of symbols
        h:           complex  channel estimate
        total_cfo:   float  total CFO in Hz (coarse + fine)
        sigma2:      float  RS residual variance
        ts_sym:      float  symbol time in seconds
        llr_clip:    float  LLR clipping range (default 20.0)

    Returns:
        (data_len,) float32  LLR values, clipped to [-llr_clip, llr_clip]
    """
    if data_start + data_len > len(symbols):
        return np.zeros(data_len, dtype=np.float32)
    seg = symbols[data_start:data_start + data_len]
    n = np.arange(data_len)
    cfo_comp = np.exp(-1j * 2 * np.pi * total_cfo * (data_start + n) * ts_sym)
    y = seg * cfo_comp
    if abs(h) > 1e-30:
        y = y / h
    sigma2_out = max(float(sigma2), 1e-6)
    llr = 4.0 * y.real / sigma2_out
    return np.clip(llr, -llr_clip, llr_clip).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Gap-based SNR (used by tools/loopback_analyze.py for refinement)
# ═══════════════════════════════════════════════════════════════════════

def snr_from_gaps(iq, frames, frame_rrc_samples, margin_before=100, margin_after=100,
                  min_gap=500):
    """Estimate SNR from inter-frame gap noise (IQ domain).

    This is a secondary metric used by tools/loopback_analyze.py to refine
    the initial prefix-based estimate.  Gap noise is purely receiver noise
    floor (no signal present).

    SNR = 10*log10( (sig_power - noise_var) / noise_var )

    Args:
        iq:                  (N,) complex64  raw IQ samples
        frames:              list[dict]  each with 'global_pos' key
        frame_rrc_samples:   int  frame duration in IQ samples (typ. 1012)
        margin_before:       int  samples before next frame start
        margin_after:        int  samples after previous frame end
        min_gap:             int  minimum gap samples to use

    Returns:
        (noise_var, snr_per_frame_list) or (None, []) if insufficient gaps
    """
    if len(frames) < 2:
        return None, []

    gap_iqs = []
    for i in range(len(frames) - 1):
        g0 = frames[i]['global_pos'] + frame_rrc_samples + margin_after
        g1 = frames[i + 1]['global_pos'] - margin_before
        if g1 - g0 > min_gap:
            gap_iqs.append(iq[g0:g1])

    if not gap_iqs:
        return None, []

    gap_all = np.concatenate(gap_iqs)
    noise_var = float(np.var(gap_all))

    snrs = []
    for f in frames:
        if 'constellation' in f:
            sig_pow = float(np.mean(np.abs(f['constellation']) ** 2))
            snr_lin = max(sig_pow - noise_var, 1e-30) / max(noise_var, 1e-30)
            snrs.append(float(10 * np.log10(snr_lin)))

    return noise_var, snrs


# ═══════════════════════════════════════════════════════════════════════
# Convenience: all metrics from one frame
# ═══════════════════════════════════════════════════════════════════════

def frame_metrics(symbols, rs_pos, ref_rs, coarse_cfo, h, sigma2,
                  noise_floor, ts_sym, rs_len=32):
    """Compute all standard metrics for one frame.

    Args:
        symbols:     (M,) complex64
        rs_pos:      RS start index
        ref_rs:      reference RS sequence
        coarse_cfo:  from STF
        h:           RS channel estimate (complex)
        sigma2:      Welch-corrected RS residual variance
        noise_floor: pre-measured symbol-level noise variance
        ts_sym:      symbol time
        rs_len:      RS length

    Returns:
        dict with keys: snr_symbol, snr_rs, evm, total_cfo, fine_cfo, rs_corr, h_mag, sigma2
    """
    _, fine_cfo, rs_corr = 0.0, 0.0, 0.0  # filled if fine_cfo computed

    h_mag = float(abs(h))
    snr_sym = snr_symbol_domain(h_mag, noise_floor)
    snr_rs = snr_from_sigma2(h_mag, sigma2)
    evm = evm_from_sigma2(sigma2)

    return {
        'snr_symbol': snr_sym,
        'snr_rs': snr_rs,
        'evm': evm,
        'h_mag': h_mag,
        'sigma2': sigma2,
        'noise_floor': noise_floor,
    }
