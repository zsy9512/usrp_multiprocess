/*
 * phy_dsp.h — BPSK PHY 公共 DSP (C++17, 零外部依赖)
 *
 * 供 tx_main / rx_main / uhd_tx_main / uhd_rx_main 共用。
 * 所有函数均为 inline，避免多目标链接冲突。
 *
 * 帧结构: STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <ctime>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using C64 = std::complex<float>;

// ===================================================================
// 1. 帧参数 (与 phy_params.py 一致)
// ===================================================================
constexpr int SPS           = 2;
constexpr float ROLLOFF     = 0.35f;
constexpr int RRC_NUM_SYM   = 10;
constexpr int STF_REP       = 16;
constexpr int STF_NUM       = 4;
constexpr int STF_LEN       = STF_REP * STF_NUM;   // 64
constexpr int PSS_LEN       = 64;
constexpr int PSS_U         = 25;
constexpr int RS_LEN        = 32;
constexpr int HEADER_LEN    = 32;
constexpr int PAYLOAD_LEN   = 256;
constexpr int CRC_LEN       = 16;
constexpr int GUARD_SYMBOLS = 32;
constexpr int FRAME_SYMBOLS = STF_LEN + PSS_LEN + RS_LEN + HEADER_LEN
                              + PAYLOAD_LEN + CRC_LEN + GUARD_SYMBOLS; // 496
constexpr int STF_DELAY     = STF_REP * SPS;       // 32
constexpr int RRC_TAPS      = RRC_NUM_SYM * SPS + 1;   // = 21
constexpr int RRC_DELAY     = (RRC_TAPS - 1) / 2;  // 10
constexpr int FRAME_RRC_SAMPLES = FRAME_SYMBOLS * SPS + (RRC_TAPS - 1); // 1012

// 可运行时覆盖的全局参数 (默认 1Msps)
static float g_ts              = 1.0f / 1.0e6f;        // 样本时间
static float g_ts_sym          = (float)SPS / 1.0e6f;  // 符号时间 = g_ts * SPS
static float g_stf_threshold   = 0.40f;
static float g_stf_min_energy  = 0.02f * STF_DELAY;
static float g_pss_ptm_thr     = 3.5f;
static float g_pss_pts_thr     = 1.5f;
static float g_rs_corr_thr     = RS_LEN * 0.3f;   // 9.6
static float g_fine_cfo_max    = 500.0f;           // 对齐 loopback_test: |fine_cfo|>500 → 拒收
static float g_sigma2_max      = 0.5f;             // 噪声方差上界裁剪

// sync parameters
constexpr int   PSS_SEARCH_WIN     = STF_LEN * SPS;       // 128
constexpr int   EXTRACT_EXTRA      = 200;                  // 对齐 loopback_test 前后裕量
constexpr int   ADVANCE_SAMPLES    = (PSS_LEN + RS_LEN + HEADER_LEN) * SPS;
constexpr int   MIN_WIN_SAMPLES    = FRAME_RRC_SAMPLES + PSS_SEARCH_WIN;

// ===================================================================
// 2. CRC16-IBM
// ===================================================================

static uint16_t crc16_table[256];

inline void init_crc16_table() {
    for (int i = 0; i < 256; i++) {
        uint16_t crc = (uint16_t)(i << 8);
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x8005) : (uint16_t)(crc << 1);
        crc16_table[i] = crc;
    }
}

inline uint16_t crc16(const uint8_t* data, int len) {
    uint16_t crc = 0;
    for (int i = 0; i < len; i++)
        crc = (uint16_t)((crc << 8) ^ crc16_table[((crc >> 8) ^ data[i]) & 0xFF]);
    return crc;
}

// ===================================================================
// 3. 参考序列生成 (固定种子, 与 Python 一致)
// ===================================================================

inline std::vector<C64> gen_stf() {
    // numpy RandomState(7), randint(0,2,16) → ±1
    int bits[STF_REP] = {1,-1,1,-1,1,1,1,1,-1,1,-1,1,-1,1,-1,-1};
    std::vector<C64> stf(STF_LEN);
    for (int i = 0; i < STF_NUM; i++)
        for (int j = 0; j < STF_REP; j++)
            stf[i*STF_REP+j] = C64((float)bits[j], 0.0f);
    return stf;
}

inline std::vector<C64> gen_pss() {
    std::vector<C64> pss(PSS_LEN);
    for (int n = 0; n < PSS_LEN; n++) {
        float phase = -(float)M_PI * PSS_U * n * (n + 1) / PSS_LEN;
        pss[n] = C64(std::cos(phase), std::sin(phase));
    }
    return pss;
}

inline std::vector<C64> gen_rs() {
    // numpy RandomState(13), randint(0,2,32), mapped to ±1
    int bits[RS_LEN] = {-1,-1,-1,-1,-1,-1,-1,1,
                        -1,-1,1,-1,-1,-1,-1,-1,
                        -1,1,1,-1,1,1,-1,-1,
                        -1,1,1,1,1,-1,1,1};
    std::vector<C64> rs(RS_LEN);
    for (int i = 0; i < RS_LEN; i++)
        rs[i] = C64((float)bits[i], 0.0f);
    return rs;
}

inline std::vector<float> design_rrc() {
    std::vector<float> h(RRC_TAPS, 0.0f);
    float half = RRC_NUM_SYM / 2.0f;
    for (int i = 0; i < RRC_TAPS; i++) {
        float ti = -half + i / (float)SPS;
        if (std::abs(ti) < 1e-12f)
            h[i] = 1.0f + ROLLOFF * (4.0f / (float)M_PI - 1.0f);
        else if (std::abs(std::abs(ti) - 1.0f / (4.0f * ROLLOFF)) < 1e-12f)
            h[i] = (ROLLOFF / std::sqrt(2.0f))
                * ((1.0f + 2.0f / (float)M_PI) * std::sin((float)M_PI / (4.0f * ROLLOFF))
                + (1.0f - 2.0f / (float)M_PI) * std::cos((float)M_PI / (4.0f * ROLLOFF)));
        else {
            float pit = (float)M_PI * ti;
            float num = std::sin(pit * (1.0f - ROLLOFF))
                      + 4.0f * ROLLOFF * ti * std::cos(pit * (1.0f + ROLLOFF));
            float den = pit * (1.0f - std::pow(4.0f * ROLLOFF * ti, 2));
            h[i] = num / den;
        }
    }
    float energy = 0.0f;
    for (float v : h) energy += v * v;
    float scale = 1.0f / std::sqrt(energy);
    for (float& v : h) v *= scale;
    return h;
}

// ===================================================================
// 4. TX 帧打包
// ===================================================================

inline std::vector<C64> build_frame(const std::vector<int>& dataBits, uint16_t frameId) {
    // payload CRC
    uint8_t payloadBytes[PAYLOAD_LEN / 8] = {};
    for (int i = 0; i < PAYLOAD_LEN; i++)
        payloadBytes[i / 8] = (uint8_t)((payloadBytes[i / 8] << 1) | (dataBits[i] & 1));
    uint16_t payCrc = crc16(payloadBytes, PAYLOAD_LEN / 8);

    // header: frame_id(16bit) + header CRC
    uint8_t hdrBytes[2] = {(uint8_t)(frameId>>8), (uint8_t)(frameId&0xFF)};
    uint16_t hdrCrc = crc16(hdrBytes, 2);

    // assemble frame symbols
    std::vector<C64> frame;
    frame.reserve(FRAME_SYMBOLS);

    auto stf = gen_stf();
    frame.insert(frame.end(), stf.begin(), stf.end());

    auto pss = gen_pss();
    frame.insert(frame.end(), pss.begin(), pss.end());

    auto rs = gen_rs();
    frame.insert(frame.end(), rs.begin(), rs.end());

    // Header: 16 frame_id + 16 CRC bits → BPSK (MSB first)
    for (int i = 15; i >= 0; i--)
        frame.push_back(C64(((frameId>>i)&1) ? -1.0f : 1.0f, 0.0f));
    for (int i = 15; i >= 0; i--) {
        int bit = (hdrCrc >> i) & 1;
        frame.push_back(C64(bit ? -1.0f : 1.0f, 0.0f));
    }

    // Payload: data bits → BPSK
    for (int bit : dataBits)
        frame.push_back(C64(bit ? -1.0f : 1.0f, 0.0f));

    // CRC bits → BPSK (MSB first)
    for (int i = 15; i >= 0; i--) {
        int bit = (payCrc >> i) & 1;
        frame.push_back(C64(bit ? -1.0f : 1.0f, 0.0f));
    }

    // Guard (zeros)
    for (int i = 0; i < GUARD_SYMBOLS; i++)
        frame.push_back(C64(0.0f, 0.0f));

    return frame;
}

// ===================================================================
// 5. RRC 脉冲成形
// ===================================================================

inline std::vector<C64> rrc_filter(const std::vector<C64>& symbols,
                                    const std::vector<float>& rrc) {
    int N = (int)symbols.size();
    int M = (int)rrc.size();
    int upLen = N * SPS;
    std::vector<C64> up(upLen, C64(0.0f, 0.0f));
    for (int i = 0; i < N; i++)
        up[i * SPS] = symbols[i];

    int convLen = upLen + M - 1;
    std::vector<C64> out(convLen, C64(0.0f, 0.0f));
    for (int i = 0; i < convLen; i++) {
        C64 sum(0, 0);
        for (int j = 0; j < M; j++) {
            int sidx = i - j;
            if (sidx >= 0 && sidx < upLen)
                sum += up[sidx] * rrc[j];
        }
        out[i] = sum;
    }
    return out;
}

// ===================================================================
// 6. RX: RRC 匹配滤波
// ===================================================================

inline std::vector<C64> rrc_match(const std::vector<C64>& samples,
                                   const std::vector<float>& rrc) {
    int N = (int)samples.size();
    int M = (int)rrc.size();
    int convLen = N + M - 1;
    std::vector<C64> filt(convLen, C64(0,0));
    for (int i = 0; i < convLen; i++) {
        C64 sum(0, 0);
        for (int j = 0; j < M; j++) {
            int sidx = i - j;
            if (sidx >= 0 && sidx < N)
                sum += samples[sidx] * rrc[M - 1 - j];
        }
        filt[i] = sum;
    }
    int delay = (M - 1) / 2;
    int nOut = (convLen - delay + SPS - 1) / SPS;
    std::vector<C64> out(nOut);
    for (int k = 0; k < nOut; k++)
        out[k] = filt[delay + k * SPS];
    return out;
}

// ===================================================================
// 7. RX: STF 延迟相关
// ===================================================================

struct StfResult {
    std::vector<float> metric;
    std::vector<C64> P;
};

inline StfResult stf_delay_correlate(const std::vector<C64>& samples) {
    int L = STF_DELAY;
    int N = (int)samples.size();
    StfResult res;
    if (N <= L) return res;

    int numP = N - 2 * L + 1;
    res.metric.resize(numP);
    res.P.resize(numP);

    std::vector<C64> prod(N - L);
    for (int i = 0; i < N - L; i++)
        prod[i] = samples[i] * std::conj(samples[i + L]);

    C64 pSum(0, 0);
    float eSum = 0.0f;
    for (int i = 0; i < L; i++) {
        pSum += prod[i];
        eSum += std::norm(samples[i + L]);
    }

    res.P[0] = pSum;
    float noiseFloor = 1e-6f * L;
    res.metric[0] = std::abs(pSum) / (eSum + noiseFloor);

    for (int d = 1; d < numP; d++) {
        pSum = pSum - prod[d - 1] + prod[d + L - 1];
        eSum = eSum - std::norm(samples[d - 1 + L]) + std::norm(samples[d + 2 * L - 1]);
        res.P[d] = pSum;
        res.metric[d] = std::abs(pSum) / (eSum + noiseFloor);
    }
    return res;
}

inline float compute_coarse_cfo(C64 P_peak, float samp_rate) {
    float phase = std::arg(P_peak);
    return -phase / (2.0f * (float)M_PI * STF_DELAY / samp_rate);
}

// ===================================================================
// 7b. STF 峰值聚类去重 (对齐 loopback_test 128-sample 窗)
// ===================================================================

struct StfClusterResult {
    std::vector<int> peaks;      // 聚类后的峰值位置
    std::vector<float> cfos;     // 对应的粗 CFO
};

inline StfClusterResult stf_cluster_peaks(const StfResult& stf,
                                           const std::vector<C64>& samples,
                                           float samp_rate) {
    StfClusterResult res;
    int L = STF_DELAY;
    int N = (int)samples.size();

    // 收集满足门限 + 能量的原始候选
    struct RawPeak { int d; float metric; C64 P; };
    std::vector<RawPeak> raw;
    for (int d = 0; d < (int)stf.metric.size(); d++) {
        if (stf.metric[d] > g_stf_threshold) {
            float localE = 0.0f;
            for (int j = L; j < 2 * L; j++) {
                if (d + j < N)
                    localE += std::norm(samples[d + j]);
            }
            if (localE > g_stf_min_energy)
                raw.push_back({d, stf.metric[d], stf.P[d]});
        }
    }
    if (raw.empty()) return res;

    // 按 M 排序, 128-sample 窗内去重保留最强
    std::sort(raw.begin(), raw.end(),
              [](const RawPeak& a, const RawPeak& b) { return a.metric > b.metric; });

    std::vector<bool> used(stf.metric.size(), false);
    for (const auto& rp : raw) {
        if (used[rp.d]) continue;
        int start = std::max(0, rp.d - 128);
        int end = std::min((int)stf.metric.size(), rp.d + 128);
        for (int i = start; i < end; i++) used[i] = true;
        res.peaks.push_back(rp.d);
        float phase = std::arg(rp.P);
        res.cfos.push_back(-phase / (2.0f * (float)M_PI * L / samp_rate));
    }
    return res;
}

// ===================================================================
// 8. RX: PSS 互相关
// ===================================================================

struct PssResult {
    int peak_idx = 0;
    float peak_to_mean = 0.0f;
    float peak_to_second = 0.0f;
};

inline PssResult pss_correlate(const std::vector<C64>& symbols,
                                const std::vector<C64>& ref_pss) {
    PssResult res;
    int N = (int)symbols.size();
    int M = PSS_LEN;
    if (N < M) return res;

    int corrLen = N - M + 1;
    std::vector<float> corrMag(corrLen, 0.0f);
    float sumAll = 0.0f;

    for (int i = 0; i < corrLen; i++) {
        C64 dot(0, 0);
        for (int j = 0; j < M; j++)
            dot += symbols[i + j] * std::conj(ref_pss[j]);
        corrMag[i] = std::abs(dot);
        sumAll += corrMag[i];
    }

    int peak = 0;
    float peakVal = corrMag[0];
    for (int i = 1; i < corrLen; i++) {
        if (corrMag[i] > peakVal) {
            peakVal = corrMag[i];
            peak = i;
        }
    }

    float meanVal = sumAll / (float)corrLen;
    res.peak_idx = peak;
    res.peak_to_mean = peakVal / (meanVal + 1e-30f);

    // find second peak (first one > PSS_LEN/2 away from main peak)
    float secondVal = 0.0f;
    std::vector<int> sortedIdx(corrLen);
    for (int i = 0; i < corrLen; i++) sortedIdx[i] = i;
    std::sort(sortedIdx.begin(), sortedIdx.end(),
              [&](int a, int b) { return corrMag[a] > corrMag[b]; });
    for (int i = 0; i < corrLen; i++) {
        int idx = sortedIdx[i];
        if (std::abs(idx - peak) > M / 2) {
            secondVal = corrMag[idx];
            break;
        }
    }
    if (secondVal < 1e-12f)
        res.peak_to_second = res.peak_to_mean;
    else
        res.peak_to_second = peakVal / secondVal;

    return res;
}

// ===================================================================
// 9. RX: RS 细 CFO + 信道估计
// ===================================================================

inline float rs_fine_cfo(const std::vector<C64>& symbols, int rs_pos,
                          float coarse_cfo,
                          const std::vector<C64>& ref_rs,
                          float& rs_corr) {
    if (rs_pos + RS_LEN > (int)symbols.size()) {
        rs_corr = 0.0f;
        return 0.0f;
    }

    std::vector<C64> rs_seg(RS_LEN);
    for (int i = 0; i < RS_LEN; i++) {
        rs_seg[i] = symbols[rs_pos + i];
        if (std::abs(coarse_cfo) > 0.0f) {
            float phase = -2.0f * (float)M_PI * coarse_cfo * (rs_pos + i) * g_ts_sym;
            rs_seg[i] *= C64(std::cos(phase), std::sin(phase));
        }
    }

    C64 dotSum(0, 0);
    for (int i = 0; i < RS_LEN; i++)
        dotSum += rs_seg[i] * std::conj(ref_rs[i]);
    rs_corr = std::abs(dotSum);

    float phases[RS_LEN];
    float prev = 0.0f;
    float accum = 0.0f;
    for (int i = 0; i < RS_LEN; i++) {
        float raw = std::arg(rs_seg[i] * std::conj(ref_rs[i]));
        float diff = raw - prev;
        if (diff > (float)M_PI) accum -= 2.0f * (float)M_PI;
        else if (diff < -(float)M_PI) accum += 2.0f * (float)M_PI;
        phases[i] = raw + accum;
        prev = raw;
    }

    float nMean = (RS_LEN - 1) / 2.0f;
    float phaseMean = 0.0f;
    for (int i = 0; i < RS_LEN; i++) phaseMean += phases[i];
    phaseMean /= RS_LEN;

    float num = 0.0f, den = 0.0f;
    for (int i = 0; i < RS_LEN; i++) {
        float dn = i - nMean;
        num += dn * (phases[i] - phaseMean);
        den += dn * dn;
    }
    float slope = num / (den + 1e-30f);
    return slope / (2.0f * (float)M_PI * g_ts_sym);
}

inline bool rs_channel_estimate(const std::vector<C64>& symbols, int rs_pos,
                                 float fine_cfo, float coarse_cfo,
                                 const std::vector<C64>& ref_rs,
                                 C64& h, float& phase_est, float& sigma2) {
    if (rs_pos + RS_LEN > (int)symbols.size()) {
        h = C64(1.0f, 0.0f); phase_est = 0.0f; sigma2 = 0.1f;
        return false;
    }

    float totalCfo = coarse_cfo + fine_cfo;
    C64 sum(0, 0);
    for (int i = 0; i < RS_LEN; i++) {
        float phase = -2.0f * (float)M_PI * totalCfo * (rs_pos + i) * g_ts_sym;
        C64 corrected = symbols[rs_pos + i] * C64(std::cos(phase), std::sin(phase));
        sum += corrected * std::conj(ref_rs[i]);
    }
    h = sum / (float)RS_LEN;

    if (std::abs(h) < 1e-6f) {
        h = C64(1.0f, 0.0f); phase_est = 0.0f; sigma2 = 0.1f;
        return false;
    }

    phase_est = std::arg(h);

    float noiseSum = 0.0f;
    for (int i = 0; i < RS_LEN; i++) {
        float phase = -2.0f * (float)M_PI * totalCfo * (rs_pos + i) * g_ts_sym;
        C64 corrected = symbols[rs_pos + i] * C64(std::cos(phase), std::sin(phase));
        C64 eq = corrected / h;
        C64 err = eq - ref_rs[i];
        noiseSum += std::norm(err);
    }
    sigma2 = noiseSum / (RS_LEN - 1);
    if (sigma2 < 1e-30f) sigma2 = 1e-30f;
    return true;
}

// ===================================================================
// 10. RX: BPSK 硬判决解调 (对齐 loopback_test _bpsk_demod)
// CFO全补偿 + 除以h (含相位校正) → sign(real)
// ===================================================================

inline std::vector<int> demod_bpsk_hard(const std::vector<C64>& symbols,
                                         int data_start, int data_len,
                                         C64 h, float total_cfo) {
    std::vector<int> bits(data_len, 0);
    if (data_start + data_len > (int)symbols.size()) return bits;

    for (int i = 0; i < data_len; i++) {
        float phase = -2.0f * (float)M_PI * total_cfo * (data_start + i) * g_ts_sym;
        C64 corrected = symbols[data_start + i] * C64(std::cos(phase), std::sin(phase));
        if (std::abs(h) > 1e-30f) corrected /= h;
        bits[i] = (corrected.real() < 0.0f) ? 1 : 0;
    }
    return bits;
}

// ===================================================================
// 11. CRC 验证
// ===================================================================

inline bool verify_header(const std::vector<int>& hdr_bits) {
    if ((int)hdr_bits.size() < HEADER_LEN) return false;
    uint8_t hdrBytes[2] = {};
    for (int i = 0; i < 16; i++)
        hdrBytes[i / 8] = (uint8_t)((hdrBytes[i / 8] << 1) | (hdr_bits[i] & 1));
    uint16_t expected = 0;
    for (int i = 16; i < 32; i++)
        expected = (uint16_t)((expected << 1) | (hdr_bits[i] & 1));
    return crc16(hdrBytes, 2) == expected;
}

inline bool verify_payload_crc(const std::vector<int>& payloadBits,
                                const std::vector<int>& crcBits) {
    uint8_t payloadBytes[PAYLOAD_LEN / 8] = {};
    for (int i = 0; i < PAYLOAD_LEN; i++)
        payloadBytes[i / 8] = (uint8_t)((payloadBytes[i / 8] << 1) | (payloadBits[i] & 1));
    uint16_t expected = 0;
    for (int i = 0; i < CRC_LEN; i++)
        expected = (uint16_t)((expected << 1) | (crcBits[i] & 1));
    return crc16(payloadBytes, PAYLOAD_LEN / 8) == expected;
}

// ===================================================================
// 12. 环形缓冲
// ===================================================================

struct RingBuf {
    std::vector<C64> buf;
    int len = 0;
    int cap = 0;

    void init(int n) { cap = n; buf.resize(n); len = 0; }

    void append(const std::vector<C64>& samples) {
        int n = (int)samples.size();
        if (n > cap - len) {
            int discard = len - cap / 2;
            if (discard > 0) {
                std::memmove(buf.data(), buf.data() + discard,
                             (len - discard) * sizeof(C64));
                len -= discard;
            }
        }
        int space = std::min(n, cap - len);
        std::memcpy(buf.data() + len, samples.data(), space * sizeof(C64));
        len += space;
    }

    void consume(int endIdx) {
        if (endIdx <= 0) return;
        if (endIdx >= len) { len = 0; return; }
        int remain = len - endIdx;
        std::memmove(buf.data(), buf.data() + endIdx, remain * sizeof(C64));
        len = remain;
    }
};

// ===================================================================
// 13. 全局参考序列 (一次性惰性生成)
// ===================================================================

inline std::vector<C64>& ref_pss_instance() {
    static std::vector<C64> v = gen_pss();
    return v;
}

inline std::vector<C64>& ref_rs_instance() {
    static std::vector<C64> v = gen_rs();
    return v;
}

inline std::vector<float>& rrc_instance() {
    static std::vector<float> v = design_rrc();
    return v;
}

// 便捷宏 (避免重复调用 instance)
#define REF_PSS ref_pss_instance()
#define REF_RS  ref_rs_instance()
#define RRC     rrc_instance()
