/*
 * rx_main.cpp — BPSK PHY 独立接收端 (C++17, 零外部依赖)
 *
 * 三级同步链:
 *   ① STF 延迟相关 → 粗包检测 + 粗 CFO
 *   ② PSS 互相关   → 精定时 + 双质量判据
 *   ③ RS 相位拟合  → 细 CFO + 信道估计 + 噪声方差
 *
 * 帧结构: STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)
 *
 * 输入:  IQ 二进制文件 (interleaved float32 I/Q, 等同 numpy complex64)
 * 输出:  每帧 256 个 float32 LLR → stdout
 *
 * 编译:  g++ -std=c++17 -O3 -march=native rx_main.cpp -o rx
 * 用法:  ./rx rx_iq.bin > llr.bin
 *        python receiver.py 生成 tx_iq.npy → 转 .bin 后验证
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using C64 = std::complex<float>;

// ===================================================================
// 1.  帧参数 (与 phy_params.py 严格一致)
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
constexpr float TS          = 1.0f / 1.0e6f;       // symbol time @ 1Msps

// RRC filter length = RRC_NUM_SYM*SPS + 1
constexpr int RRC_TAPS      = RRC_NUM_SYM * SPS + 1;   // = 21
constexpr int RRC_DELAY     = (RRC_TAPS - 1) / 2;  // 10
constexpr int FRAME_RRC_SAMPLES = FRAME_SYMBOLS * SPS + (RRC_TAPS - 1); // 1012

// sync parameters
constexpr float STF_THRESHOLD      = 0.40f;
constexpr float STF_MIN_ENERGY     = 0.10f * STF_DELAY;
constexpr float PSS_PTM_THR        = 4.0f;
constexpr int   PSS_SEARCH_WIN     = STF_LEN * SPS * 2;  // 256
constexpr int   EXTRACT_EXTRA      = FRAME_SYMBOLS * SPS;  // extra room for payload
constexpr int   ADVANCE_SAMPLES    = (PSS_LEN + RS_LEN + HEADER_LEN) * SPS;
constexpr int   MIN_WIN_SAMPLES    = FRAME_RRC_SAMPLES + PSS_SEARCH_WIN;
constexpr float RS_CORR_THRESHOLD  = RS_LEN * 0.3f;   // 9.6

// ===================================================================
// 2.  CRC16-IBM
// ===================================================================
static uint16_t crc16_table[256];

static void init_crc16_table() {
    for (int i = 0; i < 256; i++) {
        uint16_t crc = (uint16_t)(i << 8);
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x8005) : (uint16_t)(crc << 1);
        crc16_table[i] = crc;
    }
}

static uint16_t crc16(const uint8_t* data, int len) {
    uint16_t crc = 0;
    for (int i = 0; i < len; i++)
        crc = (uint16_t)((crc << 8) ^ crc16_table[((crc >> 8) ^ data[i]) & 0xFF]);
    return crc;
}

// ===================================================================
// 3.  参考序列生成 (固定种子, 与 Python 一致)
// ===================================================================
static std::vector<C64> gen_pss() {
    std::vector<C64> pss(PSS_LEN);
    for (int n = 0; n < PSS_LEN; n++) {
        float phase = -(float)M_PI * PSS_U * n * (n + 1) / PSS_LEN;
        pss[n] = C64(std::cos(phase), std::sin(phase));
    }
    return pss;
}

static std::vector<C64> gen_rs() {
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

static std::vector<float> design_rrc() {
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
// 4.  DSP 函数
// ===================================================================

// --- 4a. RRC 匹配滤波 (正确实现 mode='full' 卷积 + 抽取) ---
static std::vector<C64> rrc_match(const std::vector<C64>& samples,
                                   const std::vector<float>& rrc) {
    int N = (int)samples.size();
    int M = (int)rrc.size();
    // Step 1: full convolution with kernel = rrc[::-1]
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
    // Step 2: decimate from delay
    int delay = (M - 1) / 2;  // RRC_DELAY = 10
    int nOut = (convLen - delay + SPS - 1) / SPS;
    std::vector<C64> out(nOut);
    for (int k = 0; k < nOut; k++)
        out[k] = filt[delay + k * SPS];
    return out;
}

// --- 4b. STF 延迟相关 (包检测 + 粗 CFO) ---
struct StfResult {
    std::vector<float> metric;
    std::vector<C64> P;
};

static StfResult stf_delay_correlate(const std::vector<C64>& samples) {
    int L = STF_DELAY;
    int N = (int)samples.size();
    StfResult res;
    if (N <= L) return res;

    // P(d) = sum_{n=0}^{L-1} r[d+n] * conj(r[d+n+L])
    // E(d) = sum_{n=0}^{L-1} |r[d+n+L]|^2
    // M(d) = |P(d)| / (E(d) + noise_floor)

    // compute running sums efficiently
    int numP = N - 2 * L + 1;
    res.metric.resize(numP);
    res.P.resize(numP);

    // sliding window of prod = r[n] * conj(r[n+L])
    std::vector<C64> prod(N - L);
    for (int i = 0; i < N - L; i++)
        prod[i] = samples[i] * std::conj(samples[i + L]);

    // running sum of last L prod values
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

static float compute_coarse_cfo(C64 P_peak, float samp_rate) {
    float phase = std::arg(P_peak);
    return -phase / (2.0f * (float)M_PI * STF_DELAY / samp_rate);
}

// --- 4c. PSS 互相关 (精定时 + 质量判据) ---
struct PssResult {
    int peak_idx = 0;
    float peak_to_mean = 0.0f;
    float peak_to_second = 0.0f;
};

static PssResult pss_correlate(const std::vector<C64>& symbols,
                                const std::vector<C64>& ref_pss) {
    PssResult res;
    int N = (int)symbols.size();
    int M = PSS_LEN;
    if (N < M) return res;

    // sliding correlate: convolve(symbols, conj(ref_pss[::-1]))
    int corrLen = N - M + 1;
    std::vector<float> corrMag(corrLen, 0.0f);
    float sumAll = 0.0f;

    for (int i = 0; i < corrLen; i++) {
        C64 dot(0, 0);
        for (int j = 0; j < M; j++)
            dot += symbols[i + j] * std::conj(ref_pss[M - 1 - j]);
        // non-coherent (abs)
        corrMag[i] = std::abs(dot);
        sumAll += corrMag[i];
    }

    // peak
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

    // find second peak (sorted, first one > PSS_LEN/2 away from main peak)
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

// --- 4d. RS 细 CFO (粗 CFO 预补偿 + 线性相位拟合) ---
static float rs_fine_cfo(const std::vector<C64>& symbols, int rs_pos,
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
        // coarse CFO pre-compensation
        if (std::abs(coarse_cfo) > 0.0f) {
            float phase = -2.0f * (float)M_PI * coarse_cfo * (rs_pos + i) * TS;
            rs_seg[i] *= C64(std::cos(phase), std::sin(phase));
        }
    }

    // dot with ref
    C64 dotSum(0, 0);
    for (int i = 0; i < RS_LEN; i++)
        dotSum += rs_seg[i] * std::conj(ref_rs[i]);
    rs_corr = std::abs(dotSum);

    // unwrap phase + linear fit
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
    return slope / (2.0f * (float)M_PI * TS);
}

// --- 4e. RS 信道估计 ---
static void rs_channel_estimate(const std::vector<C64>& symbols, int rs_pos,
                                 float fine_cfo, float coarse_cfo,
                                 const std::vector<C64>& ref_rs,
                                 C64& h, float& phase_est, float& sigma2) {
    if (rs_pos + RS_LEN > (int)symbols.size()) {
        h = C64(1.0f, 0.0f); phase_est = 0.0f; sigma2 = 0.1f;
        return;
    }

    float totalCfo = coarse_cfo + fine_cfo;
    C64 sum(0, 0);
    for (int i = 0; i < RS_LEN; i++) {
        float phase = -2.0f * (float)M_PI * totalCfo * (rs_pos + i) * TS;
        C64 corrected = symbols[rs_pos + i] * C64(std::cos(phase), std::sin(phase));
        sum += corrected * std::conj(ref_rs[i]);
    }
    h = sum / (float)RS_LEN;

    if (std::abs(h) < 1e-30f) {
        h = C64(1.0f, 0.0f); phase_est = 0.0f; sigma2 = 0.1f;
        return;
    }

    phase_est = std::arg(h);

    // noise variance with Welch correction
    float noiseSum = 0.0f;
    for (int i = 0; i < RS_LEN; i++) {
        float phase = -2.0f * (float)M_PI * totalCfo * (rs_pos + i) * TS;
        C64 corrected = symbols[rs_pos + i] * C64(std::cos(phase), std::sin(phase));
        C64 eq = corrected / h;
        C64 err = eq - ref_rs[i];
        noiseSum += std::norm(err);
    }
    sigma2 = noiseSum / (RS_LEN - 1);  // Welch: divide by N-1
    if (sigma2 < 1e-30f) sigma2 = 1e-30f;
}

// --- 4f. BPSK LLR 解调 ---
static std::vector<float> demod_llr(const std::vector<C64>& symbols,
                                     int data_start, int data_len,
                                     C64 h, float phase_est,
                                     float fine_cfo, float coarse_cfo,
                                     float sigma2) {
    std::vector<float> llr(data_len, 0.0f);
    if (data_start + data_len > (int)symbols.size()) return llr;

    float totalCfo = coarse_cfo + fine_cfo;

    for (int i = 0; i < data_len; i++) {
        float phase = -2.0f * (float)M_PI * totalCfo * (data_start + i) * TS;
        C64 corrected = symbols[data_start + i] * C64(std::cos(phase), std::sin(phase));
        corrected *= C64(std::cos(-phase_est), std::sin(-phase_est)); // rotate by -phase_est
        if (std::abs(h) > 1e-30f) corrected /= h;

        float val = 2.0f * corrected.real() / sigma2;
        val = std::max(-50.0f, std::min(50.0f, val));
        llr[i] = val;
    }
    return llr;
}

// --- 4g. CRC 验证辅助 ---
static bool verify_header(const std::vector<int>& hdr_bits) {
    if ((int)hdr_bits.size() < HEADER_LEN) return false;
    // Header: bits[0:16]=reserved, bits[16:32]=CRC
    uint8_t hdrBytes[2] = {};
    for (int i = 0; i < 16; i++)
        hdrBytes[i / 8] = (uint8_t)((hdrBytes[i / 8] << 1) | (hdr_bits[i] & 1));
    uint16_t expected = 0;
    for (int i = 16; i < 32; i++)
        expected = (uint16_t)((expected << 1) | (hdr_bits[i] & 1));
    return crc16(hdrBytes, 2) == expected;
}

static bool verify_payload_crc(const std::vector<int>& payloadBits,
                                const std::vector<int>& crcBits) {
    uint8_t payloadBytes[PAYLOAD_LEN / 8] = {};
    for (int i = 0; i < PAYLOAD_LEN; i++)
        payloadBytes[i / 8] = (uint8_t)((payloadBytes[i / 8] << 1) | (payloadBits[i] & 1));
    uint16_t expected = 0;
    for (int i = 0; i < CRC_LEN; i++)
        expected = (uint16_t)((expected << 1) | (crcBits[i] & 1));
    return crc16(payloadBytes, PAYLOAD_LEN / 8) == expected;
}

static std::vector<int> hard_decision(const std::vector<float>& llr) {
    std::vector<int> bits(llr.size());
    for (size_t i = 0; i < llr.size(); i++)
        bits[i] = llr[i] < 0.0f ? 1 : 0;
    return bits;
}

// ===================================================================
// 5.  帧检测主循环
// ===================================================================

// 环形缓冲 (简易版, 固定 2秒容量)
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

// 全局只读参考序列 (一次生成)
static const std::vector<C64> REF_PSS = gen_pss();
static const std::vector<C64> REF_RS  = gen_rs();
static const std::vector<float> RRC   = design_rrc();

int main(int argc, char* argv[]) {
    init_crc16_table();

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <iq_file.bin>\n", argv[0]);
        fprintf(stderr, "  iq_file.bin: interleaved float32 I/Q pairs\n");
        fprintf(stderr, "  Output: 256 float32 LLR per frame → stdout\n");
        return 1;
    }

    // --- open IQ file ---
    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { perror("fopen"); return 1; }
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    long numFloats = fileSize / (long)sizeof(float);
    fprintf(stderr, "[rx] file: %s  samples: %ld  (%.1f ms @ 1Msps)\n",
            argv[1], numFloats / 2, (numFloats / 2) / 1000.0);

    // --- streaming state ---
    RingBuf ring;
    ring.init(2000000); // 2 seconds
    int totalFrames = 0, crcPass = 0;

    // read in chunks and process
    constexpr int CHUNK_FLOATS = 200000; // 100k samples per chunk
    std::vector<float> floatBuf(CHUNK_FLOATS);
    std::vector<C64> chunk;
    long pos = 0;

    while (pos < numFloats) {
        long toRead = std::min((long)CHUNK_FLOATS, numFloats - pos);
        size_t nread = fread(floatBuf.data(), sizeof(float), (size_t)toRead, fp);
        if (nread == 0) break;
        pos += (long)nread;

        // interleaved float → complex
        int nsamp = (int)nread / 2;
        chunk.resize(nsamp);
        for (int i = 0; i < nsamp; i++)
            chunk[i] = C64(floatBuf[2 * i], floatBuf[2 * i + 1]);

        ring.append(chunk);

        // --- process while enough data ---
        while (ring.len >= MIN_WIN_SAMPLES) {
            // stage 1: STF delay correlation
            std::vector<C64> r(ring.buf.begin(), ring.buf.begin() + ring.len);
            auto stf = stf_delay_correlate(r);
            if (stf.metric.empty()) break;

            // collect candidates
            std::vector<int> candidates;
            for (int d = 0; d < (int)stf.metric.size(); d++) {
                if (stf.metric[d] > STF_THRESHOLD) {
                    float localE = 0.0f;
                    for (int j = STF_DELAY; j < 2 * STF_DELAY; j++)
                        localE += std::norm(r[d + j]);
                    if (localE > STF_MIN_ENERGY)
                        candidates.push_back(d);
                }
            }

            if (candidates.empty()) {
                int adv = std::min(ADVANCE_SAMPLES, ring.len);
                ring.consume(adv);
                continue;
            }

            bool frameFound = false;
            int maxCand = std::min(32, (int)candidates.size());
            for (int ci = 0; ci < maxCand && !frameFound; ci++) {
                int candD = candidates[ci];

                float coarseCfo = 0.0f;
                if (candD < (int)stf.P.size())
                    coarseCfo = compute_coarse_cfo(stf.P[candD], 1e6f);

                // stage 2: PSS
                int margin = PSS_SEARCH_WIN;
                int extractStart = std::max(0, candD - margin);
                int extractEnd = std::min(ring.len,
                    candD + margin + FRAME_RRC_SAMPLES + EXTRACT_EXTRA);

                std::vector<C64> chunkR(r.begin() + extractStart,
                                        r.begin() + extractEnd);
                auto symbols = rrc_match(chunkR, RRC);
                if ((int)symbols.size() < PSS_LEN + RS_LEN) continue;

                auto pssRes = pss_correlate(symbols, REF_PSS);
                if (pssRes.peak_to_mean < PSS_PTM_THR) continue;

                int pssStart = pssRes.peak_idx;
                int frameSymStart = pssStart - STF_LEN;
                if (frameSymStart < 0) continue;

                int frameSampleStart = extractStart + frameSymStart * SPS - RRC_DELAY;
                if (frameSampleStart < 0) continue;

                // stage 3: RS CFO + channel
                int rsSymStart = frameSymStart + STF_LEN + PSS_LEN;
                float rsCorr = 0.0f;
                float fineCfo = rs_fine_cfo(symbols, rsSymStart, coarseCfo,
                                             REF_RS, rsCorr);
                if (rsCorr < RS_CORR_THRESHOLD) continue;

                C64 h;
                float phaseEst, sigma2;
                rs_channel_estimate(symbols, rsSymStart, fineCfo, coarseCfo,
                                    REF_RS, h, phaseEst, sigma2);

                // stage 4: demod + CRC
                int hdrStart = frameSymStart + STF_LEN + PSS_LEN + RS_LEN;
                auto hdrLlr = demod_llr(symbols, hdrStart, HEADER_LEN,
                                        h, phaseEst, fineCfo, coarseCfo, sigma2);
                auto hdrBits = hard_decision(hdrLlr);
                bool hdrOk = verify_header(hdrBits);

                int payStart = hdrStart + HEADER_LEN;
                auto payLlr = demod_llr(symbols, payStart,
                                        PAYLOAD_LEN + CRC_LEN,
                                        h, phaseEst, fineCfo, coarseCfo, sigma2);
                auto payBits = hard_decision(payLlr);
                std::vector<int> payloadBits(payBits.begin(),
                                             payBits.begin() + PAYLOAD_LEN);
                std::vector<int> crcBits(payBits.begin() + PAYLOAD_LEN,
                                         payBits.begin() + PAYLOAD_LEN + CRC_LEN);
                bool payCrcOk = verify_payload_crc(payloadBits, crcBits);

                totalFrames++;
                if (payCrcOk) crcPass++;

                // metrics
                float totalCfo = coarseCfo + fineCfo;
                float snrDb = 10.0f * std::log10(std::max(std::norm(h) / sigma2, 1e-30f));

                if (totalFrames <= 5 || totalFrames % 50 == 0) {
                    fprintf(stderr,
                        "  frame=%5d  ptm=%.1f  pts=%.1f  "
                        "cf0=%+.0f  cf1=%+.0f  Δf=%+.0fHz  "
                        "θ=%.3frad  |h|=%.3f  σ²=%.4f  SNR=%.1fdB  "
                        "HDR=%s  CRC=%s\n",
                        totalFrames, pssRes.peak_to_mean, pssRes.peak_to_second,
                        coarseCfo, fineCfo, totalCfo,
                        phaseEst, std::abs(h), sigma2, snrDb,
                        hdrOk ? "OK" : "XX", payCrcOk ? "OK" : "XX");
                    fflush(stderr);
                }

                // stage 5: output LLR (payload only, 256 floats)
                auto payloadLlr = std::vector<float>(
                    payLlr.begin(), payLlr.begin() + PAYLOAD_LEN);
                fwrite(payloadLlr.data(), sizeof(float), PAYLOAD_LEN, stdout);
                fflush(stdout);

                // consume window
                int consumeEnd = frameSampleStart + FRAME_RRC_SAMPLES;
                if (consumeEnd > ring.len) consumeEnd = ring.len;
                ring.consume(consumeEnd);
                frameFound = true;
            }

            if (!frameFound) {
                int adv = std::min(ADVANCE_SAMPLES, ring.len);
                ring.consume(adv);
            }
        }
    }

    fclose(fp);
    fprintf(stderr, "\n[rx] done: %d frames, CRC pass: %d/%d (%.1f%%)\n",
            totalFrames, crcPass, totalFrames,
            totalFrames > 0 ? 100.0f * crcPass / totalFrames : 0.0f);
    return 0;
}
