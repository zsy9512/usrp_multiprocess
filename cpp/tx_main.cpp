/*
 * tx_main.cpp — BPSK PHY 独立发送端 (C++17, 零外部依赖)
 *
 * 帧结构: STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)
 *
 * 输入:  stdin (256 bits per frame, 32 bytes packed as uint8)
 *         或随机生成 (--random)
 * 输出:  IQ 二进制文件 (interleaved float32 I/Q, 等同 numpy complex64)
 *
 * 编译:  g++ -std=c++17 -O3 -march=native tx_main.cpp -o tx
 * 用法:  ./tx --random --num-frames 20 -o tx_iq.bin
 *        python polar_encoder.py | ./tx -o tx_iq.bin
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <complex>
#include <cstdint>
#include <ctime>

using C64 = std::complex<float>;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===================================================================
// 1.  帧参数 (与 phy_params.py / rx_main.cpp 严格一致)
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
constexpr int RRC_TAPS      = RRC_NUM_SYM * SPS + 1;  // = 21

// ===================================================================
// 2.  CRC16-IBM
// ===================================================================
static uint16_t crc16_table[256];
static void init_crc16() {
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
// 3.  参考序列 (与 Python / rx_main.cpp 严格一致)
// ===================================================================
static std::vector<C64> gen_stf() {
    int bits[STF_REP] = {1,-1,1,-1,1,1,1,1,-1,1,-1,1,-1,1,-1,-1};
    std::vector<C64> stf(STF_LEN);
    for (int i = 0; i < STF_NUM; i++)
        for (int j = 0; j < STF_REP; j++)
            stf[i*STF_REP+j] = C64((float)bits[j], 0.0f);
    return stf;
}
static std::vector<C64> gen_pss() {
    std::vector<C64> pss(PSS_LEN);
    for (int n = 0; n < PSS_LEN; n++) {
        float phase = -(float)M_PI * PSS_U * n * (n + 1) / PSS_LEN;
        pss[n] = C64(std::cos(phase), std::sin(phase));
    }
    return pss;
}
static std::vector<C64> gen_rs() {
    int bits[RS_LEN] = {-1,-1,-1,-1,-1,-1,-1,1, -1,-1,1,-1,-1,-1,-1,-1,
                        -1,1,1,-1,1,1,-1,-1, -1,1,1,1,1,-1,1,1};
    std::vector<C64> rs(RS_LEN);
    for (int i = 0; i < RS_LEN; i++) rs[i] = C64((float)bits[i], 0.0f);
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
// 4.  帧打包
// ===================================================================
static std::vector<C64> build_frame(const std::vector<int>& dataBits, uint16_t frameId) {
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

    // STF
    auto stf = gen_stf();
    frame.insert(frame.end(), stf.begin(), stf.end());

    // PSS
    auto pss = gen_pss();
    frame.insert(frame.end(), pss.begin(), pss.end());

    // RS
    auto rs = gen_rs();
    frame.insert(frame.end(), rs.begin(), rs.end());

    // Header: 16 frame_id + 16 CRC bits → BPSK
    for (int i = 15; i >= 0; i--)
        frame.push_back(C64(((frameId>>i)&1) ? -1.0f : 1.0f, 0.0f));
    for (int i = 15; i >= 0; i--) {
        int bit = (hdrCrc >> i) & 1;
        frame.push_back(C64(bit ? -1.0f : 1.0f, 0.0f));
    }

    // Payload: data bits → BPSK
    for (int bit : dataBits)
        frame.push_back(C64(bit ? -1.0f : 1.0f, 0.0f));

    // CRC bits → BPSK
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
// 5.  RRC 脉冲成形
// ===================================================================
static std::vector<C64> rrc_filter(const std::vector<C64>& symbols,
                                    const std::vector<float>& rrc) {
    int N = (int)symbols.size();
    int M = (int)rrc.size();
    // up-sample by SPS
    int upLen = N * SPS;
    std::vector<C64> up(upLen, C64(0.0f, 0.0f));
    for (int i = 0; i < N; i++)
        up[i * SPS] = symbols[i];

    // convolve (mode='full'), matching Python
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
// 6.  比特源
// ===================================================================
static std::vector<int> random_bits(int n) {
    std::vector<int> bits(n);
    for (int i = 0; i < n; i++)
        bits[i] = rand() & 1;
    return bits;
}

static std::vector<int> read_bits_stdin(uint16_t& frameId) {
    // read 4B frame_id + 32B = 36 bytes from stdin
    uint8_t buf[36];
    size_t n = fread(buf, 1, 36, stdin);
    if (n < 36) return {};
    frameId = ((uint16_t)buf[0]<<8) | buf[1];
    std::vector<int> bits(256);
    for (int i = 0; i < 32; i++)
        for (int b = 7; b >= 0; b--)
            bits[i*8 + (7-b)] = (buf[4+i] >> b) & 1;
    return bits;
}

// ===================================================================
// 7.  Main
// ===================================================================
int main(int argc, char* argv[]) {
    init_crc16();

    bool randomMode = false;
    int numFrames = 0;
    const char* outFile = "tx_iq.bin";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--random")) randomMode = true;
        else if (!strcmp(argv[i], "--num-frames") && i + 1 < argc) numFrames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) outFile = argv[++i];
        else {
            fprintf(stderr, "Usage: %s [--random] [--num-frames N] [-o out.bin]\n", argv[0]);
            fprintf(stderr, "  --random:     generate random payload bits\n");
            fprintf(stderr, "  --num-frames: number of frames (0=infinite)\n");
            fprintf(stderr, "  -o:           output IQ file (default: tx_iq.bin)\n");
            fprintf(stderr, "  (no --random): read 32 bytes/frame from stdin\n");
            return 1;
        }
    }

    if (!randomMode && numFrames == 0) {
        fprintf(stderr, "[tx] reading payload bits from stdin (32 bytes/frame)...\n");
    }

    auto rrc = design_rrc();
    FILE* fp = fopen(outFile, "wb");
    if (!fp) { perror("fopen"); return 1; }

    srand((unsigned)time(nullptr));
    int frameCount = 0;
    int totalSamples = 0;

    fprintf(stderr, "[tx] frame=%dsym RRC=%dtaps output=%s\n",
            FRAME_SYMBOLS, RRC_TAPS, outFile);

    while (numFrames == 0 || frameCount < numFrames) {
        std::vector<int> dataBits;
        uint16_t frameId = frameCount;
        if (randomMode)
            dataBits = random_bits(PAYLOAD_LEN);
        else {
            dataBits = read_bits_stdin(frameId);
            if (dataBits.empty()) break;
        }

        auto frameSyms = build_frame(dataBits, frameId);
        auto txSig = rrc_filter(frameSyms, rrc);

        // write interleaved float32 I/Q
        for (auto& s : txSig) {
            float iq[2] = { s.real(), s.imag() };
            fwrite(iq, sizeof(float), 2, fp);
        }
        totalSamples += (int)txSig.size();
        frameCount++;

        if (frameCount % 50 == 0)
            fprintf(stderr, "[tx] %d frames, %d samples\n", frameCount, totalSamples);
    }

    fclose(fp);
    float airTime = totalSamples / 1.0e6f * 1000.0f;
    fprintf(stderr, "[tx] done: %d frames, %d samples (%.1f ms @ 1Msps) → %s\n",
            frameCount, totalSamples, airTime, outFile);
    return 0;
}
