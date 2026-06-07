/*
 * tx_main.cpp — BPSK PHY 独立发送端 (C++17)
 *
 * 帧结构: STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)
 *
 * 输入:  stdin (32 bytes + 4B frame_id per frame)
 *         或随机生成 (--random)
 * 输出:  IQ 二进制文件 (interleaved float32 I/Q, 等同 numpy complex64)
 *
 * 编译:  g++ -std=c++17 -O3 -march=native tx_main.cpp -o tx
 * 用法:  ./tx --random --num-frames 20 -o tx_iq.bin
 *        python polar_encoder.py | ./tx -o tx_iq.bin
 */
#include "phy_dsp.h"

// ===================================================================
// 比特源 (仅 tx_main 使用)
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
// Main
// ===================================================================

int main(int argc, char* argv[]) {
    init_crc16_table();

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
