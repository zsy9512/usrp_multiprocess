/*
 * rx_main.cpp — BPSK PHY 独立接收端 (C++17)
 *
 * 三级同步链:
 *   ① STF 延迟相关 → 粗包检测 + 粗 CFO
 *   ② PSS 互相关   → 精定时 + 双质量判据 (peak_to_mean + peak_to_second)
 *   ③ RS 相位拟合  → 细 CFO + 信道估计 + 噪声方差
 *
 * 帧结构: STF(64) + PSS(64) + RS(32) + Header(32) + Payload(256) + CRC(16) + Guard(32)
 *
 * 输入:  IQ 二进制文件 (interleaved float32 I/Q, 等同 numpy complex64)
 * 输出:  统计信息 (stderr)
 *
 * 编译:  g++ -std=c++17 -O3 -march=native rx_main.cpp -o rx
 * 用法:  ./rx [--stf-threshold 0.4] [--pss-ptm 3.5] [--pss-pts 1.5]
 *             [--rs-corr-thr 0.3] [--rate 1e6] <iq_file.bin>
 */
#include "phy_dsp.h"

// ===================================================================
// Main
// ===================================================================

int main(int argc, char* argv[]) {
    init_crc16_table();

    const char* iqFile = nullptr;
    float rate = 1e6f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rate") && i + 1 < argc) rate = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--stf-threshold") && i + 1 < argc)
            g_stf_threshold = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--pss-ptm") && i + 1 < argc)
            g_pss_ptm_thr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--pss-pts") && i + 1 < argc)
            g_pss_pts_thr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--rs-corr-thr") && i + 1 < argc)
            g_rs_corr_thr = RS_LEN * (float)atof(argv[++i]);
        else if (argv[i][0] != '-') iqFile = argv[i];
        else {
            fprintf(stderr, "Usage: %s [--rate R] [--stf-threshold F]\n", argv[0]);
            fprintf(stderr, "       [--pss-ptm F] [--pss-pts F] [--rs-corr-thr F] <iq_file.bin>\n");
            fprintf(stderr, "  --rate:         sample rate Hz (default 1e6)\n");
            fprintf(stderr, "  --stf-threshold: STF detection threshold (default 0.4)\n");
            fprintf(stderr, "  --pss-ptm:      PSS peak-to-mean threshold (default 4.0)\n");
            fprintf(stderr, "  --pss-pts:      PSS peak-to-second threshold (default 1.5)\n");
            fprintf(stderr, "  --rs-corr-thr:  RS correlation threshold factor (default 0.3)\n");
            fprintf(stderr, "  Output: [4B frame_id][256 float LLR][1B crc_ok] per frame → stdout\n");
            return 1;
        }
    }

    if (!iqFile) {
        fprintf(stderr, "Usage: %s [options] <iq_file.bin>\n", argv[0]);
        return 1;
    }

    // 设置采样时间
    g_ts = 1.0f / rate;
    g_ts_sym = (float)SPS / rate;

    FILE* fp = fopen(iqFile, "rb");
    if (!fp) { perror("fopen"); return 1; }
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    long numFloats = fileSize / (long)sizeof(float);
    fprintf(stderr, "[rx] file: %s  samples: %ld  rate=%.0fHz\n",
            iqFile, numFloats / 2, (double)rate);

    RingBuf ring;
    ring.init(2 * (int)rate); // 2 seconds

    int totalFrames = 0, crcOkCnt = 0, hdrOkCnt = 0, falseAlarms = 0;

    constexpr int CHUNK_FLOATS = 200000;
    std::vector<float> floatBuf(CHUNK_FLOATS);
    std::vector<C64> chunk;
    long pos = 0;

    while (pos < numFloats) {
        long toRead = std::min((long)CHUNK_FLOATS, numFloats - pos);
        size_t nread = fread(floatBuf.data(), sizeof(float), (size_t)toRead, fp);
        if (nread == 0) break;
        pos += (long)nread;

        int nsamp = (int)nread / 2;
        chunk.resize(nsamp);
        for (int i = 0; i < nsamp; i++)
            chunk[i] = C64(floatBuf[2 * i], floatBuf[2 * i + 1]);

        ring.append(chunk);

        while (ring.len >= MIN_WIN_SAMPLES) {
            std::vector<C64> r(ring.buf.begin(), ring.buf.begin() + ring.len);
            auto stf = stf_delay_correlate(r);
            if (stf.metric.empty()) break;

            auto clustered = stf_cluster_peaks(stf, r, rate);

            if (clustered.peaks.empty()) {
                int adv = std::min(ADVANCE_SAMPLES, ring.len);
                ring.consume(adv);
                continue;
            }

            bool frameFound = false;
            int maxCand = std::min(16, (int)clustered.peaks.size());
            for (int ci = 0; ci < maxCand && !frameFound; ci++) {
                int candD = clustered.peaks[ci];
                float coarseCfo = clustered.cfos[ci];

                int extractStart = std::max(0, candD - EXTRACT_EXTRA);
                int extractEnd = std::min(ring.len,
                    candD + EXTRACT_EXTRA + FRAME_RRC_SAMPLES + EXTRACT_EXTRA);

                std::vector<C64> chunkR(r.begin() + extractStart,
                                        r.begin() + extractEnd);
                if ((int)chunkR.size() < PSS_LEN * SPS) continue;
                auto symbols = rrc_match(chunkR, RRC);
                if ((int)symbols.size() < PSS_LEN + RS_LEN) continue;

                auto pssRes = pss_correlate(symbols, REF_PSS);
                bool pssOk = pssRes.peak_to_mean >= g_pss_ptm_thr
                          && pssRes.peak_to_second >= g_pss_pts_thr;
                if (!pssOk) continue;

                int pssStart = pssRes.peak_idx;
                int frameSymStart = pssStart - STF_LEN;
                if (frameSymStart < 0) continue;

                int frameSampleStart = extractStart + frameSymStart * SPS - RRC_DELAY;
                if (frameSampleStart < 0) continue;

                int rsSymStart = frameSymStart + STF_LEN + PSS_LEN;
                float rsCorr = 0.0f;
                float fineCfo = rs_fine_cfo(symbols, rsSymStart, coarseCfo,
                                             REF_RS, rsCorr);
                if (rsCorr < g_rs_corr_thr) continue;
                if (std::abs(fineCfo) > g_fine_cfo_max) continue;

                C64 h;
                float phaseEst, sigma2;
                if (!rs_channel_estimate(symbols, rsSymStart, fineCfo, coarseCfo,
                                    REF_RS, h, phaseEst, sigma2))
                    continue;
                sigma2 = std::min(sigma2, g_sigma2_max);

                float totalCfo = coarseCfo + fineCfo;

                int hdrStart = frameSymStart + STF_LEN + PSS_LEN + RS_LEN;
                auto hdrBits = demod_bpsk_hard(symbols, hdrStart, HEADER_LEN,
                                        h, totalCfo);
                bool hdrOk = verify_header(hdrBits);

                int payStart = hdrStart + HEADER_LEN;
                auto payBits = demod_bpsk_hard(symbols, payStart,
                                        PAYLOAD_LEN + CRC_LEN,
                                        h, totalCfo);
                std::vector<int> payloadBits(payBits.begin(),
                                             payBits.begin() + PAYLOAD_LEN);
                std::vector<int> crcBits(payBits.begin() + PAYLOAD_LEN,
                                         payBits.begin() + PAYLOAD_LEN + CRC_LEN);
                bool payCrcOk = verify_payload_crc(payloadBits, crcBits);

                totalFrames++;
                if (hdrOk) hdrOkCnt++;
                if (payCrcOk) crcOkCnt++;

                float hmag = std::abs(h);
                float snrDb = 10.0f * std::log10(std::max(hmag * hmag / sigma2, 1e-30f));

                if (totalFrames <= 5 || totalFrames % 100 == 0) {
                    fprintf(stderr,
                        "  frame=%5d  "
                        "ptm=%.1f  pts=%.1f  "
                        "\u0394f0=%+.0f  "
                        "\u0394f1=%+.0f  "
                        "|h|=%.3f  SNR=%.1fdB  "
                        "HDR=%s  CRC=%s\n",
                        totalFrames,
                        pssRes.peak_to_mean, pssRes.peak_to_second,
                        coarseCfo, fineCfo,
                        hmag, snrDb,
                        hdrOk ? "OK" : "XX", payCrcOk ? "OK" : "XX");
                    fflush(stderr);
                }

                int consumeEnd = extractStart + frameSymStart * SPS + FRAME_RRC_SAMPLES + 50;
                if (consumeEnd > ring.len) consumeEnd = ring.len;
                ring.consume(consumeEnd);
                frameFound = true;
            }

            if (!frameFound) {
                falseAlarms += (int)clustered.peaks.size();
                int adv = std::min(ADVANCE_SAMPLES, ring.len);
                ring.consume(adv);
            }
        }
    }

    fclose(fp);
    fprintf(stderr, "\n--- 结果 ---\n");
    fprintf(stderr, "  frames=%d  CRC=%d/%d (%.1f%%)  HDR=%d  false_alarms=%d\n",
            totalFrames, crcOkCnt, totalFrames,
            totalFrames > 0 ? 100.0f * crcOkCnt / totalFrames : 0.0f,
            hdrOkCnt, falseAlarms);
    return 0;
}
