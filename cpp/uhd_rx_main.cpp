/*
 * uhd_rx_main.cpp — B210 UHD 接收端 (C++ API, MSVC 编译)
 *
 * 编译: 同 uhd_tx_main.cpp (需 vcvars64 + boost 编译好的库)
 */
#include "phy_dsp.h"
#include <uhd/usrp/multi_usrp.hpp>
#include <chrono>
#include <thread>
#include <csignal>

static volatile bool g_running = true;
static void sigint_handler(int) { g_running = false; }

int main(int argc, char* argv[]) {
    init_crc16_table();
    signal(SIGINT, sigint_handler);

    double freq = 915e6, gain = 30.0, rate = 1e6;
    double settle_s = 1.0;
    int rx_channel = 0;
    bool timed_start = false;
    std::string sync_mode = "host", args_str, rx_antenna = "RX2";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--sync-mode" && i+1<argc) sync_mode = argv[++i];
        else if (a == "--freq" && i+1<argc) freq = atof(argv[++i]);
        else if (a == "--gain" && i+1<argc) gain = atof(argv[++i]);
        else if (a == "--rate" && i+1<argc) rate = atof(argv[++i]);
        else if (a == "--settle" && i+1<argc) settle_s = atof(argv[++i]);
        else if (a == "--stf-threshold" && i+1<argc) g_stf_threshold = (float)atof(argv[++i]);
        else if (a == "--pss-ptm" && i+1<argc) g_pss_ptm_thr = (float)atof(argv[++i]);
        else if (a == "--pss-pts" && i+1<argc) g_pss_pts_thr = (float)atof(argv[++i]);
        else if (a == "--rs-corr-thr" && i+1<argc) g_rs_corr_thr = RS_LEN * (float)atof(argv[++i]);
        else if (a == "--rx-channel" && i+1<argc) rx_channel = atoi(argv[++i]);
        else if (a == "--rx-antenna" && i+1<argc) rx_antenna = argv[++i];
        else if (a == "--timed-start") timed_start = true;
        else if (a == "--args" && i+1<argc) args_str = argv[++i];
        else if (a == "-h" || a == "--help") {
            printf("Usage: uhd_rx_msvc [options]\n");
            return 0;
        }
    }

    g_ts = 1.0f / (float)rate;
    g_ts_sym = (float)SPS / (float)rate;
    printf("[uhd_rx] rate=%.0f freq=%.3fMHz gain=%.0fdB mode=%s\n",
           rate, freq/1e6, gain, sync_mode.c_str());

    // --- USRP ---
    auto usrp = uhd::usrp::multi_usrp::make(args_str);
    usrp->set_rx_freq(uhd::tune_request_t(freq), rx_channel);
    usrp->set_rx_gain(gain, rx_channel);
    usrp->set_rx_rate(rate, rx_channel);
    usrp->set_rx_bandwidth(rate, rx_channel);
    usrp->set_rx_antenna(rx_antenna, rx_channel);

    // --- Clock ---
    if (sync_mode == "host") {
        usrp->set_clock_source("internal");
        usrp->set_time_source("internal");
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        usrp->set_time_now(uhd::time_spec_t((double)ns / 1e9));
    } else if (sync_mode == "external_ref") {
        usrp->set_clock_source("external");
        usrp->set_time_source("internal");
        std::this_thread::sleep_for(std::chrono::duration<double>(settle_s));
        bool locked = usrp->get_mboard_sensor("ref_locked").to_bool();
        printf("[uhd_rx] ref_locked=%d\n", locked);
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        usrp->set_time_now(uhd::time_spec_t((double)ns / 1e9));
    }

    // --- RX stream ---
    uhd::stream_args_t stream_args("fc32", "sc16");
    stream_args.channels = {size_t(rx_channel)};
    auto rx_stream = usrp->get_rx_stream(stream_args);

    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    if (timed_start) {
        stream_cmd.stream_now = false;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        stream_cmd.time_spec = uhd::time_spec_t((double)ns / 1e9 + 2.0);
    } else {
        stream_cmd.stream_now = true;
    }
    rx_stream->issue_stream_cmd(stream_cmd);

    // --- Receive loop ---
    RingBuf ring;
    ring.init(2 * (int)rate);

    std::vector<C64> buf(4096);
    std::vector<C64> chunk;
    uhd::rx_metadata_t md;

    int totalFrames = 0, crcOkCnt = 0, hdrOkCnt = 0, falseAlarms = 0, overflowCount = 0;
    printf("[uhd_rx] receiving...\n");

    while (g_running) {
        size_t ns = rx_stream->recv(buf.data(), buf.size(), md, 0.5);
        if (ns == 0) continue;
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
            overflowCount++; continue;
        }

        chunk.assign(buf.begin(), buf.begin() + ns);
        ring.append(chunk);

        while (ring.len >= MIN_WIN_SAMPLES && g_running) {
            std::vector<C64> r(ring.buf.begin(), ring.buf.begin() + ring.len);
            auto stf = stf_delay_correlate(r);
            if (stf.metric.empty()) break;

            auto clustered = stf_cluster_peaks(stf, r, (float)rate);

            if (clustered.peaks.empty()) {
                ring.consume(std::min(ADVANCE_SAMPLES, ring.len));
                continue;
            }

            bool frameFound = false;
            int maxCand = std::min(8, (int)clustered.peaks.size());
            for (int ci = 0; ci < maxCand && !frameFound; ci++) {
                int candD = clustered.peaks[ci];
                float coarseCfo = clustered.cfos[ci];

                int extractStart = std::max(0, candD - EXTRACT_EXTRA);
                int extractEnd = std::min(ring.len, candD + EXTRACT_EXTRA + FRAME_RRC_SAMPLES + EXTRACT_EXTRA);
                std::vector<C64> chunkR(r.begin()+extractStart, r.begin()+extractEnd);
                if ((int)chunkR.size() < PSS_LEN * SPS) continue;

                auto symbols = rrc_match(chunkR, RRC);
                if ((int)symbols.size() < PSS_LEN + RS_LEN) continue;

                auto pssRes = pss_correlate(symbols, REF_PSS);
                bool pssOk = pssRes.peak_to_mean >= g_pss_ptm_thr
                          && pssRes.peak_to_second >= g_pss_pts_thr;
                if (!pssOk) continue;

                int frameSymStart = pssRes.peak_idx - STF_LEN;
                if (frameSymStart < 0) continue;
                int frameSampleStart = extractStart + frameSymStart * SPS - RRC_DELAY;
                if (frameSampleStart < 0) continue;

                int rsSymStart = frameSymStart + STF_LEN + PSS_LEN;
                float rsCorr = 0.0f;
                float fineCfo = rs_fine_cfo(symbols, rsSymStart, coarseCfo, REF_RS, rsCorr);
                if (rsCorr < g_rs_corr_thr) continue;
                if (std::abs(fineCfo) > g_fine_cfo_max) continue;

                C64 h; float phaseEst, sigma2;
                if (!rs_channel_estimate(symbols, rsSymStart, fineCfo, coarseCfo, REF_RS, h, phaseEst, sigma2))
                    continue;
                sigma2 = std::min(sigma2, g_sigma2_max);

                float totalCfo = coarseCfo + fineCfo;

                int hdrStart = frameSymStart + STF_LEN + PSS_LEN + RS_LEN;
                auto hdrBits = demod_bpsk_hard(symbols, hdrStart, HEADER_LEN, h, totalCfo);
                bool hdrOk = verify_header(hdrBits);

                int payStart = hdrStart + HEADER_LEN;
                auto payBits = demod_bpsk_hard(symbols, payStart, PAYLOAD_LEN + CRC_LEN, h, totalCfo);
                std::vector<int> payloadBits(payBits.begin(), payBits.begin() + PAYLOAD_LEN);
                std::vector<int> crcBits(payBits.begin() + PAYLOAD_LEN, payBits.begin() + PAYLOAD_LEN + CRC_LEN);
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
                ring.consume(std::min(ADVANCE_SAMPLES, ring.len));
            }
        }
    }

    rx_stream->issue_stream_cmd(uhd::stream_cmd_t(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS));
    fprintf(stderr, "\n--- 结果 ---\n");
    fprintf(stderr, "  frames=%d  CRC=%d/%d (%.1f%%)  HDR=%d  false_alarms=%d\n",
            totalFrames, crcOkCnt, totalFrames,
            totalFrames>0?100.0*crcOkCnt/totalFrames:0.0,
            hdrOkCnt, falseAlarms);
    return 0;
}
