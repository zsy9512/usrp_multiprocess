/*
 * loopback.cpp — B210 single-device loopback (C++ version of loopback_test.py)
 *
 * TX thread: random bits → build_frame → rrc_filter → USRP TX
 * RX thread: USRP RX → ring buffer
 * Main thread: ring buffer → STF+PSS+RS sync → BPSK demod → CRC → stats
 *
 * Usage:
 *   loopback_msvc.exe --args serial=320F33F --freq 915e6 --gain-tx 65 --gain-rx 64
 *
 * Build: cl /EHsc /O2 /std:c++17 loopback.cpp /link uhd.lib ws2_32.lib boost_*.lib
 */
#include "phy_dsp.h"
#include <uhd/usrp/multi_usrp.hpp>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <csignal>
#include <cinttypes>

// ===================================================================
// Ring buffer extension (phy_dsp.h RingBuf + write/copy_to methods)
// ===================================================================
static constexpr int RING_CAP = 2'000'000;

struct SharedRing : public RingBuf {
    std::atomic<size_t> wr_count{0};

    SharedRing() { init(RING_CAP); }

    void write_samples(const C64* data, size_t n, size_t& w) {
        size_t end = w + n;
        if (end <= (size_t)cap) {
            std::memcpy(buf.data() + w, data, n * sizeof(C64));
        } else {
            size_t n1 = cap - w;
            std::memcpy(buf.data() + w, data, n1 * sizeof(C64));
            std::memcpy(buf.data(), data + n1, (n - n1) * sizeof(C64));
        }
        w = end % cap;
        wr_count.fetch_add(n, std::memory_order_release);
    }

    void copy_to_vec(std::vector<C64>& dst, size_t pos, size_t len) {
        pos %= cap;
        if (pos + len <= (size_t)cap) {
            std::memcpy(dst.data(), buf.data() + pos, len * sizeof(C64));
        } else {
            size_t n1 = cap - pos;
            std::memcpy(dst.data(), buf.data() + pos, n1 * sizeof(C64));
            std::memcpy(dst.data() + n1, buf.data(), (len - n1) * sizeof(C64));
        }
    }
};

// ===================================================================
// PHY processing
// ===================================================================
struct PhyStats {
    int total = 0, hdr_ok = 0, crc_ok = 0, false_alarms = 0;
};

static PhyStats g_stats;
static std::mutex g_print_mtx;

static void print_frame(int total, float snr_db, bool hdrOk, bool payCrcOk) {
    std::lock_guard<std::mutex> lk(g_print_mtx);
    fprintf(stderr, "  frame=%5d  SNR=%.1fdB  HDR=%s  CRC=%s\n",
        total, snr_db,
        hdrOk ? "OK" : "XX", payCrcOk ? "OK" : "XX");
    fflush(stderr);
}

// ---- RRC matched filter ----
static std::vector<C64> rrc_match_local(const std::vector<C64>& samples) {
    int N = (int)samples.size();
    int M = (int)RRC.size();
    int convLen = N + M - 1;
    std::vector<C64> filt(convLen, C64(0,0));
    for (int i = 0; i < convLen; i++) {
        C64 sum(0,0);
        for (int j = 0; j < M; j++) {
            int sidx = i - j;
            if (sidx >= 0 && sidx < N)
                sum += samples[sidx] * RRC[M - 1 - j];
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

// ---- STF clustering (wrap from phy_dsp) ----
static StfClusterResult stf_cluster_wrap(const StfResult& stf,
                                          const std::vector<C64>& samples,
                                          float samp_rate) {
    return stf_cluster_peaks(stf, samples, samp_rate);
}

// ---- RS estimate (combined coarse+fine CFO, channel, noise) ----
struct ChanEst {
    C64 h = C64(1,0);
    float phase_est = 0, sigma2 = 0.1f;
    float coarse_cfo = 0, fine_cfo = 0, total_cfo = 0;
    float rs_corr = 0;
    bool valid = false;
};

static ChanEst rs_estimate_local(const std::vector<C64>& syms, int rs_pos,
                                  float coarse_cfo, float samp_rate) {
    ChanEst c;
    if (rs_pos + RS_LEN > (int)syms.size()) return c;

    // Coarse CFO pre-compensation
    std::vector<C64> rs_seg(RS_LEN);
    for (int i = 0; i < RS_LEN; i++) {
        rs_seg[i] = syms[rs_pos + i];
        if (std::abs(coarse_cfo) > 1.0f) {
            float phase = -2.0f * (float)M_PI * coarse_cfo * (rs_pos + i) * g_ts_sym;
            rs_seg[i] *= C64(std::cos(phase), std::sin(phase));
        }
    }

    // Fine CFO via linear phase fitting
    float phases[RS_LEN];
    float prev = 0, accum = 0;
    for (int i = 0; i < RS_LEN; i++) {
        float raw = std::arg(rs_seg[i] * std::conj(REF_RS[i]));
        float diff = raw - prev;
        if (diff > (float)M_PI) accum -= 2.0f*(float)M_PI;
        else if (diff < -(float)M_PI) accum += 2.0f*(float)M_PI;
        phases[i] = raw + accum;
        prev = raw;
    }
    float nMean = (RS_LEN-1)/2.0f, pMean = 0;
    for (int i = 0; i < RS_LEN; i++) pMean += phases[i];
    pMean /= RS_LEN;
    float num = 0, den = 0;
    for (int i = 0; i < RS_LEN; i++) {
        float dn = i - nMean;
        num += dn * (phases[i] - pMean);
        den += dn * dn;
    }
    c.fine_cfo = (num / (den + 1e-30f)) / (2.0f*(float)M_PI*g_ts_sym);
    if (std::abs(c.fine_cfo) > 500.0f) return c;

    // Total CFO compensation + channel estimate
    float totalCfo = coarse_cfo + c.fine_cfo;
    C64 sum(0,0);
    for (int i = 0; i < RS_LEN; i++) {
        float phase = -2.0f*(float)M_PI*totalCfo*(rs_pos + i)*g_ts_sym;
        C64 corrected = syms[rs_pos + i] * C64(std::cos(phase), std::sin(phase));
        sum += corrected * std::conj(REF_RS[i]);
    }
    c.h = sum / (float)RS_LEN;
    if (std::abs(c.h) < 1e-6f) return c;

    c.phase_est = std::arg(c.h);
    c.total_cfo = totalCfo;
    c.coarse_cfo = coarse_cfo;

    // RS correlation quality
    C64 corrSum(0,0);
    for (int i = 0; i < RS_LEN; i++) {
        float phase = -2.0f*(float)M_PI*totalCfo*(rs_pos + i)*g_ts_sym;
        C64 corrected = syms[rs_pos + i] * C64(std::cos(phase), std::sin(phase));
        corrSum += corrected * std::conj(REF_RS[i]);
    }
    c.rs_corr = std::abs(corrSum);
    if (c.rs_corr < RS_LEN * 0.3f) return c;

    // Noise variance (Welch corrected)
    float noiseSum = 0;
    for (int i = 0; i < RS_LEN; i++) {
        float phase = -2.0f*(float)M_PI*totalCfo*(rs_pos + i)*g_ts_sym;
        C64 corrected = syms[rs_pos + i] * C64(std::cos(phase), std::sin(phase));
        C64 eq = corrected / c.h;
        C64 err = eq - REF_RS[i];
        noiseSum += std::norm(err);
    }
    c.sigma2 = noiseSum / (RS_LEN - 1);
    if (c.sigma2 < 1e-30f) c.sigma2 = 1e-30f;
    c.valid = true;
    return c;
}

// ---- BPSK hard decision ----
static std::vector<int> bpsk_demod_local(const std::vector<C64>& syms,
                                          int data_start, int data_len,
                                          C64 h, float total_cfo) {
    std::vector<int> bits(data_len, 0);
    if (data_start + data_len > (int)syms.size()) return bits;
    for (int i = 0; i < data_len; i++) {
        float phase = -2.0f*(float)M_PI*total_cfo*(data_start + i)*g_ts_sym;
        C64 corrected = syms[data_start + i] * C64(std::cos(phase), std::sin(phase));
        if (std::abs(h) > 1e-30f) corrected /= h;
        bits[i] = (corrected.real() < 0.0f) ? 1 : 0;
    }
    return bits;
}

// ---- B2I ----
static uint16_t b2i(const std::vector<int>& b, int len) {
    uint16_t v = 0;
    for (int i = 0; i < len; i++) v = (uint16_t)((v << 1) | (b[i] & 1));
    return v;
}

// ---- Header verify ----
static bool verify_header_local(const std::vector<int>& hdr) {
    if ((int)hdr.size() < HEADER_LEN) return false;
    uint8_t idBytes[2] = {};
    for (int i = 0; i < 16; i++)
        idBytes[i/8] = (uint8_t)((idBytes[i/8] << 1) | (hdr[i] & 1));
    uint16_t expected = b2i(std::vector<int>(hdr.begin()+16, hdr.begin()+32), 16);
    return crc16(idBytes, 2) == expected;
}

// ---- Payload CRC verify ----
static bool verify_payload_local(const std::vector<int>& payload,
                                  const std::vector<int>& crc_bits) {
    uint8_t payloadBytes[PAYLOAD_LEN/8] = {};
    for (int i = 0; i < PAYLOAD_LEN; i++)
        payloadBytes[i/8] = (uint8_t)((payloadBytes[i/8] << 1) | (payload[i] & 1));
    uint16_t expected = b2i(crc_bits, CRC_LEN);
    return crc16(payloadBytes, PAYLOAD_LEN/8) == expected;
}

// ===================================================================
// Processing loop (main thread)
// ===================================================================
static void process_loop(SharedRing& ring, std::atomic<bool>& running, float samp_rate) {
    constexpr int BUF_CAP = 1'000'000;
    std::vector<C64> buf(BUF_CAP);
    int buf_len = 0;
    size_t rd_count = 0;

    while (running.load(std::memory_order_relaxed)) {
        // Check for new data
        size_t wc = ring.wr_count.load(std::memory_order_acquire);
        if (wc <= rd_count) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        size_t avail = wc - rd_count;
        if (avail > (size_t)RING_CAP) {
            rd_count = wc - RING_CAP;
            avail = RING_CAP;
        }

        // Read chunk into temp buffer
        constexpr int CHUNK = 4096;
        int take = (int)(std::min)(avail, (size_t)CHUNK);
        std::vector<C64> chunk(take);
        ring.copy_to_vec(chunk, rd_count, take);
        rd_count += take;

        // Append to processing buffer
        int n = (int)chunk.size();
        if (n > BUF_CAP - buf_len) {
            int discard = buf_len - BUF_CAP / 2;
            if (discard > 0) {
                std::memmove(buf.data(), buf.data() + discard, (buf_len - discard) * sizeof(C64));
                buf_len -= discard;
            }
        }
        int space = std::min(n, BUF_CAP - buf_len);
        std::memcpy(buf.data() + buf_len, chunk.data(), space * sizeof(C64));
        buf_len += space;

        // Detect and process frames
        while (buf_len >= MIN_WIN_SAMPLES && running.load(std::memory_order_relaxed)) {
            int ws = (std::max)(0, buf_len - 5000);
            std::vector<C64> r(buf.begin() + ws, buf.begin() + buf_len);
            auto stf = stf_delay_correlate(r);
            if (stf.metric.empty()) break;

            auto clustered = stf_cluster_wrap(stf, r, samp_rate);

            if (clustered.peaks.empty()) {
                buf_len -= ws;
                if (buf_len > 0) std::memmove(buf.data(), buf.data() + ws, buf_len * sizeof(C64));
                break;
            }

            bool found = false;
            int maxCand = (std::min)(8, (int)clustered.peaks.size());
            for (int ci = 0; ci < maxCand && !found; ci++) {
                int d = clustered.peaks[ci];
                float coarse_cfo = clustered.cfos[ci];
                int coarse = ws + d;

                // Extract window
                int EXTRACT_EXTRA = 200;
                int es = (std::max)(0, coarse - EXTRACT_EXTRA);
                int ee = (std::min)(buf_len, coarse + EXTRACT_EXTRA + FRAME_RRC_SAMPLES + EXTRACT_EXTRA);
                if (ee - es < PSS_LEN * SPS) continue;

                std::vector<C64> chunkR(buf.begin() + es, buf.begin() + ee);
                auto syms = rrc_match_local(chunkR);
                if ((int)syms.size() < PSS_LEN + RS_LEN) continue;

                // PSS
                auto pssRes = pss_correlate(syms, REF_PSS);
                if (pssRes.peak_to_mean < g_pss_ptm_thr || pssRes.peak_to_second < g_pss_pts_thr)
                    continue;

                int fs = pssRes.peak_idx - STF_LEN;
                if (fs < 0) continue;

                int rp = fs + STF_LEN + PSS_LEN;
                if (rp + RS_LEN + HEADER_LEN + PAYLOAD_LEN + CRC_LEN > (int)syms.size())
                    continue;

                // RS estimate
                auto chan = rs_estimate_local(syms, rp, coarse_cfo, samp_rate);
                if (!chan.valid) continue;
                float sigma2 = (std::min)(chan.sigma2, 0.5f);

                // Demod Header
                int hdrStart = rp + RS_LEN;
                auto hdrBits = bpsk_demod_local(syms, hdrStart, HEADER_LEN, chan.h, chan.total_cfo);
                bool hdrOk = verify_header_local(hdrBits);

                // Demod Payload + CRC
                int payStart = hdrStart + HEADER_LEN;
                auto payBits = bpsk_demod_local(syms, payStart, PAYLOAD_LEN + CRC_LEN, chan.h, chan.total_cfo);
                std::vector<int> payloadBits(payBits.begin(), payBits.begin() + PAYLOAD_LEN);
                std::vector<int> crcBits(payBits.begin() + PAYLOAD_LEN, payBits.begin() + PAYLOAD_LEN + CRC_LEN);
                bool payCrcOk = verify_payload_local(payloadBits, crcBits);

                // Stats
                g_stats.total++;
                if (hdrOk) g_stats.hdr_ok++;
                if (payCrcOk) g_stats.crc_ok++;

                float hmag = std::abs(chan.h);
                float snrDb = 10.0f * std::log10(std::max(hmag * hmag / sigma2, 1e-30f));

                if (g_stats.total <= 5 || g_stats.total % 100 == 0) {
                    print_frame(g_stats.total, snrDb, hdrOk, payCrcOk);
                }

                // Consume window
                int consumeEnd = es + fs * SPS + FRAME_RRC_SAMPLES + 50;
                if (consumeEnd > buf_len) consumeEnd = buf_len;
                if (consumeEnd < buf_len) {
                    int remain = buf_len - consumeEnd;
                    std::memmove(buf.data(), buf.data() + consumeEnd, remain * sizeof(C64));
                    buf_len = remain;
                } else {
                    buf_len = 0;
                }
                found = true;
            }

            if (!found) {
                g_stats.false_alarms += maxCand;
                buf_len -= ws;
                if (buf_len > 0) std::memmove(buf.data(), buf.data() + ws, buf_len * sizeof(C64));
                break;
            }
        }
    }
}

// ===================================================================
// Threads
// ===================================================================

static void rx_thread_func(uhd::rx_streamer::sptr rx_stream,
                           SharedRing& ring,
                           std::atomic<bool>& running,
                           std::atomic<int>& overflow_count) {
    uhd::rx_metadata_t md;
    std::vector<C64> buf(4096);
    size_t w = 0;

    while (running.load(std::memory_order_relaxed)) {
        size_t ns = rx_stream->recv(buf.data(), buf.size(), md, 0.2);
        if (ns == 0) continue;
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
            overflow_count.fetch_add(1);
            continue;
        }
        ring.write_samples(buf.data(), ns, w);
    }

    uhd::stream_cmd_t stop_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
    rx_stream->issue_stream_cmd(stop_cmd);
}

static void tx_thread_func(uhd::tx_streamer::sptr tx_stream,
                           int num_frames, int gap_len,
                           std::atomic<bool>& running) {
    uhd::tx_metadata_t md;
    md.start_of_burst = true;
    md.end_of_burst = false;

    // Use fixed seed for reproducibility
    srand(42);

    for (int f = 0; f < num_frames && running.load(std::memory_order_relaxed); f++) {
        std::vector<int> bits(PAYLOAD_LEN);
        for (int i = 0; i < PAYLOAD_LEN; i++) bits[i] = rand() & 1;

        auto frame = build_frame(bits, (uint16_t)f);
        auto txSig = rrc_filter(frame, RRC);

        tx_stream->send(txSig.data(), txSig.size(), md);
        md.start_of_burst = false;

        if (gap_len > 0) {
            uhd::tx_metadata_t gap_md;
            gap_md.start_of_burst = false;
            gap_md.end_of_burst = false;
            std::vector<C64> gap(gap_len, C64(0,0));
            tx_stream->send(gap.data(), gap.size(), gap_md);
        }
    }

    uhd::tx_metadata_t eob;
    eob.end_of_burst = true;
    C64 z(0,0);
    tx_stream->send(&z, 1, eob);
}

// ===================================================================
// Main
// ===================================================================
int main(int argc, char* argv[]) {
    init_crc16_table();

    double freq = 915e6, tx_gain = 65.0, rx_gain = 64.0;
    double rate = 1e6, frame_gap_ms = 5.0;
    int num_frames = 1000;
    std::string args_str;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--freq" && i+1<argc) freq = atof(argv[++i]);
        else if (a == "--gain-tx" && i+1<argc) tx_gain = atof(argv[++i]);
        else if (a == "--gain-rx" && i+1<argc) rx_gain = atof(argv[++i]);
        else if (a == "--rate" && i+1<argc) rate = atof(argv[++i]);
        else if (a == "--num-frames" && i+1<argc) num_frames = atoi(argv[++i]);
        else if (a == "--frame-gap-ms" && i+1<argc) frame_gap_ms = atof(argv[++i]);
        else if (a == "--args" && i+1<argc) args_str = argv[++i];
        else if (a == "-h" || a == "--help") {
            printf("Usage: loopback_msvc [--freq 915e6] [--gain-tx 65] [--gain-rx 64] [--num-frames 1000]\n");
            return 0;
        }
    }

    g_ts = 1.0f / (float)rate;
    g_ts_sym = (float)SPS / (float)rate;

    printf("[loopback] rate=%.0f freq=%.3fMHz tx_gain=%.0f rx_gain=%.0f frames=%d\n",
           rate, freq/1e6, tx_gain, rx_gain, num_frames);

    // --- USRP ---
    auto usrp = uhd::usrp::multi_usrp::make(args_str);

    usrp->set_tx_freq(uhd::tune_request_t(freq));
    usrp->set_tx_gain(tx_gain);
    usrp->set_tx_rate(rate);
    usrp->set_tx_bandwidth(rate);
    usrp->set_tx_antenna("TX/RX");

    usrp->set_rx_freq(uhd::tune_request_t(freq));
    usrp->set_rx_gain(rx_gain);
    usrp->set_rx_rate(rate);
    usrp->set_rx_bandwidth(rate);
    usrp->set_rx_antenna("RX2");

    usrp->set_clock_source("internal");
    usrp->set_time_source("internal");
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    usrp->set_time_now(uhd::time_spec_t((double)ns / 1e9));

    // --- Streams ---
    uhd::stream_args_t tx_args("fc32", "sc16");
    tx_args.channels = {0};
    auto tx_stream = usrp->get_tx_stream(tx_args);

    uhd::stream_args_t rx_args("fc32", "sc16");
    rx_args.channels = {0};
    auto rx_stream = usrp->get_rx_stream(rx_args);
    rx_stream->issue_stream_cmd(uhd::stream_cmd_t(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS));

    // --- Ring buffer + threads ---
    SharedRing ring;
    std::atomic<bool> running(true);
    std::atomic<int> overflow_count(0);

    std::thread rx_th(rx_thread_func, rx_stream, std::ref(ring),
                      std::ref(running), std::ref(overflow_count));
    std::this_thread::sleep_for(std::chrono::seconds(1));

    int gap_len = std::max(16, (int)(frame_gap_ms * rate / 1000.0));
    std::thread tx_th(tx_thread_func, tx_stream, num_frames, gap_len, std::ref(running));
    std::thread proc_th(process_loop, std::ref(ring), std::ref(running), (float)rate);

    printf("[loopback] TX started, %d frames  gap=%.1fms  "
           "PSS thr=(ptm=%.1f,pts=%.1f)\n",
           num_frames, frame_gap_ms, g_pss_ptm_thr, g_pss_pts_thr);

    // --- Wait for TX, then drain buffer ---
    tx_th.join();
    printf("[loopback] TX done, draining...\n");
    std::this_thread::sleep_for(std::chrono::seconds(3));
    running.store(false);

    rx_th.join();
    proc_th.join();

    // --- Report ---
    int total = g_stats.total;
    fprintf(stderr, "\n--- results ---\n");
    fprintf(stderr, "  frames=%d  CRC=%d/%d (%.1f%%)  HDR=%d  false_alarms=%d\n",
            total, g_stats.crc_ok, total,
            total > 0 ? 100.0 * g_stats.crc_ok / total : 0.0,
            g_stats.hdr_ok, g_stats.false_alarms);

    return 0;
}
