/*
 * uhd_tx_main.cpp — B210 UHD 发送端 (C++ API, MSVC 编译)
 *
 * 使用 UHD C++ API. 编译:
 *   call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
 *   cl /EHsc /O2 /std:c++17 /I "C:\Program Files\UHD\include"
 *      /I "E:\PhD_work\code\usrp_hardware\boost_1_66_0\boost_1_66_0"
 *      uhd_tx_main.cpp /link "C:\Program Files\UHD\lib\uhd.lib" ws2_32.lib
 *      /out:uhd_tx_msvc.exe
 *
 * MinGW 编译 (需 UHD C API):
 *   备选 C API 版本见 uhd_tx_capi.cpp
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

    double freq = 915e6, gain = 60.0, rate = 1e6;
    double frame_gap_ms = 2.0, settle_s = 1.0;
    int num_frames = 0;
    bool timed_start = false;
    std::string sync_mode = "host", args_str;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--sync-mode" && i+1<argc) sync_mode = argv[++i];
        else if (a == "--freq" && i+1<argc) freq = atof(argv[++i]);
        else if (a == "--gain" && i+1<argc) gain = atof(argv[++i]);
        else if (a == "--rate" && i+1<argc) rate = atof(argv[++i]);
        else if (a == "--frame-gap-ms" && i+1<argc) frame_gap_ms = atof(argv[++i]);
        else if (a == "--settle" && i+1<argc) settle_s = atof(argv[++i]);
        else if (a == "--num-frames" && i+1<argc) num_frames = atoi(argv[++i]);
        else if (a == "--timed-start") timed_start = true;
        else if (a == "--args" && i+1<argc) args_str = argv[++i];
        else if (a == "-h" || a == "--help") {
            printf("Usage: uhd_tx_msvc [--sync-mode host|external_ref] ...\n");
            return 0;
        }
    }

    g_ts = 1.0f / (float)rate;
    g_ts_sym = (float)SPS / (float)rate;
    printf("[uhd_tx] rate=%.0f freq=%.3fMHz gain=%.0fdB gap=%.1fms mode=%s\n",
           rate, freq/1e6, gain, frame_gap_ms, sync_mode.c_str());

    auto usrp = uhd::usrp::multi_usrp::make(args_str);
    usrp->set_tx_freq(uhd::tune_request_t(freq));
    usrp->set_tx_gain(gain);
    usrp->set_tx_rate(rate);
    usrp->set_tx_bandwidth(rate);
    usrp->set_tx_antenna("TX/RX");

    uhd::tx_metadata_t md;
    md.start_of_burst = true;
    md.end_of_burst = false;
    md.has_time_spec = timed_start;

    // --- Clock ---
    if (sync_mode == "host") {
        usrp->set_clock_source("internal");
        usrp->set_time_source("internal");
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        usrp->set_time_now(uhd::time_spec_t((double)ns / 1e9));
        printf("[uhd_tx] sync=host\n");
    } else if (sync_mode == "external_ref") {
        usrp->set_clock_source("external");
        usrp->set_time_source("internal");
        std::this_thread::sleep_for(std::chrono::duration<double>(settle_s));
        bool locked = usrp->get_mboard_sensor("ref_locked").to_bool();
        printf("[uhd_tx] sync=external_ref locked=%d\n", locked);
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        usrp->set_time_now(uhd::time_spec_t((double)ns / 1e9));
    }

    uhd::stream_args_t stream_args("fc32", "sc16");
    auto tx_stream = usrp->get_tx_stream(stream_args);

    int gap_len = std::max(16, (int)(frame_gap_ms * rate / 1000.0));
    std::vector<C64> gap(gap_len, C64(0,0));

    auto rrc = design_rrc();
    int frame_count = 0;
    printf("[uhd_tx] transmitting...\n");

    while ((num_frames == 0 || frame_count < num_frames) && g_running) {
        std::vector<int> bits(PAYLOAD_LEN);
        for (int i = 0; i < PAYLOAD_LEN; i++) bits[i] = rand() & 1;
        auto syms = build_frame(bits, (uint16_t)frame_count);
        auto txSig = rrc_filter(syms, rrc);

        tx_stream->send(txSig.data(), txSig.size(), md);
        md.start_of_burst = false;

        uhd::tx_metadata_t gap_md;
        gap_md.start_of_burst = false;
        gap_md.end_of_burst = false;
        tx_stream->send(gap.data(), gap.size(), gap_md);

        frame_count++;
        if (frame_count % 50 == 0) printf("[uhd_tx] %d frames\n", frame_count);
    }

    uhd::tx_metadata_t eob_md;
    eob_md.end_of_burst = true;
    C64 z(0,0);
    tx_stream->send(&z, 1, eob_md);

    printf("[uhd_tx] done: %d frames\n", frame_count);
    return 0;
}
