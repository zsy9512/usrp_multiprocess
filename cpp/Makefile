# Makefile — BPSK PHY C++ standalone TX/RX
#
# Linux:   make
# Windows (MSYS2/MinGW): make
# Windows (MSVC): cl /EHsc /O2 /std:c++17 rx_main.cpp /Fe:rx.exe

CXX      ?= g++
CXXFLAGS  = -std=c++17 -O3 -march=native -Wall -Wextra
LDFLAGS   = -lm

ifeq ($(OS),Windows_NT)
    EXE_EXT = .exe
else
    EXE_EXT =
endif

all: rx tx

rx: rx_main.cpp phy_dsp.h
	$(CXX) $(CXXFLAGS) rx_main.cpp -o rx$(EXE_EXT) $(LDFLAGS)

tx: tx_main.cpp phy_dsp.h
	$(CXX) $(CXXFLAGS) tx_main.cpp -o tx$(EXE_EXT) $(LDFLAGS)

# UHD 目标 (使用本地 libuhd.a + uhd.dll, 需 UHD 头文件)
UHD_INC  ?= /usr/local/include
uhd_tx: uhd_tx_main.cpp phy_dsp.h
	$(CXX) $(CXXFLAGS) -I$(UHD_INC) uhd_tx_main.cpp -o uhd_tx$(EXE_EXT) $(LDFLAGS) -L. -luhd

uhd_rx: uhd_rx_main.cpp phy_dsp.h
	$(CXX) $(CXXFLAGS) -I$(UHD_INC) uhd_rx_main.cpp -o uhd_rx$(EXE_EXT) $(LDFLAGS) -L. -luhd

clean:
	rm -f rx$(EXE_EXT) tx$(EXE_EXT) uhd_tx$(EXE_EXT) uhd_rx$(EXE_EXT)

.PHONY: all clean uhd_tx uhd_rx
