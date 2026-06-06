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

rx: rx_main.cpp
	$(CXX) $(CXXFLAGS) rx_main.cpp -o rx$(EXE_EXT) $(LDFLAGS)

tx: tx_main.cpp
	$(CXX) $(CXXFLAGS) tx_main.cpp -o tx$(EXE_EXT) $(LDFLAGS)

clean:
	rm -f rx$(EXE_EXT) tx$(EXE_EXT)

.PHONY: all clean
