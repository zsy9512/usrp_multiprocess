@echo off
REM build_uhd_msvc.bat — 编译 UHD TX/RX (MSVC)
REM 
REM 前提: 1) VS2022 BuildTools 已安装
REM       2) Boost 1.66 已解压到 E:\PhD_work\code\usrp_hardware\boost_1_66_0
REM       3) 已执行: cd boost_1_66_0 && bootstrap.bat
REM                  && b2 toolset=msvc-14.44 address-model=64 link=static runtime-link=static
REM                     --with-system --with-date_time --with-thread --with-program_options stage
REM
REM 用法: 在 cpp 目录下运行此脚本

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %ERRORLEVEL% NEQ 0 (echo vcvars64.bat failed && exit /b 1)

set UHD_INC=C:\PROGRA~1\UHD\include
set UHD_LIB=C:\PROGRA~1\UHD\lib\uhd.lib
set BOOST_ROOT=E:\PhD_work\code\usrp_hardware\boost_1_66_0\boost_1_66_0
set BOOST_LIB=%BOOST_ROOT%\stage64s\lib

set CFLAGS=/EHsc /O2 /std:c++17 /I "%UHD_INC%" /I "%BOOST_ROOT%" ^
  /DBOOST_ALL_NO_LIB /DBOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE

set LFLAGS=%UHD_LIB% ws2_32.lib ^
  "%BOOST_LIB%\libboost_date_time-vc1444-mt-s-x64-1_66.lib" ^
  "%BOOST_LIB%\libboost_system-vc1444-mt-s-x64-1_66.lib" ^
  "%BOOST_LIB%\libboost_thread-vc1444-mt-s-x64-1_66.lib" ^
  "%BOOST_LIB%\libboost_program_options-vc1444-mt-s-x64-1_66.lib" ^
  "%BOOST_LIB%\libboost_chrono-vc1444-mt-s-x64-1_66.lib"

echo === Compiling uhd_tx_msvc.exe ===
cl %CFLAGS% uhd_tx_main.cpp /link %LFLAGS% /out:uhd_tx_msvc.exe
if %ERRORLEVEL% NEQ 0 (echo uhd_tx FAILED && exit /b 1)

echo === Compiling uhd_rx_msvc.exe ===
cl %CFLAGS% uhd_rx_main.cpp /link %LFLAGS% /out:uhd_rx_msvc.exe
if %ERRORLEVEL% NEQ 0 (echo uhd_rx FAILED && exit /b 1)

echo === Compiling loopback_msvc.exe ===
cl %CFLAGS% loopback.cpp /link %LFLAGS% /out:loopback_msvc.exe
if %ERRORLEVEL% NEQ 0 (echo loopback FAILED && exit /b 1)

echo === Done ===
dir uhd_tx_msvc.exe uhd_rx_msvc.exe loopback_msvc.exe
