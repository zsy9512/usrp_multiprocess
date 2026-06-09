@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %ERRORLEVEL% NEQ 0 (echo vcvars64.bat failed & exit /b 1)

set UHD_INC=C:\PROGRA~1\UHD\include
set UHD_LIB=C:\PROGRA~1\UHD\lib\uhd.lib
set BOOST_ROOT=E:\PhD_work\code\usrp_hardware\boost_1_66_0\boost_1_66_0
set BOOST_LIB=%BOOST_ROOT%\stage64s\lib

set CFLAGS=/EHsc /O2 /fp:fast /std:c++17 /I "%UHD_INC%" /I "%BOOST_ROOT%" /DBOOST_ALL_NO_LIB /DBOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE

set LFLAGS=%UHD_LIB% ws2_32.lib ^
  "%BOOST_LIB%\libboost_date_time-vc1444-mt-s-x64-1_66.lib" ^
  "%BOOST_LIB%\libboost_system-vc1444-mt-s-x64-1_66.lib" ^
  "%BOOST_LIB%\libboost_thread-vc1444-mt-s-x64-1_66.lib" ^
  "%BOOST_LIB%\libboost_program_options-vc1444-mt-s-x64-1_66.lib" ^
  "%BOOST_LIB%\libboost_chrono-vc1444-mt-s-x64-1_66.lib"

echo === Compiling loopback_msvc.exe ===
cl %CFLAGS% loopback.cpp /link %LFLAGS% /out:loopback_msvc.exe
if %ERRORLEVEL% NEQ 0 (echo loopback FAILED & exit /b 1)

dir loopback_msvc.exe
echo === Done ===
