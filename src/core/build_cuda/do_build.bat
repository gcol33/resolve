@echo off
setlocal enabledelayedexpansion

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to initialize VS 18 Community environment
    exit /b 1
)
echo VS 18 Community environment loaded

set "SRC_DIR=C:\Users\Gilles Colling\Documents\dev\RESOLVE\src\core"
set "BUILD_DIR=%SRC_DIR%\build_cuda"
set "TORCH_DIR=C:\Users\Gilles Colling\.pyenv\pyenv-win\versions\3.12.10\Lib\site-packages\torch\share\cmake\Torch"
set "PYTHON_EXE=C:\Users\Gilles Colling\.pyenv\pyenv-win\versions\3.12.10\python.exe"
set "NINJA_BIN=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "NINJA=%NINJA_BIN%\ninja.exe"
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"

set "MSVC_BIN=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
set "SDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
set "CUDA_NVVM=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\nvvm\bin"

rem Set minimal PATH with native backslashes. The cmake toolchain does NOT
rem set PATH because forward slashes break nvcc's cudafe++ subprocess resolution.
set "PATH=%MSVC_BIN%;%SDK_BIN%;%CUDA_BIN%;%CUDA_NVVM%;C:\Program Files\CMake\bin;%NINJA_BIN%;C:\Windows\System32;C:\Windows"

rem Restore PATHEXT (PowerShell may clobber it to just .CPL, breaking exe resolution)
set "PATHEXT=.COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC"

rem Explicitly set INCLUDE and LIB (vcvars64 sets these, but the inherited
rem environment from PowerShell/sandbox may corrupt them)
set "MSVC_INC=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\include"
set "WINSDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0"
set "WINSDK_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0"
set "MSVC_LIB=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\lib\x64"
set "INCLUDE=%MSVC_INC%;%WINSDK_INC%\ucrt;%WINSDK_INC%\shared;%WINSDK_INC%\um;%WINSDK_INC%\winrt"
set "LIB=%MSVC_LIB%;%WINSDK_LIB%\ucrt\x64;%WINSDK_LIB%\um\x64"

set "TMP=C:\tmp"
set "TEMP=C:\tmp"

if exist "%BUILD_DIR%\CMakeCache.txt" del /q "%BUILD_DIR%\CMakeCache.txt"
if exist "%BUILD_DIR%\CMakeFiles" rd /s /q "%BUILD_DIR%\CMakeFiles"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo.
echo === Configuring ===
"%CMAKE%" -S "%SRC_DIR%" -B "%BUILD_DIR%" ^
    -G Ninja ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DCMAKE_TOOLCHAIN_FILE="%SRC_DIR%\cuda_toolchain.cmake" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_PYTHON=ON ^
    -DBUILD_CLI=ON ^
    -DBUILD_TESTS=ON ^
    -DUSE_CUDA=ON ^
    -DTorch_DIR="%TORCH_DIR%" ^
    -DPython_EXECUTABLE="%PYTHON_EXE%"

if errorlevel 1 (
    echo.
    echo === CONFIGURE FAILED ===
    exit /b 1
)

echo.
echo === Building ===
"%CMAKE%" --build "%BUILD_DIR%" --config Release

if errorlevel 1 (
    echo.
    echo === BUILD FAILED ===
    exit /b 1
)

echo.
echo === BUILD SUCCEEDED ===
