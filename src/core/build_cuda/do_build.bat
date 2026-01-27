@echo off
setlocal enabledelayedexpansion

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to initialize VS2022 environment
    exit /b 1
)
echo VS2022 environment loaded

set "SRC_DIR=C:\Users\Gilles Colling\Documents\dev\RESOLVE\src\core"
set "BUILD_DIR=%SRC_DIR%\build_cuda"
set "TORCH_DIR=C:\Users\Gilles Colling\.pyenv\pyenv-win\versions\3.13.5\Lib\site-packages\torch\share\cmake\Torch"
set "PYTHON_EXE=C:\Users\Gilles Colling\.pyenv\pyenv-win\versions\3.13.5\python.exe"
set "NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

set "TMP=C:\tmp"
set "TEMP=C:\tmp"
if not exist C:\tmp mkdir C:\tmp

if exist "%BUILD_DIR%\CMakeCache.txt" del /q "%BUILD_DIR%\CMakeCache.txt"
if exist "%BUILD_DIR%\CMakeFiles" rd /s /q "%BUILD_DIR%\CMakeFiles"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo.
echo === Configuring ===
cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" ^
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
cmake --build "%BUILD_DIR%" --config Release

if errorlevel 1 (
    echo.
    echo === BUILD FAILED ===
    exit /b 1
)

echo.
echo === BUILD SUCCEEDED ===
