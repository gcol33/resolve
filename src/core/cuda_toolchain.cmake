# CUDA toolchain file for RESOLVE
# This sets up CUDA compiler without identification tests

# CUDA configuration
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe")
# CMAKE_CUDA_HOST_COMPILER intentionally not set - nvcc will find cl.exe from PATH
# Setting it causes 8.3 vs long path mismatch with --use-local-env
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler --use-local-env" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler --use-local-env")
set(CMAKE_CUDA_ARCHITECTURES 89)

# Ensure nvcc can find cicc.exe
set(ENV{PATH} "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/nvvm/bin;$ENV{PATH}")

# MSVC and Windows SDK paths (so cmake works outside VS Developer Command Prompt)
set(MSVC_ROOT "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207")
set(WINSDK_ROOT "C:/Program Files (x86)/Windows Kits/10")
set(WINSDK_VERSION "10.0.26100.0")

set(ENV{INCLUDE} "${MSVC_ROOT}/include;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/ucrt;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/shared;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/um;${WINSDK_ROOT}/Include/${WINSDK_VERSION}/winrt")
set(ENV{LIB} "${MSVC_ROOT}/lib/x64;${WINSDK_ROOT}/Lib/${WINSDK_VERSION}/ucrt/x64;${WINSDK_ROOT}/Lib/${WINSDK_VERSION}/um/x64")
set(ENV{PATH} "${MSVC_ROOT}/bin/Hostx64/x64;${WINSDK_ROOT}/bin/${WINSDK_VERSION}/x64;$ENV{PATH}")

# Point CMake to the resource compiler
set(CMAKE_RC_COMPILER "${WINSDK_ROOT}/bin/${WINSDK_VERSION}/x64/rc.exe")
set(CMAKE_MT "${WINSDK_ROOT}/bin/${WINSDK_VERSION}/x64/mt.exe")
