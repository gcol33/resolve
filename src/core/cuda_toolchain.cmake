# CUDA toolchain file for RESOLVE
# Sets up compilers for building outside VS Developer Command Prompt.
#
# Environment variables (PATH, INCLUDE, LIB, PATHEXT) are set by
# do_build.bat before cmake runs, to ensure native backslashes and
# correct values regardless of the parent process (PowerShell, bash, etc.).

# MSVC and Windows SDK paths
set(MSVC_ROOT "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.44.35207")
set(WINSDK_ROOT "C:/Program Files (x86)/Windows Kits/10")
set(WINSDK_VERSION "10.0.26100.0")

# Point CMake to the resource compiler and manifest tool
set(CMAKE_RC_COMPILER "${WINSDK_ROOT}/bin/${WINSDK_VERSION}/x64/rc.exe")
set(CMAKE_MT "${WINSDK_ROOT}/bin/${WINSDK_VERSION}/x64/mt.exe")

# CUDA configuration (only when USE_CUDA is ON)
if(NOT DEFINED USE_CUDA OR USE_CUDA)
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe")
    # CMAKE_CUDA_HOST_COMPILER intentionally not set - nvcc will find cl.exe from PATH
    # Setting it causes 8.3 vs long path mismatch with --use-local-env
    set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler --use-local-env" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler --use-local-env")
    # Use TORCH_CUDA_ARCH_LIST instead of CMAKE_CUDA_ARCHITECTURES: torch's
    # Caffe2 cmake ignores the latter and warns. "12.0" = Blackwell (RTX 5080).
    # To build for a different GPU (e.g. 8.9 = Ada / RTX 4090, 9.0 = Hopper),
    # change this list or pass -DTORCH_CUDA_ARCH_LIST="..." to cmake.
    set(TORCH_CUDA_ARCH_LIST "12.0" CACHE STRING "" FORCE)
endif()
