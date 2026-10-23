# CMake toolchain file for the Windows CUDA build run outside a VS Developer
# Command Prompt.
#
# The calling script (build_cuda/do_build.bat) owns every machine-specific
# location: it sets PATH, INCLUDE, LIB and PATHEXT with native backslashes before
# CMake runs (forward slashes in PATH break nvcc's cudafe++ subprocess
# resolution, so this file never sets PATH), names nvcc through CUDACXX, and
# passes the target GPU as -DTORCH_CUDA_ARCH_LIST. CMake finds rc.exe and mt.exe
# on that PATH.

# CUDA configuration (only when USE_CUDA is ON)
if(NOT DEFINED USE_CUDA OR USE_CUDA)
    # CMAKE_CUDA_HOST_COMPILER intentionally not set - nvcc will find cl.exe from PATH
    # Setting it causes 8.3 vs long path mismatch with --use-local-env
    set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler --use-local-env" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler --use-local-env")
endif()
