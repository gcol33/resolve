# Shared custom-CUDA-kernel setup, included by BOTH the main engine build
# (src/core/CMakeLists.txt) and the standalone Python extension build
# (src/core/python/CMakeLists.txt). Single source of truth so the wheel and the
# engine build agree on kernel compilation and, critically, on whether
# RESOLVE_HAS_CUDA is defined (without it, gpu.cpp compiles out and
# set_vram_fraction silently no-ops -- the standalone USE_CUDA=ON build used to
# hit exactly this).
#
# Expects, before inclusion:
#   RESOLVE_CORE_DIR  - absolute path to src/core (holds cuda/ and include/)
#   resolve_core      - an already-defined library target
#   resolve_warnings  - the INTERFACE warning target (resolve_warnings.cmake)
#   USE_CUDA          - whether CUDA is requested
#   TORCH_LIBRARIES   - from find_package(Torch)
# On success it links the custom kernels into resolve_core and defines
# RESOLVE_HAS_CUDA on it.

if(USE_CUDA AND CMAKE_CUDA_COMPILER AND NOT SKIP_CUDA_KERNELS)
    # Check nvcc actually runs before committing to the CUDA path.
    execute_process(
        COMMAND ${CMAKE_CUDA_COMPILER} --version
        RESULT_VARIABLE NVCC_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NVCC_RESULT EQUAL 0)
        enable_language(CUDA)
        # CUDAToolkit gives us CUDA::cudart (cuda_runtime_api.h include path for
        # C++ files that pull in libtorch's ATen/cuda/* headers).
        find_package(CUDAToolkit REQUIRED)

        # SASS targets for the custom (non-torch) hash kernels. Plain CUDA lib,
        # so it uses CUDA_ARCHITECTURES directly. The old hardcoded "89" (Ada)
        # produced no loadable image on the dev RTX 5080 (Blackwell, sm_120):
        # any custom-kernel launch failed with cudaErrorNoKernelImageForDevice
        # and silently fell back to cuBLAS. Default covers Ada + Blackwell real
        # SASS plus a compute_120 PTX entry for forward compat. Override with
        # -DRESOLVE_CUDA_ARCHITECTURES=... to target a single card.
        set(RESOLVE_CUDA_ARCHITECTURES "89-real;120-real;120-virtual" CACHE STRING
            "CUDA architectures for the custom hash kernels (resolve_cuda_kernels)")

        # Pure CUDA kernel library - NO PyTorch headers.
        add_library(resolve_cuda_kernels STATIC ${RESOLVE_CORE_DIR}/cuda/kernels.cu)
        target_include_directories(resolve_cuda_kernels PUBLIC
            $<BUILD_INTERFACE:${RESOLVE_CORE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        set_target_properties(resolve_cuda_kernels PROPERTIES
            CUDA_STANDARD 17
            CUDA_ARCHITECTURES "${RESOLVE_CUDA_ARCHITECTURES}")

        # C++ wrapper that includes PyTorch headers. resolve_warnings gates its
        # flags on COMPILE_LANGUAGE:CXX, so linking it into the pure-CUDA
        # kernel target too is a no-op today and covers any .cpp added later.
        target_link_libraries(resolve_cuda_kernels PRIVATE resolve_warnings)

        add_library(resolve_cuda_wrapper STATIC ${RESOLVE_CORE_DIR}/cuda/feature_hash.cpp)
        target_include_directories(resolve_cuda_wrapper PUBLIC
            $<BUILD_INTERFACE:${RESOLVE_CORE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        target_link_libraries(resolve_cuda_wrapper PUBLIC
            ${TORCH_LIBRARIES}
            resolve_cuda_kernels
            CUDA::cudart)
        target_link_libraries(resolve_cuda_wrapper PRIVATE resolve_warnings)

        target_link_libraries(resolve_core PUBLIC resolve_cuda_wrapper)
        target_compile_definitions(resolve_core PUBLIC RESOLVE_HAS_CUDA)
        message(STATUS "CUDA custom kernels enabled")
    else()
        message(STATUS "CUDA kernels disabled (nvcc not working)")
    endif()
else()
    message(STATUS "CUDA kernels disabled (USE_CUDA=${USE_CUDA}, SKIP_CUDA_KERNELS=${SKIP_CUDA_KERNELS})")
endif()
