cmake_minimum_required(VERSION 3.18)

# Options (must be set before project() for CUDA)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_R "Build R bindings" OFF)
option(BUILD_CLI "Build CLI application" ON)
option(BUILD_TESTS "Build tests" ON)
option(USE_CUDA "Enable CUDA support" ON)

# Project declaration - CXX only initially
# CUDA is enabled via project() so it's identified before PyTorch's find_package calls enable_language(CUDA)
if(USE_CUDA)
    project(resolve VERSION 0.6.0 LANGUAGES CXX CUDA)
else()
    project(resolve VERSION 0.6.0 LANGUAGES CXX)
endif()

# Skip Caffe2's CUDA detection by setting this before find_package(Torch)
# We link against PyTorch's pre-built CUDA libraries, not compile CUDA ourselves
set(CAFFE2_USE_CUDNN OFF)
set(USE_CUDNN OFF)

# libtorch 2.13+ uses C++20 features (bit-field default initializers in
# c10/core/AutogradState.h). Older libtorch still compiles with C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Tell Torch/Caffe2 that we're using CUDA
set(CAFFE2_USE_CUDA ${USE_CUDA})

# Find libtorch
# Set Torch_DIR if not found automatically:
# cmake -DTorch_DIR=/path/to/libtorch/share/cmake/Torch ..
find_package(Torch REQUIRED)

# Core library sources
set(RESOLVE_CORE_SOURCES
    cpp_src/encoder_common.cpp
    cpp_src/encoder_topk.cpp
    cpp_src/encoder_pool.cpp
    cpp_src/task_head.cpp
    cpp_src/attention.cpp
    cpp_src/gnn.cpp
    cpp_src/tabm.cpp
    cpp_src/adapter.cpp
    cpp_src/model.cpp
    cpp_src/trainer.cpp
    cpp_src/predictor.cpp
    cpp_src/loss.cpp
    cpp_src/dataset.cpp
    cpp_src/species_encoding.cpp
    cpp_src/checkpoint.cpp
    cpp_src/categorical.cpp

    cpp_src/pretraining.cpp
    cpp_src/vae.cpp

    cpp_src/fuzzy_index.cpp
    cpp_src/fuzzy_search.cpp
    cpp_src/fuzzy_automaton.cpp

    cpp_src/gpu.cpp
)

set(RESOLVE_CORE_HEADERS
    include/resolve/resolve.hpp
    include/resolve/types.hpp
    include/resolve/encoder.hpp
    include/resolve/attention.hpp
    include/resolve/model.hpp
    include/resolve/trainer.hpp
    include/resolve/predictor.hpp
    include/resolve/loss.hpp
    include/resolve/csv_reader.hpp
    include/resolve/csv_utils.hpp
    include/resolve/role_mapping.hpp
    include/resolve/dataset.hpp
    include/resolve/species_encoding.hpp
    include/resolve/checkpoint.hpp
    include/resolve/categorical.hpp

    include/resolve/tabm.hpp
    include/resolve/pretraining.hpp
    include/resolve/vae.hpp

    include/resolve/fuzzy.hpp

    include/resolve/gpu.hpp
)

# CUDA sources - split for CUDA 13.x compatibility:
# - kernels.cu: Pure CUDA code, compiled by nvcc WITHOUT PyTorch headers
# - feature_hash.cpp: PyTorch interface, compiled by C++ compiler
set(RESOLVE_CUDA_KERNEL_SOURCES
    cuda/kernels.cu
)

set(RESOLVE_CUDA_WRAPPER_SOURCES
    cuda/feature_hash.cpp
)

set(RESOLVE_CUDA_HEADERS
    include/resolve/cuda/feature_hash.hpp
)

# Build core library (C++ only)
add_library(resolve_core STATIC ${RESOLVE_CORE_SOURCES} ${RESOLVE_CORE_HEADERS})

# CUDA custom kernels (optional - skip if nvcc environment is broken)
# The core library works fine without custom kernels - it uses PyTorch's CUDA ops
# To enable custom CUDA kernels, run from VS Developer Command Prompt with CUDA in PATH
if(USE_CUDA AND CMAKE_CUDA_COMPILER AND NOT SKIP_CUDA_KERNELS)
    # Check if we can actually use nvcc
    execute_process(
        COMMAND ${CMAKE_CUDA_COMPILER} --version
        RESULT_VARIABLE NVCC_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NVCC_RESULT EQUAL 0)
        enable_language(CUDA)
        # CUDAToolkit gives us CUDA::cudart (provides cuda_runtime_api.h include path
        # for C++ files that pull in libtorch's ATen/cuda/* headers).
        find_package(CUDAToolkit REQUIRED)

        # Pure CUDA kernel library - NO PyTorch headers
        add_library(resolve_cuda_kernels STATIC ${RESOLVE_CUDA_KERNEL_SOURCES})
        target_include_directories(resolve_cuda_kernels PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        set_target_properties(resolve_cuda_kernels PROPERTIES
            CUDA_STANDARD 17
            CUDA_ARCHITECTURES "89")

        # C++ wrapper that includes PyTorch headers
        add_library(resolve_cuda_wrapper STATIC ${RESOLVE_CUDA_WRAPPER_SOURCES})
        target_include_directories(resolve_cuda_wrapper PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        target_link_libraries(resolve_cuda_wrapper PUBLIC
            ${TORCH_LIBRARIES}
            resolve_cuda_kernels
            CUDA::cudart)

        target_link_libraries(resolve_core PUBLIC resolve_cuda_wrapper)
        target_compile_definitions(resolve_core PUBLIC RESOLVE_HAS_CUDA)
        message(STATUS "CUDA custom kernels enabled")
    else()
        message(STATUS "CUDA kernels disabled (nvcc not working)")
    endif()
else()
    message(STATUS "CUDA kernels disabled (USE_CUDA=${USE_CUDA}, SKIP_CUDA_KERNELS=${SKIP_CUDA_KERNELS})")
endif()

target_include_directories(resolve_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(resolve_core PUBLIC ${TORCH_LIBRARIES})

# OpenMP for fuzzy::FuzzyIndex::query_batch. Optional: serial fallback if absent.
#
# RESOLVE_USE_OPENMP=OFF drops the OpenMP runtime entirely. The R package's
# resolve_c build sets this: the package does not expose fuzzy::query_batch, and
# stacking a third OpenMP runtime (MSVC vcomp) on top of libtorch's libiomp5md
# and R's own libgomp inside a single mingw R process crashes on the first
# parallel op. Dropping vcomp removes that conflict; query_batch falls back to
# its serial path (unused from R anyway).
option(RESOLVE_USE_OPENMP "Link OpenMP into resolve_core (fuzzy::query_batch)" ON)
if(RESOLVE_USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(resolve_core PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP enabled for resolve_core")
    else()
        message(STATUS "OpenMP not found; fuzzy::query_batch will run serially")
    endif()
else()
    message(STATUS "OpenMP disabled (RESOLVE_USE_OPENMP=OFF); fuzzy::query_batch serial")
endif()

# Ensure ABI compatibility (matches CMAKE_CXX_STANDARD above)
set_property(TARGET resolve_core PROPERTY CXX_STANDARD 20)

# Export for bindings
install(TARGETS resolve_core
    EXPORT resolve_core-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/resolve DESTINATION include)

# ----------------------------------------------------------------------------
# resolve_c: flat C ABI facade shared library (issue #17)
#
# A single extern "C" DLL/so wrapping the engine so non-MSVC clients (the R
# package's Rtools mingw build on Windows) can call it without ever touching
# libtorch's C++ ABI. libtorch only ships an MSVC build on Windows, and MSVC
# and mingw C++ ABIs do not cross; a C ABI does. Built on every platform so the
# C-client path is identical everywhere (not a Windows-only branch). See
# include/resolve/resolve_capi.h and dev_notes/issue17_c_abi_facade_design.md.
# ----------------------------------------------------------------------------
option(BUILD_R_CAPI "Build the resolve_c C ABI shared library" ON)
if(BUILD_R_CAPI)
    add_library(resolve_c SHARED cpp_src/resolve_capi.cpp)
    target_link_libraries(resolve_c PRIVATE resolve_core ${TORCH_LIBRARIES})
    target_include_directories(resolve_c PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    set_property(TARGET resolve_c PROPERTY CXX_STANDARD 20)
    # Activate __declspec(dllexport) / visibility("default") on the C API.
    target_compile_definitions(resolve_c PRIVATE RESOLVE_CAPI_BUILD=1)
    # torch TUs are large; MSVC needs /bigobj for the object that pulls in
    # the full engine + libtorch headers.
    if(MSVC)
        target_compile_options(resolve_c PRIVATE /bigobj)
        # Stage the libtorch runtime DLLs next to resolve_c.dll so it loads
        # standalone (the R package adds this dir to PATH at load time).
        file(GLOB RESOLVE_C_TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET resolve_c POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${RESOLVE_C_TORCH_DLLS} $<TARGET_FILE_DIR:resolve_c>)
    endif()
    install(TARGETS resolve_c
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
endif()

# Python bindings
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

# R bindings
if(BUILD_R)
    add_subdirectory(r)
endif()

# CLI application
if(BUILD_CLI)
    add_subdirectory(cli)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
