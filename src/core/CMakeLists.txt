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
    project(resolve VERSION 0.1.0 LANGUAGES CXX CUDA)
else()
    project(resolve VERSION 0.1.0 LANGUAGES CXX)
endif()

# Skip Caffe2's CUDA detection by setting this before find_package(Torch)
# We link against PyTorch's pre-built CUDA libraries, not compile CUDA ourselves
set(CAFFE2_USE_CUDNN OFF)
set(USE_CUDNN OFF)

set(CMAKE_CXX_STANDARD 17)
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
    cpp_src/encoder.cpp
    cpp_src/model.cpp
    cpp_src/trainer.cpp
    cpp_src/predictor.cpp
    cpp_src/loss.cpp
    cpp_src/dataset.cpp
)

set(RESOLVE_CORE_HEADERS
    include/resolve/resolve.hpp
    include/resolve/types.hpp
    include/resolve/encoder.hpp
    include/resolve/model.hpp
    include/resolve/trainer.hpp
    include/resolve/predictor.hpp
    include/resolve/loss.hpp
    include/resolve/csv_reader.hpp
    include/resolve/role_mapping.hpp
    include/resolve/dataset.hpp
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
            resolve_cuda_kernels)

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

# Ensure ABI compatibility
set_property(TARGET resolve_core PROPERTY CXX_STANDARD 17)

# Export for bindings
install(TARGETS resolve_core
    EXPORT resolve_core-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/resolve DESTINATION include)

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
