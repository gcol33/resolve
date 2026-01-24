cmake_minimum_required(VERSION 3.18)
project(resolve VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_R "Build R bindings" OFF)
option(BUILD_TESTS "Build tests" ON)
option(USE_CUDA "Enable CUDA support" ON)

# CUDA configuration - must be set BEFORE find_package(Torch)
# PyTorch was built with CUDA, so we need CUDA available even for CPU-only code
#
# Force CMake to accept CUDA compiler without identification (works around nvcc issues)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER_FORCED TRUE)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Tell Torch/Caffe2 that we're not building CUDA code ourselves
# This should let PyTorch find CUDA for linking but skip compiling .cu files
set(CAFFE2_USE_CUDA ON)

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
)

set(RESOLVE_CORE_HEADERS
    include/resolve/resolve.hpp
    include/resolve/types.hpp
    include/resolve/encoder.hpp
    include/resolve/model.hpp
    include/resolve/trainer.hpp
    include/resolve/predictor.hpp
    include/resolve/loss.hpp
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

# CUDA custom kernels with CUDA 13.x workaround:
# - Pure CUDA kernels (no PyTorch headers) are compiled separately
# - C++ wrappers call extern "C" kernel launchers
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # Pure CUDA kernel library - NO PyTorch headers
    # This compiles with nvcc and avoids the 'char' variable name bug
    add_library(resolve_cuda_kernels STATIC ${RESOLVE_CUDA_KERNEL_SOURCES})
    target_include_directories(resolve_cuda_kernels PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    set_target_properties(resolve_cuda_kernels PROPERTIES
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES "89")
    # NO torch libraries here - pure CUDA only

    # C++ wrapper that includes PyTorch headers
    add_library(resolve_cuda_wrapper STATIC ${RESOLVE_CUDA_WRAPPER_SOURCES})
    target_include_directories(resolve_cuda_wrapper PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(resolve_cuda_wrapper PUBLIC
        ${TORCH_LIBRARIES}
        resolve_cuda_kernels)

    # Link to core
    target_link_libraries(resolve_core PUBLIC resolve_cuda_wrapper)
    target_compile_definitions(resolve_core PUBLIC RESOLVE_HAS_CUDA)
    message(STATUS "CUDA kernels enabled (CUDA 13.x compatible)")
else()
    message(STATUS "CUDA kernels disabled (no CUDA compiler found)")
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

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
