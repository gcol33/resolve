cmake_minimum_required(VERSION 3.18)

# Options (must be set before project() for CUDA)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_R "Build R bindings" OFF)
option(BUILD_CLI "Build CLI application" ON)
option(BUILD_TESTS "Build tests" ON)
option(USE_CUDA "Enable CUDA support" ON)

# Version - single source of truth is the repo-root VERSION file, which
# tools/version.py keeps in sync with resolve::VERSION and the R/Python manifests
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION" RESOLVE_VERSION)
string(STRIP "${RESOLVE_VERSION}" RESOLVE_VERSION)

# Project declaration - CXX only initially
# CUDA is enabled via project() so it's identified before PyTorch's find_package calls enable_language(CUDA)
if(USE_CUDA)
    project(resolve VERSION ${RESOLVE_VERSION} LANGUAGES CXX CUDA)
else()
    project(resolve VERSION ${RESOLVE_VERSION} LANGUAGES CXX)
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

# Warning flags for RESOLVE's own targets, as the INTERFACE library
# resolve_warnings. Shared with the Python extension build so both agree.
include(${CMAKE_CURRENT_SOURCE_DIR}/resolve_warnings.cmake)

# Core library sources -- single source of truth shared with the Python
# extension build (src/core/python/CMakeLists.txt) via resolve_core_sources.cmake
# so the two lists never drift.
set(RESOLVE_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include(${CMAKE_CURRENT_SOURCE_DIR}/resolve_core_sources.cmake)

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
    include/resolve/io_retry.hpp
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
    include/resolve/process.hpp
)

# Build core library (C++ only)
add_library(resolve_core STATIC ${RESOLVE_CORE_SOURCES} ${RESOLVE_CORE_HEADERS})

# CUDA custom kernels (optional - skip if nvcc environment is broken). The core
# library works fine without them - it uses PyTorch's CUDA ops. Shared with the
# standalone Python extension build via resolve_cuda_kernels.cmake so both agree
# on kernel compilation and the RESOLVE_HAS_CUDA define.
include(${CMAKE_CURRENT_SOURCE_DIR}/resolve_cuda_kernels.cmake)

target_include_directories(resolve_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Prefer the libstdc++ the engine was compiled against over the one libtorch
# re-exports.
#
# PyTorch builds libtorch with -static-libstdc++ for portability, and
# libtorch.so / libtorch_cpu.so then export the absorbed libstdc++ symbols
# (64 and 142 std::filesystem entries respectively). The compiler links
# libstdc++ last, so libtorch precedes it in DT_NEEDED and the dynamic
# linker -- which takes the first definition it finds -- resolves the
# engine's std::filesystem calls into libtorch's copy instead of the one the
# headers were compiled against. Against a pip libtorch under GCC 13.3 that
# copy sends std::filesystem::remove_all through a null pointer and the
# process takes SIGSEGV.
#
# Naming libstdc++ in the link-options slot puts it ahead of the objects and
# of ${TORCH_LIBRARIES}, so it takes the first DT_NEEDED entry. --no-as-needed
# is required because nothing is undefined yet at that point, so the linker
# would otherwise drop it. This only reorders a library the driver already
# links; any symbol absent from it still falls through to the next definer.
if(UNIX AND NOT APPLE)
    target_link_options(resolve_core PUBLIC "-Wl,--no-as-needed,-lstdc++,--as-needed")
endif()

target_link_libraries(resolve_core PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(resolve_core PRIVATE resolve_warnings)

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

# Export for bindings. resolve_warnings rides along because resolve_core links
# it PRIVATE, which for a static library records it in the interface under
# $<LINK_ONLY:>; an install(EXPORT) of this set would otherwise fail on a
# target that is not part of it.
install(TARGETS resolve_core resolve_warnings
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
    target_link_libraries(resolve_c PRIVATE resolve_core ${TORCH_LIBRARIES} resolve_warnings)
    target_include_directories(resolve_c PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    set_property(TARGET resolve_c PROPERTY CXX_STANDARD 20)
    # Activate __declspec(dllexport) / visibility("default") on the C API.
    target_compile_definitions(resolve_c PRIVATE RESOLVE_CAPI_BUILD=1)
    # Relocatable RPATH so the shipped library finds its sibling libtorch
    # runtimes wherever the release zip is unzipped (issue #17 packaging flattens
    # resolve_c + libtorch into one dir). libc10.so is a direct DT_NEEDED of
    # libresolve_c.so, so $ORIGIN must be in this library's own DT_RUNPATH --
    # a build-tree absolute RPATH to libtorch/lib alone leaves a relocated
    # bundle unable to resolve it. macOS gets @loader_path via install_name_tool
    # in the release workflow; Windows resolves siblings via the altered search
    # path in resolve_capi_dynload.cpp.
    if(UNIX AND NOT APPLE)
        set_target_properties(resolve_c PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN")
    endif()
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
