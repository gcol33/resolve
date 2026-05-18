cmake_minimum_required(VERSION 3.18)

# Check if we're being included as a subdirectory or built standalone
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Standalone build - define project and build resolve_core
    project(resolve_python VERSION 0.5.0 LANGUAGES CXX)

    option(USE_CUDA "Enable CUDA support" ON)

    # libtorch 2.13+ requires C++20 (bit-field default initializers in
    # c10/core/AutogradState.h). Match the parent src/core/CMakeLists.txt.
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    # Skip Caffe2's CUDA detection entirely when building CPU-only
    if(NOT USE_CUDA)
        set(CAFFE2_USE_CUDA OFF CACHE BOOL "" FORCE)
        set(USE_CUDA OFF CACHE BOOL "" FORCE)
        set(CAFFE2_USE_CUDNN OFF CACHE BOOL "" FORCE)
        set(USE_CUDNN OFF CACHE BOOL "" FORCE)
        # This is the key: tell Torch not to look for CUDA
        set(TORCH_CUDA_ARCH_LIST "" CACHE STRING "" FORCE)
        set(CMAKE_CUDA_COMPILER "" CACHE FILEPATH "" FORCE)
    endif()

    # Find libtorch (will skip CUDA if above vars are set)
    find_package(Torch REQUIRED)

    # Build the core library from parent directory
    set(RESOLVE_CORE_SOURCES
        ../cpp_src/encoder_common.cpp
        ../cpp_src/encoder_topk.cpp
        ../cpp_src/encoder_pool.cpp
        ../cpp_src/task_head.cpp
        ../cpp_src/attention.cpp
        ../cpp_src/gnn.cpp
        ../cpp_src/tabm.cpp
        ../cpp_src/adapter.cpp
        ../cpp_src/model.cpp
        ../cpp_src/trainer.cpp
        ../cpp_src/predictor.cpp
        ../cpp_src/loss.cpp
        ../cpp_src/dataset.cpp
        ../cpp_src/species_encoding.cpp
        ../cpp_src/checkpoint.cpp

        ../cpp_src/pretraining.cpp
        ../cpp_src/vae.cpp

        ../cpp_src/fuzzy_index.cpp
        ../cpp_src/fuzzy_search.cpp
        ../cpp_src/fuzzy_automaton.cpp
    )

    add_library(resolve_core STATIC ${RESOLVE_CORE_SOURCES})
    target_include_directories(resolve_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    target_link_libraries(resolve_core PUBLIC ${TORCH_LIBRARIES})

    # OpenMP for fuzzy::query_batch (optional).
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(resolve_core PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP enabled for resolve_core (standalone Python build)")
    else()
        message(STATUS "OpenMP not found; fuzzy::query_batch will run serially")
    endif()

    set_property(TARGET resolve_core PROPERTY CXX_STANDARD 20)
endif()

# Python bindings using nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Fetch nanobind
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.4.0
)
FetchContent_MakeAvailable(nanobind)

# Build the extension module
set(BINDING_SOURCES
    src/bindings.cpp
    src/bindings_enums.cpp
    src/bindings_types.cpp
    src/bindings_dataset.cpp
    src/bindings_model.cpp
    src/bindings_trainer.cpp
    src/bindings_metrics.cpp
    src/bindings_pretraining.cpp
    src/bindings_fuzzy.cpp
)
nanobind_add_module(_resolve_core ${BINDING_SOURCES})

# Find torch_python library for Python tensor interop (THPVariable_Wrap, etc.)
get_filename_component(TORCH_LIB_DIR "${TORCH_INSTALL_PREFIX}/lib" ABSOLUTE)
find_library(TORCH_PYTHON_LIB torch_python PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH)
if(TORCH_PYTHON_LIB)
    message(STATUS "Found torch_python: ${TORCH_PYTHON_LIB}")
    target_link_libraries(_resolve_core PRIVATE resolve_core ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB})
else()
    message(WARNING "torch_python library not found, some features may not work")
    target_link_libraries(_resolve_core PRIVATE resolve_core ${TORCH_LIBRARIES})
endif()

target_include_directories(_resolve_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Set output directory for the built module
set_target_properties(_resolve_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/resolve_core
)

# Copy torch libraries to output (for Windows)
if(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET _resolve_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:_resolve_core>
    )
endif()

# Install target for scikit-build-core
install(TARGETS _resolve_core LIBRARY DESTINATION resolve_core)
