cmake_minimum_required(VERSION 3.18)

# Check if we're being included as a subdirectory or built standalone
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Standalone build - define project and build resolve_core
    project(resolve_python VERSION 0.7.1 LANGUAGES CXX)

    # CPU by default: the PyPI wheels link the CPU libtorch the user's `torch`
    # ships and do not bundle CUDA. Pass -DUSE_CUDA=ON for a local CUDA build.
    option(USE_CUDA "Enable CUDA support" OFF)

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

    # Locate libtorch via the installed `torch` package's CMake prefix when no
    # explicit Torch_DIR / CMAKE_PREFIX_PATH was given. This is how the PyPI
    # wheel and sdist builds find libtorch: `torch` is a build dependency
    # (pyproject.toml), so it is present in the build environment, and the
    # extension links the same libtorch the user's torch ships -- no separate
    # libtorch is downloaded or bundled.
    if(NOT Torch_DIR AND NOT CMAKE_PREFIX_PATH)
        find_package(Python COMPONENTS Interpreter REQUIRED)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE TORCH_PROBE_RC
        )
        if(TORCH_PROBE_RC EQUAL 0 AND TORCH_CMAKE_PREFIX)
            list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
            message(STATUS "Using libtorch from the torch package: ${TORCH_CMAKE_PREFIX}")
        endif()
    endif()

    # Find libtorch (will skip CUDA if the CPU-only vars above are set)
    find_package(Torch REQUIRED)

    # Core library sources: single source of truth shared with the main engine
    # build via resolve_core_sources.cmake (no drift -- this is why the wheel
    # previously failed to link set_vram_fraction / install_crash_handler).
    set(RESOLVE_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
    include(${RESOLVE_CORE_DIR}/resolve_core_sources.cmake)

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

# Runtime libtorch resolution. The wheel deliberately does NOT bundle libtorch
# -- it links the same libs the user's `torch` ships (a separate copy would be
# huge and risk an ABI clash). resolve_core/__init__.py imports torch before
# loading this extension, so torch's libs are already in the process; the RPATH
# below is a belt-and-braces pointer to the installed torch lib dir
# (site-packages/torch/lib, one level up from resolve_core/). On Windows, torch
# adds its lib dir to the DLL search at import, so no RPATH/copy is needed.
if(UNIX AND NOT APPLE)
    set_target_properties(_resolve_core PROPERTIES
        INSTALL_RPATH "$ORIGIN/../torch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE)
elseif(APPLE)
    set_target_properties(_resolve_core PROPERTIES
        INSTALL_RPATH "@loader_path/../torch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE)
    # Linking against libtorch_python pulls libtorch's bundled pybind11 into the
    # module (via torch/csrc/autograd/python_variable.h in bindings_model.cpp),
    # which references Python C-API symbols (PyInstanceMethod_*, _PyThreadState_
    # UncheckedGet) not in nanobind's restricted macOS symbol list. A Python
    # extension on macOS does not link libpython; those symbols resolve at load
    # time from the host interpreter, so allow undefined-symbol dynamic lookup
    # (the canonical macOS Python-extension link flag). Without it the arm64 link
    # fails with "Undefined symbols for architecture arm64".
    target_link_options(_resolve_core PRIVATE -Wl,-undefined,dynamic_lookup)
endif()

# Install target for scikit-build-core
install(TARGETS _resolve_core LIBRARY DESTINATION resolve_core)
