cmake_minimum_required(VERSION 3.18)
project(resolve VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_R "Build R bindings" OFF)
option(BUILD_TESTS "Build tests" ON)

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
    cpp_src/species_encoder.cpp
    cpp_src/loss.cpp
)

set(RESOLVE_CORE_HEADERS
    include/resolve/resolve.hpp
    include/resolve/types.hpp
    include/resolve/encoder.hpp
    include/resolve/model.hpp
    include/resolve/trainer.hpp
    include/resolve/predictor.hpp
    include/resolve/species_encoder.hpp
    include/resolve/loss.hpp
)

add_library(resolve_core STATIC ${RESOLVE_CORE_SOURCES} ${RESOLVE_CORE_HEADERS})

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
