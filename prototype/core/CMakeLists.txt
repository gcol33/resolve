cmake_minimum_required(VERSION 3.18)
project(resolve_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Torch REQUIRED)
find_package(nlohmann_json 3.9 REQUIRED)

# Source files
set(RESOLVE_SOURCES
    src/encoder.cpp
    src/model.cpp
    src/plot_encoder.cpp
    src/trainer.cpp
)

# Define the core library
add_library(resolve_core STATIC ${RESOLVE_SOURCES})

target_include_directories(resolve_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(resolve_core PUBLIC
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Compiler flags
if(MSVC)
    target_compile_options(resolve_core PRIVATE /W4)
else()
    target_compile_options(resolve_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install
install(TARGETS resolve_core
    EXPORT resolve_core-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/resolve
    DESTINATION include
)

install(EXPORT resolve_core-targets
    FILE resolve_core-targets.cmake
    NAMESPACE resolve::
    DESTINATION lib/cmake/resolve_core
)

# Export for build tree
export(TARGETS resolve_core
    NAMESPACE resolve::
    FILE ${CMAKE_BINARY_DIR}/resolve_core-targets.cmake
)

# Testing
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
