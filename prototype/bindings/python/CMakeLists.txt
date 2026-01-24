cmake_minimum_required(VERSION 3.18)
project(resolve_python VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# Add core library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../core ${CMAKE_BINARY_DIR}/core)

# Create Python module
pybind11_add_module(_core src/bindings.cpp)

target_link_libraries(_core PRIVATE
    resolve_core
    ${TORCH_LIBRARIES}
)

# Set output directory
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resolve
)

# Install
install(TARGETS _core
    LIBRARY DESTINATION resolve
)

install(DIRECTORY resolve/
    DESTINATION resolve
    FILES_MATCHING PATTERN "*.py"
)
