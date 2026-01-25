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
nanobind_add_module(_resolve_core src/bindings.cpp)

target_link_libraries(_resolve_core PRIVATE resolve_core ${TORCH_LIBRARIES})

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
