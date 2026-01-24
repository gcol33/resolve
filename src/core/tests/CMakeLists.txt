# Tests for resolve core library

# Download Catch2
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# Test executable
add_executable(resolve_tests
    test_species_encoder.cpp
    test_model.cpp
)

target_link_libraries(resolve_tests
    PRIVATE
        resolve_core
        ${TORCH_LIBRARIES}
        Catch2::Catch2WithMain
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(resolve_tests)
