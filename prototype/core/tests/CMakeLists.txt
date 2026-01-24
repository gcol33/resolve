cmake_minimum_required(VERSION 3.18)

# Find Catch2
find_package(Catch2 3 REQUIRED)

# Test sources
set(TEST_SOURCES
    test_plot_encoder.cpp
    test_model.cpp
    test_loss.cpp
)

# Create test executable
add_executable(resolve_tests ${TEST_SOURCES})

target_link_libraries(resolve_tests
    PRIVATE
        resolve_core
        Catch2::Catch2WithMain
)

# Register tests with CTest
include(Catch)
catch_discover_tests(resolve_tests)
