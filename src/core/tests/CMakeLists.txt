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
    test_model.cpp
)

target_link_libraries(resolve_tests
    PRIVATE
        resolve_core
        ${TORCH_LIBRARIES}
        Catch2::Catch2WithMain
)

# Register test without discovery (avoids needing to run the binary at build time)
# Tests can be run manually with the test executable
include(CTest)
add_test(NAME resolve_tests COMMAND resolve_tests)
