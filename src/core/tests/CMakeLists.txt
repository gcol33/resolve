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
    test_dataset.cpp
    test_loss.cpp
    test_new_modules.cpp
    test_rank_pool_encoder.cpp
    test_transformer_encoder.cpp
    test_mlm_pretrain.cpp
    test_spatial_cv.cpp
    test_species_encoders.cpp
    test_fuzzy.cpp
    test_categorical.cpp
    test_gpu.cpp
    test_predictor.cpp
    test_trainer_eval.cpp
    test_taxonomy_vocab.cpp
    test_oom_retry.cpp
)

# Add CUDA library directory if CUDA is available
if(USE_CUDA AND CUDAToolkit_FOUND)
    target_link_directories(resolve_tests PRIVATE ${CUDAToolkit_LIBRARY_DIR})
elseif(USE_CUDA AND DEFINED CUDA_TOOLKIT_ROOT_DIR)
    target_link_directories(resolve_tests PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
elseif(USE_CUDA AND DEFINED ENV{CUDA_PATH})
    target_link_directories(resolve_tests PRIVATE $ENV{CUDA_PATH}/lib/x64)
endif()

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

# Benchmark executable (standalone, no Catch2)
add_executable(resolve_benchmark
    benchmark_cuda.cpp
)

# Add CUDA library directory if CUDA is available
if(USE_CUDA AND CUDAToolkit_FOUND)
    target_link_directories(resolve_benchmark PRIVATE ${CUDAToolkit_LIBRARY_DIR})
elseif(USE_CUDA AND DEFINED CUDA_TOOLKIT_ROOT_DIR)
    target_link_directories(resolve_benchmark PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
elseif(USE_CUDA AND DEFINED ENV{CUDA_PATH})
    target_link_directories(resolve_benchmark PRIVATE $ENV{CUDA_PATH}/lib/x64)
endif()

target_link_libraries(resolve_benchmark
    PRIVATE
        resolve_core
        ${TORCH_LIBRARIES}
)

# Catch2 benchmarks (with statistical analysis)
add_executable(resolve_benchmark_catch2
    benchmark_catch2.cpp
)

# Add CUDA library directory if CUDA is available
if(USE_CUDA AND CUDAToolkit_FOUND)
    target_link_directories(resolve_benchmark_catch2 PRIVATE ${CUDAToolkit_LIBRARY_DIR})
elseif(USE_CUDA AND DEFINED CUDA_TOOLKIT_ROOT_DIR)
    target_link_directories(resolve_benchmark_catch2 PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
elseif(USE_CUDA AND DEFINED ENV{CUDA_PATH})
    target_link_directories(resolve_benchmark_catch2 PRIVATE $ENV{CUDA_PATH}/lib/x64)
endif()

target_link_libraries(resolve_benchmark_catch2
    PRIVATE
        resolve_core
        ${TORCH_LIBRARIES}
        Catch2::Catch2WithMain
)
