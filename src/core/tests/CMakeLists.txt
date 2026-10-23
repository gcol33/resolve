# Tests for resolve core library
#
# Every test/benchmark target links resolve_warnings (resolve_warnings.cmake),
# so the test sources compile under the same flags as the engine.
#
# RESOLVE_BUILD_BENCHMARKS adds the two benchmark executables and the
# crash-handler smoke; nothing in CTest runs them.
option(RESOLVE_BUILD_BENCHMARKS "Build the benchmark and crash-smoke executables" OFF)

# Download Catch2
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# Catch2 builds with its own flags, but its headers reach the test translation
# units through a FetchContent target, which is not IMPORTED, so CMake does not
# mark them SYSTEM the way it does libtorch's. Promote them: RESOLVE's warning
# flags are for RESOLVE's code, and with -Werror on the Linux CI job a future
# Catch2 bump that ships a header diagnostic would otherwise fail the build.
# Diagnostics raised inside a TEST_CASE / REQUIRE expansion still land on our
# source line and are still reported.
foreach(RESOLVE_CATCH_TARGET Catch2 Catch2WithMain)
    if(TARGET ${RESOLVE_CATCH_TARGET})
        get_target_property(RESOLVE_CATCH_INCS ${RESOLVE_CATCH_TARGET}
            INTERFACE_INCLUDE_DIRECTORIES)
        if(RESOLVE_CATCH_INCS)
            set_property(TARGET ${RESOLVE_CATCH_TARGET} APPEND PROPERTY
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${RESOLVE_CATCH_INCS})
        endif()
    endif()
endforeach()

# Links the CUDA toolkit's library directory into `target` on a CUDA build.
function(resolve_link_cuda_dirs target)
    if(USE_CUDA AND CUDAToolkit_FOUND)
        target_link_directories(${target} PRIVATE ${CUDAToolkit_LIBRARY_DIR})
    elseif(USE_CUDA AND DEFINED CUDA_TOOLKIT_ROOT_DIR)
        target_link_directories(${target} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
    elseif(USE_CUDA AND DEFINED ENV{CUDA_PATH})
        target_link_directories(${target} PRIVATE $ENV{CUDA_PATH}/lib/x64)
    endif()
endfunction()

# Test executable. Shared fixtures live in test_support.hpp.
add_executable(resolve_tests
    test_model.cpp
    test_dataset.cpp
    test_loss.cpp
    test_tabm.cpp
    test_pretraining.cpp
    test_vae.cpp
    test_experts.cpp
    test_gnn.cpp
    test_attention_encoders.cpp
    test_tabular_adapter.cpp
    test_cross_validation.cpp
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
    test_csv_utils.cpp
    test_process.cpp
    test_io_retry.cpp
    test_amp_norm.cpp
    test_dataframe.cpp
    test_recovery.cpp
    test_entmax.cpp
    test_pretrain_seed.cpp
    test_checkpoint_vocab.cpp
    test_effective_batch.cpp
    test_config_registry.cpp
    test_unknown_tracking.cpp
    test_env_utils.cpp
    test_species_selection.cpp
    test_moe_placement.cpp
    test_info_report.cpp
    test_missing_values.cpp
    test_freeze_composition.cpp
    test_architecture_knobs.cpp
    test_species_graph.cpp
    test_cli_config_flags.cpp
    test_suite.cpp
    test_target_masking.cpp
)

resolve_link_cuda_dirs(resolve_tests)

target_link_libraries(resolve_tests
    PRIVATE
        resolve_core
        ${TORCH_LIBRARIES}
        Catch2::Catch2WithMain
        resolve_warnings
)

# Two CTest entries over the one executable: the unit cases, and the
# parameter-recovery fits ([recovery]) that train to convergence and take
# minutes. `ctest -L unit` / `ctest -L recovery` runs one of them.
include(CTest)
add_test(NAME resolve_tests_unit COMMAND resolve_tests "~[recovery]")
add_test(NAME resolve_tests_recovery COMMAND resolve_tests "[recovery]")
set_tests_properties(resolve_tests_unit PROPERTIES LABELS unit)
set_tests_properties(resolve_tests_recovery PROPERTIES LABELS recovery)

if(RESOLVE_BUILD_BENCHMARKS)
    # Crash-handler smoke. Deliberately faults, so it is a standalone exe
    # checked from outside by its exit code, not a Catch2 case.
    # Driver: src/core/dev_notes/crash_smoke_drive.ps1.
    add_executable(resolve_crash_smoke
        crash_smoke.cpp
    )
    target_link_libraries(resolve_crash_smoke
        PRIVATE
            resolve_core
            ${TORCH_LIBRARIES}
            resolve_warnings
    )

    # Benchmark executable (standalone, no Catch2)
    add_executable(resolve_benchmark
        benchmark_cuda.cpp
    )
    resolve_link_cuda_dirs(resolve_benchmark)
    target_link_libraries(resolve_benchmark
        PRIVATE
            resolve_core
            ${TORCH_LIBRARIES}
            resolve_warnings
    )

    # Catch2 benchmarks (with statistical analysis)
    add_executable(resolve_benchmark_catch2
        benchmark_catch2.cpp
    )
    resolve_link_cuda_dirs(resolve_benchmark_catch2)
    target_link_libraries(resolve_benchmark_catch2
        PRIVATE
            resolve_core
            ${TORCH_LIBRARIES}
            Catch2::Catch2WithMain
            resolve_warnings
    )
endif()
