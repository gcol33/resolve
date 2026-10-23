# Single source of truth for the resolve_core C++ translation units and the
# public headers under include/resolve.
#
# include()d by both the main engine build (src/core/CMakeLists.txt) and the
# Python extension build (src/core/python/CMakeLists.txt) so the two never
# drift: a source missing from one list leaves its symbols out of that build.
# The check at the end fails the configure when a file on disk is not listed.
#
# The includer must set RESOLVE_CORE_DIR to the src/core directory (the parent
# of cpp_src/). Paths are absolute so the relative offset of the includer
# (src/core vs src/core/python) does not matter.
get_filename_component(RESOLVE_CORE_DIR "${RESOLVE_CORE_DIR}" ABSOLUTE)

set(RESOLVE_CORE_SOURCES
    ${RESOLVE_CORE_DIR}/cpp_src/encoder_common.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/species_encoder.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/species_pool.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/plot_encoder_hash.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/plot_encoder_embed.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/plot_encoder_sparse.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/plot_encoder_rank_pool.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/plot_encoder_transformer.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/task_head.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/ft_transformer.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/tabnet.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/saint.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/excelformer.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/traitnet.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/gnn.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/hetero_gnn.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/tabm.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/adapter.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/model.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/continuous_block.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/trainer.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/trainer_checkpoint.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/lr_schedule.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/predictor.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/retrieval.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/suite.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/json.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/sha256.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/loss.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/target_labels.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/dataset.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/species_encoding.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/species_graph.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/sparse_rows.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/checkpoint.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/categorical.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/pretraining.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/vae.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/fuzzy_index.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/fuzzy_search.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/fuzzy_automaton.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/gpu.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/process.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/fuzzy_internal.h
)

set(RESOLVE_CORE_HEADERS
    ${RESOLVE_CORE_DIR}/include/resolve/resolve.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/types.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/enum_names.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/config_registry.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/config_json.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/architectures.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/plot_inputs.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/aggregate_arity.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/utils.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/env.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/number_text.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/encoder.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/mlp_block.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/experts.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/species_encoder.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/species_pool.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/plot_encoder_hash.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/plot_encoder_embed.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/plot_encoder_sparse.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/plot_encoder_rank_pool.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/plot_encoder_transformer.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/continuous_block.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/task_head.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/ft_transformer.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/tabnet.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/saint.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/excelformer.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/traitnet.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/gnn.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/hetero_gnn.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/tabm.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/adapter.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/aux_loss.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/ln_mlp.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/model.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/trainer.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/lr_schedule.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/scalers.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/target_transform.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/predictor.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/retrieval.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/suite.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/json.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/sha256.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/loss.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/target_labels.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/csv_reader.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/csv_utils.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/row_source.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/io_retry.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/role_mapping.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/dataset.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/name_pool.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/sparse_rows.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/species_encoding.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/species_graph.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/hash_core.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/checkpoint.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/checkpoint_schema_keys.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/string_list_archive.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/categorical.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/pretraining.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/vae.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/fuzzy.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/gpu.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/process.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/cuda/feature_hash.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/cuda/hash_kernels.h
)

# The C ABI's own headers belong to the resolve_c target, not resolve_core.
set(RESOLVE_CAPI_HEADERS
    ${RESOLVE_CORE_DIR}/include/resolve/resolve_capi.h
    ${RESOLVE_CORE_DIR}/include/resolve/resolve_capi_symbols.inc
)

# Every engine source and public header on disk must be listed above.
# resolve_capi.cpp is the resolve_c target's own source.
file(GLOB RESOLVE_ON_DISK CONFIGURE_DEPENDS
    ${RESOLVE_CORE_DIR}/cpp_src/*.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/*.h
    ${RESOLVE_CORE_DIR}/include/resolve/*.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/*.h
    ${RESOLVE_CORE_DIR}/include/resolve/*.inc
    ${RESOLVE_CORE_DIR}/include/resolve/cuda/*.hpp
    ${RESOLVE_CORE_DIR}/include/resolve/cuda/*.h)
list(REMOVE_ITEM RESOLVE_ON_DISK
    ${RESOLVE_CORE_SOURCES} ${RESOLVE_CORE_HEADERS} ${RESOLVE_CAPI_HEADERS}
    ${RESOLVE_CORE_DIR}/cpp_src/resolve_capi.cpp)
if(RESOLVE_ON_DISK)
    list(JOIN RESOLVE_ON_DISK "\n  " RESOLVE_UNLISTED)
    message(FATAL_ERROR
        "resolve_core_sources.cmake does not list:\n  ${RESOLVE_UNLISTED}")
endif()
