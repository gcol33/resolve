# Single source of truth for the resolve_core C++ translation units.
#
# include()d by both the main engine build (src/core/CMakeLists.txt) and the
# Python extension build (src/core/python/CMakeLists.txt) so the two never drift
# -- the drift (the Python list lacking gpu.cpp / process.cpp) was why the PyPI
# wheel build failed to link `set_vram_fraction` / `install_crash_handler`.
#
# The includer must set RESOLVE_CORE_DIR to the src/core directory (the parent
# of cpp_src/). Sources are listed as absolute paths so the relative offset of
# the includer (src/core vs src/core/python) does not matter.
set(RESOLVE_CORE_SOURCES
    ${RESOLVE_CORE_DIR}/cpp_src/encoder_common.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/encoder_topk.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/encoder_pool.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/task_head.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/attention.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/gnn.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/tabm.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/adapter.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/model.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/trainer.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/predictor.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/loss.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/dataset.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/species_encoding.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/checkpoint.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/categorical.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/pretraining.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/vae.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/fuzzy_index.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/fuzzy_search.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/fuzzy_automaton.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/gpu.cpp
    ${RESOLVE_CORE_DIR}/cpp_src/process.cpp
)
