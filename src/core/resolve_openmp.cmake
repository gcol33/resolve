# OpenMP for fuzzy::FuzzyIndex::query_batch, included by both the main engine
# build (src/core/CMakeLists.txt) and the standalone Python extension build
# (src/core/python/CMakeLists.txt) so the two agree on the switch below.
#
# Expects, before inclusion:
#   resolve_core - an already-defined library target
#
# Optional: query_batch runs its serial loop when OpenMP is absent or off.
#
# RESOLVE_USE_OPENMP=OFF drops the OpenMP runtime entirely. The R package's
# resolve_c build sets this: the package does not expose fuzzy::query_batch, and
# stacking a third OpenMP runtime (MSVC vcomp) on top of libtorch's libiomp5md
# and R's own libgomp inside a single mingw R process crashes on the first
# parallel op. Dropping vcomp removes that conflict; query_batch falls back to
# its serial path (unused from R anyway).

option(RESOLVE_USE_OPENMP "Link OpenMP into resolve_core (fuzzy::query_batch)" ON)
if(RESOLVE_USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(resolve_core PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP enabled for resolve_core")
    else()
        message(STATUS "OpenMP not found; fuzzy::query_batch will run serially")
    endif()
else()
    message(STATUS "OpenMP disabled (RESOLVE_USE_OPENMP=OFF); fuzzy::query_batch serial")
endif()
