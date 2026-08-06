# Single source of truth for the warning flags RESOLVE's own targets compile
# with, exposed as the INTERFACE library `resolve_warnings`.
#
# include()d by the engine build (src/core/CMakeLists.txt) and by the Python
# extension build (src/core/python/CMakeLists.txt), which is also the sdist root
# and configures standalone -- hence the guard: both may include this in a
# single configure of the engine tree.
#
# Only RESOLVE's own targets link it; the FetchContent'd third-party targets
# (Catch2, nanobind) keep their own flags. Third-party HEADERS stay quiet by
# being SYSTEM includes, not by weakening the flags: libtorch reaches us through
# an IMPORTED target, whose INTERFACE_INCLUDE_DIRECTORIES CMake passes as SYSTEM
# by default (-isystem on GCC/Clang, -external:I plus -external:W0 on MSVC),
# while Catch2 and nanobind reach us through FetchContent targets, which are not
# IMPORTED and so are promoted explicitly in tests/CMakeLists.txt and
# python/CMakeLists.txt. Every target_include_directories() call in the tree
# names a RESOLVE source directory, so none of them re-adds a third-party tree
# non-SYSTEM.
#
# Flags are gated on COMPILE_LANGUAGE:CXX so a target that also carries .cu
# sources does not hand -Wall (or /W4) to nvcc, which rejects them.

if(NOT TARGET resolve_warnings)
    add_library(resolve_warnings INTERFACE)

    if(MSVC)
        # /permissive- is already implied by /std:c++20, stated here so the
        # conformance requirement does not silently depend on the standard.
        set(RESOLVE_WARNING_FLAGS /W4 /permissive-)

        # Two /W4 codes that libtorch emits on its own and that no RESOLVE
        # source line produces. Measured, not assumed: a translation unit whose
        # entire content is `#include <torch/torch.h>` plus an empty function
        # reproduces exactly C4267 x8 and C4702 x2, and across the whole build
        # neither code is ever reported at a file under src/core.
        #
        #   C4702 unreachable code -- every occurrence is the acknowledged
        #     `return false; // Horrible hack` in c10/util/irange.h. It is
        #     raised by the back end after inlining, which is past the point
        #     where /external: scoping applies, so the -external:I + -external:W0
        #     CMake already passes for libtorch cannot reach it.
        #   C4267 size_t narrowing -- raised inside <optional>, <xutility>,
        #     <utility> and <vector> while instantiating libtorch's
        #     ATen/core/function_schema.h and torch/csrc/dynamo/
        #     compiled_autograd.h. MSVC attributes an instantiation-time warning
        #     to the translation unit, so neither /external:env:INCLUDE,
        #     /external:anglebrackets, /external:templates- nor a
        #     `#pragma warning(push, 0)` around the include suppresses it (all
        #     four measured: warning count unchanged at 10).
        #
        # Neither code has a counterpart in the -Wall -Wextra set used on
        # GCC/Clang, so switching them off keeps the two toolchains on the same
        # diagnostic surface rather than dropping MSVC below it. Narrowing that
        # loses data in RESOLVE's own code is still caught: C4244 stays on, and
        # it is what flagged the int64_t -> int and double -> float sites fixed
        # alongside this change.
        list(APPEND RESOLVE_WARNING_FLAGS /wd4267 /wd4702)
    else()
        set(RESOLVE_WARNING_FLAGS -Wall -Wextra -Wshadow -Wnon-virtual-dtor)
    endif()

    # -Werror is opt-in and belongs to one CI job, not to a developer build:
    # the local Windows/CUDA toolchain emits a different diagnostic set from
    # the Linux compiler that gates pull requests, so making it the default
    # would break local builds on warnings CI never sees.
    option(RESOLVE_WERROR "Treat compiler warnings as errors" OFF)
    if(RESOLVE_WERROR)
        if(MSVC)
            list(APPEND RESOLVE_WARNING_FLAGS /WX)
        else()
            list(APPEND RESOLVE_WARNING_FLAGS -Werror)
        endif()
    endif()

    target_compile_options(resolve_warnings INTERFACE
        "$<$<COMPILE_LANGUAGE:CXX>:${RESOLVE_WARNING_FLAGS}>")
endif()
