#pragma once

// Force-included ahead of every Catch2 test source (tests/CMakeLists.txt).
//
// libtorch's c10/util/logging_is_not_google_glog.h defines CHECK(condition)
// as a fatal log. Whichever of Catch2 and libtorch is included second
// redefines the other's CHECK, so in a file that includes Catch2 first and
// torch after, CHECK(...) is libtorch's: a passing check is never counted and
// a failing one aborts the run instead of being reported. Including libtorch's
// logging header first, dropping its CHECK, and then Catch2 leaves CHECK as the
// Catch2 assertion; both headers are include-guarded, so the files' own
// includes later change nothing.

#include <c10/util/Logging.h>
#undef CHECK
#include <catch2/catch_test_macros.hpp>
