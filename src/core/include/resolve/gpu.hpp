#pragma once

#include <string>

#include "resolve/types.hpp"

namespace resolve {

// The default PYTORCH_CUDA_ALLOC_CONF value RESOLVE sets to tame WDDM
// fragmentation. Single source for the two C++ callers (the nanobind
// configure_cuda_allocator entry point and the C-ABI native shim); the Python
// package keeps its own copy in resolve_core/__init__.py because it must set the
// env var BEFORE `import torch`, i.e. before this library is loaded. Includes
// the `expandable_segments:True` prefix off Windows (the cuMemMap allocator is
// not implemented on win32).
std::string default_cuda_alloc_conf();

// Apply default_cuda_alloc_conf() to the process environment and return the
// value PYTORCH_CUDA_ALLOC_CONF ends up holding.
//
// An existing non-empty setting wins unless `force` is true, so a user's own
// allocator tuning is never overwritten silently. Late-bound: once the CUDA
// caching allocator has initialized, changing the variable no longer affects
// it, which is why resolve_core/__init__.py sets the same value before
// `import torch`. Single source for the nanobind and C-ABI entry points, which
// each carried a copy of this branch.
std::string configure_cuda_allocator(bool force);

// Limit the fraction of GPU VRAM that the PyTorch caching allocator may use
// on the given CUDA device. Wraps
// c10::cuda::CUDACachingAllocator::setMemoryFraction.
//
// fraction must be in (0, 1]; 1.0 is treated as "uncapped" and skips the call.
// device_index = -1 (default) uses the current CUDA device.
//
// On Windows the WDDM driver spills allocations beyond device VRAM into shared
// system memory, which causes the whole desktop to hang under load. Defaulting
// to a fraction < 1.0 keeps headroom for the desktop compositor and other GPU
// apps so the system stays usable while RESOLVE is training. On a dedicated
// training server set this to 1.0 to use all available VRAM.
//
// Silently no-ops when the library was built without CUDA support
// (RESOLVE_HAS_CUDA undefined), when no CUDA devices are present at runtime,
// or when fraction >= 1.0. Throws std::invalid_argument for fraction <= 0
// or fraction > 1.
void set_vram_fraction(
    double fraction,
    int device_index = -1,
    LogCallback log = default_log
);

// Decision helper for the auto-halve-on-OOM retry inside Trainer::fit.
//
// Pure logic, no torch dependency: given the batch size that just OOM'd
// (`prev_bs`), the configured retry `floor`, the batch size originally
// requested when fit() was entered (`original_bs`), and the OOM's
// `oom_what()` string, decides whether to retry with `prev_bs/2` or
// rethrow as a runtime_error.
//
// Returns true and fills `out_log_msg` if the caller should halve to
// `out_new_batch_size` and retry. Returns false and fills `out_err_msg`
// with a diagnostic that the caller must throw as std::runtime_error.
//
// Extracted into a free function so it can be unit-tested without needing
// to actually exhaust CUDA VRAM.
bool decide_oom_retry(
    int prev_bs,
    int floor,
    int original_bs,
    const char* oom_what,
    int& out_new_batch_size,
    std::string& out_log_msg,
    std::string& out_err_msg
);

} // namespace resolve
