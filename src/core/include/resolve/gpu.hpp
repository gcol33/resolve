#pragma once

#include "resolve/types.hpp"

namespace resolve {

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

} // namespace resolve
