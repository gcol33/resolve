#include "resolve/gpu.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

#ifdef RESOLVE_HAS_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif

namespace resolve {

void set_vram_fraction(double fraction, int device_index, LogCallback log) {
    if (fraction <= 0.0 || fraction > 1.0) {
        std::ostringstream err;
        err << "vram_fraction must be in (0, 1], got " << fraction;
        throw std::invalid_argument(err.str());
    }

    if (fraction >= 1.0) {
        log("VRAM cap disabled (vram_fraction = 1.0; allocator may use all "
            "available VRAM)");
        return;
    }

#ifdef RESOLVE_HAS_CUDA
    const auto n_devices = c10::cuda::device_count();
    if (n_devices == 0) {
        log("set_vram_fraction: no CUDA devices found, skipping");
        return;
    }

    const c10::DeviceIndex dev = (device_index < 0)
        ? c10::cuda::current_device()
        : static_cast<c10::DeviceIndex>(device_index);

    // The caching allocator lazy-initializes on first device touch. Calling
    // setMemoryFraction before that fails with "Allocator not initialized for
    // device" inside libtorch. Mirror torch.cuda.set_per_process_memory_fraction
    // which calls _lazy_init() first.
    c10::cuda::CUDACachingAllocator::init(n_devices);

    c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction, dev);

    std::ostringstream msg;
    msg << "VRAM cap: PyTorch CUDA caching allocator limited to "
        << (fraction * 100.0) << "% of device " << static_cast<int>(dev)
        << " VRAM";
    log(msg.str());
#else
    (void)device_index;
    log("set_vram_fraction: built without CUDA support, skipping");
#endif
}

} // namespace resolve
