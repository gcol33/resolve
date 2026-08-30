#include "resolve/gpu.hpp"
#include "resolve/env.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <cstring>

#ifdef RESOLVE_HAS_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif

namespace resolve {

std::string default_cuda_alloc_conf() {
    std::string base = "garbage_collection_threshold:0.8,max_split_size_mb:256";
#if !defined(_WIN32)
    base = "expandable_segments:True," + base;
#endif
    return base;
}

std::string configure_cuda_allocator(bool force) {
    const std::string base = default_cuda_alloc_conf();
    const auto existing = get_env("PYTORCH_CUDA_ALLOC_CONF");
    if (force || !existing.has_value() || existing->empty()) {
        set_env("PYTORCH_CUDA_ALLOC_CONF", base.c_str());
        return base;
    }
    return *existing;
}

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

    if (device_index >= static_cast<int>(n_devices)) {
        std::ostringstream err;
        err << "set_vram_fraction: device_index " << device_index
            << " out of range (only " << static_cast<int>(n_devices)
            << " CUDA device(s) present)";
        throw std::invalid_argument(err.str());
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

bool decide_oom_retry(
    int prev_bs,
    int floor,
    int original_bs,
    const char* oom_what,
    int& out_new_batch_size,
    std::string& out_log_msg,
    std::string& out_err_msg
) {
    const int new_bs = prev_bs / 2;
    if (new_bs < floor) {
        // Rethrow path: leave out_new_batch_size at the unchanged current value
        // so a caller that reads it without checking the return does not act on
        // a sub-floor size.
        out_new_batch_size = prev_bs;
        std::ostringstream err;
        err << "CUDA out of memory during Trainer::fit and the auto-halve "
               "retry hit the floor. Original batch_size="
            << original_bs
            << ", current batch_size=" << prev_bs
            << ", batch_size_floor=" << floor
            << ". Reduce model size, raise vram_fraction (if a cap is "
               "in effect), or lower batch_size_floor. Original OOM: "
            << (oom_what == nullptr ? "<null>" : oom_what);
        out_err_msg = err.str();
        out_log_msg.clear();
        return false;
    }

    out_new_batch_size = new_bs;
    std::ostringstream msg;
    msg << "OOM at batch_size=" << prev_bs
        << "; retrying at batch_size=" << new_bs
        << " (floor=" << floor << ")";
    out_log_msg = msg.str();
    out_err_msg.clear();
    return true;
}

bool use_hash_prefetch(bool use_cuda_hash, bool device_is_cuda,
                       int64_t n_train, int64_t batch_size) {
    return use_cuda_hash && device_is_cuda && n_train > batch_size;
}

} // namespace resolve
