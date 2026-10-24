#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "resolve/types.hpp"

#include <c10/util/Exception.h>

namespace resolve {

// The device a binding, the CLI or a config names as text. Accepted, exactly as
// libtorch spells them (lower case, no spaces):
//
//   "cpu"     the CPU
//   "cuda"    the current CUDA device
//   "cuda:N"  CUDA device N, N a non-negative decimal integer
//
// Anything else -- "CUDA", "gpu", "cuda:", "cuda:-1", "cuda:1x", "" -- throws
// std::invalid_argument naming the accepted forms, so a misspelled device is an
// error rather than a run on the CPU. Whether the named device exists is not
// checked here: that is a question about the machine, answered when the device
// is first used.
[[nodiscard]] torch::Device parse_device(const std::string& name);

// The text parse_device reads back to `device`: "cpu", "cuda" (no index) or
// "cuda:N".
[[nodiscard]] std::string device_name(const torch::Device& device);

// The PYTORCH_CUDA_ALLOC_CONF value RESOLVE sets to tame WDDM fragmentation:
// the one definition, read by configure_cuda_allocator and by every binding.
// Includes the `expandable_segments:True` prefix off Windows (the cuMemMap
// allocator is not implemented on win32).
std::string default_cuda_alloc_conf();

// Apply default_cuda_alloc_conf() to the process environment and return the
// value PYTORCH_CUDA_ALLOC_CONF ends up holding.
//
// An existing non-empty setting wins unless `force` is true, so a user's own
// allocator tuning is never overwritten silently. The CUDA caching allocator
// reads the variable when it first allocates on a device, so this takes effect
// whenever it runs before the process's first CUDA allocation, and not after.
// The nanobind module and the C ABI call it when they load.
std::string configure_cuda_allocator(bool force);

// Limit the fraction of GPU VRAM that the PyTorch caching allocator may use
// on the given CUDA device. Wraps
// c10::cuda::CUDACachingAllocator::setMemoryFraction.
//
// fraction must be in (0, 1]; 1.0 means uncapped and leaves the allocator
// alone. device_index = -1 (default) uses the current CUDA device.
//
// On Windows the WDDM driver spills allocations beyond device VRAM into shared
// system memory, which hangs the whole desktop under load. A fraction < 1.0
// keeps headroom for the desktop compositor and other GPU apps while RESOLVE
// trains; a dedicated training server keeps the default 1.0.
//
// Changes nothing, and says so through `log`, when fraction == 1.0, when the
// library was built without CUDA support (RESOLVE_HAS_CUDA undefined), or when
// no CUDA device is present at runtime. Throws std::invalid_argument for
// fraction <= 0 or fraction > 1, and for a device_index past the last device.
void set_vram_fraction(
    double fraction,
    int device_index = -1,
    LogCallback log = default_log
);

// Decision helper for the auto-halve-on-OOM retry (retry_halving_on_oom) of
// Trainer::fit and the pretraining loop.
//
// Pure logic, no torch dependency: given the run that failed (`where`, named
// in the diagnostic), the batch size that just OOM'd (`prev_bs`), the
// configured retry `floor`, the batch size originally requested when the run
// was entered (`original_bs`), and the OOM's `oom_what()` string, decides
// whether to retry with `prev_bs/2` or rethrow as a runtime_error.
//
// Returns true and fills `out_log_msg` if the caller should halve to
// `out_new_batch_size` and retry. Returns false and fills `out_err_msg`
// with a diagnostic that the caller must throw as std::runtime_error.
//
// A free function so it is testable without exhausting CUDA VRAM.
bool decide_oom_retry(
    const char* where,
    int prev_bs,
    int floor,
    int original_bs,
    const char* oom_what,
    int& out_new_batch_size,
    std::string& out_log_msg,
    std::string& out_err_msg
);

// Decision helper for an out-of-memory error in the trainer's held-out
// forward (the per-epoch evaluation, the baselines and the test-fold
// evaluators), which runs the fold in chunks of `prev_chunk` plots.
//
// An evaluation OOM shrinks the evaluation chunk and never the training batch:
// the forward runs without gradients, so its memory is set by the chunk alone.
// Returns true and fills `out_log_msg` when the caller should retry the failed
// chunk at `out_new_chunk` = prev_chunk / 2. Returns false and fills
// `out_err_msg` (for the caller to throw as std::runtime_error) when the chunk
// is already one plot, or when `full_batch_required`: an architecture whose
// forward must see the whole fold at once (Trainer's coordinate-kNN GNN) has
// no chunk to shrink.
bool decide_eval_chunk_retry(
    int64_t prev_chunk,
    bool full_batch_required,
    const char* oom_what,
    int64_t& out_new_chunk,
    std::string& out_log_msg,
    std::string& out_err_msg
);

// Whether `error` is a CUDA out-of-memory failure: the caching allocator's
// c10::OutOfMemoryError (its reserved cap is hit), or a c10 error carrying
// cudaErrorMemoryAllocation (a fresh cudaMalloc failed outright once the
// allocator had already returned its free blocks to the driver).
[[nodiscard]] bool is_out_of_memory(const c10::Error& error);

// Return the CUDA caching allocator's free blocks to the device, so a retry
// after an out-of-memory error has the headroom it is retried for. Does
// nothing for a CPU device or a build without CUDA.
void release_cached_device_memory(const torch::Device& device);

// The auto-halve-on-OOM retry of a training run, shared by Trainer::fit and
// the pretraining loop. Calls attempt(n) with n = 0, 1, ... until one returns.
// When attempt(n) fails with a CUDA out-of-memory error (is_out_of_memory),
// release() frees what the attempt held and the error is decided by
// decide_oom_retry(where, batch_size, floor, requested_batch_size, ...): it is
// thrown as std::runtime_error with that diagnostic, or `batch_size` is
// halved, the decision logged and attempt(n + 1) runs at the new size. An
// attempt after the first starts the run over, so it puts back whatever the
// failed attempt had already changed (weights, histories, counters). Any other
// error propagates unchanged and without release().
template <typename Attempt, typename Release>
void retry_halving_on_oom(const char* where, int& batch_size, int floor,
                          int requested_batch_size, const LogCallback& log,
                          Attempt&& attempt, Release&& release) {
    for (int n = 0;; ++n) {
        try {
            attempt(n);
            return;
        } catch (const c10::Error& error) {
            if (!is_out_of_memory(error)) {
                throw;
            }
            int new_bs = batch_size;
            std::string log_msg;
            std::string err_msg;
            const bool retry = decide_oom_retry(where, batch_size, floor, requested_batch_size,
                                                error.what(), new_bs, log_msg, err_msg);
            // Torn down whether the run is retried or the error rethrown, so
            // the caller is left in a consistent state either way.
            release();
            if (!retry) {
                throw std::runtime_error(err_msg);
            }
            log(log_msg);
            batch_size = new_bs;
        }
    }
}

// Whether Trainer::train_epoch runs the side-stream CUDA-hash prefetch, and so
// whether it acquires the two CUDA streams that path needs.
//
// The prefetch needs the CUDA hash path selected, the run actually on a CUDA
// device, and more than one batch to overlap. The middle term is the one that
// matters here: RESOLVE_HAS_CUDA is a COMPILE-time guard, so a CUDA-enabled
// build reaches the prefetch setup on a device="cpu" run too, and acquiring the
// streams there initializes the CUDA runtime -- which throws on a host with no
// usable driver, a CPU queue node or CI runner.
//
// Pure logic, so the decision is testable without a CUDA device.
bool use_hash_prefetch(
    bool use_cuda_hash,
    bool device_is_cuda,
    int64_t n_train,
    int64_t batch_size
);

} // namespace resolve
