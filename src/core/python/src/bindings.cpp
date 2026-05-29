// Main binding entry point - delegates to split modules for maintainability
//
// The actual binding code is split into:
// - bindings_enums.cpp: All enum bindings
// - bindings_types.cpp: Configuration structs and result types
// - bindings_dataset.cpp: ResolveDataset and species encoding
// - bindings_model.cpp: ResolveModel
// - bindings_trainer.cpp: Trainer and Predictor
// - bindings_metrics.cpp: Metrics classes

#include "bindings_common.hpp"

#include <cstdlib>
#include <string>

#if defined(_WIN32)
#define RESOLVE_IS_WIN32 1
#else
#define RESOLVE_IS_WIN32 0
#endif

namespace {

// Mirror of resolve_core/__init__.py::configure_cuda_allocator. The Python
// helper runs at module import (pre-torch); this C++ entry point exists so
// users who imported torch first can still call it explicitly. It is
// late-bound: if the CUDA allocator has already initialized, the new env var
// has no effect on the active allocator config. Documented as such on the
// Python wrapper.
std::string configure_cuda_allocator_impl(bool force) {
    std::string base = "garbage_collection_threshold:0.8,max_split_size_mb:256";
    if constexpr (!RESOLVE_IS_WIN32) {
        base = "expandable_segments:True," + base;
    }

    const char* existing = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (force || existing == nullptr || existing[0] == '\0') {
#if defined(_WIN32)
        _putenv_s("PYTORCH_CUDA_ALLOC_CONF", base.c_str());
#else
        setenv("PYTORCH_CUDA_ALLOC_CONF", base.c_str(), 1);
#endif
        return base;
    }
    return std::string(existing);
}

} // namespace

NB_MODULE(_resolve_core, m) {
    m.doc() = "RESOLVE C++ core library for species-composition based prediction";

    // libtorch holds global state (type registrations, dispatch keys) via shared_ptr
    // that outlives nanobind's module cleanup at interpreter shutdown. This causes
    // false-positive "leaked N types / N functions" warnings. Instance leaks are
    // fixed by returning config objects by value (not reference) in property bindings.
    nb::set_leak_warnings(false);

    // Register all bindings from split modules
    register_enums(m);
    register_types(m);
    register_dataset(m);
    register_model(m);
    register_trainer(m);
    register_metrics(m);
    register_pretraining(m);
    register_fuzzy(m);

    // Platform-aware PYTORCH_CUDA_ALLOC_CONF setter. The primary surface is
    // resolve_core.configure_cuda_allocator() in the Python __init__ which
    // runs at module import (pre-torch). This C++ entry point is the
    // late-bound fallback for users who imported torch first; in that case
    // the allocator has already initialized and changing the env var no
    // longer affects allocator behavior. Returns the resulting config
    // string for logging.
    m.def(
        "_configure_cuda_allocator_native",
        &configure_cuda_allocator_impl,
        nb::arg("force") = false,
        "Set PYTORCH_CUDA_ALLOC_CONF if unset (or force-set). Linux/macOS\n"
        "prepend expandable_segments:True; Windows omits it. Returns the\n"
        "active value. Use resolve_core.configure_cuda_allocator() instead;\n"
        "this native shim is late-bound and cannot rescue allocators that\n"
        "have already initialized."
    );

    // Top-level helper: cap the PyTorch CUDA caching allocator at a fraction
    // of device VRAM. Standalone counterpart to TrainConfig.vram_fraction for
    // users running Predictor-only workflows or wanting to apply the cap
    // before constructing any RESOLVE object.
    m.def(
        "set_vram_fraction",
        [](double fraction, int device_index) {
            resolve::set_vram_fraction(fraction, device_index);
        },
        nb::arg("fraction"),
        nb::arg("device_index") = -1,
        "Cap the PyTorch CUDA caching allocator at `fraction` of device VRAM.\n"
        "fraction must be in (0, 1]; 1.0 disables the cap. device_index = -1\n"
        "uses the current CUDA device. No-op on CPU-only builds or when no\n"
        "CUDA device is present."
    );

    // Version
    m.attr("__version__") = resolve::VERSION;
}
