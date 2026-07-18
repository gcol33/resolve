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

namespace {

// Mirror of resolve_core/__init__.py::configure_cuda_allocator. The Python
// helper runs at module import (pre-torch); this C++ entry point exists so
// users who imported torch first can still call it explicitly. It is
// late-bound: if the CUDA allocator has already initialized, the new env var
// has no effect on the active allocator config. Documented as such on the
// Python wrapper.
std::string configure_cuda_allocator_impl(bool force) {
    // Single source in resolve/gpu.hpp (shared with the C-ABI native shim).
    std::string base = resolve::default_cuda_alloc_conf();

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

    // Pin libtorch's host thread pools. Standalone counterpart used at startup
    // to avoid worker-thread teardown races on Windows (issue #18); harmless
    // elsewhere. <=0 keeps libtorch's default for the corresponding pool.
    m.def(
        "set_thread_pools",
        [](int intraop_threads, int interop_threads) {
            resolve::set_thread_pools(intraop_threads, interop_threads);
        },
        nb::arg("intraop_threads"),
        nb::arg("interop_threads") = -1,
        "Pin libtorch's intra-op / inter-op thread pools (<=0 keeps the\n"
        "default). Best-effort; call at startup before the first op."
    );

    // Windows crash hardening (issue #19): convert an unhandled native fault in
    // a headless training worker into an immediate TerminateProcess with the
    // fault's NTSTATUS, instead of an indefinite Windows-Error-Reporting /
    // JIT-debugger (vsjitdebugger) hang that holds the GPU and stalls the
    // batch. No-op off Windows. Auto-installed at import below; also exposed so
    // callers can re-arm or adjust the shutdown exit code explicitly.
    m.def(
        "install_crash_handler",
        [](int shutdown_exit_code) {
            resolve::install_crash_handler(shutdown_exit_code);
        },
        nb::arg("shutdown_exit_code") = 0,
        "Install the Windows unhandled-exception filter that fails fast via\n"
        "TerminateProcess instead of hanging on the JIT debugger. No-op off\n"
        "Windows. Idempotent."
    );

    // Internal: flip the crash handler to treat a subsequent native fault as a
    // benign teardown artifact (exit with the shutdown code, not a failure
    // code). Registered with atexit() in resolve_core/__init__.py so a clean
    // interpreter shutdown after a successful run is not misreported as a crash.
    m.def(
        "_signal_work_complete",
        []() { resolve::signal_work_complete(); },
        "Mark all engine work complete (atexit hook; see __init__.py)."
    );

    // Arm the crash handler as soon as resolve_core is imported, so a training
    // worker is hardened before Trainer::fit() ever runs (issue #19). Default
    // shutdown code 0: a clean interpreter shutdown (after _signal_work_complete
    // via atexit) exits 0; a mid-run native fault exits with its NTSTATUS.
    resolve::install_crash_handler(0);

    // Version
    m.attr("__version__") = resolve::VERSION;
}
