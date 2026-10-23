// The _resolve_core extension module. The bindings are split by area:
// - bindings_enums.cpp: enums
// - bindings_types.cpp: configuration structs and result types
// - bindings_dataset.cpp: ResolveDataset and species encoding
// - bindings_model.cpp: ResolveModel
// - bindings_trainer.cpp: Trainer and Predictor
// - bindings_metrics.cpp: metrics
// - bindings_pretraining.cpp: self-supervised pretraining
// - bindings_fuzzy.cpp: the fuzzy string index
// - bindings_suite.cpp: model suites, their manifest and results

#include "bindings_common.hpp"
#include "resolve/io_retry.hpp"

#include <string>

NB_MODULE(_resolve_core, m) {
    m.doc() = "RESOLVE C++ core library for species-composition based prediction";

    // libtorch holds global state (type registrations, dispatch keys) via shared_ptr
    // that outlives nanobind's module cleanup at interpreter shutdown. This causes
    // false-positive "leaked N types / N functions" warnings. Instance leaks are
    // fixed by returning config objects by value (not reference) in property bindings.
    nb::set_leak_warnings(false);

    // A storage read or write that failed after the engine's own retries
    // (io::with_retry): an OSError, so `except OSError` catches it.
    nb::exception<resolve::io::IOError>(m, "StorageError", PyExc_OSError);

    register_enums(m);
    register_types(m);
    register_dataset(m);
    register_model(m);
    register_trainer(m);
    register_metrics(m);
    register_pretraining(m);
    register_fuzzy(m);
    register_suite(m);

    m.def("default_cuda_alloc_conf", &resolve::default_cuda_alloc_conf,
          "The PYTORCH_CUDA_ALLOC_CONF value RESOLVE sets: garbage collection at\n"
          "0.8 and a 256 MB split limit, plus expandable segments off Windows.");

    // Sets the variable in the C runtime's environment; resolve_core's
    // configure_cuda_allocator() calls this and mirrors the result into
    // os.environ.
    m.def("_configure_cuda_allocator", &resolve::configure_cuda_allocator,
          nb::arg("force") = false);

    // Cap the PyTorch CUDA caching allocator at a fraction of device VRAM: the
    // standalone counterpart of TrainConfig.vram_fraction for Predictor-only
    // work or before any RESOLVE object exists.
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

    // Pin libtorch's host thread pools, which avoids worker-thread teardown
    // races on Windows; harmless elsewhere. <=0 keeps libtorch's default for
    // the corresponding pool.
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

    // Windows crash hardening: an unhandled native fault in a headless worker
    // becomes an immediate TerminateProcess with the fault's NTSTATUS, instead
    // of an indefinite Windows-Error-Reporting / JIT-debugger hang that holds
    // the GPU. No-op off Windows. Installed at import below; exposed so callers
    // can re-arm it or change the shutdown exit code.
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

    // Treat a later native fault as a benign teardown artifact (exit with the
    // shutdown code, not a failure code). Registered with atexit() in
    // resolve_core/__init__.py, so a clean interpreter shutdown after a
    // successful run is not reported as a crash.
    m.def(
        "_signal_work_complete",
        []() { resolve::signal_work_complete(); },
        "Mark all engine work complete (atexit hook; see __init__.py)."
    );

    // Arm the crash handler as soon as resolve_core is imported, so a training
    // worker is hardened before Trainer::fit() ever runs. Shutdown code 0: a
    // clean interpreter shutdown (after _signal_work_complete via atexit)
    // exits 0; a mid-run native fault exits with its NTSTATUS.
    resolve::install_crash_handler(0);

    m.attr("__version__") = resolve::VERSION;
}
