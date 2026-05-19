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
