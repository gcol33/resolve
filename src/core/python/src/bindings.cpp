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

    // Register all bindings from split modules
    register_enums(m);
    register_types(m);
    register_dataset(m);
    register_model(m);
    register_trainer(m);
    register_metrics(m);
    register_pretraining(m);

    // Version
    m.attr("__version__") = resolve::VERSION;
}
