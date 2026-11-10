// Fine-tuning (fine_tune.hpp): the parent a model starts from, the warm-start
// settings and the record a fine-tuned checkpoint keeps.

#include "bindings_common.hpp"

#include "resolve/config_registry.hpp"
#include "resolve/fine_tune.hpp"

void register_fine_tune(nb::module_& m) {
    using namespace resolve;

    {
        using Cfg = InitFromConfig;
        auto cls = nb::class_<Cfg>(m, "InitFromConfig",
                                   "Where a warm start reads its parent from (suite_target, "
                                   "suite_seed for a suite directory) and what it carries "
                                   "over (reuse).");
        cls.def(nb::init<>());
        RESOLVE_INIT_FROM_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }

    nb::class_<ParentRecord>(m, "ParentRecord",
                             "The model a fine-tuned run started from, as its checkpoint "
                             "records it (RunMetadata.parent).")
        .def(nb::init<>())
        .def_rw("checkpoint", &ParentRecord::checkpoint)
        .def_rw("sha256", &ParentRecord::sha256)
        .def_rw("bytes", &ParentRecord::bytes)
        .def_rw("resolve_version", &ParentRecord::resolve_version)
        .def_rw("reuse", &ParentRecord::reuse)
        .def_rw("n_tensors", &ParentRecord::n_tensors)
        .def_rw("heads", &ParentRecord::heads)
        .def_rw("suite", &ParentRecord::suite)
        .def_rw("suite_target", &ParentRecord::suite_target)
        .def_rw("suite_seed", &ParentRecord::suite_seed)
        .def_rw("licence", &ParentRecord::licence)
        .def_rw("training_scope", &ParentRecord::training_scope)
        .def("__eq__", [](const ParentRecord& a, const ParentRecord& b) { return a == b; });

    nb::class_<ParentModel>(m, "ParentModel",
                            "A trained model a fine-tuned model starts from: a checkpoint, or a "
                            "member of a suite. Load the fine-tuning dataset with its vocabs() "
                            "and dataset_config(), build the model from its model_config with "
                            "only the head, dropout and freeze settings changed, and call "
                            "Trainer.init_from(parent).")
        .def_static("open", &ParentModel::open, nb::arg("source"),
                    nb::arg("config") = InitFromConfig{},
                    nb::call_guard<nb::gil_scoped_release>(),
                    "Read the parent: a checkpoint path, or a suite directory whose member "
                    "config.suite_target / config.suite_seed names (its size and SHA-256 "
                    "checked against the manifest first).")
        .def_prop_ro("model_config", [](const ParentModel& p) { return ModelConfig(p.model_config()); })
        .def_prop_ro("schema", [](const ParentModel& p) { return ResolveSchema(p.schema()); })
        .def_prop_ro("scalers", [](const ParentModel& p) { return Scalers(p.scalers()); })
        .def_prop_ro("categorical_vocab",
                     [](const ParentModel& p) { return CategoricalVocab(p.categorical_vocab()); })
        .def("vocabs", &ParentModel::vocabs,
             "The vocabularies a fine-tuning dataset is loaded with (ExternalVocabs).")
        .def("dataset_config", &ParentModel::dataset_config,
             "The loader settings a fine-tuning dataset is loaded with.")
        .def_prop_ro("record", [](const ParentModel& p) { return ParentRecord(p.record()); },
                     "The checkpoint's path, SHA-256, size and engine version, and the suite, "
                     "target and seed it came from.");

    m.def("fine_tune_mismatches", &fine_tune_mismatches, nb::arg("parent"),
          nb::arg("model_config"), nb::arg("schema"),
          "Every way a fine-tuned model's config and data schema depart from what its parent "
          "fixes; empty when it reads the plots as the parent does.");
    m.attr("FINE_TUNE_MODEL_FIELDS") = [] {
        nb::list fields;
        for (const char* f : kFineTuneModelFields) fields.append(f);
        return fields;
    }();
}
