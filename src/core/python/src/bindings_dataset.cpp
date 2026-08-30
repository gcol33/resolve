#include "bindings_common.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/categorical.hpp"

void register_dataset(nb::module_& m) {
    // Categorical vocabulary — exposed so Predictor users can re-encode
    // raw CSV strings into the training-time codes.
    nb::class_<resolve::CategoricalVocab>(m, "CategoricalVocab")
        .def(nb::init<>())
        .def("fit_column", &resolve::CategoricalVocab::fit_column,
             nb::arg("column_name"), nb::arg("raw_values"))
        .def("encode",
             [](const resolve::CategoricalVocab& self, const std::string& col,
                const std::string& val) { return self.encode(col, val); },
             nb::arg("column_name"), nb::arg("raw_value"))
        .def("encode_batch", &resolve::CategoricalVocab::encode_batch,
             nb::arg("column_names"), nb::arg("raw_values_per_column"))
        .def("vocab_size", &resolve::CategoricalVocab::vocab_size,
             nb::arg("column_name"))
        .def("has_column", &resolve::CategoricalVocab::has_column,
             nb::arg("column_name"))
        .def_prop_ro("column_names", &resolve::CategoricalVocab::column_names)
        .def_prop_ro("vocab_sizes", &resolve::CategoricalVocab::vocab_sizes)
        .def("column_map", &resolve::CategoricalVocab::column_map,
             nb::arg("column_name"));

    nb::class_<resolve::ResolveDataset>(m, "ResolveDataset")
        .def_static("from_csv", &resolve::ResolveDataset::from_csv,
                    nb::arg("header_path"),
                    nb::arg("species_path"),
                    nb::arg("roles"),
                    nb::arg("targets"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    "Load dataset from header CSV and species CSV files")
        // from_csv_with_schema is overloaded on the vocabulary source, so each
        // overload needs an explicit function-pointer cast (a bare &member is
        // ambiguous). nanobind dispatches on the argument type, keeping one
        // Python-level verb.
        .def_static("from_csv_with_schema",
                    static_cast<resolve::ResolveDataset (*)(
                        const std::string&, const std::string&,
                        const resolve::RoleMapping&,
                        const std::vector<resolve::TargetSpec>&,
                        const resolve::ResolveDataset&,
                        const resolve::DatasetConfig&)>(
                        &resolve::ResolveDataset::from_csv_with_schema),
                    nb::arg("header_path"),
                    nb::arg("species_path"),
                    nb::arg("roles"),
                    nb::arg("targets"),
                    nb::arg("schema_source"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    "Load dataset reusing the categorical / taxonomy / species "
                    "vocabularies and classification class mappings from "
                    "`schema_source`. Use for cross-split workflows (leave-one-"
                    "dataset-out, sample efficiency, transfer) where the test "
                    "set must be encoded against the training set's vocab so "
                    "model lookup tables are indexed correctly.")
        .def_static("from_csv_with_schema",
                    static_cast<resolve::ResolveDataset (*)(
                        const std::string&, const std::string&,
                        const resolve::RoleMapping&,
                        const std::vector<resolve::TargetSpec>&,
                        const resolve::ResolveSchema&,
                        const resolve::DatasetConfig&)>(
                        &resolve::ResolveDataset::from_csv_with_schema),
                    nb::arg("header_path"),
                    nb::arg("species_path"),
                    nb::arg("roles"),
                    nb::arg("targets"),
                    nb::arg("schema"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    "Load dataset reusing the vocabularies a CHECKPOINT's schema "
                    "carries (issue #102), so the training CSVs need not exist. "
                    "Raises when the schema has no species vocabulary (a "
                    "pre-#102 checkpoint). Pass predictor.external_vocabs to "
                    "from_csv_with_vocabs instead when the model has "
                    "categorical covariates.")
        .def_static("from_csv_with_vocabs",
                    &resolve::ResolveDataset::from_csv_with_vocabs,
                    nb::arg("header_path"),
                    nb::arg("species_path"),
                    nb::arg("roles"),
                    nb::arg("targets"),
                    nb::arg("vocabs"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    "Load dataset with every vocabulary supplied explicitly "
                    "(ExternalVocabs). The complete form: use with "
                    "predictor.external_vocabs, whose categorical maps live on "
                    "the Predictor rather than the schema.")
        .def_static("from_species_csv", &resolve::ResolveDataset::from_species_csv,
                    nb::arg("species_path"),
                    nb::arg("roles"),
                    nb::arg("targets"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    "Load dataset from a single species CSV file")
        .def_static("from_species_csv_with_schema",
                    &resolve::ResolveDataset::from_species_csv_with_schema,
                    nb::arg("species_path"),
                    nb::arg("roles"),
                    nb::arg("targets"),
                    nb::arg("schema"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    "Single-table counterpart of from_csv_with_schema: reuse a "
                    "checkpoint's vocabularies when scoring a species-only CSV.")
        .def_static("from_species_csv_with_vocabs",
                    &resolve::ResolveDataset::from_species_csv_with_vocabs,
                    nb::arg("species_path"),
                    nb::arg("roles"),
                    nb::arg("targets"),
                    nb::arg("vocabs"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    "Single-table counterpart of from_csv_with_vocabs.")
        // --- In-memory (DataFrame) loaders (issue #22) ---
        // Low-level column entry points: each frame arrives as (names, columns)
        // with every cell already a string (column-major), so the result is
        // identical to loading the equivalent CSV. The high-level `from_pandas`
        // / `from_dataframe` wrappers in resolve_core/__init__.py convert pandas
        // DataFrames to these and dispatch. The conversion (no Python C-API per
        // cell on the C++ side) runs without the GIL where the engine works.
        .def_static("from_columns",
                    [](std::vector<std::string> header_names,
                       std::vector<std::vector<std::string>> header_columns,
                       std::vector<std::string> species_names,
                       std::vector<std::vector<std::string>> species_columns,
                       const resolve::RoleMapping& roles,
                       const std::vector<resolve::TargetSpec>& targets,
                       const resolve::DatasetConfig& config) {
                        resolve::ColumnTable header(std::move(header_names),
                                                    std::move(header_columns));
                        resolve::ColumnTable species(std::move(species_names),
                                                     std::move(species_columns));
                        return resolve::ResolveDataset::from_dataframe(
                            header, species, roles, targets, config);
                    },
                    nb::arg("header_names"), nb::arg("header_columns"),
                    nb::arg("species_names"), nb::arg("species_columns"),
                    nb::arg("roles"), nb::arg("targets"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    nb::call_guard<nb::gil_scoped_release>(),
                    "Load dataset from in-memory header + species column tables "
                    "(string cells, column-major). Identical to from_csv on the "
                    "equivalent CSV; no disk round-trip.")
        .def_static("from_columns_header",
                    [](std::vector<std::string> header_names,
                       std::vector<std::vector<std::string>> header_columns,
                       const std::string& species_path,
                       const resolve::RoleMapping& roles,
                       const std::vector<resolve::TargetSpec>& targets,
                       const resolve::DatasetConfig& config) {
                        resolve::ColumnTable header(std::move(header_names),
                                                    std::move(header_columns));
                        return resolve::ResolveDataset::from_dataframe_header(
                            header, species_path, roles, targets, config);
                    },
                    nb::arg("header_names"), nb::arg("header_columns"),
                    nb::arg("species_path"),
                    nb::arg("roles"), nb::arg("targets"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    nb::call_guard<nb::gil_scoped_release>(),
                    "Load dataset from an in-memory header column table plus a "
                    "species CSV path (the large species table is read once).")
        .def_static("from_species_columns",
                    [](std::vector<std::string> species_names,
                       std::vector<std::vector<std::string>> species_columns,
                       const resolve::RoleMapping& roles,
                       const std::vector<resolve::TargetSpec>& targets,
                       const resolve::DatasetConfig& config) {
                        resolve::ColumnTable species(std::move(species_names),
                                                     std::move(species_columns));
                        return resolve::ResolveDataset::from_species_dataframe(
                            species, roles, targets, config);
                    },
                    nb::arg("species_names"), nb::arg("species_columns"),
                    nb::arg("roles"), nb::arg("targets"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    nb::call_guard<nb::gil_scoped_release>(),
                    "Load dataset from a single in-memory long-format column "
                    "table (the DataFrame analog of from_species_csv).")
        .def_static("from_columns_with_schema",
                    [](std::vector<std::string> header_names,
                       std::vector<std::vector<std::string>> header_columns,
                       std::vector<std::string> species_names,
                       std::vector<std::vector<std::string>> species_columns,
                       const resolve::RoleMapping& roles,
                       const std::vector<resolve::TargetSpec>& targets,
                       const resolve::ResolveDataset& schema_source,
                       const resolve::DatasetConfig& config) {
                        resolve::ColumnTable header(std::move(header_names),
                                                    std::move(header_columns));
                        resolve::ColumnTable species(std::move(species_names),
                                                     std::move(species_columns));
                        return resolve::ResolveDataset::from_dataframe_with_schema(
                            header, species, roles, targets, schema_source, config);
                    },
                    nb::arg("header_names"), nb::arg("header_columns"),
                    nb::arg("species_names"), nb::arg("species_columns"),
                    nb::arg("roles"), nb::arg("targets"),
                    nb::arg("schema_source"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    nb::call_guard<nb::gil_scoped_release>(),
                    "In-memory analog of from_csv_with_schema: reuse the "
                    "schema_source vocabularies / class mappings.")
        .def_static("from_columns_with_vocabs",
                    [](std::vector<std::string> header_names,
                       std::vector<std::vector<std::string>> header_columns,
                       std::vector<std::string> species_names,
                       std::vector<std::vector<std::string>> species_columns,
                       const resolve::RoleMapping& roles,
                       const std::vector<resolve::TargetSpec>& targets,
                       const resolve::ExternalVocabs& vocabs,
                       const resolve::DatasetConfig& config) {
                        resolve::ColumnTable header(std::move(header_names),
                                                    std::move(header_columns));
                        resolve::ColumnTable species(std::move(species_names),
                                                     std::move(species_columns));
                        return resolve::ResolveDataset::from_dataframe_with_vocabs(
                            header, species, roles, targets, vocabs, config);
                    },
                    nb::arg("header_names"), nb::arg("header_columns"),
                    nb::arg("species_names"), nb::arg("species_columns"),
                    nb::arg("roles"), nb::arg("targets"),
                    nb::arg("vocabs"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    nb::call_guard<nb::gil_scoped_release>(),
                    "In-memory analog of from_csv_with_vocabs (issue #102).")
        .def_static("from_species_columns_with_vocabs",
                    [](std::vector<std::string> species_names,
                       std::vector<std::vector<std::string>> species_columns,
                       const resolve::RoleMapping& roles,
                       const std::vector<resolve::TargetSpec>& targets,
                       const resolve::ExternalVocabs& vocabs,
                       const resolve::DatasetConfig& config) {
                        resolve::ColumnTable species(std::move(species_names),
                                                     std::move(species_columns));
                        return resolve::ResolveDataset::from_species_dataframe_with_vocabs(
                            species, roles, targets, vocabs, config);
                    },
                    nb::arg("species_names"), nb::arg("species_columns"),
                    nb::arg("roles"), nb::arg("targets"),
                    nb::arg("vocabs"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    nb::call_guard<nb::gil_scoped_release>(),
                    "In-memory analog of from_species_csv_with_vocabs (issue #102).")
        // Tensor accessors must be wrapped via THPVariable_Wrap; nanobind
        // has no built-in caster for at::Tensor (would produce
        // "Unable to convert function return value to a Python type").
        // Undefined tensors are surfaced as None.
        .def_prop_ro("coordinates", [](const resolve::ResolveDataset& self) {
            const auto& t = self.coordinates();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("covariates", [](const resolve::ResolveDataset& self) {
            const auto& t = self.covariates();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("hash_embedding", [](const resolve::ResolveDataset& self) {
            const auto& t = self.hash_embedding();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("species_ids", [](const resolve::ResolveDataset& self) {
            const auto& t = self.species_ids();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("species_vector", [](const resolve::ResolveDataset& self) {
            const auto& t = self.species_vector();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("genus_ids", [](const resolve::ResolveDataset& self) {
            const auto& t = self.genus_ids();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("family_ids", [](const resolve::ResolveDataset& self) {
            const auto& t = self.family_ids();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("unknown_fraction", [](const resolve::ResolveDataset& self) {
            const auto& t = self.unknown_fraction();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("unknown_count", [](const resolve::ResolveDataset& self) {
            const auto& t = self.unknown_count();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("categorical_ids", [](const resolve::ResolveDataset& self) {
            const auto& t = self.categorical_ids();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        // Pool-style encoder tensors (rank_pool / transformer modes). Wrap
        // via THPVariable_Wrap inside a lambda — see pitfall #6.1 in
        // port.md (def_prop_ro(&accessor) fails for at::Tensor returns).
        .def_prop_ro("pool_genus_ids", [](const resolve::ResolveDataset& self) {
            const auto& t = self.pool_genus_ids();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("pool_family_ids", [](const resolve::ResolveDataset& self) {
            const auto& t = self.pool_family_ids();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("pool_weights", [](const resolve::ResolveDataset& self) {
            const auto& t = self.pool_weights();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("pool_mask", [](const resolve::ResolveDataset& self) {
            const auto& t = self.pool_mask();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("pool_has_cover", [](const resolve::ResolveDataset& self) {
            const auto& t = self.pool_has_cover();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def("has_pool_data", &resolve::ResolveDataset::has_pool_data)
        .def_prop_ro("categorical_vocab",
                     [](const resolve::ResolveDataset& self) -> const resolve::CategoricalVocab& {
                         return self.categorical_vocab();
                     })
        .def_prop_ro("targets", [](const resolve::ResolveDataset& d) {
            return tensor_map_to_dict(d.targets());
        })
        .def_prop_ro("schema", [](const resolve::ResolveDataset& self) -> resolve::ResolveSchema { return self.schema(); })
        .def_prop_ro("plot_ids", &resolve::ResolveDataset::plot_ids)
        .def_prop_ro("species_vocab", &resolve::ResolveDataset::species_vocab)
        .def_prop_ro("n_plots", &resolve::ResolveDataset::n_plots)
        .def_prop_ro("config", [](const resolve::ResolveDataset& self) -> resolve::DatasetConfig { return self.config(); })
        // CUDA hash accessors
        .def("has_raw_species_data", &resolve::ResolveDataset::has_raw_species_data)
        .def_prop_ro("raw_species_ids", [](const resolve::ResolveDataset& self) {
            const auto& t = self.raw_species_ids();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("raw_weights", [](const resolve::ResolveDataset& self) {
            const auto& t = self.raw_weights();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("plot_offsets", [](const resolve::ResolveDataset& self) {
            const auto& t = self.plot_offsets();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("taxonomy_vocab", [](const resolve::ResolveDataset& self) -> const resolve::TaxonomyVocab& {
            return self.taxonomy_vocab();
        })
        // Every vocabulary this dataset fitted, in the carrier the
        // *_with_vocabs loaders take (issue #102). Equivalent to passing the
        // dataset itself to from_csv_with_schema, but usable after its source
        // files are gone.
        .def("external_vocabs", &resolve::ResolveDataset::external_vocabs);

    // Species Encoding helpers
    nb::class_<resolve::TaxonomyVocab>(m, "TaxonomyVocab")
        .def(nb::init<>())
        .def("fit", &resolve::TaxonomyVocab::fit)
        .def("encode_genus", &resolve::TaxonomyVocab::encode_genus)
        .def("encode_family", &resolve::TaxonomyVocab::encode_family)
        .def("n_genera", &resolve::TaxonomyVocab::n_genera)
        .def("n_families", &resolve::TaxonomyVocab::n_families)
        .def("save", &resolve::TaxonomyVocab::save)
        .def_static("load", &resolve::TaxonomyVocab::load);

    nb::class_<resolve::SpeciesRecord>(m, "SpeciesRecord")
        .def(nb::init<>())
        .def_rw("species_id", &resolve::SpeciesRecord::species_id)
        .def_rw("genus", &resolve::SpeciesRecord::genus)
        .def_rw("family", &resolve::SpeciesRecord::family)
        .def_rw("abundance", &resolve::SpeciesRecord::abundance)
        .def_rw("plot_id", &resolve::SpeciesRecord::plot_id);

    nb::class_<resolve::SpeciesVocab>(m, "SpeciesVocab")
        .def(nb::init<>())
        .def_static("from_records", &resolve::SpeciesVocab::from_records,
                    nb::arg("records"), nb::arg("min_count") = 1)
        .def("encode", &resolve::SpeciesVocab::encode)
        .def("size", &resolve::SpeciesVocab::size)
        .def("empty", &resolve::SpeciesVocab::empty)
        .def_prop_ro("species_to_id", &resolve::SpeciesVocab::species_to_id);

    // Per-plot novelty of an assemblage against a fitted vocabulary. Same
    // definition the dataset's unknown_fraction / unknown_count columns use.
    nb::class_<resolve::UnknownSpeciesStats>(m, "UnknownSpeciesStats")
        .def(nb::init<>())
        .def_ro("fraction", &resolve::UnknownSpeciesStats::fraction)
        .def_ro("count", &resolve::UnknownSpeciesStats::count);

    m.def("compute_unknown_species_stats", &resolve::compute_unknown_species_stats,
          nb::arg("records"), nb::arg("plot_ids"), nb::arg("vocab"),
          "Per-plot unknown-species fraction (abundance-weighted) and count, "
          "measured against a fitted SpeciesVocab. Non-zero only where the vocab "
          "was fitted on other data; a vocab fitted on these records covers every "
          "name in them.");

    nb::class_<resolve::RankPoolEncodedData>(m, "RankPoolEncodedData")
        .def(nb::init<>())
        .def_ro("species_ids", &resolve::RankPoolEncodedData::species_ids)
        .def_ro("genus_ids", &resolve::RankPoolEncodedData::genus_ids)
        .def_ro("family_ids", &resolve::RankPoolEncodedData::family_ids)
        .def_ro("weights", &resolve::RankPoolEncodedData::weights)
        .def_ro("mask", &resolve::RankPoolEncodedData::mask)
        .def_ro("has_cover", &resolve::RankPoolEncodedData::has_cover)
        .def_ro("unknown_fraction", &resolve::RankPoolEncodedData::unknown_fraction)
        .def_ro("unknown_count", &resolve::RankPoolEncodedData::unknown_count)
        .def_ro("n_species_vocab", &resolve::RankPoolEncodedData::n_species_vocab)
        .def_ro("n_genera_vocab", &resolve::RankPoolEncodedData::n_genera_vocab)
        .def_ro("n_families_vocab", &resolve::RankPoolEncodedData::n_families_vocab);

    nb::class_<resolve::RankPoolEncoder>(m, "RankPoolEncoder")
        .def(nb::init<resolve::PoolWeighting, int>(),
             nb::arg("weighting") = resolve::PoolWeighting::Log1p,
             nb::arg("min_frequency") = 1)
        .def("fit", &resolve::RankPoolEncoder::fit)
        .def("transform", &resolve::RankPoolEncoder::transform,
             nb::arg("records"), nb::arg("plot_ids"),
             nb::arg("species_cap") = 0,
             nb::arg("has_abundance_column") = false)
        .def("is_fitted", &resolve::RankPoolEncoder::is_fitted)
        .def("n_species_vocab", &resolve::RankPoolEncoder::n_species_vocab)
        .def("n_genera_vocab", &resolve::RankPoolEncoder::n_genera_vocab)
        .def("n_families_vocab", &resolve::RankPoolEncoder::n_families_vocab)
        .def_prop_ro("species_vocab", &resolve::RankPoolEncoder::species_vocab)
        .def_prop_ro("taxonomy_vocab", &resolve::RankPoolEncoder::taxonomy_vocab);

    nb::class_<resolve::EmbeddingEncodedData>(m, "EmbeddingEncodedData")
        .def(nb::init<>())
        .def_ro("species_ids", &resolve::EmbeddingEncodedData::species_ids)
        .def_ro("genus_ids", &resolve::EmbeddingEncodedData::genus_ids)
        .def_ro("family_ids", &resolve::EmbeddingEncodedData::family_ids)
        .def_ro("unknown_fraction", &resolve::EmbeddingEncodedData::unknown_fraction)
        .def_ro("unknown_count", &resolve::EmbeddingEncodedData::unknown_count)
        .def_ro("n_species_vocab", &resolve::EmbeddingEncodedData::n_species_vocab)
        .def_ro("n_genera_vocab", &resolve::EmbeddingEncodedData::n_genera_vocab)
        .def_ro("n_families_vocab", &resolve::EmbeddingEncodedData::n_families_vocab);

    nb::class_<resolve::EmbeddingEncoder>(m, "EmbeddingEncoder")
        .def(nb::init<int, int, resolve::SelectionMode>(),
             nb::arg("top_k_species") = 10,
             nb::arg("top_k_taxonomy") = 3,
             nb::arg("selection") = resolve::SelectionMode::Top)
        .def("fit", &resolve::EmbeddingEncoder::fit)
        .def("transform", &resolve::EmbeddingEncoder::transform)
        .def("is_fitted", &resolve::EmbeddingEncoder::is_fitted)
        .def("n_species_vocab", &resolve::EmbeddingEncoder::n_species_vocab)
        .def("n_genera_vocab", &resolve::EmbeddingEncoder::n_genera_vocab)
        .def("n_families_vocab", &resolve::EmbeddingEncoder::n_families_vocab);

    // Vocabularies fitted at training time, in the form the ResolveDataset
    // *_with_vocabs loaders accept (issue #102). Get one from
    // `predictor.external_vocabs` (complete, including the categorical maps) or
    // `dataset.external_vocabs()`. Registered here, after the TaxonomyVocab /
    // CategoricalVocab classes it holds.
    nb::class_<resolve::ExternalVocabs>(m, "ExternalVocabs")
        .def(nb::init<>())
        .def_rw("species_vocab", &resolve::ExternalVocabs::species_vocab)
        .def_rw("taxonomy", &resolve::ExternalVocabs::taxonomy)
        .def_rw("categorical", &resolve::ExternalVocabs::categorical)
        .def_rw("targets", &resolve::ExternalVocabs::targets);

    m.def("external_vocabs_from_schema", &resolve::external_vocabs_from_schema,
          nb::arg("schema"),
          "Rebuild the vocabularies a checkpoint's schema carries. The "
          "categorical string -> code maps are not on the schema; take them "
          "from Predictor.categorical_vocab (or use Predictor.external_vocabs, "
          "which folds them in).");

    m.def("dataset_config_from_checkpoint", &resolve::dataset_config_from_checkpoint,
          nb::arg("schema"), nb::arg("model_config"),
          "Reassemble the loading-side DatasetConfig a checkpoint was built "
          "with. species_encoding / hash_dim / top_k come from the ModelConfig "
          "(they size the model); everything else the loader consumed comes "
          "from the schema. use_cuda_hash is deliberately not restored.");

    m.def("effective_selection", &resolve::effective_selection, nb::arg("config"),
          "The species selection a dataset built under this config actually "
          "applies. hash and embed always select (their widths are per-plot "
          "budgets: top_k and top_k_species). rank_pool / transformer / sparse "
          "select only when species_budget gives them a budget, and report All "
          "otherwise, because they encode every record. This is the value the "
          "dataset publishes on its schema, so a checkpoint cannot claim a "
          "selection the run never made.");
}
