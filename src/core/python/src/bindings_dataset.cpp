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
        .def_static("from_species_csv", &resolve::ResolveDataset::from_species_csv,
                    nb::arg("species_path"),
                    nb::arg("roles"),
                    nb::arg("targets"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    "Load dataset from a single species CSV file")
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
        });

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

    nb::class_<resolve::EncodedSpecies>(m, "EncodedSpecies")
        .def(nb::init<>())
        .def_ro("hash_embedding", &resolve::EncodedSpecies::hash_embedding)
        .def_ro("genus_ids", &resolve::EncodedSpecies::genus_ids)
        .def_ro("family_ids", &resolve::EncodedSpecies::family_ids)
        .def_ro("unknown_fraction", &resolve::EncodedSpecies::unknown_fraction)
        .def_ro("unknown_count", &resolve::EncodedSpecies::unknown_count)
        .def_ro("species_vector", &resolve::EncodedSpecies::species_vector)
        .def_ro("species_ids", &resolve::EncodedSpecies::species_ids)
        .def_ro("plot_ids", &resolve::EncodedSpecies::plot_ids);

    nb::class_<resolve::SpeciesVocab>(m, "SpeciesVocab")
        .def(nb::init<>())
        .def_static("from_records", &resolve::SpeciesVocab::from_records,
                    nb::arg("records"), nb::arg("min_count") = 1)
        .def("encode", &resolve::SpeciesVocab::encode)
        .def("size", &resolve::SpeciesVocab::size)
        .def("empty", &resolve::SpeciesVocab::empty)
        .def_prop_ro("species_to_id", &resolve::SpeciesVocab::species_to_id);

    nb::class_<resolve::RankPoolEncodedData>(m, "RankPoolEncodedData")
        .def(nb::init<>())
        .def_ro("species_ids", &resolve::RankPoolEncodedData::species_ids)
        .def_ro("genus_ids", &resolve::RankPoolEncodedData::genus_ids)
        .def_ro("family_ids", &resolve::RankPoolEncodedData::family_ids)
        .def_ro("weights", &resolve::RankPoolEncodedData::weights)
        .def_ro("mask", &resolve::RankPoolEncodedData::mask)
        .def_ro("has_cover", &resolve::RankPoolEncodedData::has_cover)
        .def_ro("unknown_fraction", &resolve::RankPoolEncodedData::unknown_fraction)
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
             nb::arg("species_cap") = 0)
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
}
