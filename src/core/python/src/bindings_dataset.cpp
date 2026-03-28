#include "bindings_common.hpp"
#include "resolve/species_encoding.hpp"

void register_dataset(nb::module_& m) {
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
        .def_prop_ro("coordinates", &resolve::ResolveDataset::coordinates)
        .def_prop_ro("covariates", &resolve::ResolveDataset::covariates)
        .def_prop_ro("hash_embedding", &resolve::ResolveDataset::hash_embedding)
        .def_prop_ro("species_ids", &resolve::ResolveDataset::species_ids)
        .def_prop_ro("species_vector", &resolve::ResolveDataset::species_vector)
        .def_prop_ro("genus_ids", &resolve::ResolveDataset::genus_ids)
        .def_prop_ro("family_ids", &resolve::ResolveDataset::family_ids)
        .def_prop_ro("unknown_fraction", &resolve::ResolveDataset::unknown_fraction)
        .def_prop_ro("unknown_count", &resolve::ResolveDataset::unknown_count)
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
        .def_prop_ro("raw_species_ids", &resolve::ResolveDataset::raw_species_ids)
        .def_prop_ro("raw_weights", &resolve::ResolveDataset::raw_weights)
        .def_prop_ro("plot_offsets", &resolve::ResolveDataset::plot_offsets)
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
        .def("transform", &resolve::RankPoolEncoder::transform)
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
