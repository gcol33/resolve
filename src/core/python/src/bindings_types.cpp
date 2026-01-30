#include "bindings_common.hpp"

void register_types(nb::module_& m) {
    // Role Mapping and Dataset Configuration
    nb::class_<resolve::RoleMapping>(m, "RoleMapping")
        .def(nb::init<>())
        .def_rw("plot_id", &resolve::RoleMapping::plot_id)
        .def_rw("species_id", &resolve::RoleMapping::species_id)
        .def_rw("abundance", &resolve::RoleMapping::abundance)
        .def_rw("longitude", &resolve::RoleMapping::longitude)
        .def_rw("latitude", &resolve::RoleMapping::latitude)
        .def_rw("genus", &resolve::RoleMapping::genus)
        .def_rw("family", &resolve::RoleMapping::family)
        .def_rw("covariates", &resolve::RoleMapping::covariates)
        .def_rw("targets", &resolve::RoleMapping::targets)
        .def("has_coordinates", &resolve::RoleMapping::has_coordinates)
        .def("has_taxonomy", &resolve::RoleMapping::has_taxonomy)
        .def("has_abundance", &resolve::RoleMapping::has_abundance);

    nb::class_<resolve::TargetSpec>(m, "TargetSpec")
        .def(nb::init<>())
        .def_rw("column_name", &resolve::TargetSpec::column_name)
        .def_rw("target_name", &resolve::TargetSpec::target_name)
        .def_rw("task", &resolve::TargetSpec::task)
        .def_rw("transform", &resolve::TargetSpec::transform)
        .def_rw("num_classes", &resolve::TargetSpec::num_classes)
        .def_rw("weight", &resolve::TargetSpec::weight)
        .def_static("regression", &resolve::TargetSpec::regression,
                    nb::arg("column"), nb::arg("transform") = resolve::TransformType::None)
        .def_static("classification", &resolve::TargetSpec::classification,
                    nb::arg("column"), nb::arg("num_classes"));

    nb::class_<resolve::DatasetConfig>(m, "DatasetConfig")
        .def(nb::init<>())
        .def_rw("species_encoding", &resolve::DatasetConfig::species_encoding)
        .def_rw("hash_dim", &resolve::DatasetConfig::hash_dim)
        .def_rw("top_k", &resolve::DatasetConfig::top_k)
        .def_rw("top_k_species", &resolve::DatasetConfig::top_k_species)
        .def_rw("selection", &resolve::DatasetConfig::selection)
        .def_rw("representation", &resolve::DatasetConfig::representation)
        .def_rw("normalization", &resolve::DatasetConfig::normalization)
        .def_rw("aggregation", &resolve::DatasetConfig::aggregation)
        .def_rw("track_unknown_fraction", &resolve::DatasetConfig::track_unknown_fraction)
        .def_rw("track_unknown_count", &resolve::DatasetConfig::track_unknown_count)
        .def_rw("use_taxonomy", &resolve::DatasetConfig::use_taxonomy);

    // Configuration structs
    nb::class_<resolve::TargetConfig>(m, "TargetConfig")
        .def(nb::init<>())
        .def_rw("name", &resolve::TargetConfig::name)
        .def_rw("task", &resolve::TargetConfig::task)
        .def_rw("transform", &resolve::TargetConfig::transform)
        .def_rw("num_classes", &resolve::TargetConfig::num_classes)
        .def_rw("weight", &resolve::TargetConfig::weight)
        .def_rw("class_weights", &resolve::TargetConfig::class_weights);

    nb::class_<resolve::ResolveSchema>(m, "ResolveSchema")
        .def(nb::init<>())
        .def_rw("n_plots", &resolve::ResolveSchema::n_plots)
        .def_rw("n_species", &resolve::ResolveSchema::n_species)
        .def_rw("n_species_vocab", &resolve::ResolveSchema::n_species_vocab)
        .def_rw("has_coordinates", &resolve::ResolveSchema::has_coordinates)
        .def_rw("has_abundance", &resolve::ResolveSchema::has_abundance)
        .def_rw("has_taxonomy", &resolve::ResolveSchema::has_taxonomy)
        .def_rw("n_genera", &resolve::ResolveSchema::n_genera)
        .def_rw("n_families", &resolve::ResolveSchema::n_families)
        .def_rw("n_genera_vocab", &resolve::ResolveSchema::n_genera_vocab)
        .def_rw("n_families_vocab", &resolve::ResolveSchema::n_families_vocab)
        .def_rw("covariate_names", &resolve::ResolveSchema::covariate_names)
        .def_rw("targets", &resolve::ResolveSchema::targets)
        .def_rw("track_unknown_fraction", &resolve::ResolveSchema::track_unknown_fraction)
        .def_rw("track_unknown_count", &resolve::ResolveSchema::track_unknown_count);

    // Alias for backwards compatibility
    m.attr("SpaccSchema") = m.attr("ResolveSchema");

    nb::class_<resolve::ModelConfig>(m, "ModelConfig")
        .def(nb::init<>())
        .def_rw("species_encoding", &resolve::ModelConfig::species_encoding)
        .def_rw("uses_explicit_vector", &resolve::ModelConfig::uses_explicit_vector)
        .def_rw("hash_dim", &resolve::ModelConfig::hash_dim)
        .def_rw("species_embed_dim", &resolve::ModelConfig::species_embed_dim)
        .def_rw("genus_emb_dim", &resolve::ModelConfig::genus_emb_dim)
        .def_rw("family_emb_dim", &resolve::ModelConfig::family_emb_dim)
        .def_rw("top_k", &resolve::ModelConfig::top_k)
        .def_rw("top_k_species", &resolve::ModelConfig::top_k_species)
        .def_rw("n_taxonomy_slots", &resolve::ModelConfig::n_taxonomy_slots)
        .def_rw("hidden_dims", &resolve::ModelConfig::hidden_dims)
        .def_rw("dropout", &resolve::ModelConfig::dropout);

    nb::class_<resolve::TrainConfig>(m, "TrainConfig")
        .def(nb::init<>())
        .def_rw("batch_size", &resolve::TrainConfig::batch_size)
        .def_rw("max_epochs", &resolve::TrainConfig::max_epochs)
        .def_rw("patience", &resolve::TrainConfig::patience)
        .def_rw("lr", &resolve::TrainConfig::lr)
        .def_rw("weight_decay", &resolve::TrainConfig::weight_decay)
        .def_rw("phase_boundaries", &resolve::TrainConfig::phase_boundaries)
        .def_rw("loss_config", &resolve::TrainConfig::loss_config)
        .def_rw("lr_scheduler", &resolve::TrainConfig::lr_scheduler)
        .def_rw("lr_step_size", &resolve::TrainConfig::lr_step_size)
        .def_rw("lr_gamma", &resolve::TrainConfig::lr_gamma)
        .def_rw("lr_min", &resolve::TrainConfig::lr_min);

    // Result structs
    nb::class_<resolve::TrainResult>(m, "TrainResult")
        .def(nb::init<>())
        .def_ro("best_epoch", &resolve::TrainResult::best_epoch)
        .def_ro("final_metrics", &resolve::TrainResult::final_metrics)
        .def_ro("train_loss_history", &resolve::TrainResult::train_loss_history)
        .def_ro("test_loss_history", &resolve::TrainResult::test_loss_history)
        .def_ro("train_time_seconds", &resolve::TrainResult::train_time_seconds)
        .def_ro("resumed_from_epoch", &resolve::TrainResult::resumed_from_epoch);

    nb::class_<resolve::ResolvePredictions>(m, "ResolvePredictions")
        .def(nb::init<>())
        .def_prop_ro("predictions", [](const resolve::ResolvePredictions& p) {
            return tensor_map_to_dict(p.predictions);
        })
        .def_ro("plot_ids", &resolve::ResolvePredictions::plot_ids)
        .def_ro("latent", &resolve::ResolvePredictions::latent);

    m.attr("SpaccPredictions") = m.attr("ResolvePredictions");

    nb::class_<resolve::Scalers>(m, "Scalers")
        .def(nb::init<>())
        .def_rw("continuous_mean", &resolve::Scalers::continuous_mean)
        .def_rw("continuous_scale", &resolve::Scalers::continuous_scale);
}
