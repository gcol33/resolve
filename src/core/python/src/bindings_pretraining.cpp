#include "bindings_common.hpp"
#include "resolve/pretraining.hpp"
#include "resolve/vae.hpp"

void register_pretraining(nb::module_& m) {
    // MaskStrategy enum
    nb::enum_<resolve::MaskStrategy>(m, "MaskStrategy")
        .value("Random", resolve::MaskStrategy::Random)
        .value("Block", resolve::MaskStrategy::Block)
        .value("Structured", resolve::MaskStrategy::Structured)
        .export_values();

    // PretrainConfig
    nb::class_<resolve::PretrainConfig>(m, "PretrainConfig")
        .def(nb::init<>())
        .def_rw("mask_ratio", &resolve::PretrainConfig::mask_ratio)
        .def_rw("mask_strategy", &resolve::PretrainConfig::mask_strategy)
        .def_rw("pretrain_epochs", &resolve::PretrainConfig::pretrain_epochs)
        .def_rw("pretrain_lr", &resolve::PretrainConfig::pretrain_lr)
        .def_rw("pretrain_weight_decay", &resolve::PretrainConfig::pretrain_weight_decay)
        .def_rw("batch_size", &resolve::PretrainConfig::batch_size)
        .def_rw("ema_decay", &resolve::PretrainConfig::ema_decay)
        .def_rw("ema_decay_end", &resolve::PretrainConfig::ema_decay_end)
        .def_rw("predictor_hidden_dim", &resolve::PretrainConfig::predictor_hidden_dim)
        .def_rw("predictor_n_layers", &resolve::PretrainConfig::predictor_n_layers)
        .def_rw("predictor_dropout", &resolve::PretrainConfig::predictor_dropout)
        .def_rw("corruption_rate", &resolve::PretrainConfig::corruption_rate)
        .def_rw("temperature", &resolve::PretrainConfig::temperature)
        .def_rw("projection_dim", &resolve::PretrainConfig::projection_dim);

    // PretrainResult
    nb::class_<resolve::PretrainResult>(m, "PretrainResult")
        .def_ro("loss_history", &resolve::PretrainResult::loss_history)
        .def_ro("total_time_seconds", &resolve::PretrainResult::total_time_seconds)
        .def_ro("epochs_completed", &resolve::PretrainResult::epochs_completed);

    // JEPAPretrainer
    nb::class_<resolve::JEPAPretrainer>(m, "JEPAPretrainer")
        .def(nb::init<resolve::ResolveModel, const resolve::PretrainConfig&>(),
             nb::arg("model"), nb::arg("config") = resolve::PretrainConfig{})
        .def("pretrain", [](resolve::JEPAPretrainer& self,
                           nb::object continuous_obj,
                           nb::object genus_ids_obj,
                           nb::object family_ids_obj,
                           nb::object species_ids_obj,
                           nb::object species_vector_obj) {
            // Guard the unpack (THPVariable_Unpack on None/non-tensor is UB) and
            // release the GIL around the multi-epoch loop so other Python
            // threads run, matching Trainer.fit / Model.forward.
            at::Tensor continuous = unpack_required_tensor(continuous_obj, "continuous");
            at::Tensor genus_ids = unpack_optional_tensor(genus_ids_obj);
            at::Tensor family_ids = unpack_optional_tensor(family_ids_obj);
            at::Tensor species_ids = unpack_optional_tensor(species_ids_obj);
            at::Tensor species_vector = unpack_optional_tensor(species_vector_obj);
            nb::gil_scoped_release release;
            return self.pretrain(continuous, genus_ids, family_ids, species_ids, species_vector);
        }, nb::arg("continuous"),
           nb::arg("genus_ids") = nb::none(),
           nb::arg("family_ids") = nb::none(),
           nb::arg("species_ids") = nb::none(),
           nb::arg("species_vector") = nb::none());

    // SCARFPretrainer
    nb::class_<resolve::SCARFPretrainer>(m, "SCARFPretrainer")
        .def(nb::init<resolve::ResolveModel, const resolve::PretrainConfig&>(),
             nb::arg("model"), nb::arg("config") = resolve::PretrainConfig{})
        .def("pretrain", [](resolve::SCARFPretrainer& self,
                           nb::object continuous_obj,
                           nb::object genus_ids_obj,
                           nb::object family_ids_obj,
                           nb::object species_ids_obj,
                           nb::object species_vector_obj) {
            at::Tensor continuous = unpack_required_tensor(continuous_obj, "continuous");
            at::Tensor genus_ids = unpack_optional_tensor(genus_ids_obj);
            at::Tensor family_ids = unpack_optional_tensor(family_ids_obj);
            at::Tensor species_ids = unpack_optional_tensor(species_ids_obj);
            at::Tensor species_vector = unpack_optional_tensor(species_vector_obj);
            nb::gil_scoped_release release;
            return self.pretrain(continuous, genus_ids, family_ids, species_ids, species_vector);
        }, nb::arg("continuous"),
           nb::arg("genus_ids") = nb::none(),
           nb::arg("family_ids") = nb::none(),
           nb::arg("species_ids") = nb::none(),
           nb::arg("species_vector") = nb::none());

    // =========================================================================
    // VAE Module
    // =========================================================================

    // VAEConfig
    nb::class_<resolve::VAEConfig>(m, "VAEConfig")
        .def(nb::init<>())
        .def_rw("latent_dim", &resolve::VAEConfig::latent_dim)
        .def_rw("encoder_dims", &resolve::VAEConfig::encoder_dims)
        .def_rw("decoder_dims", &resolve::VAEConfig::decoder_dims)
        .def_rw("dropout", &resolve::VAEConfig::dropout)
        .def_rw("kl_weight", &resolve::VAEConfig::kl_weight)
        .def_rw("kl_anneal_epochs", &resolve::VAEConfig::kl_anneal_epochs)
        .def_rw("pretrain_epochs", &resolve::VAEConfig::pretrain_epochs)
        .def_rw("pretrain_lr", &resolve::VAEConfig::pretrain_lr)
        .def_rw("batch_size", &resolve::VAEConfig::batch_size);

    // VAEPretrainResult
    nb::class_<resolve::VAEPretrainResult>(m, "VAEPretrainResult")
        .def_ro("loss_history", &resolve::VAEPretrainResult::loss_history)
        .def_ro("recon_loss_history", &resolve::VAEPretrainResult::recon_loss_history)
        .def_ro("kl_loss_history", &resolve::VAEPretrainResult::kl_loss_history)
        .def_ro("total_time_seconds", &resolve::VAEPretrainResult::total_time_seconds)
        .def_ro("epochs_completed", &resolve::VAEPretrainResult::epochs_completed);

    // VAEPretrainer
    nb::class_<resolve::VAEPretrainer>(m, "VAEPretrainer")
        .def(nb::init<int64_t, const resolve::VAEConfig&>(),
             nb::arg("n_species"), nb::arg("config") = resolve::VAEConfig{})
        .def("pretrain", [](resolve::VAEPretrainer& self,
                           nb::object species_vectors_obj) {
            at::Tensor species_vectors =
                unpack_required_tensor(species_vectors_obj, "species_vectors");
            nb::gil_scoped_release release;
            return self.pretrain(species_vectors);
        }, nb::arg("species_vectors"))
        .def("get_projection_weights", [](resolve::VAEPretrainer& self) {
            auto weights = self.vae()->get_projection_weights();
            return nb::steal(THPVariable_Wrap(weights));
        })
        .def("get_latent_dim", [](resolve::VAEPretrainer& self) {
            return self.vae()->latent_dim();
        });
}
