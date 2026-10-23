#include "bindings_common.hpp"
#include "bindings_plot_inputs.hpp"
#include "resolve/pretraining.hpp"
#include "resolve/vae.hpp"

namespace {

// Every PlotInputs field the call passed, all of them optional: the bundle a
// pretext task that reads a subset of the fields takes.
resolve::PlotInputs optional_plot_inputs(CallArgs& call) {
    resolve::PlotInputs inputs = call.plot_inputs(/*with_continuous=*/false);
    nb::object continuous = call.take("continuous");
    if (!continuous.is_none()) {
        inputs.continuous = unpack_required_tensor(continuous, "continuous");
    }
    return inputs;
}

// The pretrain() method of a pretrainer whose objective masks the continuous
// block: the model inputs in PlotInputs field order (continuous required) or
// by keyword. The GIL is released around the multi-epoch loop.
template <typename Pretrainer>
auto continuous_pretrain() {
    return [](Pretrainer& self, nb::args args, nb::kwargs kwargs) {
        CallArgs call("pretrain", args, kwargs, plot_input_names());
        const auto inputs = call.plot_inputs(/*with_continuous=*/true);
        call.finish();
        nb::gil_scoped_release nogil;
        return self.pretrain(inputs);
    };
}

const char* continuous_pretrain_doc() {
    return persistent_doc(
        "pretrain(" + argument_list(plot_input_names(), 1) + ") -> PretrainResult\n\n"
        "Pretrain on the plots' model inputs, positional in this order or by "
        "keyword; continuous is required. Every input the encoder reads is "
        "corrupted in the pretext view, the species-ID, pool and categorical "
        "inputs included.");
}

}  // namespace

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
        // Seeds the dedicated pretraining RNG: a fixed value reproduces the run
        // and leaves the global torch RNG stream untouched.
        .def_rw("seed", &resolve::PretrainConfig::seed)
        .def_rw("ema_decay", &resolve::PretrainConfig::ema_decay)
        .def_rw("ema_decay_end", &resolve::PretrainConfig::ema_decay_end)
        .def_rw("predictor_hidden_dim", &resolve::PretrainConfig::predictor_hidden_dim)
        .def_rw("predictor_n_layers", &resolve::PretrainConfig::predictor_n_layers)
        .def_rw("predictor_dropout", &resolve::PretrainConfig::predictor_dropout)
        .def_rw("corruption_rate", &resolve::PretrainConfig::corruption_rate)
        .def_rw("temperature", &resolve::PretrainConfig::temperature)
        .def_rw("projection_dim", &resolve::PretrainConfig::projection_dim)
        .def_rw("mixup_alpha", &resolve::PretrainConfig::mixup_alpha);

    // PretrainResult
    nb::class_<resolve::PretrainResult>(m, "PretrainResult")
        .def_ro("loss_history", &resolve::PretrainResult::loss_history)
        .def_ro("total_time_seconds", &resolve::PretrainResult::total_time_seconds)
        .def_ro("epochs_completed", &resolve::PretrainResult::epochs_completed);

    // JEPAPretrainer
    nb::class_<resolve::JEPAPretrainer>(m, "JEPAPretrainer")
        .def(nb::init<resolve::ResolveModel, const resolve::PretrainConfig&>(),
             nb::arg("model"), nb::arg("config") = resolve::PretrainConfig{})
        .def("pretrain", continuous_pretrain<resolve::JEPAPretrainer>(),
             continuous_pretrain_doc());

    // SCARFPretrainer
    nb::class_<resolve::SCARFPretrainer>(m, "SCARFPretrainer")
        .def(nb::init<resolve::ResolveModel, const resolve::PretrainConfig&>(),
             nb::arg("model"), nb::arg("config") = resolve::PretrainConfig{})
        .def("pretrain", continuous_pretrain<resolve::SCARFPretrainer>(),
             continuous_pretrain_doc());

    // Masked-species (MLM) pretraining of a transformer-encoded model
    nb::class_<resolve::MLMPretrainConfig>(m, "MLMPretrainConfig")
        .def(nb::init<>())
        .def_rw("mask_prob", &resolve::MLMPretrainConfig::mask_prob)
        .def_rw("pretrain_epochs", &resolve::MLMPretrainConfig::pretrain_epochs)
        .def_rw("pretrain_lr", &resolve::MLMPretrainConfig::pretrain_lr)
        .def_rw("pretrain_weight_decay", &resolve::MLMPretrainConfig::pretrain_weight_decay)
        .def_rw("batch_size", &resolve::MLMPretrainConfig::batch_size)
        .def_rw("seed", &resolve::MLMPretrainConfig::seed);

    nb::class_<resolve::MaskedSpeciesPretrainer>(m, "MaskedSpeciesPretrainer")
        .def(nb::init<resolve::ResolveModel, const resolve::MLMPretrainConfig&>(),
             nb::arg("model"), nb::arg("config") = resolve::MLMPretrainConfig{},
             "Masked-species pretraining of a species_encoding = transformer "
             "model's species encoder, in place.")
        .def("pretrain",
             [](resolve::MaskedSpeciesPretrainer& self, nb::args args, nb::kwargs kwargs) {
                 CallArgs call("pretrain", args, kwargs, {});
                 const auto inputs = optional_plot_inputs(call);
                 call.finish();
                 nb::gil_scoped_release nogil;
                 return self.pretrain(inputs);
             },
             persistent_doc(
                 "pretrain(*, " + argument_list(plot_input_names()) + ") -> PretrainResult\n\n"
                 "Pretrain on the plots' model inputs, by keyword. Reads species_ids "
                 "and pool_mask (required), pool_genus_ids, pool_family_ids and "
                 "pool_weights; the other fields are accepted and unused."));

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
        .def_rw("pretrain_weight_decay", &resolve::VAEConfig::pretrain_weight_decay)
        .def_rw("batch_size", &resolve::VAEConfig::batch_size)
        // Seeds the shuffle and the reparameterization noise; see
        // PretrainConfig.seed.
        .def_rw("seed", &resolve::VAEConfig::seed);

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
        .def("pretrain",
             [](resolve::VAEPretrainer& self, nb::args args, nb::kwargs kwargs) {
                 CallArgs call("pretrain", args, kwargs, {"species_vector"});
                 const auto inputs = optional_plot_inputs(call);
                 call.finish();
                 nb::gil_scoped_release nogil;
                 return self.pretrain(inputs);
             },
             persistent_doc(
                 "pretrain(species_vector, *, " +
                 argument_list(plot_input_names()) + ") -> VAEPretrainResult\n\n"
                 "Pretrain the VAE on the plots' species vectors (species_vector, "
                 "positional or by keyword, required). The other model inputs are "
                 "accepted and unused."))
        .def("get_projection_weights", [](resolve::VAEPretrainer& self) {
            auto weights = self.vae()->get_projection_weights();
            return nb::steal(THPVariable_Wrap(weights));
        })
        .def("get_latent_dim", [](resolve::VAEPretrainer& self) {
            return self.vae()->latent_dim();
        });
}
