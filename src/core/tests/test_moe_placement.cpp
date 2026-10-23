// Mixture of experts across every encoding, and one meaning for moe_routing.
//
// Before this, `moe_routing` was two different architectures depending on
// `species_encoding`: hash built a dedicated encoder whose mixture REPLACED the
// last MLP stages, while embed / sparse / rank_pool / transformer got a
// dim-preserving mixture bolted onto the finished latent, and the adapter
// architectures were refused outright. Nothing in the suite constructed a
// ResolveModel with routing on, so none of that was observable in a test.
//
// The contract these tests pin:
//
//   * MoEPlacement::Tail is available to all five species encodings. The
//     mixture is the encoder's final stage: the backbone takes hidden_dims
//     minus its last two widths and the latent is hidden_dims.back().
//   * The Tail parameter names are "backbone" + "moe" under the encoder, which
//     is what a hash-mode MoE checkpoint has always carried, so those
//     checkpoints keep loading. A run without routing still writes "mlp".
//   * The auxiliary load-balancing loss reaches the trainer from the tail, for
//     every encoding -- not just from post_moe_.
//   * MoEPlacement::Post keeps the latent width and is what an encoder with no
//     MLP tail (adapter architectures, TraitNet) uses. Asking those for Tail
//     is refused with a message naming the placement that works.
//   * TabM and a mixture both claim the same tail, so asking for both is
//     refused rather than silently dropping TabM.
//   * get_gate_probs reports real probabilities for the encoders its signature
//     can drive, and says so for the ones it cannot, instead of returning an
//     undefined tensor that reads as "MoE is off".

#include <catch2/catch_test_macros.hpp>

#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <cmath>
#include <set>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

// Small but genuinely splittable: with four widths the backbone keeps the first
// two and the mixture produces the last, so a tail that quietly ran the whole
// MLP would report a different latent.
const std::vector<int64_t> kHidden = {24, 20, 16, 12};

ModelConfig base_model(SpeciesEncodingMode mode) { return small_encoder_config(mode, kHidden); }

ModelConfig moe_model(SpeciesEncodingMode mode, MoEPlacement placement,
                      MoERoutingType routing = MoERoutingType::Soft) {
    ModelConfig cfg = base_model(mode);
    cfg.moe_routing = routing;
    cfg.moe_placement = placement;
    cfg.n_experts = 3;
    cfg.expert_hidden_dims = {10};
    cfg.moe_top_k = 2;
    cfg.moe_noise_std = 0.0f;  // deterministic gating in these checks
    return cfg;
}

// Width of the continuous block Trainer::prepare_data assembles for this
// dataset: its schema's coordinates, covariates, their missingness flags and
// unknown fraction, and for the hash encoding the hash embedding on top.
int64_t continuous_width(const ResolveDataset& ds, SpeciesEncodingMode mode) {
    return ds.continuous_block(mode == SpeciesEncodingMode::Hash).size(1);
}

// The first n_rows plots' model inputs: the species-side tensors from the
// dataset, the continuous block synthetic, since the dataset stores its parts
// separately.
PlotInputs batch_inputs(const ResolveDataset& ds, SpeciesEncodingMode mode, int64_t n_rows) {
    PlotInputs inputs = ds.plot_inputs().without_empty().slice(0, n_rows);
    inputs.continuous = torch::randn({n_rows, continuous_width(ds, mode)});
    return inputs;
}

// Run one batch through forward_with_aux, the path the trainer takes.
ModelForwardResult forward_batch(ResolveModel& model, const ResolveDataset& ds,
                                 SpeciesEncodingMode mode, int64_t n_rows) {
    return model->forward_with_aux(batch_inputs(ds, mode, n_rows));
}

}  // namespace

// ===========================================================================
// Tail: the mixture is the encoder's last stage, for every encoding
// ===========================================================================

TEST_CASE("A tail mixture builds and runs for every species encoding",
          "[moe][encoder]") {
    const SmallCorpus corpus;

    for (auto mode : all_species_encodings()) {
        INFO("encoding " << species_encoding_to_string(mode));
        torch::manual_seed(7);

        auto ds = corpus.load(mode);
        ResolveModel model(ds.schema(), moe_model(mode, MoEPlacement::Tail));
        model->eval();

        // The mixture produces the last hidden width, so the latent is
        // hidden_dims.back() exactly as it is without a mixture.
        REQUIRE(model->latent_dim() == kHidden.back());

        auto out = forward_batch(model, ds, mode, 8);
        REQUIRE(out.outputs.at("y").size(0) == 8);
        REQUIRE(out.moe_aux_loss.defined());
        REQUIRE(std::isfinite(out.moe_aux_loss.item<float>()));
    }
}

TEST_CASE("The tail's parameters are named backbone + moe, and only then",
          "[moe][encoder][checkpoint]") {
    const SmallCorpus corpus;

    for (auto mode : all_species_encodings()) {
        INFO("encoding " << species_encoding_to_string(mode));
        auto ds = corpus.load(mode);

        // A hash-mode MoE checkpoint has always carried encoder.backbone.* and
        // encoder.moe.*; keeping those names is what lets one load after the
        // dedicated MoE encoder class went away.
        ResolveModel with_moe(ds.schema(), moe_model(mode, MoEPlacement::Tail));
        const auto moe_names = parameter_names(*with_moe);
        CHECK(any_starts_with(moe_names, "encoder.backbone."));
        CHECK(any_starts_with(moe_names, "encoder.moe."));
        CHECK_FALSE(any_starts_with(moe_names, "encoder.mlp."));
        CHECK_FALSE(any_starts_with(moe_names, "post_moe."));

        // And a run without routing is untouched by any of this.
        ResolveModel plain(ds.schema(), base_model(mode));
        const auto plain_names = parameter_names(*plain);
        CHECK(any_starts_with(plain_names, "encoder.mlp."));
        CHECK_FALSE(any_starts_with(plain_names, "encoder.moe."));
        CHECK_FALSE(any_starts_with(plain_names, "encoder.backbone."));
    }
}

TEST_CASE("The tail backbone takes hidden_dims minus its last two widths",
          "[moe][encoder]") {
    const SmallCorpus corpus;
    auto ds = corpus.load(SpeciesEncodingMode::Hash);

    // {24, 20, 16, 12}: backbone 24 -> 20, mixture 20 -> 12. The split is what
    // makes a mixture a redistribution of the encoder's capacity rather than an
    // extra block on top of it.
    ResolveModel model(ds.schema(), moe_model(SpeciesEncodingMode::Hash,
                                              MoEPlacement::Tail));
    REQUIRE(model->latent_dim() == 12);

    for (const auto& item : model->named_parameters()) {
        if (item.key() == "encoder.moe.layer_weight_0") {
            // (n_experts, out, in) for the first expert layer: in == 20, the
            // backbone's last width.
            REQUIRE(item.value().size(0) == 3);
            REQUIRE(item.value().size(2) == 20);
        }
        if (item.key() == "encoder.moe.layer_weight_1") {
            REQUIRE(item.value().size(1) == 12);  // out == hidden_dims.back()
        }
    }
}

TEST_CASE("A shorter hidden_dims spec still splits into a backbone and a mixture",
          "[moe][encoder]") {
    const SmallCorpus corpus;
    auto ds = corpus.load(SpeciesEncodingMode::Hash);

    // Two widths: too short to drop two stages, so the first is the backbone
    // and the mixture produces the second.
    auto cfg = moe_model(SpeciesEncodingMode::Hash, MoEPlacement::Tail);
    cfg.hidden_dims = {16, 8};
    ResolveModel model(ds.schema(), cfg);
    REQUIRE(model->latent_dim() == 8);

    // One width: the backbone produces it and the mixture maps it to itself.
    cfg.hidden_dims = {16};
    ResolveModel single(ds.schema(), cfg);
    REQUIRE(single->latent_dim() == 16);
}

// ===========================================================================
// The auxiliary loss reaches the trainer
// ===========================================================================

TEST_CASE("A tail mixture's load-balancing loss reaches training",
          "[moe][trainer]") {
    const SmallCorpus corpus;

    for (auto mode : all_species_encodings()) {
        INFO("encoding " << species_encoding_to_string(mode));
        torch::manual_seed(11);

        auto ds = corpus.load(mode);
        ResolveModel model(ds.schema(), moe_model(mode, MoEPlacement::Tail));

        TrainConfig train_config;
        train_config.max_epochs = 2;
        train_config.batch_size = 16;
        train_config.device = torch::kCPU;

        Trainer trainer(model, train_config);
        trainer.prepare_data(ds);
        auto result = trainer.fit();
        REQUIRE_FALSE(result.train_loss_history.empty());
        for (float loss : result.train_loss_history) {
            REQUIRE(std::isfinite(loss));
        }
    }
}

TEST_CASE("The aux-loss weight changes what the tail optimizes", "[moe][trainer]") {
    const SmallCorpus corpus;
    auto ds = corpus.load(SpeciesEncodingMode::Hash);

    // Same seed, same data, same architecture: the only difference is how
    // heavily the load-balancing term counts. Weighting it must move the
    // trained weights, or the term is not reaching the optimizer.
    const auto fit_with = [&](float aux_weight) {
        torch::manual_seed(3);
        auto cfg = moe_model(SpeciesEncodingMode::Hash, MoEPlacement::Tail);
        cfg.moe_aux_loss_weight = aux_weight;
        ResolveModel model(ds.schema(), cfg);

        TrainConfig train_config;
        train_config.max_epochs = 3;
        train_config.batch_size = 16;
        train_config.device = torch::kCPU;
        Trainer trainer(model, train_config);
        trainer.prepare_data(ds);
        trainer.fit();

        for (const auto& item : model->named_parameters()) {
            if (item.key() == "encoder.moe.gate.weight") {
                return item.value().detach().clone();
            }
        }
        return torch::Tensor();
    };

    auto light = fit_with(0.0f);
    auto heavy = fit_with(10.0f);
    REQUIRE(light.defined());
    REQUIRE(heavy.defined());
    REQUIRE_FALSE(torch::allclose(light, heavy));
}

// ===========================================================================
// Post: the placement for an encoder with no MLP tail
// ===========================================================================

TEST_CASE("A post mixture preserves the latent width", "[moe][encoder]") {
    const SmallCorpus corpus;

    for (auto mode : all_species_encodings()) {
        INFO("encoding " << species_encoding_to_string(mode));
        torch::manual_seed(5);

        auto ds = corpus.load(mode);
        ResolveModel model(ds.schema(), moe_model(mode, MoEPlacement::Post));
        model->eval();

        REQUIRE(model->latent_dim() == kHidden.back());
        const auto names = parameter_names(*model);
        CHECK(any_starts_with(names, "post_moe."));
        CHECK(any_starts_with(names, "encoder.mlp."));  // the encoder keeps its tail

        auto out = forward_batch(model, ds, mode, 8);
        REQUIRE(out.moe_aux_loss.defined());
        REQUIRE(std::isfinite(out.moe_aux_loss.item<float>()));
    }
}

TEST_CASE("An encoder with no MLP tail takes the mixture after the latent",
          "[moe][adapter]") {
    const SmallCorpus corpus;
    auto ds = corpus.load(SpeciesEncodingMode::Sparse);

    // The adapter architectures were refused any mixture at all; Post is a
    // placement they can honour, so the refusal becomes a supported case.
    torch::manual_seed(2);
    auto cfg = moe_model(SpeciesEncodingMode::Sparse, MoEPlacement::Post);
    cfg.encoder_architecture = EncoderArchitecture::FTTransformer;
    cfg.ft_transformer.d_model = 16;
    cfg.ft_transformer.n_heads = 2;
    cfg.ft_transformer.n_layers = 1;

    ResolveModel model(ds.schema(), cfg);
    model->eval();
    CHECK(any_starts_with(parameter_names(*model), "post_moe."));

    auto out = forward_batch(model, ds, SpeciesEncodingMode::Sparse, 8);
    REQUIRE(out.outputs.at("y").size(0) == 8);
    REQUIRE(out.moe_aux_loss.defined());
    REQUIRE(std::isfinite(out.moe_aux_loss.item<float>()));
}

TEST_CASE("Asking a tail-less encoder for a tail mixture names the placement that works",
          "[moe][adapter]") {
    const SmallCorpus corpus;
    auto ds = corpus.load(SpeciesEncodingMode::Sparse);

    auto cfg = moe_model(SpeciesEncodingMode::Sparse, MoEPlacement::Tail);
    cfg.encoder_architecture = EncoderArchitecture::TabNet;
    REQUIRE_THROWS_AS(ResolveModel(ds.schema(), cfg), std::invalid_argument);

    cfg.encoder_architecture = EncoderArchitecture::TraitNet;
    REQUIRE_THROWS_AS(ResolveModel(ds.schema(), cfg), std::invalid_argument);

    // ... and the same architectures accept the placement the message names.
    cfg.moe_placement = MoEPlacement::Post;
    cfg.encoder_architecture = EncoderArchitecture::TabNet;
    REQUIRE_NOTHROW(ResolveModel(ds.schema(), cfg));
}

TEST_CASE("TabM and a mixture cannot both have the tail", "[moe][encoder]") {
    const SmallCorpus corpus;
    auto ds = corpus.load(SpeciesEncodingMode::Hash);

    // Both replace the encoder's MLP tail, so the combination is refused rather
    // than dropping TabM in silence.
    auto cfg = moe_model(SpeciesEncodingMode::Hash, MoEPlacement::Tail);
    cfg.tabm.enabled = true;
    REQUIRE_THROWS_AS(ResolveModel(ds.schema(), cfg), std::invalid_argument);

    // Post leaves the tail to TabM, so the two coexist there.
    cfg.moe_placement = MoEPlacement::Post;
    REQUIRE_NOTHROW(ResolveModel(ds.schema(), cfg));
}

// ===========================================================================
// Gate probabilities
// ===========================================================================

TEST_CASE("Gate probabilities come back at either placement", "[moe][gates]") {
    const SmallCorpus corpus;
    auto ds = corpus.load(SpeciesEncodingMode::Hash);
    const auto inputs = batch_inputs(ds, SpeciesEncodingMode::Hash, 8);

    for (auto placement : {MoEPlacement::Tail, MoEPlacement::Post}) {
        INFO("placement " << (placement == MoEPlacement::Tail ? "tail" : "post"));
        torch::manual_seed(13);
        ResolveModel model(ds.schema(), moe_model(SpeciesEncodingMode::Hash, placement));
        model->eval();

        auto probs = model->get_gate_probs(inputs);
        REQUIRE(probs.defined());
        REQUIRE(probs.size(0) == 8);
        REQUIRE(probs.size(1) == 3);  // n_experts
        CHECK(probs.min().item<float>() >= 0.0f);
        auto row_sums = probs.sum(/*dim=*/1);
        CHECK(torch::allclose(row_sums, torch::ones_like(row_sums), 1e-4, 1e-4));
    }
}

TEST_CASE("Gate probabilities are still empty when there is no mixture",
          "[moe][gates]") {
    const SmallCorpus corpus;
    auto ds = corpus.load(SpeciesEncodingMode::Hash);

    ResolveModel model(ds.schema(), base_model(SpeciesEncodingMode::Hash));
    auto probs = model->get_gate_probs(batch_inputs(ds, SpeciesEncodingMode::Hash, 8));
    CHECK_FALSE(probs.defined());
}

TEST_CASE("Gate probabilities come back for every species encoder", "[moe][gates]") {
    const SmallCorpus corpus;

    // The inputs carry the species IDs, the species vector and the pool
    // tensors, so the encoders that read composition from them report their
    // gates as the hash encoder does.
    for (auto mode : {SpeciesEncodingMode::Embed, SpeciesEncodingMode::Sparse,
                      SpeciesEncodingMode::RankPool,
                      SpeciesEncodingMode::Transformer}) {
        INFO("encoding " << species_encoding_to_string(mode));
        auto ds = corpus.load(mode);
        torch::manual_seed(13);
        ResolveModel model(ds.schema(), moe_model(mode, MoEPlacement::Tail));
        model->eval();
        auto probs = model->get_gate_probs(batch_inputs(ds, mode, 8));
        REQUIRE(probs.defined());
        REQUIRE(probs.size(0) == 8);
        REQUIRE(probs.size(1) == 3);  // n_experts
        auto row_sums = probs.sum(/*dim=*/1);
        CHECK(torch::allclose(row_sums, torch::ones_like(row_sums), 1e-4, 1e-4));
    }
}

// ===========================================================================
// Checkpoints
// ===========================================================================

TEST_CASE("A mixture survives the checkpoint at either placement",
          "[moe][checkpoint]") {
    const SmallCorpus corpus;

    for (auto placement : {MoEPlacement::Tail, MoEPlacement::Post}) {
        for (auto mode : {SpeciesEncodingMode::Hash, SpeciesEncodingMode::Embed,
                          SpeciesEncodingMode::RankPool}) {
            INFO("encoding " << species_encoding_to_string(mode) << ", placement "
                             << (placement == MoEPlacement::Tail ? "tail" : "post"));
            torch::manual_seed(17);

            auto ds = corpus.load(mode);
            ResolveModel model(ds.schema(), moe_model(mode, placement));

            TrainConfig train_config;
            train_config.max_epochs = 1;
            train_config.batch_size = 16;
            train_config.device = torch::kCPU;

            Trainer trainer(model, train_config);
            trainer.prepare_data(ds);
            trainer.fit();

            const TempPath checkpoint(".pt");
            trainer.save(checkpoint.path());

            // The reload rebuilds the architecture from the persisted config,
            // and Trainer::load_weights_into throws on a missing parameter, so
            // a placement that failed to round-trip cannot load quietly.
            auto predictor = Predictor::load(checkpoint.path(), torch::kCPU);
            auto reloaded = predictor.predict(ds, /*return_latent=*/true);
            REQUIRE(reloaded.predictions.at("y").size(0) == SmallCorpus::kPlots);
            REQUIRE(reloaded.latent.size(1) == kHidden.back());
        }
    }
}

TEST_CASE("A reloaded mixture carries the weights that were trained",
          "[moe][checkpoint]") {
    const SmallCorpus corpus;
    torch::manual_seed(23);

    auto ds = corpus.load(SpeciesEncodingMode::RankPool);
    ResolveModel model(ds.schema(), moe_model(SpeciesEncodingMode::RankPool,
                                              MoEPlacement::Tail));

    TrainConfig train_config;
    train_config.max_epochs = 2;
    train_config.batch_size = 16;
    train_config.device = torch::kCPU;

    Trainer trainer(model, train_config);
    trainer.prepare_data(ds);
    trainer.fit();

    const TempPath checkpoint(".pt");
    trainer.save(checkpoint.path());

    auto predictor = Predictor::load(checkpoint.path(), torch::kCPU);

    // Every trained parameter, gate and expert weights included, must come back
    // bit-for-bit. Trainer::load_weights_into throws on a MISSING parameter, so
    // only an equality check can catch one that loaded into the wrong slot.
    std::unordered_map<std::string, torch::Tensor> saved;
    for (const auto& item : model->named_parameters()) {
        saved[item.key()] = item.value().detach().clone();
    }

    int mixture_params = 0;
    for (const auto& item : predictor.model()->named_parameters()) {
        INFO("parameter " << item.key());
        REQUIRE(saved.count(item.key()) == 1);
        REQUIRE(torch::equal(saved.at(item.key()), item.value().detach()));
        if (item.key().rfind("encoder.moe.", 0) == 0) ++mixture_params;
    }
    REQUIRE(saved.size() == predictor.model()->named_parameters().size());
    REQUIRE(mixture_params > 0);  // the mixture is actually in the comparison
}

TEST_CASE("The placement a run trained with is the placement it reloads",
          "[moe][checkpoint]") {
    const SmallCorpus corpus;
    torch::manual_seed(29);

    auto ds = corpus.load(SpeciesEncodingMode::Hash);
    ResolveModel model(ds.schema(), moe_model(SpeciesEncodingMode::Hash,
                                              MoEPlacement::Post));

    TrainConfig train_config;
    train_config.max_epochs = 1;
    train_config.batch_size = 16;
    train_config.device = torch::kCPU;

    Trainer trainer(model, train_config);
    trainer.prepare_data(ds);
    trainer.fit();

    const TempPath checkpoint(".pt");
    trainer.save(checkpoint.path());

    auto predictor = Predictor::load(checkpoint.path(), torch::kCPU);
    // A Post checkpoint reloaded as the Tail default would look for
    // encoder.moe.* and find post_moe.*, so this is the load itself asserting
    // the placement came back.
    REQUIRE(predictor.model()->config().moe_placement == MoEPlacement::Post);
    REQUIRE(predictor.model()->config().moe_routing == MoERoutingType::Soft);
    REQUIRE(any_starts_with(parameter_names(*predictor.model()), "post_moe."));
}

// ===========================================================================
// TopK routing
// ===========================================================================

TEST_CASE("TopK routing runs on a tail as well as Soft", "[moe][encoder]") {
    const SmallCorpus corpus;
    torch::manual_seed(31);

    auto ds = corpus.load(SpeciesEncodingMode::Embed);
    ResolveModel model(ds.schema(), moe_model(SpeciesEncodingMode::Embed,
                                              MoEPlacement::Tail,
                                              MoERoutingType::TopK));
    model->train();

    auto out = forward_batch(model, ds, SpeciesEncodingMode::Embed, 16);
    REQUIRE(out.moe_aux_loss.defined());
    REQUIRE(std::isfinite(out.moe_aux_loss.item<float>()));
    // Switch-Transformer load balancing: E * sum(f_i * P_i), minimised at 1.0
    // when the load is even, so it is never negative.
    REQUIRE(out.moe_aux_loss.item<float>() >= 0.0f);
}
