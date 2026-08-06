#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "resolve/pretraining.hpp"
#include "resolve/vae.hpp"
#include "resolve/encoder.hpp"

using namespace resolve;
using namespace Catch::Matchers;

// =============================================================================
// Issue #107: reproducible pretraining + the shared pretrain loop
//
// Two contracts are pinned here, for each of the four pretext tasks:
//   (1) a fixed PretrainConfig / MLMPretrainConfig / VAEConfig seed reproduces
//       the run bit for bit, and a different seed changes it;
//   (2) a pretraining run leaves the GLOBAL torch RNG stream exactly where it
//       found it, so a pretrain-then-finetune sequence does not have its
//       finetune dropout draws shifted by however many batches pretraining ran.
//
// Every fixture below is built dropout-free on purpose. torch::nn::Dropout takes
// no generator argument, so a module dropout mask is the one draw in the
// pretraining path that CANNOT be moved off the global stream; a dropout-bearing
// configuration therefore does still advance it. Setting the dropout knobs to 0
// isolates exactly what the seeding fix owns: shuffles, feature masks,
// species-view masks, SCARF corruption, the BERT 80/10/10 split, the mask-token
// init and the VAE reparameterization noise.
// =============================================================================

namespace {

// Seeds the FIXTURES (inputs, model init) so a test's setup is identical across
// runs; distinct from the pretrainer seeds under test.
constexpr uint64_t kFixtureSeed = 20260807;
constexpr uint64_t kGlobalProbeSeed = 4242;

// -----------------------------------------------------------------------------
// Embed-mode fixture (JEPA + SCARF): species IDs live in their own tensors, so
// mask_species_view actually has work to do and its draws are covered.
// -----------------------------------------------------------------------------

ResolveSchema embed_schema() {
    ResolveSchema schema;
    schema.n_plots = 128;
    schema.n_species = 50;
    schema.n_species_vocab = 100;
    schema.has_coordinates = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;
    schema.n_genera_vocab = 25;
    schema.n_families_vocab = 15;
    schema.track_unknown_fraction = true;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});
    return schema;
}

ModelConfig embed_model_config() {
    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Embed;
    mcfg.species_embed_dim = 16;
    mcfg.top_k_species = 5;
    mcfg.n_taxonomy_slots = 3;
    mcfg.hidden_dims = {32, 16};
    mcfg.dropout = 0.0f;       // keeps the global RNG stream out of the forward
    mcfg.head_dropout = 0.0f;
    return mcfg;
}

struct EmbedInputs {
    torch::Tensor continuous;  // n_continuous = 2 coords + 1 unknown fraction
    torch::Tensor genus;
    torch::Tensor family;
    torch::Tensor species;
};

EmbedInputs make_embed_inputs(int64_t n_plots) {
    torch::manual_seed(kFixtureSeed);
    EmbedInputs in;
    in.continuous = torch::randn({n_plots, 3});
    in.species = torch::randint(1, 100, {n_plots, 5});
    in.genus = torch::randint(1, 25, {n_plots, 3});
    in.family = torch::randint(1, 15, {n_plots, 3});
    return in;
}

PretrainConfig base_pretrain_config() {
    PretrainConfig cfg;
    cfg.pretrain_epochs = 4;
    cfg.batch_size = 32;
    cfg.pretrain_lr = 1e-3f;
    cfg.predictor_dropout = 0.0f;  // JEPAPredictor is the other dropout site
    cfg.device = torch::kCPU;
    cfg.log = [](const std::string&) {};
    return cfg;
}

std::vector<float> run_jepa(const EmbedInputs& in, int seed, int epochs) {
    torch::manual_seed(kFixtureSeed);
    ResolveModel model(embed_schema(), embed_model_config());
    PretrainConfig cfg = base_pretrain_config();
    cfg.seed = seed;
    cfg.pretrain_epochs = epochs;
    JEPAPretrainer pretrainer(model, cfg);
    return pretrainer.pretrain(in.continuous, in.genus, in.family, in.species, {})
        .loss_history;
}

std::vector<float> run_scarf(const EmbedInputs& in, int seed, int epochs) {
    torch::manual_seed(kFixtureSeed);
    ResolveModel model(embed_schema(), embed_model_config());
    PretrainConfig cfg = base_pretrain_config();
    cfg.seed = seed;
    cfg.pretrain_epochs = epochs;
    SCARFPretrainer pretrainer(model, cfg);
    return pretrainer.pretrain(in.continuous, in.genus, in.family, in.species, {})
        .loss_history;
}

// -----------------------------------------------------------------------------
// MLM fixture: a dropout-free transformer encoder with one attention layer.
// -----------------------------------------------------------------------------

PlotEncoderTransformer make_mlm_encoder() {
    MLPBlockConfig mlp_config;
    mlp_config.dropout = 0.0f;
    return PlotEncoderTransformer(
        /*n_continuous=*/5,
        /*n_species=*/60,
        /*n_genera=*/0,
        /*n_families=*/0,
        /*d_model=*/32,
        /*n_heads=*/2,
        /*n_attention_layers=*/1,
        /*transformer_ff_dim=*/64,
        /*transformer_pooling=*/"attention",
        /*transformer_dropout=*/0.0f,
        /*hidden_dims=*/std::vector<int64_t>{16},
        /*mlp_config=*/mlp_config);
}

struct MLMInputs {
    torch::Tensor species;
    torch::Tensor valid;
};

MLMInputs make_mlm_inputs(int64_t n_plots) {
    torch::manual_seed(kFixtureSeed);
    MLMInputs in;
    in.species = torch::randint(1, 60, {n_plots, 12}, torch::kInt64);
    in.valid = (in.species != 0);
    return in;
}

MLMPretrainConfig base_mlm_config() {
    MLMPretrainConfig cfg;
    cfg.pretrain_epochs = 4;
    cfg.batch_size = 32;
    cfg.pretrain_lr = 1e-3f;
    cfg.mask_prob = 0.3f;
    cfg.device = torch::kCPU;
    cfg.log = [](const std::string&) {};
    return cfg;
}

std::vector<float> run_mlm(const MLMInputs& in, int seed, int epochs) {
    torch::manual_seed(kFixtureSeed);
    auto encoder = make_mlm_encoder();
    MLMPretrainConfig cfg = base_mlm_config();
    cfg.seed = seed;
    cfg.pretrain_epochs = epochs;
    MaskedSpeciesPretrainer pretrainer(encoder, 60, cfg);
    return pretrainer.pretrain(in.species, {}, {}, {}, in.valid);
}

// -----------------------------------------------------------------------------
// VAE fixture: sparse non-negative abundance vectors, dropout-free.
// -----------------------------------------------------------------------------

torch::Tensor make_species_vectors(int64_t n_plots, int64_t vocab) {
    torch::manual_seed(kFixtureSeed);
    auto presence = (torch::rand({n_plots, vocab}) < 0.2).to(torch::kFloat32);
    return presence * torch::rand({n_plots, vocab});
}

VAEConfig base_vae_config() {
    VAEConfig cfg;
    cfg.latent_dim = 8;
    cfg.encoder_dims = {32, 16};
    cfg.decoder_dims = {16, 32};
    cfg.dropout = 0.0f;
    cfg.kl_weight = 0.01f;
    cfg.kl_anneal_epochs = 0;  // constant weight: the ELBO is comparable per epoch
    cfg.pretrain_epochs = 4;
    cfg.batch_size = 32;
    cfg.pretrain_lr = 1e-3f;
    cfg.device = torch::kCPU;
    cfg.log = [](const std::string&) {};
    return cfg;
}

VAEPretrainResult run_vae(const torch::Tensor& vectors, int64_t vocab,
                          int seed, int epochs) {
    torch::manual_seed(kFixtureSeed);
    VAEConfig cfg = base_vae_config();
    cfg.seed = seed;
    cfg.pretrain_epochs = epochs;
    VAEPretrainer pretrainer(vocab, cfg);
    return pretrainer.pretrain(vectors);
}

// -----------------------------------------------------------------------------
// Global-stream probe
// -----------------------------------------------------------------------------

// Reset the global RNG, run `work`, then take the next global draw. Two calls
// whose `work` consumes no global randomness must return identical tensors.
torch::Tensor global_draw_after(const std::function<void()>& work) {
    torch::manual_seed(kGlobalProbeSeed);
    work();
    return torch::randn({8});
}

}  // namespace

// =============================================================================
// PretrainRng itself
// =============================================================================

TEST_CASE("PretrainRng seeded draws are reproducible", "[pretraining][seed]") {
    PretrainRng a(7);
    PretrainRng b(7);
    PretrainRng c(8);

    REQUIRE(a.is_seeded());
    REQUIRE(a.generator().has_value());

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto ra = a.rand({4, 3}, opts);
    auto rb = b.rand({4, 3}, opts);
    auto rc = c.rand({4, 3}, opts);

    REQUIRE(torch::equal(ra, rb));
    REQUIRE_FALSE(torch::equal(ra, rc));

    // randperm / randint / randn_like all ride the same stream.
    REQUIRE(torch::equal(a.randperm(16, torch::kCPU), b.randperm(16, torch::kCPU)));
    REQUIRE(torch::equal(a.randint(0, 9, {5}, torch::TensorOptions().dtype(torch::kLong)),
                         b.randint(0, 9, {5}, torch::TensorOptions().dtype(torch::kLong))));
    auto proto = torch::zeros({3, 2});
    REQUIRE(torch::equal(a.randn_like(proto), b.randn_like(proto)));
}

TEST_CASE("PretrainRng seed_epoch depends only on (seed, epoch)",
          "[pretraining][seed]") {
    // Reseeding per epoch is what keeps an epoch's stream independent of how
    // many batches the earlier epochs consumed.
    PretrainRng fresh(11);
    fresh.seed_epoch(3);
    auto expected = fresh.randperm(12, torch::kCPU);

    PretrainRng used(11);
    used.seed_epoch(0);
    for (int i = 0; i < 25; ++i) {
        (void)used.rand({7}, torch::TensorOptions().dtype(torch::kFloat32));
    }
    used.seed_epoch(3);

    REQUIRE(torch::equal(used.randperm(12, torch::kCPU), expected));
}

TEST_CASE("PretrainRng seeded mode does not touch the global stream",
          "[pretraining][seed]") {
    auto baseline = global_draw_after([] {});
    auto after = global_draw_after([] {
        PretrainRng rng(5);
        for (int epoch = 0; epoch < 3; ++epoch) {
            rng.seed_epoch(epoch);
            (void)rng.randperm(64, torch::kCPU);
            (void)rng.rand({16, 8}, torch::TensorOptions().dtype(torch::kFloat32));
            (void)rng.randint(0, 4, {16}, torch::TensorOptions().dtype(torch::kLong));
        }
    });
    REQUIRE(torch::equal(baseline, after));
}

TEST_CASE("PretrainRng default mode draws from the global stream",
          "[pretraining][seed]") {
    // The default-constructed handle is what a bare masker / corruptor /
    // mask_species_batch call outside a pretraining loop gets, and it must stay
    // on the stream torch::manual_seed controls.
    auto baseline = global_draw_after([] {});
    auto after = global_draw_after([] {
        PretrainRng rng;
        REQUIRE_FALSE(rng.is_seeded());
        (void)rng.rand({16, 8}, torch::TensorOptions().dtype(torch::kFloat32));
    });
    REQUIRE_FALSE(torch::equal(baseline, after));
}

// =============================================================================
// Config surface
// =============================================================================

TEST_CASE("Pretrain configs default to seed 42 and round-trip it",
          "[pretraining][seed][config]") {
    // 42 matches the Trainer / prepare_data seed default, so a pretrain-then-
    // finetune script that leaves both alone gets one consistent default.
    const PretrainConfig pretrain_defaults;
    const MLMPretrainConfig mlm_defaults;
    const VAEConfig vae_defaults;
    REQUIRE(pretrain_defaults.seed == 42);
    REQUIRE(mlm_defaults.seed == 42);
    REQUIRE(vae_defaults.seed == 42);

    ResolveModel model(embed_schema(), embed_model_config());

    PretrainConfig pcfg = base_pretrain_config();
    pcfg.seed = 1234;
    JEPAPretrainer jepa(model, pcfg);
    SCARFPretrainer scarf(model, pcfg);
    REQUIRE(jepa.config().seed == 1234);
    REQUIRE(scarf.config().seed == 1234);

    MLMPretrainConfig mcfg = base_mlm_config();
    mcfg.seed = -77;
    MaskedSpeciesPretrainer mlm(make_mlm_encoder(), 60, mcfg);
    REQUIRE(mlm.config().seed == -77);

    VAEConfig vcfg = base_vae_config();
    vcfg.seed = 99;
    VAEPretrainer vae(24, vcfg);
    REQUIRE(vae.config().seed == 99);
}

TEST_CASE("Pretrain config validation accepts any seed",
          "[pretraining][seed][validation]") {
    // Every int is a usable generator seed once converted to uint64, negatives
    // included, so validate() has nothing to reject here.
    for (int seed : {-2147483647 - 1, -1, 0, 1, 2147483647}) {
        PretrainConfig pcfg;
        pcfg.seed = seed;
        REQUIRE_NOTHROW(pcfg.validate());

        MLMPretrainConfig mcfg;
        mcfg.seed = seed;
        REQUIRE_NOTHROW(mcfg.validate());
    }
}

// =============================================================================
// JEPA
// =============================================================================

TEST_CASE("JEPA pretraining is reproducible from its seed",
          "[pretraining][seed][jepa]") {
    auto in = make_embed_inputs(128);

    auto a = run_jepa(in, /*seed=*/11, /*epochs=*/4);
    auto b = run_jepa(in, /*seed=*/11, /*epochs=*/4);
    auto c = run_jepa(in, /*seed=*/99, /*epochs=*/4);

    REQUIRE(a.size() == 4);
    REQUIRE(a == b);   // bit-identical, not merely close
    REQUIRE(a != c);
}

TEST_CASE("JEPA pretraining leaves the global RNG stream untouched",
          "[pretraining][seed][jepa]") {
    auto in = make_embed_inputs(128);

    // Model + pretrainer are constructed OUTSIDE the probe: their weight init
    // legitimately draws from the global stream, and only the pretrain() call is
    // under test.
    torch::manual_seed(kFixtureSeed);
    ResolveModel model(embed_schema(), embed_model_config());
    PretrainConfig cfg = base_pretrain_config();
    JEPAPretrainer pretrainer(model, cfg);

    auto baseline = global_draw_after([] {});
    auto after = global_draw_after([&] {
        pretrainer.pretrain(in.continuous, in.genus, in.family, in.species, {});
    });

    REQUIRE(torch::equal(baseline, after));
}

TEST_CASE("JEPA pretraining still converges through the shared loop",
          "[pretraining][seed][jepa]") {
    auto in = make_embed_inputs(128);
    auto history = run_jepa(in, /*seed=*/11, /*epochs=*/40);

    REQUIRE(history.size() == 40);
    for (float loss : history) REQUIRE(std::isfinite(loss));
    REQUIRE(history.back() < history.front());
}

// =============================================================================
// SCARF
// =============================================================================

TEST_CASE("SCARF pretraining is reproducible from its seed",
          "[pretraining][seed][scarf]") {
    auto in = make_embed_inputs(128);

    auto a = run_scarf(in, /*seed=*/11, /*epochs=*/4);
    auto b = run_scarf(in, /*seed=*/11, /*epochs=*/4);
    auto c = run_scarf(in, /*seed=*/99, /*epochs=*/4);

    REQUIRE(a.size() == 4);
    REQUIRE(a == b);
    REQUIRE(a != c);
}

TEST_CASE("SCARF pretraining leaves the global RNG stream untouched",
          "[pretraining][seed][scarf]") {
    auto in = make_embed_inputs(128);

    torch::manual_seed(kFixtureSeed);
    ResolveModel model(embed_schema(), embed_model_config());
    SCARFPretrainer pretrainer(model, base_pretrain_config());

    auto baseline = global_draw_after([] {});
    auto after = global_draw_after([&] {
        pretrainer.pretrain(in.continuous, in.genus, in.family, in.species, {});
    });

    REQUIRE(torch::equal(baseline, after));
}

TEST_CASE("SCARF pretraining still converges through the shared loop",
          "[pretraining][seed][scarf]") {
    auto in = make_embed_inputs(128);
    auto history = run_scarf(in, /*seed=*/11, /*epochs=*/40);

    REQUIRE(history.size() == 40);
    for (float loss : history) REQUIRE(std::isfinite(loss));
    REQUIRE(history.back() < history.front());
}

// =============================================================================
// Masked-species (BERT MLM)
// =============================================================================

TEST_CASE("MLM pretraining is reproducible from its seed",
          "[pretraining][seed][mlm]") {
    auto in = make_mlm_inputs(128);

    auto a = run_mlm(in, /*seed=*/11, /*epochs=*/4);
    auto b = run_mlm(in, /*seed=*/11, /*epochs=*/4);
    auto c = run_mlm(in, /*seed=*/99, /*epochs=*/4);

    REQUIRE(a.size() == 4);
    REQUIRE(a == b);
    REQUIRE(a != c);
}

TEST_CASE("MLM pretraining leaves the global RNG stream untouched",
          "[pretraining][seed][mlm]") {
    auto in = make_mlm_inputs(128);

    torch::manual_seed(kFixtureSeed);
    auto encoder = make_mlm_encoder();
    MaskedSpeciesPretrainer pretrainer(encoder, 60, base_mlm_config());

    auto baseline = global_draw_after([] {});
    auto after = global_draw_after([&] {
        pretrainer.pretrain(in.species, {}, {}, {}, in.valid);
    });

    REQUIRE(torch::equal(baseline, after));
}

TEST_CASE("MLM pretraining still converges through the shared loop",
          "[pretraining][seed][mlm]") {
    auto in = make_mlm_inputs(128);
    auto history = run_mlm(in, /*seed=*/11, /*epochs=*/30);

    REQUIRE(history.size() == 30);
    for (float loss : history) REQUIRE(std::isfinite(loss));
    REQUIRE(history.back() < history.front());
}

// =============================================================================
// VAE
// =============================================================================

TEST_CASE("VAE pretraining is reproducible from its seed",
          "[pretraining][seed][vae]") {
    const int64_t vocab = 24;
    auto vectors = make_species_vectors(128, vocab);

    auto a = run_vae(vectors, vocab, /*seed=*/11, /*epochs=*/4);
    auto b = run_vae(vectors, vocab, /*seed=*/11, /*epochs=*/4);
    auto c = run_vae(vectors, vocab, /*seed=*/99, /*epochs=*/4);

    REQUIRE(a.loss_history.size() == 4);
    REQUIRE(a.loss_history == b.loss_history);
    REQUIRE(a.recon_loss_history == b.recon_loss_history);
    REQUIRE(a.kl_loss_history == b.kl_loss_history);
    REQUIRE(a.loss_history != c.loss_history);
}

TEST_CASE("VAE pretraining leaves the global RNG stream untouched",
          "[pretraining][seed][vae]") {
    const int64_t vocab = 24;
    auto vectors = make_species_vectors(128, vocab);

    torch::manual_seed(kFixtureSeed);
    VAEPretrainer pretrainer(vocab, base_vae_config());

    auto baseline = global_draw_after([] {});
    auto after = global_draw_after([&] { pretrainer.pretrain(vectors); });

    REQUIRE(torch::equal(baseline, after));
}

TEST_CASE("VAE pretraining still converges through the shared loop",
          "[pretraining][seed][vae]") {
    const int64_t vocab = 24;
    auto vectors = make_species_vectors(128, vocab);
    auto result = run_vae(vectors, vocab, /*seed=*/11, /*epochs=*/60);

    REQUIRE(result.loss_history.size() == 60);
    REQUIRE(result.recon_loss_history.size() == 60);
    REQUIRE(result.kl_loss_history.size() == 60);
    for (float loss : result.loss_history) REQUIRE(std::isfinite(loss));
    REQUIRE(result.recon_loss_history.back() < result.recon_loss_history.front());
    REQUIRE(result.loss_history.back() < result.loss_history.front());
}

// =============================================================================
// Shared loop contract
// =============================================================================

TEST_CASE("run_pretrain_loop rejects an unusable spec", "[pretraining][loop]") {
    auto noop = [](const PretrainBatch&, PretrainRng&) { return torch::zeros({1}); };

    SECTION("batch_size < 1") {
        PretrainLoopSpec spec;
        spec.inputs = {torch::randn({4, 2})};
        spec.batch_size = 0;
        REQUIRE_THROWS_AS(run_pretrain_loop(spec, noop), std::invalid_argument);
    }
    SECTION("no inputs") {
        PretrainLoopSpec spec;
        REQUIRE_THROWS_AS(run_pretrain_loop(spec, noop), std::invalid_argument);
    }
    SECTION("undefined leading input") {
        PretrainLoopSpec spec;
        spec.inputs = {torch::Tensor(), torch::randn({4, 2})};
        REQUIRE_THROWS_AS(run_pretrain_loop(spec, noop), std::invalid_argument);
    }
    SECTION("empty objective") {
        PretrainLoopSpec spec;
        spec.inputs = {torch::randn({4, 2})};
        REQUIRE_THROWS_AS(run_pretrain_loop(spec, PretrainBatchFn{}),
                          std::invalid_argument);
    }
}

TEST_CASE("run_pretrain_loop slices every input and honours skipped batches",
          "[pretraining][loop]") {
    // The loop passes each row-aligned input through in order, leaves absent
    // ones undefined, and treats an undefined loss as "skip this batch".
    auto weight = torch::zeros({1, 3}, torch::requires_grad(true));

    PretrainLoopSpec spec;  // no modules: the objective owns the only parameter
    spec.params = {weight};
    spec.inputs = {torch::randn({10, 3}), torch::Tensor(), torch::randn({10, 2})};
    spec.epochs = 2;
    spec.batch_size = 4;   // 10 rows -> batches of 4, 4, 2
    spec.log = [](const std::string&) {};

    int calls = 0;
    std::vector<int64_t> batch_rows;
    auto batch_fn = [&](const PretrainBatch& batch, PretrainRng&) -> torch::Tensor {
        REQUIRE(batch.size() == 3);
        REQUIRE(batch[0].defined());
        REQUIRE_FALSE(batch[1].defined());   // undefined inputs stay undefined
        REQUIRE(batch[2].defined());
        REQUIRE(batch[0].size(0) == batch[2].size(0));
        batch_rows.push_back(batch[0].size(0));
        ++calls;
        // Skip the middle batch of every epoch.
        if (calls % 3 == 2) return {};
        return (batch[0] * weight).sum();
    };

    auto result = run_pretrain_loop(spec, batch_fn);

    const std::vector<int64_t> expected_rows{4, 4, 2, 4, 4, 2};
    REQUIRE(calls == 6);                    // 3 batches x 2 epochs
    REQUIRE(batch_rows == expected_rows);
    REQUIRE(result.epochs_completed == 2);
    REQUIRE(result.loss_history.size() == 2);
    REQUIRE(result.total_time_seconds >= 0.0f);
}

TEST_CASE("run_pretrain_loop fires its hooks in order", "[pretraining][loop]") {
    auto weight = torch::zeros({1, 2}, torch::requires_grad(true));

    PretrainLoopSpec spec;
    spec.params = {weight};
    spec.inputs = {torch::randn({8, 2})};
    spec.epochs = 3;
    spec.batch_size = 4;   // 2 batches per epoch
    spec.log_every = 0;    // silence the progress line

    int begins = 0;
    int steps = 0;
    int ends = 0;
    std::vector<int> begin_epochs;

    PretrainLoopHooks hooks;
    hooks.on_epoch_begin = [&](int epoch) { ++begins; begin_epochs.push_back(epoch); };
    hooks.on_step_end = [&]() { ++steps; };
    hooks.on_epoch_end = [&](int, float) {
        // loss_history is appended before on_epoch_end runs, which is what lets
        // the VAE push a matching entry onto its component histories here.
        ++ends;
        REQUIRE(ends == begins);
    };

    auto result = run_pretrain_loop(
        spec,
        [&](const PretrainBatch& batch, PretrainRng&) {
            return (batch[0] * weight).sum();
        },
        hooks);

    const std::vector<int> expected_epochs{0, 1, 2};
    REQUIRE(begins == 3);
    REQUIRE(ends == 3);
    REQUIRE(steps == 6);
    REQUIRE(begin_epochs == expected_epochs);
    REQUIRE(result.loss_history.size() == 3);
}
