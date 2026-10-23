// The pretext tasks over a plot's whole input bundle: how each view masks the
// species inputs (embed slots, taxonomy slots, the explicit vector, the pool
// tensors as one unit per pooled species) and the categorical covariates, the
// vectorised feature masks and corruption, and every pretrainer training the
// parameters behind every input it reads, under every species encoding.

#include <catch2/catch_test_macros.hpp>

#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/pretraining.hpp"
#include "resolve/vae.hpp"
#include "test_support.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

// Each column of `after` holds the values of the same column of `before`, in
// some order.
bool columns_are_permutations(const torch::Tensor& before, const torch::Tensor& after) {
    return torch::equal(std::get<0>(before.sort(/*dim=*/0)), std::get<0>(after.sort(/*dim=*/0)));
}

// Four plots of six pooled-species slots, the last two of each plot padding.
PlotInputs pooled_plots() {
    PlotInputs in;
    in.species_ids = torch::arange(1, 25, torch::kInt64).reshape({4, 6});
    in.pool_genus_ids = in.species_ids + 100;
    in.pool_family_ids = in.species_ids + 200;
    in.pool_weights = torch::rand({4, 6}) + 0.5f;
    in.pool_mask = torch::ones({4, 6}, torch::kBool);
    in.pool_mask.slice(1, 4, 6).fill_(false);
    in.species_ids.slice(1, 4, 6).fill_(0);
    in.pool_genus_ids.slice(1, 4, 6).fill_(0);
    in.pool_family_ids.slice(1, 4, 6).fill_(0);
    in.pool_weights.slice(1, 4, 6).fill_(0.0f);
    in.pool_has_cover = torch::ones({4});
    in.continuous = torch::randn({4, 3});
    in.categorical_ids = torch::randint(1, 4, {4, 2});
    return in;
}

// The SmallCorpus line with a categorical "soil" column, under `mode`.
struct SoilCorpus {
    TempFile header;
    TempFile species;

    SoilCorpus()
        : header(write_header()), species(cyclic_species_csv(SmallCorpus::kPlots,
                                                             SmallCorpus::kSpecies)) {}

    static std::string write_header() {
        std::ostringstream hdr;
        hdr << "plot_id,lon,lat,elev,y,soil\n";
        const std::vector<std::string> soils = {"clay", "loam", "sand"};
        for (int i = 0; i < SmallCorpus::kPlots; ++i) {
            hdr << "p" << i << "," << (10.0 + 0.1 * i) << "," << (47.0 + 0.1 * i) << ","
                << (100.0 + 5.0 * i) << "," << (1.0 + 0.25 * i) << ","
                << soils[static_cast<size_t>(i % 3)] << "\n";
        }
        return hdr.str();
    }

    [[nodiscard]] ResolveDataset load(SpeciesEncodingMode mode) const {
        DatasetConfig dcfg;
        dcfg.species_encoding = mode;
        dcfg.use_taxonomy = true;
        dcfg.hash_dim = 32;
        return ResolveDataset::from_csv(
            header.path(), species.path(),
            Roles().coordinates().covariates({"elev"}).taxonomy().categoricals({"soil"}),
            {TargetSpec::regression("y")}, dcfg);
    }
};

// The model inputs a pretext task over `model` reads from `ds`: the
// dataset's bundle with the continuous block the encoder is sized for.
PlotInputs pretext_inputs(const ResolveDataset& ds, const ResolveModel& model) {
    PlotInputs inputs = ds.plot_inputs();
    inputs.continuous = ds.continuous_block(model->hash_in_continuous());
    return inputs;
}

std::vector<torch::Tensor> snapshot(const std::vector<torch::Tensor>& tensors) {
    std::vector<torch::Tensor> out;
    for (const auto& t : tensors) out.push_back(t.detach().clone());
    return out;
}

std::vector<torch::Tensor> categorical_parameters(const ResolveModel& model) {
    std::vector<torch::Tensor> out;
    for (const auto& item : model->named_parameters()) {
        if (item.key().rfind("categorical_embedder", 0) == 0) out.push_back(item.value());
    }
    return out;
}

// Every tensor of `now` differs from its snapshot `before`.
void require_all_moved(const std::vector<torch::Tensor>& before,
                       const std::vector<torch::Tensor>& now) {
    REQUIRE(before.size() == now.size());
    for (size_t i = 0; i < before.size(); ++i) {
        CHECK_FALSE(torch::equal(before[i], now[i].detach()));
    }
}

PretrainConfig quick_pretrain_config() {
    PretrainConfig cfg;
    cfg.pretrain_epochs = 2;
    cfg.batch_size = 16;
    cfg.pretrain_lr = 1e-2f;
    cfg.log = null_log;
    return cfg;
}

}  // namespace

// =============================================================================
// Pretext views
// =============================================================================

TEST_CASE("mask_codes drops nonzero codes at the ratio and keeps padding",
          "[pretraining][views]") {
    PretrainRng rng(3);
    const auto codes = torch::tensor({{1, 0, 3}, {0, 5, 6}}, torch::kInt64);
    REQUIRE(torch::equal(mask_codes(codes, 0.0f, rng), codes));
    REQUIRE(torch::equal(mask_codes(codes, 1.0f, rng), torch::zeros_like(codes)));
    const auto half = mask_codes(torch::arange(1, 401, torch::kInt64), 0.5f, rng);
    const auto dropped = (half == 0).sum().item<int64_t>();
    CHECK(dropped > 120);
    CHECK(dropped < 280);
}

TEST_CASE("A pooled species leaves the pool as a whole", "[pretraining][views]") {
    const PlotInputs in = pooled_plots();
    PretrainRng rng(7);
    const PlotInputs view = mask_species_view(in, 0.5f, rng);

    const auto was_valid = in.pool_mask;
    const auto dropped = was_valid & view.pool_mask.logical_not();
    const auto kept = view.pool_mask;
    REQUIRE(dropped.any().item<bool>());
    REQUIRE(kept.any().item<bool>());
    // Padding stays padding.
    REQUIRE_FALSE((kept & was_valid.logical_not()).any().item<bool>());

    for (auto field : {&PlotInputs::species_ids, &PlotInputs::pool_genus_ids,
                       &PlotInputs::pool_family_ids}) {
        const auto& before = in.*field;
        const auto& after = view.*field;
        REQUIRE((after.masked_select(dropped) == 0).all().item<bool>());
        REQUIRE(torch::equal(after.masked_select(kept), before.masked_select(kept)));
    }
    REQUIRE((view.pool_weights.masked_select(dropped) == 0.0f).all().item<bool>());
    REQUIRE(torch::equal(view.pool_weights.masked_select(kept),
                         in.pool_weights.masked_select(kept)));

    // The plot-level cover flag, the continuous block and the categoricals are
    // not species inputs.
    REQUIRE(torch::equal(view.pool_has_cover, in.pool_has_cover));
    REQUIRE(torch::equal(view.continuous, in.continuous));
    REQUIRE(torch::equal(view.categorical_ids, in.categorical_ids));
}

TEST_CASE("Embed slots and taxonomy slots are masked from their own shapes",
          "[pretraining][views]") {
    PlotInputs in;
    in.species_ids = torch::randint(1, 50, {8, 5});
    in.genus_ids = torch::randint(1, 9, {8, 3});
    in.family_ids = torch::randint(1, 9, {8, 3});
    PretrainRng rng(11);
    const auto all = mask_species_view(in, 1.0f, rng);
    REQUIRE(torch::equal(all.species_ids, torch::zeros_like(in.species_ids)));
    REQUIRE(torch::equal(all.genus_ids, torch::zeros_like(in.genus_ids)));
    REQUIRE(torch::equal(all.family_ids, torch::zeros_like(in.family_ids)));
    const auto none = mask_species_view(in, 0.0f, rng);
    REQUIRE(torch::equal(none.species_ids, in.species_ids));
}

TEST_CASE("The explicit species vector is masked entry by entry", "[pretraining][views]") {
    PlotInputs in;
    const auto dense = torch::rand({6, 10}) + 0.1f;
    in.species_vector = to_sparse_rows(dense);
    PretrainRng rng(5);
    const auto all = mask_species_view(in, 1.0f, rng).species_vector;
    REQUIRE(is_sparse_rows(all));
    REQUIRE(torch::equal(sparse_rows_to_dense(all), torch::zeros_like(dense)));
    REQUIRE(torch::equal(sparse_rows_to_dense(mask_species_view(in, 0.0f, rng).species_vector),
                         dense));
    // A partial ratio drops some entries and keeps the others' values.
    const auto half = sparse_rows_to_dense(mask_species_view(in, 0.5f, rng).species_vector);
    const auto kept = half != 0;
    REQUIRE(kept.any().item<bool>());
    REQUIRE_FALSE(kept.all().item<bool>());
    REQUIRE(torch::equal(half.masked_select(kept), dense.masked_select(kept)));
}

TEST_CASE("SCARF corrupts categorical cells from their column's marginal",
          "[pretraining][views]") {
    const auto codes = torch::randint(1, 6, {32, 3});
    SCARFCorruptor none(4, 0.0f);
    REQUIRE(torch::equal(none->corrupt_categoricals(codes, PretrainRng(1)), codes));
    SCARFCorruptor all(4, 1.0f);
    const auto corrupted = all->corrupt_categoricals(codes, PretrainRng(1));
    REQUIRE(columns_are_permutations(codes, corrupted));
    REQUIRE_FALSE(torch::equal(codes, corrupted));
}

TEST_CASE("SCARF corruption resamples each feature from its own column",
          "[pretraining][views]") {
    const auto features = torch::randn({40, 6});
    SCARFCorruptor all(6, 1.0f);
    const auto corrupted = all->corrupt(features, PretrainRng(2));
    REQUIRE(columns_are_permutations(features, corrupted));
    REQUIRE_FALSE(torch::equal(features, corrupted));
}

TEST_CASE("The block mask hides one contiguous run of features per row",
          "[pretraining][views]") {
    FeatureMasker masker(20, 0.25f, MaskStrategy::Block);
    const auto mask = masker->create_mask(64, PretrainRng(4));
    const auto hidden = (mask == 0.0f);
    REQUIRE((hidden.sum(1) == 5).all().item<bool>());
    // Contiguous: the hidden positions of a row span exactly 5 columns.
    const auto columns = torch::arange(20).unsqueeze(0).expand({64, 20});
    const auto first = torch::where(hidden, columns, torch::full_like(columns, 99)).amin(1);
    const auto last = torch::where(hidden, columns, torch::full_like(columns, -1)).amax(1);
    REQUIRE(((last - first) == 4).all().item<bool>());
}

TEST_CASE("The structured mask hides whole feature groups", "[pretraining][views]") {
    // 18 features -> 4 groups of 4, the last absorbing the remainder (6);
    // mask_ratio 0.5 hides 2 groups per row.
    FeatureMasker masker(18, 0.5f, MaskStrategy::Structured);
    const auto mask = masker->create_mask(50, PretrainRng(6));
    const auto hidden = (mask == 0.0f);
    for (int64_t row = 0; row < 50; ++row) {
        int groups = 0;
        for (int64_t g = 0; g < 4; ++g) {
            const int64_t start = g * 4;
            const int64_t end = g == 3 ? 18 : start + 4;
            const auto group = hidden[row].slice(0, start, end);
            const bool all_hidden = group.all().item<bool>();
            REQUIRE((all_hidden || !group.any().item<bool>()));
            groups += all_hidden ? 1 : 0;
        }
        REQUIRE(groups == 2);
    }
}

// =============================================================================
// Every pretrainer trains what each input reaches
// =============================================================================

TEST_CASE("JEPA and SCARF train the species and categorical parameters under every encoding",
          "[pretraining][encodings]") {
    const SoilCorpus corpus;
    for (const auto mode : all_species_encodings()) {
        INFO("encoding " << species_encoding_to_string(mode));
        const auto ds = corpus.load(mode);
        REQUIRE(ds.schema().n_categoricals() == 1);

        for (const bool contrastive : {false, true}) {
            INFO((contrastive ? "SCARF" : "JEPA"));
            torch::manual_seed(5);
            ResolveModel model(ds.schema(), small_encoder_config(mode, {16, 8}));
            const auto composition = model->composition_parameters();
            const auto categorical = categorical_parameters(model);
            REQUIRE_FALSE(categorical.empty());
            const auto composition_before = snapshot(composition);
            const auto categorical_before = snapshot(categorical);

            const PlotInputs inputs = pretext_inputs(ds, model);
            PretrainResult result;
            if (contrastive) {
                SCARFPretrainer pretrainer(model, quick_pretrain_config());
                result = pretrainer.pretrain(inputs);
            } else {
                JEPAPretrainer pretrainer(model, quick_pretrain_config());
                result = pretrainer.pretrain(inputs);
            }

            REQUIRE(result.loss_history.size() == 2);
            for (float loss : result.loss_history) REQUIRE(std::isfinite(loss));
            require_all_moved(categorical_before, categorical);
            require_all_moved(composition_before, composition);
        }
    }
}

TEST_CASE("Masked-species pretraining trains a transformer model's own encoder",
          "[pretraining][mlm]") {
    const SoilCorpus corpus;
    const auto ds = corpus.load(SpeciesEncodingMode::Transformer);
    torch::manual_seed(5);
    ResolveModel model(ds.schema(),
                       small_encoder_config(SpeciesEncodingMode::Transformer, {16, 8}));
    const auto before = model->get_species_weights().detach().clone();

    MLMPretrainConfig cfg;
    cfg.pretrain_epochs = 2;
    cfg.batch_size = 16;
    cfg.mask_prob = 0.5f;
    cfg.log = null_log;
    MaskedSpeciesPretrainer pretrainer(model, cfg);
    const auto result = pretrainer.pretrain(ds.plot_inputs());

    REQUIRE(result.loss_history.size() == 2);
    REQUIRE_FALSE(torch::equal(before, model->get_species_weights().detach()));
}

TEST_CASE("Masked-species pretraining needs a transformer model and its pool inputs",
          "[pretraining][mlm]") {
    const SoilCorpus corpus;
    const auto ds = corpus.load(SpeciesEncodingMode::Embed);
    ResolveModel model(ds.schema(), small_encoder_config(SpeciesEncodingMode::Embed, {16, 8}));
    REQUIRE_THROWS_AS(MaskedSpeciesPretrainer(model), std::invalid_argument);

    const auto transformer = corpus.load(SpeciesEncodingMode::Transformer);
    ResolveModel tmodel(transformer.schema(),
                        small_encoder_config(SpeciesEncodingMode::Transformer, {16, 8}));
    MaskedSpeciesPretrainer pretrainer(tmodel);
    PlotInputs without_mask = transformer.plot_inputs();
    without_mask.pool_mask = torch::Tensor();
    REQUIRE_THROWS_AS(pretrainer.pretrain(without_mask), std::invalid_argument);
}

TEST_CASE("The VAE reads the species vector of the bundle", "[pretraining][vae]") {
    const SoilCorpus corpus;
    VAEConfig cfg;
    cfg.latent_dim = 4;
    cfg.encoder_dims = {8};
    cfg.decoder_dims = {8};
    cfg.pretrain_epochs = 2;
    cfg.batch_size = 16;
    cfg.log = null_log;

    const auto sparse = corpus.load(SpeciesEncodingMode::Sparse);
    VAEPretrainer pretrainer(sparse.schema().n_species_vocab, cfg);
    const auto result = pretrainer.pretrain(sparse.plot_inputs());
    REQUIRE(result.loss_history.size() == 2);
    REQUIRE(result.recon_loss_history.size() == 2);

    const auto hash = corpus.load(SpeciesEncodingMode::Hash);
    REQUIRE_THROWS_AS(pretrainer.pretrain(hash.plot_inputs()), std::invalid_argument);
}
