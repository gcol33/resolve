// The species-encoder registry and what every encoding shares through it: one
// effective encoding per ModelConfig, the encoder and its features built from
// the registry row, per-layer activations for the diagnostics, every encoding
// against every tabular architecture, and AggregationMode::Count in the pooled
// encodings.

#include <catch2/catch_test_macros.hpp>

#include "resolve/enum_names.hpp"
#include "resolve/model.hpp"
#include "resolve/plot_encoder_rank_pool.hpp"
#include "resolve/plot_encoder_transformer.hpp"
#include "resolve/species_encoder.hpp"
#include "resolve/species_pool.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <functional>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

constexpr int64_t kBatch = 5;
constexpr int64_t kWidth = 4;

SpeciesEncoderSpec spec_for(SpeciesEncodingMode mode) {
    SpeciesEncoderSpec spec;
    spec.n_continuous = 3;
    spec.n_species = 20;
    spec.n_genera = 6;
    spec.n_families = 4;
    spec.config = small_encoder_config(mode, {12, 8});
    spec.config.top_k_species = kWidth;
    spec.config.n_taxonomy_slots = 2;
    return spec;
}

// Inputs for every encoding of spec_for(): pool rows of 2..kWidth species.
PlotInputs inputs_for(const SpeciesEncoderSpec& spec) {
    torch::manual_seed(3);
    PlotInputs in;
    const auto& traits = species_encoding_traits(effective_species_encoding(spec.config));
    in.continuous = torch::randn(
        {kBatch, spec.n_continuous + (traits.hash_in_continuous ? spec.config.hash_dim : 0)});
    in.genus_ids = torch::randint(1, spec.n_genera, {kBatch, 2});
    in.family_ids = torch::randint(1, spec.n_families, {kBatch, 2});
    const auto lengths = torch::tensor({2, 3, 4, 4, 2}).unsqueeze(1);
    in.pool_mask = torch::arange(kWidth).unsqueeze(0) < lengths;
    in.species_ids = torch::randint(1, spec.n_species, {kBatch, kWidth}) * in.pool_mask;
    in.pool_genus_ids = torch::randint(1, spec.n_genera, {kBatch, kWidth}) * in.pool_mask;
    in.pool_family_ids = torch::randint(1, spec.n_families, {kBatch, kWidth}) * in.pool_mask;
    in.pool_weights = torch::rand({kBatch, kWidth}) * in.pool_mask;
    in.pool_has_cover = torch::ones({kBatch});
    in.species_vector = to_sparse_rows(torch::rand({kBatch, spec.n_species}) *
                                       (torch::rand({kBatch, spec.n_species}) < 0.3));
    return in;
}

const std::vector<EncoderArchitecture> kAdapterArchitectures = {
    EncoderArchitecture::FTTransformer, EncoderArchitecture::TabNet,
    EncoderArchitecture::SAINT,         EncoderArchitecture::GNN,
    EncoderArchitecture::ExcelFormer,   EncoderArchitecture::HeterogeneousGNN,
};

ModelConfig adapter_config(SpeciesEncodingMode mode, EncoderArchitecture architecture) {
    ModelConfig cfg = small_encoder_config(mode, {8});
    cfg.encoder_architecture = architecture;
    cfg.n_attention_layers = 0;
    cfg.ft_transformer.d_model = 8;
    cfg.ft_transformer.n_heads = 2;
    cfg.ft_transformer.n_layers = 1;
    cfg.tabnet.n_steps = 2;
    cfg.tabnet.n_d = 8;
    cfg.tabnet.n_a = 8;
    cfg.tabnet.virtual_batch_size = 4;
    cfg.saint.d_model = 8;
    cfg.saint.n_heads = 2;
    cfg.saint.n_layers = 1;
    cfg.saint.row_attention_rows = 4;
    cfg.gnn.hidden_dim = 8;
    cfg.gnn.n_layers = 1;
    cfg.gnn.n_heads = 2;
    cfg.gnn.k_neighbors = 3;
    cfg.excelformer.d_model = 8;
    cfg.excelformer.n_heads = 2;
    cfg.excelformer.n_layers = 1;
    cfg.heterogeneous_gnn.hidden_dim = 8;
    cfg.heterogeneous_gnn.output_dim = 4;
    cfg.heterogeneous_gnn.n_layers = 1;
    cfg.heterogeneous_gnn.n_heads = 2;
    return cfg;
}

ResolveDataset load_corpus(const SmallCorpus& corpus, SpeciesEncodingMode mode,
                           AggregationMode aggregation) {
    Roles roles;
    roles.coordinates().covariates({"elev"}).taxonomy();
    DatasetConfig cfg;
    cfg.species_encoding = mode;
    cfg.aggregation = aggregation;
    return ResolveDataset::from_csv(corpus.header_path(), corpus.species_path(), roles,
                                    {TargetSpec::regression("y")}, cfg);
}

}  // namespace

TEST_CASE("Every species encoding has one registry row", "[species_encoder]") {
    for (auto mode : all_species_encodings()) {
        INFO(species_encoding_to_string(mode));
        const auto& traits = species_encoding_traits(mode);
        CHECK(traits.mode == mode);
        CHECK(traits.make_features != nullptr);
        CHECK(traits.make_encoder != nullptr);
        // Exactly the pooled encodings read per-species taxonomy instead of
        // the fixed slots.
        CHECK(traits.pooled != traits.taxonomy_slots);
    }
    CHECK(species_encoding_traits(SpeciesEncodingMode::Hash).hash_in_continuous);
    CHECK_FALSE(species_encoding_traits(SpeciesEncodingMode::Hash).indexes_species_codes);
}

TEST_CASE("The effective encoding resolves uses_explicit_vector once", "[species_encoder]") {
    ModelConfig cfg;
    for (auto mode : all_species_encodings()) {
        cfg.species_encoding = mode;
        cfg.uses_explicit_vector = false;
        CHECK(effective_species_encoding(cfg) == mode);
    }
    cfg.uses_explicit_vector = true;
    cfg.species_encoding = SpeciesEncodingMode::Hash;
    CHECK(effective_species_encoding(cfg) == SpeciesEncodingMode::Sparse);
    cfg.species_encoding = SpeciesEncodingMode::Sparse;
    CHECK(effective_species_encoding(cfg) == SpeciesEncodingMode::Sparse);
    for (auto mode : {SpeciesEncodingMode::Embed, SpeciesEncodingMode::RankPool,
                      SpeciesEncodingMode::Transformer}) {
        cfg.species_encoding = mode;
        CHECK_THROWS_AS(effective_species_encoding(cfg), std::invalid_argument);
    }

    // The model, the hash-in-continuous test and the predictor's dataset
    // config read the same resolution.
    ResolveSchema schema;
    schema.n_species_vocab = 10;
    schema.targets.push_back({"y", TaskType::Regression, TransformType::None, 0});
    ModelConfig explicit_hash = small_encoder_config(SpeciesEncodingMode::Hash, {8});
    explicit_hash.uses_explicit_vector = true;
    ResolveModel model(schema, explicit_hash);
    CHECK(model->effective_species_encoding() == SpeciesEncodingMode::Sparse);
    CHECK_FALSE(model->hash_in_continuous());
    CHECK(any_starts_with(parameter_names(*model), "encoder.species_projection"));
    CHECK(dataset_config_from_checkpoint(model->schema(), model->config()).species_encoding ==
          SpeciesEncodingMode::Sparse);
}

TEST_CASE("Each encoder forwards through its registry row", "[species_encoder]") {
    for (auto mode : all_species_encodings()) {
        INFO(species_encoding_to_string(mode));
        const auto spec = spec_for(mode);
        auto encoder = make_species_encoder(spec);
        encoder->eval();
        const auto latent = encoder->forward(inputs_for(spec));
        CHECK(latent.sizes() == torch::IntArrayRef({kBatch, 8}));
        CHECK(encoder->latent_dim() == 8);
        CHECK(encoder->has_taxonomy());
        CHECK_FALSE(encoder->composition_parameters().empty());
        CHECK(encoder->genus_weights().size(0) == spec.n_genera);
        CHECK(encoder->family_weights().size(0) == spec.n_families);
    }
}

TEST_CASE("An encoding that indexes species codes needs a species vocabulary",
          "[species_encoder]") {
    for (auto mode : all_species_encodings()) {
        INFO(species_encoding_to_string(mode));
        auto spec = spec_for(mode);
        spec.n_species = 0;
        if (species_encoding_traits(mode).indexes_species_codes) {
            CHECK_THROWS_AS(make_species_encoder(spec), std::runtime_error);
        } else {
            CHECK_NOTHROW(make_species_encoder(spec));
        }
    }
}

TEST_CASE("rank_pool refuses a taxonomy batch without family codes", "[species_encoder]") {
    const auto spec = spec_for(SpeciesEncodingMode::RankPool);
    PlotEncoderRankPool encoder(spec);
    auto in = inputs_for(spec);
    in.pool_family_ids = torch::Tensor();
    CHECK_THROWS_WITH(encoder->forward(in),
                      Catch::Matchers::ContainsSubstring("family=no"));
}

TEST_CASE("Every encoder tail reports its hidden-layer activations", "[species_encoder][diagnostics]") {
    struct Tail {
        const char* name;
        std::function<void(ModelConfig&)> edit;
        std::vector<int64_t> widths;  // columns of each activation
        int64_t rows_per_plot;
    };
    const std::vector<Tail> tails = {
        {"mlp", [](ModelConfig&) {}, {12, 8}, 1},
        {"residual", [](ModelConfig& c) { c.use_residual = true; }, {12, 8}, 1},
        {"moe", [](ModelConfig& c) {
             c.hidden_dims = {12, 10, 8};
             c.moe_routing = MoERoutingType::Soft;
             c.n_experts = 2;
             c.expert_hidden_dims = {6};
         }, {12}, 1},
        {"tabm", [](ModelConfig& c) {
             c.tabm.enabled = true;
             c.tabm.n_ensembles = 3;
         }, {12, 8}, 3},
        {"parallel", [](ModelConfig& c) {
             ParallelBranchConfig branch;
             branch.hidden_dims = {6, 5};
             c.parallel_layers.enabled = true;
             c.parallel_layers.branches = {branch, branch};
         }, {6, 5, 6, 5}, 1},
    };
    for (auto mode : all_species_encodings()) {
        for (const auto& tail : tails) {
            INFO(species_encoding_to_string(mode) << " / " << tail.name);
            auto spec = spec_for(mode);
            tail.edit(spec.config);
            auto encoder = make_species_encoder(spec);
            encoder->eval();
            std::vector<torch::Tensor> activations;
            const auto inputs = inputs_for(spec);
            const auto out = encoder->encode(inputs, &activations);
            REQUIRE(activations.size() == tail.widths.size());
            for (size_t i = 0; i < activations.size(); ++i) {
                CHECK(activations[i].sizes() ==
                      torch::IntArrayRef({kBatch * tail.rows_per_plot, tail.widths[i]}));
            }
            // The capture reads the forward rather than changing it.
            CHECK(torch::allclose(out.latent, encoder->forward(inputs)));
        }
    }
}

TEST_CASE("compute_diagnostics reports every layer for every encoding",
          "[species_encoder][diagnostics]") {
    const SmallCorpus corpus;
    for (auto mode : all_species_encodings()) {
        INFO(species_encoding_to_string(mode));
        torch::manual_seed(11);
        auto ds = corpus.load(mode);
        ResolveModel model(ds.schema(), small_encoder_config(mode, {12, 8}));
        Trainer trainer(model, quiet_train_config(1));
        trainer.prepare_data(ds);
        trainer.fit();
        const auto diag = trainer.compute_diagnostics();
        REQUIRE(diag.layers.size() == 2);
        CHECK(diag.layers[0].n_neurons == 12);
        CHECK(diag.layers[1].n_neurons == 8);
        CHECK(diag.total_neurons == 20);
        CHECK(diag.summary.find("not available") == std::string::npos);
    }
}

TEST_CASE("Every encoding meets every tabular architecture", "[species_encoder][adapter]") {
    const SmallCorpus corpus;
    for (auto mode : all_species_encodings()) {
        auto ds = corpus.load(mode);
        for (auto architecture : kAdapterArchitectures) {
            INFO(species_encoding_to_string(mode) << " x "
                 << encoder_architecture_to_string(architecture));
            const ModelConfig cfg = adapter_config(mode, architecture);
            const bool supported = architecture != EncoderArchitecture::HeterogeneousGNN ||
                                   mode == SpeciesEncodingMode::Sparse;
            if (!supported) {
                CHECK_THROWS_WITH(ResolveModel(ds.schema(), cfg),
                                  Catch::Matchers::ContainsSubstring("does not provide"));
                continue;
            }
            torch::manual_seed(2);
            ResolveModel model(ds.schema(), cfg);
            Trainer trainer(model, quiet_train_config(1));
            trainer.prepare_data(ds);
            const auto result = trainer.fit();
            CHECK(std::isfinite(result.train_loss_history.back()));
        }
    }
}

TEST_CASE("A GNN graph mode the encoding cannot feed is refused at construction",
          "[species_encoder][adapter]") {
    const SmallCorpus corpus;
    auto pooled = corpus.load(SpeciesEncodingMode::RankPool);
    ModelConfig cfg = adapter_config(SpeciesEncodingMode::RankPool, EncoderArchitecture::GNN);
    cfg.gnn.graph_mode = GraphConstructionMode::Taxonomic;
    CHECK_THROWS_WITH(ResolveModel(pooled.schema(), cfg),
                      Catch::Matchers::ContainsSubstring("fixed genus / family slots"));
    cfg.gnn.graph_mode = GraphConstructionMode::CoOccurrence;
    CHECK_THROWS_WITH(ResolveModel(pooled.schema(), cfg),
                      Catch::Matchers::ContainsSubstring("species vector"));
}

TEST_CASE("A pooled encoding feeds the adapter its pooled embeddings",
          "[species_encoder][adapter]") {
    const SmallCorpus corpus;
    auto ds = corpus.load(SpeciesEncodingMode::RankPool);
    ModelConfig cfg = adapter_config(SpeciesEncodingMode::RankPool,
                                     EncoderArchitecture::FTTransformer);
    ResolveModel model(ds.schema(), cfg);
    const auto names = parameter_names(*model);
    CHECK(any_starts_with(names, "adapter.species_embedding"));
    CHECK(any_starts_with(names, "adapter.genus_embedding"));
    // The pooled encodings carry their taxonomy per species, so no slot
    // categoricals are tokenized.
    CHECK_FALSE(any_starts_with(names, "adapter.genus_emb_"));
}

TEST_CASE("Count pools a rank_pool plot's taxa with one weight per species",
          "[species_encoder][aggregation]") {
    auto spec = spec_for(SpeciesEncodingMode::RankPool);
    const auto in = inputs_for(spec);

    for (auto aggregation : {AggregationMode::Abundance, AggregationMode::Count}) {
        INFO(enum_to_name(aggregation));
        spec.aggregation = aggregation;
        torch::nn::Module owner;
        auto features = make_rank_pool_features(owner, spec);
        const auto out = features->forward(in, /*training=*/false);

        const auto pool = PoolInputs::from(in);
        const auto species_w = pool.normalized(pool.weights);
        const auto taxon_w = aggregation == AggregationMode::Count
            ? pool.mask.to(torch::kFloat32) /
                  pool.mask.to(torch::kFloat32).sum(1, true)
            : species_w;
        auto table = [&](const char* name) {
            return owner.named_parameters()[std::string(name) + ".weight"];
        };
        auto mean = [](const torch::Tensor& ids, const torch::Tensor& t, const torch::Tensor& w) {
            return (torch::nn::functional::embedding(ids, t) * w.unsqueeze(-1)).sum(1);
        };
        const auto expected = torch::cat({
            mean(in.species_ids, table("species_embedding"), species_w),
            mean(in.pool_genus_ids, table("genus_embedding"), taxon_w),
            mean(in.pool_family_ids, table("family_embedding"), taxon_w),
            in.pool_has_cover.unsqueeze(-1)}, 1);
        CHECK(torch::allclose(out, expected, 1e-5, 1e-6));
    }
}

TEST_CASE("Count adds a transformer plot's taxa beside the attention",
          "[species_encoder][aggregation]") {
    auto spec = spec_for(SpeciesEncodingMode::Transformer);
    spec.aggregation = AggregationMode::Count;
    torch::nn::Module owner;
    auto features = make_transformer_features(owner, spec);
    owner.eval();
    const auto in = inputs_for(spec);
    torch::NoGradGuard no_grad;

    const auto with_taxa = features->forward(in, false);
    auto genus = owner.named_parameters()["genus_embedding.weight"];
    auto family = owner.named_parameters()["family_embedding.weight"];
    const auto genus_saved = genus.clone();
    const auto family_saved = family.clone();
    genus.zero_();
    family.zero_();
    const auto species_only = features->forward(in, false);

    // The attention never read the taxonomy, so the difference is exactly the
    // count-weighted mean of the genus and family rows.
    const auto mask = in.pool_mask.to(torch::kFloat32);
    const auto w = mask / mask.sum(1, true);
    auto mean = [&](const torch::Tensor& ids, const torch::Tensor& t) {
        return (torch::nn::functional::embedding(ids, t) * w.unsqueeze(-1)).sum(1);
    };
    const auto summary = mean(in.pool_genus_ids, genus_saved) +
                         mean(in.pool_family_ids, family_saved);
    const int64_t d = spec.config.d_model;
    CHECK(torch::allclose(with_taxa.slice(1, 0, d) - species_only.slice(1, 0, d), summary,
                          1e-5, 1e-5));
}

TEST_CASE("aggregation = count trains the pooled encodings end to end",
          "[species_encoder][aggregation]") {
    const SmallCorpus corpus;
    for (auto mode : {SpeciesEncodingMode::RankPool, SpeciesEncodingMode::Transformer}) {
        INFO(species_encoding_to_string(mode));
        auto by_count = load_corpus(corpus, mode, AggregationMode::Count);
        auto by_abundance = load_corpus(corpus, mode, AggregationMode::Abundance);
        REQUIRE(by_count.schema().dataset_config.aggregation == AggregationMode::Count);

        torch::manual_seed(4);
        ResolveModel count_model(by_count.schema(), small_encoder_config(mode, {8}));
        torch::manual_seed(4);
        ResolveModel abundance_model(by_abundance.schema(), small_encoder_config(mode, {8}));
        {
            count_model->eval();
            abundance_model->eval();
            torch::NoGradGuard no_grad;
            PlotInputs inputs = by_count.plot_inputs().without_empty();
            inputs.continuous =
                torch::zeros({inputs.n_plots(), by_count.schema().continuous_width()});
            CHECK_FALSE(torch::allclose(count_model->get_latent(inputs),
                                        abundance_model->get_latent(inputs)));
        }

        Trainer trainer(count_model, quiet_train_config(1));
        trainer.prepare_data(by_count);
        CHECK(std::isfinite(trainer.fit().train_loss_history.back()));
    }
}
