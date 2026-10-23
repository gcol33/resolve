// Parameter layout and outputs of every species encoding, encoder tail and
// adapter species branch, held fixed against recorded models.
//
// tests/fixtures/encoder_parity holds one archive per case: the weights of a
// model built from the case's schema and config (under the checkpoint's
// "param_" / "buffer_" keys), a batch of inputs ("in_<field>") and the model's
// eval outputs on them ("out_<target>", "out_latent"). A model built today from
// the same schema and config must load those weights -- every tensor present
// under its name, no extra one, each of the recorded shape -- and reproduce the
// outputs. A saved
// checkpoint of any of these configurations therefore keeps loading and
// predicting the same numbers.
//
// The hidden case [.encoder_parity_fixture] writes the archives.

#include <catch2/catch_test_macros.hpp>

#include "resolve/model.hpp"
#include "resolve/plot_inputs.hpp"
#include "test_support.hpp"

#include <torch/torch.h>

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

constexpr int64_t kPlots = 12;
constexpr int64_t kSpecies = 25;
constexpr int64_t kGenera = 7;
constexpr int64_t kFamilies = 5;
constexpr int64_t kPoolWidth = 6;
constexpr int kSlots = 2;
constexpr int kHashDim = 6;

struct ParityCase {
    const char* name;
    std::function<void(ResolveSchema&)> schema_edit;
    std::function<void(ModelConfig&)> config_edit;
};

ResolveSchema base_schema() {
    ResolveSchema schema;
    schema.n_plots = kPlots;
    schema.n_species = kSpecies;
    schema.n_species_vocab = kSpecies;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = kGenera;
    schema.n_families = kFamilies;
    schema.covariate_names = {"c1", "c2"};
    schema.dataset_config.track_unknown_fraction = true;
    schema.targets = {
        TargetConfig{"y", TaskType::Regression, TransformType::None, 0},
        TargetConfig{"cls", TaskType::Classification, TransformType::None, 3},
    };
    return schema;
}

ModelConfig base_config(SpeciesEncodingMode mode) {
    ModelConfig config;
    config.species_encoding = mode;
    config.hash_dim = kHashDim;
    config.species_embed_dim = 5;
    config.genus_emb_dim = 3;
    config.family_emb_dim = 2;
    config.top_k = kSlots;
    config.n_taxonomy_slots = kSlots;
    config.top_k_species = static_cast<int>(kPoolWidth);
    config.hidden_dims = {16, 8};
    config.dropout = 0.1f;
    config.d_model = 8;
    config.n_heads = 2;
    config.transformer_ff_dim = 16;
    return config;
}

const std::vector<ParityCase>& parity_cases() {
    using M = SpeciesEncodingMode;
    auto encoding = [](M mode) {
        return [mode](ModelConfig& c) { c = base_config(mode); };
    };
    auto with = [](M mode, std::function<void(ModelConfig&)> edit) {
        return [mode, edit](ModelConfig& c) {
            c = base_config(mode);
            edit(c);
        };
    };
    auto keep = [](ResolveSchema&) {};
    static const std::vector<ParityCase> cases = {
        {"hash", keep, encoding(M::Hash)},
        {"hash_no_taxonomy",
         [](ResolveSchema& s) {
             s.has_taxonomy = false;
             s.n_genera = 0;
             s.n_families = 0;
         },
         encoding(M::Hash)},
        {"embed", keep, encoding(M::Embed)},
        {"embed_categorical_head",
         [](ResolveSchema& s) {
             s.categorical_names = {"soil"};
             s.categorical_vocab_sizes = {4};
             s.categorical_embed_dim = 3;
         },
         with(M::Embed, [](ModelConfig& c) {
             c.categorical_embed_dim = 3;
             c.head_hidden_dims = {6};
             c.head_activation = ActivationType::SiLU;
         })},
        {"sparse", keep, encoding(M::Sparse)},
        {"hash_explicit_vector", keep,
         with(M::Hash, [](ModelConfig& c) { c.uses_explicit_vector = true; })},
        {"rank_pool", keep, encoding(M::RankPool)},
        {"rank_pool_moe_tail", keep, with(M::RankPool, [](ModelConfig& c) {
             c.hidden_dims = {16, 12, 8};
             c.moe_routing = MoERoutingType::Soft;
             c.n_experts = 3;
             c.expert_hidden_dims = {6};
         })},
        {"transformer_attention", keep,
         with(M::Transformer, [](ModelConfig& c) { c.n_attention_layers = 1; })},
        {"transformer_cls", keep, with(M::Transformer, [](ModelConfig& c) {
             c.n_attention_layers = 1;
             c.transformer_pooling = TransformerPooling::Cls;
         })},
        {"transformer_post_moe", keep, with(M::Transformer, [](ModelConfig& c) {
             c.moe_routing = MoERoutingType::TopK;
             c.moe_placement = MoEPlacement::Post;
             c.n_experts = 3;
             c.moe_top_k = 2;
             c.expert_hidden_dims = {6};
         })},
        {"hash_residual_batchnorm", keep,
         with(M::Hash, [](ModelConfig& c) { c.use_residual = true; })},
        {"sparse_residual_layernorm_prelu", keep, with(M::Sparse, [](ModelConfig& c) {
             c.use_residual = true;
             c.normalization = NormLayerType::LayerNorm;
             c.activation = ActivationType::PReLU;
         })},
        {"embed_residual_rmsnorm", keep, with(M::Embed, [](ModelConfig& c) {
             c.use_residual = true;
             c.normalization = NormLayerType::RMSNorm;
         })},
        {"rank_pool_residual_groupnorm", keep, with(M::RankPool, [](ModelConfig& c) {
             c.use_residual = true;
             c.normalization = NormLayerType::GroupNorm;
             c.norm_groups = 4;
         })},
        {"embed_tabm_layernorm", keep, with(M::Embed, [](ModelConfig& c) {
             c.tabm.enabled = true;
             c.tabm.n_ensembles = 4;
             c.normalization = NormLayerType::LayerNorm;
         })},
        {"sparse_parallel", keep, with(M::Sparse, [](ModelConfig& c) {
             ParallelBranchConfig branch;
             branch.hidden_dims = {8};
             c.parallel_layers.enabled = true;
             c.parallel_layers.branches = {branch, branch};
         })},
        {"ft_transformer_embed", keep, with(M::Embed, [](ModelConfig& c) {
             c.encoder_architecture = EncoderArchitecture::FTTransformer;
             c.ft_transformer.d_model = 8;
             c.ft_transformer.n_heads = 2;
             c.ft_transformer.n_layers = 1;
         })},
        {"tabnet_sparse", keep, with(M::Sparse, [](ModelConfig& c) {
             c.encoder_architecture = EncoderArchitecture::TabNet;
             c.tabnet.n_steps = 2;
             c.tabnet.n_d = 8;
             c.tabnet.n_a = 8;
             c.tabnet.virtual_batch_size = 4;
         })},
        {"gnn_hash", keep, with(M::Hash, [](ModelConfig& c) {
             c.encoder_architecture = EncoderArchitecture::GNN;
             c.gnn.hidden_dim = 8;
             c.gnn.n_layers = 2;
             c.gnn.n_heads = 2;
             c.gnn.k_neighbors = 3;
         })},
        {"saint_sparse", keep, with(M::Sparse, [](ModelConfig& c) {
             c.encoder_architecture = EncoderArchitecture::SAINT;
             c.saint.d_model = 8;
             c.saint.n_heads = 2;
             c.saint.n_layers = 1;
             c.saint.row_attention_rows = 4;
         })},
        {"hetero_gnn_sparse", keep, with(M::Sparse, [](ModelConfig& c) {
             c.encoder_architecture = EncoderArchitecture::HeterogeneousGNN;
             c.heterogeneous_gnn.hidden_dim = 8;
             c.heterogeneous_gnn.output_dim = 4;
             c.heterogeneous_gnn.n_layers = 1;
             c.heterogeneous_gnn.n_heads = 2;
         })},
    };
    return cases;
}

std::filesystem::path fixture_path(const ParityCase& c) {
    return std::filesystem::path(RESOLVE_TEST_FIXTURE_DIR) / "encoder_parity" /
           (std::string(c.name) + ".pt");
}

ResolveSchema case_schema(const ParityCase& c) {
    ResolveSchema schema = base_schema();
    c.schema_edit(schema);
    return schema;
}

ModelConfig case_config(const ParityCase& c) {
    ModelConfig config;
    c.config_edit(config);
    return config;
}

// A batch carrying every field any encoding reads, drawn from `gen`. Pool rows
// hold 2..kPoolWidth species, padded with code 0 at the end.
PlotInputs synthetic_inputs(const ResolveSchema& schema, const ModelConfig& config,
                            at::Generator& gen) {
    auto ints = [&](int64_t high, std::vector<int64_t> shape) {
        return torch::randint(1, high, shape, gen, torch::kLong);
    };
    PlotInputs in;
    const bool hash_block = config.species_encoding == SpeciesEncodingMode::Hash &&
                            !config.uses_explicit_vector;
    const int64_t width = schema.continuous_width() - schema.categorical_embed_width() +
                          (hash_block ? config.hash_dim : 0);
    in.continuous = torch::randn({kPlots, width}, gen);
    if (schema.has_taxonomy) {
        in.genus_ids = ints(kGenera, {kPlots, kSlots});
        in.family_ids = ints(kFamilies, {kPlots, kSlots});
    }

    auto lengths = torch::randint(2, kPoolWidth + 1, {kPlots, 1}, gen, torch::kLong);
    auto mask = torch::arange(kPoolWidth).unsqueeze(0) < lengths;
    in.pool_mask = mask;
    in.species_ids = ints(kSpecies, {kPlots, kPoolWidth}) * mask;
    if (schema.has_taxonomy) {
        in.pool_genus_ids = ints(kGenera, {kPlots, kPoolWidth}) * mask;
        in.pool_family_ids = ints(kFamilies, {kPlots, kPoolWidth}) * mask;
    }
    in.pool_weights = torch::rand({kPlots, kPoolWidth}, gen) * mask;
    in.pool_has_cover = torch::ones({kPlots});

    auto present = torch::rand({kPlots, kSpecies}, gen) < 0.3;
    in.species_vector = torch::rand({kPlots, kSpecies}, gen) * present;

    if (schema.has_categoricals()) {
        in.categorical_ids = torch::randint(0, schema.categorical_vocab_sizes[0],
                                            {kPlots, 1}, gen, torch::kLong);
    }
    return in;
}

ResolveModel build_model(const ResolveSchema& schema, const ModelConfig& config) {
    ResolveModel model(schema, config);
    if (model->requires_species_graph()) {
        // Every species links to the next two, typed in turn.
        std::vector<int64_t> src, dst, type;
        for (int64_t s = 1; s < kSpecies; ++s) {
            for (int64_t step = 1; step <= 2; ++step) {
                src.push_back(s);
                dst.push_back(1 + (s - 1 + step) % (kSpecies - 1));
                type.push_back((s + step) % 3);
            }
        }
        model->set_species_graph(torch::stack({torch::tensor(src), torch::tensor(dst)}),
                                 torch::tensor(type));
    }
    return model;
}

}  // namespace

TEST_CASE("Recorded encoder fixtures are written", "[.encoder_parity_fixture]") {
    std::filesystem::create_directories(std::filesystem::path(RESOLVE_TEST_FIXTURE_DIR) /
                                        "encoder_parity");
    for (const auto& c : parity_cases()) {
        torch::manual_seed(20261023);
        const ResolveSchema schema = case_schema(c);
        const ModelConfig config = case_config(c);
        auto gen = at::detail::createCPUGenerator(7);
        const PlotInputs inputs = synthetic_inputs(schema, config, gen);
        ResolveModel model = build_model(schema, config);

        // A few training-mode forwards move the normalization statistics and
        // capture SAINT's reference rows, so the recorded buffers are not the
        // initial ones.
        torch::NoGradGuard no_grad;
        model->train();
        for (int i = 0; i < 3; ++i) model->forward(inputs.conformed());
        model->eval();

        torch::serialize::OutputArchive archive;
        for (const auto& p : model->named_parameters()) archive.write("param_" + p.key(), p.value());
        for (const auto& b : model->named_buffers()) archive.write("buffer_" + b.key(), b.value());
        inputs.for_each([&](const PlotInputField& field, const torch::Tensor& t) {
            if (t.defined()) archive.write(std::string("in_") + field.name, t);
        });
        const PlotInputs conformed = inputs.conformed();
        for (const auto& [name, out] : model->forward(conformed)) {
            archive.write("out_" + name, out);
        }
        archive.write("out_latent", model->get_latent(conformed));
        archive.save_to(fixture_path(c).string());
    }
}

TEST_CASE("Every recorded encoder configuration loads and reproduces its outputs",
          "[encoder_parity]") {
    for (const auto& c : parity_cases()) {
        INFO("case " << c.name);
        const auto path = fixture_path(c);
        REQUIRE(std::filesystem::exists(path));
        torch::serialize::InputArchive archive;
        archive.load_from(path.string(), torch::kCPU);

        ResolveModel model = build_model(case_schema(c), case_config(c));

        // Every tensor under its recorded name and shape (a data buffer takes
        // the shape it was recorded with), and no recorded tensor left over.
        const auto data_buffers = model->data_buffers();
        auto is_data_buffer = [&](const torch::Tensor& t) {
            for (const auto& d : data_buffers) {
                if (d.is_same(t)) return true;
            }
            return false;
        };
        int64_t n_model = 0;
        auto load = [&](const std::string& key, const torch::Tensor& t) {
            ++n_model;
            torch::Tensor recorded;
            INFO(key);
            REQUIRE(archive.try_read(key, recorded));
            torch::NoGradGuard no_grad;
            if (is_data_buffer(t)) {
                t.set_data(recorded.to(t.scalar_type()));
                return;
            }
            REQUIRE(recorded.sizes() == t.sizes());
            t.copy_(recorded);
        };
        for (const auto& p : model->named_parameters()) load("param_" + p.key(), p.value());
        for (const auto& b : model->named_buffers()) load("buffer_" + b.key(), b.value());
        int64_t n_recorded = 0;
        for (const auto& key : archive.keys()) {
            if (key.rfind("param_", 0) == 0 || key.rfind("buffer_", 0) == 0) ++n_recorded;
        }
        CHECK(n_recorded == n_model);
        model->eval();

        PlotInputs inputs;
        inputs.for_each([&](const PlotInputField& field, torch::Tensor& t) {
            torch::Tensor recorded;
            if (archive.try_read(std::string("in_") + field.name, recorded)) t = recorded;
        });
        inputs = inputs.conformed();

        torch::NoGradGuard no_grad;
        auto check_close = [&](const std::string& key, const torch::Tensor& now) {
            torch::Tensor recorded;
            REQUIRE(archive.try_read(key, recorded));
            INFO(key << " max |diff| " << (now - recorded).abs().max().item<double>());
            CHECK(torch::allclose(now, recorded, /*rtol=*/1e-5, /*atol=*/1e-6));
        };
        for (const auto& [name, out] : model->forward(inputs)) check_close("out_" + name, out);
        check_close("out_latent", model->get_latent(inputs));
    }
}
