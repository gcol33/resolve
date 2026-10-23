// Model buffers that hold training data rather than fixed-shape state: the
// TraitNet trait matrix and SAINT's row-attention reference set. Each follows
// the model's device, is saved with the weights and comes back from a load at
// the shape it was saved with. (The heterogeneous GNN's species graph, the
// third such buffer, is covered in test_species_graph.cpp.)

#include <catch2/catch_test_macros.hpp>

#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <torch/torch.h>

#include <set>
#include <sstream>
#include <string>

using namespace resolve;
using namespace resolve::test;

namespace {

ResolveSchema trait_schema(int64_t n_species) {
    ResolveSchema schema;
    schema.n_plots = 32;
    schema.n_species = n_species;
    schema.has_coordinates = true;
    schema.covariate_names = {"elev"};
    schema.track_unknown_fraction = false;
    schema.targets.push_back({"y", TaskType::Regression, TransformType::None, 0, 1.0f});
    return schema;
}

ModelConfig trait_config(int64_t trait_dim) {
    ModelConfig config;
    config.encoder_architecture = EncoderArchitecture::TraitNet;
    config.dropout = 0.0f;
    config.hidden_dims = {16, 8};
    config.trait_net.env_dim = 12;
    config.trait_net.trait_dim = static_cast<int>(trait_dim);
    config.trait_net.interaction_dim = 8;
    return config;
}

bool has_buffer(const torch::nn::Module& module, const std::string& name) {
    for (const auto& named : module.named_buffers()) {
        if (named.key() == name) return true;
    }
    return false;
}

// A hash-encoded dataset with coordinates, two covariates and a regression
// target, `n_plots` plots of one species each.
ResolveDataset hash_dataset(int64_t n_plots) {
    std::ostringstream header;
    header << "plot_id,lat,lon,cov1,cov2,y\n";
    std::ostringstream species;
    species << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = static_cast<double>(i % 7) * 1.1;
        const double c2 = static_cast<double>(i % 11) * 0.9;
        header << "P" << i << "," << 40.0 + 0.1 * static_cast<double>(i) << ","
               << -5.0 + 0.05 * static_cast<double>(i) << "," << c1 << "," << c2 << ","
               << c1 + c2 << "\n";
        species << "P" << i << ",sp" << (i % 13) << ",1.0\n";
    }
    TempFile header_csv(header.str());
    TempFile species_csv(species.str());
    return ResolveDataset::from_csv(header_csv.path(), species_csv.path(),
                                    Roles().coordinates().covariates({"cov1", "cov2"}),
                                    {TargetSpec::regression("y")}, hash_dataset_config());
}

ModelConfig saint_config(int row_attention_rows) {
    auto config = hash_mlp_config({8});
    config.encoder_architecture = EncoderArchitecture::SAINT;
    config.saint.d_model = 16;
    config.saint.n_heads = 2;
    config.saint.n_layers = 1;
    config.saint.attention_dropout = 0.0f;
    config.saint.row_attention_rows = row_attention_rows;
    return config;
}

}  // namespace

// ============================================================================
// TraitNet trait matrix
// ============================================================================

TEST_CASE("The TraitNet trait matrix is a model buffer", "[traitnet][data_buffer]") {
    const int64_t n_species = 5, trait_dim = 4;
    ResolveModel model(trait_schema(n_species), trait_config(trait_dim));
    REQUIRE(has_buffer(*model, "trait_net_encoder.traits"));
    REQUIRE(model->data_buffers().size() == 1);
    CHECK(model->data_buffers().front().size(0) == 0);

    auto traits = torch::randn({n_species, trait_dim});
    model->set_traits(traits);
    CHECK(torch::equal(model->data_buffers().front(), traits));

    SECTION("a wrong shape is refused") {
        CHECK_THROWS(model->set_traits(torch::randn({n_species + 1, trait_dim})));
        CHECK_THROWS(model->set_traits(torch::randn({n_species, trait_dim + 1})));
    }
    SECTION("it follows the model to a CUDA device") {
        if (!torch::cuda::is_available()) SKIP("needs a CUDA device");
        model->to(torch::kCUDA);
        CHECK(model->data_buffers().front().is_cuda());
        auto out = model->forward({.continuous = torch::randn({4, 3}, torch::kCUDA)}).at("y");
        CHECK(out.is_cuda());
    }
}

TEST_CASE("The TraitNet trait matrix travels in the checkpoint", "[traitnet][data_buffer][checkpoint]") {
    const int64_t n_species = 5, trait_dim = 4;
    torch::manual_seed(41);
    ResolveModel model(trait_schema(n_species), trait_config(trait_dim));
    model->set_traits(torch::randn({n_species, trait_dim}));
    model->eval();
    auto env = torch::randn({6, 3});
    torch::Tensor expected;
    {
        torch::NoGradGuard no_grad;
        expected = model->forward({.continuous = env}).at("y");
    }

    const TempPath ckpt(".pt");
    Trainer(model, quiet_train_config()).save(ckpt.path());

    for (const torch::Device device : {torch::Device(torch::kCPU), torch::Device(torch::kCUDA)}) {
        if (device.is_cuda() && !torch::cuda::is_available()) continue;
        INFO("device " << device);
        auto [loaded, scalers, vocab] = Trainer::load(ckpt.path(), device);
        loaded->eval();
        torch::NoGradGuard no_grad;
        auto out = loaded->forward({.continuous = env.to(device)}).at("y");
        CHECK(torch::allclose(out.cpu(), expected, 1e-5, 1e-6));
    }
}

// ============================================================================
// SAINT reference rows
// ============================================================================

TEST_CASE("SAINT keeps the leading rows of its largest training batches as the reference",
          "[saint][data_buffer]") {
    auto ds = hash_dataset(40);
    ResolveModel model(ds.schema(), saint_config(/*row_attention_rows=*/5));
    REQUIRE(model->row_reference_rows() == 4);
    auto reference_rows = [&] {
        return model->data_buffers().front().size(0);  // continuous comes first
    };
    REQUIRE(reference_rows() == 0);

    auto inputs = ds.plot_inputs();
    inputs.continuous = ds.continuous_block(/*include_hash=*/true);
    model->train();
    model->forward(inputs.slice(0, 10));
    CHECK(reference_rows() == 4);
    CHECK(torch::equal(model->data_buffers().front(), inputs.continuous.slice(0, 0, 4)));

    // A batch smaller than the set held leaves it alone; a full one refreshes it.
    model->forward(inputs.slice(10, 12));
    CHECK(torch::equal(model->data_buffers().front(), inputs.continuous.slice(0, 0, 4)));
    model->forward(inputs.slice(20, 30));
    CHECK(torch::equal(model->data_buffers().front(), inputs.continuous.slice(0, 20, 24)));

    // An eval forward reads the reference and leaves it alone.
    model->eval();
    torch::NoGradGuard no_grad;
    model->forward(inputs.slice(0, 10));
    CHECK(torch::equal(model->data_buffers().front(), inputs.continuous.slice(0, 20, 24)));
}

TEST_CASE("A SAINT prediction does not depend on which plots share its chunk",
          "[saint][predictor]") {
    auto ds = hash_dataset(60);
    torch::manual_seed(43);
    ResolveModel model(ds.schema(), saint_config(/*row_attention_rows=*/8));
    TrainConfig train = quiet_train_config(/*epochs=*/1);
    train.batch_size = 16;
    Trainer trainer(model, train);
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/0);
    trainer.fit();
    REQUIRE(model->data_buffers().front().size(0) == 7);

    const TempPath ckpt(".pt");
    trainer.save(ckpt.path());
    auto predictor = Predictor::load(ckpt.path(), torch::kCPU);

    auto one_shot = predictor.predict(ds, /*return_latent=*/false, /*batch_size=*/-1);
    for (int64_t batch_size : {1, 7, 25}) {
        INFO("batch_size " << batch_size);
        auto chunked = predictor.predict(ds, /*return_latent=*/false, batch_size);
        CHECK(torch::allclose(chunked.predictions.at("y"), one_shot.predictions.at("y"),
                              1e-5, 1e-6));
    }

    // The reference set came back with the weights: the model in memory and the
    // loaded one predict alike.
    model->eval();
    torch::NoGradGuard no_grad;
    auto inputs = ds.plot_inputs();
    inputs.continuous = ds.continuous_block(/*include_hash=*/true);
    auto direct = model->forward(inputs).at("y");
    auto reloaded = std::get<0>(Trainer::load(ckpt.path(), torch::kCPU));
    reloaded->eval();
    CHECK(torch::allclose(reloaded->forward(inputs).at("y"), direct, 1e-5, 1e-6));
}
