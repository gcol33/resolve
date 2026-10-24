// Mixed-precision training (TrainConfig::use_amp) on CUDA for every species
// encoding and every attention or message-passing architecture: one epoch
// under fp16 autocast, then a prediction from the saved checkpoint. The
// transformer species encoding's masked self-attention and attention pooling
// run in float32 inside the autocast region (run_fp32_under_autocast), since
// libtorch's multi_head_attention_forward has no Half kernel for its
// key-padding fill. Every case skips without a CUDA device.

#include <catch2/catch_test_macros.hpp>

#include "resolve/dataset.hpp"
#include "resolve/encoder.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <ATen/autocast_mode.h>

#include <cmath>
#include <string>

using namespace resolve;
using namespace resolve::test;

namespace {

TrainConfig amp_train_config() {
    TrainConfig cfg = quiet_train_config(1);
    cfg.device = torch::Device(torch::kCUDA, 0);
    cfg.use_amp = true;
    return cfg;
}

// One AMP epoch of `model` on `ds`, then a prediction on every plot from the
// checkpoint; the loss and every prediction are finite.
void require_amp_fit_and_predict(const ResolveDataset& ds, const ModelConfig& model) {
    torch::manual_seed(1);
    Trainer trainer(ResolveModel(ds.schema(), model), amp_train_config());
    trainer.prepare_data(ds, 0.25f, 3);
    const auto result = trainer.fit();
    REQUIRE(result.train_loss_history.size() == 1);
    CHECK(std::isfinite(result.train_loss_history.front()));
    REQUIRE(result.test_loss_history.size() == 1);
    CHECK(std::isfinite(result.test_loss_history.front()));

    const TempPath checkpoint(".pt");
    trainer.save(checkpoint.path());
    auto predictor = Predictor::load(checkpoint.path(), torch::Device(torch::kCUDA, 0));
    const auto predictions = predictor.predict(ds);
    const auto y = predictions.predictions.at("y");
    REQUIRE(y.size(0) == ds.schema().n_plots);
    CHECK(torch::isfinite(y).all().item<bool>());
}

ModelConfig transformer_config(TransformerPooling pooling, int n_layers) {
    ModelConfig model = small_encoder_config(SpeciesEncodingMode::Transformer, {16, 8});
    model.d_model = 16;
    model.n_heads = 4;
    model.n_attention_layers = n_layers;
    model.transformer_ff_dim = 32;
    model.transformer_pooling = pooling;
    return model;
}

}  // namespace

TEST_CASE("run_fp32_under_autocast runs float32 inside autocast and is a no-op outside",
          "[amp][cuda]") {
    if (!torch::cuda::is_available()) SKIP("needs a CUDA device");
    const auto x = torch::randn({4, 3}, torch::Device(torch::kCUDA, 0));
    const auto w = torch::randn({3, 2}, torch::Device(torch::kCUDA, 0));
    const auto matmul = [](const std::vector<torch::Tensor>& in) {
        return torch::matmul(in[0], in[1]);
    };

    // Outside autocast the inputs reach the function as they are.
    const auto plain = run_fp32_under_autocast(matmul, {x, w});
    CHECK(torch::equal(plain, torch::matmul(x, w)));

    at::autocast::set_autocast_enabled(at::kCUDA, true);
    at::autocast::increment_nesting();
    const auto half_x = x.to(torch::kHalf);
    const auto inside = run_fp32_under_autocast(matmul, {half_x, w});
    const bool still_enabled = at::autocast::is_autocast_enabled(at::kCUDA);
    const auto autocast_matmul = torch::matmul(x, w);
    at::autocast::decrement_nesting();
    at::autocast::set_autocast_enabled(at::kCUDA, false);
    at::autocast::clear_cache();

    CHECK(inside.scalar_type() == torch::kFloat32);
    CHECK(torch::equal(inside, torch::matmul(half_x.to(torch::kFloat32), w)));
    CHECK(still_enabled);
    CHECK(autocast_matmul.scalar_type() == torch::kHalf);
}

TEST_CASE("Every species encoding trains and predicts under mixed precision", "[amp][cuda]") {
    if (!torch::cuda::is_available()) SKIP("needs a CUDA device");
    const SmallCorpus corpus;
    for (const auto mode : all_species_encodings()) {
        INFO("encoding " << species_encoding_to_string(mode));
        require_amp_fit_and_predict(corpus.load(mode),
                                    small_encoder_config(mode, {16, 8}));
    }
}

TEST_CASE("The transformer species encoding trains and predicts under mixed precision",
          "[amp][cuda][transformer]") {
    if (!torch::cuda::is_available()) SKIP("needs a CUDA device");
    const SmallCorpus corpus;
    const auto ds = corpus.load(SpeciesEncodingMode::Transformer);

    SECTION("attention pooling over two self-attention layers") {
        require_amp_fit_and_predict(ds, transformer_config(TransformerPooling::Attention, 2));
    }
    SECTION("attention pooling with no self-attention layer") {
        require_amp_fit_and_predict(ds, transformer_config(TransformerPooling::Attention, 0));
    }
    SECTION("CLS pooling") {
        require_amp_fit_and_predict(ds, transformer_config(TransformerPooling::Cls, 2));
    }
    SECTION("feeding an FT-Transformer") {
        ModelConfig model = transformer_config(TransformerPooling::Attention, 1);
        model.encoder_architecture = EncoderArchitecture::FTTransformer;
        model.ft_transformer.d_model = 16;
        model.ft_transformer.n_heads = 2;
        model.ft_transformer.n_layers = 1;
        require_amp_fit_and_predict(ds, model);
    }
}

TEST_CASE("The attention and graph architectures train and predict under mixed precision",
          "[amp][cuda]") {
    if (!torch::cuda::is_available()) SKIP("needs a CUDA device");
    const SmallCorpus corpus;
    const auto hash = corpus.load(SpeciesEncodingMode::Hash);
    ModelConfig model = small_encoder_config(SpeciesEncodingMode::Hash, {16, 8});

    SECTION("FT-Transformer") {
        model.encoder_architecture = EncoderArchitecture::FTTransformer;
        model.ft_transformer.d_model = 16;
        model.ft_transformer.n_heads = 2;
        model.ft_transformer.n_layers = 1;
        require_amp_fit_and_predict(hash, model);
    }
    SECTION("SAINT") {
        model.encoder_architecture = EncoderArchitecture::SAINT;
        model.saint.d_model = 16;
        model.saint.n_heads = 2;
        model.saint.n_layers = 1;
        model.saint.row_attention_rows = 8;
        require_amp_fit_and_predict(hash, model);
    }
    SECTION("ExcelFormer") {
        model.encoder_architecture = EncoderArchitecture::ExcelFormer;
        model.excelformer.d_model = 16;
        model.excelformer.n_heads = 2;
        model.excelformer.n_layers = 1;
        require_amp_fit_and_predict(hash, model);
    }
    SECTION("TabNet") {
        model.encoder_architecture = EncoderArchitecture::TabNet;
        model.tabnet.n_steps = 2;
        model.tabnet.n_d = 8;
        model.tabnet.n_a = 8;
        model.tabnet.virtual_batch_size = 0;
        require_amp_fit_and_predict(hash, model);
    }
    SECTION("GAT plot graph") {
        model.encoder_architecture = EncoderArchitecture::GNN;
        model.gnn.gnn_type = GNNType::GAT;
        model.gnn.n_layers = 2;
        model.gnn.hidden_dim = 16;
        model.gnn.n_heads = 2;
        model.gnn.k_neighbors = 4;
        require_amp_fit_and_predict(hash, model);
    }
    SECTION("heterogeneous species graph") {
        const auto sparse = corpus.load(SpeciesEncodingMode::Sparse);
        ModelConfig hetero = small_encoder_config(SpeciesEncodingMode::Sparse, {16, 8});
        hetero.encoder_architecture = EncoderArchitecture::HeterogeneousGNN;
        hetero.heterogeneous_gnn.hidden_dim = 16;
        hetero.heterogeneous_gnn.output_dim = 8;
        hetero.heterogeneous_gnn.n_layers = 2;
        hetero.heterogeneous_gnn.n_heads = 2;
        require_amp_fit_and_predict(sparse, hetero);
    }
}
