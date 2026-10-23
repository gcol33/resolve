// Attention-based tabular encoders as standalone modules: the shared
// transformer blocks (attention, tokenizer), FT-Transformer, TabNet, SAINT and
// ExcelFormer.

#include <catch2/catch_test_macros.hpp>

#include "resolve/excelformer.hpp"
#include "resolve/ft_transformer.hpp"
#include "resolve/saint.hpp"
#include "resolve/tabnet.hpp"

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include <cmath>
#include <set>
#include <string>
#include <vector>

using namespace resolve;

namespace {

// The named parameter of `module` whose name ends in `suffix`.
torch::Tensor named_parameter(const torch::nn::Module& module, const std::string& suffix) {
    for (const auto& named : module.named_parameters()) {
        const auto& key = named.key();
        if (key.size() >= suffix.size() &&
            key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return named.value();
        }
    }
    return {};
}

// Scaled dot-product attention written out: softmax(Q K^T / sqrt(d_k) + bias) V
// through the module's own projections.
torch::Tensor reference_attention(const torch::nn::Module& mha, const torch::Tensor& query,
                                  const torch::Tensor& key, const torch::Tensor& value,
                                  int64_t n_heads, const torch::Tensor& bias) {
    auto project = [&](const char* name, const torch::Tensor& x) {
        return torch::nn::functional::linear(x, named_parameter(mha, std::string(name) + ".weight"),
                                             named_parameter(mha, std::string(name) + ".bias"));
    };
    const int64_t batch = query.size(0);
    const int64_t d_model = query.size(2);
    const int64_t d_k = d_model / n_heads;
    auto heads = [&](const torch::Tensor& x) {
        return x.view({batch, x.size(1), n_heads, d_k}).transpose(1, 2);
    };
    auto Q = heads(project("W_q", query));
    auto K = heads(project("W_k", key));
    auto V = heads(project("W_v", value));
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt(static_cast<double>(d_k));
    if (bias.defined()) scores = scores + bias;
    auto context = torch::matmul(torch::softmax(scores, -1), V);
    return project("W_o", context.transpose(1, 2).contiguous().view({batch, query.size(1), d_model}));
}

const std::vector<int64_t> kNoCategoricals;

}  // namespace

// ============================================================================
// Shared blocks
// ============================================================================

TEST_CASE("MultiHeadAttention is scaled dot-product attention", "[attention]") {
    torch::manual_seed(3);
    MultiHeadAttention mha(16, 4);
    mha->eval();
    auto q = torch::randn({3, 5, 16});
    auto kv = torch::randn({3, 7, 16});

    SECTION("without a bias") {
        auto out = mha->forward(q, kv, kv);
        auto expected = reference_attention(*mha, q, kv, kv, 4, {});
        CHECK(torch::allclose(out, expected, 1e-5, 1e-5));
    }
    SECTION("an additive bias is added to the scores, gradient included") {
        auto bias = torch::randn({1, 5, 7}, torch::requires_grad());
        auto out = mha->forward(q, kv, kv, bias);
        auto expected = reference_attention(*mha, q, kv, kv, 4, bias.detach().unsqueeze(1));
        CHECK(torch::allclose(out, expected, 1e-5, 1e-5));
        out.sum().backward();
        REQUIRE(bias.grad().defined());
        CHECK(bias.grad().abs().sum().item<float>() > 0.0f);
    }
    SECTION("a key-padding bias of shape (batch, seq_len_k)") {
        auto bias = torch::zeros({3, 7});
        bias.index_put_({torch::indexing::Slice(), torch::indexing::Slice(5)}, -1e9);
        auto out = mha->forward(q, kv, kv, bias);
        auto expected = reference_attention(*mha, q, kv.narrow(1, 0, 5), kv.narrow(1, 0, 5), 4, {});
        CHECK(torch::allclose(out, expected, 1e-5, 1e-5));
    }
}

TEST_CASE("Attending over a memory plus itself is attention over the memory with the row appended",
          "[attention][saint]") {
    torch::manual_seed(4);
    MultiHeadAttention mha(8, 2);
    mha->eval();
    auto memory = torch::randn({3, 6, 8});
    auto queries = torch::randn({3, 4, 8});

    auto out = mha->attend_memory_and_self(queries, memory);
    REQUIRE(out.sizes() == queries.sizes());
    for (int64_t i = 0; i < queries.size(1); ++i) {
        auto row = queries.narrow(1, i, 1);
        auto keys = torch::cat({memory, row}, 1);
        auto expected = mha->forward(row, keys, keys);
        CHECK(torch::allclose(out.narrow(1, i, 1), expected, 1e-5, 1e-5));
    }

    SECTION("an empty memory leaves the row attending to itself alone") {
        auto alone = mha->attend_memory_and_self(queries, memory.narrow(1, 0, 0));
        for (int64_t i = 0; i < queries.size(1); ++i) {
            auto row = queries.narrow(1, i, 1);
            CHECK(torch::allclose(alone.narrow(1, i, 1), mha->forward(row, row, row), 1e-5, 1e-5));
        }
    }
}

TEST_CASE("FeatureTokenizer tokenizes every feature in one product and one table",
          "[attention][tokenizer]") {
    torch::manual_seed(5);
    FeatureTokenizer tokenizer(3, std::vector<int64_t>{4, 6}, 8);
    REQUIRE(tokenizer->n_tokens() == 1 + 3 + 2);

    auto numerical = torch::randn({5, 3});
    auto categoricals = torch::tensor({{0, 5}, {3, 1}, {2, 0}, {1, 2}, {3, 5}}, torch::kInt64);
    auto tokens = tokenizer->forward(numerical, categoricals);
    REQUIRE(tokens.sizes() == torch::IntArrayRef({5, 6, 8}));

    auto weight = named_parameter(*tokenizer, "numerical_weight");
    auto bias = named_parameter(*tokenizer, "numerical_bias");
    auto table = named_parameter(*tokenizer, "categorical_embedding.weight");
    REQUIRE(table.size(0) == 4 + 6);

    // Token 0 is CLS, the same for every row.
    CHECK(torch::allclose(tokens.select(1, 0), tokens.select(1, 0)[0].expand({5, 8})));
    for (int64_t j = 0; j < 3; ++j) {
        auto expected = numerical.select(1, j).unsqueeze(1) * weight[j] + bias[j];
        CHECK(torch::allclose(tokens.select(1, 1 + j), expected));
    }
    // Categorical feature 1 reads the rows after feature 0's four.
    CHECK(torch::allclose(tokens.select(1, 4), table.index_select(0, categoricals.select(1, 0))));
    CHECK(torch::allclose(tokens.select(1, 5),
                          table.index_select(0, categoricals.select(1, 1) + 4)));

    SECTION("a categorical count that does not match the tables is refused") {
        CHECK_THROWS(tokenizer->forward(numerical));
        CHECK_THROWS(tokenizer->forward(numerical, categoricals.narrow(1, 0, 1)));
    }
}

// ============================================================================
// FT-Transformer, TabNet, SAINT
// ============================================================================

TEST_CASE("FTTransformerEncoder forward shape", "[attention]") {
    FTTransformerEncoder encoder(10, kNoCategoricals, TransformerBlockConfig::uniform(64, 4, 0, 0.1f),
                                 /*n_layers=*/2);
    auto out = encoder->forward(torch::randn({8, 10}));

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);  // d_model
}

TEST_CASE("TabNetEncoder forward shape and its sparsity regularizer", "[attention][tabnet]") {
    TabNetEncoder encoder(32, 3, 16, 16, 1.5f);
    auto result = encoder->forward(torch::randn({8, 32}));

    REQUIRE(result.output.size(0) == 8);
    REQUIRE(result.output.size(1) == 16);  // n_d
    REQUIRE(result.mask_entropy.dim() == 0);
    // Each step's mask is on the simplex over 32 features, so its entropy is
    // within [0, log 32].
    const float entropy = result.mask_entropy.item<float>();
    CHECK(entropy >= 0.0f);
    CHECK(entropy <= std::log(32.0f) + 1e-4f);
}

TEST_CASE("TabNet normalizes through the fp32 guard under autocast", "[tabnet][amp][gpu]") {
    if (!torch::cuda::is_available()) SKIP("needs a CUDA device");
    TabNetEncoder encoder(16, 2, 8, 8, 1.5f, true, /*virtual_batch_size=*/4);
    encoder->to(torch::kCUDA);
    auto x = torch::randn({12, 16}, torch::kCUDA);
    {
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        auto result = encoder->forward(x);
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        at::autocast::clear_cache();
        CHECK(torch::isfinite(result.output).all().item<bool>());
    }
    for (const auto& named : encoder->named_buffers()) {
        if (named.key().find("running_") == std::string::npos) continue;
        INFO(named.key());
        CHECK(named.value().scalar_type() == torch::kFloat32);
        CHECK(torch::isfinite(named.value()).all().item<bool>());
    }
}

TEST_CASE("SAINTEncoder forward shape", "[attention][saint]") {
    SAINTEncoder encoder(10, kNoCategoricals, TransformerBlockConfig::uniform(64, 4, 0, 0.1f),
                         /*n_layers=*/2, /*row_attention_rows=*/8);
    auto out = encoder->forward(torch::randn({8, 10}));

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);  // d_model
    REQUIRE(encoder->reference_rows() == 7);
}

TEST_CASE("SAINT inference does not depend on which rows share the batch", "[saint]") {
    torch::manual_seed(11);
    SAINTEncoder encoder(6, std::vector<int64_t>{3}, TransformerBlockConfig::uniform(16, 2, 0, 0.0f),
                         /*n_layers=*/2, /*row_attention_rows=*/5);
    encoder->eval();
    torch::NoGradGuard no_grad;

    auto numerical = torch::randn({9, 6});
    auto codes = torch::randint(0, 3, {9, 1}, torch::kInt64);
    auto ref_numerical = torch::randn({4, 6});
    auto ref_codes = torch::randint(0, 3, {4, 1}, torch::kInt64);

    auto whole = encoder->forward(numerical, codes, ref_numerical, ref_codes);
    SECTION("each row alone gives its row of the whole batch") {
        for (int64_t i = 0; i < 9; ++i) {
            auto alone = encoder->forward(numerical.narrow(0, i, 1), codes.narrow(0, i, 1),
                                          ref_numerical, ref_codes);
            CHECK(torch::allclose(alone, whole.narrow(0, i, 1), 1e-5, 1e-6));
        }
    }
    SECTION("a reshuffled batch gives the same rows") {
        auto order = torch::randperm(9, torch::kInt64);
        auto shuffled = encoder->forward(numerical.index_select(0, order),
                                         codes.index_select(0, order), ref_numerical, ref_codes);
        CHECK(torch::allclose(shuffled, whole.index_select(0, order), 1e-5, 1e-6));
    }
    SECTION("the reference set is what the rows attend to") {
        auto other = encoder->forward(numerical, codes, torch::randn({4, 6}), ref_codes);
        CHECK_FALSE(torch::allclose(other, whole, 1e-4, 1e-5));
    }
    SECTION("without a reference set each row attends to itself alone") {
        auto none = encoder->forward(numerical, codes);
        for (int64_t i = 0; i < 9; ++i) {
            CHECK(torch::allclose(encoder->forward(numerical.narrow(0, i, 1), codes.narrow(0, i, 1)),
                                  none.narrow(0, i, 1), 1e-5, 1e-6));
        }
    }
}

TEST_CASE("SAINT training attends within groups of row_attention_rows rows", "[saint]") {
    torch::manual_seed(12);
    SAINTEncoder encoder(4, kNoCategoricals, TransformerBlockConfig::uniform(8, 2, 0, 0.0f),
                         /*n_layers=*/1, /*row_attention_rows=*/3);
    encoder->train();

    auto numerical = torch::randn({7, 4});
    auto out = encoder->forward(numerical);
    // Rows 3-5 form the second group: changing a row of the first group or the
    // trailing single-row group leaves them alone.
    auto changed = numerical.clone();
    changed[0] = torch::randn({4});
    changed[6] = torch::randn({4});
    auto out_changed = encoder->forward(changed);
    CHECK(torch::allclose(out.narrow(0, 3, 3), out_changed.narrow(0, 3, 3), 1e-5, 1e-6));
    CHECK_FALSE(torch::allclose(out.narrow(0, 1, 2), out_changed.narrow(0, 1, 2), 1e-5, 1e-6));
}

// ============================================================================
// ExcelFormer
// ============================================================================

TEST_CASE("ExcelFormerEncoder forward shape", "[excelformer]") {
    ExcelFormerEncoder encoder(10, kNoCategoricals, TransformerBlockConfig::uniform(64, 4, 0, 0.1f),
                               /*n_layers=*/2, 0.5f);
    auto out = encoder->forward(torch::randn({8, 10}));

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);  // d_model
}

TEST_CASE("ExcelFormerEncoder with categoricals", "[excelformer]") {
    ExcelFormerEncoder encoder(8, std::vector<int64_t>{5, 10}, TransformerBlockConfig::uniform(64, 4, 0, 0.1f),
                               /*n_layers=*/2, 0.5f);
    auto categoricals = torch::stack({torch::randint(0, 5, {8}), torch::randint(0, 10, {8})}, 1);
    auto out = encoder->forward(torch::randn({8, 8}), categoricals);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);
}

TEST_CASE("ExcelFormerEncoder runs its blocks through the shared transformer stack",
          "[excelformer]") {
    ExcelFormerEncoder encoder(4, kNoCategoricals, TransformerBlockConfig::uniform(16, 2, 0, 0.1f),
                               /*n_layers=*/2, 0.5f);
    std::set<std::string> names;
    for (const auto& named : encoder->named_parameters()) names.insert(named.key());
    CHECK(names.count("encoder.layers.0.attention.W_q.weight") == 1);
    CHECK(names.count("encoder.layers.1.ffn.linear2.bias") == 1);
    CHECK(names.count("encoder.final_norm.weight") == 1);
    CHECK(names.count("importance_logits") == 1);
}

TEST_CASE("ExcelFormerEncoder importance_logits receive gradient", "[excelformer]") {
    // The semi-permeable mask must be differentiable so the learnable
    // importance parameter actually trains. A hard boolean mask leaves it
    // gradient-dead (importance stuck at its init, mask permanently open).
    ExcelFormerEncoder encoder(10, kNoCategoricals, TransformerBlockConfig::uniform(32, 4, 0, 0.1f),
                               /*n_layers=*/2, 0.5f);
    auto importance_logits = named_parameter(*encoder, "importance_logits");
    REQUIRE(importance_logits.defined());

    auto out = encoder->forward(torch::randn({8, 10}));
    out.pow(2).mean().backward();

    REQUIRE(importance_logits.grad().defined());
    REQUIRE(importance_logits.grad().abs().sum().item<float>() > 0.0f);
}

TEST_CASE("ExcelFormerEncoder does not double-register block parameters",
          "[excelformer][params]") {
    // libtorch C++ named_parameters() does not dedupe shared submodules, so a
    // block registered twice would alias its storage under two paths.
    ExcelFormerEncoder encoder(10, kNoCategoricals, TransformerBlockConfig::uniform(64, 4, 0, 0.1f),
                               /*n_layers=*/2, 0.5f);

    std::set<void*> storages;
    for (const auto& named : encoder->named_parameters()) {
        void* ptr = named.value().data_ptr();
        INFO("duplicate parameter storage under: " << named.key());
        REQUIRE(storages.insert(ptr).second);
    }
}
