// Graph encoders as standalone modules: the coordinate-kNN GNNEncoder and its
// adjacency, the typed message-passing layer, and the heterogeneous species-graph
// encoder built from it.

#include <catch2/catch_test_macros.hpp>

#include "resolve/attention.hpp"

#include <torch/torch.h>

#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

using namespace resolve;

// ============================================================================
// GNNEncoder and its kNN adjacency
// ============================================================================

TEST_CASE("GNNEncoder forward shape", "[attention]") {
    GNNEncoder encoder(16, 32, 16, 2, GNNEncoderImpl::GNNType::GCN, 4, 0.0f);

    int n_nodes = 10;
    auto x = torch::randn({n_nodes, 16});
    auto adj = torch::eye(n_nodes) + torch::randn({n_nodes, n_nodes}).abs() * 0.1f;

    auto out = encoder->forward(x, adj);

    REQUIRE(out.size(0) == n_nodes);
    REQUIRE(out.size(1) == 16);  // out_features
}

TEST_CASE("build_knn_adjacency adds self-loops", "[gnn][adjacency]") {
    // Every node must have a non-zero diagonal so GAT's masked softmax always
    // has at least the self entry (an all-masked row would be NaN).
    SECTION("multi-node graph") {
        auto coords = torch::randn({6, 2});
        auto adj = build_knn_adjacency(coords, 2);
        auto diag = adj.diagonal();
        REQUIRE((diag > 0).all().item<bool>());
    }
    SECTION("single-node graph returns a self-loop, not zeros") {
        auto coords = torch::randn({1, 2});
        auto adj = build_knn_adjacency(coords, 3);
        REQUIRE(adj.size(0) == 1);
        REQUIRE(adj[0][0].item<float>() > 0.0f);
    }
}

// ============================================================================
// TypedMessagePassingLayer
// ============================================================================

TEST_CASE("TypedMessagePassingLayer forward shape", "[hetero_gnn]") {
    TypedMessagePassingLayer layer(32, 32, 3, 4, 0.0f);

    int n_nodes = 20;
    int n_edges = 40;
    auto node_features = torch::randn({n_nodes, 32});

    // Random edges: source and target node indices
    auto edge_index = torch::randint(0, n_nodes, {2, n_edges});
    auto edge_type = torch::randint(0, 3, {n_edges});

    auto out = layer->forward(node_features, edge_index, edge_type);

    REQUIRE(out.size(0) == n_nodes);
    REQUIRE(out.size(1) == 32);
}

TEST_CASE("TypedMessagePassing rejects out-of-range edge types", "[gnn]") {
    TypedMessagePassingLayer layer(/*in=*/4, /*out=*/4, /*n_edge_types=*/2,
                                   /*n_heads=*/1, /*dropout=*/0.0f);
    auto nodes = torch::randn({3, 4});
    auto edge_index = torch::tensor({{0, 1}, {1, 2}}, torch::kLong);  // (2, 2)

    SECTION("valid edge types forward") {
        auto edge_type = torch::tensor({0, 1}, torch::kLong);
        REQUIRE_NOTHROW(layer->forward(nodes, edge_index, edge_type));
    }
    SECTION("out-of-range edge type throws") {
        auto edge_type = torch::tensor({0, 5}, torch::kLong);  // 5 >= n_edge_types
        REQUIRE_THROWS(layer->forward(nodes, edge_index, edge_type));
    }
    SECTION("negative edge type throws") {
        auto edge_type = torch::tensor({0, -1}, torch::kLong);
        REQUIRE_THROWS(layer->forward(nodes, edge_index, edge_type));
    }
}

TEST_CASE("TypedMessagePassingLayer runs n_heads attention heads", "[hetero_gnn][attention]") {
    // Attention runs over n_heads disjoint slices of the message vector, so
    // HeterogeneousGNNConfig::n_heads -- a persisted, user-facing knob --
    // changes the function, not just the stored config.
    const int64_t n_nodes = 12;
    const int64_t features = 16;
    const int64_t n_edge_types = 2;

    torch::manual_seed(1234);
    auto node_features = torch::randn({n_nodes, features});

    // Complete directed graph so every target node mixes several messages and
    // the per-(target, head) softmax is actually exercised.
    std::vector<int64_t> src, tgt, types;
    for (int64_t i = 0; i < n_nodes; ++i) {
        for (int64_t j = 0; j < n_nodes; ++j) {
            if (i == j) continue;
            src.push_back(i);
            tgt.push_back(j);
            types.push_back((i + j) % n_edge_types);
        }
    }
    auto edge_index = torch::stack({
        torch::tensor(src, torch::kInt64),
        torch::tensor(tgt, torch::kInt64)});
    auto edge_type = torch::tensor(types, torch::kInt64);

    SECTION("head geometry is reported and the output shape is unchanged") {
        TypedMessagePassingLayer layer(features, features, n_edge_types, 4, 0.0f);
        layer->eval();
        REQUIRE(layer->n_heads() == 4);
        REQUIRE(layer->head_dim() == features / 4);

        auto out = layer->forward(node_features, edge_index, edge_type);
        REQUIRE(out.size(0) == n_nodes);
        REQUIRE(out.size(1) == features);
        REQUIRE(torch::isfinite(out).all().item<bool>());
    }

    SECTION("a different head count is a different function") {
        // Same seed before each construction, so the two layers hold identical
        // weights and any difference is the head count alone.
        torch::manual_seed(7);
        TypedMessagePassingLayer single(features, features, n_edge_types, 1, 0.0f);
        torch::manual_seed(7);
        TypedMessagePassingLayer multi(features, features, n_edge_types, 4, 0.0f);
        single->eval();
        multi->eval();

        auto a = single->forward(node_features, edge_index, edge_type);
        auto b = multi->forward(node_features, edge_index, edge_type);
        REQUIRE(a.sizes() == b.sizes());
        CHECK_FALSE(torch::allclose(a, b, /*rtol=*/1e-4, /*atol=*/1e-6));
    }

    SECTION("gradients flow through every head") {
        TypedMessagePassingLayer layer(features, features, n_edge_types, 4, 0.0f);
        auto out = layer->forward(node_features, edge_index, edge_type);
        out.sum().backward();
        bool attn_has_grad = false;
        for (auto& named : layer->named_parameters()) {
            if (named.key().rfind("attn_", 0) != 0) continue;
            if (named.value().grad().defined() &&
                named.value().grad().abs().sum().item<float>() > 0) {
                attn_has_grad = true;
            }
        }
        REQUIRE(attn_has_grad);
    }

    SECTION("an unusable head count is rejected at construction") {
        REQUIRE_THROWS_AS(
            TypedMessagePassingLayer(features, 15, n_edge_types, 4, 0.0f),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            TypedMessagePassingLayer(features, features, n_edge_types, 0, 0.0f),
            std::invalid_argument);
    }
}

TEST_CASE("TypedMessagePassing attention is finite under large logits",
          "[hetero_gnn][stability]") {
    // Large attention logits would overflow exp() to inf -> NaN weights; the
    // per-target-node max subtraction keeps the softmax finite.
    HeterogeneousGNNEncoder encoder(/*n_species=*/32, 64, 32, /*n_layers=*/2, 3, 4, 0.0f);
    encoder->eval();

    // Scale the learned species embeddings up so attention logits are large.
    {
        torch::NoGradGuard ng;
        for (auto& p : encoder->parameters()) {
            p.mul_(50.0f);
        }
    }

    // Dense-ish random graph over the 32 species.
    auto src = torch::randint(0, 32, {200}, torch::kInt64);
    auto dst = torch::randint(0, 32, {200}, torch::kInt64);
    auto edge_index = torch::stack({src, dst}, 0);
    auto edge_type = torch::randint(0, 3, {200}, torch::kInt64);

    auto species_emb = encoder->forward(edge_index, edge_type);
    REQUIRE(torch::isfinite(species_emb).all().item<bool>());
}

// ============================================================================
// HeterogeneousGNNEncoder
// ============================================================================

TEST_CASE("HeterogeneousGNNEncoder forward shape", "[hetero_gnn]") {
    int n_species = 50;
    HeterogeneousGNNEncoder encoder(n_species, 64, 32, 2, 3, 4, 0.0f);

    int n_edges = 100;
    auto edge_index = torch::randint(0, n_species, {2, n_edges});
    auto edge_type = torch::randint(0, 3, {n_edges});

    auto embeddings = encoder->forward(edge_index, edge_type);

    REQUIRE(embeddings.size(0) == n_species);
    REQUIRE(embeddings.size(1) == 32);  // output_dim
}

TEST_CASE("HeterogeneousGNNEncoder aggregate_for_plots", "[hetero_gnn]") {
    int n_species = 50;
    int batch_size = 8;
    int output_dim = 32;

    // Simulate species embeddings and abundance vectors
    auto species_emb = torch::randn({n_species, output_dim});
    auto species_vector = torch::randn({batch_size, n_species}).abs();  // Positive abundances

    auto plot_features = HeterogeneousGNNEncoderImpl::aggregate_for_plots(
        species_emb, species_vector);

    REQUIRE(plot_features.size(0) == batch_size);
    REQUIRE(plot_features.size(1) == output_dim);
}

TEST_CASE("HeterogeneousGNNEncoder accessors", "[hetero_gnn]") {
    HeterogeneousGNNEncoder encoder(100, 128, 64, 3, 3, 4, 0.1f);
    REQUIRE(encoder->output_dim() == 64);
    REQUIRE(encoder->n_species() == 100);
}

TEST_CASE("HeterogeneousGNNEncoder threads n_heads into its layers", "[hetero_gnn]") {
    // The config -> encoder -> layer path carries n_heads to every layer.
    HeterogeneousGNNEncoder encoder(/*n_species=*/16, /*hidden_dim=*/32,
                                    /*output_dim=*/16, /*n_layers=*/2,
                                    /*n_edge_types=*/3, /*n_heads=*/8, 0.0f);

    int n_layers_seen = 0;
    for (auto& module : encoder->modules(/*include_self=*/false)) {
        if (auto layer = std::dynamic_pointer_cast<TypedMessagePassingLayerImpl>(module)) {
            CHECK(layer->n_heads() == 8);
            CHECK(layer->head_dim() == 32 / 8);
            ++n_layers_seen;
        }
    }
    REQUIRE(n_layers_seen == 2);
}

TEST_CASE("HeterogeneousGNNEncoder gradient flow", "[hetero_gnn]") {
    int n_species = 20;
    HeterogeneousGNNEncoder encoder(n_species, 32, 16, 2, 3, 4, 0.0f);

    int n_edges = 30;
    auto edge_index = torch::randint(0, n_species, {2, n_edges});
    auto edge_type = torch::randint(0, 3, {n_edges});

    auto embeddings = encoder->forward(edge_index, edge_type);
    auto loss = embeddings.sum();
    loss.backward();

    // Check that learnable parameters got gradients
    bool has_grad = false;
    for (auto& param : encoder->parameters()) {
        if (param.grad().defined() && param.grad().abs().sum().item<float>() > 0) {
            has_grad = true;
            break;
        }
    }
    REQUIRE(has_grad);
}

TEST_CASE("HeterogeneousGNNEncoder does not double-register layer parameters",
          "[hetero_gnn][params]") {
    HeterogeneousGNNEncoder encoder(/*n_species=*/100, 128, 64, /*n_layers=*/3, 3, 4, 0.1f);

    std::set<void*> storages;
    for (const auto& named : encoder->named_parameters()) {
        void* ptr = named.value().data_ptr();
        INFO("duplicate parameter storage under: " << named.key());
        REQUIRE(storages.insert(ptr).second);
    }
}
