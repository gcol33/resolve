// Graph encoders as standalone modules: the plot k-NN graph and the GNNEncoder
// over it, the typed message-passing layer, and the heterogeneous species-graph
// encoder built from it.

#include <catch2/catch_test_macros.hpp>

#include "resolve/gnn.hpp"
#include "resolve/hetero_gnn.hpp"
#include "resolve/sparse_rows.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

using namespace resolve;

namespace {

// The graph's adjacency entries as a dense (n_nodes, n_nodes) matrix.
torch::Tensor dense_adjacency(const KnnGraph& graph) {
    auto adj = torch::zeros({graph.n_nodes, graph.n_nodes});
    adj.index_put_({graph.target, graph.source}, graph.weight);
    return adj;
}

// The dense k-NN adjacency, computed the direct way over the full distance
// matrix: the oracle the edge-list graph has to reproduce.
torch::Tensor oracle_adjacency(const torch::Tensor& features, int64_t k, GraphMetric metric,
                               bool weighted) {
    const int64_t n = features.size(0);
    const int64_t k_eff = std::min<int64_t>(k, std::max<int64_t>(n - 1, 0));
    if (k_eff == 0) return torch::eye(n);
    torch::Tensor dist;
    if (metric == GraphMetric::Cosine) {
        auto unit = torch::nn::functional::normalize(
            features, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        dist = 1.0f - torch::matmul(unit, unit.transpose(0, 1));
    } else {
        dist = (features.unsqueeze(0) - features.unsqueeze(1)).pow(2).sum(-1).sqrt();
    }
    dist.fill_diagonal_(std::numeric_limits<float>::infinity());
    auto [kept, idx] = dist.topk(k_eff, 1, /*largest=*/false);
    torch::Tensor values;
    if (!weighted) {
        values = torch::ones_like(kept);
    } else if (metric == GraphMetric::Cosine) {
        values = (1.0f - kept).clamp_min(1e-6f);
    } else {
        auto sigma = kept.mean().clamp_min(1e-6f);
        values = torch::exp(-(kept / sigma).pow(2)).clamp_min(1e-6f);
    }
    auto adj = torch::zeros({n, n});
    auto rows = torch::arange(n).unsqueeze(1).expand_as(idx);
    adj.index_put_({rows, idx}, values);
    adj = torch::maximum(adj, adj.transpose(0, 1));
    adj = torch::maximum(adj, torch::eye(n));
    auto d = adj.sum(1).clamp_min(1.0f).pow(-0.5f);
    return d.unsqueeze(1) * adj * d.unsqueeze(0);
}

// The three layers written out on the dense adjacency.
torch::Tensor dense_gcn(const torch::Tensor& adj, const torch::Tensor& transformed) {
    return torch::matmul(adj, transformed);
}

torch::Tensor dense_sage(const torch::Tensor& adj, const torch::Tensor& x) {
    return torch::matmul(adj / adj.sum(1, true).clamp_min(1.0f), x);
}

torch::Tensor dense_gat_weights(const torch::Tensor& adj, const torch::Tensor& score_target,
                                const torch::Tensor& score_source) {
    auto scores = torch::leaky_relu(score_target.unsqueeze(1) + score_source.unsqueeze(0), 0.2);
    scores = scores.masked_fill((adj == 0).unsqueeze(-1), -1e9f);
    return torch::softmax(scores, 1);  // (n, n, heads)
}

torch::Tensor named_parameter(const torch::nn::Module& module, const std::string& name) {
    for (const auto& named : module.named_parameters()) {
        if (named.key() == name) return named.value();
    }
    throw std::runtime_error("no parameter " + name);
}

}  // namespace

// ============================================================================
// The k-NN graph
// ============================================================================

TEST_CASE("The k-NN edge list is the dense k-NN adjacency", "[gnn][adjacency]") {
    torch::manual_seed(21);
    const auto coords = torch::randn({40, 2});
    const auto composition = torch::rand({40, 12});
    for (bool weighted : {false, true}) {
        INFO("weighted " << weighted);
        CHECK(torch::allclose(dense_adjacency(build_knn_graph(coords, 5, GraphMetric::Euclidean, weighted)),
                              oracle_adjacency(coords, 5, GraphMetric::Euclidean, weighted),
                              1e-5, 1e-6));
        CHECK(torch::allclose(dense_adjacency(build_knn_graph(composition, 4, GraphMetric::Cosine, weighted)),
                              oracle_adjacency(composition, 4, GraphMetric::Cosine, weighted),
                              1e-5, 1e-6));
    }
}

TEST_CASE("The k-NN graph holds one symmetric edge per pair and a self-loop per node",
          "[gnn][adjacency]") {
    torch::manual_seed(22);
    SECTION("multi-node graph") {
        auto graph = build_knn_graph(torch::randn({30, 2}), 3);
        auto keys = graph.target * graph.n_nodes + graph.source;
        CHECK(std::get<0>(torch::_unique(keys)).numel() == keys.numel());
        auto adj = dense_adjacency(graph);
        CHECK(torch::allclose(adj, adj.transpose(0, 1)));
        CHECK((adj.diagonal() > 0).all().item<bool>());
    }
    SECTION("single-node graph is a self-loop") {
        auto graph = build_knn_graph(torch::randn({1, 2}), 3);
        REQUIRE(graph.target.numel() == 1);
        CHECK(graph.weight[0].item<float>() > 0.0f);
    }
}

TEST_CASE("A graph larger than one distance block matches the dense oracle",
          "[gnn][adjacency]") {
    // 4500 nodes need two blocks of rows at 2^24 distance entries per block.
    torch::manual_seed(23);
    const auto coords = torch::randn({4500, 2});
    auto graph = build_knn_graph(coords, 3, GraphMetric::Euclidean, true);
    CHECK(torch::allclose(dense_adjacency(graph),
                          oracle_adjacency(coords, 3, GraphMetric::Euclidean, true), 1e-5, 1e-6));
}

// ============================================================================
// GNNEncoder
// ============================================================================

TEST_CASE("GNNEncoder forward shape", "[gnn]") {
    GNNEncoder encoder(16, 32, 16, 2, GNNType::GCN, 4, 0.0f);
    auto x = torch::randn({10, 16});
    auto out = encoder->forward(x, build_knn_graph(torch::randn({10, 2}), 3));

    REQUIRE(out.size(0) == 10);
    REQUIRE(out.size(1) == 16);  // out_features
}

TEST_CASE("Message passing on the edge list is message passing on the dense adjacency",
          "[gnn]") {
    torch::manual_seed(24);
    const int64_t n = 25;
    auto x = torch::randn({n, 6});
    auto graph = build_knn_graph(torch::randn({n, 2}), 4, GraphMetric::Euclidean, true);
    auto adj = dense_adjacency(graph);

    SECTION("GCN") {
        GCNLayer layer(6, 5);
        auto transformed = torch::nn::functional::linear(
            x, named_parameter(*layer, "linear.weight"), named_parameter(*layer, "linear.bias"));
        CHECK(torch::allclose(layer->forward(x, graph), dense_gcn(adj, transformed), 1e-5, 1e-6));
    }
    SECTION("GraphSAGE") {
        GraphSAGELayer layer(6, 5);
        auto self = torch::nn::functional::linear(x, named_parameter(*layer, "linear_self.weight"),
                                                  named_parameter(*layer, "linear_self.bias"));
        auto neighbours = torch::nn::functional::linear(
            dense_sage(adj, x), named_parameter(*layer, "linear_neighbor.weight"));
        CHECK(torch::allclose(layer->forward(x, graph), self + neighbours, 1e-5, 1e-6));
    }
    SECTION("GAT, concatenated and averaged heads") {
        for (bool concat : {true, false}) {
            GATLayer layer(6, 4, /*n_heads=*/3, /*dropout=*/0.0f, concat);
            auto Wh = torch::nn::functional::linear(x, named_parameter(*layer, "W.weight"))
                          .reshape({n, 3, 4});
            auto weights = dense_gat_weights(
                adj, torch::einsum("nho,ho->nh", {Wh, named_parameter(*layer, "a_src")}),
                torch::einsum("nho,ho->nh", {Wh, named_parameter(*layer, "a_dst")}));
            auto expected = torch::einsum("nmh,mho->nho", {weights, Wh});
            expected = concat ? expected.reshape({n, 12}) : expected.mean(1);
            CHECK(torch::allclose(layer->forward(x, graph), expected, 1e-5, 1e-6));
        }
    }
}

// ============================================================================
// TypedMessagePassingLayer
// ============================================================================

TEST_CASE("Typed edges are validated and sorted by type", "[hetero_gnn]") {
    auto edge_index = torch::tensor({{0, 1, 2, 3}, {1, 2, 3, 0}}, torch::kInt64);
    auto edge_type = torch::tensor({2, 0, 1, 0}, torch::kInt64);
    auto edges = sort_typed_edges(edge_index, edge_type, /*n_nodes=*/4, /*n_edge_types=*/3);

    CHECK(torch::equal(edges.edge_type, torch::tensor({0, 0, 1, 2}, torch::kInt64)));
    CHECK(torch::equal(edges.edge_index,
                       torch::tensor({{1, 3, 2, 0}, {2, 0, 3, 1}}, torch::kInt64)));
    CHECK(edges.type_offsets == std::vector<int64_t>{0, 2, 3, 4});

    CHECK_THROWS_AS(sort_typed_edges(edge_index, torch::tensor({0, 5, 1, 0}, torch::kInt64), 4, 3),
                    std::invalid_argument);
    CHECK_THROWS_AS(sort_typed_edges(edge_index, torch::tensor({0, -1, 1, 0}, torch::kInt64), 4, 3),
                    std::invalid_argument);
    CHECK_THROWS_AS(sort_typed_edges(edge_index, edge_type, /*n_nodes=*/3, 3),
                    std::invalid_argument);
    CHECK_THROWS_AS(sort_typed_edges(edge_index.narrow(0, 0, 1), edge_type, 4, 3),
                    std::invalid_argument);
}

TEST_CASE("TypedMessagePassingLayer forward shape", "[hetero_gnn]") {
    TypedMessagePassingLayer layer(32, 32, 3, 4, 0.0f);
    const int64_t n_nodes = 20;
    auto edges = sort_typed_edges(torch::randint(0, n_nodes, {2, 40}),
                                  torch::randint(0, 3, {40}), n_nodes, 3);
    auto out = layer->forward(torch::randn({n_nodes, 32}), edges);

    REQUIRE(out.size(0) == n_nodes);
    REQUIRE(out.size(1) == 32);
}

TEST_CASE("Typed message passing does not depend on the order edges arrive in",
          "[hetero_gnn]") {
    torch::manual_seed(25);
    TypedMessagePassingLayer layer(8, 8, 3, 2, 0.0f);
    layer->eval();
    auto nodes = torch::randn({10, 8});
    auto edge_index = torch::randint(0, 10, {2, 30});
    auto edge_type = torch::randint(0, 3, {30});
    auto order = torch::randperm(30, torch::kInt64);

    auto a = layer->forward(nodes, sort_typed_edges(edge_index, edge_type, 10, 3));
    auto b = layer->forward(nodes, sort_typed_edges(edge_index.index_select(1, order),
                                                    edge_type.index_select(0, order), 10, 3));
    CHECK(torch::allclose(a, b, 1e-5, 1e-6));
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
    auto edges = sort_typed_edges(
        torch::stack({torch::tensor(src, torch::kInt64), torch::tensor(tgt, torch::kInt64)}),
        torch::tensor(types, torch::kInt64), n_nodes, n_edge_types);

    SECTION("head geometry is reported and the output shape is unchanged") {
        TypedMessagePassingLayer layer(features, features, n_edge_types, 4, 0.0f);
        layer->eval();
        REQUIRE(layer->n_heads() == 4);
        REQUIRE(layer->head_dim() == features / 4);

        auto out = layer->forward(node_features, edges);
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

        auto a = single->forward(node_features, edges);
        auto b = multi->forward(node_features, edges);
        REQUIRE(a.sizes() == b.sizes());
        CHECK_FALSE(torch::allclose(a, b, /*rtol=*/1e-4, /*atol=*/1e-6));
    }

    SECTION("gradients flow through every head") {
        TypedMessagePassingLayer layer(features, features, n_edge_types, 4, 0.0f);
        auto out = layer->forward(node_features, edges);
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

// ============================================================================
// HeterogeneousGNNEncoder
// ============================================================================

namespace {

HeterogeneousGNNEncoder encoder_with_graph(int64_t n_species, int64_t n_edges,
                                           int64_t hidden = 64, int64_t output = 32) {
    HeterogeneousGNNEncoder encoder(n_species, hidden, output, 2, 3, 4, 0.0f);
    encoder->set_graph(torch::randint(0, n_species, {2, n_edges}),
                       torch::randint(0, 3, {n_edges}));
    return encoder;
}

}  // namespace

TEST_CASE("HeterogeneousGNNEncoder forward shape", "[hetero_gnn]") {
    auto encoder = encoder_with_graph(50, 100);
    auto embeddings = encoder->forward();

    REQUIRE(embeddings.size(0) == 50);
    REQUIRE(embeddings.size(1) == 32);  // output_dim
}

TEST_CASE("HeterogeneousGNNEncoder needs a graph and validates it once", "[hetero_gnn]") {
    HeterogeneousGNNEncoder encoder(8, 16, 8, 1, 3, 2, 0.0f);
    REQUIRE_FALSE(encoder->has_graph());
    CHECK_THROWS(encoder->forward());
    CHECK_THROWS_AS(encoder->set_graph(torch::tensor({{0, 9}, {1, 2}}, torch::kInt64),
                                       torch::tensor({0, 1}, torch::kInt64)),
                    std::invalid_argument);
    CHECK_THROWS_AS(encoder->set_graph(torch::tensor({{0, 1}, {1, 2}}, torch::kInt64),
                                       torch::tensor({0, 3}, torch::kInt64)),
                    std::invalid_argument);
    REQUIRE_FALSE(encoder->has_graph());

    encoder->set_graph(torch::tensor({{0, 1, 2}, {1, 2, 3}}, torch::kInt64),
                       torch::tensor({2, 0, 1}, torch::kInt64));
    REQUIRE(encoder->has_graph());
    CHECK(torch::equal(encoder->graph_edge_type(), torch::tensor({0, 1, 2}, torch::kInt64)));
}

TEST_CASE("The species graph is a buffer of the encoder", "[hetero_gnn]") {
    auto encoder = encoder_with_graph(10, 20, 16, 8);
    std::set<std::string> buffers;
    for (const auto& named : encoder->named_buffers()) buffers.insert(named.key());
    CHECK(buffers.count("graph_edge_index") == 1);
    CHECK(buffers.count("graph_edge_type") == 1);

    SECTION("it follows the module to a CUDA device") {
        if (!torch::cuda::is_available()) SKIP("needs a CUDA device");
        encoder->to(torch::kCUDA);
        CHECK(encoder->graph_edge_index().is_cuda());
        CHECK(encoder->forward().is_cuda());
    }
}

TEST_CASE("Eval-mode species embeddings are cached until the weights change", "[hetero_gnn]") {
    auto encoder = encoder_with_graph(12, 30, 16, 8);
    encoder->eval();
    torch::NoGradGuard no_grad;

    auto first = encoder->forward();
    auto second = encoder->forward();
    CHECK(first.is_same(second));

    for (auto& p : encoder->parameters()) {
        p.add_(0.1f);
        break;
    }
    auto after_update = encoder->forward();
    CHECK_FALSE(after_update.is_same(first));
    CHECK_FALSE(torch::allclose(after_update, first));

    encoder->set_graph(torch::randint(0, 12, {2, 30}), torch::randint(0, 3, {30}));
    CHECK_FALSE(encoder->forward().is_same(after_update));

    SECTION("training mode never caches") {
        encoder->train();
        CHECK_FALSE(encoder->forward().is_same(encoder->forward()));
    }
}

TEST_CASE("TypedMessagePassing attention is finite under large logits",
          "[hetero_gnn][stability]") {
    // Large attention logits would overflow exp() to inf -> NaN weights; the
    // per-target-node max subtraction keeps the softmax finite.
    auto encoder = encoder_with_graph(32, 200);
    encoder->eval();
    {
        torch::NoGradGuard ng;
        for (auto& p : encoder->parameters()) {
            p.mul_(50.0f);
        }
    }
    REQUIRE(torch::isfinite(encoder->forward()).all().item<bool>());
}

TEST_CASE("HeterogeneousGNNEncoder aggregate_for_plots", "[hetero_gnn]") {
    auto species_emb = torch::randn({50, 32});
    auto species_vector = torch::randn({8, 50}).abs();  // Positive abundances

    auto plot_features = HeterogeneousGNNEncoderImpl::aggregate_for_plots(
        species_emb, to_sparse_rows(species_vector));

    REQUIRE(plot_features.size(0) == 8);
    REQUIRE(plot_features.size(1) == 32);
    // The value-weighted mean of each plot's species embeddings.
    const auto weights = species_vector / (species_vector.sum(1, true) + kEpsilon);
    REQUIRE(torch::allclose(plot_features, torch::matmul(weights, species_emb), 1e-5, 1e-6));
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
    auto encoder = encoder_with_graph(20, 30, 32, 16);
    encoder->forward().sum().backward();

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
