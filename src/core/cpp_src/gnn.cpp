#include "resolve/gnn.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace resolve {

namespace {

// Floor for a weighted edge: a kept neighbour is an edge, so it never carries
// exactly zero weight.
constexpr float kMinEdgeWeight = 1e-6f;

// Distance entries computed at once while searching neighbours: the block of
// rows is sized so a (block, n_nodes) slab holds about this many.
constexpr int64_t kKnnBlockEntries = int64_t{1} << 24;

}  // namespace

torch::Tensor scatter_to_targets(const torch::Tensor& values, const torch::Tensor& target,
                                 int64_t n_nodes) {
    auto shape = values.sizes().vec();
    shape[0] = n_nodes;
    return torch::zeros(shape, values.options()).index_add_(0, target, values);
}

// =============================================================================
// k-NN graph
// =============================================================================

KnnGraph build_knn_graph(torch::Tensor features, int64_t k,
                         GraphMetric metric, bool weighted) {
    const int64_t n_nodes = features.size(0);
    const auto index_options =
        torch::TensorOptions().dtype(torch::kInt64).device(features.device());
    const auto value_options = features.options().dtype(torch::kFloat32);
    auto nodes = torch::arange(n_nodes, index_options);

    // Cap k at the number of *other* nodes: a batch smaller than k (a short
    // final batch, or a single plot) has fewer neighbours to offer.
    const int64_t k_eff = std::min<int64_t>(k, std::max<int64_t>(n_nodes - 1, 0));

    torch::Tensor target = nodes;
    torch::Tensor source = nodes;
    torch::Tensor edge_value = torch::ones({n_nodes}, value_options);

    if (k_eff > 0) {
        // Cosine distance is 1 - cos(u, v): 0 for identical composition and 1
        // for disjoint composition, the natural reading for a non-negative
        // abundance or presence vector.
        auto points = features.to(torch::kFloat32);
        if (metric == GraphMetric::Cosine) {
            points = torch::nn::functional::normalize(
                points, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        }

        const int64_t block = std::max<int64_t>(1, kKnnBlockEntries / n_nodes);
        std::vector<torch::Tensor> kept_parts;
        std::vector<torch::Tensor> neighbour_parts;
        for (int64_t start = 0; start < n_nodes; start += block) {
            const int64_t rows = std::min(block, n_nodes - start);
            auto queries = points.narrow(0, start, rows);
            torch::Tensor dist = metric == GraphMetric::Cosine
                ? 1.0f - torch::matmul(queries, points.transpose(0, 1))
                : torch::cdist(queries, points, /*p=*/2.0, /*compute_mode=*/2);
            // A node is not its own neighbour.
            dist.scatter_(1, nodes.narrow(0, start, rows).unsqueeze(1),
                          std::numeric_limits<float>::infinity());
            auto [kept, neighbours] = dist.topk(k_eff, /*dim=*/1, /*largest=*/false);
            kept_parts.push_back(kept);
            neighbour_parts.push_back(neighbours);
        }
        auto kept = torch::cat(kept_parts, 0);              // (n_nodes, k_eff)
        auto neighbours = torch::cat(neighbour_parts, 0).flatten();

        // What a kept edge is worth. Unweighted, every edge counts 1. Weighted,
        // an edge carries its similarity: the cosine similarity under Cosine,
        // and under Euclidean a Gaussian kernel exp(-(d/sigma)^2) whose
        // bandwidth is the mean kept neighbour distance, so the weighting is
        // scale-free in the units the coordinates happen to be in.
        torch::Tensor values;
        if (!weighted) {
            values = torch::ones_like(kept);
        } else if (metric == GraphMetric::Cosine) {
            values = (1.0f - kept).clamp_min(kMinEdgeWeight);
        } else {
            auto sigma = kept.mean().clamp_min(kMinEdgeWeight);
            values = torch::exp(-(kept / sigma).pow(2)).clamp_min(kMinEdgeWeight);
        }
        values = values.flatten();
        auto rows = nodes.unsqueeze(1).expand({n_nodes, k_eff}).flatten();

        // Both directions of every kept edge, and a self-loop per node.
        target = torch::cat({rows, neighbours, nodes});
        source = torch::cat({neighbours, rows, nodes});
        edge_value = torch::cat({values, values, torch::ones({n_nodes}, value_options)});

        // One edge per pair, the larger value winning, so an edge kept from
        // both ends is counted once.
        auto [pairs, slot] = torch::_unique(target * n_nodes + source,
                                            /*sorted=*/true, /*return_inverse=*/true);
        edge_value = torch::zeros({pairs.size(0)}, value_options)
                         .scatter_reduce_(0, slot, edge_value, "amax",
                                          /*include_self=*/false);
        target = pairs.div(n_nodes, "floor");
        source = pairs.remainder(n_nodes);
    }

    // D^-1/2 (A + I) D^-1/2 with D the row sums.
    auto degree = scatter_to_targets(edge_value, target, n_nodes);
    auto d_inv_sqrt = degree.clamp_min(1.0f).pow(-0.5f);
    auto weight = d_inv_sqrt.index_select(0, target) * edge_value *
                  d_inv_sqrt.index_select(0, source);
    return {target, source, weight, n_nodes};
}

torch::Tensor segment_softmax(const torch::Tensor& scores, const torch::Tensor& target,
                              int64_t n_nodes) {
    auto target_cols = target.unsqueeze(1).expand_as(scores);
    auto max_per_target = torch::full({n_nodes, scores.size(1)},
        -std::numeric_limits<float>::infinity(), scores.options());
    max_per_target.scatter_reduce_(0, target_cols, scores, "amax", /*include_self=*/true);
    auto exp_scores = (scores - max_per_target.index_select(0, target)).exp();
    auto sum_per_target = scatter_to_targets(exp_scores, target, n_nodes);
    return exp_scores / (sum_per_target.index_select(0, target) + kEpsilon);
}

// =============================================================================
// GCN Layer
// =============================================================================

GCNLayerImpl::GCNLayerImpl(int64_t in_features, int64_t out_features, bool bias) {
    linear_ = register_module("linear",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias)));
}

torch::Tensor GCNLayerImpl::forward(torch::Tensor x, const KnnGraph& graph) {
    auto transformed = linear_->forward(x);
    auto messages = transformed.index_select(0, graph.source) *
                    graph.weight.to(transformed.scalar_type()).unsqueeze(1);
    return scatter_to_targets(messages, graph.target, graph.n_nodes);
}

// =============================================================================
// GAT Layer
// =============================================================================

GATLayerImpl::GATLayerImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t n_heads,
    float dropout,
    bool concat
) : out_features_(out_features), n_heads_(n_heads), concat_(concat) {
    W_ = register_module("W",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features * n_heads).bias(false)));

    a_src_ = register_parameter("a_src", torch::randn({n_heads, out_features}) * kGATInitStd);
    a_dst_ = register_parameter("a_dst", torch::randn({n_heads, out_features}) * kGATInitStd);

    if (dropout > 0) {
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }
    leaky_relu_ = register_module("leaky_relu", torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(0.2)));
}

torch::Tensor GATLayerImpl::forward(torch::Tensor x, const KnnGraph& graph) {
    const int64_t n_nodes = x.size(0);
    auto Wh = W_->forward(x).reshape({n_nodes, n_heads_, out_features_});

    // Score of edge j -> i: LeakyReLU(a_src . Wh_i + a_dst . Wh_j), the
    // aggregating node's term first.
    auto score_target = torch::einsum("nho,ho->nh", {Wh, a_src_});
    auto score_source = torch::einsum("nho,ho->nh", {Wh, a_dst_});
    auto scores = leaky_relu_->forward(score_target.index_select(0, graph.target) +
                                       score_source.index_select(0, graph.source));

    auto attn = segment_softmax(scores, graph.target, n_nodes);  // (n_edges, n_heads)
    if (dropout_) {
        attn = dropout_->forward(attn);
    }

    auto output = scatter_to_targets(
        attn.unsqueeze(-1) * Wh.index_select(0, graph.source), graph.target, n_nodes);
    return concat_ ? output.reshape({n_nodes, n_heads_ * out_features_})
                   : output.mean(/*dim=*/1);
}

// =============================================================================
// GraphSAGE Layer
// =============================================================================

GraphSAGELayerImpl::GraphSAGELayerImpl(int64_t in_features, int64_t out_features, bool bias) {
    linear_self_ = register_module("linear_self",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias)));
    linear_neighbor_ = register_module("linear_neighbor",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(false)));
}

torch::Tensor GraphSAGELayerImpl::forward(torch::Tensor x, const KnnGraph& graph) {
    // The adjacency rows rescaled by their sums (at least 1), then applied.
    auto weight = graph.weight.to(x.scalar_type());
    auto row_sum = scatter_to_targets(weight, graph.target, graph.n_nodes).clamp_min(1.0f);
    auto edge_weight = weight / row_sum.index_select(0, graph.target);
    auto neighbor_agg = scatter_to_targets(
        x.index_select(0, graph.source) * edge_weight.unsqueeze(1), graph.target,
        graph.n_nodes);
    return linear_self_->forward(x) + linear_neighbor_->forward(neighbor_agg);
}

// =============================================================================
// GNN Encoder
// =============================================================================

GNNEncoderImpl::GNNEncoderImpl(
    int64_t in_features,
    int64_t hidden_dim,
    int64_t out_features,
    int64_t n_layers,
    GNNType gnn_type,
    int64_t n_heads,
    float dropout
) : out_features_(out_features), gnn_type_(gnn_type) {
    layers_ = register_module("layers", torch::nn::ModuleList());

    int64_t current_dim = in_features;
    for (int64_t i = 0; i < n_layers; ++i) {
        int64_t next_dim = (i == n_layers - 1) ? out_features : hidden_dim;
        switch (gnn_type) {
            case GNNType::GCN:
                layers_->push_back(GCNLayer(current_dim, next_dim));
                break;
            case GNNType::GAT: {
                const bool concat = (i < n_layers - 1);
                layers_->push_back(GATLayer(current_dim, next_dim, n_heads, dropout, concat));
                if (concat) {
                    next_dim = next_dim * n_heads;
                }
                break;
            }
            case GNNType::GraphSAGE:
                layers_->push_back(GraphSAGELayer(current_dim, next_dim));
                break;
            default:
                throw std::invalid_argument(
                    "unknown GNNType value: " + std::to_string(static_cast<int>(gnn_type)));
        }
        current_dim = next_dim;
    }

    if (dropout > 0) {
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }
}

torch::Tensor GNNEncoderImpl::forward(torch::Tensor x, const KnnGraph& graph) {
    for (size_t i = 0; i < layers_->size(); ++i) {
        const auto& layer = layers_->ptr(static_cast<int64_t>(i));
        switch (gnn_type_) {
            case GNNType::GCN:
                x = layer->as<GCNLayerImpl>()->forward(x, graph);
                break;
            case GNNType::GAT:
                x = layer->as<GATLayerImpl>()->forward(x, graph);
                break;
            case GNNType::GraphSAGE:
                x = layer->as<GraphSAGELayerImpl>()->forward(x, graph);
                break;
        }
        if (i + 1 < layers_->size()) {
            x = torch::relu(x);
            if (dropout_) {
                x = dropout_->forward(x);
            }
        }
    }
    return x;
}

}  // namespace resolve
