#pragma once

// The plot-level graph encoder: a k-nearest-neighbour graph over the plots of a
// forward, held as an edge list, and GCN / GAT / GraphSAGE message passing over
// it. Memory is O(n_nodes * k), never O(n_nodes^2).

#include "resolve/types.hpp"  // GNNType

#include <torch/torch.h>

#include <cstdint>

namespace resolve {

// How a k-NN graph measures the distance between two nodes.
enum class GraphMetric {
    Euclidean,  // Straight-line distance; the metric for coordinates
    Cosine      // 1 - cosine similarity; the metric for composition vectors
};

// A graph as directed weighted edges. Edge e carries a message from
// source[e] to target[e]; the edge set is symmetric, holds a self-loop for
// every node and no duplicate pair. weight[e] is the entry of the symmetric
// degree-normalized adjacency D^-1/2 (A + I) D^-1/2 for the pair.
struct KnnGraph {
    torch::Tensor target;  // (n_edges,) int64
    torch::Tensor source;  // (n_edges,) int64
    torch::Tensor weight;  // (n_edges,) float
    int64_t n_nodes = 0;
};

// The k-NN graph over node features.
// features: (n_nodes, n_features) - coordinates, or a composition vector
// k: neighbours kept per node, capped at n_nodes - 1. Node i keeps an edge to
//   each of its k nearest other nodes, and an edge kept from either end is an
//   edge.
// metric: what "near" means for these features
// weighted: an edge's adjacency entry is its similarity (a Gaussian kernel of
//   the distance under Euclidean, whose bandwidth is the mean kept neighbour
//   distance; the cosine similarity itself under Cosine) instead of a plain 1
//   -- GNNConfig::use_edge_features. Where both ends keep the edge the larger
//   entry wins; a self-loop is 1.
// The distances are computed a block of rows at a time, so the working memory
// is a (block, n_nodes) slab rather than (n_nodes, n_nodes).
KnnGraph build_knn_graph(torch::Tensor features, int64_t k,
                         GraphMetric metric = GraphMetric::Euclidean,
                         bool weighted = false);

// Softmax of per-edge scores over the edges that share a target node, per
// column: scores (n_edges, n_columns), target (n_edges,) in [0, n_nodes).
// Each target's maximum is subtracted before exp(), so large scores stay
// finite.
torch::Tensor segment_softmax(const torch::Tensor& scores, const torch::Tensor& target,
                              int64_t n_nodes);

// Graph Convolutional Layer (GCN): out_i = sum_j A_ij (x W)_j over the
// normalized adjacency.
class GCNLayerImpl : public torch::nn::Module {
public:
    GCNLayerImpl(int64_t in_features, int64_t out_features, bool bias = true);

    // x: (n_nodes, in_features) -> (n_nodes, out_features)
    torch::Tensor forward(torch::Tensor x, const KnnGraph& graph);

private:
    torch::nn::Linear linear_{nullptr};
};

TORCH_MODULE(GCNLayer);

// Graph Attention Layer (GAT): attention over each node's neighbours (the
// adjacency decides who is a neighbour, not how much they count).
class GATLayerImpl : public torch::nn::Module {
public:
    GATLayerImpl(
        int64_t in_features,
        int64_t out_features,
        int64_t n_heads = 1,
        float dropout = 0.0f,
        bool concat = true
    );

    // x: (n_nodes, in_features)
    // Returns: (n_nodes, out_features * n_heads) if concat, else (n_nodes, out_features)
    torch::Tensor forward(torch::Tensor x, const KnnGraph& graph);

private:
    int64_t out_features_;
    int64_t n_heads_;
    bool concat_;

    torch::nn::Linear W_{nullptr};
    torch::Tensor a_src_;  // Attention parameter for source
    torch::Tensor a_dst_;  // Attention parameter for destination
    torch::nn::Dropout dropout_{nullptr};
    torch::nn::LeakyReLU leaky_relu_{nullptr};
};

TORCH_MODULE(GATLayer);

// GraphSAGE Layer: a node's own projection plus the projection of the
// adjacency-weighted mean of its neighbours.
class GraphSAGELayerImpl : public torch::nn::Module {
public:
    GraphSAGELayerImpl(int64_t in_features, int64_t out_features, bool bias = true);

    // x: (n_nodes, in_features) -> (n_nodes, out_features)
    torch::Tensor forward(torch::Tensor x, const KnnGraph& graph);

private:
    torch::nn::Linear linear_self_{nullptr};
    torch::nn::Linear linear_neighbor_{nullptr};
};

TORCH_MODULE(GraphSAGELayer);

// n_layers message-passing layers of one GNNType, ReLU and dropout between
// them.
class GNNEncoderImpl : public torch::nn::Module {
public:
    GNNEncoderImpl(
        int64_t in_features,
        int64_t hidden_dim = 64,
        int64_t out_features = 32,
        int64_t n_layers = 2,
        GNNType gnn_type = GNNType::GCN,
        int64_t n_heads = 4,  // For GAT
        float dropout = 0.1f
    );

    // x: (n_nodes, in_features) -> (n_nodes, out_features)
    torch::Tensor forward(torch::Tensor x, const KnnGraph& graph);

    [[nodiscard]] int64_t output_dim() const noexcept { return out_features_; }

private:
    int64_t out_features_;
    GNNType gnn_type_;

    torch::nn::ModuleList layers_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(GNNEncoder);

}  // namespace resolve
