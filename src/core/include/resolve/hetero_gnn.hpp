#pragma once

// The heterogeneous species-graph encoder: species embeddings learned by typed
// message passing over a graph whose edges carry a type (co-occurrence,
// same-genus, same-family; species_graph.hpp builds it), aggregated per plot by
// the plot's species vector.

#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace resolve {

// A typed edge list sorted by type, with the span of each type. Edges of type
// t are columns [type_offsets[t], type_offsets[t + 1]) of edge_index and
// edge_type.
struct TypedEdges {
    torch::Tensor edge_index;            // (2, n_edges) int64, rows [source; target]
    torch::Tensor edge_type;             // (n_edges,) int64, non-decreasing
    std::vector<int64_t> type_offsets;   // n_edge_types + 1 entries
};

// Validates an edge list against a graph of n_nodes nodes and n_edge_types
// types and sorts it by type (stable, so edges of one type keep their order).
// Throws std::invalid_argument for a malformed shape, a node index outside
// [0, n_nodes) or an edge type outside [0, n_edge_types).
TypedEdges sort_typed_edges(const torch::Tensor& edge_index, const torch::Tensor& edge_type,
                            int64_t n_nodes, int64_t n_edge_types);

// One round of typed message passing. Each edge type has its own message
// function (an MLP over the source and target features). Messages are
// aggregated by attention-weighted scatter to their target nodes; attention
// runs with n_heads heads over disjoint out_features / n_heads slices of the
// message vector, concatenated back to out_features (the GAT convention), so
// out_features must be divisible by n_heads.
class TypedMessagePassingLayerImpl : public torch::nn::Module {
public:
    TypedMessagePassingLayerImpl(
        int64_t in_features,
        int64_t out_features,
        int64_t n_edge_types,
        int64_t n_heads = 4,
        float dropout = 0.1f
    );

    [[nodiscard]] int64_t n_heads() const noexcept { return n_heads_; }
    [[nodiscard]] int64_t head_dim() const noexcept { return head_dim_; }

    // node_features: (n_nodes, in_features); edges from sort_typed_edges over
    // n_edge_types types.
    // Returns: (n_nodes, out_features)
    torch::Tensor forward(torch::Tensor node_features, const TypedEdges& edges);

private:
    int64_t n_edge_types_;
    int64_t in_features_;
    int64_t out_features_;
    int64_t n_heads_;
    int64_t head_dim_;

    // Per-edge-type message MLPs: (src_feat || tgt_feat) -> message
    torch::nn::ModuleList message_fns_{nullptr};

    // Attention for weighting incoming messages
    torch::nn::Linear attn_query_{nullptr};
    torch::nn::Linear attn_key_{nullptr};

    // Output with residual + norm
    torch::nn::Linear output_{nullptr};
    torch::nn::LayerNorm norm_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(TypedMessagePassingLayer);

// Learns one embedding per species by message passing over the species graph
// set with set_graph(), which is part of the trained model: it is held in
// buffers, so it follows the module's device and is saved with the weights.
class HeterogeneousGNNEncoderImpl : public torch::nn::Module {
public:
    HeterogeneousGNNEncoderImpl(
        int64_t n_species,            // Number of species nodes
        int64_t hidden_dim = 128,
        int64_t output_dim = 64,
        int64_t n_layers = 3,
        int64_t n_edge_types = 3,
        int64_t n_heads = 4,
        float dropout = 0.1f
    );

    // Validate, sort by type and store the species graph.
    // edge_index: (2, n_edges) [source; target] species codes
    // edge_type: (n_edges,) type of each edge
    void set_graph(const torch::Tensor& edge_index, const torch::Tensor& edge_type);

    [[nodiscard]] bool has_graph() const { return graph_edge_type_.size(0) > 0; }

    // The stored graph, sorted by edge type; (2, 0) and (0,) until set.
    [[nodiscard]] const torch::Tensor& graph_edge_index() const noexcept {
        return graph_edge_index_;
    }
    [[nodiscard]] const torch::Tensor& graph_edge_type() const noexcept {
        return graph_edge_type_;
    }

    // (n_species, output_dim) species embeddings from message passing on the
    // stored graph. In eval mode with gradients off the result is cached and
    // reused until a parameter, buffer or the graph changes.
    torch::Tensor forward();

    // Aggregate species embeddings into per-plot features
    // species_embeddings: (n_species, output_dim) from forward()
    // species_vector: (batch, n_species) sparse rows (sparse_rows.hpp), the
    //                 value each species carries
    // Returns: (batch, output_dim) plot-level features, the value-weighted
    // mean of the plot's species embeddings
    [[nodiscard]] static torch::Tensor aggregate_for_plots(
        torch::Tensor species_embeddings,
        torch::Tensor species_vector
    );

    [[nodiscard]] int64_t output_dim() const noexcept { return output_dim_; }
    [[nodiscard]] int64_t n_species() const noexcept { return n_species_; }

private:
    // The stored graph with its type spans; the spans are read back to the
    // host once per graph, not once per forward.
    const TypedEdges& typed_edges();

    // The autocast state and the identity and version of every parameter and
    // buffer, which a cached embedding table must match to be reused.
    [[nodiscard]] std::vector<std::pair<const void*, int64_t>> state_signature() const;

    int64_t n_species_;
    int64_t output_dim_;
    int64_t n_edge_types_;

    // Learnable species embeddings (initial node features)
    torch::Tensor species_embeddings_;  // (n_species, hidden_dim)

    torch::nn::Linear input_proj_{nullptr};
    torch::nn::ModuleList layers_{nullptr};
    torch::nn::Linear output_proj_{nullptr};
    torch::nn::LayerNorm final_norm_{nullptr};

    torch::Tensor graph_edge_index_;  // buffer (2, n_edges)
    torch::Tensor graph_edge_type_;   // buffer (n_edges,)

    TypedEdges typed_edges_;
    const void* typed_edges_source_ = nullptr;

    torch::Tensor cached_embeddings_;
    std::vector<std::pair<const void*, int64_t>> cached_signature_;
};

TORCH_MODULE(HeterogeneousGNNEncoder);

}  // namespace resolve
