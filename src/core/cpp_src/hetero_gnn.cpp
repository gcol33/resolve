#include "resolve/hetero_gnn.hpp"
#include "resolve/gnn.hpp"    // segment_softmax
#include "resolve/types.hpp"  // kBertInitStd, kEpsilon
#include "resolve/sparse_rows.hpp"

#include <ATen/autocast_mode.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace resolve {

// =============================================================================
// Typed edges
// =============================================================================

namespace {

// [min, max] of a non-empty int64 tensor, read back in one transfer.
std::pair<int64_t, int64_t> value_range(const torch::Tensor& values) {
    auto range = torch::stack({values.min(), values.max()}).to(torch::kCPU);
    return {range[0].item<int64_t>(), range[1].item<int64_t>()};
}

// Column boundaries of each type in a type-sorted edge list.
std::vector<int64_t> type_offsets_of(const torch::Tensor& sorted_type, int64_t n_edge_types) {
    std::vector<int64_t> offsets(static_cast<size_t>(n_edge_types) + 1, 0);
    if (sorted_type.numel() == 0) return offsets;
    auto counts = torch::bincount(sorted_type, /*weights=*/{}, n_edge_types)
                      .cumsum(0).to(torch::kCPU).to(torch::kInt64);
    const auto cumulative = counts.accessor<int64_t, 1>();
    for (int64_t t = 0; t < n_edge_types; ++t) {
        offsets[static_cast<size_t>(t) + 1] = cumulative[t];
    }
    return offsets;
}

}  // namespace

TypedEdges sort_typed_edges(const torch::Tensor& edge_index, const torch::Tensor& edge_type,
                            int64_t n_nodes, int64_t n_edge_types) {
    if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
        throw std::invalid_argument("typed message passing: edge_index must be "
                                    "(2, n_edges), got " + std::to_string(edge_index.dim()) +
                                    "-D");
    }
    if (edge_type.dim() != 1 || edge_type.size(0) != edge_index.size(1)) {
        throw std::invalid_argument("typed message passing: edge_type must be (n_edges,) "
                                    "matching edge_index");
    }
    auto index = edge_index.to(torch::kInt64);
    auto type = edge_type.to(torch::kInt64);
    if (type.numel() > 0) {
        // An edge type outside [0, n_edge_types) has no message function, and a
        // node outside [0, n_nodes) has no features.
        const auto [type_min, type_max] = value_range(type);
        if (type_min < 0 || type_max >= n_edge_types) {
            throw std::invalid_argument(
                "typed message passing: edge_type values must be in [0, " +
                std::to_string(n_edge_types) + "), but got range [" +
                std::to_string(type_min) + ", " + std::to_string(type_max) + "].");
        }
        const auto [node_min, node_max] = value_range(index);
        if (node_min < 0 || node_max >= n_nodes) {
            throw std::invalid_argument(
                "typed message passing: edge_index values must be node indices in [0, " +
                std::to_string(n_nodes) + "), but got range [" + std::to_string(node_min) +
                ", " + std::to_string(node_max) + "].");
        }
    }
    auto [sorted_type, order] = type.sort(/*stable=*/true, /*dim=*/0, /*descending=*/false);
    return {index.index_select(1, order), sorted_type,
            type_offsets_of(sorted_type, n_edge_types)};
}

// =============================================================================
// Typed Message Passing Layer
// =============================================================================

TypedMessagePassingLayerImpl::TypedMessagePassingLayerImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t n_edge_types,
    int64_t n_heads,
    float dropout
) : n_edge_types_(n_edge_types),
    in_features_(in_features),
    out_features_(out_features),
    n_heads_(n_heads),
    head_dim_(0)
{
    if (n_heads < 1) {
        throw std::invalid_argument(
            "typed message passing: n_heads must be >= 1, got " +
            std::to_string(n_heads) + ".");
    }
    if (out_features % n_heads != 0) {
        throw std::invalid_argument(
            "typed message passing: out_features (" + std::to_string(out_features) +
            ") must be divisible by n_heads (" + std::to_string(n_heads) + ").");
    }
    head_dim_ = out_features / n_heads;

    torch::nn::ModuleList msg_fns;
    for (int64_t t = 0; t < n_edge_types; ++t) {
        auto seq = torch::nn::Sequential();
        seq->push_back(torch::nn::Linear(2 * in_features, out_features));
        seq->push_back(torch::nn::GELU());
        seq->push_back(torch::nn::Linear(out_features, out_features));
        msg_fns->push_back(seq);
    }
    message_fns_ = register_module("message_fns", msg_fns);

    attn_query_ = register_module("attn_q",
        torch::nn::Linear(in_features, out_features));
    attn_key_ = register_module("attn_k",
        torch::nn::Linear(out_features, out_features));

    output_ = register_module("output",
        torch::nn::Linear(out_features, out_features));
    norm_ = register_module("norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({out_features})));
    dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
}

torch::Tensor TypedMessagePassingLayerImpl::forward(
    torch::Tensor node_features,
    const TypedEdges& edges
) {
    TORCH_CHECK(static_cast<int64_t>(edges.type_offsets.size()) == n_edge_types_ + 1,
        "typed message passing: edges sorted for ", edges.type_offsets.size() - 1,
        " edge types, layer built for ", n_edge_types_);
    const int64_t n_nodes = node_features.size(0);
    const int64_t n_edges = edges.edge_index.size(1);

    if (n_edges == 0) {
        if (in_features_ == out_features_) {
            return norm_->forward(node_features);
        }
        return norm_->forward(output_->forward(
            torch::zeros({n_nodes, out_features_}, node_features.options())));
    }

    auto src_idx = edges.edge_index[0];
    auto tgt_idx = edges.edge_index[1];
    auto edge_input = torch::cat({node_features.index_select(0, src_idx),
                                  node_features.index_select(0, tgt_idx)}, /*dim=*/1);

    // The edges are sorted by type, so each type's message function reads one
    // contiguous span and the spans concatenate back in edge order.
    std::vector<torch::Tensor> type_messages;
    type_messages.reserve(static_cast<size_t>(n_edge_types_));
    for (int64_t t = 0; t < n_edge_types_; ++t) {
        const int64_t begin = edges.type_offsets[static_cast<size_t>(t)];
        const int64_t count = edges.type_offsets[static_cast<size_t>(t) + 1] - begin;
        if (count == 0) continue;
        type_messages.push_back(message_fns_[t]->as<torch::nn::Sequential>()->forward(
            edge_input.narrow(0, begin, count)));
    }
    auto messages = torch::cat(type_messages, /*dim=*/0);  // (n_edges, out_features)

    auto query = attn_query_->forward(node_features);   // (n_nodes, out_features)
    auto key = attn_key_->forward(messages);            // (n_edges, out_features)

    // Multi-head attention: each head scores over its own out_features /
    // n_heads slice, and the per-head aggregates are concatenated back to
    // out_features (the GAT concatenation convention). n_heads == 1 reduces to
    // a single dot product over the whole vector.
    auto tgt_query = query.index_select(0, tgt_idx).view({n_edges, n_heads_, head_dim_});
    auto key_heads = key.view({n_edges, n_heads_, head_dim_});
    auto attn_scores = (tgt_query * key_heads).sum(-1) /
        std::sqrt(static_cast<float>(head_dim_));  // (n_edges, n_heads)
    auto attn_weights = segment_softmax(attn_scores, tgt_idx, n_nodes);

    auto weighted_msgs = messages.view({n_edges, n_heads_, head_dim_}) *
        attn_weights.unsqueeze(-1);
    auto aggregated = torch::zeros({n_nodes, n_heads_, head_dim_}, node_features.options())
                          .index_add_(0, tgt_idx, weighted_msgs);

    auto out = dropout_->forward(output_->forward(aggregated.reshape({n_nodes, out_features_})));
    if (in_features_ == out_features_) {
        out = out + node_features;
    }
    return norm_->forward(out);
}

// =============================================================================
// Heterogeneous GNN Encoder
// =============================================================================

HeterogeneousGNNEncoderImpl::HeterogeneousGNNEncoderImpl(
    int64_t n_species,
    int64_t hidden_dim,
    int64_t output_dim,
    int64_t n_layers,
    int64_t n_edge_types,
    int64_t n_heads,
    float dropout
) : n_species_(n_species),
    output_dim_(output_dim),
    n_edge_types_(n_edge_types)
{
    species_embeddings_ = register_parameter("species_embeddings",
        torch::randn({n_species, hidden_dim}) * kBertInitStd);

    input_proj_ = register_module("input_proj",
        torch::nn::Linear(hidden_dim, hidden_dim));

    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_layers; ++i) {
        layers_->push_back(
            TypedMessagePassingLayer(
                hidden_dim, hidden_dim, n_edge_types, n_heads, dropout));
    }

    output_proj_ = register_module("output_proj",
        torch::nn::Linear(hidden_dim, output_dim));
    final_norm_ = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({output_dim})));

    graph_edge_index_ = register_buffer("graph_edge_index",
        torch::zeros({2, 0}, torch::kInt64));
    graph_edge_type_ = register_buffer("graph_edge_type", torch::zeros({0}, torch::kInt64));
}

void HeterogeneousGNNEncoderImpl::set_graph(const torch::Tensor& edge_index,
                                            const torch::Tensor& edge_type) {
    auto sorted = sort_typed_edges(edge_index, edge_type, n_species_, n_edge_types_);
    torch::NoGradGuard no_grad;
    graph_edge_index_.set_data(sorted.edge_index.to(graph_edge_index_.device()));
    graph_edge_type_.set_data(sorted.edge_type.to(graph_edge_type_.device()));
    typed_edges_source_ = nullptr;
    cached_embeddings_ = torch::Tensor();
}

const TypedEdges& HeterogeneousGNNEncoderImpl::typed_edges() {
    // The buffers may have been moved or reloaded since the spans were read;
    // the spans depend only on the sorted types, read once per stored graph.
    const void* source = graph_edge_type_.data_ptr();
    if (typed_edges_source_ != source || typed_edges_.type_offsets.empty()) {
        typed_edges_.type_offsets = type_offsets_of(graph_edge_type_, n_edge_types_);
        typed_edges_source_ = source;
    }
    typed_edges_.edge_index = graph_edge_index_;
    typed_edges_.edge_type = graph_edge_type_;
    return typed_edges_;
}

std::vector<std::pair<const void*, int64_t>>
HeterogeneousGNNEncoderImpl::state_signature() const {
    // The autocast state decides the dtype the table is computed in.
    std::vector<std::pair<const void*, int64_t>> signature{
        {nullptr, at::autocast::is_autocast_enabled(at::kCUDA) ? 1 : 0}};
    for (const auto& p : parameters()) signature.emplace_back(p.data_ptr(), p._version());
    for (const auto& b : buffers()) signature.emplace_back(b.data_ptr(), b._version());
    return signature;
}

torch::Tensor HeterogeneousGNNEncoderImpl::forward() {
    TORCH_CHECK(has_graph(),
        "HeterogeneousGNN: species graph not set. Call set_graph() before forward().");

    const bool cacheable = !is_training() && !torch::GradMode::is_enabled();
    if (!cacheable) {
        cached_embeddings_ = torch::Tensor();
    } else if (cached_embeddings_.defined() && cached_signature_ == state_signature()) {
        return cached_embeddings_;
    }

    const TypedEdges& edges = typed_edges();
    auto x = input_proj_->forward(species_embeddings_);
    for (auto& layer : *layers_) {
        x = layer->as<TypedMessagePassingLayerImpl>()->forward(x, edges);
    }
    x = final_norm_->forward(output_proj_->forward(x));

    if (cacheable) {
        cached_embeddings_ = x;
        cached_signature_ = state_signature();
    }
    return x;
}

torch::Tensor HeterogeneousGNNEncoderImpl::aggregate_for_plots(
    torch::Tensor species_embeddings,
    torch::Tensor species_vector
) {
    return sparse_rows_matmul(sparse_rows_normalized(species_vector, kEpsilon),
                              species_embeddings);
}

}  // namespace resolve
