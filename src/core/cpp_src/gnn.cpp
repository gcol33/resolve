#include "resolve/attention.hpp"
#include "resolve/types.hpp"
#include <algorithm>
#include <limits>

namespace resolve {

// =============================================================================
// GCN Layer Implementation
// =============================================================================

GCNLayerImpl::GCNLayerImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias
) {
    linear_ = register_module("linear",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias)));
}

torch::Tensor GCNLayerImpl::forward(torch::Tensor x, torch::Tensor adj) {
    auto transformed = linear_->forward(x);
    return torch::matmul(adj, transformed);
}

// =============================================================================
// GAT Layer Implementation
// =============================================================================

GATLayerImpl::GATLayerImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t n_heads,
    float dropout,
    bool concat
) : in_features_(in_features), out_features_(out_features),
    n_heads_(n_heads), concat_(concat) {

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

torch::Tensor GATLayerImpl::forward(torch::Tensor x, torch::Tensor adj) {
    auto n_nodes = x.size(0);

    auto Wh = W_->forward(x).reshape({n_nodes, n_heads_, out_features_});

    auto attn_src = torch::einsum("nho,ho->nh", {Wh, a_src_});
    auto attn_dst = torch::einsum("nho,ho->nh", {Wh, a_dst_});

    auto attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0);
    attn = leaky_relu_->forward(attn);

    auto mask = (adj == 0).unsqueeze(-1).expand_as(attn);
    attn = attn.masked_fill(mask, kAttentionMaskFill);

    attn = torch::softmax(attn, /*dim=*/1);

    if (dropout_) {
        attn = dropout_->forward(attn);
    }

    auto output = torch::einsum("nmh,mho->nho", {attn, Wh});

    if (concat_) {
        return output.reshape({n_nodes, n_heads_ * out_features_});
    } else {
        return output.mean(/*dim=*/1);
    }
}

// =============================================================================
// GraphSAGE Layer Implementation
// =============================================================================

GraphSAGELayerImpl::GraphSAGELayerImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias
) {
    linear_self_ = register_module("linear_self",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias)));
    linear_neighbor_ = register_module("linear_neighbor",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(false)));
}

torch::Tensor GraphSAGELayerImpl::forward(torch::Tensor x, torch::Tensor adj) {
    auto self_out = linear_self_->forward(x);

    auto degree = adj.sum(/*dim=*/1, /*keepdim=*/true).clamp_min(1.0f);
    auto adj_norm = adj / degree;

    auto neighbor_agg = torch::matmul(adj_norm, x);
    auto neighbor_out = linear_neighbor_->forward(neighbor_agg);

    return self_out + neighbor_out;
}

// =============================================================================
// GNN Encoder Implementation
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
                bool concat = (i < n_layers - 1);
                layers_->push_back(GATLayer(current_dim, next_dim, n_heads, dropout, concat));
                if (concat) {
                    next_dim = next_dim * n_heads;
                }
                break;
            }
            case GNNType::GraphSAGE:
                layers_->push_back(GraphSAGELayer(current_dim, next_dim));
                break;
        }

        current_dim = next_dim;
    }

    if (dropout > 0) {
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }
}

torch::Tensor GNNEncoderImpl::forward_single_graph(torch::Tensor x, torch::Tensor adj) {
    for (size_t i = 0; i < layers_->size(); ++i) {
        switch (gnn_type_) {
            case GNNType::GCN:
                x = layers_->ptr(static_cast<int64_t>(i))->as<GCNLayerImpl>()->forward(x, adj);
                break;
            case GNNType::GAT:
                x = layers_->ptr(static_cast<int64_t>(i))->as<GATLayerImpl>()->forward(x, adj);
                break;
            case GNNType::GraphSAGE:
                x = layers_->ptr(static_cast<int64_t>(i))->as<GraphSAGELayerImpl>()->forward(x, adj);
                break;
        }

        if (i < layers_->size() - 1) {
            x = torch::relu(x);
            if (dropout_) {
                x = dropout_->forward(x);
            }
        }
    }
    return x;
}

torch::Tensor GNNEncoderImpl::forward(torch::Tensor x, torch::Tensor adj) {
    if (x.dim() == 3) {
        auto batch_size = x.size(0);
        std::vector<torch::Tensor> outputs;
        for (int64_t b = 0; b < batch_size; ++b) {
            outputs.push_back(forward_single_graph(x.select(0, b), adj.select(0, b)));
        }
        return torch::stack(outputs);
    }
    return forward_single_graph(x, adj);
}

// =============================================================================
// Typed Message Passing Layer Implementation
// =============================================================================

TypedMessagePassingLayerImpl::TypedMessagePassingLayerImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t n_edge_types,
    int64_t n_heads,
    float dropout
) : n_edge_types_(n_edge_types),
    in_features_(in_features),
    out_features_(out_features)
{
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
    torch::Tensor edge_index,
    torch::Tensor edge_type
) {
    int64_t n_nodes = node_features.size(0);
    int64_t n_edges = edge_index.size(1);

    if (n_edges == 0) {
        if (in_features_ == out_features_) {
            return norm_->forward(node_features);
        }
        return norm_->forward(output_->forward(
            torch::zeros({n_nodes, out_features_}, node_features.options())));
    }

    auto src_idx = edge_index[0];
    auto tgt_idx = edge_index[1];

    auto src_feats = node_features.index_select(0, src_idx);
    auto tgt_feats = node_features.index_select(0, tgt_idx);

    auto messages = torch::zeros({n_edges, out_features_}, node_features.options());
    auto edge_input = torch::cat({src_feats, tgt_feats}, /*dim=*/1);

    for (int64_t t = 0; t < n_edge_types_; ++t) {
        auto mask = (edge_type == t);
        if (!mask.any().item<bool>()) continue;

        auto type_input = edge_input.index({mask});
        auto type_msgs = message_fns_[t]->as<torch::nn::Sequential>()->forward(type_input);
        messages.index_put_({mask}, type_msgs);
    }

    auto query = attn_query_->forward(node_features);
    auto key = attn_key_->forward(messages);

    auto tgt_query = query.index_select(0, tgt_idx);

    auto attn_scores = (tgt_query * key).sum(-1) /
        std::sqrt(static_cast<float>(out_features_));

    // Numerically stable per-target-node softmax: subtract each target node's
    // max score before exp. Without this, large learned attention logits (only
    // scaled by 1/sqrt(out_features)) overflow to inf, and inf/inf yields NaN
    // weights that propagate into every embedding.
    auto tgt_max = torch::full({n_nodes},
        -std::numeric_limits<float>::infinity(), node_features.options());
    tgt_max.scatter_reduce_(0, tgt_idx, attn_scores, "amax", /*include_self=*/true);
    auto attn_exp = (attn_scores - tgt_max.index_select(0, tgt_idx)).exp();
    auto attn_sum = torch::zeros({n_nodes}, node_features.options());
    attn_sum.scatter_add_(0, tgt_idx, attn_exp);
    auto attn_weights = attn_exp /
        (attn_sum.index_select(0, tgt_idx) + kEpsilon);

    auto weighted_msgs = messages * attn_weights.unsqueeze(1);
    auto aggregated = torch::zeros({n_nodes, out_features_}, node_features.options());
    aggregated.scatter_add_(0,
        tgt_idx.unsqueeze(1).expand_as(weighted_msgs),
        weighted_msgs);

    auto out = output_->forward(aggregated);
    out = dropout_->forward(out);

    if (in_features_ == out_features_) {
        out = out + node_features;
    }
    out = norm_->forward(out);

    return out;
}

// =============================================================================
// Heterogeneous GNN Encoder Implementation
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
    output_dim_(output_dim)
{
    species_embeddings_ = register_parameter("species_embeddings",
        torch::randn({n_species, hidden_dim}) * kBertInitStd);

    input_proj_ = register_module("input_proj",
        torch::nn::Linear(hidden_dim, hidden_dim));

    // Register layers only through the ModuleList; an additional
    // register_module("mp_i", ...) would double-register each layer (child of
    // *this AND of layers_), duplicating every parameter in named_parameters().
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
}

torch::Tensor HeterogeneousGNNEncoderImpl::forward(
    torch::Tensor edge_index,
    torch::Tensor edge_type
) {
    auto x = input_proj_->forward(species_embeddings_);

    for (auto& layer : *layers_) {
        x = layer->as<TypedMessagePassingLayerImpl>()->forward(
            x, edge_index, edge_type);
    }

    x = output_proj_->forward(x);
    x = final_norm_->forward(x);

    return x;
}

torch::Tensor HeterogeneousGNNEncoderImpl::aggregate_for_plots(
    torch::Tensor species_embeddings,
    torch::Tensor species_vector
) {
    auto weights = species_vector / (species_vector.sum(/*dim=*/1, /*keepdim=*/true) + kEpsilon);
    return torch::matmul(weights, species_embeddings);
}

// =============================================================================
// Utility: k-NN Adjacency Construction
// =============================================================================

torch::Tensor build_knn_adjacency(torch::Tensor coords, int64_t k) {
    auto n_nodes = coords.size(0);

    // Cap k at the number of *other* nodes: a batch smaller than k (a short
    // final inference batch, or batch_size=1) would otherwise make topk throw.
    int64_t k_eff = std::min<int64_t>(k, std::max<int64_t>(n_nodes - 1, 0));
    if (k_eff == 0) {
        // 0 or 1 node: no neighbours to connect. Return self-loops (identity)
        // rather than zeros so a single-node graph still has a valid, non-empty
        // adjacency row -- an all-zero row makes GAT's masked softmax NaN.
        return torch::eye(n_nodes, coords.options());
    }

    auto diff = coords.unsqueeze(0) - coords.unsqueeze(1);
    auto dist = diff.pow(2).sum(-1).sqrt();

    dist.fill_diagonal_(std::numeric_limits<float>::infinity());
    auto [_, indices] = dist.topk(k_eff, /*dim=*/1, /*largest=*/false);

    auto adj = torch::zeros({n_nodes, n_nodes}, coords.options());

    auto rows = torch::arange(n_nodes, torch::TensorOptions().dtype(torch::kInt64).device(coords.device()))
                     .unsqueeze(1).expand_as(indices);
    adj.index_put_({rows, indices}, 1.0f);

    adj = (adj + adj.transpose(0, 1)).clamp_max(1.0f);

    // Self-loops (A + I): standard GCN normalization, and it guarantees every
    // node has at least one non-zero adjacency entry so GAT's per-row softmax
    // over masked(adj==0) scores is never taken over an all -inf row (which
    // would produce NaN for an isolated node).
    adj = (adj + torch::eye(n_nodes, coords.options())).clamp_max(1.0f);

    auto degree = adj.sum(/*dim=*/1);
    auto d_inv_sqrt = torch::pow(degree.clamp_min(1.0f), -0.5f);
    adj = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0);

    return adj;
}

} // namespace resolve
