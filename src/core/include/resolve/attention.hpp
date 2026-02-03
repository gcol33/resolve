#pragma once

#include <torch/torch.h>
#include <cmath>

namespace resolve {

// =============================================================================
// Multi-Head Attention
// =============================================================================

class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(
        int64_t d_model,
        int64_t n_heads,
        float dropout = 0.0f,
        bool bias = true
    );

    // Forward pass
    // query, key, value: (batch, seq_len, d_model)
    // mask: optional (batch, seq_len) or (batch, seq_len, seq_len)
    // Returns: (batch, seq_len, d_model)
    torch::Tensor forward(
        torch::Tensor query,
        torch::Tensor key,
        torch::Tensor value,
        torch::Tensor mask = {}
    );

private:
    int64_t d_model_;
    int64_t n_heads_;
    int64_t d_k_;  // d_model / n_heads
    float scale_;

    torch::nn::Linear W_q_{nullptr};
    torch::nn::Linear W_k_{nullptr};
    torch::nn::Linear W_v_{nullptr};
    torch::nn::Linear W_o_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(MultiHeadAttention);

// =============================================================================
// Position-wise Feed-Forward Network
// =============================================================================

class FeedForwardImpl : public torch::nn::Module {
public:
    FeedForwardImpl(
        int64_t d_model,
        int64_t d_ff,
        float dropout = 0.0f,
        bool use_gelu = true
    );

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear linear1_{nullptr};
    torch::nn::Linear linear2_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    bool use_gelu_;
};

TORCH_MODULE(FeedForward);

// =============================================================================
// Transformer Encoder Block
// =============================================================================

class TransformerBlockImpl : public torch::nn::Module {
public:
    TransformerBlockImpl(
        int64_t d_model,
        int64_t n_heads,
        int64_t d_ff = 0,  // Default: 4 * d_model
        float dropout = 0.1f,
        bool pre_norm = true  // Pre-LN (more stable) vs Post-LN
    );

    // x: (batch, seq_len, d_model)
    // mask: optional attention mask
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {});

private:
    MultiHeadAttention attention_{nullptr};
    FeedForward ffn_{nullptr};
    torch::nn::LayerNorm norm1_{nullptr};
    torch::nn::LayerNorm norm2_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    bool pre_norm_;
};

TORCH_MODULE(TransformerBlock);

// =============================================================================
// Transformer Encoder (stack of blocks)
// =============================================================================

class TransformerEncoderImpl : public torch::nn::Module {
public:
    TransformerEncoderImpl(
        int64_t d_model,
        int64_t n_heads,
        int64_t n_layers,
        int64_t d_ff = 0,
        float dropout = 0.1f,
        bool pre_norm = true
    );

    // x: (batch, seq_len, d_model)
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {});

private:
    torch::nn::ModuleList layers_{nullptr};
    torch::nn::LayerNorm final_norm_{nullptr};
    bool pre_norm_;
};

TORCH_MODULE(TransformerEncoder);

// =============================================================================
// Feature Tokenizer (for FT-Transformer)
// =============================================================================

// Converts each feature (numerical or categorical) into a d_model embedding
class FeatureTokenizerImpl : public torch::nn::Module {
public:
    FeatureTokenizerImpl(
        int64_t n_numerical,           // Number of numerical features
        std::vector<int64_t> cat_cardinalities,  // Cardinality of each categorical
        int64_t d_model,
        bool use_cls_token = true
    );

    // numerical: (batch, n_numerical)
    // categoricals: list of (batch,) tensors for each categorical feature
    // Returns: (batch, n_tokens, d_model)
    torch::Tensor forward(
        torch::Tensor numerical,
        std::vector<torch::Tensor> categoricals = {}
    );

    int64_t n_tokens() const { return n_tokens_; }

private:
    int64_t n_numerical_;
    int64_t n_categorical_;
    int64_t d_model_;
    int64_t n_tokens_;
    bool use_cls_token_;

    // Numerical feature tokenization: one linear per feature
    torch::nn::ModuleList numerical_embeddings_{nullptr};

    // Categorical feature tokenization: embedding per categorical
    torch::nn::ModuleList categorical_embeddings_{nullptr};

    // CLS token (learnable)
    torch::Tensor cls_token_;
};

TORCH_MODULE(FeatureTokenizer);

// =============================================================================
// Sparsemax and Entmax (for TabNet)
// =============================================================================

// Sparsemax: sparse alternative to softmax
// Projects onto probability simplex, producing exact zeros
torch::Tensor sparsemax(torch::Tensor input, int64_t dim = -1);

// Entmax-1.5: smoother version of sparsemax
// alpha=1 is softmax, alpha=2 is sparsemax, alpha=1.5 is in between
torch::Tensor entmax15(torch::Tensor input, int64_t dim = -1);

// =============================================================================
// Row Attention (for SAINT)
// =============================================================================

// Attention across samples (rows) instead of features
// Requires all samples in batch to attend to each other
class RowAttentionImpl : public torch::nn::Module {
public:
    RowAttentionImpl(
        int64_t d_model,
        int64_t n_heads,
        float dropout = 0.0f
    );

    // x: (batch, n_features, d_model)
    // Returns: (batch, n_features, d_model)
    // Attention is computed across the batch dimension for each feature
    torch::Tensor forward(torch::Tensor x);

private:
    MultiHeadAttention attention_{nullptr};
    torch::nn::LayerNorm norm_{nullptr};
};

TORCH_MODULE(RowAttention);

// =============================================================================
// FT-Transformer Encoder (Feature Tokenizer + Transformer)
// =============================================================================

// Complete FT-Transformer for tabular data
// Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data"
class FTTransformerEncoderImpl : public torch::nn::Module {
public:
    FTTransformerEncoderImpl(
        int64_t n_numerical,
        std::vector<int64_t> cat_cardinalities,
        int64_t d_model = 192,
        int64_t n_heads = 8,
        int64_t n_layers = 3,
        int64_t d_ff = 0,  // Default: 4 * d_model
        float dropout = 0.1f,
        bool use_cls_token = true,
        bool pre_norm = true
    );

    // numerical: (batch, n_numerical)
    // categoricals: list of (batch,) tensors
    // Returns: (batch, output_dim) if use_cls_token, else (batch, n_tokens, d_model)
    torch::Tensor forward(
        torch::Tensor numerical,
        std::vector<torch::Tensor> categoricals = {}
    );

    [[nodiscard]] int64_t output_dim() const { return d_model_; }
    [[nodiscard]] int64_t n_tokens() const { return tokenizer_->n_tokens(); }

private:
    int64_t d_model_;
    bool use_cls_token_;

    FeatureTokenizer tokenizer_{nullptr};
    TransformerEncoder encoder_{nullptr};
};

TORCH_MODULE(FTTransformerEncoder);

// =============================================================================
// TabNet Encoder
// =============================================================================

// Feature step for TabNet sequential attention
class TabNetStepImpl : public torch::nn::Module {
public:
    TabNetStepImpl(
        int64_t input_dim,
        int64_t n_d,  // Decision layer dimension
        int64_t n_a,  // Attention layer dimension
        float relaxation_factor = 1.5f
    );

    // x: (batch, input_dim)
    // prior_scales: (batch, input_dim) - accumulated attention from previous steps
    // Returns: (decision_output, attention_output, new_prior_scales)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor x,
        torch::Tensor prior_scales
    );

private:
    int64_t input_dim_;
    int64_t n_d_;
    int64_t n_a_;
    float relaxation_factor_;

    torch::nn::Linear shared_fc_{nullptr};
    torch::nn::Linear decision_fc_{nullptr};
    torch::nn::Linear attention_fc_{nullptr};
    torch::nn::BatchNorm1d bn_shared_{nullptr};
    torch::nn::BatchNorm1d bn_decision_{nullptr};
    torch::nn::BatchNorm1d bn_attention_{nullptr};
};

TORCH_MODULE(TabNetStep);

// Complete TabNet encoder
// Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning"
class TabNetEncoderImpl : public torch::nn::Module {
public:
    TabNetEncoderImpl(
        int64_t input_dim,
        int64_t n_steps = 3,
        int64_t n_d = 64,
        int64_t n_a = 64,
        float relaxation_factor = 1.5f,
        float sparsity_coefficient = 1e-3f
    );

    // x: (batch, input_dim)
    // Returns: (output, feature_importance)
    //   output: (batch, n_d) - aggregated decision output
    //   feature_importance: (batch, input_dim) - interpretable feature masks
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    // Sparsity regularization loss
    [[nodiscard]] torch::Tensor sparsity_loss() const { return sparsity_loss_; }

    [[nodiscard]] int64_t output_dim() const { return n_d_; }

private:
    int64_t input_dim_;
    int64_t n_steps_;
    int64_t n_d_;
    int64_t n_a_;
    float sparsity_coefficient_;

    torch::nn::Linear initial_bn_{nullptr};  // Initial batch norm (linear with 1 weight)
    torch::nn::ModuleList steps_{nullptr};

    // Accumulated during forward pass
    mutable torch::Tensor sparsity_loss_;
};

TORCH_MODULE(TabNetEncoder);

// =============================================================================
// SAINT Encoder (Self-Attention + Inter-sample Attention)
// =============================================================================

// SAINT block: column attention followed by row attention
class SAINTBlockImpl : public torch::nn::Module {
public:
    SAINTBlockImpl(
        int64_t d_model,
        int64_t n_heads,
        int64_t d_ff = 0,
        float dropout = 0.1f,
        bool use_row_attention = true
    );

    // x: (batch, n_features, d_model)
    torch::Tensor forward(torch::Tensor x);

private:
    bool use_row_attention_;

    TransformerBlock col_attention_{nullptr};  // Attention across features
    RowAttention row_attention_{nullptr};       // Attention across samples
};

TORCH_MODULE(SAINTBlock);

// Complete SAINT encoder
// Somepalli et al., "SAINT: Improved Neural Networks for Tabular Data"
class SAINTEncoderImpl : public torch::nn::Module {
public:
    SAINTEncoderImpl(
        int64_t n_numerical,
        std::vector<int64_t> cat_cardinalities,
        int64_t d_model = 128,
        int64_t n_heads = 8,
        int64_t n_layers = 6,
        int64_t d_ff = 0,
        float dropout = 0.1f,
        bool use_row_attention = true,
        bool use_cls_token = true
    );

    torch::Tensor forward(
        torch::Tensor numerical,
        std::vector<torch::Tensor> categoricals = {}
    );

    [[nodiscard]] int64_t output_dim() const { return d_model_; }

private:
    int64_t d_model_;
    bool use_cls_token_;

    FeatureTokenizer tokenizer_{nullptr};
    torch::nn::ModuleList layers_{nullptr};
    torch::nn::LayerNorm final_norm_{nullptr};
};

TORCH_MODULE(SAINTEncoder);

// =============================================================================
// Trait-based Multi-species Network
// =============================================================================

// Bilinear interaction between environment and traits
class BilinearTraitInteractionImpl : public torch::nn::Module {
public:
    BilinearTraitInteractionImpl(
        int64_t env_dim,
        int64_t trait_dim,
        int64_t output_dim
    );

    // env: (batch, env_dim) - environmental features
    // traits: (n_species, trait_dim) - species traits
    // Returns: (batch, n_species, output_dim)
    torch::Tensor forward(torch::Tensor env, torch::Tensor traits);

private:
    int64_t env_dim_;
    int64_t trait_dim_;
    int64_t output_dim_;

    // Bilinear weight: (output_dim, env_dim, trait_dim)
    torch::Tensor weight_;
    torch::Tensor bias_;
};

TORCH_MODULE(BilinearTraitInteraction);

// Complete Trait-based network for multi-species modeling
class TraitNetEncoderImpl : public torch::nn::Module {
public:
    TraitNetEncoderImpl(
        int64_t env_dim,           // Number of environmental features
        int64_t trait_dim,         // Number of trait features per species
        int64_t n_species,         // Number of species
        int64_t hidden_dim = 128,
        int64_t n_layers = 2,
        float dropout = 0.1f
    );

    // env: (batch, env_dim) - environmental features
    // traits: (n_species, trait_dim) - species traits (can be set once via set_traits)
    // Returns: (batch, n_species) - species predictions
    torch::Tensor forward(torch::Tensor env, torch::Tensor traits = {});

    // Set species traits (stored and reused across forward calls)
    void set_traits(torch::Tensor traits);

    [[nodiscard]] int64_t output_dim() const { return n_species_; }

private:
    int64_t env_dim_;
    int64_t trait_dim_;
    int64_t n_species_;
    int64_t hidden_dim_;

    // Stored traits
    torch::Tensor traits_;

    // Environment encoder
    torch::nn::Sequential env_encoder_{nullptr};

    // Trait encoder
    torch::nn::Sequential trait_encoder_{nullptr};

    // Bilinear interaction
    BilinearTraitInteraction interaction_{nullptr};

    // Output projection
    torch::nn::Linear output_proj_{nullptr};
};

TORCH_MODULE(TraitNetEncoder);

// =============================================================================
// Graph Neural Network (GNN) Encoder
// =============================================================================

// Graph Convolutional Layer (GCN)
class GCNLayerImpl : public torch::nn::Module {
public:
    GCNLayerImpl(
        int64_t in_features,
        int64_t out_features,
        bool bias = true
    );

    // x: (n_nodes, in_features) - node features
    // adj: (n_nodes, n_nodes) - adjacency matrix (normalized)
    // Returns: (n_nodes, out_features)
    torch::Tensor forward(torch::Tensor x, torch::Tensor adj);

private:
    torch::nn::Linear linear_{nullptr};
};

TORCH_MODULE(GCNLayer);

// Graph Attention Layer (GAT)
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
    // adj: (n_nodes, n_nodes) - adjacency matrix (binary mask)
    // Returns: (n_nodes, out_features * n_heads) if concat, else (n_nodes, out_features)
    torch::Tensor forward(torch::Tensor x, torch::Tensor adj);

private:
    int64_t in_features_;
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

// GraphSAGE Layer
class GraphSAGELayerImpl : public torch::nn::Module {
public:
    GraphSAGELayerImpl(
        int64_t in_features,
        int64_t out_features,
        bool bias = true
    );

    // x: (n_nodes, in_features)
    // adj: (n_nodes, n_nodes) - adjacency matrix
    torch::Tensor forward(torch::Tensor x, torch::Tensor adj);

private:
    torch::nn::Linear linear_self_{nullptr};
    torch::nn::Linear linear_neighbor_{nullptr};
};

TORCH_MODULE(GraphSAGELayer);

// Complete GNN encoder
class GNNEncoderImpl : public torch::nn::Module {
public:
    enum class GNNType { GCN, GAT, GraphSAGE };

    GNNEncoderImpl(
        int64_t in_features,
        int64_t hidden_dim = 64,
        int64_t out_features = 32,
        int64_t n_layers = 2,
        GNNType gnn_type = GNNType::GCN,
        int64_t n_heads = 4,  // For GAT
        float dropout = 0.1f
    );

    // x: (n_nodes, in_features) or (batch, n_nodes, in_features)
    // adj: (n_nodes, n_nodes) or (batch, n_nodes, n_nodes)
    // Returns: (n_nodes, out_features) or (batch, n_nodes, out_features)
    torch::Tensor forward(torch::Tensor x, torch::Tensor adj);

    [[nodiscard]] int64_t output_dim() const { return out_features_; }

private:
    int64_t out_features_;
    GNNType gnn_type_;

    torch::nn::ModuleList layers_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(GNNEncoder);

// Utility: Build k-NN adjacency from coordinates
// coords: (n_nodes, 2) - spatial coordinates
// k: number of neighbors
// Returns: (n_nodes, n_nodes) normalized adjacency
torch::Tensor build_knn_adjacency(torch::Tensor coords, int64_t k);

}  // namespace resolve
