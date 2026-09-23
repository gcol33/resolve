#pragma once

// TraitInteractionMode and the other configuration enums the encoders below
// take as constructor arguments.
#include "resolve/types.hpp"

#include <torch/torch.h>
#include <cmath>
#include <vector>

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
// Transformer Block Configuration
// =============================================================================

// How one transformer block is shaped and where it drops activations. The three
// rates act at different places -- attention dropout on the attention weights,
// FFN dropout inside the feed-forward hidden layer, and residual dropout on a
// sublayer's output before it joins the residual stream -- and FT-Transformer
// (Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
// NeurIPS 2021, arXiv:2106.11959) tunes them separately. An architecture whose
// configuration carries one rate builds this with `uniform`.
struct TransformerBlockConfig {
    int64_t d_model = 0;
    int64_t n_heads = 8;
    int64_t d_ff = 0;  // 0 = 4 * d_model
    float attention_dropout = 0.1f;
    float ffn_dropout = 0.1f;
    float residual_dropout = 0.1f;
    bool pre_norm = true;  // Pre-LN (more stable) vs Post-LN

    // One rate at all three places.
    static TransformerBlockConfig uniform(int64_t d_model, int64_t n_heads,
                                          int64_t d_ff, float dropout,
                                          bool pre_norm = true) {
        return {d_model, n_heads, d_ff, dropout, dropout, dropout, pre_norm};
    }

    // The feed-forward width this block runs at: d_ff, or the standard
    // 4 * d_model when d_ff is left at 0.
    [[nodiscard]] int64_t ffn_dim() const {
        return d_ff == 0 ? 4 * d_model : d_ff;
    }
};

// =============================================================================
// Transformer Encoder Block
// =============================================================================

class TransformerBlockImpl : public torch::nn::Module {
public:
    explicit TransformerBlockImpl(const TransformerBlockConfig& config);

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
    TransformerEncoderImpl(const TransformerBlockConfig& config, int64_t n_layers);

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

// Entmax-1.5: the alpha = 1.5 member of the alpha-entmax family
//     alpha-entmax(z) = [ (alpha - 1) z - tau * 1 ]_+ ^ (1/(alpha - 1))
// (Peters, Niculae & Martins, "Sparse Sequence-to-Sequence Models", ACL 2019,
// arXiv:1905.05702, Eq. 13), i.e. [ z/2 - tau ]_+^2 here. alpha = 1 is softmax
// and alpha = 2 is sparsemax, so 1.5-entmax is sparse but keeps a strictly
// larger support than sparsemax on the same scores.
//
// Computed exactly by the paper's sort-based Algorithm 2 (never by bisection,
// which is the general-alpha `entmax_bisect` routine), with the closed-form
// Proposition 1 Jacobian as the backward. Differentiable, and reduces over an
// arbitrary `dim` of an arbitrarily strided input. See cpp_src/attention.cpp
// for the transcribed algorithm and gradient.
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
        const TransformerBlockConfig& block,
        int64_t n_layers = 3,
        bool use_cls_token = true
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

// GLU block of the TabNet feature transformer: FC -> BN -> GLU.
// The FC produces 2*out_dim; the gated linear unit halves it back to out_dim
// (Arik & Pfister, Fig. 4).
// Ghost batch normalization: normalize each slice of `virtual_batch_size` rows
// with `bn` independently instead of the whole batch at once (Arik & Pfister,
// "TabNet: Attentive Interpretable Tabular Learning", AAAI 2021,
// arXiv:1908.07442, section 3.2, which applies it to every block of the feature
// and attentive transformers -- the initial input normalization stays
// full-batch). One module, so the affine parameters and running statistics are
// shared across the slices and the parameter names are the plain BatchNorm1d
// ones.
//
// A `virtual_batch_size` of 0 (or one no smaller than the batch) normalizes the
// batch in one piece, i.e. plain batch normalization. In eval mode the running
// statistics already describe the training distribution, so the split would
// change nothing and is skipped. A trailing slice of one row would make
// BatchNorm1d throw ("Expected more than 1 value per channel"), so it is folded
// into the slice before it.
torch::Tensor apply_ghost_batch_norm(torch::nn::BatchNorm1d bn,
                                     torch::Tensor x,
                                     int64_t virtual_batch_size);

class TabNetGLUBlockImpl : public torch::nn::Module {
public:
    TabNetGLUBlockImpl(int64_t in_dim, int64_t out_dim,
                       int64_t virtual_batch_size = 0);
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t virtual_batch_size_;

    torch::nn::Linear fc_{nullptr};
    torch::nn::BatchNorm1d bn_{nullptr};
};

TORCH_MODULE(TabNetGLUBlock);

// One TabNet decision step. Holds the attentive transformer (produces a sparse
// feature mask over the ORIGINAL features from the previous step's attention
// split and the prior scale) and the step-specific ("independent") half of the
// feature transformer. The shared half is owned by the encoder and reused
// across steps.
class TabNetStepImpl : public torch::nn::Module {
public:
    TabNetStepImpl(
        int64_t input_dim,
        int64_t n_d,            // Decision layer dimension
        int64_t n_a,            // Attention layer dimension
        int64_t n_independent,  // Step-specific feature-transformer GLU blocks
        bool use_sparsemax = true,  // sparsemax (alpha=2) vs 1.5-entmax mask
        int64_t virtual_batch_size = 0  // Ghost batch norm slice; 0 = full batch
    );

    // att_prev: (batch, n_a) previous step's attention split.
    // prior_scales: (batch, input_dim) feature availability.
    // Returns the simplex mask over features: (batch, input_dim), produced by
    // sparsemax when use_sparsemax, otherwise by entmax15.
    torch::Tensor attentive_forward(torch::Tensor att_prev, torch::Tensor prior_scales);

    // Apply this step's independent GLU blocks after the encoder's shared blocks.
    // shared_out and the return are (batch, n_d + n_a).
    torch::Tensor feature_independent(torch::Tensor shared_out);

    [[nodiscard]] bool use_sparsemax() const noexcept { return use_sparsemax_; }

private:
    int64_t input_dim_;
    int64_t n_d_;
    int64_t n_a_;
    bool use_sparsemax_;
    int64_t virtual_batch_size_;

    torch::nn::Linear attention_fc_{nullptr};
    torch::nn::BatchNorm1d bn_attention_{nullptr};
    torch::nn::ModuleList independent_{nullptr};
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
        float sparsity_coefficient = 1e-3f,
        bool use_sparsemax = true,  // TabNetConfig::use_sparsemax
        int64_t virtual_batch_size = 0  // TabNetConfig::virtual_batch_size
    );

    // x: (batch, input_dim)
    // Returns: (output, feature_importance)
    //   output: (batch, n_d) - aggregated decision output
    //   feature_importance: (batch, input_dim) - interpretable feature masks
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    // Sparsity regularization loss
    [[nodiscard]] torch::Tensor sparsity_loss() const { return sparsity_loss_; }

    [[nodiscard]] int64_t output_dim() const { return n_d_; }

    // Which simplex mapping every step's attentive transformer applies.
    [[nodiscard]] bool use_sparsemax() const noexcept { return use_sparsemax_; }

    // The ghost batch norm slice every block below the initial normalization
    // runs at. 0 = plain batch normalization.
    [[nodiscard]] int64_t virtual_batch_size() const noexcept {
        return virtual_batch_size_;
    }

private:
    // Run the shared feature-transformer GLU blocks (input_dim -> n_d + n_a),
    // with sqrt(0.5) residual scaling between blocks after the first.
    torch::Tensor run_shared(torch::Tensor x) const;

    int64_t input_dim_;
    int64_t n_steps_;
    int64_t n_d_;
    int64_t n_a_;
    float relaxation_factor_;
    float sparsity_coefficient_;
    bool use_sparsemax_;
    int64_t virtual_batch_size_;

    torch::nn::BatchNorm1d initial_bn_{nullptr};  // Input feature batch norm
    torch::nn::ModuleList shared_{nullptr};       // Shared feature-transformer blocks
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
    SAINTBlockImpl(const TransformerBlockConfig& block,
                   bool use_row_attention = true);

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
        const TransformerBlockConfig& block,
        int64_t n_layers = 6,
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
// A trait encoder with its own weights for every species: one
// (trait_dim, hidden_dim) matrix and bias per species per layer, applied with
// an einsum, where the shared encoder applies one matrix to every species. This
// is what TraitNetConfig::shared_trait_encoder = false asks for, and it costs
// n_species times the parameters, so it suits a small species pool rather than
// a whole regional vocabulary.
class PerSpeciesTraitEncoderImpl : public torch::nn::Module {
public:
    PerSpeciesTraitEncoderImpl(
        int64_t n_species,
        int64_t trait_dim,
        int64_t hidden_dim,
        int64_t n_layers = 2,
        float dropout = 0.1f
    );

    // traits: (n_species, trait_dim) -> (n_species, hidden_dim)
    torch::Tensor forward(torch::Tensor traits);

private:
    int64_t n_species_;
    int64_t trait_dim_;

    // Per layer: weight (n_species, in_dim, hidden_dim), bias (n_species,
    // hidden_dim), plus a normalization shared across species.
    std::vector<torch::Tensor> weights_;
    std::vector<torch::Tensor> biases_;
    torch::nn::ModuleList norms_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(PerSpeciesTraitEncoder);

class TraitNetEncoderImpl : public torch::nn::Module {
public:
    TraitNetEncoderImpl(
        int64_t env_dim,           // Number of environmental features
        int64_t trait_dim,         // Number of trait features per species
        int64_t n_species,         // Number of species
        int64_t hidden_dim = 128,
        int64_t n_layers = 2,
        float dropout = 0.1f,
        // TraitNetConfig::interaction_dim: the width the environment-trait
        // combination produces, which the head then reads.
        int64_t interaction_dim = 0,  // 0 = hidden_dim
        // TraitNetConfig::interaction: how the two are combined.
        TraitInteractionMode interaction = TraitInteractionMode::Bilinear,
        // TraitNetConfig::shared_trait_encoder: one trait encoder for every
        // species, or one per species.
        bool shared_trait_encoder = true
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

    // Trait encoder: one of the two, by shared_trait_encoder.
    torch::nn::Sequential trait_encoder_{nullptr};
    PerSpeciesTraitEncoder per_species_trait_encoder_{nullptr};

    // How the environment and a species' traits are combined, and which of the
    // three below is built.
    TraitInteractionMode interaction_mode_;
    int64_t interaction_dim_;

    // Bilinear interaction
    BilinearTraitInteraction interaction_{nullptr};
    // MLP interaction: the two representations concatenated and projected.
    torch::nn::Linear interaction_mlp_{nullptr};
    // Attention interaction: the environment queries each species' traits and
    // scales that species' value vector by the attention it receives.
    torch::nn::Linear interaction_query_{nullptr};
    torch::nn::Linear interaction_key_{nullptr};
    torch::nn::Linear interaction_value_{nullptr};

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
    torch::Tensor forward_single_graph(torch::Tensor x, torch::Tensor adj);

    int64_t out_features_;
    GNNType gnn_type_;

    torch::nn::ModuleList layers_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(GNNEncoder);

// =============================================================================
// ExcelFormer Encoder (Semi-Permeable Attention)
// =============================================================================

// ExcelFormer: FT-Transformer variant where informative features attend to all
// features, while non-informative features only attend to more-informative ones.
// Feature importance is learned via gradient signal during training.
// Chen et al., "ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
class ExcelFormerEncoderImpl : public torch::nn::Module {
public:
    ExcelFormerEncoderImpl(
        int64_t n_numerical,
        std::vector<int64_t> cat_cardinalities,
        const TransformerBlockConfig& block,
        int64_t n_layers = 3,
        float importance_threshold = 0.5f,  // Features above this are "informative"
        bool use_cls_token = true
    );

    // Forward pass
    torch::Tensor forward(
        torch::Tensor numerical,
        std::vector<torch::Tensor> categoricals = {}
    );

    // Get feature importance scores (for interpretability)
    [[nodiscard]] torch::Tensor feature_importance() const;

    [[nodiscard]] int64_t output_dim() const { return d_model_; }

private:
    int64_t d_model_;
    int64_t n_tokens_;
    float importance_threshold_;
    bool use_cls_token_;
    bool pre_norm_;

    FeatureTokenizer tokenizer_{nullptr};

    // Learnable feature importance scores
    torch::Tensor importance_logits_;  // (n_tokens,) before sigmoid

    // Transformer layers (custom forward with semi-permeable mask)
    torch::nn::ModuleList layers_{nullptr};  // TransformerBlocks
    torch::nn::LayerNorm final_norm_{nullptr};

    // Build the semi-permeable attention bias from importance scores.
    // Returns a (1, n_tokens, n_tokens) additive log-bias tensor (leading 1
    // broadcasts over batch) that is ADDED to the pre-softmax scores -- a soft
    // down-weighting, not a hard -inf block.
    [[nodiscard]] torch::Tensor build_attention_mask() const;
};

TORCH_MODULE(ExcelFormerEncoder);

// =============================================================================
// Typed Message Passing Layer (for Heterogeneous GNN)
// =============================================================================

// Each edge type has its own message function (MLP).
// Messages are aggregated via attention-weighted scatter to target nodes.
// Attention runs with n_heads heads over disjoint out_features / n_heads
// slices of the message vector, concatenated back to out_features (the GAT
// convention); out_features must be divisible by n_heads.
class TypedMessagePassingLayerImpl : public torch::nn::Module {
public:
    TypedMessagePassingLayerImpl(
        int64_t in_features,
        int64_t out_features,
        int64_t n_edge_types,
        int64_t n_heads = 4,
        float dropout = 0.1f
    );

    int64_t n_heads() const { return n_heads_; }
    int64_t head_dim() const { return head_dim_; }

    // node_features: (n_nodes, in_features)
    // edge_index: (2, n_edges) - [source, target] node indices
    // edge_type: (n_edges,) - type ID for each edge
    // Returns: (n_nodes, out_features)
    torch::Tensor forward(
        torch::Tensor node_features,
        torch::Tensor edge_index,
        torch::Tensor edge_type
    );

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

// =============================================================================
// Heterogeneous GNN Encoder
// =============================================================================

// Learns species embeddings from a heterogeneous graph with typed edges
// (co-occurrence, same-genus, same-family). The embeddings are aggregated
// per-plot using species abundance vectors.
//
// Node types: species (each species is a node)
// Edge types: co-occurrence, same-genus, same-family
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

    // Run message passing on the species graph
    // edge_index: (2, n_edges) - edge endpoints
    // edge_type: (n_edges,) - type of each edge
    // Returns: (n_species, output_dim) species embeddings
    torch::Tensor forward(
        torch::Tensor edge_index,
        torch::Tensor edge_type
    );

    // Aggregate species embeddings into per-plot features
    // species_embeddings: (n_species, output_dim) from forward()
    // species_vector: (batch, n_species) abundance/presence vector
    // Returns: (batch, output_dim) plot-level features
    [[nodiscard]] static torch::Tensor aggregate_for_plots(
        torch::Tensor species_embeddings,
        torch::Tensor species_vector
    );

    [[nodiscard]] int64_t output_dim() const noexcept { return output_dim_; }
    [[nodiscard]] int64_t n_species() const noexcept { return n_species_; }

private:
    int64_t n_species_;
    int64_t output_dim_;

    // Learnable species embeddings (initial node features)
    torch::Tensor species_embeddings_;  // (n_species, hidden_dim)

    // Input projection from hidden_dim
    torch::nn::Linear input_proj_{nullptr};

    // Message passing layers
    torch::nn::ModuleList layers_{nullptr};

    // Final projection to output_dim
    torch::nn::Linear output_proj_{nullptr};
    torch::nn::LayerNorm final_norm_{nullptr};
};

TORCH_MODULE(HeterogeneousGNNEncoder);

// How a k-NN graph measures the distance between two nodes.
enum class GraphMetric {
    Euclidean,  // Straight-line distance; the metric for coordinates
    Cosine      // 1 - cosine similarity; the metric for composition vectors
};

// Utility: build a k-NN adjacency over node features
// features: (n_nodes, n_features) - coordinates, or a composition vector
// k: number of neighbors
// metric: what "near" means for these features
// weighted: keep each kept edge's similarity as its weight (a Gaussian kernel
//   of the distance under Euclidean, the cosine similarity itself under
//   Cosine) instead of a plain 1 -- GNNConfig::use_edge_features
// Returns: (n_nodes, n_nodes) symmetric, self-looped, degree-normalized
// adjacency
torch::Tensor build_knn_adjacency(torch::Tensor features, int64_t k,
                                  GraphMetric metric = GraphMetric::Euclidean,
                                  bool weighted = false);

}  // namespace resolve
