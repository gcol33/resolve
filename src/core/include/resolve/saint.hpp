#pragma once

// SAINT (Somepalli et al., "SAINT: Improved Neural Networks for Tabular Data via
// Row Attention and Contrastive Pre-Training", arXiv:2106.01342): column
// attention over a row's feature tokens, followed by row (inter-sample)
// attention that lets a row borrow from other rows.
//
// The row attention here runs per token: token t of a row attends over token t
// of the other rows, with weights shared across tokens.
//
// Which rows a row attends to:
// - Training: the batch is cut into groups of row_attention_rows consecutive
//   (shuffled) rows, and every row attends over its group, itself included.
//   The group bound keeps the (tokens, heads, rows, rows) scores bounded at
//   any batch size.
// - Inference: every row attends over a fixed reference set of training rows
//   plus itself, the same shape of neighbourhood it had in training. The
//   reference rows attend among themselves only, so a row's output depends on
//   the row and the reference set and never on which other rows share its
//   batch. Without a reference set a row attends to itself alone.
// The paper attends test rows to the other rows of their test batch, which
// makes a prediction depend on its batch-mates; the fixed reference set keeps
// its nearest-neighbour reading (a row is compared with training rows) while
// making inference a function of the row.

#include "resolve/ft_transformer.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace resolve {

// Row attention for one SAINT block: pre-LayerNorm attention across rows with
// a residual connection.
class RowAttentionImpl : public torch::nn::Module {
public:
    RowAttentionImpl(int64_t d_model, int64_t n_heads, float dropout,
                     int64_t group_rows);

    // x: (rows, n_tokens, d_model). Every row attends over its group of
    // group_rows consecutive rows. Returns the same shape.
    torch::Tensor forward(torch::Tensor x);

    // x: (rows, n_tokens, d_model); reference: (n_reference, n_tokens,
    // d_model), possibly empty. Every row of x attends over the reference rows
    // and itself; the reference is read, not updated.
    torch::Tensor forward_against(torch::Tensor x, torch::Tensor reference);

private:
    int64_t group_rows_;

    MultiHeadAttention attention_{nullptr};
    torch::nn::LayerNorm norm_{nullptr};
};

TORCH_MODULE(RowAttention);

// SAINT block: column attention followed by row attention.
class SAINTBlockImpl : public torch::nn::Module {
public:
    // row_attention_rows <= 0 builds a block without row attention.
    SAINTBlockImpl(const TransformerBlockConfig& block, int64_t row_attention_rows);

    // x: (batch, n_tokens, d_model). Training-time row attention.
    torch::Tensor forward(torch::Tensor x);

    // Inference: x and the reference rows pass column attention alike; row
    // attention updates the reference among itself and attends every row of x
    // over the updated reference and itself. Returns (x, reference).
    std::pair<torch::Tensor, torch::Tensor> forward_with_reference(
        torch::Tensor x, torch::Tensor reference);

private:
    TransformerBlock col_attention_{nullptr};  // Attention across tokens
    RowAttention row_attention_{nullptr};      // Attention across rows
};

TORCH_MODULE(SAINTBlock);

class SAINTEncoderImpl : public torch::nn::Module {
public:
    // row_attention_rows: SAINTConfig::row_attention_rows when
    // SAINTConfig::use_row_attention, else 0 (no row attention).
    SAINTEncoderImpl(
        int64_t n_numerical,
        const std::vector<int64_t>& cat_cardinalities,
        const TransformerBlockConfig& block,
        int64_t n_layers,
        int64_t row_attention_rows
    );

    // numerical: (batch, n_numerical); categoricals as in FeatureTokenizer.
    // reference_numerical / reference_categoricals: the reference rows in the
    // same layout, read only in eval mode (undefined or empty = none).
    // Returns: (batch, d_model), the CLS token.
    torch::Tensor forward(
        torch::Tensor numerical,
        torch::Tensor categoricals = {},
        torch::Tensor reference_numerical = {},
        torch::Tensor reference_categoricals = {}
    );

    [[nodiscard]] int64_t output_dim() const noexcept { return d_model_; }

    // The number of reference rows inference attends over: one group less
    // the row itself. 0 without row attention.
    [[nodiscard]] int64_t reference_rows() const noexcept {
        return row_attention_rows_ > 0 ? row_attention_rows_ - 1 : 0;
    }

private:
    int64_t d_model_;
    int64_t row_attention_rows_;

    FeatureTokenizer tokenizer_{nullptr};
    torch::nn::ModuleList layers_{nullptr};
    torch::nn::LayerNorm final_norm_{nullptr};
};

TORCH_MODULE(SAINTEncoder);

}  // namespace resolve
