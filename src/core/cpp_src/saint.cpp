#include "resolve/saint.hpp"

#include <algorithm>

namespace resolve {

// =============================================================================
// Row Attention
// =============================================================================

RowAttentionImpl::RowAttentionImpl(int64_t d_model, int64_t n_heads, float dropout,
                                   int64_t group_rows)
    : group_rows_(group_rows) {
    TORCH_CHECK(group_rows > 0, "SAINT row attention needs groups of at least one row, got ",
                group_rows);
    attention_ = register_module("attention", MultiHeadAttention(d_model, n_heads, dropout));
    norm_ = register_module("norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
}

torch::Tensor RowAttentionImpl::forward(torch::Tensor x) {
    // (rows, tokens, d) -> (tokens, rows, d): the tokens become the attention
    // batch, so each token attends across rows.
    const int64_t n_rows = x.size(0);
    std::vector<torch::Tensor> groups;
    groups.reserve(static_cast<size_t>((n_rows + group_rows_ - 1) / group_rows_));
    for (int64_t start = 0; start < n_rows; start += group_rows_) {
        auto group = x.narrow(0, start, std::min(group_rows_, n_rows - start)).transpose(0, 1);
        auto normed = norm_->forward(group);
        groups.push_back((group + attention_->forward(normed, normed, normed)).transpose(0, 1));
    }
    return groups.size() == 1 ? groups.front() : torch::cat(groups, /*dim=*/0);
}

torch::Tensor RowAttentionImpl::forward_against(torch::Tensor x, torch::Tensor reference) {
    auto memory = norm_->forward(reference.transpose(0, 1));  // (tokens, n_reference, d)
    // Rows in chunks of one group, so the (tokens, heads, rows, n_reference + 1)
    // scores stay the size a training group's are.
    const int64_t n_rows = x.size(0);
    std::vector<torch::Tensor> chunks;
    for (int64_t start = 0; start < n_rows; start += group_rows_) {
        auto rows = x.narrow(0, start, std::min(group_rows_, n_rows - start)).transpose(0, 1);
        auto attended = attention_->attend_memory_and_self(norm_->forward(rows), memory);
        chunks.push_back((rows + attended).transpose(0, 1));
    }
    if (chunks.empty()) return x;
    return chunks.size() == 1 ? chunks.front() : torch::cat(chunks, /*dim=*/0);
}

// =============================================================================
// SAINT Block
// =============================================================================

SAINTBlockImpl::SAINTBlockImpl(const TransformerBlockConfig& block,
                               int64_t row_attention_rows) {
    col_attention_ = register_module("col_attention", TransformerBlock(block));
    if (row_attention_rows > 0) {
        row_attention_ = register_module("row_attention",
            RowAttention(block.d_model, block.n_heads, block.attention_dropout,
                         row_attention_rows));
    }
}

torch::Tensor SAINTBlockImpl::forward(torch::Tensor x) {
    x = col_attention_->forward(x);
    return row_attention_ ? row_attention_->forward(x) : x;
}

std::pair<torch::Tensor, torch::Tensor> SAINTBlockImpl::forward_with_reference(
    torch::Tensor x, torch::Tensor reference
) {
    const int64_t n_rows = x.size(0);
    // Column attention is per row, so the two sets go through it as one batch.
    auto both = col_attention_->forward(torch::cat({x, reference}, /*dim=*/0));
    x = both.narrow(0, 0, n_rows);
    reference = both.narrow(0, n_rows, both.size(0) - n_rows);
    if (!row_attention_) return {x, reference};

    // The rows read the reference as this block's column attention left it, the
    // way a training row reads its group-mates.
    auto attended = row_attention_->forward_against(x, reference);
    auto updated_reference =
        reference.size(0) > 0 ? row_attention_->forward(reference) : reference;
    return {attended, updated_reference};
}

// =============================================================================
// SAINT Encoder
// =============================================================================

SAINTEncoderImpl::SAINTEncoderImpl(
    int64_t n_numerical,
    const std::vector<int64_t>& cat_cardinalities,
    const TransformerBlockConfig& block,
    int64_t n_layers,
    int64_t row_attention_rows
) : d_model_(block.d_model), row_attention_rows_(std::max<int64_t>(row_attention_rows, 0)) {
    tokenizer_ = register_module("tokenizer",
        FeatureTokenizer(n_numerical, cat_cardinalities, block.d_model));

    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_layers; ++i) {
        layers_->push_back(SAINTBlock(block, row_attention_rows_));
    }

    final_norm_ = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({block.d_model})));
}

torch::Tensor SAINTEncoderImpl::forward(
    torch::Tensor numerical,
    torch::Tensor categoricals,
    torch::Tensor reference_numerical,
    torch::Tensor reference_categoricals
) {
    auto tokens = tokenizer_->forward(numerical, categoricals);

    if (is_training() || row_attention_rows_ == 0) {
        for (auto& layer : *layers_) {
            tokens = layer->as<SAINTBlockImpl>()->forward(tokens);
        }
        return cls_readout(final_norm_->forward(tokens));
    }

    const bool has_reference =
        reference_numerical.defined() && reference_numerical.size(0) > 0;
    auto reference = has_reference
        ? tokenizer_->forward(reference_numerical, reference_categoricals)
              .narrow(0, 0, std::min(reference_numerical.size(0), reference_rows()))
        : tokens.narrow(0, 0, 0);
    for (auto& layer : *layers_) {
        std::tie(tokens, reference) =
            layer->as<SAINTBlockImpl>()->forward_with_reference(tokens, reference);
    }
    return cls_readout(final_norm_->forward(tokens));
}

}  // namespace resolve
