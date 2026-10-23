#include "resolve/excelformer.hpp"

namespace resolve {

namespace {
// Temperature of the smooth comparisons that build the attention gate.
constexpr float kGateTemperature = 0.1f;
// Keeps the log of a fully closed gate finite.
constexpr float kGateFloor = 1e-9f;
}  // namespace

ExcelFormerEncoderImpl::ExcelFormerEncoderImpl(
    int64_t n_numerical,
    const std::vector<int64_t>& cat_cardinalities,
    const TransformerBlockConfig& block,
    int64_t n_layers,
    float importance_threshold
) : d_model_(block.d_model),
    importance_threshold_(importance_threshold) {
    tokenizer_ = register_module("tokenizer",
        FeatureTokenizer(n_numerical, cat_cardinalities, block.d_model));
    importance_logits_ = register_parameter("importance_logits",
        torch::zeros({tokenizer_->n_tokens()}));
    encoder_ = register_module("encoder", TransformerEncoder(block, n_layers));
}

torch::Tensor ExcelFormerEncoderImpl::build_attention_bias() const {
    auto importance = torch::sigmoid(importance_logits_);  // (n_tokens,)
    auto query_importance = importance.unsqueeze(1);       // token i, (n_tokens, 1)
    auto key_importance = importance.unsqueeze(0);         // token j, (1, n_tokens)

    // Token i may attend to token j when j is informative OR at least as
    // important as i. Smooth comparisons keep the gate differentiable: a hard
    // (imp >= thr) has zero gradient, so the importance would never train.
    auto is_informative =
        torch::sigmoid((key_importance - importance_threshold_) / kGateTemperature);
    auto is_more_important =
        torch::sigmoid((key_importance - query_importance) / kGateTemperature);
    // Soft OR, a + b - ab, in (0, 1).
    auto can_attend = is_informative + is_more_important - is_informative * is_more_important;

    return torch::log(can_attend + kGateFloor).unsqueeze(0);
}

torch::Tensor ExcelFormerEncoderImpl::forward(torch::Tensor numerical,
                                              torch::Tensor categoricals) {
    auto tokens = tokenizer_->forward(numerical, categoricals);
    return cls_readout(encoder_->forward(tokens, build_attention_bias()));
}

}  // namespace resolve
