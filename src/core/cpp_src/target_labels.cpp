#include "resolve/target_labels.hpp"

namespace resolve {

torch::Tensor labelled_mask(const torch::Tensor& target, TaskType task) {
    const auto flat = target.reshape({-1});
    if (task == TaskType::Classification) {
        return flat.to(torch::kLong) >= 0;
    }
    return torch::isfinite(flat);
}

torch::Tensor LabelledRows::select(const torch::Tensor& rows) const {
    if (!index.defined()) return rows;
    return rows.index_select(0, index.to(rows.device()));
}

torch::Tensor LabelledRows::positions() const {
    if (index.defined()) return index.to(torch::kCPU).to(torch::kLong).contiguous();
    return torch::arange(n_rows, torch::kLong);
}

LabelledRows labelled_rows(const TargetConfig& cfg, const torch::Tensor& target) {
    LabelledRows out;
    out.n_rows = target.defined() && target.dim() > 0 ? target.size(0) : 0;
    if (cfg.missing != TargetMissingPolicy::Mask) {
        out.n_labelled = out.n_rows;
        return out;
    }
    const auto index = labelled_mask(target, cfg.task).nonzero().reshape({-1});
    out.n_labelled = index.size(0);
    if (out.n_labelled < out.n_rows) out.index = index;
    return out;
}

}  // namespace resolve
