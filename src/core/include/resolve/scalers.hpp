#pragma once

#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <utility>

namespace resolve {

// Standardisation fitted on the training rows, applied to every row the model
// later reads.
struct Scalers {
    // Per column of the continuous block: the value a missing cell is filled
    // with before standardisation (the mean of that column's recorded values
    // on the fitting rows), then the mean and scale of the filled column.
    // continuous_fill is undefined for a checkpoint written before missing
    // values were filled; such a block never carries a missing cell.
    torch::Tensor continuous_fill;
    torch::Tensor continuous_mean;
    torch::Tensor continuous_scale;
    std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>> target_scalers;
};

}  // namespace resolve
