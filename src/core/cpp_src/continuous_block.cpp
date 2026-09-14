#include "resolve/continuous_block.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace resolve {

namespace {

bool present(const torch::Tensor& t) {
    return t.defined() && t.numel() > 0;
}

// A row is flagged when any of its value columns is missing.
torch::Tensor missing_flag(const torch::Tensor& values) {
    auto missing = torch::isnan(values);
    if (missing.dim() == 2 && values.size(1) > 1) {
        missing = missing.any(/*dim=*/1, /*keepdim=*/true);
    }
    return missing.to(torch::kFloat32);
}

}  // namespace

torch::Tensor assemble_continuous(const ContinuousInputs& inputs,
                                  MissingValuePolicy policy,
                                  int64_t n_rows) {
    const bool indicate = policy == MissingValuePolicy::Indicate;
    std::vector<torch::Tensor> parts;

    if (present(inputs.coordinates)) {
        auto coords = inputs.coordinates.to(torch::kFloat32);
        if (indicate) {
            parts.push_back(coords);
            parts.push_back(missing_flag(coords));
        } else {
            parts.push_back(torch::nan_to_num(coords, 0.0));
        }
    }
    if (inputs.covariates.defined() && inputs.covariates.dim() == 2 &&
        inputs.covariates.size(1) > 0 && inputs.covariates.size(0) > 0) {
        auto covariates = inputs.covariates.to(torch::kFloat32);
        if (indicate) {
            parts.push_back(covariates);
            parts.push_back(torch::isnan(covariates).to(torch::kFloat32));
        } else {
            parts.push_back(torch::nan_to_num(covariates, 0.0));
        }
    }
    if (present(inputs.unknown_fraction)) {
        parts.push_back(inputs.unknown_fraction.to(torch::kFloat32).reshape({-1, 1}));
    }
    if (present(inputs.unknown_count)) {
        parts.push_back(inputs.unknown_count.to(torch::kFloat32).reshape({-1, 1}));
    }
    if (present(inputs.hash_embedding)) {
        parts.push_back(inputs.hash_embedding.to(torch::kFloat32));
    }

    if (parts.empty()) {
        return torch::zeros({n_rows, 0}, torch::kFloat32);
    }
    for (const auto& part : parts) {
        if (part.size(0) != parts.front().size(0)) {
            throw std::invalid_argument(
                "assemble_continuous: the continuous inputs disagree on the number "
                "of rows (" + std::to_string(parts.front().size(0)) + " and " +
                std::to_string(part.size(0)) + ")");
        }
    }
    return torch::cat(parts, /*dim=*/1);
}

void fit_continuous_scalers(Scalers& scalers, const torch::Tensor& fitting_rows) {
    if (!fitting_rows.defined() || fitting_rows.size(1) == 0) {
        return;
    }
    auto observed = ~torch::isnan(fitting_rows);
    auto count = observed.sum(/*dim=*/0).to(torch::kFloat32);
    auto total = torch::where(observed, fitting_rows,
                              torch::zeros_like(fitting_rows)).sum(/*dim=*/0);
    scalers.continuous_fill = torch::where(count > 0, total / count.clamp_min(1.0),
                                           torch::zeros_like(total));
    auto filled = torch::where(observed, fitting_rows,
                               scalers.continuous_fill.expand_as(fitting_rows));
    scalers.continuous_mean = filled.mean(/*dim=*/0);
    scalers.continuous_scale = filled.std(/*dim=*/0) + 1e-8f;
}

torch::Tensor standardize_continuous(const torch::Tensor& block, const Scalers& scalers) {
    if (!block.defined() || block.size(1) == 0) {
        return block;
    }
    torch::Tensor filled;
    if (scalers.continuous_fill.defined()) {
        filled = torch::where(torch::isnan(block),
                              scalers.continuous_fill.to(block.device()).expand_as(block),
                              block);
    } else {
        filled = torch::nan_to_num(block, 0.0);
    }
    if (!scalers.continuous_mean.defined()) {
        return filled;
    }
    return (filled - scalers.continuous_mean.to(block.device())) /
           scalers.continuous_scale.to(block.device());
}

std::vector<std::pair<std::vector<int64_t>, int64_t>>
missing_flag_columns(const ResolveSchema& schema) {
    std::vector<std::pair<std::vector<int64_t>, int64_t>> columns;
    if (schema.missing_values != MissingValuePolicy::Indicate) {
        return columns;
    }
    int64_t at = 0;
    if (schema.has_coordinates) {
        columns.push_back({{0, 1}, 2});
        at = 3;
    }
    const auto k = static_cast<int64_t>(schema.covariate_names.size());
    for (int64_t i = 0; i < k; ++i) {
        columns.push_back({{at + i}, at + k + i});
    }
    return columns;
}

torch::Tensor unstandardize_continuous(const torch::Tensor& block,
                                       const Scalers& scalers,
                                       const ResolveSchema& schema) {
    if (!block.defined() || block.size(1) == 0) {
        return block;
    }
    auto raw = block;
    if (scalers.continuous_mean.defined() && scalers.continuous_scale.defined()) {
        raw = block * scalers.continuous_scale.to(block.device()) +
              scalers.continuous_mean.to(block.device());
    }
    const auto flags = missing_flag_columns(schema);
    if (flags.empty()) {
        return raw;
    }
    raw = raw.clone();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    for (const auto& [values, flag] : flags) {
        // Rescaled, a flag column holds 0.0 and 1.0 again.
        auto missing = raw.select(1, flag) > 0.5f;
        for (int64_t col : values) {
            raw.select(1, col).masked_fill_(missing, nan);
        }
    }
    return raw;
}

}  // namespace resolve
