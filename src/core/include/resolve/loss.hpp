#pragma once

#include "resolve/types.hpp"
#include <torch/torch.h>

namespace resolve {

// Phased loss for regression targets
// Phase 1: MAE only
// Phase 2: MAE + SMAPE
// Phase 3: MAE + SMAPE + band penalty
class PhasedLoss {
public:
    PhasedLoss(
        std::pair<int, int> phase_boundaries = {100, 300},
        float smape_weight_p2 = 0.2f,
        float smape_weight_p3 = 0.15f,
        float band_weight_p3 = 0.05f,
        float band_threshold = 0.25f,
        float eps = 1e-8f
    );

    // Factory method to create loss from config mode
    static PhasedLoss from_config(LossConfigMode mode, std::pair<int, int> phase_boundaries = {100, 300});

    // Get current phase (1, 2, or 3)
    int get_phase(int epoch) const;

    // Compute regression loss
    torch::Tensor regression_loss(
        torch::Tensor pred,
        torch::Tensor target,
        int epoch,
        torch::Tensor scaler_mean = {},
        torch::Tensor scaler_scale = {},
        TransformType transform = TransformType::None
    ) const;

    // Compute classification loss
    torch::Tensor classification_loss(
        torch::Tensor pred,
        torch::Tensor target
    ) const;

private:
    std::pair<int, int> phase_boundaries_;
    float smape_weight_p2_;
    float smape_weight_p3_;
    float band_weight_p3_;
    float band_threshold_;
    float eps_;
};

// Multi-task loss combiner
class MultiTaskLoss {
public:
    MultiTaskLoss(
        const std::vector<TargetConfig>& targets,
        std::pair<int, int> phase_boundaries = {100, 300},
        LossConfigMode loss_config = LossConfigMode::Combined
    );

    // Compute combined loss
    // Returns (total_loss, individual_losses)
    std::pair<torch::Tensor, std::unordered_map<std::string, torch::Tensor>> compute(
        const std::unordered_map<std::string, torch::Tensor>& predictions,
        const std::unordered_map<std::string, torch::Tensor>& targets,
        int epoch,
        const std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>>& scalers = {}
    ) const;

private:
    std::vector<TargetConfig> targets_;
    PhasedLoss phased_loss_;
};

// Metrics computation
struct Metrics {
    static float band_accuracy(torch::Tensor pred, torch::Tensor target, float threshold = 0.25f);
    static float mae(torch::Tensor pred, torch::Tensor target);
    static float rmse(torch::Tensor pred, torch::Tensor target);
    static float smape(torch::Tensor pred, torch::Tensor target, float eps = 1e-8f);
    static float accuracy(torch::Tensor pred, torch::Tensor target);

    // Compute all metrics for a target
    static std::unordered_map<std::string, float> compute(
        torch::Tensor pred,
        torch::Tensor target,
        TaskType task,
        TransformType transform = TransformType::None
    );
};

} // namespace resolve
