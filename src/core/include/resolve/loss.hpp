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

    // Compute classification loss (with optional class weights for imbalanced data)
    torch::Tensor classification_loss(
        torch::Tensor pred,
        torch::Tensor target,
        torch::Tensor class_weights = {}
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

// Classification metrics result
struct ClassificationMetrics {
    float accuracy;
    float macro_f1;
    float weighted_f1;
    std::vector<float> per_class_precision;
    std::vector<float> per_class_recall;
    std::vector<float> per_class_f1;
    std::vector<int64_t> per_class_support;  // Number of true samples per class
    torch::Tensor confusion_matrix;          // (num_classes, num_classes)
};

// Confidence threshold metrics result (for Exp 5)
struct ConfidenceMetrics {
    float accuracy;      // Accuracy on samples above threshold
    float coverage;      // Fraction of samples above threshold (0-1)
    int64_t n_samples;   // Number of samples above threshold
    int64_t n_total;     // Total number of samples
};

// Metrics computation
struct Metrics {
    // Regression metrics
    static float band_accuracy(torch::Tensor pred, torch::Tensor target, float threshold);
    static float mae(torch::Tensor pred, torch::Tensor target);
    static float rmse(torch::Tensor pred, torch::Tensor target);
    static float smape(torch::Tensor pred, torch::Tensor target, float eps = 1e-8f);
    static float r_squared(torch::Tensor pred, torch::Tensor target);

    // Classification metrics
    static float accuracy(torch::Tensor pred, torch::Tensor target);
    static torch::Tensor confusion_matrix(torch::Tensor pred, torch::Tensor target, int num_classes);
    static ClassificationMetrics classification_metrics(torch::Tensor pred, torch::Tensor target, int num_classes);

    // Confidence threshold metrics (for accuracy-coverage curves)
    // confidence: per-sample confidence values (e.g., 1 - unknown_fraction or softmax max)
    // threshold: minimum confidence to include sample
    static ConfidenceMetrics accuracy_at_threshold(
        torch::Tensor pred,
        torch::Tensor target,
        torch::Tensor confidence,
        float threshold
    );

    // Compute accuracy-coverage curve at multiple thresholds
    static std::vector<ConfidenceMetrics> accuracy_coverage_curve(
        torch::Tensor pred,
        torch::Tensor target,
        torch::Tensor confidence,
        const std::vector<float>& thresholds = {0.0f, 0.5f, 0.8f, 0.9f, 0.95f}
    );

    // Compute all metrics for a target
    // band_thresholds: vector of thresholds for band accuracy (e.g., {0.25, 0.50, 0.75})
    //                  metric names will be "band_25", "band_50", etc. (threshold * 100)
    // num_classes: required for classification tasks to compute per-class F1
    static std::unordered_map<std::string, float> compute(
        torch::Tensor pred,
        torch::Tensor target,
        TaskType task,
        TransformType transform = TransformType::None,
        const std::vector<float>& band_thresholds = {0.25f, 0.50f, 0.75f},
        int num_classes = 0
    );
};

} // namespace resolve
