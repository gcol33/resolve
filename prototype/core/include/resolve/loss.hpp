#pragma once

#include "types.hpp"

#include <torch/torch.h>
#include <unordered_map>
#include <vector>
#include <optional>
#include <cmath>
#include <stdexcept>

namespace resolve {

/**
 * Configuration for loss components in a single training phase.
 *
 * All weights should sum to 1.0 for interpretability, but this is not enforced.
 *
 * Available loss components:
 *   - mae: Mean Absolute Error (L1) - robust to outliers
 *   - mse: Mean Squared Error (L2) - penalizes large errors more
 *   - huber: Huber loss - combines MAE/MSE, smooth transition at delta
 *   - smape: Symmetric Mean Absolute Percentage Error - relative accuracy
 *   - band: Band penalty - penalizes predictions outside threshold
 */
struct PhaseConfig {
    float mae = 0.0f;
    float mse = 0.0f;
    float huber = 0.0f;
    float smape = 0.0f;
    float band = 0.0f;

    // Huber delta (transition point between L1 and L2)
    float huber_delta = 1.0f;
    // Band threshold (relative error threshold for penalty)
    float band_threshold = 0.25f;

    /**
     * Check if at least one loss component has non-zero weight.
     */
    bool is_valid() const {
        return (mae + mse + huber + smape + band) > 0.0f;
    }

    /**
     * Whether this phase needs original-scale values (for SMAPE/band).
     */
    bool needs_original_scale() const {
        return smape > 0.0f || band > 0.0f;
    }

    // Convenience factory methods
    static PhaseConfig mae_only() {
        return PhaseConfig{1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    static PhaseConfig combined(float mae_w = 0.7f, float smape_w = 0.2f, float band_w = 0.1f) {
        return PhaseConfig{mae_w, 0.0f, 0.0f, smape_w, band_w};
    }
};


/**
 * Phased loss for regression targets with configurable loss components.
 *
 * Users configure which loss functions to use in each phase via PhaseConfig.
 * Supports any number of phases (1, 2, 3, ...).
 *
 * Example:
 *   // Single phase (no boundaries needed)
 *   PhasedLoss loss({{1, PhaseConfig::mae_only()}});
 *
 *   // Two phases
 *   PhasedLoss loss(
 *       {{1, PhaseConfig::mae_only()},
 *        {2, PhaseConfig::combined()}},
 *       {100}
 *   );
 *
 *   // Three phases
 *   PhasedLoss loss(
 *       {{1, PhaseConfig{1.0f}},                           // MAE only
 *        {2, PhaseConfig{0.8f, 0.0f, 0.0f, 0.2f}},         // MAE + SMAPE
 *        {3, PhaseConfig{0.7f, 0.0f, 0.0f, 0.2f, 0.1f}}},  // MAE + SMAPE + band
 *       {100, 300}
 *   );
 */
class PhasedLoss {
public:
    /**
     * Construct phased loss.
     *
     * @param phases Map from phase number (1, 2, ...) to PhaseConfig
     * @param phase_boundaries List of epoch thresholds for phase transitions
     *                         Length must be num_phases - 1
     * @param eps Small constant for numerical stability
     */
    PhasedLoss(
        std::unordered_map<int, PhaseConfig> phases = {{1, PhaseConfig::mae_only()}},
        std::vector<int> phase_boundaries = {},
        float eps = 1e-8f
    ) : phases_(std::move(phases)),
        phase_boundaries_(std::move(phase_boundaries)),
        eps_(eps)
    {
        validate();
    }

    /**
     * Get current training phase (1-indexed).
     */
    int get_phase(int epoch) const {
        for (size_t i = 0; i < phase_boundaries_.size(); ++i) {
            if (epoch < phase_boundaries_[i]) {
                return static_cast<int>(i + 1);
            }
        }
        return static_cast<int>(phases_.size());
    }

    /**
     * Get phase configuration for given epoch.
     */
    const PhaseConfig& get_config(int epoch) const {
        int phase = get_phase(epoch);
        return phases_.at(phase);
    }

    /**
     * Compute phased regression loss.
     *
     * @param pred Predictions (scaled)
     * @param target Targets (scaled)
     * @param epoch Current epoch for phase determination
     * @param scaler_mean Target scaler mean for inverse transform (optional)
     * @param scaler_scale Target scaler scale for inverse transform (optional)
     * @param transform "log1p" or empty
     */
    torch::Tensor regression_loss(
        const torch::Tensor& pred,
        const torch::Tensor& target,
        int epoch,
        const std::optional<torch::Tensor>& scaler_mean = std::nullopt,
        const std::optional<torch::Tensor>& scaler_scale = std::nullopt,
        TransformType transform = TransformType::None
    ) const {
        const auto& config = get_config(epoch);

        // Compute original-scale values if needed for SMAPE/band
        torch::Tensor pred_raw, target_raw;
        if (config.needs_original_scale() && scaler_mean.has_value() && scaler_scale.has_value()) {
            auto pred_orig = pred * scaler_scale.value() + scaler_mean.value();
            auto target_orig = target * scaler_scale.value() + scaler_mean.value();

            if (transform == TransformType::Log1p) {
                // Clamp to avoid overflow in expm1
                pred_raw = torch::expm1(torch::clamp(pred_orig, -87.0f, 87.0f));
                target_raw = torch::expm1(torch::clamp(target_orig, -87.0f, 87.0f));
            } else {
                pred_raw = pred_orig;
                target_raw = target_orig;
            }
        } else {
            pred_raw = pred;
            target_raw = target;
        }

        return compute_loss_components(pred, target, pred_raw, target_raw, config);
    }

    /**
     * Compute classification loss (CrossEntropy).
     */
    torch::Tensor classification_loss(
        const torch::Tensor& pred,
        const torch::Tensor& target
    ) const {
        return torch::nn::functional::cross_entropy(pred, target);
    }

    int num_phases() const { return static_cast<int>(phases_.size()); }

    const std::vector<int>& phase_boundaries() const { return phase_boundaries_; }

private:
    std::unordered_map<int, PhaseConfig> phases_;
    std::vector<int> phase_boundaries_;
    float eps_;

    void validate() const {
        int num = static_cast<int>(phases_.size());
        int expected_boundaries = num - 1;

        if (static_cast<int>(phase_boundaries_.size()) != expected_boundaries) {
            throw std::invalid_argument(
                "phases has " + std::to_string(num) + " entries, so phase_boundaries must have " +
                std::to_string(expected_boundaries) + " entries, got " +
                std::to_string(phase_boundaries_.size())
            );
        }

        // Validate phases dict has consecutive keys starting at 1
        for (int i = 1; i <= num; ++i) {
            if (phases_.find(i) == phases_.end()) {
                throw std::invalid_argument(
                    "phases keys must be consecutive integers starting at 1"
                );
            }
            if (!phases_.at(i).is_valid()) {
                throw std::invalid_argument(
                    "Phase " + std::to_string(i) + " must have at least one non-zero loss weight"
                );
            }
        }
    }

    torch::Tensor compute_loss_components(
        const torch::Tensor& pred,
        const torch::Tensor& target,
        const torch::Tensor& pred_raw,
        const torch::Tensor& target_raw,
        const PhaseConfig& config
    ) const {
        // Mask for valid (non-NaN, non-Inf) values
        auto valid_mask = torch::isfinite(pred) & torch::isfinite(target);
        if (!valid_mask.any().item<bool>()) {
            // Return zero loss that maintains gradient chain
            return (pred * 0.0f).sum();
        }

        auto pred_valid = pred.masked_select(valid_mask);
        auto target_valid = target.masked_select(valid_mask);

        std::vector<torch::Tensor> components;

        // MAE (scaled space)
        if (config.mae > 0) {
            auto mae_loss = torch::nn::functional::l1_loss(pred_valid, target_valid);
            components.push_back(config.mae * mae_loss);
        }

        // MSE (scaled space)
        if (config.mse > 0) {
            auto mse_loss = torch::nn::functional::mse_loss(pred_valid, target_valid);
            components.push_back(config.mse * mse_loss);
        }

        // Huber (scaled space)
        if (config.huber > 0) {
            auto huber_loss = torch::nn::functional::huber_loss(
                pred_valid, target_valid,
                torch::nn::functional::HuberLossFuncOptions().delta(config.huber_delta)
            );
            components.push_back(config.huber * huber_loss);
        }

        // SMAPE (original scale)
        if (config.smape > 0) {
            auto pred_raw_valid = pred_raw.masked_select(valid_mask);
            auto target_raw_valid = target_raw.masked_select(valid_mask);

            auto abs_diff = torch::abs(pred_raw_valid - target_raw_valid);
            auto denominator = (torch::abs(pred_raw_valid) + torch::abs(target_raw_valid)) / 2 + eps_;
            denominator = torch::clamp(denominator, eps_ * 10);
            auto smape_loss = (abs_diff / denominator).mean();
            components.push_back(config.smape * smape_loss);
        }

        // Band penalty (original scale)
        if (config.band > 0) {
            auto pred_raw_valid = pred_raw.masked_select(valid_mask);
            auto target_raw_valid = target_raw.masked_select(valid_mask);

            auto abs_diff = torch::abs(pred_raw_valid - target_raw_valid);
            auto rel_error = abs_diff / torch::clamp(torch::abs(target_raw_valid), eps_ * 10);
            auto band_violation = torch::relu(rel_error - config.band_threshold);
            auto band_loss = band_violation.mean();
            components.push_back(config.band * band_loss);
        }

        // Sum all components
        if (!components.empty()) {
            torch::Tensor total = components[0];
            for (size_t i = 1; i < components.size(); ++i) {
                total = total + components[i];
            }
            return total;
        }

        // Fallback
        return (pred * 0.0f).sum();
    }
};


/**
 * Combines losses across multiple targets.
 *
 * Applies appropriate loss function per task type and weights by target configuration.
 */
class MultiTaskLoss {
public:
    MultiTaskLoss(
        std::vector<TargetConfig> target_configs,
        std::unordered_map<int, PhaseConfig> phases = {{1, PhaseConfig::mae_only()}},
        std::vector<int> phase_boundaries = {}
    ) : target_configs_(std::move(target_configs)),
        phased_loss_(std::move(phases), std::move(phase_boundaries))
    {
        // Check for fast path (single regression target, MAE-only, single phase)
        use_fast_path_ = (
            target_configs_.size() == 1 &&
            target_configs_[0].task == TaskType::Regression &&
            phased_loss_.num_phases() == 1 &&
            phased_loss_.get_config(0).mae == 1.0f &&
            phased_loss_.get_config(0).mse == 0.0f &&
            phased_loss_.get_config(0).huber == 0.0f &&
            phased_loss_.get_config(0).smape == 0.0f &&
            phased_loss_.get_config(0).band == 0.0f
        );
    }

    /**
     * Compute combined multi-task loss.
     *
     * @return {total_loss, {target_name: individual_loss}}
     */
    std::pair<torch::Tensor, std::unordered_map<std::string, torch::Tensor>> operator()(
        const std::unordered_map<std::string, torch::Tensor>& predictions,
        const std::unordered_map<std::string, torch::Tensor>& targets,
        int epoch,
        const std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>>& scalers = {}
    ) const {
        // Fast path: single regression target with MAE-only
        if (use_fast_path_) {
            const auto& name = target_configs_[0].name;
            auto pred = predictions.at(name);
            auto target = targets.at(name);
            auto loss = torch::nn::functional::l1_loss(pred, target);
            return {loss, {{name, loss}}};
        }

        // Standard path
        std::unordered_map<std::string, torch::Tensor> losses;
        torch::Tensor total;
        bool first = true;

        for (const auto& cfg : target_configs_) {
            auto pred_it = predictions.find(cfg.name);
            auto target_it = targets.find(cfg.name);
            if (pred_it == predictions.end() || target_it == targets.end()) {
                continue;
            }

            const auto& pred = pred_it->second;
            const auto& target = target_it->second;

            torch::Tensor loss;
            if (cfg.task == TaskType::Regression) {
                std::optional<torch::Tensor> scaler_mean, scaler_scale;
                auto scaler_it = scalers.find(cfg.name);
                if (scaler_it != scalers.end()) {
                    scaler_mean = scaler_it->second.first;
                    scaler_scale = scaler_it->second.second;
                }

                loss = phased_loss_.regression_loss(
                    pred, target, epoch, scaler_mean, scaler_scale, cfg.transform
                );
            } else {
                loss = torch::nn::functional::cross_entropy(pred, target);
            }

            losses[cfg.name] = loss;
            auto weighted = cfg.weight * loss;

            if (first) {
                total = weighted;
                first = false;
            } else {
                total = total + weighted;
            }
        }

        if (first) {
            // No valid targets - return zero
            total = torch::zeros({1}, torch::kFloat32);
        }

        return {total, losses};
    }

private:
    std::vector<TargetConfig> target_configs_;
    PhasedLoss phased_loss_;
    bool use_fast_path_;
};


/**
 * Metrics computation utilities.
 */
struct Metrics {
    /**
     * Band accuracy: fraction of predictions within threshold of target.
     */
    static float band_accuracy(
        const torch::Tensor& pred,
        const torch::Tensor& target,
        float threshold = 0.25f
    ) {
        auto valid_mask = torch::isfinite(pred) & torch::isfinite(target);
        if (!valid_mask.any().item<bool>()) {
            return 0.0f;
        }

        auto pred_valid = pred.masked_select(valid_mask);
        auto target_valid = target.masked_select(valid_mask);

        auto abs_diff = torch::abs(pred_valid - target_valid);
        auto rel_error = abs_diff / torch::clamp(torch::abs(target_valid), 1e-8f);
        auto within_band = (rel_error <= threshold).to(torch::kFloat32);

        return within_band.mean().item<float>();
    }

    /**
     * Mean Absolute Error.
     */
    static float mae(const torch::Tensor& pred, const torch::Tensor& target) {
        auto valid_mask = torch::isfinite(pred) & torch::isfinite(target);
        if (!valid_mask.any().item<bool>()) {
            return 0.0f;
        }
        auto pred_valid = pred.masked_select(valid_mask);
        auto target_valid = target.masked_select(valid_mask);
        return torch::nn::functional::l1_loss(pred_valid, target_valid).item<float>();
    }

    /**
     * Root Mean Squared Error.
     */
    static float rmse(const torch::Tensor& pred, const torch::Tensor& target) {
        auto valid_mask = torch::isfinite(pred) & torch::isfinite(target);
        if (!valid_mask.any().item<bool>()) {
            return 0.0f;
        }
        auto pred_valid = pred.masked_select(valid_mask);
        auto target_valid = target.masked_select(valid_mask);
        return std::sqrt(torch::nn::functional::mse_loss(pred_valid, target_valid).item<float>());
    }

    /**
     * Symmetric Mean Absolute Percentage Error.
     */
    static float smape(const torch::Tensor& pred, const torch::Tensor& target, float eps = 1e-8f) {
        auto valid_mask = torch::isfinite(pred) & torch::isfinite(target);
        if (!valid_mask.any().item<bool>()) {
            return 0.0f;
        }
        auto pred_valid = pred.masked_select(valid_mask);
        auto target_valid = target.masked_select(valid_mask);

        auto abs_diff = torch::abs(pred_valid - target_valid);
        auto denominator = (torch::abs(pred_valid) + torch::abs(target_valid)) / 2 + eps;
        return (abs_diff / denominator).mean().item<float>();
    }

    /**
     * Classification accuracy.
     */
    static float accuracy(const torch::Tensor& pred, const torch::Tensor& target) {
        auto pred_classes = pred.argmax(1);
        return (pred_classes == target).to(torch::kFloat32).mean().item<float>();
    }

    /**
     * Compute all metrics for regression.
     */
    static std::unordered_map<std::string, float> compute_regression(
        const torch::Tensor& pred,
        const torch::Tensor& target
    ) {
        return {
            {"mae", mae(pred, target)},
            {"rmse", rmse(pred, target)},
            {"smape", smape(pred, target)},
            {"band_10", band_accuracy(pred, target, 0.10f)},
            {"band_25", band_accuracy(pred, target, 0.25f)},
            {"band_50", band_accuracy(pred, target, 0.50f)}
        };
    }

    /**
     * Compute all metrics for classification.
     */
    static std::unordered_map<std::string, float> compute_classification(
        const torch::Tensor& pred,
        const torch::Tensor& target
    ) {
        return {
            {"accuracy", accuracy(pred, target)}
        };
    }
};

} // namespace resolve
