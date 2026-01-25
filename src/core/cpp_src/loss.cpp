#include "resolve/loss.hpp"
#include <cmath>

namespace resolve {

// PhasedLoss implementation

PhasedLoss::PhasedLoss(
    std::pair<int, int> phase_boundaries,
    float smape_weight_p2,
    float smape_weight_p3,
    float band_weight_p3,
    float band_threshold,
    float eps
) : phase_boundaries_(phase_boundaries),
    smape_weight_p2_(smape_weight_p2),
    smape_weight_p3_(smape_weight_p3),
    band_weight_p3_(band_weight_p3),
    band_threshold_(band_threshold),
    eps_(eps)
{}

PhasedLoss PhasedLoss::from_config(LossConfigMode mode, std::pair<int, int> phase_boundaries) {
    switch (mode) {
        case LossConfigMode::MAE:
            // Pure MAE: no SMAPE, no band penalty (set weights to 0)
            return PhasedLoss({9999, 9999}, 0.0f, 0.0f, 0.0f);
        case LossConfigMode::SMAPE:
            // SMAPE as primary: high SMAPE weight from start
            return PhasedLoss({0, 0}, 1.0f, 1.0f, 0.0f);
        case LossConfigMode::Combined:
        default:
            // Default phased training
            return PhasedLoss(phase_boundaries);
    }
}

int PhasedLoss::get_phase(int epoch) const {
    if (epoch < phase_boundaries_.first) return 1;
    if (epoch < phase_boundaries_.second) return 2;
    return 3;
}

torch::Tensor PhasedLoss::regression_loss(
    torch::Tensor pred,
    torch::Tensor target,
    int epoch,
    torch::Tensor scaler_mean,
    torch::Tensor scaler_scale,
    TransformType transform
) const {
    int phase = get_phase(epoch);

    // Squeeze prediction if needed
    if (pred.dim() == 2 && pred.size(1) == 1) {
        pred = pred.squeeze(1);
    }

    // MAE loss (always)
    auto mae = torch::abs(pred - target).mean();

    if (phase == 1) {
        return mae;
    }

    // For phases 2 and 3, we need original scale values
    torch::Tensor pred_orig, target_orig;

    if (scaler_mean.defined() && scaler_scale.defined()) {
        pred_orig = pred * scaler_scale + scaler_mean;
        target_orig = target * scaler_scale + scaler_mean;
    } else {
        pred_orig = pred;
        target_orig = target;
    }

    // Apply inverse transform if log1p was used
    if (transform == TransformType::Log1p) {
        pred_orig = torch::expm1(torch::clamp(pred_orig, /*min=*/-88.0f, /*max=*/88.0f));
        target_orig = torch::expm1(target_orig);
    }

    // SMAPE loss
    auto numerator = torch::abs(pred_orig - target_orig);
    auto denominator = torch::abs(pred_orig) + torch::abs(target_orig) + eps_;
    auto smape = (numerator / denominator).mean();

    if (phase == 2) {
        return mae + smape_weight_p2_ * smape;
    }

    // Phase 3: add band penalty
    auto ratio = pred_orig / (target_orig + eps_);
    auto outside_band = (ratio < (1.0f - band_threshold_)) | (ratio > (1.0f + band_threshold_));
    auto band_penalty = outside_band.to(torch::kFloat32).mean();

    return mae + smape_weight_p3_ * smape + band_weight_p3_ * band_penalty;
}

torch::Tensor PhasedLoss::classification_loss(
    torch::Tensor pred,
    torch::Tensor target
) const {
    return torch::nn::functional::cross_entropy(pred, target);
}

// MultiTaskLoss implementation

MultiTaskLoss::MultiTaskLoss(
    const std::vector<TargetConfig>& targets,
    std::pair<int, int> phase_boundaries,
    LossConfigMode loss_config
) : targets_(targets), phased_loss_(PhasedLoss::from_config(loss_config, phase_boundaries))
{}

std::pair<torch::Tensor, std::unordered_map<std::string, torch::Tensor>> MultiTaskLoss::compute(
    const std::unordered_map<std::string, torch::Tensor>& predictions,
    const std::unordered_map<std::string, torch::Tensor>& targets,
    int epoch,
    const std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>>& scalers
) const {
    std::unordered_map<std::string, torch::Tensor> losses;
    torch::Tensor total_loss = torch::zeros({1}, predictions.begin()->second.options());

    for (const auto& cfg : targets_) {
        auto pred_it = predictions.find(cfg.name);
        auto target_it = targets.find(cfg.name);

        if (pred_it == predictions.end() || target_it == targets.end()) {
            continue;
        }

        torch::Tensor loss;
        if (cfg.task == TaskType::Regression) {
            torch::Tensor scaler_mean, scaler_scale;
            auto scaler_it = scalers.find(cfg.name);
            if (scaler_it != scalers.end()) {
                scaler_mean = scaler_it->second.first;
                scaler_scale = scaler_it->second.second;
            }
            loss = phased_loss_.regression_loss(
                pred_it->second, target_it->second, epoch,
                scaler_mean, scaler_scale, cfg.transform
            );
        } else {
            loss = phased_loss_.classification_loss(pred_it->second, target_it->second);
        }

        losses[cfg.name] = loss;
        total_loss = total_loss + cfg.weight * loss;
    }

    return {total_loss, losses};
}

// Metrics implementation

float Metrics::band_accuracy(torch::Tensor pred, torch::Tensor target, float threshold) {
    auto ratio = pred / (target + 1e-8f);
    auto in_band = (ratio >= (1.0f - threshold)) & (ratio <= (1.0f + threshold));
    return in_band.to(torch::kFloat32).mean().item<float>();
}

float Metrics::mae(torch::Tensor pred, torch::Tensor target) {
    return torch::abs(pred - target).mean().item<float>();
}

float Metrics::rmse(torch::Tensor pred, torch::Tensor target) {
    return torch::sqrt(torch::pow(pred - target, 2).mean()).item<float>();
}

float Metrics::smape(torch::Tensor pred, torch::Tensor target, float eps) {
    auto numerator = torch::abs(pred - target);
    auto denominator = torch::abs(pred) + torch::abs(target) + eps;
    return (numerator / denominator).mean().item<float>();
}

float Metrics::accuracy(torch::Tensor pred, torch::Tensor target) {
    auto pred_classes = torch::argmax(pred, /*dim=*/1);
    return (pred_classes == target).to(torch::kFloat32).mean().item<float>();
}

std::unordered_map<std::string, float> Metrics::compute(
    torch::Tensor pred,
    torch::Tensor target,
    TaskType task,
    TransformType transform
) {
    std::unordered_map<std::string, float> metrics;

    if (task == TaskType::Classification) {
        metrics["accuracy"] = accuracy(pred, target);
    } else {
        // Squeeze if needed
        if (pred.dim() == 2 && pred.size(1) == 1) {
            pred = pred.squeeze(1);
        }

        // Apply inverse transform for original scale metrics
        torch::Tensor pred_orig = pred;
        torch::Tensor target_orig = target;

        if (transform == TransformType::Log1p) {
            pred_orig = torch::expm1(torch::clamp(pred, /*min=*/-88.0f, /*max=*/88.0f));
            target_orig = torch::expm1(target);
        }

        metrics["mae"] = mae(pred, target);
        metrics["rmse"] = rmse(pred, target);
        metrics["band_accuracy"] = band_accuracy(pred_orig, target_orig);
        metrics["smape"] = smape(pred_orig, target_orig);
    }

    return metrics;
}

} // namespace resolve
