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
    torch::Tensor target,
    torch::Tensor class_weights
) const {
    if (class_weights.defined() && class_weights.numel() > 0) {
        return torch::nn::functional::cross_entropy(pred, target,
            torch::nn::functional::CrossEntropyFuncOptions().weight(class_weights));
    }
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
            // Convert class_weights vector to tensor if provided
            torch::Tensor class_weights_tensor;
            if (!cfg.class_weights.empty()) {
                class_weights_tensor = torch::from_blob(
                    const_cast<float*>(cfg.class_weights.data()),
                    {static_cast<int64_t>(cfg.class_weights.size())},
                    torch::kFloat32
                ).clone().to(pred_it->second.device());
            }
            loss = phased_loss_.classification_loss(pred_it->second, target_it->second, class_weights_tensor);
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

float Metrics::r_squared(torch::Tensor pred, torch::Tensor target) {
    auto ss_res = torch::pow(target - pred, 2).sum();
    auto ss_tot = torch::pow(target - target.mean(), 2).sum();
    // Handle edge case where ss_tot is zero (constant target)
    if (ss_tot.item<float>() < 1e-10f) {
        return 1.0f;  // Perfect fit if target is constant and pred matches
    }
    return 1.0f - (ss_res / ss_tot).item<float>();
}

float Metrics::accuracy(torch::Tensor pred, torch::Tensor target) {
    auto pred_classes = torch::argmax(pred, /*dim=*/1);
    return (pred_classes == target).to(torch::kFloat32).mean().item<float>();
}

torch::Tensor Metrics::confusion_matrix(torch::Tensor pred, torch::Tensor target, int num_classes) {
    auto pred_classes = torch::argmax(pred, /*dim=*/1);
    auto cm = torch::zeros({num_classes, num_classes}, torch::kInt64);

    auto pred_cpu = pred_classes.to(torch::kCPU);
    auto target_cpu = target.to(torch::kCPU);

    auto pred_accessor = pred_cpu.accessor<int64_t, 1>();
    auto target_accessor = target_cpu.accessor<int64_t, 1>();

    for (int64_t i = 0; i < pred_cpu.size(0); ++i) {
        int64_t true_class = target_accessor[i];
        int64_t pred_class = pred_accessor[i];
        if (true_class >= 0 && true_class < num_classes &&
            pred_class >= 0 && pred_class < num_classes) {
            cm[true_class][pred_class] += 1;
        }
    }

    return cm;
}

ClassificationMetrics Metrics::classification_metrics(torch::Tensor pred, torch::Tensor target, int num_classes) {
    ClassificationMetrics result;

    result.accuracy = accuracy(pred, target);
    result.confusion_matrix = confusion_matrix(pred, target, num_classes);

    result.per_class_precision.resize(num_classes, 0.0f);
    result.per_class_recall.resize(num_classes, 0.0f);
    result.per_class_f1.resize(num_classes, 0.0f);
    result.per_class_support.resize(num_classes, 0);

    auto cm = result.confusion_matrix.to(torch::kCPU);
    auto cm_accessor = cm.accessor<int64_t, 2>();

    int64_t total_samples = 0;
    float macro_precision_sum = 0.0f;
    float macro_recall_sum = 0.0f;
    float weighted_f1_sum = 0.0f;
    int valid_classes = 0;

    for (int c = 0; c < num_classes; ++c) {
        int64_t tp = cm_accessor[c][c];

        int64_t fp = 0;
        for (int i = 0; i < num_classes; ++i) {
            fp += cm_accessor[i][c];
        }
        fp -= tp;

        int64_t fn = 0;
        for (int j = 0; j < num_classes; ++j) {
            fn += cm_accessor[c][j];
        }
        fn -= tp;

        int64_t support = tp + fn;
        result.per_class_support[c] = support;
        total_samples += support;

        float precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
        result.per_class_precision[c] = precision;

        float recall = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
        result.per_class_recall[c] = recall;

        float f1 = (precision + recall > 0) ? 2.0f * precision * recall / (precision + recall) : 0.0f;
        result.per_class_f1[c] = f1;

        if (support > 0) {
            macro_precision_sum += precision;
            macro_recall_sum += recall;
            weighted_f1_sum += f1 * support;
            valid_classes++;
        }
    }

    if (valid_classes > 0) {
        float macro_precision = macro_precision_sum / valid_classes;
        float macro_recall = macro_recall_sum / valid_classes;
        result.macro_f1 = (macro_precision + macro_recall > 0) ?
            2.0f * macro_precision * macro_recall / (macro_precision + macro_recall) : 0.0f;
    } else {
        result.macro_f1 = 0.0f;
    }

    result.weighted_f1 = (total_samples > 0) ? weighted_f1_sum / total_samples : 0.0f;

    return result;
}

ConfidenceMetrics Metrics::accuracy_at_threshold(
    torch::Tensor pred,
    torch::Tensor target,
    torch::Tensor confidence,
    float threshold
) {
    ConfidenceMetrics result;
    result.n_total = pred.size(0);

    // Create mask for samples above threshold
    auto mask = confidence >= threshold;
    result.n_samples = mask.sum().item<int64_t>();
    result.coverage = static_cast<float>(result.n_samples) / result.n_total;

    if (result.n_samples == 0) {
        result.accuracy = 0.0f;
        return result;
    }

    // Get predicted classes
    auto pred_classes = torch::argmax(pred, /*dim=*/1);

    // Filter by mask and compute accuracy
    auto filtered_pred = pred_classes.index({mask});
    auto filtered_target = target.index({mask});

    result.accuracy = (filtered_pred == filtered_target).to(torch::kFloat32).mean().item<float>();

    return result;
}

std::vector<ConfidenceMetrics> Metrics::accuracy_coverage_curve(
    torch::Tensor pred,
    torch::Tensor target,
    torch::Tensor confidence,
    const std::vector<float>& thresholds
) {
    std::vector<ConfidenceMetrics> results;
    results.reserve(thresholds.size());

    for (float threshold : thresholds) {
        results.push_back(accuracy_at_threshold(pred, target, confidence, threshold));
    }

    return results;
}

std::unordered_map<std::string, float> Metrics::compute(
    torch::Tensor pred,
    torch::Tensor target,
    TaskType task,
    TransformType transform,
    const std::vector<float>& band_thresholds,
    int num_classes
) {
    std::unordered_map<std::string, float> metrics;

    if (task == TaskType::Classification) {
        metrics["accuracy"] = accuracy(pred, target);

        if (num_classes > 0) {
            auto clf_metrics = classification_metrics(pred, target, num_classes);

            metrics["macro_f1"] = clf_metrics.macro_f1;
            metrics["weighted_f1"] = clf_metrics.weighted_f1;

            for (int c = 0; c < num_classes; ++c) {
                metrics["precision_" + std::to_string(c)] = clf_metrics.per_class_precision[c];
                metrics["recall_" + std::to_string(c)] = clf_metrics.per_class_recall[c];
                metrics["f1_" + std::to_string(c)] = clf_metrics.per_class_f1[c];
            }
        }
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
        metrics["r2"] = r_squared(pred_orig, target_orig);
        metrics["smape"] = smape(pred_orig, target_orig);

        for (float threshold : band_thresholds) {
            int pct = static_cast<int>(threshold * 100 + 0.5f);
            std::string key = "band_" + std::to_string(pct);
            metrics[key] = band_accuracy(pred_orig, target_orig, threshold);
        }
    }

    return metrics;
}

} // namespace resolve
