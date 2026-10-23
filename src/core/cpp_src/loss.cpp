#include "resolve/loss.hpp"
#include "resolve/target_labels.hpp"
#include "resolve/target_transform.hpp"
#include <algorithm>
#include <cmath>

namespace resolve {

namespace {

// Elementwise symmetric mean absolute percentage error, standard definition:
//   |pred - target| / ((|pred| + |target|) / 2 + eps)
// The /2 in the denominator gives the conventional [0, 2] range that sklearn and
// the SMAPE literature use. Shared by the reported Metrics::smape and the
// phased-loss SMAPE term.
inline torch::Tensor smape_terms(const torch::Tensor& pred, const torch::Tensor& target, float eps) {
    auto numerator = torch::abs(pred - target);
    auto denominator = (torch::abs(pred) + torch::abs(target)) * 0.5f + eps;
    return numerator / denominator;
}

// The term target `cfg` adds to the multi-task total for its loss `loss`.
// Shared by MultiTaskLoss::compute, which sums it, and
// MultiTaskLoss::contributions, which reports it, so the reported terms are
// the summed ones bit for bit.
inline torch::Tensor weighted_term(const TargetConfig& cfg, const torch::Tensor& loss) {
    return cfg.weight * loss;
}

}  // namespace

// PhasedLoss implementation

PhasedLoss::PhasedLoss(
    std::pair<int, int> phase_boundaries,
    float smape_weight_p2,
    float smape_weight_p3,
    float band_weight_p3,
    float band_threshold,
    float eps,
    NCATerm nca
) : phase_boundaries_(phase_boundaries),
    smape_weight_p2_(smape_weight_p2),
    smape_weight_p3_(smape_weight_p3),
    band_weight_p3_(band_weight_p3),
    band_threshold_(band_threshold),
    eps_(eps),
    nca_(nca)
{}

PhasedLoss PhasedLoss::from_config(LossConfigMode mode, std::pair<int, int> phase_boundaries,
                                   float band_threshold, NCATerm nca) {
    switch (mode) {
        case LossConfigMode::MAE:
            // Pure MAE: no SMAPE, no band penalty (set weights to 0)
            return PhasedLoss({9999, 9999}, 0.0f, 0.0f, 0.0f, band_threshold);
        case LossConfigMode::SMAPE:
            // SMAPE as primary: high SMAPE weight from start
            return PhasedLoss({0, 0}, 1.0f, 1.0f, 0.0f, band_threshold);
        case LossConfigMode::NCA: {
            // The NCA objective is defined over class labels, so it applies to
            // classification targets only; regression targets keep the default
            // phased schedule (taken from the Combined case rather than
            // restated, so the two cannot drift). The caller's hyperparameters
            // are adopted wholesale and only `enabled` is decided here, which is
            // what keeps the mode the single switch for the term.
            PhasedLoss loss = from_config(LossConfigMode::Combined, phase_boundaries,
                                          band_threshold);
            loss.nca_ = nca;
            loss.nca_.enabled = true;
            return loss;
        }
        case LossConfigMode::Combined:
        default:
            // Default phased training
            return PhasedLoss(phase_boundaries, 0.2f, 0.15f, 0.05f, band_threshold);
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

    // Phases 2 and 3 score the values in original units.
    const auto pred_orig = to_original_units(pred, transform, scaler_mean, scaler_scale);
    const auto target_orig = to_original_units(target, transform, scaler_mean, scaler_scale);

    // SMAPE loss (standard symmetric form, shared with Metrics::smape)
    auto smape = smape_terms(pred_orig, target_orig, eps_).mean();

    if (phase == 2) {
        return mae + smape_weight_p2_ * smape;
    }

    // Phase 3: differentiable band hinge. Penalize predictions whose ratio to
    // the target falls outside [1 - thr, 1 + thr], proportional to how far
    // outside they are. A hard indicator (ratio-outside-band cast to float) has
    // zero gradient, so it shifts the reported loss without ever steering the
    // optimizer toward the band the model is later scored on.
    auto ratio = pred_orig / (target_orig + eps_);
    auto band_penalty = torch::relu(torch::abs(ratio - 1.0f) - band_threshold_).mean();

    return mae + smape_weight_p3_ * smape + band_weight_p3_ * band_penalty;
}

torch::Tensor PhasedLoss::classification_loss(
    torch::Tensor pred,
    torch::Tensor target,
    torch::Tensor class_weights
) const {
    torch::Tensor loss;
    if (class_weights.defined() && class_weights.numel() > 0) {
        loss = torch::nn::functional::cross_entropy(pred, target,
            torch::nn::functional::CrossEntropyFuncOptions().weight(class_weights));
    } else {
        loss = torch::nn::functional::cross_entropy(pred, target);
    }

    if (!nca_.enabled) {
        return loss;
    }

    // The NCA term acts on the head's own output space: it pulls same-class
    // samples together and pushes different-class ones apart on the unit
    // sphere, a pressure cross-entropy does not apply (cross-entropy is driven
    // by logit magnitude along one coordinate, and is blind to how the rest of
    // the output vector is arranged).
    //
    // Cross-entropy stays in the sum because the NCA objective alone is
    // invariant under a permutation of the output coordinates: it constrains
    // only the neighbourhood structure, never which coordinate belongs to which
    // class. A head trained on the NCA term alone therefore carries no
    // coordinate -> class mapping, and argmax over it -- the rule every
    // prediction, accuracy and calibration path in the engine uses -- would be
    // meaningless. Cross-entropy fixes that mapping; NCA shapes the space.
    return loss + nca_.weight * nca_objective(pred, target,
                                              nca_.temperature, nca_.n_neighbors);
}

// MultiTaskLoss implementation

MultiTaskLoss::MultiTaskLoss(
    const std::vector<TargetConfig>& targets,
    std::pair<int, int> phase_boundaries,
    LossConfigMode loss_config,
    float band_threshold,
    NCATerm nca
) : targets_(targets),
    phased_loss_(PhasedLoss::from_config(loss_config, phase_boundaries, band_threshold, nca)),
    class_weights_(targets.size())
{
    for (size_t i = 0; i < targets_.size(); ++i) {
        const auto& weights = targets_[i].class_weights;
        if (weights.empty()) continue;
        class_weights_[i] = torch::tensor(weights, torch::kFloat32);
    }
}

const torch::Tensor& MultiTaskLoss::class_weights_on(size_t i, torch::Device device) const {
    auto& weights = class_weights_[i];
    if (weights.defined() && weights.device() != device) {
        weights = weights.to(device);
    }
    return weights;
}

bool MultiTaskLoss::objective_settled(int epoch, int last_epoch) const {
    const bool has_regression = std::any_of(targets_.begin(), targets_.end(),
        [](const TargetConfig& t) { return t.task == TaskType::Regression; });
    if (!has_regression || !phased_loss_.varies_by_phase()) {
        return true;
    }
    return phased_loss_.get_phase(epoch) == phased_loss_.get_phase(last_epoch);
}

std::pair<torch::Tensor, std::unordered_map<std::string, torch::Tensor>> MultiTaskLoss::compute(
    const std::unordered_map<std::string, torch::Tensor>& predictions,
    const std::unordered_map<std::string, torch::Tensor>& targets,
    int epoch,
    const std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>>& scalers
) const {
    std::unordered_map<std::string, torch::Tensor> losses;

    // Seed the accumulator deterministically. `predictions` is an unordered_map,
    // so predictions.begin() picks an arbitrary head's device/dtype and seeds a
    // shape-{1} tensor; instead derive the device from the first target in the
    // (ordered) targets_ list that has a prediction, and seed a true scalar.
    torch::Device acc_device = torch::kCPU;
    for (const auto& cfg : targets_) {
        auto it = predictions.find(cfg.name);
        if (it != predictions.end()) { acc_device = it->second.device(); break; }
    }
    torch::Tensor total_loss = torch::zeros(
        {}, torch::TensorOptions().dtype(torch::kFloat32).device(acc_device));

    for (size_t i = 0; i < targets_.size(); ++i) {
        const auto& cfg = targets_[i];
        auto pred_it = predictions.find(cfg.name);
        auto target_it = targets.find(cfg.name);

        if (pred_it == predictions.end() || target_it == targets.end()) {
            continue;
        }

        // A masked target is scored on the rows of this batch that carry its
        // label: every term below is then the same mean as for a Drop target,
        // taken over those rows (cross-entropy weighted by class weight, as
        // torch normalises it; NCA over the labelled rows' neighbourhoods).
        // With no labelled row the target adds an exact zero and no gradient,
        // where the mean over an empty set would be NaN. The batch's other
        // targets, their weights and the phase schedule are untouched.
        const auto rows = labelled_rows(cfg, target_it->second);
        if (rows.n_labelled == 0) {
            losses[cfg.name] = torch::zeros({}, total_loss.options());
            continue;
        }
        const torch::Tensor pred = rows.select(pred_it->second);
        const torch::Tensor target = rows.select(target_it->second);

        torch::Tensor loss;
        if (cfg.task == TaskType::Regression) {
            torch::Tensor scaler_mean, scaler_scale;
            auto scaler_it = scalers.find(cfg.name);
            if (scaler_it != scalers.end()) {
                scaler_mean = scaler_it->second.first;
                scaler_scale = scaler_it->second.second;
            }
            loss = phased_loss_.regression_loss(
                pred, target, epoch,
                scaler_mean, scaler_scale, cfg.transform
            );
        } else {
            loss = phased_loss_.classification_loss(pred, target,
                                                    class_weights_on(i, pred.device()));
        }

        losses[cfg.name] = loss;
        total_loss = total_loss + weighted_term(cfg, loss);
    }

    return {total_loss, losses};
}

torch::Tensor MultiTaskLoss::contributions(
    const std::unordered_map<std::string, torch::Tensor>& losses) const {
    torch::NoGradGuard no_grad;

    torch::Device device = torch::kCPU;
    for (const auto& cfg : targets_) {
        auto it = losses.find(cfg.name);
        if (it != losses.end()) { device = it->second.device(); break; }
    }
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    std::vector<torch::Tensor> terms;
    terms.reserve(targets_.size());
    for (const auto& cfg : targets_) {
        auto it = losses.find(cfg.name);
        terms.push_back(it == losses.end()
            ? torch::zeros({}, options)
            : weighted_term(cfg, it->second.detach()).to(torch::kFloat32));
    }
    if (terms.empty()) {
        return torch::zeros({0}, options);
    }
    return torch::stack(terms);
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

float Metrics::mse(torch::Tensor pred, torch::Tensor target) {
    return torch::pow(pred - target, 2).mean().item<float>();
}

float Metrics::rmse(torch::Tensor pred, torch::Tensor target) {
    return torch::sqrt(torch::pow(pred - target, 2).mean()).item<float>();
}

float Metrics::smape(torch::Tensor pred, torch::Tensor target, float eps) {
    return smape_terms(pred, target, eps).mean().item<float>();
}

float Metrics::r_squared(torch::Tensor pred, torch::Tensor target) {
    auto ss_res = torch::pow(target - pred, 2).sum();
    auto ss_tot = torch::pow(target - target.mean(), 2).sum();
    // Constant target: R^2 is undefined (no variance to explain). Report a
    // perfect fit only when the predictions also match the constant; otherwise
    // there is no explained variance, so report 0 rather than a spurious 1.0.
    if (ss_tot.item<float>() < 1e-10f) {
        return (ss_res.item<float>() < 1e-10f) ? 1.0f : 0.0f;
    }
    return 1.0f - (ss_res / ss_tot).item<float>();
}

float Metrics::accuracy(torch::Tensor pred, torch::Tensor target) {
    auto pred_classes = torch::argmax(pred, /*dim=*/1);
    return (pred_classes == target).to(torch::kFloat32).mean().item<float>();
}

torch::Tensor Metrics::confusion_matrix(torch::Tensor pred, torch::Tensor target, int num_classes) {
    // Cell (true, predicted) counts the rows with that pair; a row whose code
    // lies outside [0, num_classes) is not counted.
    const auto pred_classes = torch::argmax(pred, /*dim=*/1).to(torch::kLong);
    const auto true_classes = target.to(pred_classes.device()).to(torch::kLong).reshape({-1});
    const auto in_range = (true_classes >= 0) & (true_classes < num_classes) &
                          (pred_classes >= 0) & (pred_classes < num_classes);
    const auto cell = (true_classes * num_classes + pred_classes).masked_select(in_range);
    const int64_t n_cells = static_cast<int64_t>(num_classes) * num_classes;
    return torch::bincount(cell, /*weights=*/{}, n_cells)
        .reshape({num_classes, num_classes})
        .to(torch::kCPU);
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
    float macro_f1_sum = 0.0f;
    float weighted_f1_sum = 0.0f;

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

        // Macro-F1 is the mean of per-class F1 scores, not the F1 of the
        // macro-averaged precision and recall (those differ whenever the
        // per-class P/R are imbalanced). Average per_class_f1 directly over ALL
        // labels: a class with no support in this fold contributes f1=0 (its
        // recall is 0), matching sklearn f1_score(average='macro',
        // labels=range(num_classes)). Weighted-F1 weights by support, so an
        // absent class adds zero and needs no special case.
        macro_f1_sum += f1;
        weighted_f1_sum += f1 * support;
    }

    result.macro_f1 = (num_classes > 0) ? macro_f1_sum / num_classes : 0.0f;

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
    int num_classes,
    torch::Tensor scaler_mean,
    torch::Tensor scaler_scale
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

        // Every reported metric is in original units: the model predicts in
        // standardised(-log1p) space, and SMAPE / band take ratios that are
        // meaningless on values standardised around zero.
        const auto pred_orig = to_original_units(pred, transform, scaler_mean, scaler_scale);
        const auto target_orig = to_original_units(target, transform, scaler_mean, scaler_scale);

        metrics["mae"] = mae(pred_orig, target_orig);
        metrics["rmse"] = rmse(pred_orig, target_orig);
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

// =============================================================================
// NCA objective
// =============================================================================

torch::Tensor nca_objective(
    torch::Tensor embeddings,
    torch::Tensor targets,
    float temperature,
    int n_neighbors
) {
    const int64_t batch_size = embeddings.size(0);

    // Unit-norm rows, so the scaled dot product below is the paper's softmax
    // over Euclidean distances in the transformed space (see the header: for
    // unit vectors the two differ by a constant that cancels in the softmax).
    auto emb_norm = torch::nn::functional::normalize(embeddings,
        torch::nn::functional::NormalizeFuncOptions().dim(1));

    // Similarity matrix: (batch, batch)
    auto sim = torch::mm(emb_norm, emb_norm.t()) / temperature;

    // p_ii = 0 (paper eq. 1): exclude self from both the numerator and the
    // normalizing sum by pushing the diagonal below the softmax floor.
    auto mask = torch::eye(batch_size, sim.options()).to(torch::kBool);
    sim.masked_fill_(mask, -1e9f);

    // Truncated neighbour set: restrict each sample to its n_neighbors most
    // similar (non-self) samples and mask the rest, so n_neighbors bounds the
    // neighbourhood. n_neighbors <= 0 or >= batch_size - 1 keeps the full set.
    if (n_neighbors > 0 && n_neighbors < batch_size - 1) {
        const int64_t k = n_neighbors;
        auto topk_idx = std::get<1>(sim.topk(k, /*dim=*/1, /*largest=*/true));  // (batch, k)
        auto keep = torch::zeros({batch_size, batch_size},
                                 sim.options().dtype(torch::kBool));
        keep.scatter_(1, topk_idx,
                      torch::ones({batch_size, k}, keep.options()));
        sim.masked_fill_(keep.logical_not(), -1e9f);
    }

    // p_ij for every pair, in log space (paper eq. 1).
    auto log_probs = torch::log_softmax(sim, /*dim=*/1);  // (batch, batch)

    // C_i = { j : c_i = c_j }, self excluded.
    auto targets_row = targets.unsqueeze(1);  // (batch, 1)
    auto targets_col = targets.unsqueeze(0);  // (1, batch)
    auto same_class = (targets_row == targets_col).to(torch::kFloat32);  // (batch, batch)
    same_class.masked_fill_(mask, 0.0f);

    // log p_i = log sum_{j in C_i} p_ij (paper eq. 2), via logsumexp for
    // numerical stability. The 1e-10 floor keeps log() finite on the
    // out-of-class entries, which the exponential then drives to ~0.
    auto masked_log_probs = log_probs + torch::log(same_class + 1e-10f);
    auto log_prob_correct = torch::logsumexp(masked_log_probs, /*dim=*/1);  // (batch,)

    // Negated eq. (6) objective. Samples with no same-class neighbour in the
    // batch contribute 0 and are excluded from the denominator, so the scale
    // does not depend on how many samples lack an in-batch partner.
    auto has_neighbor = same_class.sum(1) > 0;
    auto nca_loss = -log_prob_correct * has_neighbor.to(torch::kFloat32);

    auto n_contrib = has_neighbor.to(torch::kFloat32).sum().clamp_min(1.0f);
    return nca_loss.sum() / n_contrib;
}

} // namespace resolve
