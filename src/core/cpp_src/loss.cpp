#include "resolve/loss.hpp"
#include <cmath>

namespace resolve {

namespace {

// Elementwise symmetric mean absolute percentage error, standard definition:
//   |pred - target| / ((|pred| + |target|) / 2 + eps)
// The /2 in the denominator gives the conventional [0, 2] range that sklearn and
// the SMAPE literature use; omitting it halves every value (issue #95). Single
// source of truth for the reported Metrics::smape and the phased-loss SMAPE term
// so the two cannot drift apart.
inline torch::Tensor smape_terms(const torch::Tensor& pred, const torch::Tensor& target, float eps) {
    auto numerator = torch::abs(pred - target);
    auto denominator = (torch::abs(pred) + torch::abs(target)) * 0.5f + eps;
    return numerator / denominator;
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
    phased_loss_(PhasedLoss::from_config(loss_config, phase_boundaries, band_threshold, nca))
{}

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
    return smape_terms(pred, target, eps).mean().item<float>();
}

float Metrics::r_squared(torch::Tensor pred, torch::Tensor target) {
    auto ss_res = torch::pow(target - pred, 2).sum();
    auto ss_tot = torch::pow(target - target.mean(), 2).sum();
    // Constant target: R^2 is undefined (no variance to explain). Report a
    // perfect fit only when the predictions also match the constant; otherwise
    // there is no explained variance, so report 0 rather than a spurious 1.0.
    // (Matches fit()'s baseline-R^2 path, which leaves R^2 at its default when
    // ss_tot ~ 0 instead of asserting a perfect fit.)
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
        // labels=range(num_classes)). Excluding zero-support classes would read
        // systematically higher and diverge from the reported convention
        // (issue #75). Weighted-F1 weights by support, so an absent class adds
        // zero and needs no special case.
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

        // Map back to original units so every reported metric is in the same,
        // interpretable space. Training stores regression targets standardized
        // ((x - mean) / scale) and, for Log1p targets, log1p-transformed first;
        // the model therefore predicts in standardized(-log) space. Undo the
        // standardization, then the transform, matching PhasedLoss. Without the
        // scalers, mae/rmse were reported in standardized units and smape/band
        // took ratios of values straddling zero -- both meaningless.
        torch::Tensor pred_orig = pred;
        torch::Tensor target_orig = target;

        if (scaler_mean.defined() && scaler_scale.defined()) {
            pred_orig = pred_orig * scaler_scale + scaler_mean;
            target_orig = target_orig * scaler_scale + scaler_mean;
        }

        if (transform == TransformType::Log1p) {
            pred_orig = torch::expm1(torch::clamp(pred_orig, /*min=*/-88.0f, /*max=*/88.0f));
            target_orig = torch::expm1(target_orig);
        }

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
// NCA Loss implementation
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
    // batch contribute 0 and are excluded from the denominator (issue #90);
    // dividing by batch_size instead diluted the loss/gradient scale by the
    // fraction of samples lacking an in-batch same-class neighbour, making the
    // magnitude depend on class rarity within the batch.
    auto has_neighbor = same_class.sum(1) > 0;
    auto nca_loss = -log_prob_correct * has_neighbor.to(torch::kFloat32);

    auto n_contrib = has_neighbor.to(torch::kFloat32).sum().clamp_min(1.0f);
    return nca_loss.sum() / n_contrib;
}

NCALossImpl::NCALossImpl(
    int64_t latent_dim,
    int64_t n_classes,
    float temperature,
    int n_neighbors
) : n_classes_(n_classes),
    temperature_(temperature),
    n_neighbors_(n_neighbors)
{
    // Initialize reference set as empty
    ref_embeddings_ = register_buffer("ref_embeddings",
        torch::empty({0, latent_dim}));
    ref_labels_ = register_buffer("ref_labels",
        torch::empty({0}, torch::kLong));
}

torch::Tensor NCALossImpl::forward(
    torch::Tensor latent,
    torch::Tensor targets
) {
    return nca_objective(std::move(latent), std::move(targets),
                         temperature_, n_neighbors_);
}

torch::Tensor NCALossImpl::predict(torch::Tensor latent) {
    if (ref_embeddings_.size(0) == 0) {
        throw std::runtime_error(
            "NCALoss::predict requires reference set. Call update_references() first.");
    }

    // Normalize
    auto latent_norm = torch::nn::functional::normalize(latent,
        torch::nn::functional::NormalizeFuncOptions().dim(1));
    auto ref_norm = torch::nn::functional::normalize(ref_embeddings_,
        torch::nn::functional::NormalizeFuncOptions().dim(1));

    // Similarity to reference set: (n_query, n_ref)
    auto sim = torch::mm(latent_norm, ref_norm.t()) / temperature_;
    auto probs = torch::softmax(sim, /*dim=*/1);  // (n_query, n_ref)

    // Aggregate probabilities by class -- the paper's eq. (2) p_i, evaluated
    // against the stored reference set instead of the training batch, which is
    // the stochastic-neighbour classification rule NCA is trained for.
    auto class_probs = torch::zeros({latent.size(0), n_classes_}, latent.options());
    for (int64_t c = 0; c < n_classes_; ++c) {
        auto class_mask = (ref_labels_ == c).to(torch::kFloat32);  // (n_ref,)
        class_probs.select(1, c) = torch::mv(probs, class_mask);
    }

    return class_probs;  // (n_query, n_classes)
}

void NCALossImpl::update_references(torch::Tensor latent, torch::Tensor targets) {
    torch::NoGradGuard no_grad;
    ref_embeddings_ = latent.detach().clone();
    ref_labels_ = targets.detach().clone();
}

} // namespace resolve
