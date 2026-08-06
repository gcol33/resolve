#pragma once

#include "resolve/types.hpp"
#include <torch/torch.h>

namespace resolve {

// Neighbourhood Components Analysis objective (Goldberger, Roweis, Hinton &
// Salakhutdinov, "Neighbourhood Components Analysis", NIPS 2004).
//
// Each sample's stochastic neighbour distribution is the paper's eq. (1),
//     p_ij = exp(-||A x_i - A x_j||^2) / sum_{k != i} exp(-||A x_i - A x_k||^2),
//     p_ii = 0,
// and the returned value is the negated eq. (6) log objective,
//     g(A) = sum_i log( sum_{j in C_i} p_ij ),   C_i = { j : c_i = c_j },
// averaged over the samples that have at least one same-class neighbour, so it
// is minimized rather than maximized and its scale does not depend on how many
// samples happen to lack an in-batch partner.
//
// `embeddings` (batch, dim) plays the role of A x. The single matrix A of the
// paper is here the network that produced `embeddings`, followed by L2
// normalization and the temperature; that composition reproduces eq. (1)
// exactly rather than approximating it. For unit-norm u, v,
// ||u - v||^2 = 2 - 2 u.v, and the constant cancels in the softmax over j, so
//     softmax_j( u_i . u_j / temperature ) = softmax_j( -||A x_i - A x_j||^2 )
// with A = (2 * temperature)^(-1/2) * (normalize o network). Section 2 of the
// paper notes that the overall scale of A sets the effective number of
// neighbours, which is the quantity `temperature` tunes.
//
// Both truncations are the ones the paper's section 2 sanctions ("the sums that
// appear in equations (5) and (7) over the data points and over the neighbours
// of each point can be truncated"): the sum over data points is the mini-batch,
// and each sample's neighbour set is its `n_neighbors` most similar in-batch
// samples. n_neighbors <= 0, or >= batch - 1, keeps the full in-batch set.
//
// `targets` is (batch,) int64 class labels.
[[nodiscard]] torch::Tensor nca_objective(
    torch::Tensor embeddings,
    torch::Tensor targets,
    float temperature = kNCATemperature,
    int n_neighbors = kNCANeighbors
);

// The NCA term PhasedLoss adds to its classification loss. Off by default;
// LossConfigMode::NCA turns it on (see PhasedLoss::from_config). The three
// hyperparameters default to the shared kNCA* constants in types.hpp and are
// user-tunable through TrainConfig's nca_* fields.
struct NCATerm {
    bool enabled = false;
    float weight = kNCAWeight;
    float temperature = kNCATemperature;
    int n_neighbors = kNCANeighbors;
};

// The NCA hyperparameters a TrainConfig carries, as the term PhasedLoss takes.
// `enabled` is left off here: the mode alone decides whether the term acts, so
// PhasedLoss::from_config turns it on for LossConfigMode::NCA and leaves it off
// for every other preset while the knobs travel with the config either way.
[[nodiscard]] inline NCATerm nca_term_of(const TrainConfig& config) {
    NCATerm term;
    term.weight = config.nca_weight;
    term.temperature = config.nca_temperature;
    term.n_neighbors = config.nca_neighbors;
    return term;
}

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
        float eps = 1e-8f,
        NCATerm nca = {}
    );

    // Factory method to create loss from config mode. band_threshold sets the
    // phase-3 band-penalty tolerance (issue #99); the MAE/SMAPE modes zero the
    // band weight so the threshold is inert there. LossConfigMode::NCA takes the
    // Combined regression schedule and enables the NCA classification term with
    // `nca`'s hyperparameters; the other modes carry `nca` disabled.
    static PhasedLoss from_config(LossConfigMode mode, std::pair<int, int> phase_boundaries = {100, 300},
                                  float band_threshold = 0.25f, NCATerm nca = {});

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

    // Compute classification loss (with optional class weights for imbalanced
    // data). When the NCA term is enabled, the NCA objective over `pred` is
    // added to the cross-entropy; see the impl for why both terms are present.
    torch::Tensor classification_loss(
        torch::Tensor pred,
        torch::Tensor target,
        torch::Tensor class_weights = {}
    ) const;

    // Whether the NCA neighbourhood term is part of classification_loss.
    [[nodiscard]] bool uses_nca() const noexcept { return nca_.enabled; }

private:
    std::pair<int, int> phase_boundaries_;
    float smape_weight_p2_;
    float smape_weight_p3_;
    float band_weight_p3_;
    float band_threshold_;
    float eps_;
    NCATerm nca_;
};

// Multi-task loss combiner
class MultiTaskLoss {
public:
    MultiTaskLoss(
        const std::vector<TargetConfig>& targets,
        std::pair<int, int> phase_boundaries = {100, 300},
        LossConfigMode loss_config = LossConfigMode::Combined,
        float band_threshold = 0.25f,
        NCATerm nca = {}
    );

    // Compute combined loss
    // Returns (total_loss, individual_losses)
    std::pair<torch::Tensor, std::unordered_map<std::string, torch::Tensor>> compute(
        const std::unordered_map<std::string, torch::Tensor>& predictions,
        const std::unordered_map<std::string, torch::Tensor>& targets,
        int epoch,
        const std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>>& scalers = {}
    ) const;

    // Phase (1/2/3) the regression loss is in at `epoch`, using the effective
    // phase boundaries this loss was built with (which from_config remaps for
    // MAE/SMAPE modes). Single source of truth for phase-aware training logic
    // in the Trainer (best-model selection / early-stopping gating).
    int phase_for(int epoch) const { return phased_loss_.get_phase(epoch); }

    // Whether the NCA neighbourhood term is active on classification targets
    // (i.e. this loss was built with LossConfigMode::NCA).
    [[nodiscard]] bool uses_nca() const noexcept { return phased_loss_.uses_nca(); }

private:
    std::vector<TargetConfig> targets_;
    PhasedLoss phased_loss_;
};

// Module wrapper around nca_objective that also carries the reference set the
// paper's stochastic-neighbour classification rule needs at inference: class
// probabilities are the neighbour probability mass per class (eq. 2) taken
// against stored reference embeddings rather than the current batch.
//
// The module owns no learnable parameters -- the transformation NCA learns is
// whatever network produces `latent` -- so `forward` is exactly nca_objective.
class NCALossImpl : public torch::nn::Module {
public:
    NCALossImpl(
        int64_t latent_dim,
        int64_t n_classes,
        float temperature = kNCATemperature,
        int n_neighbors = kNCANeighbors  // truncated neighbour set per sample
    );

    // Compute NCA loss from latent representations and targets
    // latent: (batch, latent_dim) - encoder output
    // targets: (batch,) - class labels
    [[nodiscard]] torch::Tensor forward(
        torch::Tensor latent,
        torch::Tensor targets
    );

    // Get predicted class probabilities via NCA (for inference)
    [[nodiscard]] torch::Tensor predict(torch::Tensor latent);

    // Update reference set from training batch
    void update_references(torch::Tensor latent, torch::Tensor targets);

private:
    int64_t n_classes_;
    float temperature_;
    int n_neighbors_;

    // Reference embeddings and labels (maintained from training batches)
    torch::Tensor ref_embeddings_;  // (n_ref, latent_dim)
    torch::Tensor ref_labels_;      // (n_ref,)
};

TORCH_MODULE(NCALoss);

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
    // Symmetric MAPE, standard definition |p-t| / ((|p|+|t|)/2 + eps), range
    // [0, 2] (multiply by 100 for the [0, 200%] convention). Matches sklearn.
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
    // scaler_mean/scaler_scale: when defined, regression pred/target are mapped
    //   back to original units via x*scale + mean before any transform inversion,
    //   so every reported regression metric is in original units. Mirrors the
    //   unscale convention in PhasedLoss::regression_loss. Omit (undefined) to
    //   report in whatever space pred/target are already in.
    static std::unordered_map<std::string, float> compute(
        torch::Tensor pred,
        torch::Tensor target,
        TaskType task,
        TransformType transform = TransformType::None,
        const std::vector<float>& band_thresholds = {0.25f, 0.50f, 0.75f},
        int num_classes = 0,
        torch::Tensor scaler_mean = {},
        torch::Tensor scaler_scale = {}
    );
};

} // namespace resolve
