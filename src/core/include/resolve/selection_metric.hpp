#pragma once

// The held-out value a run selects by (TrainConfig::selection_metric): the
// value early stopping, best_epoch (the returned weights and best.pt) and a
// ReduceOnPlateau schedule follow, and the direction it improves in. Loss is
// the phase-invariant selection loss Trainer::eval_epoch computes, minimized;
// Accuracy is the held-out accuracy of the schema's one classification target,
// as Metrics::compute reports it in the same evaluation, maximized.

#include "resolve/types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace resolve {

// One epoch's held-out scores, by target and metric name.
using HeldOutMetrics =
    std::unordered_map<std::string, std::unordered_map<std::string, float>>;

// Whether `metric` improves upward (Accuracy) rather than downward (Loss).
[[nodiscard]] bool selection_maximizes(SelectionMetric metric) noexcept;

// The best value a run starts from, worse than any value an epoch can score:
// +infinity for a value minimized, -infinity for one maximized
// (ReduceLROnPlateau's `mode_worse`).
[[nodiscard]] float selection_start(SelectionMetric metric) noexcept;

// Whether `value` improves on `best` by more than the relative `threshold`,
// the rule of PyTorch's ReduceLROnPlateau with threshold_mode "rel": in mode
// "min" value < best * (1 - threshold), in mode "max" (`maximize`)
// value > best * (1 + threshold). A threshold of 0 is a strict improvement,
// the rule early stopping and best_epoch select by. A NaN value never
// improves.
[[nodiscard]] bool improves_on(double value, double best, double threshold,
                               bool maximize) noexcept;

// The name of the classification target Accuracy reads: the one target of
// `targets` with TaskType::Classification. Throws std::invalid_argument when
// there is none or more than one, since the accuracy to select by is then
// undefined.
[[nodiscard]] const std::string& accuracy_target(const std::vector<TargetConfig>& targets);

// The value an epoch is selected by: `held_out_loss` under Loss; under
// Accuracy the accuracy of accuracy_target(targets) in `metrics`, NaN when
// the epoch scored none (a fold without a plot labelled for a masked target),
// which never improves.
[[nodiscard]] float selected_value(SelectionMetric metric, float held_out_loss,
                                   const HeldOutMetrics& metrics,
                                   const std::vector<TargetConfig>& targets);

// Refuses, with std::invalid_argument, a selection metric the run cannot
// follow: Accuracy without a held-out fold, or for a schema without exactly
// one classification target. Loss passes.
void validate_selection_metric(const TrainConfig& config,
                               const std::vector<TargetConfig>& targets, bool has_held_out);

}  // namespace resolve
