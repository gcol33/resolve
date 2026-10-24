#include "resolve/selection_metric.hpp"
#include "resolve/loss.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace resolve {

bool selection_maximizes(SelectionMetric metric) noexcept {
    return metric == SelectionMetric::Accuracy;
}

float selection_start(SelectionMetric metric) noexcept {
    const float inf = std::numeric_limits<float>::infinity();
    return selection_maximizes(metric) ? -inf : inf;
}

bool improves_on(double value, double best, double threshold, bool maximize) noexcept {
    return maximize ? value > best * (1.0 + threshold) : value < best * (1.0 - threshold);
}

const std::string& accuracy_target(const std::vector<TargetConfig>& targets) {
    const TargetConfig* found = nullptr;
    int n_classification = 0;
    for (const auto& target : targets) {
        if (target.task != TaskType::Classification) continue;
        ++n_classification;
        if (found == nullptr) found = &target;
    }
    if (n_classification != 1) {
        throw std::invalid_argument(
            "selection_metric = accuracy selects by the held-out accuracy of the "
            "schema's one classification target, and the schema has " +
            std::to_string(n_classification) +
            " classification targets. Select by the loss (selection_metric = loss), "
            "or train a schema with exactly one classification target.");
    }
    return found->name;
}

float selected_value(SelectionMetric metric, float held_out_loss, const HeldOutMetrics& metrics,
                     const std::vector<TargetConfig>& targets) {
    if (metric == SelectionMetric::Loss) {
        return held_out_loss;
    }
    const auto target_it = metrics.find(accuracy_target(targets));
    if (target_it == metrics.end()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const auto value_it = target_it->second.find(kAccuracyMetric);
    return value_it == target_it->second.end() ? std::numeric_limits<float>::quiet_NaN()
                                               : value_it->second;
}

void validate_selection_metric(const TrainConfig& config,
                               const std::vector<TargetConfig>& targets, bool has_held_out) {
    if (config.selection_metric == SelectionMetric::Loss) {
        return;
    }
    if (!has_held_out) {
        throw std::invalid_argument(
            "selection_metric = accuracy selects by the held-out fold's accuracy, and "
            "the data was prepared with no held-out fold (test_size = 0), so there is "
            "nothing to select by. Hold out a fold, or select by the loss "
            "(selection_metric = loss).");
    }
    static_cast<void>(accuracy_target(targets));
}

}  // namespace resolve
