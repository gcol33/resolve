#include "resolve/lr_schedule.hpp"
#include "resolve/utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace resolve {

bool improves_on(double value, double best, double threshold) noexcept {
    return value < best * (1.0 - threshold);
}

PlateauState plateau_start(const TrainConfig& config) {
    PlateauState state;
    state.lr = static_cast<double>(config.lr);
    return state;
}

bool plateau_step(PlateauState& state, const TrainConfig& config, double value, bool counts) {
    if (improves_on(value, state.best, static_cast<double>(config.lr_plateau_threshold))) {
        state.best = value;
        state.bad_epochs = 0;
    } else if (counts) {
        ++state.bad_epochs;
    }

    if (state.cooldown_counter > 0) {
        --state.cooldown_counter;
        state.bad_epochs = 0;
    }

    if (state.bad_epochs <= config.lr_plateau_patience) {
        return false;
    }
    const double decayed = std::max(state.lr * static_cast<double>(config.lr_plateau_factor),
                                    static_cast<double>(config.lr_min));
    const bool changed = state.lr - decayed > kPlateauMinDecay;
    if (changed) {
        state.lr = decayed;
    }
    state.cooldown_counter = config.lr_plateau_cooldown;
    state.bad_epochs = 0;
    return changed;
}

double scheduled_learning_rate(const TrainConfig& config, int epoch,
                               const PlateauState& plateau) {
    switch (config.lr_scheduler) {
        case LRSchedulerType::StepLR: {
            // Multiply by gamma every lr_step_size epochs.
            const int n_decays = config.lr_step_size > 0 ? epoch / config.lr_step_size : 0;
            return config.lr * std::pow(config.lr_gamma, static_cast<float>(n_decays));
        }
        case LRSchedulerType::CosineAnnealing: {
            // From lr at epoch 0 to lr_min at the final epoch: progress is
            // divided by (max_epochs - 1) so it reaches 1 on the last epoch.
            const float progress = config.max_epochs > 1
                ? static_cast<float>(epoch) / static_cast<float>(config.max_epochs - 1)
                : 1.0f;
            return config.lr_min + (config.lr - config.lr_min) * cosine_ramp(progress);
        }
        case LRSchedulerType::ReduceOnPlateau:
            return plateau.lr;
        case LRSchedulerType::None:
        default:
            return config.lr;
    }
}

void validate_lr_schedule(const TrainConfig& config, bool has_held_out) {
    if (config.lr_scheduler != LRSchedulerType::ReduceOnPlateau) {
        return;
    }
    if (!has_held_out) {
        throw std::invalid_argument(
            "the reduce-on-plateau learning-rate schedule (lr_scheduler = plateau) "
            "decays on the held-out fold's loss, and the data was prepared with no "
            "held-out fold (test_size = 0), so it has nothing to follow. Hold out a "
            "fold, or choose a schedule laid out over max_epochs (step, cosine).");
    }
    if (!(config.lr_plateau_factor > 0.0f && config.lr_plateau_factor < 1.0f)) {
        throw std::invalid_argument(
            "lr_plateau_factor = " + std::to_string(config.lr_plateau_factor) +
            "; the factor a plateau multiplies the learning rate by lies in (0, 1)");
    }
    if (config.lr_plateau_patience < 0) {
        throw std::invalid_argument(
            "lr_plateau_patience = " + std::to_string(config.lr_plateau_patience) +
            "; it counts epochs, so it is 0 or positive");
    }
    if (config.lr_plateau_cooldown < 0) {
        throw std::invalid_argument(
            "lr_plateau_cooldown = " + std::to_string(config.lr_plateau_cooldown) +
            "; it counts epochs, so it is 0 or positive");
    }
    if (!(config.lr_plateau_threshold >= 0.0f && config.lr_plateau_threshold < 1.0f)) {
        throw std::invalid_argument(
            "lr_plateau_threshold = " + std::to_string(config.lr_plateau_threshold) +
            "; the relative improvement an epoch must make lies in [0, 1)");
    }
    if (!(config.lr_min >= 0.0f)) {
        throw std::invalid_argument(
            "lr_min = " + std::to_string(config.lr_min) +
            "; the learning-rate floor is 0 or positive");
    }
}

}  // namespace resolve
