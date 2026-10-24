#include "resolve/lr_schedule.hpp"
#include "resolve/utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace resolve {

PlateauState plateau_start(const TrainConfig& config) {
    PlateauState state;
    state.best = static_cast<double>(selection_start(config.selection_metric));
    state.lr = static_cast<double>(config.lr);
    return state;
}

bool plateau_step(PlateauState& state, const TrainConfig& config, double value, bool counts) {
    if (improves_on(value, state.best, static_cast<double>(config.lr_plateau_threshold),
                    selection_maximizes(config.selection_metric))) {
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
        case LRSchedulerType::LinearDecay:
            // The decay is per step (step_learning_rate); config.lr is its peak.
        case LRSchedulerType::None:
        default:
            return config.lr;
    }
}

int64_t steps_per_epoch(int64_t n_rows, int64_t batch_size) {
    if (n_rows <= 0 || batch_size <= 0) return 1;
    return std::max<int64_t>(1, (n_rows + batch_size - 1) / batch_size);
}

int64_t warmup_step_count(const TrainConfig& config, int64_t steps_per_epoch) {
    if (config.warmup_epochs > 0) {
        return static_cast<int64_t>(config.warmup_epochs) * steps_per_epoch;
    }
    return std::max<int64_t>(0, config.warmup_steps);
}

double warmup_factor(int64_t step, int64_t warmup_steps) {
    if (warmup_steps <= 0 || step >= warmup_steps) return 1.0;
    return static_cast<double>(step + 1) / static_cast<double>(warmup_steps);
}

bool epoch_in_warmup(const TrainConfig& config, int epoch, int64_t steps_per_epoch) {
    return static_cast<int64_t>(epoch) * steps_per_epoch <
           warmup_step_count(config, steps_per_epoch);
}

double step_learning_rate(const TrainConfig& config, int epoch, int64_t step,
                          int64_t steps_per_epoch, const PlateauState& plateau) {
    const int64_t warmup = warmup_step_count(config, steps_per_epoch);
    double rate = scheduled_learning_rate(config, epoch, plateau);
    if (config.lr_scheduler == LRSchedulerType::LinearDecay && step >= warmup) {
        // From lr at the first step after warmup to lr_min at the last step of
        // max_epochs, the convention CosineAnnealing keeps for its epochs.
        const int64_t last_step =
            static_cast<int64_t>(std::max(config.max_epochs, 1)) * steps_per_epoch - 1;
        const int64_t span = last_step - warmup;
        const double progress = span > 0
            ? std::clamp(static_cast<double>(step - warmup) / static_cast<double>(span), 0.0, 1.0)
            : 1.0;
        const double lr = static_cast<double>(config.lr);
        const double lr_min = static_cast<double>(config.lr_min);
        rate = lr - (lr - lr_min) * progress;
    }
    return rate * warmup_factor(step, warmup);
}

void set_learning_rate(torch::optim::AdamW& optimizer, double lr) {
    for (auto& group : optimizer.param_groups()) {
        static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
    }
}

void validate_lr_schedule(const TrainConfig& config, bool has_held_out) {
    if (config.warmup_steps < 0) {
        throw std::invalid_argument(
            "warmup_steps = " + std::to_string(config.warmup_steps) +
            "; it counts optimizer steps, so it is 0 (no warmup) or positive");
    }
    if (config.warmup_epochs < 0) {
        throw std::invalid_argument(
            "warmup_epochs = " + std::to_string(config.warmup_epochs) +
            "; it counts epochs, so it is 0 (no warmup) or positive");
    }
    if (config.warmup_steps > 0 && config.warmup_epochs > 0) {
        throw std::invalid_argument(
            "warmup_steps = " + std::to_string(config.warmup_steps) + " and warmup_epochs = " +
            std::to_string(config.warmup_epochs) + " both set the warmup; give one of them");
    }
    const bool floored = config.lr_scheduler == LRSchedulerType::ReduceOnPlateau ||
                         config.lr_scheduler == LRSchedulerType::LinearDecay;
    if (floored && !(config.lr_min >= 0.0f)) {
        throw std::invalid_argument(
            "lr_min = " + std::to_string(config.lr_min) +
            "; the learning-rate floor is 0 or positive");
    }
    if (config.lr_scheduler != LRSchedulerType::ReduceOnPlateau) {
        return;
    }
    if (!has_held_out) {
        throw std::invalid_argument(
            "the reduce-on-plateau learning-rate schedule (lr_scheduler = plateau) "
            "decays on the held-out fold's selection value, and the data was prepared with no "
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
}

}  // namespace resolve
