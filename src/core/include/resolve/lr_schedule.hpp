#pragma once

// The learning rate an epoch trains at. StepLR and CosineAnnealing depend on
// the epoch alone; ReduceOnPlateau follows the held-out fold's selection loss
// and so carries state from epoch to epoch (PlateauState), which the run's
// progress and its checkpoints hold.

#include "resolve/types.hpp"

#include <limits>

namespace resolve {

// Whether `value` improves on `best` by more than the relative `threshold`:
// value < best * (1 - threshold), the rule of PyTorch's ReduceLROnPlateau in
// mode "min" with threshold_mode "rel". A threshold of 0 is a strict decrease,
// the rule early stopping and best_epoch select by. A NaN value never
// improves.
[[nodiscard]] bool improves_on(double value, double best, double threshold) noexcept;

// The smallest learning-rate change a plateau decay applies; a decay that
// would change the rate by this much or less (the rate already at lr_min) is
// skipped. ReduceLROnPlateau's `eps` default.
inline constexpr double kPlateauMinDecay = 1e-8;

// A ReduceOnPlateau schedule between epochs, ReduceLROnPlateau's state: the
// best loss so far, the bad epochs counted since the last improvement or
// decay, the cooldown epochs left, and the rate the next epoch trains at.
struct PlateauState {
    double best = std::numeric_limits<double>::infinity();
    int bad_epochs = 0;
    int cooldown_counter = 0;
    double lr = 0.0;
};

// The state a run starts from: no best yet, the rate config.lr.
[[nodiscard]] PlateauState plateau_start(const TrainConfig& config);

// One ReduceLROnPlateau.step(value) at the end of an epoch whose held-out
// selection loss is `value`. An improvement (improves_on with
// config.lr_plateau_threshold) becomes the best and clears the bad-epoch
// count; otherwise the epoch is a bad one when `counts`. During a cooldown
// the counter decrements and the bad epochs are dropped. More than
// config.lr_plateau_patience bad epochs multiply the rate by
// config.lr_plateau_factor, floored at config.lr_min, and start a cooldown of
// config.lr_plateau_cooldown epochs. `counts` is false while the loss
// objective is still changing phase (MultiTaskLoss::objective_settled), so an
// epoch then never counts against the rate, as it never counts against
// early stopping's patience. Returns whether the rate changed.
bool plateau_step(PlateauState& state, const TrainConfig& config, double value, bool counts);

// The rate epoch `epoch` trains at under config.lr_scheduler: config.lr for
// None, the step or cosine schedule laid out over config.max_epochs, the
// plateau state's rate for ReduceOnPlateau.
[[nodiscard]] double scheduled_learning_rate(const TrainConfig& config, int epoch,
                                             const PlateauState& plateau);

// Refuses a schedule the run cannot follow, with std::invalid_argument: a
// ReduceOnPlateau schedule without a held-out fold (there is no loss to
// follow), a plateau factor outside (0, 1), a negative patience or cooldown,
// a threshold outside [0, 1), and a negative lr_min. Other schedules pass.
void validate_lr_schedule(const TrainConfig& config, bool has_held_out);

}  // namespace resolve
