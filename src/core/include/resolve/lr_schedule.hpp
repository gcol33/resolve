#pragma once

// The learning rate an optimizer step trains at. StepLR and CosineAnnealing
// depend on the epoch alone; ReduceOnPlateau follows the held-out fold's
// selection value (TrainConfig::selection_metric, selection_metric.hpp) and so
// carries state from epoch to epoch (PlateauState), which the run's progress
// and its checkpoints hold; LinearDecay depends on the step. A linear warmup
// scales whichever schedule over the run's first steps.
//
// A run counts its optimizer steps from 0: batch b of epoch e is step
// e * steps_per_epoch + b. The step count is therefore the epoch the run is
// at and the batch size it trains with, both of which a checkpoint's progress
// carries, so a resumed run continues the warmup at the step it stopped.

#include "resolve/selection_metric.hpp"
#include "resolve/types.hpp"

#include <limits>

namespace resolve {

// The smallest learning-rate change a plateau decay applies; a decay that
// would change the rate by this much or less (the rate already at lr_min) is
// skipped. ReduceLROnPlateau's `eps` default.
inline constexpr double kPlateauMinDecay = 1e-8;

// A ReduceOnPlateau schedule between epochs, ReduceLROnPlateau's state: the
// best selection value so far (plateau_start sets the start for the metric's
// direction), the bad epochs counted since the last improvement or decay, the
// cooldown epochs left, and the rate the next epoch trains at.
struct PlateauState {
    double best = std::numeric_limits<double>::infinity();
    int bad_epochs = 0;
    int cooldown_counter = 0;
    double lr = 0.0;
};

// The state a run starts from: no best yet (selection_start of
// config.selection_metric), the rate config.lr.
[[nodiscard]] PlateauState plateau_start(const TrainConfig& config);

// One ReduceLROnPlateau.step(value) at the end of an epoch whose held-out
// selection value is `value`. An improvement (improves_on with
// config.lr_plateau_threshold, in mode "max" when config.selection_metric
// maximizes, "min" otherwise) becomes the best and clears the bad-epoch
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
// plateau state's rate for ReduceOnPlateau, and config.lr, the peak, for
// LinearDecay, whose decay step_learning_rate applies per step.
[[nodiscard]] double scheduled_learning_rate(const TrainConfig& config, int epoch,
                                             const PlateauState& plateau);

// The optimizer steps an epoch over `n_rows` rows takes in batches of
// `batch_size`: ceil(n_rows / batch_size), and at least 1.
[[nodiscard]] int64_t steps_per_epoch(int64_t n_rows, int64_t batch_size);

// The run's warmup in optimizer steps: config.warmup_steps, or
// config.warmup_epochs epochs of `steps_per_epoch` steps.
[[nodiscard]] int64_t warmup_step_count(const TrainConfig& config, int64_t steps_per_epoch);

// The factor linear warmup scales step `step` (0-based) by: (step + 1) /
// warmup_steps below warmup_steps, 1 from there on and for warmup_steps <= 0.
[[nodiscard]] double warmup_factor(int64_t step, int64_t warmup_steps);

// Whether epoch `epoch` begins inside the warmup, so it counts no bad epoch
// against patience or a ReduceOnPlateau schedule.
[[nodiscard]] bool epoch_in_warmup(const TrainConfig& config, int epoch,
                                   int64_t steps_per_epoch);

// The rate step `step` of epoch `epoch` trains at, epochs of
// `steps_per_epoch` steps: the epoch's scheduled_learning_rate, or under
// LinearDecay the rate falling linearly from config.lr at the first step after
// warmup to config.lr_min at the last step of config.max_epochs, times
// warmup_factor. Without warmup and outside LinearDecay it is
// scheduled_learning_rate itself.
[[nodiscard]] double step_learning_rate(const TrainConfig& config, int epoch, int64_t step,
                                        int64_t steps_per_epoch, const PlateauState& plateau);

// Sets the rate every parameter group of `optimizer` steps at.
void set_learning_rate(torch::optim::AdamW& optimizer, double lr);

// Refuses a schedule the run cannot follow, with std::invalid_argument: a
// ReduceOnPlateau schedule without a held-out fold (there is no value to
// follow), a plateau factor outside (0, 1), a negative patience or cooldown,
// a threshold outside [0, 1), a negative lr_min under a plateau or linear
// schedule, and a negative warmup_steps or warmup_epochs
// or both positive. Other schedules pass.
void validate_lr_schedule(const TrainConfig& config, bool has_held_out);

}  // namespace resolve
