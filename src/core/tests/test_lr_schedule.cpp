// The learning-rate schedules (lr_schedule.hpp). ReduceOnPlateau is PyTorch's
// ReduceLROnPlateau in mode "min" on the held-out selection loss: the scripted
// traces below are what torch.optim.lr_scheduler.ReduceLROnPlateau (torch
// 2.13) reports after each step(value) for the same values and knobs. In a
// run the schedule reads the loss early stopping reads, counts a bad epoch
// only where patience does, and refuses a split with nothing held out.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "resolve/dataset.hpp"
#include "resolve/lr_schedule.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

// One step of a scripted trace: the value stepped with, and the state after.
struct PlateauStep {
    double value;
    double lr;
    double best;
    int bad_epochs;
    int cooldown_counter;
};

TrainConfig plateau_config(float lr, float factor, int patience, float threshold, int cooldown,
                           float lr_min) {
    TrainConfig cfg;
    cfg.lr_scheduler = LRSchedulerType::ReduceOnPlateau;
    cfg.lr = lr;
    cfg.lr_plateau_factor = factor;
    cfg.lr_plateau_patience = patience;
    cfg.lr_plateau_threshold = threshold;
    cfg.lr_plateau_cooldown = cooldown;
    cfg.lr_min = lr_min;
    return cfg;
}

void require_trace(const TrainConfig& cfg, const std::vector<PlateauStep>& trace) {
    PlateauState state = plateau_start(cfg);
    REQUIRE(state.lr == Catch::Approx(cfg.lr));
    for (size_t i = 0; i < trace.size(); ++i) {
        INFO("step " << i << ", value " << trace[i].value);
        const double lr_before = state.lr;
        const bool changed = plateau_step(state, cfg, trace[i].value, /*counts=*/true);
        CHECK(state.lr == Catch::Approx(trace[i].lr).epsilon(1e-6));
        CHECK(changed == (state.lr != lr_before));
        CHECK(state.best == trace[i].best);
        CHECK(state.bad_epochs == trace[i].bad_epochs);
        CHECK(state.cooldown_counter == trace[i].cooldown_counter);
    }
}

// A short regression fit on SmallCorpus under `cfg`, 30 of its 40 plots
// fitting and 10 held out.
struct PlateauRun {
    std::unique_ptr<Trainer> trainer;
    TrainResult result;
};

PlateauRun plateau_run(const SmallCorpus& corpus, const TrainConfig& cfg) {
    const auto ds = corpus.load(SpeciesEncodingMode::Hash);
    torch::manual_seed(3);
    ModelConfig model = small_encoder_config(SpeciesEncodingMode::Hash, {16, 8});
    model.dropout = 0.2f;
    PlateauRun run;
    run.trainer = std::make_unique<Trainer>(ResolveModel(ds.schema(), model), cfg);
    run.trainer->prepare_data(ds, 0.25f, 11);
    run.result = run.trainer->fit();
    return run;
}

// The rate each epoch of `result` should have trained at: the run's held-out
// losses replayed through the schedule, epoch e counting from `first_counted`.
std::vector<float> replayed_rates(const TrainConfig& cfg, const TrainResult& result,
                                  int first_counted) {
    std::vector<float> rates;
    PlateauState state = plateau_start(cfg);
    for (size_t e = 0; e < result.test_loss_history.size(); ++e) {
        rates.push_back(static_cast<float>(state.lr));
        plateau_step(state, cfg, result.test_loss_history[e],
                     static_cast<int>(e) >= first_counted);
    }
    return rates;
}

}  // namespace

TEST_CASE("improves_on is ReduceLROnPlateau's min / rel rule", "[lr_schedule]") {
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(improves_on(5.0, inf, 0.0));
    CHECK(improves_on(5.0, inf, 0.5));
    CHECK(improves_on(0.99, 1.0, 0.0));
    CHECK_FALSE(improves_on(1.0, 1.0, 0.0));
    CHECK_FALSE(improves_on(1.01, 1.0, 0.0));
    // 10 % below the best is needed under a 0.1 threshold.
    CHECK_FALSE(improves_on(0.95, 1.0, 0.1));
    CHECK(improves_on(0.85, 1.0, 0.1));
    CHECK_FALSE(improves_on(std::nan(""), 1.0, 0.0));
    CHECK_FALSE(improves_on(std::nan(""), inf, 0.0));
}

TEST_CASE("A plateau schedule steps as ReduceLROnPlateau with a threshold, cooldown and floor",
          "[lr_schedule]") {
    // ReduceLROnPlateau(mode="min", factor=0.5, patience=2, threshold=0.1,
    // threshold_mode="rel", cooldown=1, min_lr=0.2) on an optimizer at lr 1.0.
    // A decay comes on the third bad epoch in a row (bad > patience); the
    // cooldown epoch after it drops its bad epoch; the third decay is floored
    // at 0.2 and the fourth changes the rate by less than eps, so it is
    // skipped (the cooldown still starts).
    const TrainConfig cfg = plateau_config(1.0f, 0.5f, 2, 0.1f, 1, 0.2f);
    require_trace(cfg, {
        {10.0, 1.0, 10.0, 0, 0},
        {9.5, 1.0, 10.0, 1, 0},
        {9.2, 1.0, 10.0, 2, 0},
        {9.1, 0.5, 10.0, 0, 1},
        {8.0, 0.5, 8.0, 0, 0},
        {8.5, 0.5, 8.0, 1, 0},
        {8.6, 0.5, 8.0, 2, 0},
        {8.7, 0.25, 8.0, 0, 1},
        {8.8, 0.25, 8.0, 0, 0},
        {7.0, 0.25, 7.0, 0, 0},
        {7.5, 0.25, 7.0, 1, 0},
        {7.6, 0.25, 7.0, 2, 0},
        {7.7, 0.2, 7.0, 0, 1},
        {7.8, 0.2, 7.0, 0, 0},
        {7.9, 0.2, 7.0, 1, 0},
        {8.0, 0.2, 7.0, 2, 0},
        {8.1, 0.2, 7.0, 0, 1},
        {8.2, 0.2, 7.0, 0, 0},
        {8.3, 0.2, 7.0, 1, 0},
        {8.4, 0.2, 7.0, 2, 0},
    });
}

TEST_CASE("Under a zero threshold a plateau counts a tie as a bad epoch", "[lr_schedule]") {
    // ReduceLROnPlateau(factor=0.1, patience=1, threshold=0, min_lr=1e-6) at
    // lr 0.1: early stopping's rule, a strict decrease.
    const TrainConfig cfg = plateau_config(0.1f, 0.1f, 1, 0.0f, 0, 1e-6f);
    require_trace(cfg, {
        {1.0, 0.1, 1.0, 0, 0},
        {1.0, 0.1, 1.0, 1, 0},
        {1.0, 0.01, 1.0, 0, 0},
        {0.9, 0.01, 0.9, 0, 0},
        {0.95, 0.01, 0.9, 1, 0},
        {0.9, 0.001, 0.9, 0, 0},
        {0.89, 0.001, 0.89, 0, 0},
        {0.89, 0.001, 0.89, 1, 0},
        {0.89, 1e-4, 0.89, 0, 0},
        {0.89, 1e-4, 0.89, 1, 0},
        {0.89, 1e-5, 0.89, 0, 0},
        {0.89, 1e-5, 0.89, 1, 0},
    });
}

TEST_CASE("A step that does not count keeps the best but adds no bad epoch", "[lr_schedule]") {
    const TrainConfig cfg = plateau_config(1.0f, 0.5f, 0, 0.0f, 0, 0.0f);
    PlateauState state = plateau_start(cfg);

    CHECK_FALSE(plateau_step(state, cfg, 2.0, /*counts=*/false));
    CHECK(state.best == 2.0);
    CHECK_FALSE(plateau_step(state, cfg, 3.0, /*counts=*/false));
    CHECK_FALSE(plateau_step(state, cfg, 1.5, /*counts=*/false));
    CHECK(state.best == 1.5);
    CHECK(state.bad_epochs == 0);
    CHECK(state.lr == 1.0);

    // Counted, the same non-improving value decays the rate at patience 0.
    CHECK(plateau_step(state, cfg, 3.0, /*counts=*/true));
    CHECK(state.lr == 0.5);
}

TEST_CASE("The epoch schedules take their rate from the epoch alone", "[lr_schedule]") {
    TrainConfig cfg;
    cfg.lr = 0.1f;
    const PlateauState plateau = plateau_start(cfg);
    CHECK(scheduled_learning_rate(cfg, 7, plateau) == Catch::Approx(0.1));

    cfg.lr_scheduler = LRSchedulerType::StepLR;
    cfg.lr_step_size = 3;
    cfg.lr_gamma = 0.5f;
    CHECK(scheduled_learning_rate(cfg, 2, plateau) == Catch::Approx(0.1));
    CHECK(scheduled_learning_rate(cfg, 3, plateau) == Catch::Approx(0.05));
    CHECK(scheduled_learning_rate(cfg, 7, plateau) == Catch::Approx(0.025));

    cfg.lr_scheduler = LRSchedulerType::CosineAnnealing;
    cfg.max_epochs = 11;
    cfg.lr_min = 0.0f;
    CHECK(scheduled_learning_rate(cfg, 0, plateau) == Catch::Approx(0.1));
    CHECK(scheduled_learning_rate(cfg, 5, plateau) == Catch::Approx(0.05));
    CHECK(scheduled_learning_rate(cfg, 10, plateau) == Catch::Approx(0.0).margin(1e-9));

    cfg.lr_scheduler = LRSchedulerType::ReduceOnPlateau;
    PlateauState decayed = plateau;
    decayed.lr = 0.003;
    CHECK(scheduled_learning_rate(cfg, 0, decayed) == 0.003);
}

TEST_CASE("A run trains each epoch at the rate the held-out losses set",
          "[lr_schedule][trainer]") {
    // A threshold no epoch can meet (a 1000-fold drop on the best) makes every
    // epoch after the first a bad one, so the rate halves on a fixed rhythm:
    // two bad epochs, a decay, a cooldown epoch.
    const SmallCorpus corpus;
    TrainConfig cfg = quiet_train_config(12);
    cfg.loss_config = LossConfigMode::MAE;
    cfg.lr = 1e-2f;
    cfg.lr_scheduler = LRSchedulerType::ReduceOnPlateau;
    cfg.lr_plateau_factor = 0.5f;
    cfg.lr_plateau_patience = 1;
    cfg.lr_plateau_threshold = 0.999f;
    cfg.lr_plateau_cooldown = 1;

    const auto run = plateau_run(corpus, cfg);
    const auto& r = run.result;

    REQUIRE(r.lr_history.size() == 12);
    REQUIRE(r.lr_history == replayed_rates(cfg, r, /*first_counted=*/0));
    // Epoch 0 sets the best and epochs 1 and 2 are bad, so epoch 3 trains at
    // half the rate; epoch 3 is the cooldown epoch, 4 and 5 are bad again.
    const double lr0 = static_cast<double>(cfg.lr);
    for (int e = 0; e < 12; ++e) {
        INFO("epoch " << e);
        CHECK(r.lr_history[static_cast<size_t>(e)] ==
              static_cast<float>(lr0 * std::pow(0.5, e / 3)));
    }
}

TEST_CASE("A plateau counts no bad epoch before the loss objective settles",
          "[lr_schedule][trainer]") {
    // The combined loss changes phase at epochs 2 and 4, so patience counts
    // from epoch 4 on and so does the schedule: with no epoch able to improve
    // and patience 0, the first decay follows epoch 4 and every later epoch
    // decays again.
    const SmallCorpus corpus;
    TrainConfig cfg = quiet_train_config(10);
    cfg.loss_config = LossConfigMode::Combined;
    cfg.phase_boundaries = {2, 4};
    cfg.lr = 1e-2f;
    cfg.lr_scheduler = LRSchedulerType::ReduceOnPlateau;
    cfg.lr_plateau_factor = 0.5f;
    cfg.lr_plateau_patience = 0;
    cfg.lr_plateau_threshold = 0.999f;

    const auto run = plateau_run(corpus, cfg);
    const auto& r = run.result;

    REQUIRE(r.lr_history.size() == 10);
    REQUIRE(r.lr_history == replayed_rates(cfg, r, /*first_counted=*/4));
    const double lr0 = static_cast<double>(cfg.lr);
    for (int e = 0; e < 10; ++e) {
        INFO("epoch " << e);
        CHECK(r.lr_history[static_cast<size_t>(e)] ==
              static_cast<float>(lr0 * std::pow(0.5, std::max(0, e - 4))));
    }
}

TEST_CASE("Early stopping and the plateau read one held-out loss", "[lr_schedule][trainer]") {
    // Leblanc et al.'s protocol at a smaller scale: the rate decays after more
    // than 2 epochs without a new best, training stops after 4. A high rate
    // and dropout keep the held-out loss moving. Whatever the run does, its
    // rates are the replay of its own held-out losses, best_epoch is their
    // first minimum, and the run ends where 4 epochs in a row failed to beat
    // the best -- the bad epochs the schedule counts.
    const SmallCorpus corpus;
    TrainConfig cfg = quiet_train_config(60);
    cfg.loss_config = LossConfigMode::MAE;
    cfg.patience = 4;
    cfg.lr = 5e-2f;
    cfg.lr_scheduler = LRSchedulerType::ReduceOnPlateau;
    cfg.lr_plateau_factor = 0.1f;
    cfg.lr_plateau_patience = 2;

    const auto run = plateau_run(corpus, cfg);
    const auto& r = run.result;
    const auto& losses = r.test_loss_history;

    REQUIRE(r.lr_history == replayed_rates(cfg, r, /*first_counted=*/0));

    float best = std::numeric_limits<float>::infinity();
    int best_epoch = 0;
    int since_best = 0;
    int stop = static_cast<int>(losses.size());
    for (size_t e = 0; e < losses.size(); ++e) {
        if (losses[e] < best) {
            best = losses[e];
            best_epoch = static_cast<int>(e);
            since_best = 0;
        } else if (++since_best >= cfg.patience) {
            stop = static_cast<int>(e) + 1;
            break;
        }
    }
    CHECK(r.best_epoch == best_epoch);
    CHECK(static_cast<int>(losses.size()) == stop);
}

TEST_CASE("prepare_data refuses a plateau schedule it cannot drive",
          "[lr_schedule][trainer]") {
    const SmallCorpus corpus;
    const auto ds = corpus.load(SpeciesEncodingMode::Hash);
    const ModelConfig model = small_encoder_config(SpeciesEncodingMode::Hash, {8});

    auto refused = [&](const TrainConfig& cfg, float test_size, const std::string& needle) {
        Trainer trainer(ResolveModel(ds.schema(), model), cfg);
        REQUIRE_THROWS_WITH(trainer.prepare_data(ds, test_size, 3),
                            Catch::Matchers::ContainsSubstring(needle));
        // A refused call leaves nothing prepared.
        REQUIRE_THROWS_WITH(trainer.fit(), Catch::Matchers::ContainsSubstring("prepared"));
    };

    TrainConfig plateau = quiet_train_config(4);
    plateau.lr_scheduler = LRSchedulerType::ReduceOnPlateau;

    SECTION("a fixed duration with nothing held out") {
        TrainConfig cfg = plateau;
        cfg.fixed_epochs = 4;
        refused(cfg, 0.0f, "no held-out fold");
    }
    SECTION("nothing held out under early stopping") {
        refused(plateau, 0.0f, "no held-out fold");
    }
    SECTION("knobs out of range") {
        TrainConfig cfg = plateau;
        cfg.lr_plateau_factor = 1.0f;
        refused(cfg, 0.25f, "lr_plateau_factor");
        cfg.lr_plateau_factor = 0.0f;
        refused(cfg, 0.25f, "lr_plateau_factor");
        cfg = plateau;
        cfg.lr_plateau_patience = -1;
        refused(cfg, 0.25f, "lr_plateau_patience");
        cfg = plateau;
        cfg.lr_plateau_cooldown = -1;
        refused(cfg, 0.25f, "lr_plateau_cooldown");
        cfg = plateau;
        cfg.lr_plateau_threshold = 1.0f;
        refused(cfg, 0.25f, "lr_plateau_threshold");
        cfg.lr_plateau_threshold = -0.1f;
        refused(cfg, 0.25f, "lr_plateau_threshold");
        cfg = plateau;
        cfg.lr_min = -1e-3f;
        refused(cfg, 0.25f, "lr_min");
    }
    SECTION("an epoch schedule still trains a fixed duration on every plot") {
        TrainConfig cfg = quiet_train_config(4);
        cfg.lr_scheduler = LRSchedulerType::CosineAnnealing;
        cfg.fixed_epochs = 2;
        Trainer trainer(ResolveModel(ds.schema(), model), cfg);
        REQUIRE_NOTHROW(trainer.prepare_data(ds, 0.0f, 3));
        const auto result = trainer.fit();
        CHECK(result.lr_history.size() == 2);
    }
}
