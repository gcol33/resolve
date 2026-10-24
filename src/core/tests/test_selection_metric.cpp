// The held-out value a run selects by (selection_metric.hpp). Under Loss the
// run follows the phase-invariant selection loss, minimized; under Accuracy
// the held-out accuracy of the schema's one classification target,
// maximized: an epoch improves when its accuracy is strictly above the best,
// and early stopping, best_epoch and the returned weights follow that value.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "resolve/dataset.hpp"
#include "resolve/loss.hpp"
#include "resolve/model.hpp"
#include "resolve/selection_metric.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

std::vector<TargetSpec> class_target() {
    return {TargetSpec::classification("cls", ClassificationCorpus::kClasses)};
}

// A noisy fit: a high rate and dropout keep the held-out loss and accuracy
// moving from epoch to epoch.
TrainConfig noisy_train_config(int epochs, SelectionMetric metric) {
    TrainConfig cfg = quiet_train_config(epochs);
    cfg.lr = 5e-2f;
    cfg.selection_metric = metric;
    return cfg;
}

struct SelectedRun {
    std::unique_ptr<Trainer> trainer;
    TrainResult result;
};

SelectedRun selected_run(const ResolveDataset& ds, const TrainConfig& cfg) {
    torch::manual_seed(5);
    ModelConfig model = hash_mlp_config({32, 16});
    model.dropout = 0.3f;
    SelectedRun run;
    run.trainer = std::make_unique<Trainer>(ResolveModel(ds.schema(), model), cfg);
    run.trainer->prepare_data(ds, 0.3f, 7);
    run.result = run.trainer->fit();
    return run;
}

// The first epoch whose value is strictly better than every earlier one's
// best, in `metric`'s direction.
int first_best(const std::vector<float>& values, SelectionMetric metric) {
    const bool maximize = selection_maximizes(metric);
    float best = selection_start(metric);
    int best_epoch = 0;
    for (size_t e = 0; e < values.size(); ++e) {
        if (improves_on(values[e], best, 0.0, maximize)) {
            best = values[e];
            best_epoch = static_cast<int>(e);
        }
    }
    return best_epoch;
}

}  // namespace

TEST_CASE("improves_on in mode max is ReduceLROnPlateau's max / rel rule",
          "[selection_metric]") {
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(improves_on(0.5, -inf, 0.0, true));
    CHECK(improves_on(0.0, -inf, 0.5, true));
    CHECK(improves_on(0.51, 0.5, 0.0, true));
    CHECK_FALSE(improves_on(0.5, 0.5, 0.0, true));
    CHECK_FALSE(improves_on(0.49, 0.5, 0.0, true));
    // 10 % above the best is needed under a 0.1 threshold.
    CHECK_FALSE(improves_on(0.54, 0.5, 0.1, true));
    CHECK(improves_on(0.56, 0.5, 0.1, true));
    CHECK_FALSE(improves_on(std::nan(""), 0.5, 0.0, true));
    CHECK_FALSE(improves_on(std::nan(""), -inf, 0.0, true));

    // Mode "min" on the negated value is a different rule under a relative
    // threshold: -0.54 is more than 10 % below -0.5, while 0.54 is not 10 %
    // above 0.5.
    CHECK(improves_on(-0.54, -0.5, 0.1, false));
    CHECK_FALSE(improves_on(0.54, 0.5, 0.1, true));
}

TEST_CASE("Each selection metric starts from the value no epoch is worse than",
          "[selection_metric]") {
    const float inf = std::numeric_limits<float>::infinity();
    CHECK_FALSE(selection_maximizes(SelectionMetric::Loss));
    CHECK(selection_maximizes(SelectionMetric::Accuracy));
    CHECK(selection_start(SelectionMetric::Loss) == inf);
    CHECK(selection_start(SelectionMetric::Accuracy) == -inf);
    CHECK(enum_to_name(SelectionMetric::Loss) == std::string("loss"));
    CHECK(enum_from_name<SelectionMetric>("accuracy") == SelectionMetric::Accuracy);
}

TEST_CASE("The selected value is the loss or the one classification target's accuracy",
          "[selection_metric]") {
    TargetConfig y;
    y.name = "y";
    TargetConfig cls;
    cls.name = "cls";
    cls.task = TaskType::Classification;
    cls.num_classes = 3;
    const std::vector<TargetConfig> targets = {y, cls};
    const HeldOutMetrics metrics = {{"y", {{"mae", 0.25f}}},
                                    {"cls", {{kAccuracyMetric, 0.625f}, {"macro_f1", 0.5f}}}};

    CHECK(accuracy_target(targets) == "cls");
    CHECK(selected_value(SelectionMetric::Loss, 1.5f, metrics, targets) == 1.5f);
    CHECK(selected_value(SelectionMetric::Accuracy, 1.5f, metrics, targets) == 0.625f);
    // A fold that scored the target on no plot leaves nothing to select by.
    CHECK(std::isnan(selected_value(SelectionMetric::Accuracy, 1.5f, {{"y", {}}}, targets)));

    REQUIRE_THROWS_WITH(accuracy_target({y}),
                        Catch::Matchers::ContainsSubstring("0 classification targets"));
    TargetConfig other = cls;
    other.name = "other";
    REQUIRE_THROWS_WITH(accuracy_target({cls, other}),
                        Catch::Matchers::ContainsSubstring("2 classification targets"));
}

TEST_CASE("Selecting by accuracy keeps the epoch of the highest held-out accuracy",
          "[selection_metric][trainer]") {
    // Patience longer than the run, so both runs train every epoch on the
    // same weights and differ only in the epoch they keep: the first lowest
    // held-out loss under Loss, the first highest held-out accuracy under
    // Accuracy. On this corpus the two are different epochs.
    const ClassificationCorpus corpus;
    const auto ds = corpus.load(class_target());
    TrainConfig by_loss = noisy_train_config(30, SelectionMetric::Loss);
    by_loss.patience = 100;
    TrainConfig by_accuracy = by_loss;
    by_accuracy.selection_metric = SelectionMetric::Accuracy;

    const auto loss_run = selected_run(ds, by_loss);
    const auto accuracy_run = selected_run(ds, by_accuracy);
    const auto& l = loss_run.result;
    const auto& a = accuracy_run.result;

    REQUIRE(l.test_loss_history.size() == 30);
    REQUIRE(a.test_loss_history == l.test_loss_history);
    REQUIRE(a.train_loss_history == l.train_loss_history);

    // Under Loss the selected values are the held-out losses.
    REQUIRE(l.selection_history == l.test_loss_history);
    REQUIRE(l.best_epoch == first_best(l.test_loss_history, SelectionMetric::Loss));

    REQUIRE(a.selection_history.size() == 30);
    REQUIRE(a.best_epoch == first_best(a.selection_history, SelectionMetric::Accuracy));
    const float best_accuracy = a.selection_history[static_cast<size_t>(a.best_epoch)];
    for (const float accuracy : a.selection_history) CHECK(accuracy <= best_accuracy);
    CHECK(a.final_metrics.at("cls").at(kAccuracyMetric) == best_accuracy);

    CHECK(a.best_epoch != l.best_epoch);

    // The returned weights are the best-accuracy epoch's: the baselines score
    // them again on the held-out fold and reproduce its accuracy.
    CHECK(a.baselines.at("cls").model_accuracy == best_accuracy);
}

TEST_CASE("Early stopping under accuracy stops after patience epochs without a higher one",
          "[selection_metric][trainer]") {
    const ClassificationCorpus corpus;
    const auto ds = corpus.load(class_target());
    TrainConfig cfg = noisy_train_config(80, SelectionMetric::Accuracy);
    cfg.patience = 4;

    const auto run = selected_run(ds, cfg);
    const auto& accuracies = run.result.selection_history;

    // A tie with the best is not an improvement, so it counts against
    // patience.
    float best = -std::numeric_limits<float>::infinity();
    int best_epoch = 0;
    int since_best = 0;
    int stop = static_cast<int>(accuracies.size());
    for (size_t e = 0; e < accuracies.size(); ++e) {
        if (accuracies[e] > best) {
            best = accuracies[e];
            best_epoch = static_cast<int>(e);
            since_best = 0;
        } else if (++since_best >= cfg.patience) {
            stop = static_cast<int>(e) + 1;
            break;
        }
    }
    REQUIRE(stop < cfg.max_epochs);
    CHECK(run.result.best_epoch == best_epoch);
    CHECK(static_cast<int>(accuracies.size()) == stop);
    CHECK(run.result.test_loss_history.size() == accuracies.size());
}

TEST_CASE("Selecting by accuracy beside a regression target reads the classification target",
          "[selection_metric][trainer]") {
    const ClassificationCorpus corpus;
    const auto ds = corpus.load({TargetSpec::regression("y"),
                                 TargetSpec::classification("cls", ClassificationCorpus::kClasses)});
    TrainConfig cfg = noisy_train_config(6, SelectionMetric::Accuracy);
    cfg.loss_config = LossConfigMode::MAE;
    cfg.patience = 100;

    const auto run = selected_run(ds, cfg);
    const auto& r = run.result;
    REQUIRE(r.selection_history.size() == 6);
    CHECK(r.best_epoch == first_best(r.selection_history, SelectionMetric::Accuracy));
    CHECK(r.final_metrics.at("cls").at(kAccuracyMetric) ==
          r.selection_history[static_cast<size_t>(r.best_epoch)]);
}

TEST_CASE("prepare_data refuses a selection by accuracy it cannot follow",
          "[selection_metric][trainer]") {
    const ClassificationCorpus corpus;
    const ModelConfig model = hash_mlp_config({8});

    auto refused = [&](const ResolveDataset& ds, const TrainConfig& cfg, float test_size,
                       const std::string& needle) {
        Trainer trainer(ResolveModel(ds.schema(), model), cfg);
        REQUIRE_THROWS_AS(trainer.prepare_data(ds, test_size, 3), std::invalid_argument);
        Trainer again(ResolveModel(ds.schema(), model), cfg);
        REQUIRE_THROWS_WITH(again.prepare_data(ds, test_size, 3),
                            Catch::Matchers::ContainsSubstring(needle));
        // A refused call leaves nothing prepared.
        REQUIRE_THROWS_WITH(again.fit(), Catch::Matchers::ContainsSubstring("prepared"));
    };

    TrainConfig accuracy = quiet_train_config(3);
    accuracy.selection_metric = SelectionMetric::Accuracy;

    SECTION("nothing held out") {
        TrainConfig cfg = accuracy;
        cfg.fixed_epochs = 3;
        refused(corpus.load(class_target()), cfg, 0.0f, "no held-out fold");
    }
    SECTION("no classification target") {
        refused(corpus.load({TargetSpec::regression("y")}), accuracy, 0.25f,
                "0 classification targets");
    }
    SECTION("two classification targets") {
        refused(corpus.load({TargetSpec::classification("cls", ClassificationCorpus::kClasses),
                             TargetSpec::classification("cls2", 2)}),
                accuracy, 0.25f, "2 classification targets");
    }
    SECTION("the loss needs neither") {
        TrainConfig cfg = quiet_train_config(2);
        cfg.fixed_epochs = 2;
        const auto ds = corpus.load({TargetSpec::regression("y")});
        Trainer trainer(ResolveModel(ds.schema(), model), cfg);
        REQUIRE_NOTHROW(trainer.prepare_data(ds, 0.0f, 3));
        const auto result = trainer.fit();
        CHECK(result.selection_history.empty());
    }
}
