#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "resolve/loss.hpp"

using namespace resolve;
using namespace Catch::Matchers;

// ============================================================================
// Regression Metrics Tests
// ============================================================================

TEST_CASE("Metrics::mae", "[metrics][regression]") {
    auto pred = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto target = torch::tensor({1.5f, 2.0f, 2.5f, 4.5f, 5.0f});

    float mae = Metrics::mae(pred, target);

    // |1-1.5| + |2-2| + |3-2.5| + |4-4.5| + |5-5| = 0.5 + 0 + 0.5 + 0.5 + 0 = 1.5
    // Mean = 1.5 / 5 = 0.3
    REQUIRE_THAT(mae, WithinAbs(0.3f, 1e-5));
}

TEST_CASE("Metrics::rmse", "[metrics][regression]") {
    auto pred = torch::tensor({1.0f, 2.0f, 3.0f});
    auto target = torch::tensor({2.0f, 2.0f, 4.0f});

    float rmse = Metrics::rmse(pred, target);

    // (1-2)^2 + (2-2)^2 + (3-4)^2 = 1 + 0 + 1 = 2
    // Mean = 2/3, sqrt = 0.8165
    REQUIRE_THAT(rmse, WithinAbs(0.8165f, 0.001));
}

TEST_CASE("Metrics::r_squared", "[metrics][regression]") {
    SECTION("perfect fit") {
        auto vals = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        REQUIRE(Metrics::r_squared(vals, vals) == 1.0f);
    }

    SECTION("good fit") {
        auto pred = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        auto target = torch::tensor({1.1f, 2.0f, 2.9f, 4.1f, 5.0f});

        float r2 = Metrics::r_squared(pred, target);
        REQUIRE_THAT(r2, WithinAbs(0.99f, 0.02));
    }

    SECTION("poor fit") {
        auto pred = torch::tensor({5.0f, 4.0f, 3.0f, 2.0f, 1.0f});  // Inverted
        auto target = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        float r2 = Metrics::r_squared(pred, target);
        REQUIRE(r2 < 0.0f);  // Negative R² for worse-than-mean predictions
    }

    SECTION("constant target") {
        auto pred = torch::tensor({1.0f, 1.0f, 1.0f});
        auto target = torch::tensor({1.0f, 1.0f, 1.0f});

        float r2 = Metrics::r_squared(pred, target);
        REQUIRE(r2 == 1.0f);  // Edge case handling
    }
}

TEST_CASE("Metrics::smape", "[metrics][regression]") {
    auto pred = torch::tensor({100.0f, 200.0f, 300.0f});
    auto target = torch::tensor({110.0f, 200.0f, 280.0f});

    float smape = Metrics::smape(pred, target);

    // SMAPE formula: |pred - target| / (|pred| + |target|)
    // (10/210 + 0/400 + 20/580) / 3
    REQUIRE(smape > 0.0f);
    REQUIRE(smape < 0.1f);  // Should be small for close predictions
}

TEST_CASE("Metrics::band_accuracy", "[metrics][regression]") {
    auto pred = torch::tensor({100.0f, 200.0f, 300.0f, 400.0f});
    auto target = torch::tensor({100.0f, 180.0f, 250.0f, 500.0f});

    SECTION("25% band") {
        // 100/100 = 1.0 (in band)
        // 200/180 = 1.11 (in band, within 25%)
        // 300/250 = 1.2 (in band, within 25%)
        // 400/500 = 0.8 (in band, within 25%)
        float acc = Metrics::band_accuracy(pred, target, 0.25f);
        REQUIRE_THAT(acc, WithinAbs(1.0f, 0.01));
    }

    SECTION("10% band") {
        // Fewer predictions should be in a tighter band
        float acc = Metrics::band_accuracy(pred, target, 0.10f);
        REQUIRE(acc < 1.0f);
    }
}

// ============================================================================
// Classification Metrics Tests
// ============================================================================

TEST_CASE("Metrics::accuracy", "[metrics][classification]") {
    // 3 classes, batch of 5
    auto pred = torch::tensor({
        {0.9f, 0.05f, 0.05f},  // pred: 0
        {0.1f, 0.8f, 0.1f},    // pred: 1
        {0.1f, 0.1f, 0.8f},    // pred: 2
        {0.7f, 0.2f, 0.1f},    // pred: 0 (wrong)
        {0.1f, 0.7f, 0.2f}     // pred: 1
    });
    auto target = torch::tensor(std::vector<int64_t>{0, 1, 2, 1, 1});

    float acc = Metrics::accuracy(pred, target);

    // Correct: 0, 1, 2, 1 (4 out of 5)
    REQUIRE_THAT(acc, WithinAbs(0.8f, 1e-5));
}

TEST_CASE("Metrics::confusion_matrix", "[metrics][classification]") {
    auto pred = torch::tensor({
        {0.9f, 0.1f},  // pred: 0
        {0.2f, 0.8f},  // pred: 1
        {0.6f, 0.4f},  // pred: 0
        {0.3f, 0.7f}   // pred: 1
    });
    auto target = torch::tensor(std::vector<int64_t>{0, 1, 1, 0});  // true: 0, 1, 1, 0

    auto cm = Metrics::confusion_matrix(pred, target, 2);

    // Confusion matrix:
    //           Pred 0  Pred 1
    // True 0      1       1
    // True 1      1       1

    REQUIRE(cm[0][0].item<int64_t>() == 1);  // TP for class 0
    REQUIRE(cm[0][1].item<int64_t>() == 1);  // FN for class 0 (pred as 1)
    REQUIRE(cm[1][0].item<int64_t>() == 1);  // FN for class 1 (pred as 0)
    REQUIRE(cm[1][1].item<int64_t>() == 1);  // TP for class 1
}

TEST_CASE("Metrics::classification_metrics", "[metrics][classification]") {
    // Perfect predictions for class 0, all wrong for class 1
    auto pred = torch::tensor({
        {0.9f, 0.1f},  // pred: 0, true: 0 ✓
        {0.9f, 0.1f},  // pred: 0, true: 0 ✓
        {0.9f, 0.1f},  // pred: 0, true: 1 ✗
        {0.9f, 0.1f}   // pred: 0, true: 1 ✗
    });
    auto target = torch::tensor(std::vector<int64_t>{0, 0, 1, 1});

    auto metrics = Metrics::classification_metrics(pred, target, 2);

    SECTION("accuracy") {
        REQUIRE_THAT(metrics.accuracy, WithinAbs(0.5f, 1e-5));  // 2/4
    }

    SECTION("per-class support") {
        REQUIRE(metrics.per_class_support[0] == 2);
        REQUIRE(metrics.per_class_support[1] == 2);
    }

    SECTION("per-class precision") {
        // Class 0: 2 TP, 2 FP -> precision = 2/4 = 0.5
        REQUIRE_THAT(metrics.per_class_precision[0], WithinAbs(0.5f, 1e-5));
        // Class 1: 0 TP -> precision = 0
        REQUIRE(metrics.per_class_precision[1] == 0.0f);
    }

    SECTION("per-class recall") {
        // Class 0: 2 TP, 0 FN -> recall = 1.0
        REQUIRE_THAT(metrics.per_class_recall[0], WithinAbs(1.0f, 1e-5));
        // Class 1: 0 TP, 2 FN -> recall = 0
        REQUIRE(metrics.per_class_recall[1] == 0.0f);
    }

    SECTION("macro F1") {
        // F1 class 0 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 0.667
        // F1 class 1 = 0
        // Macro F1 = (0.667 + 0) / 2 = 0.333
        REQUIRE(metrics.macro_f1 > 0.0f);
        REQUIRE(metrics.macro_f1 < 0.5f);
    }
}

TEST_CASE("Metrics::macro_f1 is mean of per-class F1, not F1 of averaged P/R",
          "[metrics][classification]") {
    // Samples: (true,pred) = (0,0), (0,1), (1,1).
    //   class 0: P=1.0, R=0.5 -> F1 = 0.667
    //   class 1: P=0.5, R=1.0 -> F1 = 0.667
    // Correct macro-F1 = mean(F1) = 0.667.
    // The old F1-of-averaged-P/R gave 2*0.75*0.75/1.5 = 0.75 (distinguishable).
    auto pred = torch::tensor({
        {0.9f, 0.1f},  // -> 0
        {0.1f, 0.9f},  // -> 1
        {0.1f, 0.9f}   // -> 1
    });
    auto target = torch::tensor(std::vector<int64_t>{0, 0, 1});
    auto m = Metrics::classification_metrics(pred, target, 2);

    REQUIRE_THAT(m.per_class_f1[0], WithinAbs(0.6667f, 1e-3));
    REQUIRE_THAT(m.per_class_f1[1], WithinAbs(0.6667f, 1e-3));
    // Mean of per-class F1, NOT the 0.75 the F1-of-means formula produced.
    REQUIRE_THAT(m.macro_f1, WithinAbs(0.6667f, 1e-3));
}

TEST_CASE("Metrics::r_squared on a constant target", "[metrics][regression]") {
    auto constant = torch::tensor({3.0f, 3.0f, 3.0f, 3.0f});

    SECTION("exact match -> 1.0") {
        REQUIRE_THAT(Metrics::r_squared(constant, constant), WithinAbs(1.0f, 1e-5));
    }
    SECTION("bad predictions -> 0.0, not a spurious 1.0") {
        auto pred = torch::tensor({1.0f, 2.0f, 4.0f, 5.0f});
        REQUIRE_THAT(Metrics::r_squared(pred, constant), WithinAbs(0.0f, 1e-5));
    }
}

// ============================================================================
// Confidence Metrics Tests
// ============================================================================

TEST_CASE("Metrics::accuracy_at_threshold", "[metrics][confidence]") {
    auto pred = torch::tensor({
        {0.9f, 0.1f},  // pred: 0, correct
        {0.6f, 0.4f},  // pred: 0, wrong (true: 1)
        {0.3f, 0.7f},  // pred: 1, correct
        {0.5f, 0.5f}   // pred: 0, wrong (true: 1)
    });
    auto target = torch::tensor(std::vector<int64_t>{0, 1, 1, 1});
    auto confidence = torch::tensor({0.9f, 0.6f, 0.7f, 0.5f});

    SECTION("threshold 0.0 includes all") {
        auto result = Metrics::accuracy_at_threshold(pred, target, confidence, 0.0f);
        REQUIRE(result.n_samples == 4);
        REQUIRE_THAT(result.coverage, WithinAbs(1.0f, 1e-5));
        REQUIRE_THAT(result.accuracy, WithinAbs(0.5f, 1e-5));  // 2/4
    }

    SECTION("threshold 0.65 filters low confidence") {
        auto result = Metrics::accuracy_at_threshold(pred, target, confidence, 0.65f);
        REQUIRE(result.n_samples == 2);  // Only 0.9 and 0.7
        REQUIRE_THAT(result.coverage, WithinAbs(0.5f, 1e-5));
        REQUIRE_THAT(result.accuracy, WithinAbs(1.0f, 1e-5));  // Both correct
    }

    SECTION("threshold 1.0 includes none") {
        auto result = Metrics::accuracy_at_threshold(pred, target, confidence, 1.0f);
        REQUIRE(result.n_samples == 0);
        REQUIRE(result.coverage == 0.0f);
    }
}

TEST_CASE("Metrics::accuracy_coverage_curve", "[metrics][confidence]") {
    auto pred = torch::tensor({
        {0.9f, 0.1f},
        {0.6f, 0.4f},
        {0.3f, 0.7f}
    });
    auto target = torch::tensor(std::vector<int64_t>{0, 0, 1});
    auto confidence = torch::tensor({0.9f, 0.6f, 0.7f});

    std::vector<float> thresholds = {0.0f, 0.5f, 0.8f, 1.0f};
    auto curve = Metrics::accuracy_coverage_curve(pred, target, confidence, thresholds);

    REQUIRE(curve.size() == 4);

    // Coverage should decrease as threshold increases
    REQUIRE(curve[0].coverage >= curve[1].coverage);
    REQUIRE(curve[1].coverage >= curve[2].coverage);
    REQUIRE(curve[2].coverage >= curve[3].coverage);
}

// ============================================================================
// PhasedLoss Tests
// ============================================================================

TEST_CASE("PhasedLoss phase boundaries", "[loss]") {
    PhasedLoss loss({100, 300});

    REQUIRE(loss.get_phase(0) == 1);
    REQUIRE(loss.get_phase(50) == 1);
    REQUIRE(loss.get_phase(99) == 1);
    REQUIRE(loss.get_phase(100) == 2);
    REQUIRE(loss.get_phase(200) == 2);
    REQUIRE(loss.get_phase(299) == 2);
    REQUIRE(loss.get_phase(300) == 3);
    REQUIRE(loss.get_phase(500) == 3);
}

TEST_CASE("PhasedLoss::from_config", "[loss]") {
    SECTION("MAE mode") {
        auto loss = PhasedLoss::from_config(LossConfigMode::MAE);
        // MAE mode should always be phase 1 (no SMAPE, no band)
        REQUIRE(loss.get_phase(0) == 1);
        REQUIRE(loss.get_phase(1000) == 1);
    }

    SECTION("Combined mode") {
        auto loss = PhasedLoss::from_config(LossConfigMode::Combined, {50, 150});
        REQUIRE(loss.get_phase(25) == 1);
        REQUIRE(loss.get_phase(100) == 2);
        REQUIRE(loss.get_phase(200) == 3);
    }
}

TEST_CASE("PhasedLoss regression_loss", "[loss]") {
    PhasedLoss loss({10, 20});

    auto pred = torch::tensor({1.0f, 2.0f, 3.0f});
    auto target = torch::tensor({1.1f, 2.0f, 2.9f});

    SECTION("phase 1 is pure MAE") {
        auto l1 = loss.regression_loss(pred, target, 5);
        auto expected_mae = torch::abs(pred - target).mean();
        REQUIRE_THAT(l1.item<float>(), WithinAbs(expected_mae.item<float>(), 1e-5));
    }

    SECTION("phase 2 adds SMAPE") {
        auto l2 = loss.regression_loss(pred, target, 15);
        auto mae = torch::abs(pred - target).mean();
        REQUIRE(l2.item<float>() >= mae.item<float>());  // Should be >= MAE alone
    }
}

TEST_CASE("PhasedLoss classification_loss", "[loss]") {
    PhasedLoss loss;

    auto pred = torch::tensor({
        {2.0f, 0.5f, 0.1f},
        {0.1f, 2.0f, 0.5f}
    });
    auto target = torch::tensor(std::vector<int64_t>{0, 1});

    SECTION("without class weights") {
        auto l = loss.classification_loss(pred, target);
        REQUIRE(l.item<float>() > 0.0f);
    }

    SECTION("with class weights") {
        auto weights = torch::tensor({1.0f, 2.0f, 1.0f});
        auto l = loss.classification_loss(pred, target, weights);
        REQUIRE(l.item<float>() > 0.0f);
    }
}

// ============================================================================
// Metrics::compute Integration Tests
// ============================================================================

TEST_CASE("Metrics::compute regression", "[metrics][integration]") {
    auto pred = torch::tensor({{1.0f}, {2.0f}, {3.0f}, {4.0f}});
    auto target = torch::tensor({1.1f, 2.0f, 2.9f, 4.1f});

    auto metrics = Metrics::compute(pred, target, TaskType::Regression);

    REQUIRE(metrics.count("mae") > 0);
    REQUIRE(metrics.count("rmse") > 0);
    REQUIRE(metrics.count("r2") > 0);
    REQUIRE(metrics.count("smape") > 0);
    REQUIRE(metrics.count("band_25") > 0);
    REQUIRE(metrics.count("band_50") > 0);
    REQUIRE(metrics.count("band_75") > 0);
}

TEST_CASE("Metrics::compute reports regression metrics in original units",
          "[metrics][regression][recovery]") {
    // Reproduces the training pipeline: original-units targets are log1p'd then
    // standardized, and the model predicts in that standardized-log space. The
    // reported metrics must be mapped back to original units via the scalers,
    // NOT left in standardized space (where mae/rmse are unitless and smape/band
    // take ratios of values straddling zero).
    auto y = torch::tensor({10.0f, 50.0f, 200.0f, 1000.0f, 5000.0f});  // e.g. plot areas
    auto t = torch::log1p(y);
    auto mean = t.mean();
    auto scale = t.std() + 1e-8f;
    auto t_std = (t - mean) / scale;          // == Trainer::test_targets_

    // Model prediction in standardized-log space with a small systematic offset.
    auto pred_std = t_std + 0.1f;

    auto metrics = Metrics::compute(pred_std.unsqueeze(1), t_std,
                                    TaskType::Regression, TransformType::Log1p,
                                    {0.25f, 0.50f, 0.75f}, /*num_classes=*/0,
                                    mean, scale);

    // Ground truth computed independently in original units.
    auto pred_orig = torch::expm1(pred_std * scale + mean);
    float expected_mae = torch::abs(pred_orig - y).mean().item<float>();
    float expected_rmse = torch::sqrt(torch::pow(pred_orig - y, 2).mean()).item<float>();

    REQUIRE_THAT(metrics["mae"], WithinRel(expected_mae, 1e-4f));
    REQUIRE_THAT(metrics["rmse"], WithinRel(expected_rmse, 1e-4f));

    // Original-units error is on the scale of the targets (hundreds+), not the
    // ~0.1 standardized-space number the pre-fix path reported.
    REQUIRE(metrics["mae"] > 1.0f);

    // smape and band are finite and within their natural ranges.
    REQUIRE(metrics["smape"] >= 0.0f);
    REQUIRE(metrics["smape"] <= 1.0f);
    REQUIRE(metrics["band_25"] >= 0.0f);
    REQUIRE(metrics["band_25"] <= 1.0f);

    // Contrast: omitting the scalers leaves the metrics in standardized-log
    // space, where the same inputs report a tiny, meaningless mae. This is the
    // pre-fix behavior and confirms the unscale is load-bearing.
    auto unscaled = Metrics::compute(pred_std.unsqueeze(1), t_std,
                                     TaskType::Regression, TransformType::Log1p,
                                     {0.25f}, 0);
    REQUIRE(unscaled["mae"] < 1.0f);
}

TEST_CASE("Metrics::compute classification", "[metrics][integration]") {
    auto pred = torch::tensor({
        {0.9f, 0.1f, 0.0f},
        {0.1f, 0.8f, 0.1f},
        {0.0f, 0.1f, 0.9f}
    });
    auto target = torch::tensor(std::vector<int64_t>{0, 1, 2});

    auto metrics = Metrics::compute(pred, target, TaskType::Classification,
                                    TransformType::None, {}, 3);

    REQUIRE(metrics.count("accuracy") > 0);
    REQUIRE(metrics.count("macro_f1") > 0);
    REQUIRE(metrics.count("weighted_f1") > 0);
    REQUIRE(metrics.count("precision_0") > 0);
    REQUIRE(metrics.count("recall_0") > 0);
    REQUIRE(metrics.count("f1_0") > 0);

    // All predictions correct
    REQUIRE_THAT(metrics["accuracy"], WithinAbs(1.0f, 1e-5));
}

// ============================================================================
// MultiTaskLoss accumulator seeding
// ============================================================================

TEST_CASE("MultiTaskLoss::compute seeds a deterministic scalar accumulator",
          "[loss][multitask]") {
    // The accumulator used to be seeded from predictions.begin()->second
    // (an arbitrary head from an unordered_map) with shape {1}. It is now a
    // true scalar on a device chosen deterministically from the first target
    // in the configured order.
    std::vector<TargetConfig> targets;
    TargetConfig a; a.name = "a"; a.task = TaskType::Classification;
    a.num_classes = 3; a.weight = 1.0f;
    TargetConfig b; b.name = "b"; b.task = TaskType::Classification;
    b.num_classes = 3; b.weight = 1.0f;
    targets.push_back(a);
    targets.push_back(b);

    MultiTaskLoss loss(targets);

    torch::manual_seed(0);
    std::unordered_map<std::string, torch::Tensor> preds;
    preds["a"] = torch::randn({4, 3});
    preds["b"] = torch::randn({4, 3});
    std::unordered_map<std::string, torch::Tensor> tgts;
    tgts["a"] = torch::randint(0, 3, {4}, torch::kInt64);
    tgts["b"] = torch::randint(0, 3, {4}, torch::kInt64);

    auto [total, individual] = loss.compute(preds, tgts, /*epoch=*/0);

    // Seed is a true scalar (shape {}), not {1}.
    REQUIRE(total.dim() == 0);
    REQUIRE(total.device().is_cpu());
    REQUIRE(total.dtype() == torch::kFloat32);

    // Total equals the weighted sum of the per-target losses (weights = 1).
    float expected = individual["a"].item<float>() + individual["b"].item<float>();
    REQUIRE_THAT(total.item<float>(), WithinAbs(expected, 1e-5));

    // Gradient flows through the scalar accumulator.
    preds["a"].requires_grad_(true);
    preds["b"].requires_grad_(true);
    auto [total2, ignored] = loss.compute(preds, tgts, /*epoch=*/0);
    total2.backward();
    REQUIRE(preds["a"].grad().defined());
    REQUIRE(preds["b"].grad().defined());
}
