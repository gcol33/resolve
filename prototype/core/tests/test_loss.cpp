#include <catch2/catch.hpp>

#include <resolve/loss.hpp>
#include <resolve/types.hpp>

using namespace resolve;

TEST_CASE("PhaseConfig construction", "[loss]") {

    SECTION("default construction") {
        PhaseConfig config;
        REQUIRE(config.mae == 0.0f);
        REQUIRE(config.mse == 0.0f);
        REQUIRE(config.huber == 0.0f);
        REQUIRE(config.smape == 0.0f);
        REQUIRE(config.band == 0.0f);
    }

    SECTION("custom construction") {
        PhaseConfig config{0.5f, 0.3f, 0.0f, 0.1f, 0.1f};
        REQUIRE(config.mae == 0.5f);
        REQUIRE(config.mse == 0.3f);
        REQUIRE(config.smape == 0.1f);
        REQUIRE(config.band == 0.1f);
    }
}

TEST_CASE("PhasedLoss computation", "[loss]") {

    SECTION("single phase MAE loss") {
        std::vector<PhaseConfig> phases = {
            {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}  // Pure MAE
        };
        std::vector<int> boundaries = {};

        PhasedLoss loss_fn(phases, boundaries, TransformType::None);

        auto pred = torch::tensor({1.0f, 2.0f, 3.0f});
        auto target = torch::tensor({1.0f, 3.0f, 5.0f});

        auto loss = loss_fn(pred, target, 0);

        // MAE = (0 + 1 + 2) / 3 = 1.0
        REQUIRE(loss.item<float>() == Approx(1.0f));
    }

    SECTION("single phase MSE loss") {
        std::vector<PhaseConfig> phases = {
            {0.0f, 1.0f, 0.0f, 0.0f, 0.0f}  // Pure MSE
        };
        std::vector<int> boundaries = {};

        PhasedLoss loss_fn(phases, boundaries, TransformType::None);

        auto pred = torch::tensor({1.0f, 2.0f, 3.0f});
        auto target = torch::tensor({1.0f, 3.0f, 5.0f});

        auto loss = loss_fn(pred, target, 0);

        // MSE = (0 + 1 + 4) / 3 = 1.667
        REQUIRE(loss.item<float>() == Approx(5.0f / 3.0f).epsilon(0.01));
    }

    SECTION("multi-phase loss transitions") {
        std::vector<PhaseConfig> phases = {
            {1.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Phase 1: Pure MAE
            {0.0f, 1.0f, 0.0f, 0.0f, 0.0f},  // Phase 2: Pure MSE
        };
        std::vector<int> boundaries = {5};  // Switch at epoch 5

        PhasedLoss loss_fn(phases, boundaries, TransformType::None);

        auto pred = torch::tensor({1.0f, 2.0f, 3.0f});
        auto target = torch::tensor({1.0f, 3.0f, 5.0f});

        // Epoch 0 should use MAE
        auto loss_epoch0 = loss_fn(pred, target, 0);
        REQUIRE(loss_epoch0.item<float>() == Approx(1.0f));

        // Epoch 10 should use MSE
        auto loss_epoch10 = loss_fn(pred, target, 10);
        REQUIRE(loss_epoch10.item<float>() == Approx(5.0f / 3.0f).epsilon(0.01));
    }
}

TEST_CASE("MultiTaskLoss computation", "[loss]") {

    std::vector<TargetConfig> targets = {
        {"ph", TaskType::Regression, 1, TransformType::None},
        {"nitrogen", TaskType::Regression, 1, TransformType::Log1p},
    };

    std::vector<PhaseConfig> phases = {
        {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}  // Pure MAE
    };

    MultiTaskLoss loss_fn(targets, phases, {});

    SECTION("computes loss for all targets") {
        std::unordered_map<std::string, torch::Tensor> preds = {
            {"ph", torch::tensor({{5.0f}, {6.0f}, {7.0f}})},
            {"nitrogen", torch::tensor({{1.0f}, {2.0f}, {3.0f}})},
        };

        std::unordered_map<std::string, torch::Tensor> targets_map = {
            {"ph", torch::tensor({5.0f, 7.0f, 8.0f})},
            {"nitrogen", torch::tensor({1.5f, 2.5f, 3.5f})},
        };

        auto [loss, task_losses] = loss_fn(preds, targets_map, 0);

        REQUIRE(loss.item<float>() > 0.0f);
        REQUIRE(task_losses.count("ph") == 1);
        REQUIRE(task_losses.count("nitrogen") == 1);
    }
}

TEST_CASE("Metrics computation", "[metrics]") {

    SECTION("band_accuracy") {
        auto pred = torch::tensor({100.0f, 200.0f, 300.0f, 400.0f});
        auto target = torch::tensor({100.0f, 180.0f, 400.0f, 600.0f});

        // 25% band: |pred - target| / |target| <= 0.25
        // 100 vs 100: 0% error - in band
        // 200 vs 180: 11% error - in band
        // 300 vs 400: 25% error - in band
        // 400 vs 600: 33% error - out of band
        float accuracy = Metrics::band_accuracy(pred, target, 0.25f);
        REQUIRE(accuracy == Approx(0.75f));
    }

    SECTION("mae") {
        auto pred = torch::tensor({1.0f, 2.0f, 3.0f});
        auto target = torch::tensor({1.0f, 3.0f, 5.0f});

        float mae = Metrics::mae(pred, target);
        REQUIRE(mae == Approx(1.0f));
    }

    SECTION("rmse") {
        auto pred = torch::tensor({1.0f, 2.0f, 3.0f});
        auto target = torch::tensor({1.0f, 3.0f, 5.0f});

        float rmse = Metrics::rmse(pred, target);
        // sqrt((0 + 1 + 4) / 3) = sqrt(5/3) ≈ 1.29
        REQUIRE(rmse == Approx(std::sqrt(5.0f / 3.0f)).epsilon(0.01));
    }

    SECTION("smape") {
        auto pred = torch::tensor({100.0f, 200.0f});
        auto target = torch::tensor({110.0f, 180.0f});

        float smape = Metrics::smape(pred, target);
        // SMAPE = mean(|pred - target| / ((|pred| + |target|) / 2))
        // = mean(10/105, 20/190) = mean(0.095, 0.105) ≈ 0.1
        REQUIRE(smape > 0.0f);
        REQUIRE(smape < 1.0f);
    }

    SECTION("compute_regression returns all metrics") {
        auto pred = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        auto target = torch::tensor({1.1f, 2.2f, 2.8f, 4.5f, 4.8f});

        auto metrics = Metrics::compute_regression(pred, target);

        REQUIRE(metrics.count("mae") == 1);
        REQUIRE(metrics.count("rmse") == 1);
        REQUIRE(metrics.count("smape") == 1);
        REQUIRE(metrics.count("band_10") == 1);
        REQUIRE(metrics.count("band_25") == 1);
        REQUIRE(metrics.count("band_50") == 1);
    }
}

TEST_CASE("Classification metrics", "[metrics]") {

    SECTION("accuracy") {
        // Logits (batch_size=4, n_classes=3)
        auto logits = torch::tensor({
            {2.0f, 1.0f, 0.5f},  // pred: class 0
            {0.5f, 2.0f, 1.0f},  // pred: class 1
            {1.0f, 0.5f, 2.0f},  // pred: class 2
            {2.0f, 0.5f, 1.0f},  // pred: class 0
        });
        auto targets = torch::tensor({0, 1, 2, 1});  // correct: 0, 1, 2; wrong: 3

        auto metrics = Metrics::compute_classification(logits, targets);

        REQUIRE(metrics.count("accuracy") == 1);
        REQUIRE(metrics["accuracy"] == Approx(0.75f));
    }
}
