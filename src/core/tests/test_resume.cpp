// Trainer::resume continues a run from a checkpoint fit() wrote while it
// trained. On the CPU a resumed run is the uninterrupted run, bit for bit:
// same loss histories, best epoch, metrics and final weights, whether it
// continues from a periodic checkpoint or from best.pt, under early stopping
// or a fixed duration, with dropout drawing from the restored RNG streams.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

// 60 plots, y = 3 cov1 - cov2 + noise-free offset, one of eight species.
ResolveDataset resume_dataset() {
    std::ostringstream hdr, spc;
    hdr << "plot_id,lat,lon,cov1,cov2,y\n";
    spc << "plot_id,sp,cover\n";
    for (int i = 0; i < 60; ++i) {
        const double c1 = std::sin(0.37 * i);
        const double c2 = std::cos(0.11 * i);
        hdr << "P" << i << "," << (45.0 + 0.01 * i) << "," << (7.0 + 0.02 * i) << ","
            << c1 << "," << c2 << "," << (3.0 * c1 - c2 + 1.0) << "\n";
        spc << "P" << i << ",sp" << (i % 8) << "," << (1 + i % 4) << "\n";
    }
    const TempFile header(hdr.str()), species(spc.str());
    return ResolveDataset::from_csv(header.path(), species.path(),
                                    Roles().coordinates().covariates({"cov1", "cov2"}),
                                    {TargetSpec::regression("y")}, hash_dataset_config());
}

// Dropout on, so the continued epochs draw from the global RNG stream.
ModelConfig resume_model_config() {
    ModelConfig cfg = hash_mlp_config({16, 8});
    cfg.dropout = 0.3f;
    return cfg;
}

TrainConfig resume_train_config(const std::string& dir, torch::Device device = torch::kCPU) {
    TrainConfig cfg = quiet_train_config(8);
    cfg.patience = 50;
    cfg.checkpoint_dir = dir;
    cfg.checkpoint_every = 2;
    cfg.lr_scheduler = LRSchedulerType::CosineAnnealing;
    cfg.device = device;
    return cfg;
}

std::unique_ptr<Trainer> make_trainer(const ResolveDataset& ds, const TrainConfig& tc,
                                      uint64_t init_seed) {
    torch::manual_seed(init_seed);
    return std::make_unique<Trainer>(ResolveModel(ds.schema(), resume_model_config()), tc);
}

std::vector<torch::Tensor> parameters_of(Trainer& trainer) {
    std::vector<torch::Tensor> out;
    for (const auto& p : trainer.model()->parameters()) out.push_back(p.detach().cpu().clone());
    return out;
}

void require_same_run(const TrainResult& a, const TrainResult& b) {
    REQUIRE(a.train_loss_history == b.train_loss_history);
    REQUIRE(a.test_loss_history == b.test_loss_history);
    REQUIRE(a.lr_history == b.lr_history);
    REQUIRE(a.selection_history == b.selection_history);
    REQUIRE((a.train_loss_history_by_target == b.train_loss_history_by_target));
    REQUIRE((a.test_loss_history_by_target == b.test_loss_history_by_target));
    REQUIRE(a.best_epoch == b.best_epoch);
    REQUIRE((a.final_metrics == b.final_metrics));
}

void require_same_weights(Trainer& a, Trainer& b) {
    const auto pa = parameters_of(a);
    const auto pb = parameters_of(b);
    REQUIRE(pa.size() == pb.size());
    for (size_t i = 0; i < pa.size(); ++i) {
        REQUIRE(torch::equal(pa[i], pb[i]));
    }
}

// The uninterrupted run, and a fresh trainer resumed from `checkpoint` of it.
struct ResumedRun {
    std::unique_ptr<Trainer> full_trainer;
    std::unique_ptr<Trainer> resumed_trainer;
    TrainResult full;
    TrainResult resumed;
};

ResumedRun run_and_resume(const ResolveDataset& ds, const TrainConfig& tc,
                          const std::string& checkpoint) {
    ResumedRun run;
    run.full_trainer = make_trainer(ds, tc, 1);
    run.full_trainer->prepare_data(ds, 0.25f, 17);
    run.full = run.full_trainer->fit();

    // A different initialisation: the weights have to come from the checkpoint.
    run.resumed_trainer = make_trainer(ds, tc, 2);
    run.resumed_trainer->prepare_data(ds, 0.25f, 17);
    run.resumed = run.resumed_trainer->resume(checkpoint);
    return run;
}

}  // namespace

TEST_CASE("A run resumed from a periodic checkpoint is the uninterrupted run",
          "[trainer][resume]") {
    const auto ds = resume_dataset();
    const TempDir dir;
    const TrainConfig tc = resume_train_config(dir.path());

    const auto run = run_and_resume(ds, tc, dir.file("checkpoint_4.pt"));

    REQUIRE(run.full.resumed_from_epoch == 0);
    REQUIRE(run.resumed.resumed_from_epoch == 4);
    REQUIRE(run.resumed.train_loss_history.size() == 8);
    require_same_run(run.full, run.resumed);
    require_same_weights(*run.full_trainer, *run.resumed_trainer);
    REQUIRE(run.resumed.baselines.at("y").model_mae == run.full.baselines.at("y").model_mae);
}

TEST_CASE("A run resumed from best.pt replays the epochs after its best",
          "[trainer][resume]") {
    const auto ds = resume_dataset();
    const TempDir dir;
    const TrainConfig tc = resume_train_config(dir.path());

    const auto run = run_and_resume(ds, tc, dir.file("best.pt"));

    // best.pt is the last new best, so the run continues at the epoch after it.
    REQUIRE(run.resumed.resumed_from_epoch == run.full.best_epoch + 1);
    require_same_run(run.full, run.resumed);
    require_same_weights(*run.full_trainer, *run.resumed_trainer);
}

TEST_CASE("A run under a reduce-on-plateau schedule resumes with the schedule's state",
          "[trainer][resume][lr_schedule]") {
    // No epoch can beat its best by the threshold, so the rate decays on a
    // fixed rhythm (two bad epochs, a decay, a cooldown epoch): checkpoint_6
    // is written right after the second decay with its cooldown pending, and
    // the epochs after it only match the uninterrupted run if the best, the
    // bad-epoch count, the cooldown and the rate all come back.
    const auto ds = resume_dataset();
    const TempDir dir;
    TrainConfig tc = resume_train_config(dir.path());
    tc.max_epochs = 12;
    tc.lr = 1e-2f;
    tc.lr_scheduler = LRSchedulerType::ReduceOnPlateau;
    tc.lr_plateau_factor = 0.5f;
    tc.lr_plateau_patience = 1;
    tc.lr_plateau_threshold = 0.999f;
    tc.lr_plateau_cooldown = 1;

    const auto run = run_and_resume(ds, tc, dir.file("checkpoint_6.pt"));

    REQUIRE(run.resumed.resumed_from_epoch == 6);
    REQUIRE(run.full.lr_history.size() == 12);
    REQUIRE(run.full.lr_history.front() == 1e-2f);
    REQUIRE(run.full.lr_history.back() < run.full.lr_history[6]);
    require_same_run(run.full, run.resumed);
    require_same_weights(*run.full_trainer, *run.resumed_trainer);
}

TEST_CASE("A run selecting by accuracy resumes selecting by accuracy",
          "[trainer][resume][selection_metric]") {
    // Selected by the held-out accuracy, with a plateau in mode max: the
    // epochs after checkpoint_6 only match the uninterrupted run if the best
    // accuracy (not a loss), its direction and the schedule's state come back.
    const ClassificationCorpus corpus;
    const auto ds =
        corpus.load({TargetSpec::classification("cls", ClassificationCorpus::kClasses)});
    const TempDir dir;
    TrainConfig tc = resume_train_config(dir.path());
    tc.max_epochs = 12;
    tc.lr = 5e-2f;
    tc.selection_metric = SelectionMetric::Accuracy;
    tc.lr_scheduler = LRSchedulerType::ReduceOnPlateau;
    tc.lr_plateau_factor = 0.5f;
    tc.lr_plateau_patience = 1;
    tc.lr_plateau_threshold = 1e-4f;

    const auto run = run_and_resume(ds, tc, dir.file("checkpoint_6.pt"));

    REQUIRE(run.resumed.resumed_from_epoch == 6);
    REQUIRE(run.full.selection_history.size() == 12);
    REQUIRE(run.resumed.selection_history == run.full.selection_history);
    REQUIRE(run.full.selection_history != run.full.test_loss_history);
    require_same_run(run.full, run.resumed);
    require_same_weights(*run.full_trainer, *run.resumed_trainer);

    SECTION("resuming it selecting by the loss is refused") {
        TrainConfig by_loss = tc;
        by_loss.selection_metric = SelectionMetric::Loss;
        auto trainer = make_trainer(ds, by_loss, 2);
        trainer->prepare_data(ds, 0.25f, 17);
        REQUIRE_THROWS_WITH(trainer->resume(dir.file("checkpoint_6.pt")),
                            Catch::Matchers::ContainsSubstring("selection metric"));
    }
}

TEST_CASE("A fixed-duration run without a held-out fold resumes",
          "[trainer][resume][fixed_epochs]") {
    const auto ds = resume_dataset();
    const TempDir dir;
    TrainConfig tc = resume_train_config(dir.path());
    tc.fixed_epochs = 6;

    auto full = make_trainer(ds, tc, 1);
    full->prepare_data(ds, /*test_size=*/0.0f, 17);
    const auto full_result = full->fit();

    auto resumed = make_trainer(ds, tc, 2);
    resumed->prepare_data(ds, /*test_size=*/0.0f, 17);
    const auto resumed_result = resumed->resume(dir.file("checkpoint_2.pt"));

    REQUIRE(resumed_result.resumed_from_epoch == 2);
    REQUIRE(resumed_result.train_loss_history == full_result.train_loss_history);
    REQUIRE(resumed_result.best_epoch == 5);
    require_same_weights(*full, *resumed);
}

TEST_CASE("resume refuses what it cannot continue", "[trainer][resume]") {
    const auto ds = resume_dataset();
    const TempDir dir;
    const TrainConfig tc = resume_train_config(dir.path());
    {
        auto full = make_trainer(ds, tc, 1);
        full->prepare_data(ds, 0.25f, 17);
        full->fit();
    }

    SECTION("nothing prepared") {
        auto trainer = make_trainer(ds, tc, 2);
        REQUIRE_THROWS_AS(trainer->resume(dir.file("checkpoint_4.pt")), std::runtime_error);
    }
    SECTION("a final checkpoint carries no run progress") {
        auto trainer = make_trainer(ds, tc, 2);
        trainer->prepare_data(ds, 0.25f, 17);
        REQUIRE_THROWS_WITH(trainer->resume(dir.file("checkpoint.pt")),
                            Catch::Matchers::ContainsSubstring("no run progress"));
    }
    SECTION("a different split") {
        auto trainer = make_trainer(ds, tc, 2);
        trainer->prepare_data(ds, 0.25f, /*seed=*/18);
        REQUIRE_THROWS_WITH(trainer->resume(dir.file("checkpoint_4.pt")),
                            Catch::Matchers::ContainsSubstring("split differs"));
    }
}

TEST_CASE("A run resumed on the GPU continues the uninterrupted run",
          "[trainer][resume][cuda]") {
    if (!torch::cuda::is_available()) {
        SKIP("needs a CUDA device");
    }
    const auto ds = resume_dataset();
    const TempDir dir;
    const TrainConfig tc = resume_train_config(dir.path(), torch::kCUDA);

    const auto run = run_and_resume(ds, tc, dir.file("checkpoint_4.pt"));

    REQUIRE(run.resumed.resumed_from_epoch == 4);
    REQUIRE(run.resumed.train_loss_history.size() == run.full.train_loss_history.size());
    // The epochs before the checkpoint are the run's own record.
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(run.resumed.train_loss_history[i] == run.full.train_loss_history[i]);
    }
    // The continued epochs follow the same trajectory (CUDA kernels need not
    // reduce in a fixed order, so equal up to rounding).
    for (size_t i = 4; i < run.full.train_loss_history.size(); ++i) {
        REQUIRE(std::abs(run.resumed.train_loss_history[i] - run.full.train_loss_history[i]) <=
                1e-3f * (1.0f + std::abs(run.full.train_loss_history[i])));
    }
}
