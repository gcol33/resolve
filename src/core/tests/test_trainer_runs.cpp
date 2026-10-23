// What a Trainer carries from one call to the next: a new prepare_data
// replaces every member of the last one (on the host and in the GPU mirrors),
// load_state re-expresses a prepared split in the loaded scalers, every split
// goes through one apply_split with one set of guards, cross-validation hands
// back the trainer's own run record, the held-out forward is chunked without
// changing a prediction, and the baselines score the model exactly as
// final_metrics does.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/target_labels.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

constexpr int kClasses = 3;

// Plot i ("P<i>") sits at lat 40 + 0.01 i, lon -5 + 0.01 i (or at the cell
// `cell(i)` gives, when set), with cov1 = sin(0.3 i), y = exp(cov1) + 2 (blank
// where `labelled(i)` is false) and class i % 3; one of eight species.
struct Corpus {
    TempFile header;
    TempFile species;
};

Corpus write_corpus(int n_plots, const std::function<bool(int)>& labelled,
                    const std::function<std::pair<double, double>(int)>& cell = {}) {
    std::ostringstream hdr, spc;
    hdr << "plot_id,lat,lon,cov1,y,cls\n";
    spc << "plot_id,sp,cover\n";
    for (int i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(0.3 * i);
        const auto [lat, lon] = cell ? cell(i) : std::pair{40.0 + 0.01 * i, -5.0 + 0.01 * i};
        hdr << "P" << i << "," << lat << "," << lon << "," << c1 << ",";
        if (labelled(i)) hdr << (std::exp(c1) + 2.0);
        hdr << "," << (i % kClasses) << "\n";
        spc << "P" << i << ",sp" << (i % 8) << "," << (1 + i % 3) << "\n";
    }
    return {TempFile(hdr.str()), TempFile(spc.str())};
}

TargetSpec y_spec(TargetMissingPolicy missing, TransformType transform = TransformType::None) {
    TargetSpec spec = TargetSpec::regression("y", transform);
    spec.missing = missing;
    return spec;
}

ResolveDataset load_corpus(const Corpus& corpus, std::vector<TargetSpec> targets) {
    return ResolveDataset::from_csv(corpus.header.path(), corpus.species.path(),
                                    Roles().coordinates().covariates({"cov1"}),
                                    std::move(targets), hash_dataset_config());
}

ResolveDataset labelled_corpus(int n_plots, std::vector<TargetSpec> targets) {
    const Corpus corpus = write_corpus(n_plots, [](int) { return true; });
    return load_corpus(corpus, std::move(targets));
}

Trainer make_trainer(const ResolveDataset& ds, const TrainConfig& tc, uint64_t seed = 11) {
    torch::manual_seed(seed);
    return Trainer(ResolveModel(ds.schema(), hash_mlp_config({16, 8})), tc);
}

// The labelled plots among the IDs, under the "labelled on even plots" rule.
int64_t count_even(const std::vector<std::string>& ids) {
    int64_t n = 0;
    for (const auto& id : ids) {
        if (std::stoi(id.substr(1)) % 2 == 0) ++n;
    }
    return n;
}

template <typename Fn>
std::string invalid_argument_message(Fn&& fn) {
    bool thrown = false;
    std::string message;
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        thrown = true;
        message = e.what();
    }
    REQUIRE(thrown);
    return message;
}

// The masked y beside the class target every plot carries, which keeps the
// plots y is missing on in the dataset.
std::vector<TargetSpec> masked_y_and_class() {
    return {y_spec(TargetMissingPolicy::Mask), TargetSpec::classification("cls", kClasses)};
}

// Two corpora on the same schema: 60 fully labelled plots, and 30 plots
// labelled on the even ones, for a masked target whose labelled count the
// metrics report.
struct TwoSplits {
    Corpus first_files = write_corpus(60, [](int) { return true; });
    Corpus second_files = write_corpus(30, [](int i) { return i % 2 == 0; });
    ResolveDataset first = load_corpus(first_files, masked_y_and_class());
    ResolveDataset second = load_corpus(second_files, masked_y_and_class());
};

// prepare_data(first), fit, prepare_data(second), fit on `device`: the second
// fit must train and score on the second split alone.
void check_second_split_wins(torch::Device device) {
    TwoSplits data;
    TrainConfig tc = quiet_train_config(2);
    tc.device = device;
    Trainer trainer = make_trainer(data.first, tc);

    trainer.prepare_data(data.first, 0.25f, /*seed=*/3);
    trainer.fit();

    trainer.prepare_data(data.second, 0.25f, /*seed=*/5);
    const auto result = trainer.fit();

    const auto test_ids = trainer.test_plot_ids();
    REQUIRE(test_ids.size() == 7);  // int(30 * 0.25)
    REQUIRE(trainer.run_metadata().n_plots_train + trainer.run_metadata().n_plots_test == 30);
    // The held-out metrics are the second fold's: its labelled count, not the
    // first fold's 15.
    REQUIRE(result.final_metrics.at("y").at(kLabelledMetric) ==
            static_cast<float>(count_even(test_ids)));
    REQUIRE(static_cast<int64_t>(trainer.compute_residuals("y").test_rows.size()) ==
            count_even(test_ids));
}

}  // namespace

// =============================================================================
// A new prepare_data replaces the last one
// =============================================================================

TEST_CASE("Two consecutive prepare_data calls train on the second split",
          "[trainer][prepare]") {
    check_second_split_wins(torch::kCPU);
}

TEST_CASE("Two consecutive prepare_data calls refresh the GPU mirror",
          "[trainer][prepare][cuda]") {
    if (!torch::cuda::is_available()) {
        SKIP("needs a CUDA device");
    }
    check_second_split_wins(torch::kCUDA);
}

TEST_CASE("The tensor prepare_data clears what only a dataset carries",
          "[trainer][prepare]") {
    TwoSplits data;
    Trainer trainer = make_trainer(data.first, quiet_train_config());
    trainer.prepare_data(data.first, 0.25f, 3);
    REQUIRE_FALSE(trainer.test_plot_ids().empty());

    trainer.prepare_data(data.second.continuous_inputs(), data.second.plot_inputs(),
                         data.second.targets(), 0.25f, 5);
    // Tensors carry no plot IDs: the first dataset's would name the wrong plots.
    REQUIRE(trainer.test_plot_ids().empty());
    REQUIRE(trainer.train_plot_ids().empty());
    REQUIRE(trainer.test_indices().numel() == 7);
}

TEST_CASE("load_state re-expresses a prepared split in the loaded scalers",
          "[trainer][load_state]") {
    const auto ds = labelled_corpus(80, {y_spec(TargetMissingPolicy::Drop)});
    const TempPath ckpt(".pt");

    // A trained model with the scalers of seed 1's split.
    Trainer source = make_trainer(ds, quiet_train_config(2));
    source.prepare_data(ds, 0.25f, /*seed=*/1);
    source.fit();
    source.save(ckpt.path());

    // A trainer prepared on another split (other scalers), then loaded.
    Trainer loaded = make_trainer(ds, quiet_train_config(2), /*seed=*/99);
    loaded.prepare_data(ds, 0.25f, /*seed=*/2);
    loaded.fit();
    loaded.load_state(ckpt.path());

    // Its residuals on its own fold are the source model's predictions there.
    const auto residuals = loaded.compute_residuals("y");
    Predictor predictor(source.model(), source.scalers(), torch::kCPU);
    const auto all = predictor.predict(ds, false, -1).predictions.at("y");
    const auto test_idx = loaded.test_indices();
    REQUIRE(static_cast<int64_t>(residuals.predictions.size()) == test_idx.numel());
    for (int64_t i = 0; i < test_idx.numel(); ++i) {
        const float expected = all[test_idx[i].item<int64_t>()].item<float>();
        REQUIRE(residuals.predictions[static_cast<size_t>(i)] ==
                Catch::Approx(expected).epsilon(1e-4).margin(1e-5));
    }
}

// =============================================================================
// One split step, one set of guards
// =============================================================================

TEST_CASE("A split with no fitting rows is refused", "[trainer][split]") {
    const auto ds = labelled_corpus(20, {y_spec(TargetMissingPolicy::Drop)});
    Trainer trainer = make_trainer(ds, quiet_train_config());
    const auto message =
        invalid_argument_message([&] { trainer.prepare_data(ds, /*test_size=*/1.0f, 1); });
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("no fitting rows"));
}

TEST_CASE("prepare_data and cross-validation refuse a split with no labelled fitting row",
          "[trainer][split][cv]") {
    // y is labelled on P0 and P1 only. P0 and P1 share one spatial block and
    // every other plot has a block of its own.
    const Corpus files = write_corpus(
        12, [](int i) { return i < 2; },
        [](int i) {
            return i < 2 ? std::pair{40.5, -4.5}
                         : std::pair{40.5 + 2.0 * i, -4.5 + 2.0 * i};
        });
    const auto ds = load_corpus(files, masked_y_and_class());

    SECTION("prepare_data, when the fold holds every labelled plot") {
        Trainer trainer = make_trainer(ds, quiet_train_config());
        const auto message = invalid_argument_message(
            [&] { trainer.prepare_data(ds, std::vector<int64_t>{0, 1}, 1); });
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("training plots"));
    }

    SECTION("cross-validation, in the fold that holds out the labelled block") {
        Trainer trainer = make_trainer(ds, quiet_train_config());
        trainer.prepare_data(ds, std::vector<int64_t>{1, 5}, 1);
        SpatialBlockConfig blocks;
        blocks.lat_size = 1.0f;
        blocks.lon_size = 1.0f;
        const auto message = invalid_argument_message(
            [&] { (void)trainer.cross_validate_spatial(blocks, /*n_folds=*/2, 1); });
        REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("training plots"));
    }
}

TEST_CASE("Cross-validation hands back the trainer's own run record",
          "[trainer][cv]") {
    const auto ds = labelled_corpus(40, {y_spec(TargetMissingPolicy::Drop)});

    SECTION("before any fit") {
        Trainer trainer = make_trainer(ds, quiet_train_config(1));
        trainer.prepare_data(ds, 0.25f, 4);
        (void)trainer.cross_validate(2, 4);
        // Each fold's fit() set them; the trainer itself has not trained.
        REQUIRE(trainer.effective_batch_size() == 0);
        REQUIRE(trainer.run_metadata().created_at.empty());
        const TempPath path(".pt");
        trainer.save(path.path());
        REQUIRE(Trainer::load_train_config(path.path()).batch_size == 8);
    }

    SECTION("after a fit") {
        Trainer trainer = make_trainer(ds, quiet_train_config(1));
        trainer.prepare_data(ds, 0.25f, 4);
        trainer.fit();
        const auto meta = trainer.run_metadata();
        const int effective = trainer.effective_batch_size();
        (void)trainer.cross_validate(2, 4);
        REQUIRE(trainer.effective_batch_size() == effective);
        REQUIRE(trainer.run_metadata().created_at == meta.created_at);
        REQUIRE(trainer.run_metadata().completed_at == meta.completed_at);
    }
}

// =============================================================================
// The held-out forward and the baselines
// =============================================================================

TEST_CASE("The chunked held-out forward predicts what a single forward does",
          "[trainer][held_out]") {
    // 4800 held-out plots: more than one chunk of Trainer::kHeldOutChunk.
    const auto ds = labelled_corpus(6000, {y_spec(TargetMissingPolicy::Drop)});
    TrainConfig tc = quiet_train_config(1);
    tc.batch_size = 256;
    Trainer trainer = make_trainer(ds, tc);
    trainer.prepare_data(ds, /*test_size=*/0.8f, 7);
    REQUIRE(trainer.test_indices().numel() > Trainer::kHeldOutChunk);
    trainer.fit();

    const auto residuals = trainer.compute_residuals("y");
    Predictor predictor(trainer.model(), trainer.scalers(), torch::kCPU);
    const auto all = predictor.predict(ds, false, -1).predictions.at("y");
    const auto expected = all.index_select(0, trainer.test_indices());
    const auto chunked = torch::tensor(residuals.predictions);
    REQUIRE(chunked.numel() == expected.numel());
    REQUIRE(torch::allclose(chunked, expected, 1e-5, 1e-5));
}

TEST_CASE("The baselines score the model exactly as final_metrics does, in original units",
          "[trainer][baselines]") {
    const auto ds = labelled_corpus(
        80, {y_spec(TargetMissingPolicy::Drop, TransformType::Log1p),
             TargetSpec::classification("cls", kClasses)});
    Trainer trainer = make_trainer(ds, quiet_train_config(4));
    trainer.prepare_data(ds, 0.25f, 9);
    const auto result = trainer.fit();

    const auto& y = result.baselines.at("y");
    REQUIRE(y.model_mae == result.final_metrics.at("y").at("mae"));
    REQUIRE(y.r_squared == result.final_metrics.at("y").at("r2"));
    const auto& cls = result.baselines.at("cls");
    REQUIRE(cls.model_accuracy == result.final_metrics.at("cls").at("accuracy"));

    // The baseline constant is the training mean of the recorded values, and
    // its errors are taken against the recorded test values.
    const auto raw = ds.targets().at("y");
    const auto train_y = raw.index_select(0, trainer.train_indices());
    const auto test_y = raw.index_select(0, trainer.test_indices());
    const float train_mean = train_y.mean().item<float>();
    REQUIRE(y.training_mean == Catch::Approx(train_mean).epsilon(1e-4));
    REQUIRE(y.baseline_mae ==
            Catch::Approx((test_y - train_mean).abs().mean().item<float>()).epsilon(1e-4));
    REQUIRE(y.baseline_mse ==
            Catch::Approx((test_y - train_mean).pow(2).mean().item<float>()).epsilon(1e-4));
    REQUIRE(y.skill_score == Catch::Approx(1.0f - y.model_mse / y.baseline_mse).margin(1e-6));
}
