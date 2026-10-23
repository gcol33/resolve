// An explicit stopping fold: Trainer::prepare_data(dataset, stopping_indices,
// seed) holds out exactly the plots the caller names instead of a random
// fraction. The cases pin that the fold is used as given, that a fold equal to
// the random one reproduces the random run exactly (so everything downstream
// of the split behaves as under a random split), the input checks, and that
// the split reaches run_metadata() and the checkpoint.

#include <catch2/catch_test_macros.hpp>

#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <cstdint>
#include <filesystem>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

std::vector<int64_t> to_vector(const torch::Tensor& t) {
    auto c = t.to(torch::kCPU).contiguous();
    return std::vector<int64_t>(c.data_ptr<int64_t>(), c.data_ptr<int64_t>() + c.numel());
}

ModelConfig hash_model_config(const ResolveDataset& ds) {
    ModelConfig cfg = small_encoder_config(SpeciesEncodingMode::Hash, {16, 8});
    cfg.hash_dim = ds.config().hash_dim;
    return cfg;
}

// A trainer over a fresh model, seeded so two calls build identical weights.
Trainer make_trainer(const ResolveDataset& ds, const TrainConfig& tc) {
    torch::manual_seed(11);
    return Trainer(ResolveModel(ds.schema(), hash_model_config(ds)), tc);
}

// Message of the std::invalid_argument `fn` throws; fails the case otherwise.
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

// The run metadata `trainer` writes into a checkpoint, read back from one.
RunMetadata saved_run_metadata(const Trainer& trainer) {
    const TempPath path(".pt");
    trainer.save(path.path(), &trainer.run_metadata());
    RunMetadata meta = Trainer::load_run_metadata(path.path());
    std::filesystem::path sidecar(path.path());
    sidecar.replace_extension(".json");
    std::error_code ec;
    std::filesystem::remove(sidecar, ec);
    return meta;
}

}  // namespace

TEST_CASE("An explicit stopping fold holds out exactly the plots given",
          "[stopping_fold]") {
    const SmallCorpus corpus;
    const auto ds = corpus.load(SpeciesEncodingMode::Hash, /*taxonomy=*/false);
    const auto n = static_cast<int64_t>(ds.n_plots());

    const std::vector<int64_t> fold = {31, 4, 17, 0, 22, 9};
    Trainer trainer = make_trainer(ds, quiet_train_config());
    trainer.prepare_data(ds, fold, 7);

    REQUIRE(to_vector(trainer.test_indices()) == fold);

    const auto train = to_vector(trainer.train_indices());
    REQUIRE(static_cast<int64_t>(train.size()) == n - static_cast<int64_t>(fold.size()));
    std::set<int64_t> seen(train.begin(), train.end());
    for (int64_t i : fold) seen.insert(i);
    REQUIRE(static_cast<int64_t>(seen.size()) == n);

    std::vector<std::string> fold_ids;
    for (int64_t i : fold) fold_ids.push_back(ds.plot_ids()[static_cast<size_t>(i)]);
    REQUIRE(trainer.test_plot_ids() == fold_ids);
    REQUIRE(ds.plot_indices(fold_ids) == fold);

    const auto& meta = trainer.run_metadata();
    REQUIRE(meta.hold_out == HoldOutMode::Explicit);
    REQUIRE(meta.test_size == 0.0f);
    REQUIRE(meta.split_seed == 7);
    REQUIRE(meta.n_plots_test == static_cast<int64_t>(fold.size()));
    REQUIRE(meta.n_plots_train == n - static_cast<int64_t>(fold.size()));
}

TEST_CASE("A stopping fold equal to the random fold reproduces the random run",
          "[stopping_fold]") {
    const SmallCorpus corpus;
    const auto ds = corpus.load(SpeciesEncodingMode::Hash, /*taxonomy=*/false);
    TrainConfig tc = quiet_train_config(6);
    tc.patience = 2;

    Trainer random_trainer = make_trainer(ds, tc);
    random_trainer.prepare_data(ds, 0.25f, 3);
    const auto random_fold = to_vector(random_trainer.test_indices());
    const TrainResult random_result = random_trainer.fit();

    Trainer explicit_trainer = make_trainer(ds, tc);
    explicit_trainer.prepare_data(ds, random_fold, 3);
    REQUIRE(torch::equal(explicit_trainer.train_indices(), random_trainer.train_indices()));
    REQUIRE(torch::equal(explicit_trainer.test_indices(), random_trainer.test_indices()));
    const TrainResult explicit_result = explicit_trainer.fit();

    REQUIRE(explicit_result.train_loss_history == random_result.train_loss_history);
    REQUIRE(explicit_result.test_loss_history == random_result.test_loss_history);
    REQUIRE(explicit_result.best_epoch == random_result.best_epoch);
    // Compared as a bool: Catch cannot print the nested metric map.
    const bool same_final_metrics = explicit_result.final_metrics == random_result.final_metrics;
    REQUIRE(same_final_metrics);
    REQUIRE(explicit_trainer.test_plot_ids() == random_trainer.test_plot_ids());

    const auto& random_scalers = random_trainer.scalers();
    const auto& explicit_scalers = explicit_trainer.scalers();
    for (const auto& [name, scaler] : random_scalers.target_scalers) {
        const auto it = explicit_scalers.target_scalers.find(name);
        REQUIRE(it != explicit_scalers.target_scalers.end());
        REQUIRE(torch::equal(it->second.first, scaler.first));
        REQUIRE(torch::equal(it->second.second, scaler.second));
    }

    // The two runs differ only in how the fold was asked for.
    REQUIRE(random_trainer.run_metadata().hold_out == HoldOutMode::Random);
    REQUIRE(random_trainer.run_metadata().test_size == 0.25f);
    REQUIRE(explicit_trainer.run_metadata().hold_out == HoldOutMode::Explicit);
    REQUIRE(explicit_trainer.run_metadata().best_epoch ==
          random_trainer.run_metadata().best_epoch);
}

TEST_CASE("An unusable stopping fold is refused before anything is prepared",
          "[stopping_fold]") {
    const SmallCorpus corpus;
    const auto ds = corpus.load(SpeciesEncodingMode::Hash, /*taxonomy=*/false);
    const auto n = static_cast<int64_t>(ds.n_plots());
    Trainer trainer = make_trainer(ds, quiet_train_config());

    REQUIRE(invalid_argument_message([&] {
              trainer.prepare_data(ds, std::vector<int64_t>{}, 1);
          }).find("empty") != std::string::npos);

    const auto low = invalid_argument_message([&] {
        trainer.prepare_data(ds, std::vector<int64_t>{3, -1}, 1);
    });
    REQUIRE(low.find("index -1") != std::string::npos);
    REQUIRE(low.find("outside") != std::string::npos);

    const auto high = invalid_argument_message([&] {
        trainer.prepare_data(ds, std::vector<int64_t>{n}, 1);
    });
    REQUIRE(high.find("index " + std::to_string(n)) != std::string::npos);

    const auto twice = invalid_argument_message([&] {
        trainer.prepare_data(ds, std::vector<int64_t>{5, 2, 5}, 1);
    });
    REQUIRE(twice.find("more than once") != std::string::npos);
    REQUIRE(twice.find("'" + ds.plot_ids()[5] + "'") != std::string::npos);

    std::vector<int64_t> every(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) every[static_cast<size_t>(i)] = i;
    REQUIRE(invalid_argument_message([&] { trainer.prepare_data(ds, every, 1); })
              .find("none to fit") != std::string::npos);

    // Nothing was prepared by the refused calls.
    REQUIRE_THROWS(trainer.fit());

    REQUIRE(invalid_argument_message([&] {
              (void)ds.plot_indices({ds.plot_ids()[0], "no_such_plot"});
          }).find("no_such_plot") != std::string::npos);
}

TEST_CASE("A stopping fold without a masked target's labels is refused",
          "[stopping_fold]") {
    // y is recorded on the even plots only and masked on the odd ones; z is
    // recorded everywhere, so no plot is dropped as labelled for no target.
    std::ostringstream header, species;
    header << "plot_id,lat,lon,y,z\n";
    species << "plot_id,sp,cover\n";
    for (int i = 0; i < 24; ++i) {
        header << "P" << i << "," << (47.0 + 0.01 * i) << "," << (10.0 + 0.01 * i) << ",";
        if (i % 2 == 0) header << (1.0 + 0.1 * i);
        header << "," << (2.0 - 0.05 * i) << "\n";
        species << "P" << i << ",sp" << (i % 5) << ",1.0\n";
    }
    const TempFile header_csv(header.str(), ".csv", WriteMode::Binary);
    const TempFile species_csv(species.str(), ".csv", WriteMode::Binary);
    auto y = TargetSpec::regression("y");
    y.missing = TargetMissingPolicy::Mask;
    const auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(),
                                             Roles().coordinates(),
                                             {y, TargetSpec::regression("z")},
                                             hash_dataset_config());
    REQUIRE(ds.n_plots() == 24);

    Trainer trainer = make_trainer(ds, quiet_train_config());
    const auto odd = ds.plot_indices({"P1", "P3", "P5", "P7"});
    const auto message = invalid_argument_message([&] { trainer.prepare_data(ds, odd, 1); });
    REQUIRE(message.find("'y'") != std::string::npos);
    REQUIRE(message.find("stopping fold") != std::string::npos);

    // One labelled plot in the fold is enough.
    const auto mixed = ds.plot_indices({"P1", "P3", "P4"});
    REQUIRE_NOTHROW(trainer.prepare_data(ds, mixed, 1));
    REQUIRE(to_vector(trainer.test_indices()) == mixed);
}

TEST_CASE("The checkpoint records how the plots were split", "[stopping_fold]") {
    const SmallCorpus corpus;
    const auto ds = corpus.load(SpeciesEncodingMode::Hash, /*taxonomy=*/false);

    SECTION("explicit fold") {
        Trainer trainer = make_trainer(ds, quiet_train_config(2));
        const std::vector<int64_t> fold = {1, 2, 3, 5, 8, 13, 21, 34};
        trainer.prepare_data(ds, fold, 19);
        const TrainResult result = trainer.fit();

        const RunMetadata meta = saved_run_metadata(trainer);
        REQUIRE(meta.hold_out == HoldOutMode::Explicit);
        REQUIRE(meta.test_size == 0.0f);
        REQUIRE(meta.split_seed == 19);
        REQUIRE(meta.n_plots_test == static_cast<int64_t>(fold.size()));
        REQUIRE(meta.n_plots_train ==
              static_cast<int64_t>(ds.n_plots()) - static_cast<int64_t>(fold.size()));
        REQUIRE(meta.best_epoch == result.best_epoch);
        REQUIRE(meta.total_epochs == static_cast<int>(result.train_loss_history.size()));
    }

    SECTION("random fold") {
        Trainer trainer = make_trainer(ds, quiet_train_config(1));
        trainer.prepare_data(ds, 0.3f, 4);
        trainer.fit();

        const RunMetadata meta = saved_run_metadata(trainer);
        REQUIRE(meta.hold_out == HoldOutMode::Random);
        REQUIRE(meta.test_size == 0.3f);
        REQUIRE(meta.split_seed == 4);
    }
}

TEST_CASE("Cross-validation leaves the explicit split and its run metadata in place",
          "[stopping_fold]") {
    const SmallCorpus corpus;
    const auto ds = corpus.load(SpeciesEncodingMode::Hash, /*taxonomy=*/false);
    Trainer trainer = make_trainer(ds, quiet_train_config(1));
    const std::vector<int64_t> fold = {6, 12, 18, 24, 30, 36};
    trainer.prepare_data(ds, fold, 2);

    (void)trainer.cross_validate(2, 2);

    REQUIRE(to_vector(trainer.test_indices()) == fold);
    REQUIRE(trainer.run_metadata().hold_out == HoldOutMode::Explicit);
    REQUIRE(trainer.run_metadata().split_seed == 2);
    REQUIRE(trainer.run_metadata().n_plots_test == static_cast<int64_t>(fold.size()));
    REQUIRE(trainer.run_metadata().total_epochs == 0);
}
