// Trainer::cross_validate and cross_validate_spatial on a short schedule: the
// folds they return, the per-fold reset to the as-constructed weights, and the
// trainer's own split surviving the run.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <cmath>
#include <sstream>

using namespace resolve;
using namespace resolve::test;

namespace {

// 400 plots on a 20 x 20 grid of `cell_deg`-degree cells starting at
// (40N, 5W), with cov1 = sin(freq i) and y = slope * cov1 + intercept, and one
// of eight species per plot.
ResolveDataset grid_dataset(double cell_deg, double freq, double slope, double intercept) {
    const int64_t n_plots = 400;
    std::ostringstream hdr, spc;
    hdr << "plot_id,lat,lon,cov1,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(i * freq);
        const double y = slope * c1 + intercept;
        hdr << "P" << i << "," << (40.0 + (i % 20) * cell_deg) << ","
            << (-5.0 + (i / 20) * cell_deg) << "," << c1 << "," << y << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());
    return ResolveDataset::from_csv(header_csv.path(), species_csv.path(),
                                    Roles().coordinates().covariates({"cov1"}),
                                    {TargetSpec::regression("y")}, hash_dataset_config());
}

}  // namespace

TEST_CASE("cross_validate returns folds and restores split state", "[trainer][cv]") {
    auto ds = grid_dataset(/*cell_deg=*/0.5, /*freq=*/0.13, /*slope=*/2.0, /*intercept=*/1.0);

    ResolveModel model(ds.schema(), hash_mlp_config({16, 8}));
    Trainer trainer(model, recovery_train_config(20));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/9);

    const int64_t test_n_before = trainer.compute_residuals("y").predictions.size();

    auto cv = trainer.cross_validate(/*n_folds=*/3, /*seed=*/9);
    REQUIRE(cv.n_folds == 3);
    REQUIRE(cv.fold_results.size() == 3);
    REQUIRE(cv.mean_metrics.count("y") == 1);
    for (const auto& [name, val] : cv.mean_metrics.at("y")) {
        REQUIRE(std::isfinite(val));
    }

    // Post-CV, the original split is restored: the held-out fold size and
    // predictions match the pre-CV split, not the last CV fold.
    const int64_t test_n_after = trainer.compute_residuals("y").predictions.size();
    REQUIRE(test_n_after == test_n_before);
}

// cross_validate resets each fold to the as-constructed (untrained) weights, so
// running it after fit() does NOT warm-start folds from trained weights (which
// would leak the fold's held-out rows into its starting point and bias the
// metrics). CV on a fresh trainer and CV re-run after a full fit() produce
// identical fold metrics.
TEST_CASE("cross_validate resets folds to pristine init after fit", "[trainer][cv]") {
    auto ds = grid_dataset(/*cell_deg=*/0.5, /*freq=*/0.13, /*slope=*/2.0, /*intercept=*/1.0);

    ResolveModel model(ds.schema(), hash_mlp_config({16, 8}));
    Trainer trainer(model, recovery_train_config(15));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/9);

    torch::manual_seed(123);
    auto cv_fresh = trainer.cross_validate(/*n_folds=*/3, /*seed=*/7);

    // Train to convergence, mutating the model's weights.
    trainer.fit();

    // Re-run identical CV. The per-fold pristine reset makes the fold metrics
    // reproduce the fresh run; a warm start from the trained weights would not.
    torch::manual_seed(123);
    auto cv_after_fit = trainer.cross_validate(/*n_folds=*/3, /*seed=*/7);

    REQUIRE(cv_fresh.mean_metrics.count("y") == 1);
    REQUIRE(cv_after_fit.mean_metrics.count("y") == 1);
    for (const auto& [name, val] : cv_fresh.mean_metrics.at("y")) {
        REQUIRE(cv_after_fit.mean_metrics.at("y").count(name) == 1);
        REQUIRE(cv_after_fit.mean_metrics.at("y").at(name) ==
                Catch::Approx(val).margin(1e-4));
    }
}

// cross_validate_spatial applies the fold indices in the order of the shuffled
// split tensors, so on shuffled data it returns the requested folds with finite
// metrics and leaves the trainer's state intact.
TEST_CASE("cross_validate_spatial runs on shuffled data", "[trainer][cv]") {
    // One-degree cells, so 1-degree spatial blocks form real folds.
    auto ds = grid_dataset(/*cell_deg=*/1.0, /*freq=*/0.11, /*slope=*/1.5, /*intercept=*/0.5);

    ResolveModel model(ds.schema(), hash_mlp_config({16, 8}));
    Trainer trainer(model, recovery_train_config(15));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/4);  // shuffles

    SpatialBlockConfig scfg;
    scfg.lat_size = 1.0f; scfg.lon_size = 1.0f;

    auto cv = trainer.cross_validate_spatial(scfg, /*n_folds=*/3, /*seed=*/4);
    REQUIRE(cv.n_folds == 3);
    REQUIRE(cv.fold_results.size() == 3);
    REQUIRE(cv.mean_metrics.count("y") == 1);
    for (const auto& [name, val] : cv.mean_metrics.at("y")) {
        REQUIRE(std::isfinite(val));
    }
    // Post-CV state intact.
    REQUIRE_NOTHROW(trainer.compute_residuals("y"));
}
