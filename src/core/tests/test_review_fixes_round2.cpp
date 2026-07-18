// Regression + recovery tests for the 2026-07-18 review-sweep fixes
// (issues #70-#78). Fast deterministic guards for the correctness fixes, plus
// parameter-recovery for a non-MLP architecture and a non-hash encoder (the
// #78 gap: previously only Hash + MLP was ever trained to recover a signal).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "resolve/dataset.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "resolve/loss.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

using namespace resolve;

namespace {

class TempFile {
public:
    explicit TempFile(const std::string& content) {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_rf2_" + std::to_string(counter_++) + ".csv");
        std::ofstream file(path_);
        file << content;
    }
    ~TempFile() { std::filesystem::remove(path_); }
    [[nodiscard]] std::string path() const { return path_.string(); }
private:
    std::filesystem::path path_;
    static int counter_;
};
int TempFile::counter_ = 0;

double pearson(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = a.size();
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        sab += da * db; saa += da * da; sbb += db * db;
    }
    if (saa <= 0 || sbb <= 0) return 0.0;
    return sab / std::sqrt(saa * sbb);
}

TrainConfig cpu_train_config(int max_epochs) {
    TrainConfig tcfg;
    tcfg.batch_size = 64;
    tcfg.max_epochs = max_epochs;
    tcfg.patience = 40;
    tcfg.lr = 1e-2f;
    tcfg.device = torch::kCPU;
    return tcfg;
}

}  // namespace

// =============================================================================
// #72 - select_top_k breaks abundance ties deterministically by name, so the
// selected set no longer depends on input (CSV row) order.
// =============================================================================
TEST_CASE("select_top_k is order-independent on abundance ties", "[review2][determinism]") {
    // Five species, all equal abundance (presence/absence data): the top-3 must
    // be the three alphabetically-first names regardless of input order.
    std::vector<std::pair<std::string, float>> a = {
        {"zeta", 1.0f}, {"alpha", 1.0f}, {"gamma", 1.0f}, {"beta", 1.0f}, {"delta", 1.0f}};
    std::vector<std::pair<std::string, float>> b = {
        {"delta", 1.0f}, {"gamma", 1.0f}, {"beta", 1.0f}, {"alpha", 1.0f}, {"zeta", 1.0f}};

    auto sa = select_top_k(a, 3);
    auto sb = select_top_k(b, 3);

    std::vector<std::string> na, nb;
    for (auto& p : sa) na.push_back(p.first);
    for (auto& p : sb) nb.push_back(p.first);
    std::sort(na.begin(), na.end());
    std::sort(nb.begin(), nb.end());

    REQUIRE(na == nb);
    const std::vector<std::string> expected = {"alpha", "beta", "delta"};
    REQUIRE(na == expected);
}

// Mixed abundances still rank by abundance first; only genuine ties use the name.
TEST_CASE("select_top_k ranks by abundance, name only on ties", "[review2][determinism]") {
    std::vector<std::pair<std::string, float>> in = {
        {"low", 0.1f}, {"tieB", 2.0f}, {"tieA", 2.0f}, {"high", 5.0f}};
    auto s = select_top_k(in, 3);
    REQUIRE(s.size() == 3);
    REQUIRE(s[0].first == "high");   // strictly largest
    REQUIRE(s[1].first == "tieA");   // tie at 2.0 -> name ascending
    REQUIRE(s[2].first == "tieB");
}

// =============================================================================
// #75 - macro-F1 averages over ALL classes; a class with no support in the fold
// contributes F1 = 0 (sklearn macro convention), not excluded.
// =============================================================================
TEST_CASE("macro_f1 includes zero-support classes", "[review2][metrics]") {
    // 3-class head, but class 2 never appears in target or prediction.
    // Classes 0 and 1 are predicted perfectly. classification_metrics expects
    // (N, num_classes) logits/probabilities (it argmaxes over dim 1), so encode
    // the predicted classes as one-hot rows.
    auto target = torch::tensor({0, 0, 1, 1, 0, 1}, torch::kInt64);
    auto pred_classes = torch::tensor({0, 0, 1, 1, 0, 1}, torch::kInt64);
    auto pred = torch::one_hot(pred_classes, /*num_classes=*/3).to(torch::kFloat32);

    auto m = Metrics::classification_metrics(pred, target, /*num_classes=*/3);

    REQUIRE(m.accuracy == Catch::Approx(1.0f));
    // Per-class F1: class0=1, class1=1, class2=0 (no support). Old (excluding
    // zero-support) gave 1.0; sklearn macro over all 3 labels is 2/3.
    REQUIRE(m.macro_f1 == Catch::Approx(2.0f / 3.0f).margin(1e-5));
    // weighted-F1 weights by support, so the absent class contributes nothing.
    REQUIRE(m.weighted_f1 == Catch::Approx(1.0f));
}

// =============================================================================
// #74 - classification_with_mapping sizes the head from max_code+1, so a
// non-dense mapping ({0,2,5}) builds a 6-output head and a code-5 target does
// NOT index out of bounds (before the fix num_classes was mapping.size()==3).
// =============================================================================
TEST_CASE("sparse class_mapping sizes the head correctly", "[review2][classmap]") {
    std::unordered_map<std::string, int64_t> mapping = {{"a", 0}, {"b", 2}, {"c", 5}};
    auto spec = TargetSpec::classification_with_mapping("hab", mapping);
    REQUIRE(spec.num_classes == 6);

    std::ostringstream hdr, spc;
    hdr << "plot_id,hab\n";
    spc << "plot_id,sp,cover\n";
    const char* labels[] = {"a", "b", "c"};
    for (int i = 0; i < 120; ++i) {
        hdr << "P" << i << "," << labels[i % 3] << "\n";
        spc << "P" << i << ",sp" << (i % 5) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "sp"; roles.abundance = "cover";

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4; dcfg.use_taxonomy = false;
    dcfg.track_unknown_fraction = false; dcfg.track_unknown_count = false;

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles, {spec}, dcfg);
    REQUIRE(ds.schema().targets.size() == 1);
    REQUIRE(ds.schema().targets[0].num_classes == 6);
    REQUIRE(ds.schema().targets[0].class_names.size() == 6);

    // A short fit must not throw: with the pre-fix num_classes==3 the head had
    // 3 outputs and CrossEntropy on a target of 5 asserts out of bounds.
    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.hash_dim = 4; mcfg.hidden_dims = {16, 8};
    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, cpu_train_config(3));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/3);
    REQUIRE_NOTHROW(trainer.fit());
}

// =============================================================================
// #74 - explicit class_mapping rejects a negative code (would index the head
// OOB via static_cast<size_t> of a negative).
// =============================================================================
TEST_CASE("explicit class_mapping rejects negative codes", "[review2][classmap]") {
    std::unordered_map<std::string, int64_t> mapping = {{"a", 0}, {"b", -1}};
    auto spec = TargetSpec::classification_with_mapping("hab", mapping);

    std::ostringstream hdr, spc;
    hdr << "plot_id,hab\n";
    spc << "plot_id,sp,cover\n";
    for (int i = 0; i < 20; ++i) {
        hdr << "P" << i << "," << ((i % 2) ? "a" : "b") << "\n";
        spc << "P" << i << ",sp" << (i % 3) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "sp"; roles.abundance = "cover";
    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4; dcfg.use_taxonomy = false;

    REQUIRE_THROWS_AS(
        ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles, {spec}, dcfg),
        std::invalid_argument);
}

// =============================================================================
// #71 - species records for plots absent from the kept set are filtered out of
// the vocab even when the species file covers fewer plots than are kept (the
// case the old size-guard wrongly skipped).
// =============================================================================
TEST_CASE("phantom-plot species do not inflate the vocab", "[review2][vocab]") {
    // Header keeps 4 plots (P0..P3). Species file covers only P0, P1, and a
    // PHANTOM plot P9 not in the header: 3 record-plots <= 4 kept, so the old
    // `plot_records.size() <= kept_ids.size()` guard skipped filtering and let
    // spPhantom into the vocab.
    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "sp"; roles.abundance = "cover";
    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4; dcfg.use_taxonomy = false;

    std::string header = "plot_id,y\nP0,1\nP1,2\nP2,3\nP3,4\n";

    std::string species_phantom =
        "plot_id,sp,cover\nP0,spA,1.0\nP1,spB,1.0\nP9,spPhantom,1.0\n";
    std::string species_clean =
        "plot_id,sp,cover\nP0,spA,1.0\nP1,spB,1.0\n";

    TempFile hdr(header);
    TempFile spc_phantom(species_phantom), spc_clean(species_clean);

    auto ds_phantom = ResolveDataset::from_csv(
        hdr.path(), spc_phantom.path(), roles, {TargetSpec::regression("y")}, dcfg);
    auto ds_clean = ResolveDataset::from_csv(
        hdr.path(), spc_clean.path(), roles, {TargetSpec::regression("y")}, dcfg);

    // The phantom species must not appear: the vocab matches the clean load.
    REQUIRE(ds_phantom.schema().n_species_vocab == ds_clean.schema().n_species_vocab);
}

// =============================================================================
// #70 / #45 - cross_validate runs, returns per-fold metrics, and leaves the
// trainer's split intact so post-CV evaluators use the original held-out fold.
// =============================================================================
TEST_CASE("cross_validate returns folds and restores split state", "[review2][cv]") {
    const int64_t n_plots = 400;
    std::ostringstream hdr, spc;
    hdr << "plot_id,lat,lon,cov1,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(i * 0.13);
        const double y = 2.0 * c1 + 1.0;
        hdr << "P" << i << "," << (40.0 + (i % 20) * 0.5) << "," << (-5.0 + (i / 20) * 0.5)
            << "," << c1 << "," << y << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "sp"; roles.abundance = "cover";
    roles.latitude = "lat"; roles.longitude = "lon"; roles.covariates = {"cov1"};

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4; dcfg.use_taxonomy = false;
    dcfg.track_unknown_fraction = false; dcfg.track_unknown_count = false;

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.hash_dim = 4; mcfg.hidden_dims = {16, 8};

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, cpu_train_config(20));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/9);

    const int64_t test_n_before = trainer.compute_residuals("y").predictions.size();

    auto cv = trainer.cross_validate(/*n_folds=*/3, /*seed=*/9);
    REQUIRE(cv.n_folds == 3);
    REQUIRE(cv.fold_results.size() == 3);
    REQUIRE(cv.mean_metrics.count("y") == 1);
    for (const auto& [name, val] : cv.mean_metrics.at("y")) {
        REQUIRE(std::isfinite(val));
    }

    // Post-CV, the original split is restored (#45): the held-out fold size and
    // predictions match the pre-CV split, not the last CV fold.
    const int64_t test_n_after = trainer.compute_residuals("y").predictions.size();
    REQUIRE(test_n_after == test_n_before);
}

// =============================================================================
// #70 - cross_validate_spatial runs to completion on shuffled data (the fold
// indices are now applied in the same order as the split tensors) and returns
// valid folds. Before the fix this silently scrambled the split; here we assert
// it produces the requested folds with finite metrics and intact post-state.
// =============================================================================
TEST_CASE("cross_validate_spatial runs on shuffled data", "[review2][cv]") {
    const int64_t n_plots = 400;
    std::ostringstream hdr, spc;
    hdr << "plot_id,lat,lon,cov1,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        // Plots laid out over a lat/lon grid so 1-degree blocks form real
        // spatial folds.
        const double lat = 40.0 + static_cast<double>(i % 20);
        const double lon = -5.0 + static_cast<double>(i / 20);
        const double c1 = std::sin(i * 0.11);
        const double y = 1.5 * c1 + 0.5;
        hdr << "P" << i << "," << lat << "," << lon << "," << c1 << "," << y << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "sp"; roles.abundance = "cover";
    roles.latitude = "lat"; roles.longitude = "lon"; roles.covariates = {"cov1"};

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4; dcfg.use_taxonomy = false;
    dcfg.track_unknown_fraction = false; dcfg.track_unknown_count = false;

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.hash_dim = 4; mcfg.hidden_dims = {16, 8};

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, cpu_train_config(15));
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

// =============================================================================
// #78 (P1) - a non-MLP architecture (FT-Transformer) recovers a known covariate
// signal. Previously only MLP was ever trained to recover anything.
// =============================================================================
TEST_CASE("FT-Transformer recovers a covariate signal", "[review2][recovery][arch]") {
    const int64_t n_plots = 600;
    std::ostringstream hdr, spc;
    hdr << "plot_id,cov1,cov2,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(i * 0.13);
        const double c2 = std::cos(i * 0.07);
        const double y = 2.0 * c1 - 1.5 * c2 + 3.0;
        hdr << "P" << i << "," << c1 << "," << c2 << "," << y << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "sp"; roles.abundance = "cover";
    roles.covariates = {"cov1", "cov2"};

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4; dcfg.use_taxonomy = false;
    dcfg.track_unknown_fraction = false; dcfg.track_unknown_count = false;

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::FTTransformer;
    mcfg.hash_dim = 4; mcfg.hidden_dims = {32};

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, cpu_train_config(300));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/11);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() > 10);
    // Lenient: a working architecture correlates strongly with the linear
    // signal; a broken adapter (ignores its input) sits near zero.
    REQUIRE(pearson(res.predictions, res.actuals) > 0.6);
}

// =============================================================================
// #78 (P2) - a non-hash species encoder (Embed) recovers a species-driven
// target end-to-end. Previously only the Hash pathway was proven to carry
// species signal.
// =============================================================================
TEST_CASE("Embed encoder recovers a species-driven target", "[review2][recovery][encoder]") {
    const int64_t n_plots = 600;
    const int n_species = 10;
    std::vector<double> species_value(n_species);
    for (int s = 0; s < n_species; ++s) species_value[s] = std::sin(s * 1.7) * 2.0;

    std::ostringstream hdr, spc;
    hdr << "plot_id,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        int s0 = static_cast<int>(i % n_species);
        int s1 = static_cast<int>((i / n_species) % n_species);
        double y = species_value[s0] + species_value[s1];
        hdr << "P" << i << "," << y << "\n";
        spc << "P" << i << ",sp" << s0 << ",1.0\n";
        spc << "P" << i << ",sp" << s1 << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "sp"; roles.abundance = "cover";

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Embed;
    dcfg.top_k_species = 4; dcfg.use_taxonomy = false;
    dcfg.track_unknown_fraction = false; dcfg.track_unknown_count = false;

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Embed;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.top_k_species = 4; mcfg.species_embed_dim = 16; mcfg.hidden_dims = {64, 32};

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, cpu_train_config(400));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/5);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() > 10);
    REQUIRE(pearson(res.predictions, res.actuals) > 0.6);
}
