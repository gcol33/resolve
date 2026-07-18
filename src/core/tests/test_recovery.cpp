// Parameter-recovery tests for the fitter (issue #35).
//
// The existing suite validates plumbing (shapes, dtypes, checkpoints, encoder
// equivalence) and the pure metric functions, but never trained the model to
// convergence and asserted it recovers a known signal. These tests do exactly
// that: they simulate data with a known relationship, fit to convergence on
// CPU, and assert the trained model recovers it on the held-out fold, well
// above chance. They are the counterpart to the repo's "Statistical Code Needs
// Recovery Tests" bar.
//
// Kept small and seeded so they run in a few seconds and are deterministic.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "resolve/dataset.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"

#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

using namespace resolve;

namespace {

class TempFile {
public:
    explicit TempFile(const std::string& content, const std::string& suffix = ".csv") {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_recovery_" + std::to_string(counter_++) + suffix);
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

// Pearson correlation between two equal-length vectors.
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

TrainConfig recovery_train_config(int max_epochs = 300) {
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
// 1. Regression head recovers a known linear signal in the covariates.
// =============================================================================
TEST_CASE("Regression recovers a known covariate signal", "[recovery][regression]") {
    const int64_t n_plots = 600;
    std::ostringstream hdr, spc;
    hdr << "plot_id,lat,lon,cov1,cov2,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(i * 0.13);
        const double c2 = std::cos(i * 0.07);
        const double y = 2.0 * c1 - 1.5 * c2 + 3.0;  // deterministic linear signal
        hdr << "P" << i << "," << (40.0 + i * 0.001) << "," << (-5.0 + i * 0.001)
            << "," << c1 << "," << c2 << "," << y << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "sp"; roles.abundance = "cover";
    roles.latitude = "lat"; roles.longitude = "lon";
    roles.covariates = {"cov1", "cov2"};

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4; dcfg.use_taxonomy = false;
    dcfg.track_unknown_fraction = false; dcfg.track_unknown_count = false;

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.hash_dim = 4; mcfg.hidden_dims = {32, 16};

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, recovery_train_config());
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/11);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() == res.actuals.size());
    REQUIRE(res.predictions.size() > 10);
    // A model that learned the signal correlates strongly with the truth on the
    // held-out fold; an untrained / broken fitter would sit near zero.
    REQUIRE(pearson(res.predictions, res.actuals) > 0.9);
}

// =============================================================================
// 2. Classification head separates a synthetic, separable class signal.
// =============================================================================
TEST_CASE("Classification recovers a separable class signal", "[recovery][classification]") {
    const int64_t n_plots = 600;
    const int n_classes = 3;
    std::ostringstream hdr, spc;
    hdr << "plot_id,lat,lon,cov1,cov2,hab\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(i * 0.13);          // in [-1, 1]
        const double c2 = std::cos(i * 0.07);
        // Class is a clean function of cov1 (three separable bands).
        int hab = (c1 < -0.33) ? 0 : (c1 < 0.33 ? 1 : 2);
        hdr << "P" << i << "," << (40.0 + i * 0.001) << "," << (-5.0 + i * 0.001)
            << "," << c1 << "," << c2 << "," << hab << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "sp"; roles.abundance = "cover";
    roles.latitude = "lat"; roles.longitude = "lon";
    roles.covariates = {"cov1", "cov2"};

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4; dcfg.use_taxonomy = false;
    dcfg.track_unknown_fraction = false; dcfg.track_unknown_count = false;

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::classification("hab", n_classes)}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.hash_dim = 4; mcfg.hidden_dims = {32, 16};

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, recovery_train_config());
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/11);
    trainer.fit();

    auto pred = trainer.compute_classification_predictions("hab");
    REQUIRE(pred.predicted_classes.defined());
    const int64_t n_test = pred.predicted_classes.size(0);
    REQUIRE(n_test > 10);
    const int64_t n_correct = (pred.predicted_classes == pred.actuals).sum().item<int64_t>();
    const double acc = static_cast<double>(n_correct) / static_cast<double>(n_test);
    // Chance is 1/3; a model that learned the separable signal is far above it.
    REQUIRE(acc > 0.75);
}

// =============================================================================
// 3. Species composition alone predicts the target (the central claim), via
//    the hash encoder: y is a deterministic function of the plot's species.
// =============================================================================
TEST_CASE("Species composition recovers a species-driven target", "[recovery][species]") {
    const int64_t n_plots = 600;
    const int n_species = 12;
    // Per-species contribution; the plot target is the sum over its species.
    std::vector<double> species_value(n_species);
    for (int s = 0; s < n_species; ++s) species_value[s] = std::sin(s * 1.7) * 2.0;

    std::ostringstream hdr, spc;
    hdr << "plot_id,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        // Each plot has 3 species determined by i (deterministic composition).
        int s0 = static_cast<int>(i % n_species);
        int s1 = static_cast<int>((i / n_species) % n_species);
        int s2 = static_cast<int>((i / (n_species * n_species) + i) % n_species);
        double y = species_value[s0] + species_value[s1] + species_value[s2];
        hdr << "P" << i << "," << y << "\n";
        spc << "P" << i << ",sp" << s0 << ",1.0\n";
        spc << "P" << i << ",sp" << s1 << ",1.0\n";
        spc << "P" << i << ",sp" << s2 << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "sp"; roles.abundance = "cover";

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 32; dcfg.use_taxonomy = false;
    dcfg.track_unknown_fraction = false; dcfg.track_unknown_count = false;

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.hash_dim = 32; mcfg.hidden_dims = {64, 32};

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, recovery_train_config(/*max_epochs=*/400));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/5);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() > 10);
    // The species-only hash pathway must carry enough signal to predict the
    // composition-determined target on held-out plots (well above the ~0
    // a broken species pathway would give).
    REQUIRE(pearson(res.predictions, res.actuals) > 0.8);
}

// =============================================================================
// 4. fit() persists run metadata to the checkpoint (issue #54): load_run_metadata
//    recovers the run's best epoch / total epochs / train time / metrics, which
//    fit() previously never wrote.
// =============================================================================
TEST_CASE("fit persists run metadata to the checkpoint", "[recovery][metadata]") {
    const int64_t n_plots = 300;
    std::ostringstream hdr, spc;
    hdr << "plot_id,lat,lon,cov1,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(i * 0.11);
        const double y = 1.7 * c1 + 2.0;
        hdr << "P" << i << "," << (40.0 + i * 0.001) << "," << (-5.0 + i * 0.001)
            << "," << c1 << "," << y << "\n";
        spc << "P" << i << ",sp" << (i % 6) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "sp"; roles.abundance = "cover";
    roles.latitude = "lat"; roles.longitude = "lon";
    roles.covariates = {"cov1"};

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

    const auto ckpt_dir = std::filesystem::temp_directory_path() /
        ("resolve_meta_" + std::to_string(::time(nullptr)) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&ds)));
    std::filesystem::create_directories(ckpt_dir);

    TrainConfig tcfg = recovery_train_config(/*max_epochs=*/30);
    tcfg.checkpoint_dir = ckpt_dir.string();

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, tcfg);
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/7);
    auto result = trainer.fit();

    const std::string ckpt = (ckpt_dir / "checkpoint.pt").string();
    REQUIRE(std::filesystem::exists(ckpt));

    const RunMetadata meta = Trainer::load_run_metadata(ckpt);
    // These were all-default (zeros/empty) before fit() passed a RunMetadata.
    REQUIRE(meta.total_epochs > 0);
    REQUIRE(meta.total_epochs == static_cast<int>(result.train_loss_history.size()));
    REQUIRE(meta.best_epoch == result.best_epoch);
    REQUIRE(meta.train_time_seconds > 0.0f);
    REQUIRE(meta.n_plots_train > 0);
    REQUIRE(meta.n_plots_test > 0);
    REQUIRE(meta.final_metrics.count("y") == 1);
    REQUIRE_FALSE(meta.created_at.empty());

    // JSON sidecar is written alongside the checkpoint.
    REQUIRE(std::filesystem::exists(ckpt_dir / "checkpoint.json"));

    std::filesystem::remove_all(ckpt_dir);
}
