// Tests for Predictor::predict(ResolveDataset, return_latent, batch_size).
//
// Covers the chunked-forward port (issue #2). The behavioral contract is:
//   - batch_size = -1 runs a single forward over the whole dataset (legacy).
//   - batch_size > 0 chunks the dataset along dim 0 and concats outputs on
//     CPU. The chunked output MUST equal the one-shot output bit-for-bit
//     up to libtorch's nondeterminism budget (we use a tight allclose).
//   - batch_size > n_plots collapses to the one-shot path with no extra
//     concat needed; predictions must still match.
//   - batch_size = 1 forwards each plot individually; predictions must
//     still match (covers the "tiny input" edge case).
//   - batch_size = 0 or any negative non-(-1) value is rejected.
//
// Construction strategy: synthetic in-memory ResolveDataset with both a
// covariate and a hash-mode species column. We never actually call fit()
// here — the test is about the chunking math being a no-op on the
// predictions, not about the model converging. Random weights are fine
// because both code paths share the same model_->forward() call.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/dataset.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "resolve/predictor.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace resolve;

namespace {

// TempFile helper (same shape as the other test files; kept local so this
// file builds independently).
class TempFile {
public:
    explicit TempFile(const std::string& content,
                      const std::string& suffix = ".csv") {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_pred_test_" + std::to_string(counter_++) + suffix);
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

// Build a synthetic ResolveDataset with `n_plots` rows, two covariates,
// lat/lon, and a hash species column with one species per plot. Used to
// exercise the chunked predict path against deterministic data.
ResolveDataset make_synthetic_dataset(int64_t n_plots) {
    std::ostringstream hdr;
    hdr << "plot_id,lat,lon,cov1,cov2,y\n";
    std::ostringstream spc;
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        // Spread coords + covariates over a wide range so the scaler has
        // something non-degenerate to fit on.
        double lat = 40.0 + (static_cast<double>(i) / n_plots) * 10.0;
        double lon = -5.0 + (static_cast<double>(i) / n_plots) * 10.0;
        double c1 = static_cast<double>(i % 7) * 1.1;
        double c2 = static_cast<double>(i % 11) * 0.9;
        double y = c1 + c2;
        hdr << "P" << i << "," << lat << "," << lon << ","
            << c1 << "," << c2 << "," << y << "\n";
        // One species per plot, cycling through a small pool. Hash
        // encoding hashes the string so collisions are not a correctness
        // concern.
        spc << "P" << i << ",sp" << (i % 13) << ",1.0\n";
    }

    TempFile header_csv(hdr.str());
    TempFile species_csv(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "sp";
    roles.abundance = "cover";
    roles.latitude = "lat";
    roles.longitude = "lon";
    roles.covariates = {"cov1", "cov2"};

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4;
    dcfg.use_taxonomy = false;
    dcfg.track_unknown_fraction = false;
    dcfg.track_unknown_count = false;

    return ResolveDataset::from_csv(
        header_csv.path(), species_csv.path(), roles,
        {TargetSpec::regression("y")}, dcfg);
}

// Build a minimal Predictor over the given dataset. Random model weights
// are fine — both predict paths share the same forward and the test
// asserts agreement between paths, not against any ground truth.
Predictor make_test_predictor(const ResolveDataset& ds) {
    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.hash_dim = 4;
    mcfg.hidden_dims = {8, 4};

    ResolveModel model(ds.schema(), mcfg);

    TrainConfig tcfg;
    tcfg.batch_size = 4;
    tcfg.max_epochs = 1;
    tcfg.patience = 1;
    tcfg.lr = 1e-3f;

    Trainer trainer(model, tcfg);
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/0);

    // Persist + reload to get a real Predictor (mirrors the path the
    // chunking actually runs through in production). Skipping fit() so
    // the test stays under a second.
    auto ckpt_path =
        (std::filesystem::temp_directory_path() / "pred_batch_test.pt")
            .string();
    trainer.save(ckpt_path);
    Predictor p = Predictor::load(ckpt_path, torch::kCPU);
    std::filesystem::remove(ckpt_path);
    return p;
}

}  // namespace

// =============================================================================
// 1. batch_size = -1 keeps the one-shot path
// =============================================================================

TEST_CASE("Predictor::predict batch_size=-1 runs the one-shot forward",
          "[predictor][batch_size]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/64);
    auto predictor = make_test_predictor(ds);

    auto preds = predictor.predict(ds, /*return_latent=*/false,
                                   /*batch_size=*/-1);
    REQUIRE(preds.predictions.count("y") == 1);
    REQUIRE(preds.predictions.at("y").size(0) == 64);
}

// =============================================================================
// 2. batch_size = 4096 matches batch_size = -1 (the central contract)
// =============================================================================

TEST_CASE("Predictor::predict chunked output matches one-shot output",
          "[predictor][batch_size]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/250);
    auto predictor = make_test_predictor(ds);

    auto one_shot = predictor.predict(ds, /*return_latent=*/false,
                                       /*batch_size=*/-1);
    auto chunked = predictor.predict(ds, /*return_latent=*/false,
                                      /*batch_size=*/64);

    REQUIRE(one_shot.predictions.size() == chunked.predictions.size());

    for (const auto& [name, one_shot_t] : one_shot.predictions) {
        REQUIRE(chunked.predictions.count(name) == 1);
        const auto& chunked_t = chunked.predictions.at(name);
        REQUIRE(one_shot_t.sizes() == chunked_t.sizes());
        // Both paths must produce the same predictions: same model,
        // same inputs, just sliced differently. Allow a tiny tolerance
        // for any floating-point reordering libtorch might do.
        REQUIRE(torch::allclose(
            one_shot_t.to(torch::kCPU), chunked_t.to(torch::kCPU),
            /*rtol=*/1e-5, /*atol=*/1e-6));
    }
    REQUIRE(chunked.plot_ids.size() == 250);
}

// =============================================================================
// 3. batch_size > n_plots is the no-op chunking case
// =============================================================================

TEST_CASE("Predictor::predict batch_size > n_plots collapses to one-shot",
          "[predictor][batch_size]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/40);
    auto predictor = make_test_predictor(ds);

    auto one_shot = predictor.predict(ds, /*return_latent=*/false,
                                       /*batch_size=*/-1);
    auto big_bs = predictor.predict(ds, /*return_latent=*/false,
                                     /*batch_size=*/100000);

    for (const auto& [name, one_shot_t] : one_shot.predictions) {
        const auto& big_bs_t = big_bs.predictions.at(name);
        REQUIRE(torch::allclose(
            one_shot_t.to(torch::kCPU), big_bs_t.to(torch::kCPU),
            /*rtol=*/1e-5, /*atol=*/1e-6));
    }
}

// =============================================================================
// 4. batch_size = 1 forwards one plot at a time, tiny input
// =============================================================================

TEST_CASE("Predictor::predict batch_size=1 matches one-shot on tiny inputs",
          "[predictor][batch_size]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/8);
    auto predictor = make_test_predictor(ds);

    auto one_shot = predictor.predict(ds, /*return_latent=*/false,
                                       /*batch_size=*/-1);
    auto bs1 = predictor.predict(ds, /*return_latent=*/false,
                                  /*batch_size=*/1);

    for (const auto& [name, one_shot_t] : one_shot.predictions) {
        const auto& bs1_t = bs1.predictions.at(name);
        REQUIRE(one_shot_t.sizes() == bs1_t.sizes());
        REQUIRE(torch::allclose(
            one_shot_t.to(torch::kCPU), bs1_t.to(torch::kCPU),
            /*rtol=*/1e-5, /*atol=*/1e-6));
    }
}

// =============================================================================
// 5. return_latent=true respects chunking too
// =============================================================================

TEST_CASE("Predictor::predict return_latent matches one-shot under chunking",
          "[predictor][batch_size][latent]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/96);
    auto predictor = make_test_predictor(ds);

    auto one_shot = predictor.predict(ds, /*return_latent=*/true,
                                       /*batch_size=*/-1);
    auto chunked = predictor.predict(ds, /*return_latent=*/true,
                                      /*batch_size=*/32);

    REQUIRE(one_shot.latent.defined());
    REQUIRE(chunked.latent.defined());
    REQUIRE(one_shot.latent.sizes() == chunked.latent.sizes());
    REQUIRE(torch::allclose(
        one_shot.latent.to(torch::kCPU), chunked.latent.to(torch::kCPU),
        /*rtol=*/1e-5, /*atol=*/1e-6));
}

// =============================================================================
// 6. invalid batch_size is rejected
// =============================================================================

TEST_CASE("Predictor::predict rejects batch_size=0 / negative non-(-1)",
          "[predictor][batch_size]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/8);
    auto predictor = make_test_predictor(ds);

    REQUIRE_THROWS_AS(
        predictor.predict(ds, /*return_latent=*/false, /*batch_size=*/0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        predictor.predict(ds, /*return_latent=*/false, /*batch_size=*/-2),
        std::invalid_argument);
}

// =============================================================================
// 7. optimize_for_inference (Linear+BatchNorm fusion) preserves predictions
// =============================================================================

TEST_CASE("Predictor::optimize_for_inference preserves predictions",
          "[predictor][inference]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/80);
    auto predictor = make_test_predictor(ds);

    auto before = predictor.predict(ds, /*return_latent=*/false, /*batch_size=*/-1);
    predictor.optimize_for_inference();  // folds BatchNorm into the preceding Linear
    auto after = predictor.predict(ds, /*return_latent=*/false, /*batch_size=*/-1);

    REQUIRE(before.predictions.size() == after.predictions.size());
    for (const auto& [name, t] : before.predictions) {
        REQUIRE(after.predictions.count(name) == 1);
        REQUIRE(torch::allclose(
            t.to(torch::kCPU), after.predictions.at(name).to(torch::kCPU),
            /*rtol=*/1e-4, /*atol=*/1e-5));
    }
}
