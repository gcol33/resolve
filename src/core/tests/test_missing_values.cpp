// Missing covariates and coordinates.
//
// A blank covariate or coordinate cell used to be read as 0.0, so a recorded
// zero and a missing value were one input, and a plot without coordinates sat
// at (0, 0). The loader now keeps the cell as NaN and DatasetConfig::
// missing_values decides how the model reads it:
//
//   Indicate  the continuous block carries a 0/1 column per covariate and one
//             for the coordinate pair; the value is filled with the mean of
//             that column's recorded values on the fitting rows before
//             standardisation.
//   Zero      the value is read as 0.0 with no flag -- the earlier behaviour,
//             and what a checkpoint written before the policy existed reloads
//             as.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/checkpoint.hpp"
#include "resolve/continuous_block.hpp"
#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/trainer.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;
using Catch::Matchers::WithinAbs;

namespace {

const float kNaN = std::numeric_limits<float>::quiet_NaN();

class TempFile {
public:
    explicit TempFile(const std::string& content) {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_missing_" + std::to_string(counter_++) + ".csv");
        std::ofstream file(path_);
        file << content;
    }
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    [[nodiscard]] std::string path() const { return path_.string(); }
private:
    std::filesystem::path path_;
    static int counter_;
};
int TempFile::counter_ = 0;

void null_log(const std::string&) {}

constexpr int kPlots = 60;

// Plot i: lat/lon blank when i % 10 == 3, cov1 blank when i % 5 == 1, cov2
// blank when i % 4 == 2, a recorded cov1 of 0 when i % 5 == 4. y depends on
// cov1 where it is recorded.
std::string header_csv() {
    std::ostringstream out;
    out << "plot_id,lat,lon,cov1,cov2,y\n";
    for (int i = 0; i < kPlots; ++i) {
        const bool no_coords = i % 10 == 3;
        const bool no_cov1 = i % 5 == 1;
        const bool no_cov2 = i % 4 == 2;
        const double cov1 = (i % 5 == 4) ? 0.0 : 1.0 + (i % 7);
        out << "P" << i << ',';
        if (no_coords) out << ",,";
        else out << 45.0 + 0.1 * i << ',' << 10.0 + 0.05 * i << ',';
        if (!no_cov1) out << cov1;
        out << ',';
        if (!no_cov2) out << 2.0 + 0.5 * (i % 3);
        out << ',' << (no_cov1 ? 5.0 : 2.0 * cov1 + 1.0) << '\n';
    }
    return out.str();
}

std::string species_csv() {
    std::ostringstream out;
    out << "plot_id,sp,cover,genus,family\n";
    for (int i = 0; i < kPlots; ++i) {
        out << "P" << i << ",sp_a," << 1 + i % 3 << ",ga,fa\n";
        out << "P" << i << ",sp_" << (i % 4 == 0 ? "b" : "c") << ",2,gb,fb\n";
    }
    return out.str();
}

RoleMapping roles() {
    RoleMapping r;
    r.plot_id = "plot_id";
    r.species_id = "sp";
    r.abundance = "cover";
    r.latitude = "lat";
    r.longitude = "lon";
    r.genus = "genus";
    r.family = "family";
    r.covariates = {"cov1", "cov2"};
    return r;
}

DatasetConfig dataset_config(MissingValuePolicy policy) {
    DatasetConfig cfg;
    cfg.species_encoding = SpeciesEncodingMode::Hash;
    cfg.hash_dim = 8;
    cfg.top_k = 2;
    cfg.track_unknown_fraction = false;
    cfg.track_unknown_count = false;
    cfg.missing_values = policy;
    return cfg;
}

ResolveDataset load(MissingValuePolicy policy) {
    TempFile header(header_csv());
    TempFile species(species_csv());
    return ResolveDataset::from_csv(header.path(), species.path(), roles(),
                                    {TargetSpec::regression("y")},
                                    dataset_config(policy));
}

ModelConfig model_config() {
    ModelConfig cfg;
    cfg.species_encoding = SpeciesEncodingMode::Hash;
    cfg.hash_dim = 8;
    cfg.top_k = 2;
    cfg.n_taxonomy_slots = 2;
    cfg.genus_emb_dim = 4;
    cfg.family_emb_dim = 4;
    cfg.hidden_dims = {16, 8};
    return cfg;
}

TrainConfig train_config(int epochs) {
    TrainConfig cfg;
    cfg.batch_size = 8;
    cfg.max_epochs = epochs;
    cfg.patience = epochs;
    cfg.lr = 1e-3f;
    cfg.device = torch::kCPU;
    cfg.log = null_log;
    return cfg;
}

int64_t row_of(const ResolveDataset& ds, const std::string& plot_id) {
    const auto& ids = ds.plot_ids();
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == plot_id) return static_cast<int64_t>(i);
    }
    FAIL("plot " << plot_id << " not in the dataset");
    return -1;
}

}  // namespace

TEST_CASE("the loader keeps a missing covariate or coordinate as NaN",
          "[missing_values]") {
    auto ds = load(MissingValuePolicy::Indicate);
    REQUIRE(ds.n_plots() == kPlots);

    const auto cov = ds.covariates();
    const auto coords = ds.coordinates();
    // P1: cov1 blank. P4: cov1 recorded as 0. P3: no coordinates.
    CHECK(std::isnan(cov[row_of(ds, "P1")][0].item<float>()));
    CHECK(cov[row_of(ds, "P4")][0].item<float>() == 0.0f);
    CHECK(std::isnan(coords[row_of(ds, "P3")][0].item<float>()));
    CHECK(std::isnan(coords[row_of(ds, "P3")][1].item<float>()));
    CHECK_FALSE(std::isnan(coords[row_of(ds, "P4")][0].item<float>()));

    // The schema records the policy and sizes the flags from it.
    CHECK(ds.schema().missing_values == MissingValuePolicy::Indicate);
    CHECK(ds.schema().missing_flag_width() == 3);
    CHECK(load(MissingValuePolicy::Zero).schema().missing_flag_width() == 0);
}

TEST_CASE("assemble_continuous places the flags beside their values",
          "[missing_values]") {
    auto coords = torch::tensor({{1.0f, 2.0f}, {kNaN, kNaN}, {3.0f, 4.0f}});
    auto covariates = torch::tensor({{0.0f, kNaN}, {5.0f, 6.0f}, {kNaN, 7.0f}});
    auto unknown = torch::tensor({0.1f, 0.2f, 0.3f});

    SECTION("indicate") {
        auto block = assemble_continuous({coords, covariates, unknown, {}, {}},
                                         MissingValuePolicy::Indicate, 3);
        // coords (2) | coord flag | covariates (2) | covariate flags (2) | unknown
        REQUIRE(block.size(1) == 8);
        auto flags_coord = block.select(1, 2);
        CHECK(flags_coord[0].item<float>() == 0.0f);
        CHECK(flags_coord[1].item<float>() == 1.0f);
        CHECK(block[0][5].item<float>() == 0.0f);   // cov1 recorded 0: no flag
        CHECK(block[0][6].item<float>() == 1.0f);   // cov2 missing
        CHECK(block[2][5].item<float>() == 1.0f);   // cov1 missing
        CHECK(block[0][3].item<float>() == 0.0f);   // the recorded 0 stays a value
        CHECK(std::isnan(block[2][3].item<float>()));
        CHECK_THAT(block[1][7].item<float>(), WithinAbs(0.2, 1e-6));
    }

    SECTION("zero reads a missing value as 0.0 and adds no flag") {
        auto block = assemble_continuous({coords, covariates, unknown, {}, {}},
                                         MissingValuePolicy::Zero, 3);
        auto expected = torch::cat({torch::nan_to_num(coords, 0.0),
                                    torch::nan_to_num(covariates, 0.0),
                                    unknown.unsqueeze(1)}, 1);
        REQUIRE(block.sizes() == expected.sizes());
        CHECK(torch::equal(block, expected));
    }
}

TEST_CASE("the fill is the mean of a column's recorded values on the fitting rows",
          "[missing_values]") {
    // Column 0: fitting rows 0-2 hold 1, NaN, 3 -> fill 2. Row 3 is not fitted
    // on, so its 100 cannot move the fill. Column 1 has no recorded value.
    auto block = torch::tensor({{1.0f, kNaN}, {kNaN, kNaN}, {3.0f, kNaN}, {100.0f, kNaN}});
    Scalers scalers;
    fit_continuous_scalers(scalers, block.slice(0, 0, 3));
    CHECK_THAT(scalers.continuous_fill[0].item<float>(), WithinAbs(2.0, 1e-6));
    CHECK(scalers.continuous_fill[1].item<float>() == 0.0f);

    auto scaled = standardize_continuous(block, scalers);
    REQUIRE(torch::isfinite(scaled).all().item<bool>());
    // A filled cell reads the same as a recorded value equal to the fill.
    auto recorded_fill = torch::tensor({{2.0f, 0.0f}});
    CHECK(torch::allclose(scaled.slice(0, 1, 2), standardize_continuous(recorded_fill, scalers)));

    SECTION("unstandardize restores the missing cells") {
        ResolveSchema schema;
        schema.missing_values = MissingValuePolicy::Indicate;
        schema.has_coordinates = true;
        schema.covariate_names = {"cov1", "cov2"};
        auto coords = torch::tensor({{1.0f, 2.0f}, {kNaN, kNaN}, {5.0f, 6.0f}, {7.0f, 8.0f}});
        auto covs = torch::tensor({{0.0f, kNaN}, {5.0f, 6.0f}, {kNaN, 7.0f}, {1.0f, 2.0f}});
        auto raw = assemble_continuous({coords, covs, {}, {}, {}},
                                       MissingValuePolicy::Indicate, 4);
        Scalers s;
        fit_continuous_scalers(s, raw);
        auto back = unstandardize_continuous(standardize_continuous(raw, s), s, schema);
        CHECK(torch::equal(torch::isnan(back), torch::isnan(raw)));
        CHECK(torch::allclose(torch::nan_to_num(back, 0.0), torch::nan_to_num(raw, 0.0),
                              1e-4, 1e-4));
    }
}

TEST_CASE("the model is sized for the flags, trains, and reloads through missing values",
          "[missing_values][predictor]") {
    for (auto policy : {MissingValuePolicy::Indicate, MissingValuePolicy::Zero}) {
        DYNAMIC_SECTION((policy == MissingValuePolicy::Indicate ? "indicate" : "zero")) {
            auto ds = load(policy);
            torch::manual_seed(0);
            ResolveModel model(ds.schema(), model_config());
            Trainer trainer(model, train_config(3));
            trainer.prepare_data(ds, 0.25f, 7);
            auto result = trainer.fit();
            CHECK(std::isfinite(result.final_metrics.at("y").at("mae")));

            const auto& scalers = trainer.scalers();
            REQUIRE(scalers.continuous_fill.defined());
            REQUIRE(torch::isfinite(scalers.continuous_fill).all().item<bool>());

            const auto path = (std::filesystem::temp_directory_path() /
                               ("resolve_missing_ckpt_" +
                                std::to_string(static_cast<int>(policy)) + ".pt")).string();
            trainer.save(path);
            auto predictor = Predictor::load(path, torch::kCPU);
            CHECK(predictor.schema().missing_values == policy);
            REQUIRE(predictor.scalers().continuous_fill.defined());
            CHECK(torch::allclose(predictor.scalers().continuous_fill,
                                  scalers.continuous_fill));

            auto one_shot = predictor.predict(ds, false, -1);
            auto chunked = predictor.predict(ds, false, 7);
            const auto& a = one_shot.predictions.at("y");
            REQUIRE(torch::isfinite(a).all().item<bool>());
            CHECK(torch::allclose(a, chunked.predictions.at("y"), 1e-5, 1e-6));
            std::filesystem::remove(path);
        }
    }
}

TEST_CASE("a flagged plot and its filled twin differ only through the flag",
          "[missing_values]") {
    // Under Indicate the model can tell a missing cov1 from a recorded value
    // equal to the fill; under Zero a missing cov1 and a recorded 0 are the same
    // input.
    auto ds = load(MissingValuePolicy::Zero);
    const int64_t missing = row_of(ds, "P1");
    const int64_t zero = row_of(ds, "P4");
    auto block = assemble_continuous(
        {ds.coordinates(), ds.covariates(), {}, {}, {}}, MissingValuePolicy::Zero,
        ds.n_plots());
    CHECK(block[missing][2].item<float>() == block[zero][2].item<float>());

    auto flagged = assemble_continuous(
        {ds.coordinates(), ds.covariates(), {}, {}, {}}, MissingValuePolicy::Indicate,
        ds.n_plots());
    CHECK(flagged[missing][5].item<float>() == 1.0f);
    CHECK(flagged[zero][5].item<float>() == 0.0f);
}

TEST_CASE("cross-validation refits the fill on each fold and stays finite",
          "[missing_values][cv]") {
    auto ds = load(MissingValuePolicy::Indicate);
    torch::manual_seed(0);
    ResolveModel model(ds.schema(), model_config());
    Trainer trainer(model, train_config(2));
    trainer.prepare_data(ds, 0.2f, 3);
    const Scalers before = trainer.scalers();
    auto cv = trainer.cross_validate(3, 11);
    REQUIRE(cv.n_folds == 3);
    for (const auto& fold : cv.fold_results) {
        CHECK(std::isfinite(fold.final_metrics.at("y").at("mae")));
    }
    // cross_validate restores the prepared split and its scalers.
    CHECK(torch::allclose(trainer.scalers().continuous_fill, before.continuous_fill));
}

TEST_CASE("spatial blocking keeps plots without coordinates out of every test fold",
          "[missing_values][spatial_cv]") {
    std::vector<float> values;
    std::set<int64_t> unplaced;
    for (int i = 0; i < 40; ++i) {
        if (i % 7 == 0) {
            values.push_back(kNaN);
            values.push_back(kNaN);
            unplaced.insert(i);
        } else {
            values.push_back(static_cast<float>(i % 5));
            values.push_back(static_cast<float>(i % 4));
        }
    }
    auto coords = torch::tensor(values).reshape({40, 2});
    SpatialBlockSplitter splitter(1.0f, 1.0f, 4, 42, false);
    auto folds = splitter.split(coords);
    REQUIRE(folds.size() == 4);
    for (const auto& [train, test] : folds) {
        REQUIRE_FALSE(test.empty());
        for (int64_t i : test) CHECK(unplaced.count(i) == 0);
        for (int64_t i : unplaced) {
            CHECK(std::find(train.begin(), train.end(), i) != train.end());
        }
    }
}

TEST_CASE("the policy round-trips through a checkpoint's schema",
          "[missing_values][checkpoint]") {
    for (auto policy : {MissingValuePolicy::Indicate, MissingValuePolicy::Zero}) {
        ResolveSchema schema;
        schema.missing_values = policy;
        torch::serialize::OutputArchive out;
        save_schema(out, schema);
        std::stringstream buffer;
        out.save_to(buffer);
        torch::serialize::InputArchive in;
        in.load_from(buffer);
        CHECK(load_schema(in).missing_values == policy);
    }
    // dataset_config_from_checkpoint hands the policy to an inference load.
    ResolveSchema schema;
    schema.missing_values = MissingValuePolicy::Zero;
    CHECK(dataset_config_from_checkpoint(schema, ModelConfig{}).missing_values ==
          MissingValuePolicy::Zero);
}
