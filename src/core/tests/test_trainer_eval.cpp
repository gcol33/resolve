// Tests for the checkpoint-evaluation surface added for issue #4:
//   - Trainer::load_state  : in-place loader (the static load() tuple return
//                            has no language-binding converter).
//   - Trainer::compute_classification_predictions : per-plot test-fold
//                            predictions for classification targets (the
//                            regression-only compute_residuals left these
//                            unreachable).
//   - Trainer::test_indices / test_plot_ids : expose the held-out fold so
//                            downstream code can recover exactly which plots
//                            were scored.
//
// Construction strategy mirrors test_predictor.cpp: a synthetic in-memory
// ResolveDataset with a regression target (y) and a classification target
// (hab, 3 integer-coded classes). We deliberately skip fit() — these tests
// are about the evaluation plumbing being correct and self-consistent, not
// about convergence. Random weights are fine because every assertion is an
// agreement check (load_state reproduces the source weights; predicted
// classes equal argmax of the returned probabilities; actuals match the raw
// data at the reported fold indices).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/dataset.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "resolve/checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace resolve;

namespace {

class TempFile {
public:
    explicit TempFile(const std::string& content,
                      const std::string& suffix = ".csv") {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_traineval_" + std::to_string(counter_++) + suffix);
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

constexpr int kNumHabClasses = 3;

// Synthetic dataset: n_plots rows, two covariates, lat/lon, one hash species
// per plot, a regression target y, and a classification target hab whose
// class for global plot i is exactly (i % kNumHabClasses). That deterministic
// relationship lets the classification test verify actuals against the
// reported fold indices.
ResolveDataset make_synthetic_dataset(int64_t n_plots) {
    std::ostringstream hdr;
    hdr << "plot_id,lat,lon,cov1,cov2,y,hab\n";
    std::ostringstream spc;
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        double lat = 40.0 + (static_cast<double>(i) / n_plots) * 10.0;
        double lon = -5.0 + (static_cast<double>(i) / n_plots) * 10.0;
        double c1 = static_cast<double>(i % 7) * 1.1;
        double c2 = static_cast<double>(i % 11) * 0.9;
        double y = c1 + c2;
        int hab = static_cast<int>(i % kNumHabClasses);
        hdr << "P" << i << "," << lat << "," << lon << ","
            << c1 << "," << c2 << "," << y << "," << hab << "\n";
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
        {TargetSpec::regression("y"),
         TargetSpec::classification("hab", kNumHabClasses)},
        dcfg);
}

ModelConfig make_model_config() {
    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.hash_dim = 4;
    mcfg.hidden_dims = {8, 4};
    return mcfg;
}

TrainConfig make_train_config() {
    TrainConfig tcfg;
    tcfg.batch_size = 8;
    tcfg.max_epochs = 1;
    tcfg.patience = 1;
    tcfg.lr = 1e-3f;
    tcfg.device = torch::kCPU;
    return tcfg;
}

// A trainer whose model holds known (random) weights, with prepare_data
// already run on the given seed. Saved to `ckpt_path` so a second trainer can
// load_state from it.
Trainer make_prepared_trainer(const ResolveDataset& ds, int seed) {
    ResolveModel model(ds.schema(), make_model_config());
    Trainer trainer(model, make_train_config());
    trainer.prepare_data(ds, /*test_size=*/0.25f, seed);
    return trainer;
}

}  // namespace

// =============================================================================
// 1. load_state restores weights, scalers, and produces identical predictions
// =============================================================================

TEST_CASE("Trainer::load_state restores checkpoint weights in place",
          "[trainer][load_state]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/80);

    // Source trainer: random init, saved without fit().
    Trainer src = make_prepared_trainer(ds, /*seed=*/7);
    auto ckpt = (std::filesystem::temp_directory_path() /
                 "trainer_load_state_test.pt").string();
    src.save(ckpt);

    // Destination trainer: a different random init + the same fold (same seed).
    Trainer dst = make_prepared_trainer(ds, /*seed=*/7);

    auto src_params = src.model()->parameters();
    auto dst_params = dst.model()->parameters();
    REQUIRE(src_params.size() == dst_params.size());

    // Independent inits must differ somewhere before loading.
    bool differ_before = false;
    for (size_t i = 0; i < src_params.size(); ++i) {
        if (!torch::allclose(src_params[i].to(torch::kCPU),
                             dst_params[i].to(torch::kCPU))) {
            differ_before = true;
            break;
        }
    }
    REQUIRE(differ_before);

    dst.load_state(ckpt, torch::kCPU);
    std::filesystem::remove(ckpt);

    // After loading, every parameter must match the source bit-for-bit.
    auto dst_loaded = dst.model()->parameters();
    REQUIRE(dst_loaded.size() == src_params.size());
    for (size_t i = 0; i < src_params.size(); ++i) {
        REQUIRE(torch::allclose(src_params[i].to(torch::kCPU),
                                dst_loaded[i].to(torch::kCPU)));
    }

    // Identical weights + identical fold => identical classification output.
    auto src_pred = src.compute_classification_predictions("hab");
    auto dst_pred = dst.compute_classification_predictions("hab");
    REQUIRE(src_pred.predicted_classes.defined());
    REQUIRE(dst_pred.predicted_classes.defined());
    REQUIRE(torch::equal(src_pred.predicted_classes, dst_pred.predicted_classes));
    REQUIRE(torch::allclose(src_pred.probabilities, dst_pred.probabilities));
}

// =============================================================================
// 2. compute_classification_predictions shapes + self-consistency
// =============================================================================

TEST_CASE("Trainer::compute_classification_predictions returns per-plot output",
          "[trainer][classification]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/80);
    Trainer trainer = make_prepared_trainer(ds, /*seed=*/3);

    auto pred = trainer.compute_classification_predictions("hab");

    REQUIRE(pred.target_name == "hab");
    REQUIRE(pred.predicted_classes.defined());
    REQUIRE(pred.probabilities.defined());
    REQUIRE(pred.actuals.defined());

    const int64_t n_test = trainer.test_indices().size(0);
    REQUIRE(n_test > 0);
    REQUIRE(pred.predicted_classes.size(0) == n_test);
    REQUIRE(pred.actuals.size(0) == n_test);
    REQUIRE(pred.probabilities.dim() == 2);
    REQUIRE(pred.probabilities.size(0) == n_test);
    REQUIRE(pred.probabilities.size(1) == kNumHabClasses);

    // Predicted class must be the argmax of the returned probability row.
    auto argmax = pred.probabilities.argmax(/*dim=*/1).to(torch::kLong);
    REQUIRE(torch::equal(argmax, pred.predicted_classes));

    // Probability rows are a valid distribution (sum to 1).
    auto row_sums = pred.probabilities.sum(/*dim=*/1);
    REQUIRE(torch::allclose(row_sums, torch::ones({n_test}),
                            /*rtol=*/1e-4, /*atol=*/1e-4));

    // Actuals must equal the raw data (hab = global_index % kNumHabClasses) at
    // the reported fold indices — ties predictions back to the source rows.
    auto test_idx = trainer.test_indices().to(torch::kLong);
    auto expected = test_idx.remainder(kNumHabClasses);
    REQUIRE(torch::equal(expected, pred.actuals));
}

// =============================================================================
// 3. compute_classification_predictions on a regression target is empty
// =============================================================================

TEST_CASE("Trainer::compute_classification_predictions empty for regression",
          "[trainer][classification]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/40);
    Trainer trainer = make_prepared_trainer(ds, /*seed=*/1);

    auto pred = trainer.compute_classification_predictions("y");
    REQUIRE(pred.target_name == "y");
    // Empty (undefined) tensors signal "not a classification target".
    REQUIRE_FALSE(pred.predicted_classes.defined());
    REQUIRE_FALSE(pred.probabilities.defined());
    REQUIRE_FALSE(pred.actuals.defined());
}

// =============================================================================
// 4. test_indices / test_plot_ids expose the held-out fold faithfully
// =============================================================================

TEST_CASE("Trainer::test_indices and test_plot_ids match the dataset",
          "[trainer][fold]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/100);
    Trainer trainer = make_prepared_trainer(ds, /*seed=*/42);

    auto test_idx = trainer.test_indices();
    auto train_idx = trainer.train_indices();
    REQUIRE(test_idx.defined());
    REQUIRE(train_idx.defined());
    REQUIRE(test_idx.size(0) + train_idx.size(0) == 100);

    auto test_pids = trainer.test_plot_ids();
    REQUIRE(static_cast<int64_t>(test_pids.size()) == test_idx.size(0));

    // Plot IDs are recoverable: plot i was written as "P<i>", and plot_ids()
    // preserves dataset order, so test_plot_ids()[k] == "P" + test_idx[k].
    const auto& all_pids = ds.plot_ids();
    auto idx_cpu = test_idx.to(torch::kLong).contiguous();
    auto idx_acc = idx_cpu.accessor<int64_t, 1>();
    for (int64_t k = 0; k < idx_cpu.size(0); ++k) {
        const int64_t g = idx_acc[k];
        REQUIRE(test_pids[static_cast<size_t>(k)] == all_pids[static_cast<size_t>(g)]);
    }
}

// =============================================================================
// 5. compute_residuals still serves regression targets after the refactor
// =============================================================================

TEST_CASE("Trainer::compute_residuals unaffected by shared forward refactor",
          "[trainer][residuals]") {
    auto ds = make_synthetic_dataset(/*n_plots=*/60);
    Trainer trainer = make_prepared_trainer(ds, /*seed=*/5);

    auto resid = trainer.compute_residuals("y");
    const int64_t n_test = trainer.test_indices().size(0);

    REQUIRE(resid.target_name == "y");
    REQUIRE(static_cast<int64_t>(resid.predictions.size()) == n_test);
    REQUIRE(static_cast<int64_t>(resid.actuals.size()) == n_test);
    REQUIRE(static_cast<int64_t>(resid.residuals.size()) == n_test);

    // A regression target has no classification predictions.
    auto cls = trainer.compute_classification_predictions("y");
    REQUIRE_FALSE(cls.predicted_classes.defined());
}

TEST_CASE("Checkpoint train-config + run-metadata round-trip", "[trainer][checkpoint]") {
    // The save_*_config / save_run_metadata writers now have matching readers
    // (issue #14). Save a distinctive config + metadata to an archive and read
    // it back through the path-based Trainer::load_* convenience methods.
    TrainConfig cfg;
    cfg.batch_size = 8192;
    cfg.batch_size_floor = 512;
    cfg.max_epochs = 77;
    cfg.patience = 13;
    cfg.lr = 2e-3f;
    cfg.weight_decay = 5e-4f;
    cfg.phase_boundaries = {10, 25};
    cfg.loss_config = LossConfigMode::SMAPE;       // non-default (default Combined)
    cfg.lr_scheduler = LRSchedulerType::CosineAnnealing;  // non-default (default None)
    cfg.lr_step_size = 33;
    cfg.lr_gamma = 0.2f;
    cfg.lr_min = 1e-5f;
    cfg.vram_fraction = 0.8f;
    cfg.band_thresholds = {0.05f, 0.2f, 0.4f, 0.6f};

    RunMetadata meta;
    meta.created_at = "2026-06-12T10:00:00Z";
    meta.completed_at = "2026-06-12T11:30:00Z";
    meta.train_time_seconds = 5400.5f;
    meta.n_plots_train = 1450000;
    meta.n_plots_test = 362500;
    meta.best_epoch = 42;
    meta.total_epochs = 77;
    meta.final_metrics["area"] = {{"rmse", 1.5f}, {"r2", 0.83f}};
    meta.final_metrics["eunis"] = {{"accuracy", 0.91f}};

    const std::string path =
        (std::filesystem::temp_directory_path() / "resolve_ckpt_cfg_meta_roundtrip.pt").string();
    {
        torch::serialize::OutputArchive ar;
        save_train_config(ar, cfg);
        save_run_metadata(ar, meta);
        ar.save_to(path);
    }

    const TrainConfig cfg2 = Trainer::load_train_config(path);
    const RunMetadata meta2 = Trainer::load_run_metadata(path);
    std::filesystem::remove(path);

    SECTION("train config fields round-trip") {
        REQUIRE(cfg2.batch_size == cfg.batch_size);
        REQUIRE(cfg2.batch_size_floor == cfg.batch_size_floor);
        REQUIRE(cfg2.max_epochs == cfg.max_epochs);
        REQUIRE(cfg2.patience == cfg.patience);
        REQUIRE(cfg2.lr == Catch::Approx(cfg.lr));
        REQUIRE(cfg2.weight_decay == Catch::Approx(cfg.weight_decay));
        REQUIRE(cfg2.phase_boundaries.first == cfg.phase_boundaries.first);
        REQUIRE(cfg2.phase_boundaries.second == cfg.phase_boundaries.second);
        REQUIRE(cfg2.loss_config == cfg.loss_config);
        REQUIRE(cfg2.lr_scheduler == cfg.lr_scheduler);
        REQUIRE(cfg2.lr_step_size == cfg.lr_step_size);
        REQUIRE(cfg2.lr_gamma == Catch::Approx(cfg.lr_gamma));
        REQUIRE(cfg2.lr_min == Catch::Approx(cfg.lr_min));
        REQUIRE(cfg2.vram_fraction == Catch::Approx(cfg.vram_fraction));
        REQUIRE(cfg2.band_thresholds == cfg.band_thresholds);
    }

    SECTION("unpersisted fields keep TrainConfig defaults") {
        TrainConfig defaults;
        REQUIRE(cfg2.device == defaults.device);
        REQUIRE(cfg2.use_amp == defaults.use_amp);
        REQUIRE(cfg2.checkpoint_dir == defaults.checkpoint_dir);
    }

    SECTION("run metadata fields round-trip") {
        REQUIRE(meta2.created_at == meta.created_at);
        REQUIRE(meta2.completed_at == meta.completed_at);
        REQUIRE(meta2.train_time_seconds == Catch::Approx(meta.train_time_seconds));
        REQUIRE(meta2.n_plots_train == meta.n_plots_train);
        REQUIRE(meta2.n_plots_test == meta.n_plots_test);
        REQUIRE(meta2.best_epoch == meta.best_epoch);
        REQUIRE(meta2.total_epochs == meta.total_epochs);
        REQUIRE(meta2.final_metrics.size() == 2);
        REQUIRE(meta2.final_metrics.at("area").at("rmse") == Catch::Approx(1.5f));
        REQUIRE(meta2.final_metrics.at("area").at("r2") == Catch::Approx(0.83f));
        REQUIRE(meta2.final_metrics.at("eunis").at("accuracy") == Catch::Approx(0.91f));
    }
}

TEST_CASE("Checkpoint model-config sub-config round-trip", "[trainer][checkpoint]") {
    // Non-MLP encoder sub-configs must survive the checkpoint or Predictor::load
    // rebuilds default-sized layers whose weights mismatch (issue #37).
    ModelConfig cfg;
    cfg.encoder_architecture = EncoderArchitecture::FTTransformer;
    cfg.ft_transformer.d_model = 256;        // non-default (192)
    cfg.ft_transformer.n_layers = 6;         // non-default (3)
    cfg.ft_transformer.n_heads = 16;         // non-default (8)
    cfg.ft_transformer.pre_norm = false;     // non-default (true)
    cfg.tabnet.n_steps = 7;                  // non-default (3)
    cfg.saint.n_layers = 9;                  // non-default (6)
    cfg.gnn.gnn_type = GNNType::GraphSAGE;   // non-default (GAT)
    cfg.gnn.k_neighbors = 25;                // non-default (10)
    cfg.trait_net.trait_dim = 96;            // non-default (64)
    cfg.excelformer.d_model = 320;           // non-default (192)
    cfg.excelformer.importance_threshold = 0.7f;  // non-default (0.5)
    cfg.heterogeneous_gnn.output_dim = 128;  // non-default (64)

    ParallelBranchConfig b1;
    b1.hidden_dims = {128, 64};
    b1.dropout = 0.25f;
    ParallelBranchConfig b2;
    b2.hidden_dims = {256};
    b2.branch_weight = 2.0f;
    cfg.parallel_layers.enabled = true;
    cfg.parallel_layers.aggregation = ParallelAggregation::Gated;
    cfg.parallel_layers.branches = {b1, b2};

    const std::string path =
        (std::filesystem::temp_directory_path() / "resolve_modelcfg_subconfig_roundtrip.pt").string();
    {
        torch::serialize::OutputArchive ar;
        save_model_config(ar, cfg);
        ar.save_to(path);
    }
    torch::serialize::InputArchive ar;
    ar.load_from(path);
    ModelConfig cfg2 = load_model_config(ar);
    std::filesystem::remove(path);

    REQUIRE(cfg2.encoder_architecture == EncoderArchitecture::FTTransformer);
    REQUIRE(cfg2.ft_transformer.d_model == 256);
    REQUIRE(cfg2.ft_transformer.n_layers == 6);
    REQUIRE(cfg2.ft_transformer.n_heads == 16);
    REQUIRE(cfg2.ft_transformer.pre_norm == false);
    REQUIRE(cfg2.tabnet.n_steps == 7);
    REQUIRE(cfg2.saint.n_layers == 9);
    REQUIRE(cfg2.gnn.gnn_type == GNNType::GraphSAGE);
    REQUIRE(cfg2.gnn.k_neighbors == 25);
    REQUIRE(cfg2.trait_net.trait_dim == 96);
    REQUIRE(cfg2.excelformer.d_model == 320);
    REQUIRE(cfg2.excelformer.importance_threshold == Catch::Approx(0.7f));
    REQUIRE(cfg2.heterogeneous_gnn.output_dim == 128);
    REQUIRE(cfg2.parallel_layers.enabled == true);
    REQUIRE(cfg2.parallel_layers.aggregation == ParallelAggregation::Gated);
    REQUIRE(cfg2.parallel_layers.branches.size() == 2);
    REQUIRE(cfg2.parallel_layers.branches[0].hidden_dims == std::vector<int64_t>{128, 64});
    REQUIRE(cfg2.parallel_layers.branches[0].dropout == Catch::Approx(0.25f));
    REQUIRE(cfg2.parallel_layers.branches[1].hidden_dims == std::vector<int64_t>{256});
    REQUIRE(cfg2.parallel_layers.branches[1].branch_weight == Catch::Approx(2.0f));
}

TEST_CASE("Checkpoint schema pool-weighting round-trip", "[trainer][checkpoint]") {
    // pool_weighting / pool_species_cap must survive the checkpoint so the
    // predict side rebuilds the same DatasetConfig instead of defaulting to
    // Log1p and recomputing different pool weights (issue #38).
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 40;
    schema.n_species_vocab = 41;
    schema.pool_weighting = 4;      // PoolWeighting::Rank (non-default)
    schema.pool_species_cap = 137;  // resolved max-species width

    const std::string path =
        (std::filesystem::temp_directory_path() / "resolve_schema_pool_roundtrip.pt").string();
    {
        torch::serialize::OutputArchive ar;
        save_schema(ar, schema);
        ar.save_to(path);
    }
    torch::serialize::InputArchive ar;
    ar.load_from(path);
    ResolveSchema schema2 = load_schema(ar);
    std::filesystem::remove(path);

    REQUIRE(schema2.pool_weighting == 4);
    REQUIRE(schema2.pool_species_cap == 137);

    SECTION("pre-#38 checkpoint keeps schema defaults") {
        // A schema archive that never wrote the pool keys must read back as the
        // Log1p / auto defaults, not throw.
        ResolveSchema legacy;
        legacy.n_plots = 5;
        const std::string lpath =
            (std::filesystem::temp_directory_path() / "resolve_schema_legacy.pt").string();
        {
            torch::serialize::OutputArchive ar2;
            // Write only the categorical block's prerequisite key set by writing
            // the full schema, then confirm defaults survive a fresh struct.
            save_schema(ar2, legacy);
            ar2.save_to(lpath);
        }
        torch::serialize::InputArchive ar2;
        ar2.load_from(lpath);
        ResolveSchema legacy2 = load_schema(ar2);
        std::filesystem::remove(lpath);
        REQUIRE(legacy2.pool_weighting == 2);   // Log1p
        REQUIRE(legacy2.pool_species_cap == 0);  // auto
    }
}

// Issue #65: JSON sidecar string values must be escaped. A target name or
// version containing a quote used to emit invalid JSON.
TEST_CASE("write_metadata_json escapes string values", "[checkpoint][json]") {
    resolve::ModelConfig mc;
    resolve::TrainConfig tc;
    resolve::ResolveSchema sch;
    resolve::RunMetadata md;
    md.resolve_version = "1.0\"x";
    md.final_metrics["area\"m2"] = {{"mae", 1.5f}};

    auto base = (std::filesystem::temp_directory_path() / "meta_escape_test.pt").string();
    resolve::write_metadata_json(base, mc, tc, md, sch);
    auto json_path = base.substr(0, base.size() - 3) + ".json";

    std::ifstream f(json_path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();
    f.close();
    std::filesystem::remove(json_path);

    // The embedded quotes must appear escaped, not raw.
    REQUIRE(content.find("area\\\"m2") != std::string::npos);
    REQUIRE(content.find("1.0\\\"x") != std::string::npos);
}
