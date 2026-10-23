// Parameter-recovery tests for the fitter.
//
// Each case simulates data with a known relationship, fits to convergence on
// CPU, and asserts the trained model recovers it on the held-out fold, well
// above chance: the statistical counterpart to the plumbing tests (shapes,
// dtypes, checkpoints, encoder equivalence) and the pure metric functions.
//
// Kept small and seeded so they run in a few seconds and are deterministic.
// Seeded means BOTH halves: prepare_data's seed fixes the split, and
// seed_recovery() below fixes the weight initialisation, which draws from the
// process-global torch RNG. Without the second one a threshold is evaluated on
// a different random draw every run.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "resolve/dataset.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "resolve/pretraining.hpp"
#include "resolve/vae.hpp"
#include "test_support.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

// Fix the weights every recovery test starts from.
//
// These tests fit to convergence and assert a correlation or accuracy
// threshold, so they need the same starting point every run. The seed passed
// to prepare_data covers the SPLIT; model weight initialisation draws from
// the process-global torch RNG, so without this a recovered correlation moves
// from run to run and a threshold can miss by a hair on one CI leg and pass on
// the next. Call this BEFORE constructing the model, which is where the draw
// happens.
void seed_recovery(uint64_t seed = 20260830) { torch::manual_seed(seed); }

// Header (plot_id,lat,lon,cov1,cov2,y) + species (plot_id,sp,cover) pair with
// the deterministic linear signal y = 2 cov1 - 1.5 cov2 + 3 and a neutral
// species pathway (one of eight species per plot, presence only).
void make_linear_signal_csv(std::ostringstream& hdr, std::ostringstream& spc,
                            int64_t n_plots) {
    hdr << "plot_id,lat,lon,cov1,cov2,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(i * 0.13);
        const double c2 = std::cos(i * 0.07);
        const double y = 2.0 * c1 - 1.5 * c2 + 3.0;
        hdr << "P" << i << "," << (40.0 + i * 0.001) << "," << (-5.0 + i * 0.001)
            << "," << c1 << "," << c2 << "," << y << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
}

// Header (plot_id,y) + species (plot_id,sp,cover) pair where y is the sum of
// per-species contributions over each plot's (fixed-size, three-species) set.
void make_species_signal_csv(std::ostringstream& hdr, std::ostringstream& spc,
                             int64_t n_plots, int n_species) {
    std::vector<double> species_value(n_species);
    for (int s = 0; s < n_species; ++s) species_value[s] = std::sin(s * 1.7) * 2.0;
    hdr << "plot_id,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        int s0 = static_cast<int>(i % n_species);
        int s1 = static_cast<int>((i / n_species) % n_species);
        int s2 = static_cast<int>((i / (n_species * n_species) + i) % n_species);
        double y = species_value[s0] + species_value[s1] + species_value[s2];
        hdr << "P" << i << "," << y << "\n";
        spc << "P" << i << ",sp" << s0 << ",1.0\n";
        spc << "P" << i << ",sp" << s1 << ",1.0\n";
        spc << "P" << i << ",sp" << s2 << ",1.0\n";
    }
}

}  // namespace

// =============================================================================
// 1. Regression head recovers a known linear signal in the covariates.
// =============================================================================
TEST_CASE("Regression recovers a known covariate signal", "[recovery][regression]") {
    seed_recovery();
    const int64_t n_plots = 600;
    std::ostringstream hdr, spc;
    make_linear_signal_csv(hdr, spc, n_plots);
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles().coordinates().covariates({"cov1", "cov2"});

    const DatasetConfig dcfg = hash_dataset_config();

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg = hash_mlp_config({32, 16});

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
    seed_recovery();
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

    const RoleMapping roles = Roles().coordinates().covariates({"cov1", "cov2"});

    const DatasetConfig dcfg = hash_dataset_config();

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::classification("hab", n_classes)}, dcfg);

    ModelConfig mcfg = hash_mlp_config({32, 16});

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
    seed_recovery();
    const int64_t n_plots = 600;
    const int n_species = 12;
    std::ostringstream hdr, spc;
    make_species_signal_csv(hdr, spc, n_plots, n_species);
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles();

    const DatasetConfig dcfg = hash_dataset_config(32);

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg = hash_mlp_config({64, 32}, 32);

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
// 4. fit() persists run metadata to the checkpoint: load_run_metadata recovers
//    the run's best epoch / total epochs / train time / metrics.
// =============================================================================
TEST_CASE("fit persists run metadata to the checkpoint", "[recovery][metadata]") {
    seed_recovery();
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

    const RoleMapping roles = Roles().coordinates().covariates({"cov1"});

    const DatasetConfig dcfg = hash_dataset_config();

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg = hash_mlp_config({16, 8});

    const TempDir ckpt_dir;

    TrainConfig tcfg = recovery_train_config(/*max_epochs=*/30);
    tcfg.checkpoint_dir = ckpt_dir.path();

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, tcfg);
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/7);
    auto result = trainer.fit();

    const std::string ckpt = ckpt_dir.file("checkpoint.pt");
    REQUIRE(std::filesystem::exists(ckpt));

    const RunMetadata meta = Trainer::load_run_metadata(ckpt);
    // fit() fills every field in; none is left at its default.
    REQUIRE(meta.total_epochs > 0);
    REQUIRE(meta.total_epochs == static_cast<int>(result.train_loss_history.size()));
    REQUIRE(meta.best_epoch == result.best_epoch);
    REQUIRE(meta.train_time_seconds > 0.0f);
    REQUIRE(meta.n_plots_train > 0);
    REQUIRE(meta.n_plots_test > 0);
    REQUIRE(meta.final_metrics.count("y") == 1);
    REQUIRE_FALSE(meta.created_at.empty());

    // JSON sidecar is written alongside the checkpoint.
    REQUIRE(std::filesystem::exists(ckpt_dir.file("checkpoint.json")));
}

// =============================================================================
// 5-6. Rank-pool and transformer species encoders recover a composition signal:
//      the target is a deterministic weighted-pool function of a plot's species,
//      so an encoder that actually learns composition recovers it on held-out
//      plots, well above the ~0 a broken pooling path would give.
// =============================================================================

TEST_CASE("Rank-pool encoder recovers a species-driven target", "[recovery][rank_pool]") {
    seed_recovery();
    const int64_t n_plots = 600;
    const int n_species = 12;
    std::ostringstream hdr, spc;
    make_species_signal_csv(hdr, spc, n_plots, n_species);
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles();

    const DatasetConfig dcfg = bare_dataset_config(SpeciesEncodingMode::RankPool);

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::RankPool;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.species_embed_dim = 16;
    mcfg.hidden_dims = {64, 32};

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, recovery_train_config(/*max_epochs=*/400));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/7);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() > 10);
    REQUIRE(pearson(res.predictions, res.actuals) > 0.8);
}

TEST_CASE("Transformer encoder recovers a species-driven target", "[recovery][transformer]") {
    seed_recovery();
    const int64_t n_plots = 600;
    const int n_species = 12;
    std::ostringstream hdr, spc;
    make_species_signal_csv(hdr, spc, n_plots, n_species);
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles();

    const DatasetConfig dcfg = bare_dataset_config(SpeciesEncodingMode::Transformer);

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Transformer;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.d_model = 32;
    mcfg.n_heads = 4;
    mcfg.n_attention_layers = 1;
    mcfg.transformer_ff_dim = 64;
    mcfg.transformer_pooling = "attention";
    mcfg.hidden_dims = {32};

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, recovery_train_config(/*max_epochs=*/400));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/7);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() > 10);
    REQUIRE(pearson(res.predictions, res.actuals) > 0.7);
}

// =============================================================================
// 7. JEPA target encoder BatchNorm buffers track the online encoder.
//    The target encoder runs in eval(), so if its BN running stats were left at
//    the construction-time init (mean 0, var 1) while the online encoder's BN
//    adapts to the data, the "target = slow copy of online" invariant breaks.
//    update_target_encoder must copy buffers, not only EMA the parameters.
// =============================================================================
TEST_CASE("JEPA target encoder syncs BatchNorm buffers", "[recovery][jepa]") {
    seed_recovery();
    const int64_t n_plots = 256;
    std::ostringstream hdr, spc;
    hdr << "plot_id,cov1,cov2,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        // Large offset so the online encoder's BN running_mean moves clearly away
        // from the init value of 0, making the buffer-sync check meaningful.
        const double c1 = 10.0 + std::sin(i * 0.13);
        const double c2 = -7.0 + std::cos(i * 0.07);
        hdr << "P" << i << "," << c1 << "," << c2 << "," << (c1 - c2) << "\n";
        spc << "P" << i << ",sp" << (i % 6) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles().covariates({"cov1", "cov2"});

    const DatasetConfig dcfg = hash_dataset_config();

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg = hash_mlp_config({16, 8});
    mcfg.normalization = NormLayerType::BatchNorm;  // default; BN has running buffers

    ResolveModel model(ds.schema(), mcfg);

    PretrainConfig pcfg;
    pcfg.pretrain_epochs = 3;
    pcfg.batch_size = 32;
    pcfg.device = torch::kCPU;

    JEPAPretrainer pretrainer(model, pcfg);
    // Hash-mode get_latent expects the hash embedding folded into continuous.
    auto cont = ds.continuous_block(/*include_hash=*/true);
    pretrainer.pretrain(cont);

    // Find a BatchNorm running_mean buffer and compare online vs target.
    auto ctx_bufs = pretrainer.model()->named_buffers();
    auto tgt_bufs = pretrainer.target_encoder()->named_buffers();
    bool checked = false;
    for (const auto& b : ctx_bufs) {
        if (b.key().find("running_mean") == std::string::npos) continue;
        auto tgt = tgt_bufs.find(b.key());
        REQUIRE(tgt != nullptr);
        // Online BN stats actually moved from the init (mean 0)...
        REQUIRE(b.value().abs().sum().item<float>() > 1e-3f);
        // ...and the target encoder's buffers were synced to them, not left at
        // init.
        REQUIRE(torch::allclose(b.value(), *tgt, 1e-5, 1e-6));
        checked = true;
    }
    REQUIRE(checked);  // the model must actually contain BatchNorm buffers
}

// =============================================================================
// 8. cudnn_benchmark=false survives the CUDA training loop: cache_data_to_gpu()
//    must not re-enable the cuDNN auto-tuner over the determinism knob fit()
//    set from config. The global benchmark flag stays off across a CUDA fit
//    that goes through cache_data_to_gpu().
// =============================================================================
TEST_CASE("cudnn_benchmark=false is not overridden by the CUDA cache path",
          "[recovery][cuda]") {
    seed_recovery();
    if (!torch::cuda::is_available()) {
        SKIP("No CUDA device available");
    }

    const int64_t n_plots = 256;
    std::ostringstream hdr, spc;
    make_linear_signal_csv(hdr, spc, n_plots);
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles().coordinates().covariates({"cov1", "cov2"});

    const DatasetConfig dcfg = hash_dataset_config();

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg = hash_mlp_config({16, 8});

    ResolveModel model(ds.schema(), mcfg);

    TrainConfig tcfg;
    tcfg.batch_size = 64;
    tcfg.max_epochs = 3;
    tcfg.patience = 5;
    tcfg.device = torch::kCUDA;
    tcfg.cudnn_benchmark = false;  // request deterministic cuDNN

    // Poison the global flag so a failure to honor the config is visible.
    at::globalContext().setBenchmarkCuDNN(true);

    Trainer trainer(model, tcfg);
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/11);
    trainer.fit();  // goes through cache_data_to_gpu() on the CUDA path

    REQUIRE_FALSE(at::globalContext().benchmarkCuDNN());
}

// =============================================================================
// 9. GNN trains full-batch and embeds taxonomy.
//    The coordinate-kNN GNN forces full-batch training
//    (requires_full_batch_training) so its spatial graph spans all plots rather
//    than an arbitrary mini-batch, and it embeds genus/family IDs instead of
//    feeding raw integers as magnitudes. This trains a GNN end-to-end and asserts
//    it recovers a per-plot covariate signal on the held-out fold (self-loops let
//    each node carry its own features, and spatial neighbors share the signal).
// =============================================================================
TEST_CASE("GNN trains full-batch and recovers a covariate signal", "[recovery][gnn]") {
    seed_recovery();
    const int64_t n_plots = 240;
    std::ostringstream hdr, spc;
    hdr << "plot_id,lat,lon,cov1,cov2,y\n";
    spc << "plot_id,sp,cover,genus,family\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(i * 0.13);
        const double c2 = std::cos(i * 0.07);
        const double y = 2.0 * c1 - 1.5 * c2 + 3.0;  // deterministic linear signal
        hdr << "P" << i << "," << (40.0 + i * 0.01) << "," << (-5.0 + i * 0.017)
            << "," << c1 << "," << c2 << "," << y << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0,g" << (i % 4) << ",f" << (i % 2) << "\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles().taxonomy().coordinates().covariates({"cov1", "cov2"});

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4; dcfg.use_taxonomy = true;
    dcfg.track_unknown_fraction = false; dcfg.track_unknown_count = false;

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);
    REQUIRE(ds.schema().has_taxonomy);       // taxonomy embeddings are exercised

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::GNN;
    mcfg.hash_dim = 4;
    mcfg.genus_emb_dim = 4; mcfg.family_emb_dim = 4;
    mcfg.n_taxonomy_slots = 3;
    mcfg.gnn.hidden_dim = 32;
    mcfg.gnn.n_layers = 2;
    mcfg.gnn.k_neighbors = 8;

    ResolveModel model(ds.schema(), mcfg);
    REQUIRE(model->requires_full_batch_training());

    Trainer trainer(model, recovery_train_config(120));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/11);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() > 10);
    for (float p : res.predictions) REQUIRE(std::isfinite(p));
    // Learned the signal well above chance on the held-out fold.
    REQUIRE(pearson(res.predictions, res.actuals) > 0.5);
}

// =============================================================================
// Parameter recovery for non-MLP architectures and non-Hash species encoders,
// model-level calibration/coverage, categorical recovery, a real CV run, and a
// pretext-learning check.
// =============================================================================

namespace {

// Build the standard linear-covariate-signal dataset (Hash species pathway kept
// neutral), fit `mcfg`, and return held-out Pearson(pred, actual) for "y". The
// signal lives entirely in cov1/cov2, so any encoder architecture that routes
// the continuous features correctly recovers it; a broken adapter (ignores its
// input / wrong feature routing) collapses to ~0.
double covariate_recovery_pearson(const ModelConfig& mcfg, int max_epochs = 150,
                                  int seed = 11) {
    const int64_t n_plots = 300;
    std::ostringstream hdr, spc;
    make_linear_signal_csv(hdr, spc, n_plots);
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles().coordinates().covariates({"cov1", "cov2"});

    const DatasetConfig dcfg = hash_dataset_config();

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, recovery_train_config(max_epochs));
    trainer.prepare_data(ds, /*test_size=*/0.25f, seed);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() > 10);
    for (float p : res.predictions) REQUIRE(std::isfinite(p));
    return pearson(res.predictions, res.actuals);
}

}  // namespace

// -----------------------------------------------------------------------------
// P1: every non-MLP TABULAR architecture recovers the covariate signal, which a
// shape or gradient check cannot show for a subtly broken adapter. (GNN and the
// full-batch graph path are covered by the dedicated GNN test above.)
// -----------------------------------------------------------------------------
TEST_CASE("Non-MLP tabular architectures recover a covariate signal",
          "[recovery][architecture]") {
    seed_recovery();
    auto base = []() {
        ModelConfig mcfg;
        mcfg.species_encoding = SpeciesEncodingMode::Hash;
        mcfg.hash_dim = 4;
        mcfg.hidden_dims = {32, 16};
        return mcfg;
    };

    SECTION("FTTransformer") {
        ModelConfig mcfg = base();
        mcfg.encoder_architecture = EncoderArchitecture::FTTransformer;
        mcfg.ft_transformer.d_model = 24;
        mcfg.ft_transformer.n_heads = 4;
        mcfg.ft_transformer.n_layers = 1;
        REQUIRE(covariate_recovery_pearson(mcfg) > 0.75);
    }
    SECTION("TabNet") {
        ModelConfig mcfg = base();
        mcfg.encoder_architecture = EncoderArchitecture::TabNet;
        mcfg.tabnet.n_d = 12;
        mcfg.tabnet.n_a = 12;
        mcfg.tabnet.n_steps = 2;
        REQUIRE(covariate_recovery_pearson(mcfg) > 0.75);
    }
    SECTION("SAINT") {
        ModelConfig mcfg = base();
        mcfg.encoder_architecture = EncoderArchitecture::SAINT;
        mcfg.saint.d_model = 24;
        mcfg.saint.n_heads = 4;
        mcfg.saint.n_layers = 1;
        REQUIRE(covariate_recovery_pearson(mcfg) > 0.75);
    }
    SECTION("ExcelFormer") {
        ModelConfig mcfg = base();
        mcfg.encoder_architecture = EncoderArchitecture::ExcelFormer;
        mcfg.excelformer.d_model = 24;
        mcfg.excelformer.n_heads = 4;
        mcfg.excelformer.n_layers = 1;
        REQUIRE(covariate_recovery_pearson(mcfg) > 0.75);
    }
}

// -----------------------------------------------------------------------------
// P2: embed and sparse species-encoding modes carry the composition signal
// end-to-end, the embed one through EmbeddingEncoder.
// -----------------------------------------------------------------------------
TEST_CASE("Embed encoder recovers a species-driven target", "[recovery][embed]") {
    seed_recovery();
    const int64_t n_plots = 600;
    const int n_species = 12;
    std::ostringstream hdr, spc;
    make_species_signal_csv(hdr, spc, n_plots, n_species);
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles();

    DatasetConfig dcfg = bare_dataset_config(SpeciesEncodingMode::Embed);
    dcfg.top_k_species = 5;

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Embed;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.species_embed_dim = 16;
    mcfg.top_k_species = 5;
    mcfg.hidden_dims = {64, 32};

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, recovery_train_config(/*max_epochs=*/400));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/7);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() > 10);
    REQUIRE(pearson(res.predictions, res.actuals) > 0.7);
}

TEST_CASE("Sparse encoder recovers a species-driven target", "[recovery][sparse]") {
    seed_recovery();
    const int64_t n_plots = 600;
    const int n_species = 12;
    std::ostringstream hdr, spc;
    make_species_signal_csv(hdr, spc, n_plots, n_species);
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles();

    const DatasetConfig dcfg = bare_dataset_config(SpeciesEncodingMode::Sparse);

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Sparse;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.hidden_dims = {64, 32};

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, recovery_train_config(/*max_epochs=*/400));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/7);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() > 10);
    REQUIRE(pearson(res.predictions, res.actuals) > 0.7);
}

// -----------------------------------------------------------------------------
// P3: model-level calibration / coverage. On a cleanly separable class signal a
// trained model should be confident AND accurate, so its reliability curve is
// well-calibrated (low ECE) and its high-confidence predictions are almost all
// correct (empirical accuracy tracks confidence). For regression, the realized
// band coverage on a learnable signal is high.
// -----------------------------------------------------------------------------
TEST_CASE("Classification predictions are calibrated on a separable signal",
          "[recovery][calibration]") {
    seed_recovery();
    const int64_t n_plots = 600;
    const int n_classes = 3;
    std::ostringstream hdr, spc;
    hdr << "plot_id,cov1,cov2,hab\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(i * 0.13);
        const double c2 = std::cos(i * 0.07);
        int hab = (c1 < -0.33) ? 0 : (c1 < 0.33 ? 1 : 2);
        hdr << "P" << i << "," << c1 << "," << c2 << "," << hab << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles().covariates({"cov1", "cov2"});

    const DatasetConfig dcfg = hash_dataset_config();

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::classification("hab", n_classes)}, dcfg);

    ModelConfig mcfg = hash_mlp_config({32, 16});

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, recovery_train_config());
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/11);
    trainer.fit();

    // Reliability curve: bins sum to the test set, ECE is finite and low.
    auto cal = trainer.compute_calibration("hab", /*n_bins=*/10);
    REQUIRE_FALSE(cal.bins.empty());
    int64_t binned = 0;
    for (const auto& b : cal.bins) binned += b.count;
    auto pred = trainer.compute_classification_predictions("hab");
    const int64_t n_test = pred.predicted_classes.size(0);
    REQUIRE(binned == n_test);
    REQUIRE(std::isfinite(cal.expected_calibration_error));
    REQUIRE(cal.expected_calibration_error < 0.3f);

    // Empirical accuracy tracks confidence: bin held-out predictions by their
    // max softmax probability and require the high-confidence bin (>= 0.8) to be
    // almost always correct.
    auto probs = pred.probabilities.cpu().contiguous();   // (n_test, n_classes)
    auto max_probs = std::get<0>(probs.max(1)).contiguous();
    auto pred_cls = probs.argmax(1);
    auto correct = (pred_cls == pred.actuals).to(torch::kBool).contiguous();
    auto mp = max_probs.accessor<float, 1>();
    auto cc = correct.accessor<bool, 1>();
    int64_t hi_conf = 0, hi_conf_correct = 0;
    for (int64_t s = 0; s < n_test; ++s) {
        if (mp[s] >= 0.8f) { ++hi_conf; if (cc[s]) ++hi_conf_correct; }
    }
    REQUIRE(hi_conf > 0);
    REQUIRE(static_cast<double>(hi_conf_correct) / hi_conf > 0.85);
}

TEST_CASE("Regression band coverage is high on a learnable signal",
          "[recovery][coverage]") {
    seed_recovery();
    const int64_t n_plots = 600;
    std::ostringstream hdr, spc;
    hdr << "plot_id,cov1,cov2,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(i * 0.13);
        const double c2 = std::cos(i * 0.07);
        // Positive target (band coverage is a relative-error criterion) with a
        // small deterministic perturbation so predictions are not exact.
        const double y = 10.0 + 2.0 * c1 + 1.5 * c2 + 0.05 * std::sin(i * 0.5);
        hdr << "P" << i << "," << c1 << "," << c2 << "," << y << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles().covariates({"cov1", "cov2"});

    const DatasetConfig dcfg = hash_dataset_config();

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg = hash_mlp_config({32, 16});

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, recovery_train_config());
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/11);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() > 10);
    // Realized coverage of the 25% relative-error band on the held-out fold.
    int64_t within = 0;
    for (size_t i = 0; i < res.predictions.size(); ++i) {
        const float denom = std::abs(res.actuals[i]) + 1e-6f;
        if (std::abs(res.predictions[i] - res.actuals[i]) / denom <= 0.25f) ++within;
    }
    const double coverage = static_cast<double>(within) / res.predictions.size();
    REQUIRE(coverage > 0.9);
}

// -----------------------------------------------------------------------------
// P4: categorical-covariate recovery. The target is a function of the category
// alone, so a CategoricalEmbedder that fuses zeros (or is ignored) collapses to
// the mean; a working one recovers the per-category value.
// -----------------------------------------------------------------------------
TEST_CASE("Categorical covariate alone recovers the target",
          "[recovery][categorical]") {
    seed_recovery();
    const int64_t n_plots = 600;
    const char* regions[] = {"north", "south", "east", "west"};
    const double region_value[] = {1.0, 4.0, 7.0, 10.0};
    std::ostringstream hdr, spc;
    hdr << "plot_id,cov1,region,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const int r = static_cast<int>(i % 4);
        // y depends ONLY on region; cov1 is uninformative noise-like oscillation.
        const double y = region_value[r];
        hdr << "P" << i << "," << std::sin(i * 0.31) << "," << regions[r] << "," << y << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles().covariates({"cov1"}).categoricals({"region"});

    const DatasetConfig dcfg = hash_dataset_config();

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);
    REQUIRE(ds.schema().categorical_names.size() == 1);

    ModelConfig mcfg = hash_mlp_config({32, 16});
    mcfg.categorical_embed_dim = 8;

    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, recovery_train_config());
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/11);
    trainer.fit();

    auto res = trainer.compute_residuals("y");
    REQUIRE(res.predictions.size() > 10);
    REQUIRE(pearson(res.predictions, res.actuals) > 0.9);
}

// -----------------------------------------------------------------------------
// P5: cross_validate / cross_validate_spatial actually train folds and return
// sensible per-fold metrics, and leave the trainer's post-CV split intact so the
// checkpoint evaluators still work (the restore-split and per-fold pristine
// reset, end to end).
// -----------------------------------------------------------------------------
TEST_CASE("cross_validate trains folds and restores post-CV state",
          "[recovery][cv]") {
    seed_recovery();
    const int64_t n_plots = 400;
    std::ostringstream hdr, spc;
    hdr << "plot_id,lat,lon,cov1,cov2,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = std::sin(i * 0.13);
        const double c2 = std::cos(i * 0.07);
        const double y = 2.0 * c1 - 1.5 * c2 + 3.0;
        // Well-distributed coordinates: a 10x10 degree grid so spatial CV has
        // far more blocks than folds.
        const double lat = 40.0 + static_cast<double>(i % 10);
        const double lon = -5.0 + static_cast<double>((i / 10) % 10);
        hdr << "P" << i << "," << lat << "," << lon << "," << c1 << "," << c2
            << "," << y << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles().coordinates().covariates({"cov1", "cov2"});

    const DatasetConfig dcfg = hash_dataset_config();

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg = hash_mlp_config({32, 16});

    SECTION("random k-fold") {
        ResolveModel model(ds.schema(), mcfg);
        Trainer trainer(model, recovery_train_config(/*max_epochs=*/60));
        trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/11);

        auto cv = trainer.cross_validate(/*n_folds=*/3, /*seed=*/7);
        REQUIRE(cv.n_folds == 3);
        REQUIRE(cv.fold_results.size() == 3);
        REQUIRE(cv.mean_metrics.count("y") == 1);
        for (const auto& [name, value] : cv.mean_metrics.at("y")) {
            INFO("metric " << name);
            REQUIRE(std::isfinite(value));
        }
        // Post-CV state restored: the held-out evaluator still runs on the
        // original split rather than the last fold's leftover state.
        auto res = trainer.compute_residuals("y");
        REQUIRE(res.predictions.size() > 10);
        for (float p : res.predictions) REQUIRE(std::isfinite(p));
    }

    SECTION("spatial block CV") {
        ResolveModel model(ds.schema(), mcfg);
        Trainer trainer(model, recovery_train_config(/*max_epochs=*/60));
        trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/11);

        SpatialBlockConfig scfg;
        scfg.lat_size = 1.0f; scfg.lon_size = 1.0f; scfg.balance = true;
        auto cv = trainer.cross_validate_spatial(scfg, /*n_folds=*/3, /*seed=*/7);
        REQUIRE(cv.n_folds == 3);
        REQUIRE(cv.mean_metrics.count("y") == 1);
        for (const auto& [name, value] : cv.mean_metrics.at("y")) {
            REQUIRE(std::isfinite(value));
        }
    }
}

// -----------------------------------------------------------------------------
// P6: a pretrainer actually learns its pretext task (not just "loss is finite").
// The VAE reconstruction loss must fall over pretraining; a decoder that ignores
// the latent (or a broken ELBO) leaves it flat.
// -----------------------------------------------------------------------------
TEST_CASE("VAE pretraining reduces reconstruction loss", "[recovery][pretrain][vae]") {
    seed_recovery();
    const int64_t n_plots = 400;
    const int n_species = 16;
    std::ostringstream hdr, spc;
    hdr << "plot_id,y\n";
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        hdr << "P" << i << ",0.0\n";
        // Structured composition (correlated species) so there is something to
        // reconstruct beyond noise.
        const int base = static_cast<int>(i % 4) * 4;
        for (int j = 0; j < 4; ++j) {
            spc << "P" << i << ",sp" << (base + j) % n_species << ",1.0\n";
        }
    }
    TempFile header_csv(hdr.str()), species_csv(spc.str());

    const RoleMapping roles = Roles();

    const DatasetConfig dcfg = bare_dataset_config(SpeciesEncodingMode::Sparse);  // populates species_vector()

    auto ds = ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                       {TargetSpec::regression("y")}, dcfg);
    REQUIRE(ds.species_vector().defined());
    const int64_t vocab = ds.schema().n_species_vocab;

    VAEConfig vcfg;
    vcfg.latent_dim = 8;
    vcfg.encoder_dims = {32, 16};
    vcfg.kl_anneal_epochs = 5;
    vcfg.pretrain_epochs = 60;
    vcfg.batch_size = 64;
    vcfg.device = torch::kCPU;

    VAEPretrainer pretrainer(vocab, vcfg);
    auto result = pretrainer.pretrain(ds.species_vector());
    REQUIRE(result.recon_loss_history.size() >= 2);
    const float first = result.recon_loss_history.front();
    const float last = result.recon_loss_history.back();
    REQUIRE(std::isfinite(first));
    REQUIRE(std::isfinite(last));
    // The pretext task is learned: reconstruction improves by a clear margin.
    REQUIRE(last < first * 0.9f);
}
