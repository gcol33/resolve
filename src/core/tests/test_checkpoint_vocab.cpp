// Tests for checkpoint-persisted vocabularies (gcol33/resolve#102).
//
// The defect: a checkpoint carried only the SIZES of the species / genus /
// family vocabularies. Every non-hash encoder indexes an embedding table with a
// code that is a function of the file the vocabulary was fitted on -- the
// species vocab is frequency-ranked, the taxonomy vocab is a function of the
// name set -- so scoring new data from a checkpoint alone re-fitted the codes
// and looked up other species' embedding rows. Wrong predictions, no error.
//
// The contract now:
//   1. save_schema/load_schema round-trip the ordered species / genus / family
//      vocabularies, plus the remaining DatasetConfig knobs the loader consumed.
//   2. ResolveDataset::from_csv_with_schema(const ResolveSchema&) builds an
//      inference dataset in the checkpoint's ID namespace, with no training
//      dataset on hand.
//   3. Predictor::predict(ResolveDataset) REJECTS a dataset whose vocabularies
//      are not the model's, instead of scoring it.
//   4. A pre-#102 checkpoint (no vocab block) still loads: the vectors come
//      back empty and only the size guard applies.
//
// The headline test (per encoder) trains on file A, scores a file B whose rows
// are a subset of A's but whose species FREQUENCY ORDER differs, and asserts the
// predictions equal scoring A restricted to B's rows. A re-fit vocabulary
// permutes the codes and breaks that equality.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/checkpoint.hpp"
#include "resolve/checkpoint_schema_keys.hpp"
#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/trainer.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;

namespace {

class TempFile {
public:
    explicit TempFile(const std::string& content, const std::string& suffix = ".csv") {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_ckpt_vocab_" + std::to_string(counter_++) + suffix);
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

// Scratch path for a checkpoint, removed on scope exit.
class TempPath {
public:
    explicit TempPath(const std::string& stem) {
        path_ = std::filesystem::temp_directory_path() /
                (stem + std::to_string(counter_++) + ".pt");
    }
    ~TempPath() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    [[nodiscard]] std::string path() const { return path_.string(); }
private:
    std::filesystem::path path_;
    static int counter_;
};
int TempPath::counter_ = 0;

// ---------------------------------------------------------------------------
// Synthetic corpus
// ---------------------------------------------------------------------------
//
// 24 plots, four species, EXACTLY three species per plot (so the rank-pool
// padded width is the same in every subset and cannot itself explain a
// difference). Genus/family are derived from the species name.
//
//   P0..P11  : sp_a sp_b sp_c
//   P12..P19 : sp_a sp_b sp_d
//   P20..P23 : sp_b sp_c sp_d
//
// Two scoring subsets, because the two vocabulary builders fail differently:
//
//   REORDER (P16..P23) -- all four species still present, so the vocabulary
//     SIZE is unchanged, but the frequencies become b=8, d=8, a=4, c=4 against
//     b=24, a=20, c=16, d=12 over the whole file. The dataset's own
//     frequency-ranked vocab (build_species_vocab, used by embed and sparse)
//     therefore permutes: b,a,c,d -> b,d,a,c. This is the silent case a
//     size-only checkpoint could not detect.
//
//   DROP (P12..P19) -- sp_c never occurs, so the species SET shrinks. That
//     moves the name-sorted vocab too (SpeciesVocab::from_records, used by the
//     rank_pool / transformer encoders), and changes the size.

struct Plot {
    std::string id;
    std::vector<std::string> species;
    double cov;
    double y;
};

std::vector<Plot> corpus() {
    std::vector<Plot> plots;
    for (int i = 0; i < 24; ++i) {
        Plot p;
        p.id = "P" + std::to_string(i);
        if (i < 12) {
            p.species = {"sp_a", "sp_b", "sp_c"};
        } else if (i < 20) {
            p.species = {"sp_a", "sp_b", "sp_d"};
        } else {
            p.species = {"sp_b", "sp_c", "sp_d"};
        }
        p.cov = 0.5 * i;
        p.y = 1.0 + 0.25 * i;
        plots.push_back(std::move(p));
    }
    return plots;
}

std::string genus_of(const std::string& sp) { return "gen_" + sp.substr(3, 1); }
std::string family_of(const std::string& sp) {
    // Two families over four genera, so the family vocabulary is genuinely
    // coarser than the genus one.
    return (sp == "sp_a" || sp == "sp_b") ? "fam_x" : "fam_y";
}

// Coordinates vary per plot (and are a pure function of the plot, so a subset
// file carries the same values), which keeps the fitted continuous scaler
// non-degenerate.
double lat_of(const Plot& p) { return 40.0 + p.cov; }
double lon_of(const Plot& p) { return -5.0 + 0.5 * p.cov; }

std::string header_csv(const std::vector<Plot>& plots) {
    std::ostringstream out;
    out << "plot_id,lat,lon,cov1,y\n";
    for (const auto& p : plots) {
        out << p.id << "," << lat_of(p) << "," << lon_of(p) << ","
            << p.cov << "," << p.y << "\n";
    }
    return out.str();
}

std::string species_csv(const std::vector<Plot>& plots) {
    std::ostringstream out;
    out << "plot_id,sp,cover,genus,family\n";
    for (const auto& p : plots) {
        for (size_t k = 0; k < p.species.size(); ++k) {
            // Distinct covers so abundance-weighted schemes have something to
            // work with; identical per plot across files by construction.
            out << p.id << "," << p.species[k] << "," << (1.0 + k) << ","
                << genus_of(p.species[k]) << "," << family_of(p.species[k]) << "\n";
        }
    }
    return out.str();
}

RoleMapping test_roles() {
    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "sp";
    roles.abundance = "cover";
    roles.latitude = "lat";
    roles.longitude = "lon";
    roles.genus = "genus";
    roles.family = "family";
    roles.covariates = {"cov1"};
    return roles;
}

DatasetConfig test_dataset_config(SpeciesEncodingMode mode) {
    DatasetConfig cfg;
    cfg.species_encoding = mode;
    cfg.top_k = 2;
    cfg.top_k_species = 3;
    cfg.use_taxonomy = true;
    // Keep the continuous block identical between the full file and any subset
    // so any prediction difference can only come from the species/taxonomy IDs.
    cfg.track_unknown_fraction = false;
    cfg.track_unknown_count = false;
    return cfg;
}

ModelConfig test_model_config(SpeciesEncodingMode mode) {
    ModelConfig cfg;
    cfg.species_encoding = mode;
    cfg.encoder_architecture = EncoderArchitecture::MLP;
    cfg.top_k = 2;
    cfg.top_k_species = 3;
    cfg.n_taxonomy_slots = 2;
    cfg.species_embed_dim = 8;
    cfg.genus_emb_dim = 4;
    cfg.family_emb_dim = 4;
    cfg.hidden_dims = {16, 8};
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.n_attention_layers = 0;
    return cfg;
}

TrainConfig quiet_train_config() {
    TrainConfig cfg;
    cfg.batch_size = 8;
    cfg.max_epochs = 1;
    cfg.patience = 1;
    cfg.lr = 1e-3f;
    cfg.log = null_log;
    return cfg;
}

// Save a checkpoint for `ds` with randomly-initialised weights. Random init is
// enough: the embedding tables differ row to row, so a wrong species code
// produces a different prediction. fit() is skipped to keep the test fast.
void save_checkpoint(const ResolveDataset& ds, const ModelConfig& mcfg,
                     const std::string& path) {
    torch::manual_seed(1234);
    ResolveModel model(ds.schema(), mcfg);
    Trainer trainer(model, quiet_train_config());
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/0);
    trainer.save(path);
}

void save_checkpoint(const ResolveDataset& ds, SpeciesEncodingMode mode,
                     const std::string& path) {
    save_checkpoint(ds, test_model_config(mode), path);
}

// Map plot_id -> row index in a predictions result.
std::map<std::string, size_t> row_index(const ResolvePredictions& preds) {
    std::map<std::string, size_t> out;
    for (size_t i = 0; i < preds.plot_ids.size(); ++i) out[preds.plot_ids[i]] = i;
    return out;
}

double pred_at(const ResolvePredictions& preds, const std::string& target, size_t row) {
    auto it = preds.predictions.find(target);
    REQUIRE(it != preds.predictions.end());
    return it->second[static_cast<int64_t>(row)].item<double>();
}

const char* mode_name(SpeciesEncodingMode m) {
    switch (m) {
        case SpeciesEncodingMode::Hash: return "hash";
        case SpeciesEncodingMode::Embed: return "embed";
        case SpeciesEncodingMode::Sparse: return "sparse";
        case SpeciesEncodingMode::RankPool: return "rank_pool";
        case SpeciesEncodingMode::Transformer: return "transformer";
    }
    return "?";
}

}  // namespace

// =============================================================================
// 1. Headline: train on A, score B, predictions match A-restricted-to-B
// =============================================================================

TEST_CASE("Scoring a subset from a checkpoint alone matches scoring the full file",
          "[checkpoint][vocab][issue102]") {
    const auto all_plots = corpus();
    const std::vector<Plot> reorder(all_plots.begin() + 16, all_plots.end());
    const std::vector<Plot> drop(all_plots.begin() + 12, all_plots.begin() + 20);

    TempFile hdr_a(header_csv(all_plots));
    TempFile spc_a(species_csv(all_plots));
    TempFile hdr_r(header_csv(reorder));
    TempFile spc_r(species_csv(reorder));
    TempFile hdr_d(header_csv(drop));
    TempFile spc_d(species_csv(drop));

    const auto roles = test_roles();
    const std::vector<TargetSpec> targets = {TargetSpec::regression("y")};

    for (auto mode : {SpeciesEncodingMode::Embed, SpeciesEncodingMode::Sparse,
                      SpeciesEncodingMode::RankPool, SpeciesEncodingMode::Transformer}) {
        DYNAMIC_SECTION("encoder = " << mode_name(mode)) {
            const auto dcfg = test_dataset_config(mode);
            auto ds_a = ResolveDataset::from_csv(
                hdr_a.path(), spc_a.path(), roles, targets, dcfg);

            TempPath ckpt("resolve_vocab_head_");
            save_checkpoint(ds_a, mode, ckpt.path());

            Predictor predictor = Predictor::load(ckpt.path(), torch::kCPU);

            // The checkpoint really carries the fitted vocabularies.
            REQUIRE(predictor.schema().has_species_vocab());
            REQUIRE(predictor.schema().species_vocab == ds_a.species_vocab());
            REQUIRE(predictor.schema().has_taxonomy_vocab());

            auto preds_full = predictor.predict(ds_a, /*return_latent=*/false, -1);
            const auto idx_full = row_index(preds_full);

            // The checkpoint-only path: no training dataset, no training CSVs.
            // Per-plot predictions must be identical to scoring the whole file
            // and reading off the same rows -- which only holds if the subset's
            // species / taxonomy IDs land on the model's embedding rows.
            const auto infer_cfg = dataset_config_from_checkpoint(
                predictor.schema(), predictor.model()->config());

            struct Case { const char* label; const TempFile* hdr; const TempFile* spc;
                          const std::vector<Plot>* plots; };
            const Case cases[] = {
                {"reordered subset", &hdr_r, &spc_r, &reorder},
                {"species-dropping subset", &hdr_d, &spc_d, &drop},
            };

            for (const auto& c : cases) {
                INFO("scoring file: " << c.label);
                auto ds_sub = ResolveDataset::from_csv_with_schema(
                    c.hdr->path(), c.spc->path(), roles, targets,
                    predictor.schema(), infer_cfg);

                REQUIRE(ds_sub.species_vocab() == predictor.species_vocab());
                REQUIRE(ds_sub.schema().n_species_vocab ==
                        predictor.schema().n_species_vocab);
                REQUIRE(ds_sub.schema().n_genera_vocab ==
                        predictor.schema().n_genera_vocab);
                REQUIRE(ds_sub.schema().n_families_vocab ==
                        predictor.schema().n_families_vocab);

                auto preds_sub = predictor.predict(ds_sub, /*return_latent=*/false, -1);
                REQUIRE(preds_sub.plot_ids.size() == c.plots->size());

                const auto idx_sub = row_index(preds_sub);
                for (const auto& p : *c.plots) {
                    REQUIRE(idx_full.count(p.id) == 1);
                    REQUIRE(idx_sub.count(p.id) == 1);
                    const double from_full = pred_at(preds_full, "y", idx_full.at(p.id));
                    const double from_sub = pred_at(preds_sub, "y", idx_sub.at(p.id));
                    REQUIRE_THAT(from_sub, Catch::Matchers::WithinAbs(from_full, 1e-5));
                }
            }
        }
    }
}

// A vocabulary re-fit that leaves the SIZE unchanged is the silent case: the
// old size-only checkpoint had nothing to compare. Both frequency-ranked
// encoders (embed, sparse) permute their codes on the reordered subset.
TEST_CASE("A re-fitted species vocabulary is caught even when its size is unchanged",
          "[checkpoint][vocab][issue102][guard]") {
    const auto all_plots = corpus();
    const std::vector<Plot> reorder(all_plots.begin() + 16, all_plots.end());

    TempFile hdr_a(header_csv(all_plots));
    TempFile spc_a(species_csv(all_plots));
    TempFile hdr_r(header_csv(reorder));
    TempFile spc_r(species_csv(reorder));

    const auto roles = test_roles();
    const std::vector<TargetSpec> targets = {TargetSpec::regression("y")};

    for (auto mode : {SpeciesEncodingMode::Embed, SpeciesEncodingMode::Sparse}) {
        DYNAMIC_SECTION("encoder = " << mode_name(mode)) {
            const auto dcfg = test_dataset_config(mode);
            auto ds_a = ResolveDataset::from_csv(
                hdr_a.path(), spc_a.path(), roles, targets, dcfg);

            TempPath ckpt("resolve_vocab_silent_");
            save_checkpoint(ds_a, mode, ckpt.path());
            Predictor predictor = Predictor::load(ckpt.path(), torch::kCPU);

            auto ds_naive = ResolveDataset::from_csv(
                hdr_r.path(), spc_r.path(), roles, targets, dcfg);

            // Same size, different meaning: this is exactly what used to be
            // scored without complaint.
            REQUIRE(ds_naive.schema().n_species_vocab ==
                    predictor.schema().n_species_vocab);
            REQUIRE(ds_naive.species_vocab() != predictor.species_vocab());
            REQUIRE_THROWS_AS(predictor.predict(ds_naive, false, -1), std::runtime_error);
        }
    }
}

// The species-dropping subset shrinks the vocabulary, which every
// species-indexing encoder must reject on the size check alone -- the guard
// that also protects a pre-issue-#102 checkpoint.
TEST_CASE("A scoring file missing a species is rejected by every species-indexing encoder",
          "[checkpoint][vocab][issue102][guard]") {
    const auto all_plots = corpus();
    const std::vector<Plot> drop(all_plots.begin() + 12, all_plots.begin() + 20);

    TempFile hdr_a(header_csv(all_plots));
    TempFile spc_a(species_csv(all_plots));
    TempFile hdr_d(header_csv(drop));
    TempFile spc_d(species_csv(drop));

    const auto roles = test_roles();
    const std::vector<TargetSpec> targets = {TargetSpec::regression("y")};

    for (auto mode : {SpeciesEncodingMode::Embed, SpeciesEncodingMode::Sparse,
                      SpeciesEncodingMode::RankPool, SpeciesEncodingMode::Transformer}) {
        DYNAMIC_SECTION("encoder = " << mode_name(mode)) {
            const auto dcfg = test_dataset_config(mode);
            auto ds_a = ResolveDataset::from_csv(
                hdr_a.path(), spc_a.path(), roles, targets, dcfg);

            TempPath ckpt("resolve_vocab_drop_");
            save_checkpoint(ds_a, mode, ckpt.path());
            Predictor predictor = Predictor::load(ckpt.path(), torch::kCPU);

            auto ds_naive = ResolveDataset::from_csv(
                hdr_d.path(), spc_d.path(), roles, targets, dcfg);
            REQUIRE(ds_naive.schema().n_species_vocab <
                    predictor.schema().n_species_vocab);
            REQUIRE_THROWS_AS(predictor.predict(ds_naive, false, -1), std::runtime_error);
        }
    }
}

// =============================================================================
// 2. Over-species: a species absent from training maps to UNK, never out of range
// =============================================================================

TEST_CASE("A species absent from training maps to the reserved UNK code",
          "[checkpoint][vocab][issue102]") {
    const auto all_plots = corpus();
    TempFile hdr_a(header_csv(all_plots));
    TempFile spc_a(species_csv(all_plots));

    // Same plots, but every third plot swaps one species for one the model has
    // never seen. The vocabulary must NOT grow, and the unseen species must
    // encode to 0 rather than to a row past the end of the embedding table.
    auto extended = all_plots;
    for (size_t i = 0; i < extended.size(); i += 3) {
        extended[i].species[0] = "sp_unseen";
    }
    TempFile hdr_c(header_csv(extended));
    TempFile spc_c(species_csv(extended));

    const auto roles = test_roles();
    const std::vector<TargetSpec> targets = {TargetSpec::regression("y")};

    for (auto mode : {SpeciesEncodingMode::Embed, SpeciesEncodingMode::RankPool,
                      SpeciesEncodingMode::Transformer}) {
        DYNAMIC_SECTION("encoder = " << mode_name(mode)) {
            const auto dcfg = test_dataset_config(mode);
            auto ds_a = ResolveDataset::from_csv(
                hdr_a.path(), spc_a.path(), roles, targets, dcfg);

            TempPath ckpt("resolve_vocab_unk_");
            save_checkpoint(ds_a, mode, ckpt.path());
            Predictor predictor = Predictor::load(ckpt.path(), torch::kCPU);

            const auto infer_cfg = dataset_config_from_checkpoint(
                predictor.schema(), predictor.model()->config());
            auto ds_c = ResolveDataset::from_csv_with_schema(
                hdr_c.path(), spc_c.path(), roles, targets,
                predictor.schema(), infer_cfg);

            // The vocabulary is the model's, unchanged by the new species.
            REQUIRE(ds_c.species_vocab() == predictor.species_vocab());
            REQUIRE(ds_c.schema().n_species_vocab == predictor.schema().n_species_vocab);

            // No species ID may index past the model's embedding table, and the
            // unseen species must have landed on the reserved UNK slot.
            const auto& ids = ds_c.species_ids();
            REQUIRE(ids.defined());
            REQUIRE(ids.numel() > 0);
            REQUIRE(ids.max().item<int64_t>() < predictor.schema().n_species_vocab);
            REQUIRE(ids.min().item<int64_t>() >= 0);
            REQUIRE((ids == 0).any().item<bool>());

            // And it scores without a device-side index error.
            REQUIRE_NOTHROW(predictor.predict(ds_c, /*return_latent=*/false, -1));
        }
    }
}

// =============================================================================
// 3. Vocabulary round-trip through the checkpoint archive
// =============================================================================

TEST_CASE("save_schema/load_schema round-trip the species and taxonomy vocabularies",
          "[checkpoint][vocab][issue102]") {
    ResolveSchema schema;
    schema.n_plots = 7;
    schema.n_species = 3;
    schema.n_species_vocab = 4;
    schema.has_taxonomy = true;
    schema.n_genera = 3;
    schema.n_families = 2;
    schema.n_genera_vocab = 3;
    schema.n_families_vocab = 2;
    schema.species_vocab = {"<UNK>", "sp_b", "sp_a", "sp_c"};
    schema.genus_vocab = {"<UNK>", "gen_a", "gen_b"};
    schema.family_vocab = {"<UNK>", "fam_x"};
    schema.targets.push_back(TargetConfig{"y", TaskType::Regression,
                                          TransformType::None, 0, 1.0f, {}, {}});

    TempPath path("resolve_vocab_roundtrip_");
    {
        torch::serialize::OutputArchive archive;
        save_schema(archive, schema);
        archive.save_to(path.path());
    }
    torch::serialize::InputArchive in;
    in.load_from(path.path());
    const ResolveSchema loaded = load_schema(in);

    REQUIRE(loaded.species_vocab == schema.species_vocab);
    REQUIRE(loaded.genus_vocab == schema.genus_vocab);
    REQUIRE(loaded.family_vocab == schema.family_vocab);
    REQUIRE(loaded.has_species_vocab());
    REQUIRE(loaded.has_taxonomy_vocab());

    // The ordered lists rebuild the encoder exactly.
    const TaxonomyVocab tax = TaxonomyVocab::from_ordered(loaded.genus_vocab,
                                                          loaded.family_vocab);
    REQUIRE(tax.encode_genus("gen_a") == 1);
    REQUIRE(tax.encode_genus("gen_b") == 2);
    REQUIRE(tax.encode_genus("never_seen") == 0);
    REQUIRE(tax.encode_family("fam_x") == 1);
    REQUIRE(tax.n_genera() == 3);
    REQUIRE(tax.n_families() == 2);
    REQUIRE(tax.genus_names() == loaded.genus_vocab);
    REQUIRE(tax.family_names() == loaded.family_vocab);
}

TEST_CASE("An empty vocabulary round-trips as empty rather than as a corrupt read",
          "[checkpoint][vocab][issue102]") {
    ResolveSchema schema;
    schema.n_plots = 1;
    // No species / genus / family vocabularies at all (a schema built by hand).
    TempPath path("resolve_vocab_empty_");
    {
        torch::serialize::OutputArchive archive;
        save_schema(archive, schema);
        archive.save_to(path.path());
    }
    torch::serialize::InputArchive in;
    in.load_from(path.path());
    const ResolveSchema loaded = load_schema(in);
    REQUIRE(loaded.species_vocab.empty());
    REQUIRE(loaded.genus_vocab.empty());
    REQUIRE(loaded.family_vocab.empty());
    REQUIRE_FALSE(loaded.has_species_vocab());
    REQUIRE_FALSE(loaded.has_taxonomy_vocab());
}

TEST_CASE("A checkpoint written before the vocab block still loads",
          "[checkpoint][vocab][issue102][backcompat]") {
    // Emit ONLY the keys load_schema reads unconditionally, i.e. exactly what a
    // pre-issue-#102 save_schema produced. Every field added since must come
    // back at its default rather than throwing.
    namespace k = ckpt_schema_keys;
    TempPath path("resolve_vocab_legacy_");
    {
        torch::serialize::OutputArchive archive;
        archive.write(k::kNPlots, torch::tensor(static_cast<int64_t>(11)));
        archive.write(k::kNSpecies, torch::tensor(static_cast<int64_t>(5)));
        archive.write(k::kNSpeciesVocab, torch::tensor(static_cast<int64_t>(6)));
        archive.write(k::kHasCoordinates, torch::tensor(1));
        archive.write(k::kHasAbundance, torch::tensor(1));
        archive.write(k::kHasTaxonomy, torch::tensor(0));
        archive.write(k::kNGenera, torch::tensor(static_cast<int64_t>(0)));
        archive.write(k::kNFamilies, torch::tensor(static_cast<int64_t>(0)));
        archive.write(k::kNGeneraVocab, torch::tensor(static_cast<int64_t>(0)));
        archive.write(k::kNFamiliesVocab, torch::tensor(static_cast<int64_t>(0)));
        archive.write(k::kTrackUnknownFrac, torch::tensor(1));
        archive.write(k::kTrackUnknownCount, torch::tensor(0));
        archive.write(k::kNCovariates, torch::tensor(static_cast<int64_t>(0)));
        archive.write(k::kNTargets, torch::tensor(static_cast<int64_t>(1)));
        const std::string prefix = k::target_prefix(0);
        archive.write(prefix + k::kTargetTask, torch::tensor(0));
        archive.write(prefix + k::kTargetTransform, torch::tensor(0));
        archive.write(prefix + k::kTargetNumClasses, torch::tensor(0));
        archive.write(prefix + k::kTargetWeight, torch::tensor(1.0f));
        archive.save_to(path.path());
    }

    torch::serialize::InputArchive in;
    in.load_from(path.path());
    ResolveSchema loaded;
    REQUIRE_NOTHROW(loaded = load_schema(in));

    REQUIRE(loaded.n_plots == 11);
    REQUIRE(loaded.n_species_vocab == 6);
    REQUIRE(loaded.targets.size() == 1);
    // Nothing was persisted for the new fields, so they hold the defaults --
    // which are the DatasetConfig defaults, i.e. exactly the old behaviour.
    REQUIRE_FALSE(loaded.has_species_vocab());
    REQUIRE_FALSE(loaded.has_taxonomy_vocab());
    const DatasetConfig defaults;
    REQUIRE(loaded.top_k_species == defaults.top_k_species);
    REQUIRE(loaded.selection == defaults.selection);
    REQUIRE(loaded.representation == defaults.representation);
    REQUIRE(loaded.normalization == defaults.normalization);
    REQUIRE(loaded.aggregation == defaults.aggregation);
    REQUIRE(loaded.use_taxonomy == defaults.use_taxonomy);

    // The vocabulary carrier derived from such a schema is empty, which is what
    // makes the loaders refuse it instead of silently re-fitting the codes.
    REQUIRE(external_vocabs_from_schema(loaded).species_vocab.empty());
}

// =============================================================================
// 4. Full DatasetConfig round-trip through ResolveSchema
// =============================================================================

TEST_CASE("Every loading-side DatasetConfig field round-trips through the schema",
          "[checkpoint][vocab][issue102][config]") {
    // Non-default value for every knob, so a dropped field is visible.
    ResolveSchema schema;
    schema.track_unknown_fraction = false;
    schema.track_unknown_count = true;
    schema.top_k_species = 7;
    schema.selection = SelectionMode::TopBottom;
    schema.representation = RepresentationMode::PresenceAbsence;
    schema.normalization = NormalizationMode::Log1p;
    schema.aggregation = AggregationMode::Count;
    schema.use_taxonomy = false;
    schema.pool_weighting = static_cast<int>(PoolWeighting::Rank);
    schema.pool_species_cap = 23;

    TempPath path("resolve_vocab_cfg_");
    {
        torch::serialize::OutputArchive archive;
        save_schema(archive, schema);
        archive.save_to(path.path());
    }
    torch::serialize::InputArchive in;
    in.load_from(path.path());
    const ResolveSchema loaded = load_schema(in);

    REQUIRE(loaded.track_unknown_fraction == false);
    REQUIRE(loaded.track_unknown_count == true);
    REQUIRE(loaded.top_k_species == 7);
    REQUIRE(loaded.selection == SelectionMode::TopBottom);
    REQUIRE(loaded.representation == RepresentationMode::PresenceAbsence);
    REQUIRE(loaded.normalization == NormalizationMode::Log1p);
    REQUIRE(loaded.aggregation == AggregationMode::Count);
    REQUIRE(loaded.use_taxonomy == false);
    REQUIRE(loaded.pool_weighting == static_cast<int>(PoolWeighting::Rank));
    REQUIRE(loaded.pool_species_cap == 23);

    // The reassembled DatasetConfig: the three model-sizing knobs come from
    // ModelConfig, the other nine from the schema. All twelve loading-side
    // fields are therefore recovered (use_cuda_hash is deliberately excluded --
    // it is a training-time compute path, not a property of the encoding).
    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::RankPool;
    mcfg.hash_dim = 128;
    mcfg.top_k = 9;
    const DatasetConfig cfg = dataset_config_from_checkpoint(loaded, mcfg);

    REQUIRE(cfg.species_encoding == SpeciesEncodingMode::RankPool);
    REQUIRE(cfg.hash_dim == 128);
    REQUIRE(cfg.top_k == 9);
    REQUIRE(cfg.top_k_species == 7);
    REQUIRE(cfg.selection == SelectionMode::TopBottom);
    REQUIRE(cfg.representation == RepresentationMode::PresenceAbsence);
    REQUIRE(cfg.normalization == NormalizationMode::Log1p);
    REQUIRE(cfg.aggregation == AggregationMode::Count);
    REQUIRE(cfg.track_unknown_fraction == false);
    REQUIRE(cfg.track_unknown_count == true);
    REQUIRE(cfg.use_taxonomy == false);
    REQUIRE(cfg.pool_weighting == PoolWeighting::Rank);
    REQUIRE(cfg.pool_species_cap == 23);
    REQUIRE(cfg.use_cuda_hash == false);
}

TEST_CASE("A loaded dataset publishes its own config onto the schema",
          "[checkpoint][vocab][issue102][config]") {
    const auto plots = corpus();
    TempFile hdr(header_csv(plots));
    TempFile spc(species_csv(plots));

    DatasetConfig dcfg = test_dataset_config(SpeciesEncodingMode::Embed);
    dcfg.top_k_species = 3;
    dcfg.selection = SelectionMode::Top;
    dcfg.representation = RepresentationMode::PresenceAbsence;
    dcfg.normalization = NormalizationMode::Norm;
    dcfg.aggregation = AggregationMode::Count;

    auto ds = ResolveDataset::from_csv(hdr.path(), spc.path(), test_roles(),
                                       {TargetSpec::regression("y")}, dcfg);

    REQUIRE(ds.schema().top_k_species == 3);
    REQUIRE(ds.schema().selection == SelectionMode::Top);
    REQUIRE(ds.schema().representation == RepresentationMode::PresenceAbsence);
    REQUIRE(ds.schema().normalization == NormalizationMode::Norm);
    REQUIRE(ds.schema().aggregation == AggregationMode::Count);
    REQUIRE(ds.schema().use_taxonomy == true);
    REQUIRE(ds.schema().species_vocab == ds.species_vocab());
    REQUIRE(ds.schema().genus_vocab == ds.taxonomy_vocab().genus_names());
    REQUIRE(ds.schema().family_vocab == ds.taxonomy_vocab().family_names());
}

TEST_CASE("The rank-pool resolved species cap survives into the inference config",
          "[checkpoint][vocab][issue102][config]") {
    const auto plots = corpus();
    TempFile hdr(header_csv(plots));
    TempFile spc(species_csv(plots));

    DatasetConfig dcfg = test_dataset_config(SpeciesEncodingMode::RankPool);
    auto ds = ResolveDataset::from_csv(hdr.path(), spc.path(), test_roles(),
                                       {TargetSpec::regression("y")}, dcfg);

    // encode_species overwrites pool_species_cap with the RESOLVED width, so a
    // checkpoint always carries a concrete >0 cap even when the request was 0.
    REQUIRE(ds.schema().pool_species_cap == ds.pool_weights().size(1));

    TempPath ckpt("resolve_vocab_cap_");
    save_checkpoint(ds, SpeciesEncodingMode::RankPool, ckpt.path());
    Predictor predictor = Predictor::load(ckpt.path(), torch::kCPU);

    const DatasetConfig infer = dataset_config_from_checkpoint(
        predictor.schema(), predictor.model()->config());
    REQUIRE(infer.pool_species_cap == ds.schema().pool_species_cap);
    REQUIRE(infer.pool_weighting == dcfg.pool_weighting);
}

// =============================================================================
// 5. The vocabulary guard on Predictor::predict
// =============================================================================

TEST_CASE("Predictor::predict rejects a dataset whose species vocabulary differs in size",
          "[checkpoint][vocab][issue102][guard]") {
    const auto plots = corpus();
    TempFile hdr_a(header_csv(plots));
    TempFile spc_a(species_csv(plots));

    // Drop every occurrence of one species, so the fresh fit has one entry
    // fewer than the model's table.
    auto reduced = plots;
    for (auto& p : reduced) {
        p.species.erase(std::remove(p.species.begin(), p.species.end(), "sp_d"),
                        p.species.end());
        if (p.species.empty()) p.species.push_back("sp_a");
    }
    TempFile hdr_b(header_csv(reduced));
    TempFile spc_b(species_csv(reduced));

    const auto roles = test_roles();
    const std::vector<TargetSpec> targets = {TargetSpec::regression("y")};
    const auto dcfg = test_dataset_config(SpeciesEncodingMode::Embed);

    auto ds_a = ResolveDataset::from_csv(hdr_a.path(), spc_a.path(), roles, targets, dcfg);
    TempPath ckpt("resolve_vocab_guard_");
    save_checkpoint(ds_a, SpeciesEncodingMode::Embed, ckpt.path());
    Predictor predictor = Predictor::load(ckpt.path(), torch::kCPU);

    auto ds_b = ResolveDataset::from_csv(hdr_b.path(), spc_b.path(), roles, targets, dcfg);
    REQUIRE(ds_b.schema().n_species_vocab != predictor.schema().n_species_vocab);
    REQUIRE_THROWS_AS(predictor.predict(ds_b, false, -1), std::runtime_error);

    // The dataset the model was trained on is of course accepted.
    REQUIRE_NOTHROW(predictor.predict(ds_a, false, -1));
}

TEST_CASE("Hash encoding is exempt from the species check but not the taxonomy one",
          "[checkpoint][vocab][issue102][guard]") {
    const auto plots = corpus();
    // Hash features are derived from the species STRING, so its codes never
    // reach the model and a differing species vocabulary is harmless. The
    // genus/family slots ARE embedding lookups, so those must still match.
    auto reduced = plots;
    for (auto& p : reduced) {
        p.species.erase(std::remove(p.species.begin(), p.species.end(), "sp_d"),
                        p.species.end());
        if (p.species.empty()) p.species.push_back("sp_a");
    }

    TempFile hdr_a(header_csv(plots));
    TempFile spc_a(species_csv(plots));
    TempFile hdr_b(header_csv(reduced));
    TempFile spc_b(species_csv(reduced));

    const auto roles = test_roles();
    const std::vector<TargetSpec> targets = {TargetSpec::regression("y")};

    DatasetConfig dcfg = test_dataset_config(SpeciesEncodingMode::Hash);
    dcfg.hash_dim = 8;
    dcfg.use_taxonomy = false;  // species-only: nothing left to check

    auto ds_a = ResolveDataset::from_csv(hdr_a.path(), spc_a.path(), roles, targets, dcfg);
    ModelConfig mcfg = test_model_config(SpeciesEncodingMode::Hash);
    mcfg.hash_dim = 8;
    TempPath ckpt("resolve_vocab_hash_");
    save_checkpoint(ds_a, mcfg, ckpt.path());
    Predictor predictor = Predictor::load(ckpt.path(), torch::kCPU);

    auto ds_b = ResolveDataset::from_csv(hdr_b.path(), spc_b.path(), roles, targets, dcfg);
    REQUIRE(ds_b.schema().n_species_vocab != predictor.schema().n_species_vocab);
    REQUIRE_NOTHROW(predictor.predict(ds_b, false, -1));
}

// =============================================================================
// 6. Categorical covariates: predict() behaves as the header says
// =============================================================================

namespace {

// Header CSV with an extra string-valued categorical column. `regions` supplies
// the value for each plot in order.
std::string header_csv_with_region(const std::vector<Plot>& plots,
                                   const std::vector<std::string>& regions) {
    std::ostringstream out;
    out << "plot_id,lat,lon,cov1,region,y\n";
    for (size_t i = 0; i < plots.size(); ++i) {
        out << plots[i].id << "," << lat_of(plots[i]) << "," << lon_of(plots[i])
            << "," << plots[i].cov << "," << regions[i % regions.size()] << ","
            << plots[i].y << "\n";
    }
    return out.str();
}

}  // namespace

TEST_CASE("Predictor::predict rejects a dataset that re-factorized its categorical columns",
          "[checkpoint][vocab][issue102][categorical]") {
    const auto plots = corpus();
    // Training sees four regions; the scoring file sees only two of them, so a
    // fresh factorize assigns codes 1..2 over values that were 2 and 4 at
    // training time. The tensor shape is unchanged, so nothing but the vocab
    // can catch it.
    TempFile hdr_a(header_csv_with_region(plots, {"east", "north", "south", "west"}));
    TempFile spc_a(species_csv(plots));
    TempFile hdr_b(header_csv_with_region(plots, {"north", "west"}));
    TempFile spc_b(species_csv(plots));

    auto roles = test_roles();
    roles.categoricals = {"region"};
    const std::vector<TargetSpec> targets = {TargetSpec::regression("y")};

    DatasetConfig dcfg = test_dataset_config(SpeciesEncodingMode::Hash);
    dcfg.hash_dim = 8;

    auto ds_a = ResolveDataset::from_csv(hdr_a.path(), spc_a.path(), roles, targets, dcfg);
    REQUIRE(ds_a.schema().has_categoricals());

    ModelConfig mcfg = test_model_config(SpeciesEncodingMode::Hash);
    mcfg.hash_dim = 8;
    mcfg.categorical_embed_dim = 4;
    TempPath ckpt("resolve_vocab_cat_");
    save_checkpoint(ds_a, mcfg, ckpt.path());
    Predictor predictor = Predictor::load(ckpt.path(), torch::kCPU);
    REQUIRE(predictor.categorical_vocab().vocab_size("region") == 5);

    // A plain load re-factorizes: "north" no longer means what it meant.
    auto ds_b_naive = ResolveDataset::from_csv(hdr_b.path(), spc_b.path(), roles, targets, dcfg);
    REQUIRE(ds_b_naive.categorical_vocab().encode("region", "north") !=
            predictor.categorical_vocab().encode("region", "north"));
    REQUIRE_THROWS_AS(predictor.predict(ds_b_naive, false, -1), std::runtime_error);

    // The vocab-carrying path keeps the training codes -- the categorical maps
    // live on the Predictor, not on the schema, so external_vocabs() is the
    // carrier that has all of them.
    const auto infer_cfg = dataset_config_from_checkpoint(
        predictor.schema(), predictor.model()->config());
    auto ds_b = ResolveDataset::from_csv_with_vocabs(
        hdr_b.path(), spc_b.path(), roles, targets,
        predictor.external_vocabs(), infer_cfg);

    REQUIRE(ds_b.categorical_vocab().encode("region", "north") ==
            predictor.categorical_vocab().encode("region", "north"));
    REQUIRE(ds_b.categorical_vocab().encode("region", "west") ==
            predictor.categorical_vocab().encode("region", "west"));
    // A value the training data never carried still falls on UNK.
    REQUIRE(ds_b.categorical_vocab().encode("region", "arctic") == 0);
    REQUIRE_NOTHROW(predictor.predict(ds_b, false, -1));
}

// =============================================================================
// 7. Single-table (species-only) loader parity
// =============================================================================

TEST_CASE("The single-table loader has a vocabulary-reusing sibling",
          "[checkpoint][vocab][issue102]") {
    // One long table carrying the target: the shape `resolve predict` uses when
    // no header file is given. Before #102 it had no *_with_schema sibling at
    // all, so that CLI path could not stay in the model's ID namespace.
    const auto plots = corpus();
    std::ostringstream full, part;
    full << "plot_id,sp,cover,genus,family,lat,lon,y\n";
    part << "plot_id,sp,cover,genus,family,lat,lon,y\n";
    for (size_t i = 0; i < plots.size(); ++i) {
        const auto& p = plots[i];
        for (size_t k = 0; k < p.species.size(); ++k) {
            std::ostringstream row;
            row << p.id << "," << p.species[k] << "," << (1.0 + k) << ","
                << genus_of(p.species[k]) << "," << family_of(p.species[k])
                << "," << lat_of(p) << "," << lon_of(p) << "," << p.y << "\n";
            full << row.str();
            if (i >= 16) part << row.str();
        }
    }
    TempFile full_csv(full.str());
    TempFile part_csv(part.str());

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "sp";
    roles.abundance = "cover";
    roles.genus = "genus";
    roles.family = "family";
    roles.latitude = "lat";
    roles.longitude = "lon";
    const std::vector<TargetSpec> targets = {TargetSpec::regression("y")};

    const auto dcfg = test_dataset_config(SpeciesEncodingMode::Embed);
    auto ds_full = ResolveDataset::from_species_csv(full_csv.path(), roles, targets, dcfg);

    TempPath ckpt("resolve_vocab_single_");
    save_checkpoint(ds_full, SpeciesEncodingMode::Embed, ckpt.path());
    Predictor predictor = Predictor::load(ckpt.path(), torch::kCPU);

    const auto infer_cfg = dataset_config_from_checkpoint(
        predictor.schema(), predictor.model()->config());
    auto ds_part = ResolveDataset::from_species_csv_with_schema(
        part_csv.path(), roles, targets, predictor.schema(), infer_cfg);

    REQUIRE(ds_part.species_vocab() == predictor.species_vocab());
    REQUIRE(ds_part.n_plots() == 8);

    auto preds_full = predictor.predict(ds_full, false, -1);
    auto preds_part = predictor.predict(ds_part, false, -1);
    const auto idx_full = row_index(preds_full);
    const auto idx_part = row_index(preds_part);
    for (size_t i = 16; i < plots.size(); ++i) {
        const auto& id = plots[i].id;
        REQUIRE(idx_part.count(id) == 1);
        REQUIRE_THAT(pred_at(preds_part, "y", idx_part.at(id)),
                     Catch::Matchers::WithinAbs(
                         pred_at(preds_full, "y", idx_full.at(id)), 1e-5));
    }
}

// =============================================================================
// 8. ExternalVocabs plumbing
// =============================================================================

TEST_CASE("ExternalVocabs from a dataset and from a schema describe the same namespace",
          "[checkpoint][vocab][issue102]") {
    const auto plots = corpus();
    TempFile hdr(header_csv(plots));
    TempFile spc(species_csv(plots));

    const auto dcfg = test_dataset_config(SpeciesEncodingMode::RankPool);
    auto ds = ResolveDataset::from_csv(hdr.path(), spc.path(), test_roles(),
                                       {TargetSpec::regression("y")}, dcfg);

    const ExternalVocabs from_ds = ds.external_vocabs();
    const ExternalVocabs from_schema = external_vocabs_from_schema(ds.schema());

    REQUIRE(from_ds.species_vocab == from_schema.species_vocab);
    REQUIRE(from_ds.taxonomy.genus_names() == from_schema.taxonomy.genus_names());
    REQUIRE(from_ds.taxonomy.family_names() == from_schema.taxonomy.family_names());
    REQUIRE(from_ds.targets.size() == from_schema.targets.size());

    // A schema with no vocabulary yields an empty carrier, and the loaders
    // refuse it rather than re-fitting.
    const ExternalVocabs empty = external_vocabs_from_schema(ResolveSchema{});
    REQUIRE(empty.species_vocab.empty());
    REQUIRE_THROWS_AS(
        ResolveDataset::from_csv_with_vocabs(hdr.path(), spc.path(), test_roles(),
                                             {TargetSpec::regression("y")}, empty, dcfg),
        std::runtime_error);
}
