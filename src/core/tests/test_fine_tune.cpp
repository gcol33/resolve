// Fine-tuning from a trained model (fine_tune.hpp) and suites derived from one.
//
// Contract:
//   * Trainer::init_from copies every parameter and buffer outside the task
//     heads from the parent, starts every new head fresh, and with
//     InitReuse::EncoderHeads carries a shared target's head and its scaler;
//   * the continuous block is standardised with the parent's scalers;
//   * ModelConfig::freeze_encoder keeps everything outside the heads
//     bit-identical through a fit (batch normalisation statistics included)
//     while the heads train;
//   * the checkpoint records its parent (SHA-256, size, the suite, target and
//     seed it came from), in the archive and the JSON sidecar;
//   * a model that does not read the plots as its parent does is refused,
//     every departure listed;
//   * a frozen encoder from a parent trained on one target lets new heads
//     recover a rule of the same composition gradient, for a regression and a
//     classification target ([recovery]);
//   * ParentModel::open reads a suite member after checking its checksum;
//     write_derived_suite releases fine-tuned members as a suite that
//     SuitePredictor scores, and loading refuses a member whose recorded
//     parent or classes differ from the manifest's.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "resolve/checkpoint.hpp"
#include "resolve/dataset.hpp"
#include "resolve/fine_tune.hpp"
#include "resolve/json.hpp"
#include "resolve/loss.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/process.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/sha256.hpp"
#include "resolve/suite.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;
using Catch::Matchers::ContainsSubstring;

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// A composition gradient
// ---------------------------------------------------------------------------
//
// Plot i sits at a position t_i in [0, 1) along a gradient (a golden-ratio
// sequence, so any run of plots covers it evenly). Species sp_j (j < 24) has
// its optimum at j / 23 and is recorded where |t - j / 23| < 0.22, its cover
// falling with the distance, so a plot's species composition fixes t. The
// coordinates are unrelated to t. Targets: g = t (what the parent learns),
// soil = 1 + 3 t, zone = the third of the gradient t falls in (Z0, Z1, Z2).

constexpr int kSpecies = 24;

double gradient(int i) { return std::fmod(0.6180339887498949 * (i + 1), 1.0); }

std::string header_csv(int first, int last) {
    std::ostringstream out;
    out << "plot_id,lat,lon,g,soil,zone\n";
    for (int i = first; i < last; ++i) {
        const double t = gradient(i);
        out << "p" << i << "," << (46.0 + std::fmod(0.37 * i, 1.0)) << ","
            << (10.0 + std::fmod(0.53 * i, 1.0)) << "," << t << "," << (1.0 + 3.0 * t) << ",Z"
            << std::min(2, static_cast<int>(3.0 * t)) << "\n";
    }
    return out.str();
}

std::string species_csv(int first, int last) {
    std::ostringstream out;
    out << "plot_id,sp,cover,genus,family\n";
    for (int i = first; i < last; ++i) {
        const double t = gradient(i);
        for (int j = 0; j < kSpecies; ++j) {
            const double d = std::fabs(t - j / 23.0);
            if (d >= 0.22) continue;
            out << "p" << i << ",sp_" << j << "," << (1.0 + 10.0 * (1.0 - d / 0.22)) << ",gen_"
                << (j / 4) << ",fam_" << (j / 8) << "\n";
        }
    }
    return out.str();
}

DatasetConfig dataset_config() {
    DatasetConfig cfg = vocab_dataset_config(SpeciesEncodingMode::Embed, /*track_unknown=*/true);
    cfg.top_k_species = 12;
    return cfg;
}

ModelConfig parent_model_config() {
    ModelConfig cfg = vocab_model_config(SpeciesEncodingMode::Embed);
    cfg.top_k_species = 12;
    cfg.hidden_dims = {32, 16};
    return cfg;
}

// The plots of [first, last) on disk, read with the targets a case names.
class Plots {
public:
    Plots(int first, int last) {
        dir_.write("header.csv", header_csv(first, last));
        dir_.write("species.csv", species_csv(first, last));
    }
    [[nodiscard]] ResolveDataset load(const std::vector<TargetSpec>& targets,
                                      const DatasetConfig& config = dataset_config(),
                                      const ExternalVocabs* vocabs = nullptr) const {
        return ResolveDataset::load(TableSource::csv(dir_.file("header.csv")),
                                    TableSource::csv(dir_.file("species.csv")),
                                    Roles().taxonomy().coordinates(), targets, config, vocabs);
    }
    [[nodiscard]] std::string header() const { return dir_.file("header.csv"); }
    [[nodiscard]] std::string species() const { return dir_.file("species.csv"); }

private:
    TempDir dir_;
};

TargetSpec zone_target() {
    TargetSpec t = TargetSpec::classification("zone", 3);
    return t;
}

// A parent trained on g over plots [0, 300) for `epochs`, saved to `path`.
void train_parent(const std::string& path, int epochs, uint64_t seed = 7) {
    fs::create_directories(fs::path(path).parent_path());
    manual_seed(seed);
    const Plots plots(0, 300);
    const ResolveDataset ds = plots.load({TargetSpec::regression("g")});
    TrainConfig tc = epochs > 10 ? recovery_train_config(epochs) : quiet_train_config(epochs);
    tc.log = null_log;
    Trainer trainer(ResolveModel(ds.schema(), model_config_for_dataset(ds, parent_model_config())), tc);
    trainer.prepare_data(ds, 0.2f, 1);
    trainer.fit();
    trainer.save(path, &trainer.run_metadata());
}

// A fine-tuning dataset over plots [first, last) in the parent's namespace.
ResolveDataset child_dataset(const ParentModel& parent, const Plots& plots,
                             const std::vector<TargetSpec>& targets) {
    const ExternalVocabs vocabs = parent.vocabs();
    return plots.load(targets, parent.dataset_config(), &vocabs);
}

Trainer child_trainer(const ParentModel& parent, const ResolveDataset& ds, bool freeze,
                      TrainConfig tc = quiet_train_config(3)) {
    ModelConfig mc = parent.model_config();
    mc.freeze_encoder = freeze;
    tc.log = null_log;
    return Trainer(ResolveModel(ds.schema(), model_config_for_dataset(ds, mc)), tc);
}

bool same(const torch::Tensor& a, const torch::Tensor& b) {
    return a.sizes() == b.sizes() && torch::equal(a.to(torch::kCPU), b.to(torch::kCPU));
}

}  // namespace

// =============================================================================
// Warm start
// =============================================================================

TEST_CASE("A warm start copies every tensor outside the heads and starts new heads fresh",
          "[fine_tune]") {
    TempDir dir;
    train_parent(dir.file("parent.pt"), 2);
    const ParentModel parent = ParentModel::open(dir.file("parent.pt"));
    const Plots plots(300, 380);
    const ResolveDataset ds =
        child_dataset(parent, plots, {TargetSpec::regression("soil"), zone_target()});

    Trainer trainer = child_trainer(parent, ds, false);
    trainer.init_from(parent);

    int64_t n_shared = 0;
    for (const auto& pair : trainer.model()->named_parameters()) {
        if (trainer.model()->head_of(pair.key())) {
            CHECK(parent.parameters().count(pair.key()) == 0);
            continue;
        }
        ++n_shared;
        INFO(pair.key());
        REQUIRE(parent.parameters().count(pair.key()) == 1);
        CHECK(same(pair.value(), parent.parameters().at(pair.key())));
    }
    for (const auto& pair : trainer.model()->named_buffers()) {
        if (trainer.model()->head_of(pair.key())) continue;
        INFO(pair.key());
        REQUIRE(parent.buffers().count(pair.key()) == 1);
        CHECK(same(pair.value(), parent.buffers().at(pair.key())));
    }
    CHECK(n_shared > 0);
    REQUIRE(trainer.parent().has_value());
    CHECK(trainer.parent()->reuse == InitReuse::Encoder);
    CHECK(trainer.parent()->heads.empty());
    CHECK(trainer.parent()->sha256 == sha256_file(dir.file("parent.pt")));
}

TEST_CASE("The parent's scalers standardise a fine-tuned model's plots", "[fine_tune]") {
    TempDir dir;
    train_parent(dir.file("parent.pt"), 2);
    const ParentModel parent = ParentModel::open(dir.file("parent.pt"));
    const Plots plots(300, 380);
    const ResolveDataset ds = child_dataset(parent, plots, {TargetSpec::regression("soil")});

    SECTION("init_from before prepare_data") {
        Trainer trainer = child_trainer(parent, ds, false);
        trainer.init_from(parent);
        trainer.prepare_data(ds, 0.25f, 3);
        CHECK(same(trainer.scalers().continuous_mean, parent.scalers().continuous_mean));
        CHECK(same(trainer.scalers().continuous_scale, parent.scalers().continuous_scale));
        CHECK(trainer.scalers().target_scalers.count("soil") == 1);
    }
    SECTION("init_from after prepare_data re-expresses the split") {
        Trainer before = child_trainer(parent, ds, false);
        before.init_from(parent);
        before.prepare_data(ds, 0.25f, 3);
        Trainer after = child_trainer(parent, ds, false);
        after.prepare_data(ds, 0.25f, 3);
        after.init_from(parent);
        CHECK(same(after.scalers().continuous_mean, parent.scalers().continuous_mean));
        CHECK(same(before.scalers().target_scalers.at("soil").first,
                   after.scalers().target_scalers.at("soil").first));
    }
}

TEST_CASE("A frozen encoder stays bit-identical through a fit while the heads train",
          "[fine_tune]") {
    TempDir dir;
    train_parent(dir.file("parent.pt"), 2);
    const ParentModel parent = ParentModel::open(dir.file("parent.pt"));
    const Plots plots(300, 380);
    const ResolveDataset ds = child_dataset(parent, plots, {TargetSpec::regression("soil")});

    TrainConfig tc = quiet_train_config(4);
    tc.weight_decay = 0.1f;
    tc.lr = 1e-2f;
    Trainer trainer = child_trainer(parent, ds, true, tc);
    trainer.init_from(parent);
    trainer.prepare_data(ds, 0.25f, 3);

    std::vector<std::pair<std::string, torch::Tensor>> heads_before;
    for (const auto& pair : trainer.model()->named_parameters()) {
        if (trainer.model()->head_of(pair.key())) {
            heads_before.emplace_back(pair.key(), pair.value().detach().clone());
            CHECK(pair.value().requires_grad());
        } else {
            CHECK_FALSE(pair.value().requires_grad());
        }
    }
    trainer.fit();

    for (const auto& pair : trainer.model()->named_parameters()) {
        if (trainer.model()->head_of(pair.key())) continue;
        INFO(pair.key());
        CHECK(same(pair.value(), parent.parameters().at(pair.key())));
    }
    for (const auto& pair : trainer.model()->named_buffers()) {
        if (trainer.model()->head_of(pair.key())) continue;
        INFO(pair.key());
        CHECK(same(pair.value(), parent.buffers().at(pair.key())));
    }
    bool moved = false;
    const auto params = trainer.model()->named_parameters();
    for (const auto& [name, before] : heads_before) {
        if (!same(*params.find(name), before)) moved = true;
    }
    CHECK(moved);

    // In training mode only the heads train.
    trainer.model()->train();
    for (const auto& child : trainer.model()->named_children()) {
        INFO(child.key());
        CHECK(child.value()->is_training() == (child.key().rfind("head_", 0) == 0));
    }
}

TEST_CASE("reuse = encoder_heads carries a shared target's head and its scaler", "[fine_tune]") {
    TempDir dir;
    train_parent(dir.file("parent.pt"), 2);
    const ParentModel parent = ParentModel::open(dir.file("parent.pt"));
    const Plots plots(300, 380);
    const ResolveDataset ds =
        child_dataset(parent, plots, {TargetSpec::regression("g"), TargetSpec::regression("soil")});

    Trainer trainer = child_trainer(parent, ds, false);
    trainer.init_from(parent, InitReuse::EncoderHeads);
    REQUIRE(trainer.parent()->heads == std::vector<std::string>{"g"});
    for (const auto& pair : trainer.model()->named_parameters()) {
        const auto head = trainer.model()->head_of(pair.key());
        if (head && *head == "g") {
            INFO(pair.key());
            CHECK(same(pair.value(), parent.parameters().at(pair.key())));
        }
    }
    trainer.prepare_data(ds, 0.25f, 3);
    CHECK(same(trainer.scalers().target_scalers.at("g").first,
               parent.scalers().target_scalers.at("g").first));
    CHECK(same(trainer.scalers().target_scalers.at("g").second,
               parent.scalers().target_scalers.at("g").second));
    CHECK_FALSE(same(trainer.scalers().target_scalers.at("soil").first,
                     parent.scalers().target_scalers.at("g").first));
}

TEST_CASE("A shared target name of another task cannot carry its head", "[fine_tune]") {
    TempDir dir;
    train_parent(dir.file("parent.pt"), 2);
    const ParentModel parent = ParentModel::open(dir.file("parent.pt"));
    // A plot table whose "g" column holds classes.
    TempDir data;
    std::string header = header_csv(300, 380);
    {
        std::istringstream in(header);
        std::ostringstream out;
        std::string line;
        std::getline(in, line);
        out << "plot_id,lat,lon,g\n";
        while (std::getline(in, line)) {
            std::vector<std::string> cells;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) cells.push_back(cell);
            out << cells[0] << "," << cells[1] << "," << cells[2] << "," << cells[5] << "\n";
        }
        header = out.str();
    }
    data.write("header.csv", header);
    data.write("species.csv", species_csv(300, 380));
    const ExternalVocabs vocabs = parent.vocabs();
    ExternalVocabs without_targets = vocabs;
    without_targets.targets.clear();
    const ResolveDataset ds = ResolveDataset::load(
        TableSource::csv(data.file("header.csv")), TableSource::csv(data.file("species.csv")),
        Roles().taxonomy().coordinates(), {TargetSpec::classification("g", 3)},
        parent.dataset_config(), &without_targets);
    Trainer trainer = child_trainer(parent, ds, false);
    CHECK_THROWS_WITH(trainer.init_from(parent, InitReuse::EncoderHeads),
                      ContainsSubstring("target 'g'") && ContainsSubstring("cannot be carried over"));
    // The encoder alone starts fine.
    CHECK_NOTHROW(trainer.init_from(parent, InitReuse::Encoder));
}

TEST_CASE("A model that does not read the plots as its parent does is refused", "[fine_tune]") {
    TempDir dir;
    train_parent(dir.file("parent.pt"), 2);
    const ParentModel parent = ParentModel::open(dir.file("parent.pt"));
    const Plots plots(300, 380);

    SECTION("its own species vocabulary") {
        const ResolveDataset own = plots.load({TargetSpec::regression("soil")});
        Trainer trainer = child_trainer(parent, own, false);
        CHECK_THROWS_WITH(trainer.init_from(parent), ContainsSubstring("species vocabularies"));
    }
    SECTION("an encoder of other widths") {
        const ResolveDataset ds = child_dataset(parent, plots, {TargetSpec::regression("soil")});
        ModelConfig mc = parent.model_config();
        mc.hidden_dims = {24, 16};
        Trainer trainer(ResolveModel(ds.schema(), model_config_for_dataset(ds, mc)),
                        quiet_train_config());
        CHECK_THROWS_WITH(trainer.init_from(parent),
                          ContainsSubstring("model config 'hidden_dims'") &&
                              ContainsSubstring("[24,16] here and [32,16] in the parent"));
    }
    SECTION("other loader settings") {
        DatasetConfig cfg = parent.dataset_config();
        cfg.top_k_species = 8;
        const ExternalVocabs vocabs = parent.vocabs();
        const ResolveDataset ds = plots.load({TargetSpec::regression("soil")}, cfg, &vocabs);
        ModelConfig mc = parent.model_config();
        Trainer trainer(ResolveModel(ds.schema(), model_config_for_dataset(ds, mc)),
                        quiet_train_config());
        CHECK_THROWS_WITH(trainer.init_from(parent),
                          ContainsSubstring("loader setting 'top_k_species'"));
    }
    SECTION("head settings may differ") {
        const ResolveDataset ds = child_dataset(parent, plots, {TargetSpec::regression("soil")});
        ModelConfig mc = parent.model_config();
        mc.head_hidden_dims = {8};
        mc.dropout = 0.3f;
        Trainer trainer(ResolveModel(ds.schema(), model_config_for_dataset(ds, mc)),
                        quiet_train_config());
        CHECK_NOTHROW(trainer.init_from(parent));
    }
    SECTION("suite settings with a checkpoint") {
        InitFromConfig cfg;
        cfg.suite_target = "g";
        CHECK_THROWS_AS(ParentModel::open(dir.file("parent.pt"), cfg), std::invalid_argument);
    }
}

TEST_CASE("A fine-tuned checkpoint records its parent", "[fine_tune][checkpoint]") {
    TempDir dir;
    train_parent(dir.file("parent.pt"), 2);
    const ParentModel parent = ParentModel::open(dir.file("parent.pt"));
    const Plots plots(300, 380);
    const ResolveDataset ds = child_dataset(parent, plots, {TargetSpec::regression("soil")});
    Trainer trainer = child_trainer(parent, ds, true);
    trainer.init_from(parent);
    trainer.prepare_data(ds, 0.25f, 3);
    trainer.fit();

    SECTION("with run metadata") {
        trainer.save(dir.file("child.pt"), &trainer.run_metadata());
        const RunMetadata meta = Trainer::load_run_metadata(dir.file("child.pt"));
        REQUIRE(meta.parent.has_value());
        CHECK(*meta.parent == *trainer.parent());
        CHECK(meta.parent->sha256 == sha256_file(dir.file("parent.pt")));
        CHECK(meta.parent->bytes == static_cast<int64_t>(fs::file_size(dir.file("parent.pt"))));
        CHECK(meta.parent->n_tensors > 0);
        CHECK(meta.parent->suite.empty());
        const json::Value sidecar = json::read_file(dir.file("child.json"));
        REQUIRE(sidecar.find("parent") != nullptr);
        CHECK(sidecar.find("parent")->find("sha256")->as_string() == meta.parent->sha256);
        CHECK(sidecar.find("model_config")->find("freeze_encoder")->as_bool());
    }
    SECTION("without run metadata") {
        trainer.save(dir.file("child.pt"));
        const RunMetadata meta = Trainer::load_run_metadata(dir.file("child.pt"));
        REQUIRE(meta.parent.has_value());
        CHECK(meta.parent->sha256 == sha256_file(dir.file("parent.pt")));
    }
    SECTION("a fresh model records none") {
        const ResolveDataset own = plots.load({TargetSpec::regression("soil")});
        Trainer fresh(ResolveModel(own.schema(), model_config_for_dataset(own, parent_model_config())),
                      quiet_train_config());
        fresh.prepare_data(own, 0.25f, 3);
        fresh.save(dir.file("fresh.pt"), &fresh.run_metadata());
        CHECK_FALSE(Trainer::load_run_metadata(dir.file("fresh.pt")).parent.has_value());
    }
}

TEST_CASE("A frozen encoder lets new heads recover a rule of the parent's gradient",
          "[fine_tune][recovery]") {
    TempDir dir;
    train_parent(dir.file("parent.pt"), 200);
    const ParentModel parent = ParentModel::open(dir.file("parent.pt"));
    const Plots tune(300, 400);
    const Plots held_out(400, 500);
    const std::vector<TargetSpec> targets{TargetSpec::regression("soil"), zone_target()};
    const ResolveDataset ds = child_dataset(parent, tune, targets);

    manual_seed(11);
    TrainConfig tc = recovery_train_config(300);
    Trainer trainer = child_trainer(parent, ds, true, tc);
    trainer.init_from(parent);
    trainer.prepare_data(ds, 0.25f, 5);
    trainer.fit();
    trainer.save(dir.file("child.pt"), &trainer.run_metadata());

    Predictor predictor = Predictor::load(dir.file("child.pt"));
    const ResolveDataset test = child_dataset(parent, held_out, targets);
    const ResolvePredictions out = predictor.predict(test);

    const auto soil = out.predictions.at("soil");
    const auto soil_true = test.targets().at("soil");
    const float r2 = Metrics::r_squared(soil, soil_true);
    INFO("soil R^2 " << r2);
    CHECK(r2 > 0.9f);

    const auto zone = out.predictions.at("zone");
    const auto zone_true = test.targets().at("zone");
    const float acc = zone.eq(zone_true.to(zone.scalar_type())).to(torch::kFloat32).mean().item<float>();
    INFO("zone accuracy " << acc);
    CHECK(acc > 0.85f);
}

// =============================================================================
// Suites
// =============================================================================

namespace {

// A parent suite of two members (seeds 1 and 2) trained on g, sealed in `dir`.
SuiteManifest write_parent_suite(const TempDir& dir) {
    for (int seed : {1, 2}) {
        train_parent(dir.file("g/seed_" + std::to_string(seed) + ".pt"), 2,
                     static_cast<uint64_t>(seed));
    }
    SuiteManifest m;
    m.name = "gradient_parent";
    m.licence = "CC-BY-4.0";
    m.engine_version = VERSION;
    m.taxonomy = "synthetic";
    m.training_scope = "300 synthetic plots";
    m.inputs.plot_id = "plot_id";
    m.inputs.species = "sp";
    m.inputs.abundance = "cover";
    m.inputs.genus = "genus";
    m.inputs.family = "family";
    m.inputs.latitude = "lat";
    m.inputs.longitude = "lon";
    SuiteTarget t;
    t.name = "g";
    t.task = TaskType::Regression;
    t.combine = SuiteCombine::Mean;
    t.outputs = {"g"};
    for (int seed : {1, 2}) {
        SuiteMember member;
        member.seed = seed;
        member.file = "g/seed_" + std::to_string(seed) + ".pt";
        t.members.push_back(member);
    }
    m.targets = {t};
    m.seal(dir.path());
    m.write(dir.path());
    return m;
}

// Members fine-tuned from each parent seed, saved under `out`.
std::vector<DerivedSuiteMember> tune_members(const std::string& parent_dir, const Plots& plots,
                                             const TempDir& out) {
    std::vector<DerivedSuiteMember> members;
    for (int seed : {1, 2}) {
        InitFromConfig cfg;
        cfg.suite_target = "g";
        cfg.suite_seed = seed;
        const ParentModel parent = ParentModel::open(parent_dir, cfg);
        const ResolveDataset ds =
            child_dataset(parent, plots, {TargetSpec::regression("soil"), zone_target()});
        manual_seed(static_cast<uint64_t>(seed));
        Trainer trainer = child_trainer(parent, ds, true);
        trainer.init_from(parent);
        trainer.prepare_data(ds, 0.25f, seed);
        trainer.fit();
        const std::string path = out.file("tuned_" + std::to_string(seed) + ".pt");
        trainer.save(path, &trainer.run_metadata());
        DerivedSuiteMember member;
        member.checkpoint = path;
        member.seed = seed;
        members.push_back(member);
    }
    return members;
}

}  // namespace

TEST_CASE("A suite member is opened as a parent after its checksum", "[fine_tune][suite]") {
    TempDir dir;
    const SuiteManifest manifest = write_parent_suite(dir);

    InitFromConfig cfg;
    cfg.suite_seed = 2;
    const ParentModel parent = ParentModel::open(dir.path(), cfg);
    CHECK(parent.record().suite == "gradient_parent");
    CHECK(parent.record().suite_target == "g");
    CHECK(parent.record().suite_seed == 2);
    CHECK(parent.record().licence == "CC-BY-4.0");
    CHECK(parent.record().training_scope == "300 synthetic plots");
    CHECK(parent.record().sha256 == manifest.targets[0].members[1].sha256);

    cfg.suite_seed = 9;
    CHECK_THROWS_WITH(ParentModel::open(dir.path(), cfg), ContainsSubstring("no member of seed 9"));
    cfg.suite_seed = -1;
    cfg.suite_target = "habitat";
    CHECK_THROWS_WITH(ParentModel::open(dir.path(), cfg), ContainsSubstring("habitat"));

    // A member that differs from its manifest entry is not read.
    {
        std::ofstream out(dir.file("g/seed_1.pt"), std::ios::binary | std::ios::app);
        out << "x";
    }
    CHECK_THROWS_WITH(ParentModel::open(dir.path(), InitFromConfig{}),
                      ContainsSubstring("does not match its manifest"));
}

TEST_CASE("Fine-tuned members are released as a suite derived from their parent",
          "[fine_tune][suite]") {
    TempDir parent_dir;
    const SuiteManifest parent_manifest = write_parent_suite(parent_dir);
    const Plots plots(300, 380);
    TempDir work;
    const std::vector<DerivedSuiteMember> members = tune_members(parent_dir.path(), plots, work);

    DerivedSuiteSpec spec;
    spec.parent = parent_dir.path();
    spec.name = "gradient_soil";
    spec.licence = "CC-BY-4.0";
    spec.training_scope = "80 synthetic plots";
    DerivedSuiteTarget soil;
    soil.name = "soil";
    soil.units = "pH";
    soil.members = members;
    DerivedSuiteTarget zone;
    zone.name = "zone";
    zone.members = members;
    spec.targets = {soil, zone};

    TempDir out;
    const std::string derived = out.file("suite");
    const SuiteManifest manifest = write_derived_suite(derived, spec);

    CHECK(manifest.format_version() == 5);
    CHECK(json::read_file(derived + "/manifest.json").find("format_version")->as_int() == 5);
    REQUIRE(manifest.targets.size() == 2);
    const SuiteTarget& s = manifest.targets[0];
    CHECK(s.combine == SuiteCombine::Mean);
    CHECK(s.units == "pH");
    REQUIRE(s.derived_from.has_value());
    CHECK(s.derived_from->suite == "gradient_parent");
    CHECK(s.derived_from->target == "g");
    CHECK(s.derived_from->licence == "CC-BY-4.0");
    CHECK(s.derived_from->training_scope == "300 synthetic plots");
    CHECK(s.members[0].parent_sha256 == parent_manifest.targets[0].members[0].sha256);
    CHECK(s.members[1].parent_sha256 == parent_manifest.targets[0].members[1].sha256);
    const SuiteTarget& z = manifest.targets[1];
    CHECK(z.combine == SuiteCombine::Vote);
    CHECK(z.classes == std::vector<std::string>{"Z0", "Z1", "Z2"});
    CHECK(manifest.inputs.latitude == "lat");
    CHECK(manifest.training_scope == "80 synthetic plots");

    SuitePredictor suite = SuitePredictor::load(derived);
    const SuitePredictions p = suite.predict(SuiteInput::csv(plots.header(), plots.species()));
    CHECK(p.target("soil").value.size(0) == 80);
    CHECK(p.target("zone").class_names == std::vector<std::string>{"Z0", "Z1", "Z2"});

    SECTION("a member whose recorded parent differs is refused") {
        json::Value text = json::read_file(derived + "/manifest.json");
        SuiteManifest edited = SuiteManifest::from_json(text);
        edited.targets[0].members[0].parent_sha256 = std::string(64, 'a');
        edited.write(derived);
        CHECK_THROWS_WITH(SuitePredictor::load(derived),
                          ContainsSubstring("started from a parent of SHA-256"));
    }
    SECTION("classes that differ from the checkpoint's are refused") {
        SuiteManifest edited = SuiteManifest::read(derived);
        edited.targets[1].classes = {"Z0", "Z2", "Z1"};
        edited.write(derived);
        CHECK_THROWS_WITH(SuitePredictor::load(derived), ContainsSubstring("classes"));
    }
    SECTION("a member started from another suite is refused") {
        DerivedSuiteSpec other = spec;
        TempDir unrelated;
        SuiteManifest renamed = SuiteManifest::read(parent_dir.path());
        renamed.name = "another_suite";
        renamed.write(parent_dir.path());
        CHECK_THROWS_WITH(write_derived_suite(unrelated.file("suite"), other),
                          ContainsSubstring("started from suite 'gradient_parent'"));
    }
    SECTION("a directory holding a manifest is not overwritten") {
        CHECK_THROWS_WITH(write_derived_suite(derived, spec),
                          ContainsSubstring("already holds a suite manifest"));
    }
}

TEST_CASE("A manifest uses format 5 only for a class list or a derived target", "[fine_tune][suite]") {
    SuiteManifest m;
    SuiteTarget t;
    t.combine = SuiteCombine::Vote;
    t.task = TaskType::Classification;
    m.targets = {t};
    CHECK(m.format_version() == 1);
    m.targets[0].classes = {"a", "b"};
    CHECK(m.format_version() == 5);
    m.targets[0].classes.clear();
    m.targets[0].derived_from = SuiteDerivation{"parent", "habitat", "1.1.0", "CC-BY-4.0", "EVA"};
    CHECK(m.format_version() == 5);
}
