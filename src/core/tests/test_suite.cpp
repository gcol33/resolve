// Tests for model suites (suite.hpp) and the pieces they stand on: the JSON
// reader/writer, SHA-256, the combination rules, the per-plot species
// recognition report, and DatasetConfig::zero_abundance_as.
//
// The end-to-end cases build a real suite on disk: three targets -- a class
// vote, a mean, and a circular mean read from a sine / cosine pair -- whose
// members are checkpoints of different seeds. The vote target is trained on a
// subset of the plots, so its species vocabulary differs from the other two
// and the suite must encode the input twice. Every combined prediction is
// checked against the same rule applied to predictions made member by member
// through the single-model Predictor.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "resolve/checkpoint.hpp"
#include "resolve/csv_reader.hpp"
#include "resolve/dataset.hpp"
#include "resolve/json.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/sha256.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/suite.hpp"
#include "resolve/trainer.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace fs = std::filesystem;

namespace {

// A scratch directory, removed with everything in it on scope exit.
class TempDir {
public:
    TempDir() {
        path_ = fs::temp_directory_path() / ("resolve_suite_" + std::to_string(counter_++) +
                                             "_" + std::to_string(::time(nullptr)));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    [[nodiscard]] std::string path() const { return path_.string(); }
    [[nodiscard]] std::string file(const std::string& name) const { return (path_ / name).string(); }
    void write(const std::string& name, const std::string& content) const {
        fs::create_directories((path_ / name).parent_path());
        std::ofstream out(path_ / name, std::ios::binary);
        out << content;
    }

private:
    fs::path path_;
    static int counter_;
};
int TempDir::counter_ = 0;

// ---------------------------------------------------------------------------
// Synthetic plots
// ---------------------------------------------------------------------------
//
// 30 plots over five species, three to four species per plot, with a class
// label, a value and a bearing. Plots P20..P29 are the only ones that record
// sp_e, and the vote target trains on P0..P19, so its vocabulary lacks sp_e
// while the other two targets' vocabularies hold it.

struct Plot {
    std::string id;
    std::vector<std::string> species;
    double lat, lon, cov;
    std::string hab;
    double y;
    double bearing;
};

std::vector<Plot> corpus() {
    std::vector<Plot> plots;
    const std::vector<std::vector<std::string>> lists = {
        {"sp_a", "sp_b", "sp_c"},
        {"sp_a", "sp_b", "sp_d"},
        {"sp_b", "sp_c", "sp_d", "sp_a"},
    };
    for (int i = 0; i < 30; ++i) {
        Plot p;
        p.id = "P" + std::to_string(i);
        p.species = lists[static_cast<size_t>(i % 3)];
        if (i >= 20) p.species.push_back("sp_e");
        p.lat = 45.0 + 0.1 * i;
        p.lon = 10.0 - 0.05 * i;
        p.cov = 0.3 * i;
        p.hab = std::string(1, static_cast<char>('A' + i % 3));
        p.y = 2.0 + 0.5 * i;
        p.bearing = std::fmod(37.0 * i, 360.0);
        plots.push_back(std::move(p));
    }
    return plots;
}

std::string genus_of(const std::string& sp) { return "gen_" + sp.substr(3, 1); }
std::string family_of(const std::string& sp) {
    return (sp == "sp_a" || sp == "sp_b") ? "fam_x" : "fam_y";
}

std::string header_csv(const std::vector<Plot>& plots, bool with_targets) {
    std::ostringstream out;
    out << "plot,lat,lon,cov1";
    if (with_targets) out << ",hab,y,asp_sin,asp_cos";
    out << "\n";
    for (const auto& p : plots) {
        out << p.id << "," << p.lat << "," << p.lon << "," << p.cov;
        if (with_targets) {
            const double r = p.bearing * std::numbers::pi / 180.0;
            out << "," << p.hab << "," << p.y << "," << std::sin(r) << "," << std::cos(r);
        }
        out << "\n";
    }
    return out.str();
}

std::string species_csv(const std::vector<Plot>& plots) {
    std::ostringstream out;
    out << "plot,taxon,cover,genus,family\n";
    for (const auto& p : plots) {
        for (size_t k = 0; k < p.species.size(); ++k) {
            out << p.id << "," << p.species[k] << "," << (1.0 + 2.0 * k) << ","
                << genus_of(p.species[k]) << "," << family_of(p.species[k]) << "\n";
        }
    }
    return out.str();
}

RoleMapping training_roles() {
    RoleMapping roles;
    roles.plot_id = "plot";
    roles.species_id = "taxon";
    roles.abundance = "cover";
    roles.latitude = "lat";
    roles.longitude = "lon";
    roles.genus = "genus";
    roles.family = "family";
    roles.covariates = {"cov1"};
    return roles;
}

DatasetConfig training_dataset_config() {
    DatasetConfig cfg;
    cfg.species_encoding = SpeciesEncodingMode::RankPool;
    cfg.selection = SelectionMode::All;
    cfg.pool_weighting = PoolWeighting::Log1p;
    cfg.use_taxonomy = true;
    return cfg;
}

ModelConfig tiny_model() {
    ModelConfig cfg;
    cfg.species_encoding = SpeciesEncodingMode::RankPool;
    cfg.encoder_architecture = EncoderArchitecture::MLP;
    cfg.species_embed_dim = 8;
    cfg.genus_emb_dim = 4;
    cfg.family_emb_dim = 4;
    cfg.hidden_dims = {16, 8};
    return cfg;
}

// A member checkpoint with seed-dependent random weights: enough for the
// members to disagree, and fit() is not needed to test how they combine.
void save_member(const ResolveDataset& ds, int seed, const std::string& path) {
    torch::manual_seed(static_cast<uint64_t>(seed));
    ResolveModel model(ds.schema(), tiny_model());
    TrainConfig tc;
    tc.batch_size = 8;
    tc.max_epochs = 1;
    tc.log = null_log;
    Trainer trainer(model, tc);
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/seed);
    fs::create_directories(fs::path(path).parent_path());
    trainer.save(path);
}

SuiteManifest base_manifest() {
    SuiteManifest m;
    m.name = "test_suite";
    m.description = "three targets over two vocabularies";
    m.licence = "CC-BY-4.0";
    m.engine_version = VERSION;
    m.taxonomy = "synthetic";
    m.training_scope = "30 synthetic plots";
    m.inputs.plot_id = "plot";
    m.inputs.species = "taxon";
    m.inputs.abundance = "cover";
    m.inputs.genus = "genus";
    m.inputs.family = "family";
    m.inputs.latitude = "lat";
    m.inputs.longitude = "lon";
    m.inputs.covariates = {"cov1"};
    m.inputs.abundance_units = "percent cover";
    m.inputs.zero_abundance_as = 1.0f;
    return m;
}

// Build the three-target suite under `dir`. Returns the manifest written.
SuiteManifest build_suite(const TempDir& dir) {
    const auto plots = corpus();
    const std::vector<Plot> first20(plots.begin(), plots.begin() + 20);
    dir.write("train/header.csv", header_csv(plots, true));
    dir.write("train/species.csv", species_csv(plots));
    dir.write("train/header20.csv", header_csv(first20, true));
    dir.write("train/species20.csv", species_csv(first20));

    const auto roles = training_roles();
    const auto cfg = training_dataset_config();
    const auto hab_ds = ResolveDataset::from_csv(
        dir.file("train/header20.csv"), dir.file("train/species20.csv"), roles,
        {TargetSpec::classification("hab", 0)}, cfg);
    const auto y_ds = ResolveDataset::from_csv(
        dir.file("train/header.csv"), dir.file("train/species.csv"), roles,
        {TargetSpec::regression("y")}, cfg);
    const auto asp_ds = ResolveDataset::from_csv(
        dir.file("train/header.csv"), dir.file("train/species.csv"), roles,
        {TargetSpec::regression("asp_sin"), TargetSpec::regression("asp_cos")}, cfg);

    SuiteManifest m = base_manifest();
    auto add = [&](const std::string& name, TaskType task, SuiteCombine combine,
                   std::vector<std::string> outputs, const ResolveDataset& ds,
                   std::vector<int> seeds) {
        SuiteTarget t;
        t.name = name;
        t.task = task;
        t.combine = combine;
        t.outputs = std::move(outputs);
        for (const int seed : seeds) {
            const std::string file = name + "/seed_" + std::to_string(seed) + "/model_final.pt";
            save_member(ds, seed, dir.file(file));
            t.members.push_back(SuiteMember{seed, file, "", 0});
        }
        m.targets.push_back(std::move(t));
        return &m.targets.back();
    };
    add("hab", TaskType::Classification, SuiteCombine::Vote, {"hab"}, hab_ds, {0, 1, 2});
    SuiteTarget* y = add("y", TaskType::Regression, SuiteCombine::Mean, {"y"}, y_ds, {3, 4});
    y->units = "m";
    y->status = SuiteTargetStatus::Experimental;
    y->limit = "reliable within the synthetic corpus only";
    y->validation.set("canonical", json::parse(R"({"mae": 1.5, "n": 6})"));
    SuiteTarget* asp = add("aspect", TaskType::Regression, SuiteCombine::CircularMean,
                           {"asp_sin", "asp_cos"}, asp_ds, {5, 6});
    asp->units = "degrees";
    asp->period = 360.0;

    m.seal(dir.path());
    m.write(dir.path());
    return m;
}

// The scoring input: all 30 plots plus one plot recording an unseen species,
// with NO target column (the plots to score carry no answer).
struct ScoringFiles {
    std::string header, species;
};

ScoringFiles scoring_files(const TempDir& dir) {
    auto plots = corpus();
    Plot novel;
    novel.id = "NEW";
    novel.species = {"sp_a", "sp_z"};
    novel.lat = 46.0;
    novel.lon = 9.5;
    novel.cov = 1.0;
    plots.push_back(novel);
    dir.write("score/header.csv", header_csv(plots, false));
    dir.write("score/species.csv", species_csv(plots));
    return {dir.file("score/header.csv"), dir.file("score/species.csv")};
}

// A member's own predictions through the single-model path.
ResolvePredictions predict_member(const std::string& path, const ScoringFiles& in) {
    Predictor p = Predictor::load(path);
    auto cfg = dataset_config_from_checkpoint(p.schema(), p.model()->config());
    cfg.zero_abundance_as = 1.0f;
    const auto ds = ResolveDataset::from_csv_with_vocabs(
        in.header, in.species, training_roles(), {}, p.external_vocabs(), cfg);
    return p.predict(ds);
}

std::map<std::string, int64_t> index_of(const std::vector<std::string>& ids) {
    std::map<std::string, int64_t> out;
    for (size_t i = 0; i < ids.size(); ++i) out[ids[i]] = static_cast<int64_t>(i);
    return out;
}

}  // namespace

// =============================================================================
// JSON
// =============================================================================

TEST_CASE("json round-trips every kind and keeps member order", "[suite][json]") {
    const std::string text = R"({
  "z": 1,
  "a": [true, false, null, -2.5, 1e-3, "x"],
  "nested": {"k": {"deep": [1, 2, {"e": "\u00e9\ud83c\udf3f"}]}},
  "big": 9007199254740991
})";
    const json::Value v = json::parse(text);
    REQUIRE(v.members().front().first == "z");
    REQUIRE(v.members()[1].first == "a");
    REQUIRE(v.at("a").items()[3].as_number() == -2.5);
    REQUIRE(v.at("big").as_int() == 9007199254740991LL);
    const std::string e = v.at("nested").at("k").at("deep").items()[2].at("e").as_string();
    REQUIRE(e == "\xC3\xA9\xF0\x9F\x8C\xBF");  // é then U+1F33F, as UTF-8

    for (int indent : {0, 2}) {
        REQUIRE(json::parse(json::dump(v, indent)) == v);
    }
    // 0.1 has no exact binary form; 17 significant digits read back exactly.
    REQUIRE(json::parse(json::dump(json::Value(0.1))).as_number() == 0.1);
    REQUIRE(json::dump(json::Value(3.0)) == "3");
    const std::string escaped = "\"a\\\"b\\\\c\\n\"";
    REQUIRE(json::dump(json::Value("a\"b\\c\n")) == escaped);
}

TEST_CASE("json rejects what is not JSON, naming where", "[suite][json]") {
    REQUIRE_THROWS_WITH(json::parse("{\"a\": 1,\n  \"a\": 2}"),
                        ContainsSubstring("duplicate member 'a'") &&
                            ContainsSubstring("line 2"));
    REQUIRE_THROWS_WITH(json::parse("[1, 2] x"), ContainsSubstring("after the JSON value"));
    REQUIRE_THROWS(json::parse("{'a': 1}"));
    REQUIRE_THROWS(json::parse("[01]"));
    REQUIRE_THROWS(json::parse("\"\\ud800\""));
    REQUIRE_THROWS(json::parse("[1,]"));
    REQUIRE_THROWS(json::parse(""));
    REQUIRE_THROWS(json::dump(json::Value(std::nan(""))));
    REQUIRE_THROWS_WITH(json::Value(1.5).as_int(), ContainsSubstring("integer"));
    REQUIRE_THROWS_WITH(json::Value("x").as_number(), ContainsSubstring("a string"));
    // A byte-order mark is tolerated.
    REQUIRE(json::parse("\xEF\xBB\xBF{\"a\": 1}").at("a").as_int() == 1);
}

// =============================================================================
// SHA-256
// =============================================================================

TEST_CASE("SHA-256 matches the FIPS 180-4 test vectors", "[suite][sha256]") {
    REQUIRE(sha256_hex("") ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    REQUIRE(sha256_hex("abc") ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    REQUIRE(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // One million 'a', fed in uneven pieces, and read back from a file.
    const std::string million(1000000, 'a');
    Sha256 streamed;
    for (size_t at = 0; at < million.size();) {
        const size_t take = std::min<size_t>(1 + at % 97, million.size() - at);
        streamed.update(million.data() + at, take);
        at += take;
    }
    const std::string expected =
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
    REQUIRE(streamed.hex_digest() == expected);
    TempDir dir;
    dir.write("a.bin", million);
    REQUIRE(sha256_file(dir.file("a.bin")) == expected);
    REQUIRE_THROWS(sha256_file(dir.file("missing.bin")));
}

// =============================================================================
// Combination rules
// =============================================================================

TEST_CASE("a vote takes the majority class, ties to the lowest code", "[suite][combine]") {
    // Three members over four plots.
    const auto codes = torch::tensor({2, 1, 0, 3, 2, 1, 1, 3, 0, 2, 2, 3}, torch::kLong)
                           .reshape({3, 4});
    const auto c = combine_vote(codes, 4);
    REQUIRE(c.value[0].item<int64_t>() == 2);  // 2, 2, 0
    REQUIRE(c.value[1].item<int64_t>() == 1);  // 1, 1, 2
    REQUIRE(c.value[2].item<int64_t>() == 0);  // 0, 1, 2 -- three-way tie
    REQUIRE(c.value[3].item<int64_t>() == 3);  // unanimous
    REQUIRE_THAT(c.agreement[0].item<double>(), WithinAbs(2.0 / 3.0, 1e-6));
    REQUIRE_THAT(c.agreement[2].item<double>(), WithinAbs(1.0 / 3.0, 1e-6));
    REQUIRE_THAT(c.agreement[3].item<double>(), WithinAbs(1.0, 1e-6));
    REQUIRE_FALSE(c.dispersion.defined());
    REQUIRE_THROWS(combine_vote(codes, 3));  // code 3 outside [0, 3)
}

TEST_CASE("a mean reports the members' standard deviation", "[suite][combine]") {
    const auto v = torch::tensor({1.0, 10.0, 3.0, 10.0}, torch::kFloat32).reshape({2, 2});
    const auto c = combine_mean(v);
    REQUIRE_THAT(c.value[0].item<double>(), WithinAbs(2.0, 1e-6));
    REQUIRE_THAT(c.value[1].item<double>(), WithinAbs(10.0, 1e-6));
    REQUIRE_THAT(c.dispersion[0].item<double>(), WithinAbs(std::sqrt(2.0), 1e-6));
    REQUIRE_THAT(c.dispersion[1].item<double>(), WithinAbs(0.0, 1e-6));
    // One member has no spread to report.
    REQUIRE(std::isnan(combine_mean(v.slice(0, 0, 1)).dispersion[0].item<double>()));
}

TEST_CASE("a circular mean averages across the wrap", "[suite][combine]") {
    // 350 and 10 degrees meet at 0; an arithmetic mean would say 180.
    const auto b = torch::tensor({350.0, 90.0, 10.0, 90.0}, torch::kFloat32).reshape({2, 2});
    const auto c = combine_circular(b, 360.0);
    const double m0 = c.value[0].item<double>();
    REQUIRE((m0 < 1e-3 || m0 > 360.0 - 1e-3));
    REQUIRE_THAT(c.value[1].item<double>(), WithinAbs(90.0, 1e-4));
    // sqrt(-2 ln cos 10deg) in degrees.
    const double pi = std::numbers::pi;
    const double expected = std::sqrt(-2.0 * std::log(std::cos(10.0 * pi / 180.0))) *
                            180.0 / pi;
    REQUIRE_THAT(c.dispersion[0].item<double>(), WithinAbs(expected, 1e-3));
    REQUIRE_THAT(c.dispersion[1].item<double>(), WithinAbs(0.0, 1e-3));

    // A bearing from a sine / cosine pair of any length.
    const auto s = torch::tensor({1.0, 0.0, -2.0, 0.3}, torch::kFloat32);
    const auto k = torch::tensor({0.0, -1.0, 0.0, 0.3}, torch::kFloat32);
    const auto bearing = bearing_from_components(s, k, 360.0);
    REQUIRE_THAT(bearing[0].item<double>(), WithinAbs(90.0, 1e-4));
    REQUIRE_THAT(bearing[1].item<double>(), WithinAbs(180.0, 1e-4));
    REQUIRE_THAT(bearing[2].item<double>(), WithinAbs(270.0, 1e-4));
    REQUIRE_THAT(bearing[3].item<double>(), WithinAbs(45.0, 1e-4));
}

// =============================================================================
// Manifest
// =============================================================================

TEST_CASE("a manifest round-trips through JSON", "[suite][manifest]") {
    SuiteManifest m = base_manifest();
    SuiteTarget t;
    t.name = "aspect";
    t.combine = SuiteCombine::CircularMean;
    t.outputs = {"s", "c"};
    t.units = "degrees";
    t.period = 360.0;
    t.status = SuiteTargetStatus::Experimental;
    t.limit = "within represented databases";
    t.members.push_back({0, "aspect/seed_0/model_final.pt", std::string(64, 'a'), 10});
    t.training.set("fixed_epochs", 13);
    t.validation.set("canonical", json::parse(R"({"circular_mae": 40.1})"));
    m.targets.push_back(t);
    REQUIRE_NOTHROW(m.validate());

    const SuiteManifest back = SuiteManifest::from_json(json::parse(json::dump(m.to_json())));
    REQUIRE(back.to_json() == m.to_json());
    REQUIRE(back.inputs.zero_abundance_as == 1.0f);
    REQUIRE(back.targets[0].combine == SuiteCombine::CircularMean);
    REQUIRE(back.targets[0].validation.at("canonical").at("circular_mae").as_number() == 40.1);
}

TEST_CASE("manifest validation reports every problem at once", "[suite][manifest]") {
    SuiteManifest m = base_manifest();
    m.licence.clear();
    m.inputs.longitude.clear();
    SuiteTarget vote;
    vote.name = "hab";
    vote.task = TaskType::Regression;  // a vote needs classification
    vote.combine = SuiteCombine::Vote;
    vote.outputs = {"hab"};
    vote.members.push_back({0, "../outside.pt", "nothex", 0});
    vote.members.push_back({0, "hab/b.pt", std::string(64, 'b'), 5});
    SuiteTarget circ;
    circ.name = "hab";  // duplicate name
    circ.combine = SuiteCombine::CircularMean;
    circ.outputs = {"a", "b", "c"};
    circ.status = SuiteTargetStatus::Experimental;  // no limit
    circ.members.push_back({1, "C:/abs.pt", std::string(64, 'c'), 5});
    m.targets = {vote, circ};

    try {
        m.validate();
        FAIL("validate() accepted an invalid manifest");
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        for (const char* expected :
             {"licence: is required", "latitude and longitude", "task must be classification",
              "'..' segment", "64 lower-case hex", "bytes: must be positive",
              "seed: 0 appears twice", "'hab' names two targets", "one output (the bearing)",
              "positive period", "states its limit", "must not name a drive"}) {
            INFO(expected);
            REQUIRE_THAT(msg, ContainsSubstring(expected));
        }
    }
}

TEST_CASE("a manifest with an unknown member or a newer format is refused",
          "[suite][manifest]") {
    json::Value v = base_manifest().to_json();
    v.set("targets", json::Value::array());
    json::Value typo = v;
    typo.set("licnce", "x");
    REQUIRE_THROWS_WITH(SuiteManifest::from_json(typo),
                        ContainsSubstring("licnce is not a member"));
    json::Value newer = v;
    newer.set("format_version", kSuiteFormatVersion + 1);
    REQUIRE_THROWS_WITH(SuiteManifest::from_json(newer), ContainsSubstring("newer engine"));
    json::Value other = v;
    other.set("format", "something-else");
    REQUIRE_THROWS(SuiteManifest::from_json(other));
}

// =============================================================================
// Recognition and the zero-abundance reading
// =============================================================================

TEST_CASE("species recognition counts distinct names and recognised abundance",
          "[suite][recognition]") {
    const auto vocab = SpeciesVocab::from_map({{"sp_a", 1}, {"sp_b", 2}});
    const std::vector<SpeciesRecord> records = {
        {"sp_a", "", "", 2.0f, "p1"}, {"sp_a", "", "", 1.0f, "p1"},  // two layers, one species
        {"sp_x", "", "", 1.0f, "p1"},
        {"sp_b", "", "", 4.0f, "p2"},
        {"sp_y", "", "", 0.0f, "p4"},
    };
    const auto r = compute_species_recognition(records, {"p1", "p2", "p3", "p4"}, vocab);
    REQUIRE(r.n_species[0].item<float>() == 2.0f);
    REQUIRE(r.n_recognised[0].item<float>() == 1.0f);
    REQUIRE_THAT(r.count_share[0].item<double>(), WithinAbs(0.5, 1e-6));
    REQUIRE_THAT(r.abundance_share[0].item<double>(), WithinAbs(0.75, 1e-6));
    REQUIRE(r.count_share[1].item<float>() == 1.0f);
    REQUIRE(r.abundance_share[1].item<float>() == 1.0f);
    // A plot with no records has no share to report.
    REQUIRE(r.n_species[2].item<float>() == 0.0f);
    REQUIRE(std::isnan(r.count_share[2].item<float>()));
    REQUIRE(std::isnan(r.abundance_share[2].item<float>()));
    // A plot whose abundances sum to zero has a count share, no abundance share.
    REQUIRE(r.count_share[3].item<float>() == 0.0f);
    REQUIRE(std::isnan(r.abundance_share[3].item<float>()));

    // The abundance share is the complement of the model's unknown fraction.
    const auto u = compute_unknown_species_stats(records, {"p1", "p2"}, vocab);
    REQUIRE_THAT(u.fraction[0].item<double>(), WithinAbs(1.0 - 0.75, 1e-6));
}

TEST_CASE("zero_abundance_as reads a recorded 0 at the given abundance",
          "[suite][zero_abundance]") {
    TempDir dir;
    // Four plots, so the checkpoint written at the end has a test fold.
    const std::string rest = "p2,sp_a,3\np3,sp_b,2\np4,sp_a,4\n";
    dir.write("h.csv", "plot,y\np1,1\np2,2\np3,3\np4,4\n");
    dir.write("zero.csv", "plot,taxon,cover\np1,sp_a,0\np1,sp_b,5\n" + rest);
    dir.write("one.csv", "plot,taxon,cover\np1,sp_a,1\np1,sp_b,5\n" + rest);
    RoleMapping roles;
    roles.plot_id = "plot";
    roles.species_id = "taxon";
    roles.abundance = "cover";
    DatasetConfig cfg = training_dataset_config();
    cfg.use_taxonomy = false;

    cfg.zero_abundance_as = 1.0f;
    const auto zero = ResolveDataset::from_csv(dir.file("h.csv"), dir.file("zero.csv"), roles,
                                               {TargetSpec::regression("y")}, cfg);
    const auto one = ResolveDataset::from_csv(dir.file("h.csv"), dir.file("one.csv"), roles,
                                              {TargetSpec::regression("y")}, cfg);
    REQUIRE(torch::equal(zero.pool_weights(), one.pool_weights()));
    REQUIRE(zero.schema().zero_abundance_as == 1.0f);

    cfg.zero_abundance_as = 0.0f;
    const auto left = ResolveDataset::from_csv(dir.file("h.csv"), dir.file("zero.csv"), roles,
                                               {TargetSpec::regression("y")}, cfg);
    REQUIRE_FALSE(torch::equal(left.pool_weights(), one.pool_weights()));

    cfg.zero_abundance_as = -1.0f;
    REQUIRE_THROWS_WITH(
        ResolveDataset::from_csv(dir.file("h.csv"), dir.file("zero.csv"), roles,
                                 {TargetSpec::regression("y")}, cfg),
        ContainsSubstring("zero_abundance_as"));

    // The knob travels in the checkpoint and back into the loader config.
    cfg.zero_abundance_as = 1.0f;
    const auto ds = ResolveDataset::from_csv(dir.file("h.csv"), dir.file("zero.csv"), roles,
                                             {TargetSpec::regression("y")}, cfg);
    save_member(ds, 0, dir.file("m/model.pt"));
    const Predictor p = Predictor::load(dir.file("m/model.pt"));
    REQUIRE(p.schema().zero_abundance_as == 1.0f);
    REQUIRE(dataset_config_from_checkpoint(p.schema(), p.model()->config()).zero_abundance_as ==
            1.0f);
}

// =============================================================================
// End to end
// =============================================================================

TEST_CASE("a suite combines its members' predictions by each target's rule",
          "[suite][predict]") {
    TempDir dir;
    const SuiteManifest written = build_suite(dir);
    const ScoringFiles in = scoring_files(dir);

    SuitePredictor suite = SuitePredictor::load(dir.path());
    REQUIRE(suite.target_names() == std::vector<std::string>{"hab", "y", "aspect"});
    // hab trained on 20 plots, the others on all 30: two vocabularies.
    REQUIRE(suite.n_encodings() == 2);

    SuitePredictOptions opts;
    opts.keep_members = true;
    const SuitePredictions out = suite.predict(SuiteInput::csv(in.header, in.species), opts);

    // Every plot of the header is scored, in header order, though it carries
    // no target column.
    REQUIRE(out.plot_ids.size() == 31);
    REQUIRE(out.plot_ids.front() == "P0");
    REQUIRE(out.plot_ids.back() == "NEW");

    const auto& hab = out.target("hab");
    const auto& y = out.target("y");
    const auto& asp = out.target("aspect");
    REQUIRE(hab.class_names == std::vector<std::string>{"A", "B", "C"});
    REQUIRE(hab.member_seeds == std::vector<int64_t>{0, 1, 2});
    REQUIRE(y.status == SuiteTargetStatus::Experimental);
    REQUIRE(y.limit == "reliable within the synthetic corpus only");
    REQUIRE(y.units == "m");
    REQUIRE(asp.members.size(0) == 2);
    REQUIRE(asp.members.size(1) == 31);

    // Recompute every rule from the members scored one at a time.
    std::vector<torch::Tensor> hab_codes, hab_probs, y_vals, asp_bearings;
    for (const auto& t : written.targets) {
        for (const auto& m : t.members) {
            const auto p = predict_member(dir.file(m.file), in);
            const auto idx = index_of(p.plot_ids);
            std::vector<int64_t> order;
            for (const auto& id : out.plot_ids) order.push_back(idx.at(id));
            const auto perm = torch::tensor(order, torch::kLong);
            if (t.name == "hab") {
                hab_codes.push_back(p.predictions.at("hab").index_select(0, perm));
                hab_probs.push_back(p.probabilities.at("hab").index_select(0, perm));
            } else if (t.name == "y") {
                y_vals.push_back(p.predictions.at("y").index_select(0, perm));
            } else {
                asp_bearings.push_back(bearing_from_components(
                    p.predictions.at("asp_sin").index_select(0, perm),
                    p.predictions.at("asp_cos").index_select(0, perm), 360.0));
            }
        }
    }
    const auto vote = combine_vote(torch::stack(hab_codes), 3);
    REQUIRE(torch::equal(hab.value, vote.value));
    REQUIRE(torch::allclose(hab.agreement, vote.agreement));
    REQUIRE(torch::allclose(hab.probabilities, torch::stack(hab_probs).mean(0), 1e-5, 1e-6));
    REQUIRE(torch::allclose(hab.probabilities.sum(1), torch::ones({31}), 1e-5, 1e-5));
    REQUIRE(torch::equal(hab.members, torch::stack(hab_codes)));

    const auto mean = combine_mean(torch::stack(y_vals));
    REQUIRE(torch::allclose(y.value, mean.value, 1e-5, 1e-5));
    REQUIRE(torch::allclose(y.dispersion, mean.dispersion, 1e-5, 1e-5));

    const auto circ = combine_circular(torch::stack(asp_bearings), 360.0);
    REQUIRE(torch::allclose(asp.value, circ.value, 1e-4, 1e-4));
    REQUIRE(torch::allclose(asp.dispersion, circ.dispersion, 1e-4, 1e-4));
    REQUIRE((asp.value >= 0).all().item<bool>());
    REQUIRE((asp.value < 360).all().item<bool>());

    // Recognition is per target vocabulary: sp_e is unknown only to hab, and
    // sp_z to every target.
    const auto row = index_of(out.plot_ids);
    const int64_t p21 = row.at("P21");  // sp_a, sp_b, sp_c and sp_e
    const int64_t fresh = row.at("NEW");
    REQUIRE(hab.recognition.n_species[p21].item<float>() == 4.0f);
    REQUIRE(hab.recognition.n_recognised[p21].item<float>() == 3.0f);
    REQUIRE(y.recognition.n_recognised[p21].item<float>() == 4.0f);
    REQUIRE(y.recognition.count_share[p21].item<float>() == 1.0f);
    REQUIRE_THAT(y.recognition.count_share[fresh].item<double>(), WithinAbs(0.5, 1e-6));
    REQUIRE(y.recognition.abundance_share[fresh].item<float>() < 1.0f);

    SECTION("in-memory tables score the same as the CSVs") {
        auto table = [](const std::string& path) {
            CSVReader reader(path);
            std::vector<std::vector<std::string>> columns(reader.columns().size());
            reader.read_rows([&](size_t, const std::vector<std::string>& r) {
                for (size_t c = 0; c < r.size(); ++c) columns[c].push_back(r[c]);
            });
            return ColumnTable(reader.columns(), std::move(columns));
        };
        const ColumnTable header = table(in.header);
        const ColumnTable species = table(in.species);
        const auto again = suite.predict(SuiteInput::tables(&header, species), opts);
        REQUIRE(again.plot_ids == out.plot_ids);
        for (size_t t = 0; t < out.targets.size(); ++t) {
            REQUIRE(torch::allclose(again.targets[t].value.to(torch::kDouble),
                                    out.targets[t].value.to(torch::kDouble)));
        }
    }

    SECTION("renamed columns are read through SuiteColumns") {
        dir.write("renamed/header.csv", [&] {
            std::ifstream f(in.header);
            std::string text((std::istreambuf_iterator<char>(f)), {});
            text.replace(0, text.find('\n'), "PLOT,LAT,LON,COV");
            return text;
        }());
        SuitePredictOptions renamed = opts;
        renamed.columns.plot_id = "PLOT";
        renamed.columns.latitude = "LAT";
        renamed.columns.longitude = "LON";
        renamed.columns.covariates = std::vector<std::string>{"COV"};
        // The species file keeps "plot": both tables are read with one plot id
        // name, so it is renamed there too.
        dir.write("renamed/species.csv", [&] {
            std::ifstream f(in.species);
            std::string text((std::istreambuf_iterator<char>(f)), {});
            text.replace(0, 4, "PLOT");
            return text;
        }());
        const auto again = suite.predict(
            SuiteInput::csv(dir.file("renamed/header.csv"), dir.file("renamed/species.csv")),
            renamed);
        REQUIRE(torch::allclose(again.target("y").value, y.value));
        renamed.columns.covariates = std::vector<std::string>{"COV", "extra"};
        REQUIRE_THROWS_WITH(suite.predict(SuiteInput::csv(dir.file("renamed/header.csv"),
                                                          dir.file("renamed/species.csv")),
                                          renamed),
                            ContainsSubstring("reads 1 covariate"));
    }

    SECTION("a subset of targets loads only those members") {
        SuiteLoadOptions lo;
        lo.targets = {"aspect"};
        SuitePredictor one = SuitePredictor::load(dir.path(), lo);
        REQUIRE(one.target_names() == std::vector<std::string>{"aspect"});
        REQUIRE(one.n_encodings() == 1);
        const auto p = one.predict(SuiteInput::csv(in.header, in.species));
        REQUIRE(torch::allclose(p.target("aspect").value, asp.value));
        lo.targets = {"nope"};
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path(), lo),
                            ContainsSubstring("has no target 'nope'"));
    }

    SECTION("plot-level columns make the header required") {
        REQUIRE_THROWS_WITH(suite.predict(SuiteInput::csv("", in.species)),
                            ContainsSubstring("header table is required"));
    }
}

TEST_CASE("a suite refuses members that do not match its manifest", "[suite][predict]") {
    TempDir dir;
    const SuiteManifest written = build_suite(dir);

    SECTION("a changed weight file fails the checksum") {
        const std::string victim = dir.file(written.targets[1].members[0].file);
        {
            std::fstream f(victim, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(100);
            f.put('\x7f');
        }
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()), ContainsSubstring("SHA-256"));
        REQUIRE(written.verify(dir.path()).size() == 1);
    }

    SECTION("a missing member is reported") {
        fs::remove(dir.file(written.targets[2].members[1].file));
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()), ContainsSubstring("file missing"));
        SuiteLoadOptions lo;
        lo.verify_checksums = false;
        lo.targets = {"hab"};
        REQUIRE_NOTHROW(SuitePredictor::load(dir.path(), lo));
    }

    SECTION("an input contract the members were not trained on") {
        SuiteManifest m = written;
        m.inputs.covariates = {"other"};
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("does not list in that order"));
        m = written;
        m.inputs.latitude.clear();
        m.inputs.longitude.clear();
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("reads coordinates"));
    }

    SECTION("an output the checkpoint has no head for, or of the wrong task") {
        SuiteManifest m = written;
        m.targets[1].outputs = {"z"};
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("reads output 'z'"));
        m = written;
        m.targets[0].combine = SuiteCombine::Mean;
        m.targets[0].task = TaskType::Regression;
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("is a classification head"));
    }

    SECTION("members of one target on different vocabularies") {
        // A hab member trained on all 30 plots carries sp_e in its vocabulary;
        // the three written ones, trained on 20, do not.
        const auto wide = ResolveDataset::from_csv(
            dir.file("train/header.csv"), dir.file("train/species.csv"), training_roles(),
            {TargetSpec::classification("hab", 0)}, training_dataset_config());
        save_member(wide, 7, dir.file("hab/seed_7/model_final.pt"));
        SuiteManifest m = written;
        m.targets[0].members.push_back({7, "hab/seed_7/model_final.pt", "", 0});
        m.seal(dir.path());
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("encodes its input differently"));
    }
}
