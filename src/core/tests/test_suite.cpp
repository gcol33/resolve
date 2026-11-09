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
#include "resolve/name_resolution.hpp"
#include "resolve/number_text.hpp"
#include "resolve/predictor.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/sha256.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/suite.hpp"
#include "resolve/terrain_field.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <clocale>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <fstream>
#include <iterator>
#include <map>
#include <numbers>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace fs = std::filesystem;

namespace {

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
    return Roles("plot", "taxon").taxonomy().coordinates().covariates({"cov1"});
}

// Read a recorded 0 as the input contract of base_manifest() does.
DatasetConfig training_dataset_config() {
    DatasetConfig cfg;
    cfg.species_encoding = SpeciesEncodingMode::RankPool;
    cfg.selection = SelectionMode::All;
    cfg.pool_weighting = PoolWeighting::Log1p;
    cfg.use_taxonomy = true;
    cfg.zero_abundance_as = 1.0f;
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

TEST_CASE("json numbers: shortest exact spelling, null for non-finite, any locale",
          "[suite][json]") {
    CHECK(json::dump(json::Value(0.1f), 0) == "0.1");
    // The float's shortest spelling: nine digits at most, reading back exactly.
    CHECK(json::dump(json::Value(47.123456f), 0).size() <= 10);
    CHECK(static_cast<float>(json::parse(json::dump(json::Value(47.123456f), 0)).as_number()) ==
          47.123456f);
    CHECK(json::dump(json::Value(0.1), 0) == "0.1");
    CHECK(json::dump(json::Value(8192), 0) == "8192");
    CHECK(json::dump(json::Value(1e300), 0) == "1e+300");
    CHECK(json::Value(std::numeric_limits<double>::infinity()).is_null());
    CHECK(json::Value(std::numeric_limits<float>::quiet_NaN()).is_null());
    json::Value o = json::Value::object();
    o.set("loss", std::numeric_limits<float>::infinity());
    CHECK(json::dump(o, 0) == "{\"loss\":null}");
    CHECK(o.erase("loss"));
    CHECK_FALSE(o.erase("loss"));

    // A process locale with a comma decimal point changes neither direction.
    const char* german = std::setlocale(LC_NUMERIC, "de-DE");
    if (german == nullptr) german = std::setlocale(LC_NUMERIC, "de_DE.UTF-8");
    if (german == nullptr) SKIP("no locale with a comma decimal point is installed");
    CHECK(json::dump(json::Value(2.5), 0) == "2.5");
    CHECK(json::parse("2.5").as_number() == 2.5);
    float parsed = 0.0f;
    CHECK(decimal_to_number(std::string_view("0.25"), parsed));
    CHECK(parsed == 0.25f);
    std::setlocale(LC_NUMERIC, "C");
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
    REQUIRE(json::dump(json::Value(std::nan(""))) == "null");
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

TEST_CASE("a vote takes the majority class, ties to the higher mean probability",
          "[suite][combine]") {
    // Three members over four plots.
    const auto codes = torch::tensor({2, 1, 0, 3, 2, 1, 1, 3, 0, 2, 2, 3}, torch::kLong)
                           .reshape({3, 4});
    const auto c = combine_vote(codes, 4);
    REQUIRE(c.value[0].item<int64_t>() == 2);  // 2, 2, 0
    REQUIRE(c.value[1].item<int64_t>() == 1);  // 1, 1, 2
    REQUIRE(c.value[2].item<int64_t>() == 0);  // 0, 1, 2 -- three-way tie, no probabilities
    REQUIRE(c.value[3].item<int64_t>() == 3);  // unanimous
    REQUIRE_THAT(c.agreement[0].item<double>(), WithinAbs(2.0 / 3.0, 1e-6));
    REQUIRE_THAT(c.agreement[2].item<double>(), WithinAbs(1.0 / 3.0, 1e-6));
    REQUIRE_THAT(c.agreement[3].item<double>(), WithinAbs(1.0, 1e-6));
    REQUIRE_FALSE(c.dispersion.defined());
    REQUIRE_THROWS(combine_vote(codes, 3));  // code 3 outside [0, 3)

    // Plot 2's tie among classes 0, 1 and 2 goes to class 2, the most probable
    // of them; class 3 is more probable still but holds no vote. Plot 0's
    // majority stands against a class the probabilities favour.
    const auto probs = torch::tensor({0.1f, 0.1f, 0.2f, 0.6f,
                                      0.2f, 0.5f, 0.3f, 0.0f,
                                      0.2f, 0.2f, 0.3f, 0.3f,
                                      0.0f, 0.0f, 0.0f, 1.0f}).reshape({4, 4});
    const auto p = combine_vote(codes, 4, probs);
    REQUIRE(p.value[0].item<int64_t>() == 2);
    REQUIRE(p.value[1].item<int64_t>() == 1);
    REQUIRE(p.value[2].item<int64_t>() == 2);
    REQUIRE(p.value[3].item<int64_t>() == 3);
    REQUIRE(torch::equal(p.agreement, c.agreement));
    // Classes 0 and 1 tie in probability too: the lowest code wins.
    const auto even = torch::tensor({0.4f, 0.4f, 0.2f, 0.0f}).reshape({1, 4});
    const auto two = torch::tensor({int64_t{0}, int64_t{1}}, torch::kLong).reshape({2, 1});
    REQUIRE(combine_vote(two, 4, even).value[0].item<int64_t>() == 0);
    REQUIRE_THROWS(combine_vote(codes, 4, probs.narrow(0, 0, 3)));
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

TEST_CASE("a spherical mean averages points as unit vectors", "[suite][combine]") {
    // Two members either side of the antimeridian on the equator meet at 180;
    // an arithmetic mean of the longitudes would say 0. Two members on one
    // meridian, 2 degrees of latitude apart, meet halfway.
    const auto c = torch::tensor({0.0, 179.0, 10.0, 5.0,
                                  0.0, -179.0, 12.0, 5.0}, torch::kFloat64)
                       .reshape({2, 2, 2});
    const auto s = combine_spherical(c);
    REQUIRE(s.value.sizes() == torch::IntArrayRef({2, 2}));
    REQUIRE_THAT(s.value[0][0].item<double>(), WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(std::fabs(s.value[0][1].item<double>()), WithinAbs(180.0, 1e-4));
    REQUIRE_THAT(s.value[1][0].item<double>(), WithinAbs(11.0, 1e-4));
    REQUIRE_THAT(s.value[1][1].item<double>(), WithinAbs(5.0, 1e-4));
    // Each member lies one degree of arc from the combined point.
    const double one_degree_km = kEarthRadiusKm * std::numbers::pi / 180.0;
    REQUIRE_THAT(s.dispersion[0].item<double>(), WithinAbs(one_degree_km, 1e-2));
    REQUIRE_THAT(s.dispersion[1].item<double>(), WithinAbs(one_degree_km, 1e-2));
    REQUIRE_FALSE(s.agreement.defined());

    // The median distance over three members is the middle one.
    const auto three = torch::tensor({0.0, 0.0, 0.0, 1.0, 0.0, -3.0}, torch::kFloat64)
                           .reshape({3, 1, 2});
    const auto m = combine_spherical(three);
    const double centre = m.value[0][1].item<double>();
    std::vector<double> d = {std::fabs(0.0 - centre), std::fabs(1.0 - centre),
                             std::fabs(-3.0 - centre)};
    std::sort(d.begin(), d.end());
    REQUIRE_THAT(m.dispersion[0].item<double>(), WithinAbs(d[1] * one_degree_km, 1e-2));

    // One member has no spread; members that cancel name no point.
    REQUIRE(std::isnan(combine_spherical(c.slice(0, 0, 1)).dispersion[0].item<double>()));
    const auto opposite = torch::tensor({0.0, 0.0, 0.0, 180.0}, torch::kFloat64).reshape({2, 1, 2});
    const auto none = combine_spherical(opposite);
    REQUIRE(std::isnan(none.value[0][0].item<double>()));
    REQUIRE(std::isnan(none.dispersion[0].item<double>()));
    REQUIRE_THROWS(combine_spherical(torch::zeros({2, 3})));
    REQUIRE_THROWS(combine_spherical(torch::zeros({2, 3, 3})));
}

TEST_CASE("every combination rule leaves the dispersion of one member undefined",
          "[suite][combine]") {
    const auto values = torch::tensor({1.0f, 5.0f}).reshape({1, 2});
    REQUIRE(std::isnan(combine_mean(values).dispersion[0].item<double>()));
    REQUIRE(std::isnan(combine_circular(values, 360.0).dispersion[1].item<double>()));
    REQUIRE_THAT(combine_circular(values, 360.0).value[1].item<double>(), WithinAbs(5.0, 1e-4));
    const auto point = torch::tensor({10.0f, 20.0f}).reshape({1, 1, 2});
    REQUIRE(std::isnan(combine_spherical(point).dispersion[0].item<double>()));
    // A vote of one member names its class with every member behind it.
    const auto vote = combine_vote(torch::tensor(std::vector<int64_t>{2}).reshape({1, 1}), 3);
    REQUIRE(vote.value[0].item<int64_t>() == 2);
    REQUIRE(vote.agreement[0].item<float>() == 1.0f);
}

TEST_CASE("a circular mean of cancelling bearings names no bearing", "[suite][combine]") {
    // 0 and 180 degrees, and 0 / 120 / 240: the unit vectors sum to zero.
    const auto pair = torch::tensor({0.0f, 180.0f}).reshape({2, 1});
    const auto c = combine_circular(pair, 360.0);
    REQUIRE(std::isnan(c.value[0].item<double>()));
    REQUIRE(std::isnan(c.dispersion[0].item<double>()));
    const auto three = torch::tensor({0.0f, 120.0f, 240.0f}).reshape({3, 1});
    REQUIRE(std::isnan(combine_circular(three, 360.0).value[0].item<double>()));
    // Nearly opposite bearings still name a direction.
    const auto near = torch::tensor({0.0f, 179.0f}).reshape({2, 1});
    REQUIRE_THAT(combine_circular(near, 360.0).value[0].item<double>(), WithinAbs(89.5, 1e-3));
}

TEST_CASE("every combine rule has one descriptor", "[suite][combine]") {
    const auto rules = suite_combine_rules();
    REQUIRE(rules.size() == std::size(kSuiteCombineNames));
    for (const auto& entry : kSuiteCombineNames) {
        INFO("rule " << entry.name);
        const SuiteCombineRule& rule = combine_rule(entry.value);
        CHECK(rule.combine == entry.value);
        CHECK(rule.extract != nullptr);
        CHECK(rule.combine_members != nullptr);
        CHECK(rule.min_outputs <= rule.max_outputs);
        // A classification rule reports agreement, every other rule a
        // dispersion in named units.
        CHECK((rule.dispersion_column == nullptr) == (rule.task == TaskType::Classification));
        CHECK((rule.dispersion_units == DispersionUnits::None) ==
              (rule.task == TaskType::Classification));
    }
    CHECK(combine_rule(SuiteCombine::SphericalMean).n_value_columns() == 2);
    CHECK(combine_rule(SuiteCombine::CircularMean).needs_period);
    CHECK(combine_rule(SuiteCombine::SphericalMean).format_version == 2);
}

TEST_CASE("suite predictions flatten to one row per plot with every rule's columns",
          "[suite][columns]") {
    SuitePredictions p;
    p.plot_ids = {"a", "b"};
    const auto recognition = SpeciesRecognition{
        torch::tensor({3.0f, 0.0f}), torch::tensor({2.0f, 0.0f}),
        torch::tensor({2.0f / 3.0f, std::nanf("")}), torch::tensor({0.5f, std::nanf("")})};

    SuiteTargetPrediction vote;
    vote.name = "hab";
    vote.task = TaskType::Classification;
    vote.combine = SuiteCombine::Vote;
    vote.class_names = {"M"};  // code 1 has no label
    vote.member_seeds = {7, 8};
    vote.value = torch::tensor(std::vector<int64_t>{0, 1});
    vote.agreement = torch::tensor({1.0f, 0.5f});
    vote.probabilities = torch::tensor({0.9f, 0.1f, 0.4f, 0.6f}).reshape({2, 2});
    vote.members = torch::tensor(std::vector<int64_t>{0, 1, 0, 0}).reshape({2, 2});
    vote.recognition = recognition;
    p.targets.push_back(vote);

    SuiteTargetPrediction where;
    where.name = "where";
    where.combine = SuiteCombine::SphericalMean;
    where.member_seeds = {3};
    where.value = torch::tensor({47.123456f, 9.5f, -33.25f, 151.0f}).reshape({2, 2});
    where.dispersion = torch::tensor({std::nanf(""), std::nanf("")});
    where.members = where.value.unsqueeze(0);
    where.recognition = recognition;
    p.targets.push_back(where);

    SuiteTargetPrediction y;
    y.name = "y";
    y.combine = SuiteCombine::Mean;
    y.value = torch::tensor({1.5f, 2.5f});
    y.recognition = recognition;
    p.targets.push_back(y);

    const auto columns = suite_prediction_columns(p, /*probabilities=*/true);
    std::vector<std::string> names;
    for (const auto& c : columns) names.push_back(c.name);
    const std::vector<std::string> expected = {
        "plot_id", "hab", "hab_code", "hab_agreement", "hab_prob_M", "hab_prob_1",
        "hab_n_species", "hab_n_recognised", "hab_recognised_share",
        "hab_recognised_abundance_share", "hab_seed7", "hab_seed8",
        "where_latitude", "where_longitude", "where_spread_km", "where_n_species",
        "where_n_recognised", "where_recognised_share", "where_recognised_abundance_share",
        "where_seed3_latitude", "where_seed3_longitude",
        "y", "y_sd", "y_n_species", "y_n_recognised", "y_recognised_share",
        "y_recognised_abundance_share"};
    REQUIRE(names == expected);

    auto column = [&](const std::string& name) -> const SuiteColumn& {
        for (const auto& c : columns) {
            if (c.name == name) return c;
        }
        FAIL("no column " << name);
        return columns.front();
    };
    // A class code without a label is written as the code.
    CHECK(column("hab").text == std::vector<std::string>{"M", "1"});
    CHECK(column("hab_code").integers == std::vector<int64_t>{0, 1});
    CHECK(column("hab_seed7").text == std::vector<std::string>{"M", "1"});
    CHECK(column("hab_prob_1").probability);
    CHECK_FALSE(column("hab_agreement").probability);
    CHECK(column("where_latitude").numbers[0] == 47.123456f);
    CHECK(column("where_seed3_longitude").numbers[1] == 151.0f);
    CHECK(std::isnan(column("where_spread_km").numbers[0]));
    // An undefined statistic is NaN in every row.
    CHECK(std::isnan(column("y_sd").numbers[1]));
    CHECK(std::isnan(column("y_recognised_share").numbers[1]));

    const auto without = suite_prediction_columns(p, /*probabilities=*/false);
    CHECK(without.size() == columns.size() - 2);
}

TEST_CASE("a member may state the zero abundance it was trained under; that is format 3",
          "[suite][manifest]") {
    SuiteManifest m = base_manifest();
    SuiteTarget y;
    y.name = "y";
    y.outputs = {"y"};
    y.members.push_back({0, "y/m.pt", std::string(64, 'a'), 10});
    m.targets.push_back(y);
    REQUIRE(m.format_version() == 1);

    m.targets[0].members[0].zero_abundance_as = 1.0f;
    REQUIRE_NOTHROW(m.validate());
    REQUIRE(m.format_version() == 3);
    const SuiteManifest back = SuiteManifest::from_json(json::parse(json::dump(m.to_json())));
    REQUIRE(back.targets[0].members[0].zero_abundance_as == std::optional<float>(1.0f));
    REQUIRE(back.to_json() == m.to_json());

    json::Value stated = m.to_json();
    stated.set("format_version", 2);
    REQUIRE_THROWS_WITH(SuiteManifest::from_json(stated), ContainsSubstring("need version 3"));

    m.targets[0].members[0].zero_abundance_as = -1.0f;
    REQUIRE_THROWS_WITH(m.validate(), ContainsSubstring("zero_abundance_as"));
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

TEST_CASE("per-target inputs and a spherical mean are format 2; a manifest without is 1",
          "[suite][manifest]") {
    SuiteManifest m = base_manifest();
    SuiteTarget y;
    y.name = "y";
    y.outputs = {"y"};
    y.members.push_back({0, "y/m.pt", std::string(64, 'a'), 10});
    m.targets.push_back(y);
    REQUIRE(m.format_version() == 1);
    REQUIRE(m.to_json().at("format_version").as_int() == 1);
    REQUIRE(m.to_json().at("targets").items()[0].find("inputs") == nullptr);
    REQUIRE(m.inputs_of(m.targets[0]) ==
            SuiteTargetInputs{true, {"cov1"}, {}});

    SuiteTarget where;
    where.name = "where";
    where.combine = SuiteCombine::SphericalMean;
    where.outputs = {"lat", "lon"};
    where.units = "degrees";
    where.inputs = SuiteTargetInputs{false, {"cov1"}, {}};
    where.members.push_back({0, "where/m.pt", std::string(64, 'b'), 10});
    m.targets[0].inputs = SuiteTargetInputs{true, {}, {}};
    m.targets.push_back(where);
    REQUIRE_NOTHROW(m.validate());
    REQUIRE(m.format_version() == 2);

    const SuiteManifest back = SuiteManifest::from_json(json::parse(json::dump(m.to_json())));
    REQUIRE(back.to_json() == m.to_json());
    REQUIRE(back.targets[0].inputs == SuiteTargetInputs{true, {}, {}});
    REQUIRE(back.targets[1].inputs == SuiteTargetInputs{false, {"cov1"}, {}});
    REQUIRE(back.targets[1].combine == SuiteCombine::SphericalMean);

    // A manifest that states version 1 and uses a version-2 feature is refused.
    json::Value stated = m.to_json();
    stated.set("format_version", 1);
    REQUIRE_THROWS_WITH(SuiteManifest::from_json(stated), ContainsSubstring("need version 2"));
}

TEST_CASE("per-target inputs are checked against the contract and cover it",
          "[suite][manifest]") {
    SuiteManifest m = base_manifest();
    m.inputs.categoricals = {"hab"};
    SuiteTarget a;
    a.name = "a";
    a.outputs = {"a"};
    a.inputs = SuiteTargetInputs{true, {"cov1", "nope", "cov1"}, {}};
    a.members.push_back({0, "a/m.pt", std::string(64, 'a'), 10});
    SuiteTarget where;
    where.name = "where";
    where.combine = SuiteCombine::SphericalMean;
    where.outputs = {"lat"};
    where.units = "m";
    where.inputs = SuiteTargetInputs{true, {}, {}};
    where.members.push_back({0, "w/m.pt", std::string(64, 'b'), 10});
    m.targets = {a, where};
    try {
        m.validate();
        FAIL("validate() accepted an invalid manifest");
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        for (const char* expected :
             {"'nope' is not a column of the input contract", "'cov1' appears twice",
              "'hab' is read by no target", "reads two outputs", "units must be 'degrees'"}) {
            INFO(expected);
            REQUIRE_THAT(msg, ContainsSubstring(expected));
        }
    }

    SuiteManifest bare = base_manifest();
    bare.inputs.latitude.clear();
    bare.inputs.longitude.clear();
    a.inputs = SuiteTargetInputs{true, {"cov1"}, {}};
    bare.targets = {a};
    REQUIRE_THROWS_WITH(bare.validate(), ContainsSubstring("the input contract names none"));

    SuiteManifest unread = base_manifest();
    a.inputs = SuiteTargetInputs{false, {"cov1"}, {}};
    unread.targets = {a};
    REQUIRE_THROWS_WITH(unread.validate(), ContainsSubstring("names coordinates, but no target"));
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
    const auto vocab = SpeciesVocab::from_ordered({"<UNK>", "sp_a", "sp_b"});
    const std::vector<SpeciesRecord> records = {
        {"sp_a", "", "", 2.0f, "p1"}, {"sp_a", "", "", 1.0f, "p1"},  // two layers, one species
        {"sp_x", "", "", 1.0f, "p1"},
        {"sp_b", "", "", 4.0f, "p2"},
        {"sp_y", "", "", 0.0f, "p4"},
    };
    const auto r = compute_species_recognition(
        PlotAssemblages::build(records, {"p1", "p2", "p3", "p4"}), vocab);
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
    const auto u = compute_unknown_species_stats(PlotAssemblages::build(records, {"p1", "p2"}),
                                                 vocab);
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
    REQUIRE(zero.schema().dataset_config.zero_abundance_as == 1.0f);

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
    REQUIRE(p.schema().dataset_config.zero_abundance_as == 1.0f);
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
    const auto vote = combine_vote(torch::stack(hab_codes), 3, torch::stack(hab_probs).mean(0));
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
            reader.read_rows([&](size_t, RowFields r) {
                for (size_t c = 0; c < r.size(); ++c) columns[c].emplace_back(r[c]);
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

    SECTION("verify checks the members of the targets it is given") {
        fs::remove(dir.file(written.targets[2].members[1].file));
        REQUIRE(written.verify(dir.path()).size() == 1);
        REQUIRE(written.verify(dir.path(), {"hab", "y"}).empty());
        REQUIRE(written.verify(dir.path(), {"aspect"}).size() == 1);
    }

    SECTION("a member trained reading a zero abundance as the contract does not") {
        SuiteManifest m = written;
        m.inputs.zero_abundance_as = 0.5f;
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("reading a zero abundance as 1"));
        // A statement in the manifest that the checkpoint contradicts.
        m = written;
        m.targets[1].members[0].zero_abundance_as = 0.0f;
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("the checkpoint records 1"));
        m.targets[1].members[0].zero_abundance_as = 1.0f;
        m.write(dir.path());
        REQUIRE_NOTHROW(SuitePredictor::load(dir.path()));
    }

    SECTION("a contract that names fewer taxonomy levels or no abundance") {
        SuiteManifest m = written;
        m.inputs.family.clear();
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("reads family, the input contract names no family"));
        m = written;
        m.inputs.genus.clear();
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("reads genus, the input contract names no genus"));
        m = written;
        m.inputs.abundance.clear();
        m.inputs.zero_abundance_as = 0.0f;
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("no abundance column"));
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

TEST_CASE("declaring the whole contract per target predicts bit for bit as declaring none",
          "[suite][predict]") {
    TempDir dir;
    const SuiteManifest written = build_suite(dir);
    const ScoringFiles in = scoring_files(dir);
    SuitePredictOptions opts;
    opts.keep_members = true;
    SuitePredictor implicit = SuitePredictor::load(dir.path());
    const SuitePredictions a = implicit.predict(SuiteInput::csv(in.header, in.species), opts);

    SuiteManifest m = written;
    REQUIRE(m.format_version() == 1);
    for (auto& t : m.targets) t.inputs = m.inputs_of(t);
    m.write(dir.path());
    REQUIRE(SuiteManifest::read(dir.path()).format_version() == 2);
    SuitePredictor declared = SuitePredictor::load(dir.path());
    REQUIRE(declared.n_encodings() == implicit.n_encodings());
    const SuitePredictions b = declared.predict(SuiteInput::csv(in.header, in.species), opts);

    REQUIRE(a.plot_ids == b.plot_ids);
    REQUIRE(a.targets.size() == b.targets.size());
    auto same = [](const torch::Tensor& x, const torch::Tensor& y) {
        if (x.defined() != y.defined()) return false;
        return !x.defined() || (x.dtype() == y.dtype() && torch::equal(x, y));
    };
    for (size_t t = 0; t < a.targets.size(); ++t) {
        INFO(a.targets[t].name);
        const auto& x = a.targets[t];
        const auto& y = b.targets[t];
        REQUIRE(same(x.value, y.value));
        REQUIRE(same(x.agreement, y.agreement));
        REQUIRE(same(x.probabilities, y.probabilities));
        REQUIRE(same(x.dispersion, y.dispersion));
        REQUIRE(same(x.members, y.members));
        REQUIRE(same(x.recognition.count_share, y.recognition.count_share));
        REQUIRE(x.dispersion_units == y.dispersion_units);
    }
    REQUIRE(a.target("y").dispersion_units == "m");
    REQUIRE(a.target("aspect").dispersion_units == "degrees");
    REQUIRE(a.target("hab").dispersion_units.empty());
}

namespace {

// A suite whose two targets read different parts of one contract, as a model
// of the survey year cannot read the year and a model of the location cannot
// read the coordinates: "y" reads species and coordinates, "where" reads
// species and cov1 and predicts the plot's latitude / longitude.
RoleMapping roles_without_covariates() {
    RoleMapping roles = training_roles();
    roles.covariates.clear();
    return roles;
}

RoleMapping roles_without_coordinates() {
    RoleMapping roles = training_roles();
    roles.latitude.reset();
    roles.longitude.reset();
    return roles;
}

SuiteManifest build_split_suite(const TempDir& dir) {
    const auto plots = corpus();
    dir.write("train/header.csv", header_csv(plots, true));
    dir.write("train/species.csv", species_csv(plots));
    const auto cfg = training_dataset_config();
    const auto y_ds = ResolveDataset::from_csv(
        dir.file("train/header.csv"), dir.file("train/species.csv"), roles_without_covariates(),
        {TargetSpec::regression("y")}, cfg);
    const auto where_ds = ResolveDataset::from_csv(
        dir.file("train/header.csv"), dir.file("train/species.csv"), roles_without_coordinates(),
        {TargetSpec::regression("lat"), TargetSpec::regression("lon")}, cfg);

    SuiteManifest m = base_manifest();
    SuiteTarget y;
    y.name = "y";
    y.outputs = {"y"};
    y.units = "m";
    y.inputs = SuiteTargetInputs{true, {}, {}};
    SuiteTarget where;
    where.name = "where";
    where.combine = SuiteCombine::SphericalMean;
    where.outputs = {"lat", "lon"};
    where.units = "degrees";
    where.inputs = SuiteTargetInputs{false, {"cov1"}, {}};
    for (const int seed : {0, 1}) {
        const std::string yf = "y/seed_" + std::to_string(seed) + "/model_final.pt";
        save_member(y_ds, seed, dir.file(yf));
        y.members.push_back(SuiteMember{seed, yf, "", 0});
    }
    for (const int seed : {2, 3, 4}) {
        const std::string wf = "where/seed_" + std::to_string(seed) + "/model_final.pt";
        save_member(where_ds, seed, dir.file(wf));
        where.members.push_back(SuiteMember{seed, wf, "", 0});
    }
    m.targets = {y, where};
    m.seal(dir.path());
    m.write(dir.path());
    return m;
}

ResolvePredictions predict_member_with(const std::string& path, const ScoringFiles& in,
                                       const RoleMapping& roles) {
    Predictor p = Predictor::load(path);
    auto cfg = dataset_config_from_checkpoint(p.schema(), p.model()->config());
    cfg.zero_abundance_as = 1.0f;
    const auto ds = ResolveDataset::from_csv_with_vocabs(in.header, in.species, roles, {},
                                                         p.external_vocabs(), cfg);
    return p.predict(ds);
}

}  // namespace

TEST_CASE("each target reads its own part of the contract", "[suite][predict]") {
    TempDir dir;
    const SuiteManifest written = build_split_suite(dir);
    REQUIRE(written.format_version() == 2);
    const ScoringFiles in = scoring_files(dir);

    SuitePredictor suite = SuitePredictor::load(dir.path());
    // One vocabulary, two sets of columns read: two encodings.
    REQUIRE(suite.n_encodings() == 2);
    SuitePredictOptions opts;
    opts.keep_members = true;
    const SuitePredictions out = suite.predict(SuiteInput::csv(in.header, in.species), opts);
    REQUIRE(out.plot_ids.size() == 31);

    std::vector<torch::Tensor> y_vals, where_points;
    for (const auto& t : written.targets) {
        const RoleMapping roles = t.name == "y" ? roles_without_covariates()
                                                : roles_without_coordinates();
        for (const auto& member : t.members) {
            const auto p = predict_member_with(dir.file(member.file), in, roles);
            const auto idx = index_of(p.plot_ids);
            std::vector<int64_t> order;
            for (const auto& id : out.plot_ids) order.push_back(idx.at(id));
            const auto perm = torch::tensor(order, torch::kLong);
            if (t.name == "y") {
                y_vals.push_back(p.predictions.at("y").index_select(0, perm));
            } else {
                where_points.push_back(torch::stack({p.predictions.at("lat").index_select(0, perm),
                                                     p.predictions.at("lon").index_select(0, perm)},
                                                    1));
            }
        }
    }
    const auto& y = out.target("y");
    const auto mean = combine_mean(torch::stack(y_vals));
    REQUIRE(torch::allclose(y.value, mean.value, 1e-5, 1e-5));
    REQUIRE(torch::allclose(y.dispersion, mean.dispersion, 1e-5, 1e-5));

    const auto& where = out.target("where");
    REQUIRE(where.combine == SuiteCombine::SphericalMean);
    REQUIRE(where.dispersion_units == "km");
    REQUIRE(where.value.sizes() == torch::IntArrayRef({31, 2}));
    REQUIRE(where.members.sizes() == torch::IntArrayRef({3, 31, 2}));
    const auto sphere = combine_spherical(torch::stack(where_points));
    REQUIRE(torch::allclose(where.value, sphere.value, 1e-4, 1e-4));
    REQUIRE(torch::allclose(where.dispersion, sphere.dispersion, 1e-3, 1e-3));
    REQUIRE(torch::allclose(where.members, torch::stack(where_points), 1e-5, 1e-5));

    SECTION("a renamed covariate reaches the target that reads it") {
        dir.write("renamed/header.csv", [&] {
            std::ifstream f(in.header);
            std::string text((std::istreambuf_iterator<char>(f)), {});
            text.replace(0, text.find('\n'), "plot,lat,lon,COV");
            return text;
        }());
        SuitePredictOptions renamed = opts;
        renamed.columns.covariates = std::vector<std::string>{"COV"};
        const auto again = suite.predict(
            SuiteInput::csv(dir.file("renamed/header.csv"), in.species), renamed);
        REQUIRE(torch::equal(again.target("where").value, where.value));
        REQUIRE(torch::equal(again.target("y").value, y.value));
    }

    SECTION("declared inputs the checkpoint was not trained on are refused") {
        SuiteManifest m = written;
        m.targets[0].inputs = SuiteTargetInputs{true, {"cov1"}, {}};
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("the target's declared inputs"));
        m = written;
        m.targets[1].inputs = SuiteTargetInputs{true, {"cov1"}, {}};
        m.write(dir.path());
        REQUIRE_THROWS_WITH(SuitePredictor::load(dir.path()),
                            ContainsSubstring("names coordinates, the checkpoint reads none"));
    }
}

TEST_CASE("suite gap fitting fits the named targets from the dataset's complete plots",
          "[suite][predict][gap_fitting]") {
    TempDir dir;
    build_suite(dir);
    const ScoringFiles in = scoring_files(dir);
    // The same plots with recorded values: y on every plot but every fourth,
    // the bearing on every plot but every fifth.
    const auto plots = corpus();
    std::ostringstream header;
    header << "plot,lat,lon,cov1,y_rec,asp_rec,year\n";
    for (size_t i = 0; i < plots.size(); ++i) {
        const auto& p = plots[i];
        header << p.id << "," << p.lat << "," << p.lon << "," << p.cov << ","
               << (i % 4 == 0 ? std::string("NA") : std::to_string(p.y)) << ","
               << (i % 5 == 0 ? std::string("") : std::to_string(p.bearing)) << ","
               << (1980 + i) << "\n";
    }
    dir.write("score/header_rec.csv", header.str());
    dir.write("score/species_rec.csv", species_csv(plots));
    const SuiteInput input =
        SuiteInput::csv(dir.file("score/header_rec.csv"), dir.file("score/species_rec.csv"));

    SuitePredictor suite = SuitePredictor::load(dir.path());
    SuitePredictOptions opts;
    opts.gap_fitting = GapFittingConfig{};
    opts.columns.recorded = {{"y", {"y_rec"}}, {"aspect", {"asp_rec"}}};
    opts.columns.survey_year = "year";
    const SuitePredictions out = suite.predict(input, opts);

    const auto& y = out.target("y");
    const auto& asp = out.target("aspect");
    REQUIRE(y.gap_fit.has_value());
    REQUIRE(asp.gap_fit.has_value());
    CHECK_FALSE(out.target("hab").gap_fit.has_value());
    CHECK(y.gap_fit->report.n_gaps == 8);
    CHECK(asp.gap_fit->report.n_gaps == 6);
    CHECK(asp.gap_fit->report.scale == GapScale::Circular);

    // The suite's fit is the free function's over the combined prediction.
    GapColumns columns;
    columns.plot_id = "plot";
    columns.species_id = "taxon";
    columns.latitude = "lat";
    columns.longitude = "lon";
    columns.survey_year = "year";
    columns.recorded = {{"y", {"y_rec"}}};
    const GapTables tables = read_gap_tables(TableSource::csv(dir.file("score/header_rec.csv")),
                                             TableSource::csv(dir.file("score/species_rec.csv")),
                                             columns, out.plot_ids);
    const GapFit direct = fit_gaps({"y"}, y.value, tables.recorded.at("y"), tables.plots);
    CHECK(torch::allclose(direct.fitted, y.gap_fit->fitted, 0.0, 1e-12, /*equal_nan=*/true));
    CHECK(torch::equal(direct.method, y.gap_fit->method));
    // Complete plots keep their record; the raw prediction is untouched.
    CHECK_THAT(y.gap_fit->fitted[1].item<double>(), WithinAbs(plots[1].y, 1e-4));
    CHECK(torch::equal(y.value, suite.predict(input, SuitePredictOptions{}).target("y").value));

    const auto table = suite_prediction_columns(out, false);
    auto column = [&](const std::string& name) -> const SuiteColumn& {
        for (const auto& c : table) {
            if (c.name == name) return c;
        }
        FAIL("no column " << name);
        return table.front();
    };
    CHECK(column("y_fitted").numbers.size() == 30);
    CHECK(column("y_fit_method").text[1] == "recorded");
    CHECK(column("y_fit_method").text[0] != "recorded");
    CHECK(column("aspect_fit_method").text[0] != "recorded");
    // The fit method already says which plots record the target.
    for (const auto& c : table) CHECK(c.name.find("_recorded") == std::string::npos);
    CHECK(torch::equal(y.recorded, y.gap_fit->method == kGapRecorded));

    // Gap fitting reads regression targets only, and needs the recorded column.
    SuitePredictOptions vote = opts;
    vote.columns.recorded = {{"hab", {"y_rec"}}};
    CHECK_THROWS_AS(suite.predict(input, vote), std::invalid_argument);
    SuitePredictOptions none = opts;
    none.columns.recorded.clear();
    CHECK_THROWS_AS(suite.predict(input, none), std::invalid_argument);
}

TEST_CASE("suite recorded columns without gap fitting flag the plots that record each target",
          "[suite][predict][gap_fitting]") {
    TempDir dir;
    build_suite(dir);
    // y recorded on every plot but every fourth, the bearing on every plot but
    // every fifth.
    const auto plots = corpus();
    std::ostringstream header;
    header << "plot,lat,lon,cov1,y_rec,asp_rec\n";
    for (size_t i = 0; i < plots.size(); ++i) {
        const auto& p = plots[i];
        header << p.id << "," << p.lat << "," << p.lon << "," << p.cov << ","
               << (i % 4 == 0 ? std::string("NA") : std::to_string(p.y)) << ","
               << (i % 5 == 0 ? std::string("") : std::to_string(p.bearing)) << "\n";
    }
    dir.write("score/header_rec.csv", header.str());
    dir.write("score/species_rec.csv", species_csv(plots));
    const SuiteInput input =
        SuiteInput::csv(dir.file("score/header_rec.csv"), dir.file("score/species_rec.csv"));

    SuitePredictor suite = SuitePredictor::load(dir.path());
    SuitePredictOptions opts;
    opts.columns.recorded = {{"y", {"y_rec"}}, {"aspect", {"asp_rec"}}};
    const SuitePredictions out = suite.predict(input, opts);

    const auto& y = out.target("y");
    const auto& asp = out.target("aspect");
    CHECK_FALSE(y.gap_fit.has_value());
    CHECK_FALSE(out.target("hab").recorded.defined());
    REQUIRE(y.recorded.defined());
    REQUIRE(asp.recorded.defined());
    REQUIRE(y.recorded.size(0) == static_cast<int64_t>(plots.size()));
    // Plots keep the header's order here, so plot i is row i.
    REQUIRE(out.plot_ids[1] == plots[1].id);
    for (int64_t i = 0; i < y.recorded.size(0); ++i) {
        INFO(i);
        CHECK(y.recorded[i].item<bool>() == (i % 4 != 0));
        CHECK(asp.recorded[i].item<bool>() == (i % 5 != 0));
    }
    // The prediction itself is the one without recorded columns.
    CHECK(torch::equal(y.value, suite.predict(input, SuitePredictOptions{}).target("y").value));

    const auto table = suite_prediction_columns(out, false);
    auto column = [&](const std::string& name) -> const SuiteColumn& {
        for (const auto& c : table) {
            if (c.name == name) return c;
        }
        FAIL("no column " << name);
        return table.front();
    };
    const SuiteColumn& flag = column("y_recorded");
    CHECK(flag.kind == SuiteColumn::Kind::Logical);
    REQUIRE(flag.logical.size() == plots.size());
    CHECK(flag.logical[0] == 0);
    CHECK(flag.logical[1] == 1);
    CHECK(column("aspect_recorded").logical[5] == 0);
    // Right after the target's dispersion, before the recognition columns.
    auto position = [&](const std::string& name) {
        return std::find_if(table.begin(), table.end(), [&](const SuiteColumn& c) { return c.name == name; }) -
               table.begin();
    };
    CHECK(position("y_recorded") == position("y_sd") + 1);
    CHECK(position("y_n_species") == position("y_recorded") + 1);

    // A classification, a column count that does not fit the target, an
    // unknown target and a target named twice are refused.
    SuitePredictOptions bad = opts;
    bad.columns.recorded = {{"hab", {"y_rec"}}};
    CHECK_THROWS_WITH(suite.predict(input, bad), ContainsSubstring("is a classification"));
    bad.columns.recorded = {{"y", {"y_rec", "asp_rec"}}};
    CHECK_THROWS_WITH(suite.predict(input, bad), ContainsSubstring("needs one recorded column"));
    bad.columns.recorded = {{"nope", {"y_rec"}}};
    CHECK_THROWS_WITH(suite.predict(input, bad), ContainsSubstring("no loaded target 'nope'"));
    bad.columns.recorded = {{"y", {"y_rec"}}, {"y", {"y_rec"}}};
    CHECK_THROWS_WITH(suite.predict(input, bad), ContainsSubstring("named twice"));
    bad.columns.recorded = {{"y", {"no_such_column"}}};
    CHECK_THROWS(suite.predict(input, bad));
}

TEST_CASE("suite names resolve per target vocabulary and predict as the tokens would",
          "[suite][predict][names]") {
    TempDir dir;
    build_suite(dir);
    const ScoringFiles in = scoring_files(dir);

    // The scoring species written with an entered name for sp_a and a
    // misspelling of sp_e, which only y's and aspect's vocabularies hold.
    std::ifstream file(in.species, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    auto replace_all = [&text](const std::string& from, const std::string& to) {
        for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
            text.replace(at, from.size(), to);
        }
    };
    replace_all(",sp_a,", ",Species a entered,");
    replace_all(",sp_e,", ",sp_ex,");
    dir.write("score/species_written.csv", text);
    const std::string written = dir.file("score/species_written.csv");

    SuitePredictor suite = SuitePredictor::load(dir.path());
    const SuitePredictions expected = suite.predict(SuiteInput::csv(in.header, in.species));
    SuitePredictOptions opts;
    opts.names = SpeciesNames{std::make_shared<const NameTable>(NameTable::from_columns(
                                  {"Species a entered"}, {"sp_a"}, {"eva"})),
                              NameResolverOptions{}};
    const SuitePredictions resolved = suite.predict(SuiteInput::csv(in.header, written), opts);
    const SuitePredictions plain = suite.predict(SuiteInput::csv(in.header, written));

    REQUIRE(resolved.plot_ids == expected.plot_ids);
    for (const std::string name : {"hab", "y", "aspect"}) {
        INFO(name);
        CHECK(torch::equal(resolved.target(name).value, expected.target(name).value));
        CHECK(torch::equal(resolved.target(name).recognition.n_recognised,
                           expected.target(name).recognition.n_recognised));
        CHECK(plain.target(name).species_names.size() == 0);
        CHECK(plain.target(name).recognition.n_recognised.sum().item<float>() <
              expected.target(name).recognition.n_recognised.sum().item<float>());
    }

    const SpeciesNameReport& hab = resolved.target("hab").species_names;
    // sp_z, one edit from every token, stays as written everywhere.
    REQUIRE(hab.size() == 3);
    CHECK(hab.name == std::vector<std::string>{"Species a entered", "sp_ex", "sp_z"});
    CHECK(hab.token == std::vector<std::string>{"sp_a", "sp_ex", "sp_z"});
    CHECK(hab.route[0] == NameRoute::Alias);
    CHECK(hab.route[1] == NameRoute::Unresolved);
    CHECK(hab.route[2] == NameRoute::Unresolved);
    CHECK(hab.n_records == std::vector<int64_t>{31, 10, 1});

    const SpeciesNameReport& y = resolved.target("y").species_names;
    REQUIRE(y.size() == 3);
    CHECK(y.token == std::vector<std::string>{"sp_a", "sp_e", "sp_z"});
    CHECK(y.route[1] == NameRoute::Spelling);
    CHECK(y.via[1] == "sp_e");
    CHECK(y.distance[1] == 1);
}

// =============================================================================
// Terrain correction
// =============================================================================

namespace {

// The plane-terrain degree squares the corpus' coordinates (45.0..47.9 N,
// 8.55..10.0 E) and their 1 km windows fall in; 44 N holds the cells on
// 45 N, a tile's southern edge.
const std::vector<std::pair<int, int>> kCorpusSquares = {{44, 8}, {44, 9}, {44, 10}, {45, 8}, {45, 9}, {45, 10}, {46, 8}, {46, 9},
                                                          {46, 10}, {47, 8}, {47, 9}, {47, 10}};

// The three-target suite with a terrain correction on "y", fitted on
// synthetic plots, and a terrain field of the plane terrain, written under
// `dir`; the plane grids sit in `<dir>/grids`.
SuiteManifest build_terrain_suite(const TempDir& dir, const TempDir& grids_dir) {
    SuiteManifest m = build_suite(dir);
    write_plane_terrain(grids_dir, kCorpusSquares, 24);

    auto gen = at::make_generator<at::CPUGeneratorImpl>(41);
    const int64_t n = 400;
    TerrainCorrectionInputs in;
    const torch::Tensor truth = 10.0 * torch::rand({n}, gen, torch::kFloat64);
    in.prediction = truth + torch::randn({n}, gen, torch::kFloat64);
    in.dem = truth + 0.1 * torch::randn({n}, gen, torch::kFloat64);
    in.dem_1km = truth + 0.5 * torch::randn({n}, gen, torch::kFloat64);
    in.spread = torch::rand({n}, gen, torch::kFloat64);
    TerrainCorrectionConfig config;
    config.scale = 10.0f;
    config.epochs = 3;
    config.batch_size = 64;
    config.hidden_dim = 16;
    fs::create_directories(dir.file("y"));
    fit_terrain_correction("y", TerrainDescriptor::Elevation, in, truth, config).save(dir.file("y/terrain.pack"));

    TerrainFieldConfig field;
    field.levels = 4;
    field.log2_table_size = 8;
    field.coarsest_resolution = 4;
    field.hidden_dim = 8;
    field.base_cells = 4;
    field.steps = 30;
    field.batch_size = 256;
    const TerrainGrids grids(grids_dir.path());
    std::vector<std::string> tiles;
    fs::create_directories(dir.file("terrain"));
    for (int64_t t = 0; t < grids.index().size(); ++t) {
        const TerrainTile& tile = grids.index().tiles()[static_cast<size_t>(t)];
        tiles.push_back(dir.file("terrain/" + tile.name + ".tile"));
        fit_terrain_tile(grids.grid(t), tile, field, tiles.back());
    }
    assemble_terrain_field(tiles, dir.file("terrain/field.bin"));
    for (const auto& t : tiles) fs::remove(t);

    for (auto& t : m.targets) {
        if (t.name == "y") t.terrain_correction = SuiteFile{"y/terrain.pack", "", 0};
    }
    m.terrain_field = SuiteFile{"terrain/field.bin", "", 0};
    m.seal(dir.path());
    m.write(dir.path());
    return m;
}

}  // namespace

TEST_CASE("a terrain field and terrain correction are format 4, round-trip and are checked like members",
          "[suite][manifest][terrain]") {
    TempDir dir, grids;
    const SuiteManifest m = build_terrain_suite(dir, grids);
    CHECK(m.format_version() == 4);
    const SuiteManifest back = SuiteManifest::from_json(m.to_json());
    REQUIRE(back.terrain_field.has_value());
    CHECK(back.terrain_field->sha256 == m.terrain_field->sha256);
    CHECK(back.target("y").terrain_correction->file == "y/terrain.pack");
    CHECK_FALSE(back.target("hab").terrain_correction.has_value());
    CHECK(m.verify(dir.path()).empty());

    // A changed correction file fails the checksum, like a member.
    {
        std::ofstream out(dir.file("y/terrain.pack"), std::ios::binary | std::ios::app);
        out << "x";
    }
    const auto problems = m.verify(dir.path(), {"y"});
    REQUIRE(problems.size() == 1);
    CHECK_THAT(problems[0], Catch::Matchers::ContainsSubstring("terrain correction"));

    SuiteManifest vote = m;
    for (auto& t : vote.targets) {
        if (t.name == "hab") t.terrain_correction = m.target("y").terrain_correction;
    }
    CHECK_THROWS_WITH(vote.validate(), Catch::Matchers::ContainsSubstring("mean-combined target"));
}

TEST_CASE("suite terrain correction reads the descriptors from columns, grids or the shipped field",
          "[suite][predict][terrain]") {
    TempDir dir, grids;
    build_terrain_suite(dir, grids);
    const auto plots = corpus();
    std::ostringstream header;
    header << "plot,lat,lon,cov1,elev,elev1km,unc\n";
    for (size_t i = 0; i < plots.size(); ++i) {
        const auto& p = plots[i];
        header << p.id << "," << p.lat << "," << p.lon << "," << p.cov << "," << terrain_plane(p.lat, p.lon)
               << "," << (i % 3 == 0 ? std::string("NA") : std::to_string(terrain_plane(p.lat, p.lon) + 2.0))
               << "," << (i % 2 == 0 ? std::string("") : std::to_string(10.0 * (i + 1))) << "\n";
    }
    dir.write("score/header_terrain.csv", header.str());
    dir.write("score/species_terrain.csv", species_csv(plots));
    const SuiteInput input =
        SuiteInput::csv(dir.file("score/header_terrain.csv"), dir.file("score/species_terrain.csv"));
    SuitePredictor suite = SuitePredictor::load(dir.path());

    SuitePredictOptions columns;
    columns.terrain = SuiteTerrainOptions{TerrainSourceKind::Columns, "", torch::kCPU};
    columns.columns.terrain.elevation = "elev";
    columns.columns.terrain.elevation_1km = "elev1km";
    columns.columns.terrain.uncertainty = "unc";
    const SuitePredictions out = suite.predict(input, columns);
    const auto& y = out.target("y");
    REQUIRE(y.terrain.has_value());
    CHECK_FALSE(out.target("hab").terrain.has_value());
    CHECK(y.terrain->source == TerrainSourceKind::Columns);
    CHECK(y.terrain->result.applied.all().item<bool>());
    // The suite's correction is the stored correction applied to the combined
    // prediction with the columns' values.
    const std::map<std::string, int64_t> row = index_of(out.plot_ids);
    std::vector<double> elev(out.plot_ids.size()), elev1(out.plot_ids.size()), unc(out.plot_ids.size());
    for (size_t i = 0; i < plots.size(); ++i) {
        const auto r = static_cast<size_t>(row.at(plots[i].id));
        elev[r] = terrain_plane(plots[i].lat, plots[i].lon);
        elev1[r] = i % 3 == 0 ? std::nan("") : elev[r] + 2.0;
        unc[r] = i % 2 == 0 ? std::nan("") : 10.0 * static_cast<double>(i + 1);
    }
    TerrainCorrectionInputs direct;
    direct.prediction = y.value.to(torch::kFloat64);
    direct.dem = torch::tensor(elev, torch::kFloat64);
    direct.dem_1km = torch::tensor(elev1, torch::kFloat64);
    direct.uncertainty_m = torch::tensor(unc, torch::kFloat64);
    direct.spread = y.dispersion.to(torch::kFloat64);
    direct.recognised_share = y.recognition.count_share.to(torch::kFloat64);
    direct.n_species = y.recognition.n_species.to(torch::kFloat64);
    const TerrainCorrectionResult expected = TerrainCorrection::load(dir.file("y/terrain.pack")).apply(direct);
    CHECK(torch::allclose(expected.corrected, y.terrain->result.corrected, 1e-6, 1e-6));
    CHECK(torch::allclose(direct.dem, y.terrain->dem, 0.0, 1e-3));
    // The raw prediction is left as it was.
    CHECK(torch::equal(y.value, suite.predict(input, SuitePredictOptions{}).target("y").value));

    // The grids source reads the plane at the plots' coordinates.
    SuitePredictOptions from_grids;
    from_grids.terrain = SuiteTerrainOptions{TerrainSourceKind::Grids, grids.path(), torch::kCPU};
    const SuitePredictions g = suite.predict(input, from_grids);
    const TerrainGrids grid_source(grids.path());
    std::vector<double> lat(out.plot_ids.size()), lon(out.plot_ids.size());
    for (const auto& p : plots) {
        lat[static_cast<size_t>(row.at(p.id))] = p.lat;
        lon[static_cast<size_t>(row.at(p.id))] = p.lon;
    }
    const TerrainDescriptors d =
        terrain_descriptors(grid_source, torch::tensor(lat, torch::kFloat64), torch::tensor(lon, torch::kFloat64));
    CHECK(torch::allclose(g.target("y").terrain->dem, d.elevation, 0.0, 1e-6));
    CHECK(torch::allclose(g.target("y").terrain->dem_1km, d.elevation_1km, 0.0, 1e-6));

    // The shipped field: every plot inside its tiles gets descriptors.
    SuitePredictOptions from_field;
    from_field.terrain = SuiteTerrainOptions{};
    const SuitePredictions f = suite.predict(input, from_field);
    CHECK(torch::isfinite(f.target("y").terrain->dem).all().item<bool>());
    CHECK(f.target("y").terrain->result.applied.all().item<bool>());

    const auto table = suite_prediction_columns(out, false);
    std::set<std::string> names;
    for (const auto& c : table) names.insert(c.name);
    for (const char* c : {"y_terrain", "y_terrain_w_resolve", "y_terrain_w_dem", "y_terrain_w_dem_1km", "y_dem",
                          "y_dem_1km"}) {
        CHECK(names.count(c) == 1);
    }
    CHECK(names.count("hab_terrain") == 0);

    // Columns source without the elevation column named is refused.
    SuitePredictOptions unnamed;
    unnamed.terrain = SuiteTerrainOptions{TerrainSourceKind::Columns, "", torch::kCPU};
    CHECK_THROWS_AS(suite.predict(input, unnamed), std::invalid_argument);
}

TEST_CASE("terrain correction on a suite without one is refused", "[suite][predict][terrain]") {
    TempDir dir;
    build_suite(dir);
    const ScoringFiles in = scoring_files(dir);
    SuitePredictor suite = SuitePredictor::load(dir.path());
    SuitePredictOptions opts;
    opts.terrain = SuiteTerrainOptions{};
    CHECK_THROWS_WITH(suite.predict(SuiteInput::csv(in.header, in.species), opts),
                      Catch::Matchers::ContainsSubstring("no loaded target a terrain correction"));
}
