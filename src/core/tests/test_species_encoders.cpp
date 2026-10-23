#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "resolve/dataset.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/sparse_rows.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>

using namespace resolve;
using namespace resolve::test;
using Catch::Matchers::WithinAbs;

namespace {

std::vector<SpeciesRecord> make_test_records() {
    // SpeciesRecord: {species_id, genus, family, abundance, plot_id}
    return {
        {"sp_a", "genus_1", "family_1", 10.0f, "P1"},
        {"sp_b", "genus_1", "family_1", 5.0f,  "P1"},
        {"sp_c", "genus_2", "family_2", 2.0f,  "P1"},
        {"sp_a", "genus_1", "family_1", 8.0f,  "P2"},
        {"sp_d", "genus_2", "family_2", 3.0f,  "P2"},
        {"sp_b", "genus_1", "family_1", 7.0f,  "P3"},
        {"sp_c", "genus_2", "family_2", 4.0f,  "P3"},
        {"sp_e", "genus_3", "family_2", 1.0f,  "P3"},
    };
}

std::vector<std::string> plot_species(const PlotAssemblages& plots, int64_t p) {
    std::vector<std::string> out;
    for (int32_t s : plots.species(p)) out.push_back(plots.species_name(s));
    return out;
}

// The genus / family name of the species of entry i of plot p, "" for none.
std::string entry_genus(const PlotAssemblages& plots, int64_t p, size_t i) {
    const int32_t g = plots.genus_of(plots.species(p)[i]);
    return g >= 0 ? plots.genus_name(g) : std::string();
}
std::string entry_family(const PlotAssemblages& plots, int64_t p, size_t i) {
    const int32_t f = plots.family_of(plots.species(p)[i]);
    return f >= 0 ? plots.family_name(f) : std::string();
}

// The value entry i of plot p carries into the encodings.
float entry_value(const PlotAssemblages& plots, int64_t p, size_t i) {
    return plots.value(plots.ranking(p)[i]);
}

// Adds `weight` at the feature-hash slot of `name` in `row`.
void add_hashed(std::vector<float>& row, std::string_view name, float weight) {
    const HashBucketSign slot = feature_hash_bucket_sign(name, static_cast<int32_t>(row.size()));
    row[static_cast<size_t>(slot.bucket)] += slot.sign * weight;
}

// The dense row of plot p of a dataset's sparse species vector.
torch::Tensor species_row(const ResolveDataset& ds, int64_t p) {
    return sparse_rows_to_dense(ds.species_vector())[p];
}

using Names = std::vector<std::string>;

}  // namespace

// ============================================================================
// PlotAssemblages: grouping and the one duplicate rule
// ============================================================================

TEST_CASE("PlotAssemblages groups records by plot in the order asked for",
          "[assemblages]") {
    const auto plots = PlotAssemblages::build(make_test_records(), {"P3", "P9", "P1"});
    REQUIRE(plots.n_plots() == 3);
    REQUIRE(plot_species(plots, 0) == Names{"sp_b", "sp_c", "sp_e"});
    REQUIRE(plots.species(1).empty());                 // no records
    REQUIRE(plot_species(plots, 2) == Names{"sp_a", "sp_b", "sp_c"});
    // P2 is not asked for, so its records are dropped.
    REQUIRE(plots.n_entries() == 6);
    REQUIRE(plots.plots_per_species()[static_cast<size_t>(plots.species(0)[0])] == 2);

    // In the order the plots first appear: P1, P2, P3.
    const auto by_appearance = PlotAssemblages::build(make_test_records());
    REQUIRE(by_appearance.n_plots() == 3);
    REQUIRE(plot_species(by_appearance, 0) == Names{"sp_a", "sp_b", "sp_c"});
    REQUIRE(plot_species(by_appearance, 1) == Names{"sp_a", "sp_d"});
    REQUIRE(plot_species(by_appearance, 2) == Names{"sp_b", "sp_c", "sp_e"});
}

TEST_CASE("PlotAssemblages::Builder groups coded records and drops unmapped plots",
          "[assemblages]") {
    PlotAssemblages::Builder builder;
    builder.add(0, "sp_a", 1.0f, "gen_z", "");
    builder.add(2, "sp_b", 2.0f, "gen_b", "fam_b");
    builder.add(1, "sp_a", 4.0f, "gen_a", "fam_a");  // code 1 is dropped below
    builder.add(0, "sp_c", 3.0f, "", "");
    builder.add(2, "sp_b", 5.0f, "", "");
    REQUIRE(builder.n_records() == 5);
    // Code 0 is plot 1, code 2 is plot 0, code 1 is dropped.
    const std::vector<int32_t> plot_of = {1, -1, 0};
    const auto plots = std::move(builder).finish(2, plot_of, RepresentationMode::Abundance);

    REQUIRE(plots.n_plots() == 2);
    REQUIRE(plot_species(plots, 0) == Names{"sp_b"});
    REQUIRE(plots.ranking(0)[0] == 7.0f);
    REQUIRE(plot_species(plots, 1) == Names{"sp_a", "sp_c"});
    // The dropped plot's record names no taxonomy: sp_a keeps gen_z, not gen_a.
    REQUIRE(entry_genus(plots, 1, 0) == "gen_z");
    REQUIRE(entry_family(plots, 1, 0).empty());
    REQUIRE(entry_genus(plots, 0, 0) == "gen_b");
    REQUIRE(plots.names_taxa());
    // Name order for ties: sp_a < sp_b < sp_c.
    const int32_t a = plots.species(1)[0];
    const int32_t c = plots.species(1)[1];
    const int32_t b = plots.species(0)[0];
    REQUIRE(plots.species_rank(a) < plots.species_rank(b));
    REQUIRE(plots.species_rank(b) < plots.species_rank(c));
}

TEST_CASE("PlotAssemblages merges a species' records within a plot", "[assemblages][duplicates]") {
    // sp_x is recorded in two layers of P1 and once in P2.
    const std::vector<SpeciesRecord> records = {
        {"sp_x", "gen_b", "fam_b", 2.0f, "P1"},
        {"sp_y", "gen_y", "fam_y", 4.0f, "P1"},
        {"sp_x", "gen_a", "",      3.0f, "P1"},
        {"sp_x", "gen_c", "fam_a", 1.0f, "P2"},
    };
    const auto plots = PlotAssemblages::build(records, {"P1", "P2"});

    SECTION("one entry per species, where its first record stood") {
        REQUIRE(plot_species(plots, 0) == Names{"sp_x", "sp_y"});
    }
    SECTION("the abundance is the sum over the species' records") {
        REQUIRE(entry_value(plots, 0, 0) == 5.0f);
        REQUIRE(plots.ranking(0)[0] == 5.0f);
        REQUIRE(entry_value(plots, 0, 1) == 4.0f);
    }
    SECTION("every entry of a species carries the smallest genus and family named anywhere") {
        REQUIRE(entry_genus(plots, 0, 0) == "gen_a");
        REQUIRE(entry_family(plots, 0, 0) == "fam_a");
        REQUIRE(entry_genus(plots, 1, 0) == "gen_a");
        REQUIRE(entry_family(plots, 1, 0) == "fam_a");
    }
    SECTION("presence/absence carries 1.0 and keeps the summed ranking") {
        const auto presence =
            PlotAssemblages::build(records, {"P1", "P2"}, RepresentationMode::PresenceAbsence);
        REQUIRE(entry_value(presence, 0, 0) == 1.0f);
        REQUIRE(entry_value(presence, 0, 1) == 1.0f);
        REQUIRE(presence.ranking(0)[0] == 5.0f);
        REQUIRE(presence.representation() == RepresentationMode::PresenceAbsence);
    }
}

// ============================================================================
// Feature-hash aggregation
// ============================================================================

TEST_CASE("The hash encoding adds each species at its hash_bucket_sign slot",
          "[hash][parity]") {
    const int hash_dim = 64;
    const std::vector<std::pair<std::string, float>> sp = {
        {"Quercus robur", 3.0f}, {"Fagus sylvatica", 2.0f}, {"Picea abies", 5.0f},
        {"Ranunculus acris", 1.5f}
    };
    std::string species_csv = "plot_id,sp,cover\n";
    for (const auto& [name, abd] : sp) species_csv += "P1," + name + "," + std::to_string(abd) + "\n";
    TempFile header("plot_id,y\nP1,1\n");
    TempFile species(species_csv);
    DatasetConfig cfg;
    cfg.hash_dim = hash_dim;
    cfg.selection = SelectionMode::All;
    cfg.track_unknown_fraction = false;
    const auto ds = ResolveDataset::from_csv(header.path(), species.path(), Roles(),
                                             {TargetSpec::regression("y")}, cfg);
    const auto emb = ds.hash_embedding()[0].contiguous();

    std::vector<float> expected(hash_dim, 0.0f);
    bool finalizer_moves_a_bucket = false;
    for (const auto& [name, abd] : sp) {
        const HashBucketSign slot = hash_bucket_sign(feature_hash_key(name), hash_dim);
        expected[slot.bucket] += slot.sign * abd;
        // The bucket comes from the finalized key, not from the string hash
        // modulo hash_dim.
        const int unfinalized = static_cast<int>(feature_hash(name) % static_cast<uint32_t>(hash_dim));
        if (unfinalized != slot.bucket) finalizer_moves_a_bucket = true;
    }
    for (int i = 0; i < hash_dim; ++i) {
        REQUIRE(emb[i].item<float>() == expected[static_cast<size_t>(i)]);
    }
    REQUIRE(finalizer_moves_a_bucket);
}

// ============================================================================
// SpeciesVocab: the one vocabulary policy
// ============================================================================

TEST_CASE("SpeciesVocab ranks species by plot count, ties by name", "[species_vocab]") {
    std::vector<SpeciesRecord> records = {
        {"zeta", "", "", 1.0f, "P1"}, {"zeta", "", "", 1.0f, "P2"}, {"zeta", "", "", 1.0f, "P3"},
        {"beta", "", "", 1.0f, "P1"}, {"beta", "", "", 1.0f, "P2"},
        {"alpha", "", "", 1.0f, "P1"}, {"alpha", "", "", 1.0f, "P2"},
        // Two layers of one species in one plot count one plot.
        {"gamma", "", "", 1.0f, "P1"}, {"gamma", "", "", 1.0f, "P1"},
    };
    auto vocab = SpeciesVocab::from_records(records);

    REQUIRE(vocab.names() == Names{"<UNK>", "zeta", "alpha", "beta", "gamma"});
    REQUIRE(vocab.size() == 5);
    REQUIRE(vocab.encode("zeta") == 1);
    REQUIRE(vocab.encode("unknown_species") == 0);
}

TEST_CASE("SpeciesVocab min_count filters rare species", "[species_vocab]") {
    auto vocab = SpeciesVocab::from_records(make_test_records(), /*min_count=*/2);
    REQUIRE(vocab.encode("sp_a") > 0);  // P1, P2
    REQUIRE(vocab.encode("sp_b") > 0);  // P1, P3
    REQUIRE(vocab.encode("sp_c") > 0);  // P1, P3
    REQUIRE(vocab.encode("sp_d") == 0); // only P2
    REQUIRE(vocab.encode("sp_e") == 0); // only P3
}

TEST_CASE("SpeciesVocab round-trips through its ordered names", "[species_vocab]") {
    const auto fitted = SpeciesVocab::from_records(make_test_records());
    const auto adopted = SpeciesVocab::from_ordered(fitted.names());
    REQUIRE(adopted.names() == fitted.names());
    for (const auto& name : fitted.names()) {
        REQUIRE(adopted.encode(name) == fitted.encode(name));
    }
    REQUIRE(SpeciesVocab().empty());
    REQUIRE(SpeciesVocab().size() == 1);
}

// ============================================================================
// Taxonomy slots and AggregationMode
// ============================================================================

TEST_CASE("encode_taxonomy_slots ranks taxa by abundance or by species count",
          "[taxonomy][aggregation]") {
    // gen_big holds one species at 10; gen_many three species at 1 each.
    const std::vector<SpeciesRecord> records = {
        {"s1", "gen_big", "fam_x", 10.0f, "P"},
        {"s2", "gen_many", "fam_y", 1.0f, "P"},
        {"s3", "gen_many", "fam_y", 1.0f, "P"},
        {"s4", "gen_many", "fam_y", 1.0f, "P"},
    };
    const auto plots = PlotAssemblages::build(records, {"P"});
    TaxonomyVocab taxonomy;
    taxonomy.fit(records);

    const auto by_abundance =
        encode_taxonomy_slots(plots, taxonomy, AggregationMode::Abundance, 1);
    REQUIRE(by_abundance.genus_ids[0][0].item<int64_t>() == taxonomy.encode_genus("gen_big"));
    REQUIRE(by_abundance.family_ids[0][0].item<int64_t>() == taxonomy.encode_family("fam_x"));

    const auto by_count = encode_taxonomy_slots(plots, taxonomy, AggregationMode::Count, 1);
    REQUIRE(by_count.genus_ids[0][0].item<int64_t>() == taxonomy.encode_genus("gen_many"));
    REQUIRE(by_count.family_ids[0][0].item<int64_t>() == taxonomy.encode_family("fam_y"));

    // Under presence/absence every species weighs 1, so abundance ranks like count.
    const auto presence = PlotAssemblages::build(records, {"P"}, RepresentationMode::PresenceAbsence);
    const auto presence_slots =
        encode_taxonomy_slots(presence, taxonomy, AggregationMode::Abundance, 1);
    REQUIRE(presence_slots.genus_ids[0][0].item<int64_t>() == taxonomy.encode_genus("gen_many"));

    // Slots past the plot's taxa stay 0.
    const auto wide = encode_taxonomy_slots(plots, taxonomy, AggregationMode::Count, 4);
    REQUIRE(wide.genus_ids[0][1].item<int64_t>() == taxonomy.encode_genus("gen_big"));
    REQUIRE(wide.genus_ids[0][2].item<int64_t>() == 0);
}

// ============================================================================
// RankPoolEncoder
// ============================================================================

TEST_CASE("RankPoolEncoder fit + transform produces correct shapes", "[rank_pool_enc]") {
    auto records = make_test_records();
    std::vector<std::string> plot_ids = {"P1", "P2", "P3"};

    RankPoolEncoder encoder(PoolWeighting::Log1p);
    encoder.fit(records);

    REQUIRE(encoder.is_fitted());
    REQUIRE(encoder.n_species_vocab() == 6);

    auto result = encoder.transform(records, plot_ids);

    REQUIRE(result.species_ids.size(0) == 3);   // 3 plots
    REQUIRE(result.species_ids.size(1) == 3);   // widest plot
    REQUIRE(result.genus_ids.sizes() == result.species_ids.sizes());
    REQUIRE(result.family_ids.sizes() == result.species_ids.sizes());
    REQUIRE(result.weights.sizes() == result.species_ids.sizes());
    REQUIRE(result.mask.sizes() == result.species_ids.sizes());
    REQUIRE(result.has_cover.size(0) == 3);
}

TEST_CASE("RankPoolEncoder mask is correct", "[rank_pool_enc]") {
    auto records = make_test_records();
    std::vector<std::string> plot_ids = {"P1", "P2", "P3"};

    RankPoolEncoder encoder(PoolWeighting::Binary);
    encoder.fit(records);
    auto result = encoder.transform(records, plot_ids);

    REQUIRE(result.mask[0].sum().item<int64_t>() == 3);
    REQUIRE(result.mask[1].sum().item<int64_t>() == 2);
}

TEST_CASE("RankPoolEncoder rank weighting assigns 1/rank", "[rank_pool_enc]") {
    auto records = make_test_records();
    RankPoolEncoder encoder(PoolWeighting::Rank);
    encoder.fit(records);
    auto result = encoder.transform(records, {"P1"});

    // P1 has sp_a(10), sp_b(5), sp_c(2) in that order -> weights 1, 1/2, 1/3.
    REQUIRE_THAT(result.weights[0][0].item<float>(), WithinAbs(1.0f, 1e-6));
    REQUIRE_THAT(result.weights[0][1].item<float>(), WithinAbs(0.5f, 1e-6));
    REQUIRE_THAT(result.weights[0][2].item<float>(), WithinAbs(1.0f / 3.0f, 1e-6));
}

TEST_CASE("RankPoolEncoder rank weighting is dense (ties share rank, no gaps)", "[rank_pool_enc]") {
    // Tied abundances use DENSE ranking (1,2,2,3), not competition ranking
    // with gaps (1,2,2,4): the smallest weight is 1/3, not 1/4.
    std::vector<SpeciesRecord> records = {
        {"sp_a", "g_a", "f_a", 10.0f, "P1"},
        {"sp_b", "g_b", "f_b", 5.0f,  "P1"},
        {"sp_c", "g_c", "f_c", 5.0f,  "P1"},
        {"sp_d", "g_d", "f_d", 2.0f,  "P1"},
    };
    RankPoolEncoder encoder(PoolWeighting::Rank);
    encoder.fit(records);
    auto result = encoder.transform(records, {"P1"});
    REQUIRE_THAT(result.weights[0].min().item<float>(), WithinAbs(1.0f / 3.0f, 1e-6));
}

TEST_CASE("RankPoolEncoder transform before a vocabulary throws", "[rank_pool_enc]") {
    RankPoolEncoder encoder;
    REQUIRE_THROWS(encoder.transform(std::vector<SpeciesRecord>{}, std::vector<std::string>{}));
}

TEST_CASE("RankPoolEncoder gives a species recorded in two layers one slot",
          "[rank_pool_enc][duplicates]") {
    const std::vector<SpeciesRecord> records = {
        {"sp_a", "", "", 2.0f, "P1"}, {"sp_b", "", "", 1.0f, "P1"}, {"sp_a", "", "", 3.0f, "P1"}};
    RankPoolEncoder encoder(PoolWeighting::Abundance);
    encoder.fit(records);
    auto result = encoder.transform(records, {"P1"});
    REQUIRE(result.species_ids.size(1) == 2);
    REQUIRE(result.species_ids[0][0].item<int64_t>() == encoder.species_vocab().encode("sp_a"));
    REQUIRE(result.weights[0][0].item<float>() == 5.0f);
}

TEST_CASE("RankPoolEncoder under presence/absence weighs every species alike",
          "[rank_pool_enc][representation]") {
    auto records = make_test_records();
    RankPoolEncoder encoder(PoolWeighting::Abundance);
    encoder.fit(records);

    auto abundance = encoder.transform(records, {"P1"}, 0, /*has_abundance_column=*/true);
    REQUIRE(abundance.weights[0][0].item<float>() == 10.0f);
    REQUIRE(abundance.has_cover[0].item<float>() == 1.0f);

    auto presence = encoder.transform(records, {"P1"}, 0, /*has_abundance_column=*/true,
                                      RepresentationMode::PresenceAbsence);
    for (int64_t j = 0; j < 3; ++j) REQUIRE(presence.weights[0][j].item<float>() == 1.0f);
    // No cover reaches the pool, so the model reads the plot as presence-only.
    REQUIRE(presence.has_cover[0].item<float>() == 0.0f);
}

// ============================================================================
// EmbeddingEncoder
// ============================================================================

TEST_CASE("EmbeddingEncoder fit + transform produces correct shapes", "[embed_enc]") {
    auto records = make_test_records();
    std::vector<std::string> plot_ids = {"P1", "P2", "P3"};

    EmbeddingEncoder encoder(/*top_k_species=*/3, /*top_k_taxonomy=*/2);
    encoder.fit(records);
    REQUIRE(encoder.is_fitted());

    auto result = encoder.transform(records, plot_ids);
    REQUIRE(result.species_ids.size(0) == 3);
    REQUIRE(result.species_ids.size(1) == 3);
    REQUIRE(result.genus_ids.size(0) == 3);
    REQUIRE(result.genus_ids.size(1) == 2);
    REQUIRE(result.family_ids.size(0) == 3);
    REQUIRE(result.family_ids.size(1) == 2);
}

TEST_CASE("EmbeddingEncoder slots run from the most abundant species, whatever the plot size",
          "[embed_enc][ranking]") {
    auto records = make_test_records();
    EmbeddingEncoder encoder(/*top_k_species=*/5, /*top_k_taxonomy=*/2);
    encoder.fit(records);
    const auto& vocab = encoder.species_vocab();

    // Every plot holds fewer species than the 5 slots; the slots still carry
    // the abundance ranking, then padding.
    auto result = encoder.transform(records, {"P1", "P2", "P3"});
    auto ids = result.species_ids.accessor<int64_t, 2>();
    REQUIRE(ids[0][0] == vocab.encode("sp_a"));  // 10
    REQUIRE(ids[0][1] == vocab.encode("sp_b"));  // 5
    REQUIRE(ids[0][2] == vocab.encode("sp_c"));  // 2
    REQUIRE(ids[0][3] == 0);
    REQUIRE(ids[2][0] == vocab.encode("sp_b"));  // 7
    REQUIRE(ids[2][1] == vocab.encode("sp_c"));  // 4
    REQUIRE(ids[2][2] == vocab.encode("sp_e"));  // 1

    EmbeddingEncoder bottom(/*top_k_species=*/5, /*top_k_taxonomy=*/2, SelectionMode::Bottom);
    bottom.set_vocab(encoder.vocab());
    auto least_first = bottom.transform(records, {"P1"});
    REQUIRE(least_first.species_ids[0][0].item<int64_t>() == vocab.encode("sp_c"));
    REQUIRE(least_first.species_ids[0][2].item<int64_t>() == vocab.encode("sp_a"));
}

TEST_CASE("EmbeddingEncoder gives a species recorded in two layers one slot",
          "[embed_enc][duplicates]") {
    // sp_b's two layers (3 + 3) outrank sp_a (5) once merged.
    const std::vector<SpeciesRecord> records = {
        {"sp_a", "", "", 5.0f, "P1"}, {"sp_b", "", "", 3.0f, "P1"}, {"sp_b", "", "", 3.0f, "P1"},
        {"sp_c", "", "", 1.0f, "P1"}};
    EmbeddingEncoder encoder(/*top_k_species=*/3, /*top_k_taxonomy=*/1);
    encoder.fit(records);
    auto ids = encoder.transform(records, {"P1"}).species_ids;
    const auto& vocab = encoder.species_vocab();
    REQUIRE(ids[0][0].item<int64_t>() == vocab.encode("sp_b"));
    REQUIRE(ids[0][1].item<int64_t>() == vocab.encode("sp_a"));
    REQUIRE(ids[0][2].item<int64_t>() == vocab.encode("sp_c"));
}

TEST_CASE("EmbeddingEncoder transform before a vocabulary throws", "[embed_enc]") {
    EmbeddingEncoder encoder;
    REQUIRE_THROWS(encoder.transform(std::vector<SpeciesRecord>{}, std::vector<std::string>{}));
}

// ============================================================================
// Unknown species are counted once per species
// ============================================================================

TEST_CASE("Unknown-species statistics count species, not records", "[unknown][duplicates]") {
    const std::vector<SpeciesRecord> records = {
        {"known", "", "", 2.0f, "P1"},
        {"novel", "", "", 1.0f, "P1"}, {"novel", "", "", 1.0f, "P1"},  // two layers
    };
    const auto plots = PlotAssemblages::build(records, {"P1"});
    const auto vocab = SpeciesVocab::from_ordered({"<UNK>", "known"});
    const auto stats = compute_unknown_species_stats(plots, vocab);
    REQUIRE(stats.count[0].item<float>() == 1.0f);
    REQUIRE_THAT(stats.fraction[0].item<float>(), WithinAbs(0.5f, 1e-6));

    const auto presence =
        PlotAssemblages::build(records, {"P1"}, RepresentationMode::PresenceAbsence);
    REQUIRE_THAT(compute_unknown_species_stats(presence, vocab).fraction[0].item<float>(),
                 WithinAbs(0.5f, 1e-6));
}

// ============================================================================
// Every dataset encoding: duplicates, representation, aggregation
// ============================================================================

namespace {

// Two plots. P1 records sp_a in two layers (2 + 3), sp_b at 4 and the three
// species of gen_many (s1..s3) at 1 each; P2 records sp_a at 1. sp_a and sp_b
// belong to gen_big.
const char* kLayeredSpecies =
    "plot_id,sp,cover,genus,family\n"
    "P1,sp_a,2,gen_big,fam_x\n"
    "P1,sp_b,4,gen_big,fam_x\n"
    "P1,sp_a,3,gen_big,fam_x\n"
    "P1,s1,1,gen_many,fam_y\n"
    "P1,s2,1,gen_many,fam_y\n"
    "P1,s3,1,gen_many,fam_y\n"
    "P2,sp_a,1,gen_big,fam_x\n";

ResolveDataset load_layered(DatasetConfig cfg) {
    TempFile header("plot_id,y\nP1,1\nP2,2\n");
    TempFile species(kLayeredSpecies);
    cfg.track_unknown_fraction = false;
    return ResolveDataset::from_csv(header.path(), species.path(), Roles().taxonomy(),
                                    {TargetSpec::regression("y")}, cfg);
}

DatasetConfig encoding(SpeciesEncodingMode mode) {
    DatasetConfig cfg;
    cfg.species_encoding = mode;
    cfg.selection = SelectionMode::All;
    cfg.use_taxonomy = true;
    return cfg;
}

int64_t code_of(const ResolveDataset& ds, const std::string& name) {
    const auto& vocab = ds.species_vocab();
    return std::find(vocab.begin(), vocab.end(), name) - vocab.begin();
}

}  // namespace

TEST_CASE("Every encoding reads a species' layers as one summed entry",
          "[dataset][duplicates]") {
    SECTION("hash sums the layers into one weight") {
        auto cfg = encoding(SpeciesEncodingMode::Hash);
        cfg.hash_dim = 64;
        cfg.normalization = NormalizationMode::Log1p;
        auto ds = load_layered(cfg);
        // log1p of the summed cover, not log1p(2) + log1p(3).
        std::vector<float> row(64, 0.0f);
        add_hashed(row, "sp_a", std::log1p(5.0f));
        add_hashed(row, "sp_b", std::log1p(4.0f));
        for (const char* s : {"s1", "s2", "s3"}) add_hashed(row, s, std::log1p(1.0f));
        for (int i = 0; i < 64; ++i) {
            REQUIRE_THAT(ds.hash_embedding()[0][i].item<float>(), WithinAbs(row[static_cast<size_t>(i)], 1e-5));
        }
    }
    SECTION("sparse writes the summed cover") {
        auto ds = load_layered(encoding(SpeciesEncodingMode::Sparse));
        REQUIRE(species_row(ds, 0)[code_of(ds, "sp_a")].item<float>() == 5.0f);
    }
    SECTION("rank_pool gives the species one slot") {
        auto ds = load_layered(encoding(SpeciesEncodingMode::RankPool));
        REQUIRE(ds.pool_mask()[0].sum().item<int64_t>() == 5);
        REQUIRE(ds.species_ids()[0][0].item<int64_t>() == code_of(ds, "sp_a"));
    }
    SECTION("embed ranks the summed cover first") {
        auto cfg = encoding(SpeciesEncodingMode::Embed);
        cfg.selection = SelectionMode::Top;
        cfg.top_k_species = 2;
        auto ds = load_layered(cfg);
        REQUIRE(ds.species_ids()[0][0].item<int64_t>() == code_of(ds, "sp_a"));
        REQUIRE(ds.species_ids()[0][1].item<int64_t>() == code_of(ds, "sp_b"));
    }
}

TEST_CASE("PresenceAbsence reaches every encoding", "[dataset][representation]") {
    const auto presence = [](SpeciesEncodingMode mode) {
        auto cfg = encoding(mode);
        cfg.representation = RepresentationMode::PresenceAbsence;
        return cfg;
    };

    SECTION("hash hashes 1.0 per species") {
        auto cfg = presence(SpeciesEncodingMode::Hash);
        cfg.hash_dim = 64;
        auto ds = load_layered(cfg);
        std::vector<float> row(64, 0.0f);
        for (const char* s : {"sp_a", "sp_b", "s1", "s2", "s3"}) add_hashed(row, s, 1.0f);
        for (int i = 0; i < 64; ++i) {
            REQUIRE(ds.hash_embedding()[0][i].item<float>() == row[static_cast<size_t>(i)]);
        }
    }
    SECTION("sparse writes 1.0") {
        auto ds = load_layered(presence(SpeciesEncodingMode::Sparse));
        REQUIRE(species_row(ds, 0)[code_of(ds, "sp_a")].item<float>() == 1.0f);
        REQUIRE(species_row(ds, 0)[code_of(ds, "sp_b")].item<float>() == 1.0f);
    }
    SECTION("rank_pool and transformer weigh species alike and report no cover") {
        for (auto mode : {SpeciesEncodingMode::RankPool, SpeciesEncodingMode::Transformer}) {
            auto cfg = presence(mode);
            cfg.pool_weighting = PoolWeighting::Abundance;
            auto ds = load_layered(cfg);
            REQUIRE(ds.pool_weights()[0].sum().item<float>() == 5.0f);
            REQUIRE(ds.pool_has_cover()[0].item<float>() == 0.0f);
        }
    }
    SECTION("embed ranks its taxonomy slots by species present") {
        auto cfg = presence(SpeciesEncodingMode::Embed);
        cfg.selection = SelectionMode::Top;
        cfg.top_k = 1;
        auto ds = load_layered(cfg);
        // gen_many holds three species, gen_big two.
        REQUIRE(ds.genus_ids()[0][0].item<int64_t>() ==
                ds.taxonomy_vocab().encode_genus("gen_many"));
    }
    SECTION("the schema records it") {
        auto ds = load_layered(presence(SpeciesEncodingMode::Hash));
        REQUIRE(ds.schema().dataset_config.representation == RepresentationMode::PresenceAbsence);
    }
}

TEST_CASE("AggregationMode ranks the fixed taxonomy slots of hash, sparse and embed",
          "[dataset][aggregation]") {
    for (auto mode : {SpeciesEncodingMode::Hash, SpeciesEncodingMode::Sparse,
                      SpeciesEncodingMode::Embed}) {
        DYNAMIC_SECTION("encoding " << static_cast<int>(mode)) {
            auto cfg = encoding(mode);
            cfg.selection = SelectionMode::Top;
            cfg.top_k = 1;
            if (mode == SpeciesEncodingMode::Sparse) cfg.selection = SelectionMode::All;

            // Abundance: gen_big covers 5 + 4 = 9 against gen_many's 3.
            auto by_abundance = load_layered(cfg);
            REQUIRE(by_abundance.genus_ids().size(1) == 1);
            REQUIRE(by_abundance.genus_ids()[0][0].item<int64_t>() ==
                    by_abundance.taxonomy_vocab().encode_genus("gen_big"));

            // Count: gen_many holds three species against gen_big's two.
            cfg.aggregation = AggregationMode::Count;
            auto by_count = load_layered(cfg);
            REQUIRE(by_count.genus_ids()[0][0].item<int64_t>() ==
                    by_count.taxonomy_vocab().encode_genus("gen_many"));
            REQUIRE(by_count.family_ids()[0][0].item<int64_t>() ==
                    by_count.taxonomy_vocab().encode_family("fam_y"));
            REQUIRE(by_count.schema().dataset_config.aggregation == AggregationMode::Count);
        }
    }
}

// The per-species taxonomy encodings encode the same tensors under either
// aggregation; Count changes how the model pools a plot's genera and families
// (test_species_encoder.cpp), so the dataset records it for the model.
TEST_CASE("AggregationMode::Count reaches the per-species taxonomy encodings' schema",
          "[dataset][aggregation]") {
    for (auto mode : {SpeciesEncodingMode::RankPool, SpeciesEncodingMode::Transformer}) {
        auto cfg = encoding(mode);
        auto by_abundance = load_layered(cfg);
        cfg.aggregation = AggregationMode::Count;
        auto by_count = load_layered(cfg);
        REQUIRE(by_count.schema().dataset_config.aggregation == AggregationMode::Count);
        REQUIRE(torch::equal(by_count.pool_weights(), by_abundance.pool_weights()));
        REQUIRE(torch::equal(by_count.pool_genus_ids(), by_abundance.pool_genus_ids()));
    }
}

TEST_CASE("Fixed taxonomy slots follow the species' resolved taxonomy",
          "[dataset][taxonomy]") {
    // sp_a names gen_b in P1 and gen_a in P2: every slot reads gen_a, the
    // lexicographically smallest, the same genus the species' code maps to.
    TempFile header("plot_id,y\nP1,1\nP2,2\n");
    TempFile species("plot_id,sp,cover,genus,family\n"
                     "P1,sp_a,5,gen_b,fam\n"
                     "P2,sp_a,5,gen_a,fam\n");
    for (auto mode : {SpeciesEncodingMode::Hash, SpeciesEncodingMode::Sparse,
                      SpeciesEncodingMode::Embed}) {
        auto cfg = encoding(mode);
        cfg.selection = mode == SpeciesEncodingMode::Sparse ? SelectionMode::All : SelectionMode::Top;
        cfg.top_k = 1;
        auto ds = ResolveDataset::from_csv(header.path(), species.path(), Roles().taxonomy(),
                                           {TargetSpec::regression("y")}, cfg);
        const int64_t gen_a = ds.taxonomy_vocab().encode_genus("gen_a");
        REQUIRE(ds.genus_ids()[0][0].item<int64_t>() == gen_a);
        REQUIRE(ds.genus_ids()[1][0].item<int64_t>() == gen_a);
        REQUIRE(ds.species_genus_ids()[code_of(ds, "sp_a")].item<int64_t>() == gen_a);
    }
}

// ============================================================================
// percentile_linear_trunc — numpy int(np.percentile(x, q, "linear")) parity
// ============================================================================

TEST_CASE("percentile_linear_trunc matches int(np.percentile) on skewed input",
          "[percentile]") {
    // The auto species cap (-1) takes the 99th percentile of per-plot lengths.
    // Reference values are int(np.percentile(values, 99)) with numpy's linear
    // interpolation default.

    // sorted [1,5,100]: rank=1.98 -> 5 + 0.98*(100-5) = 98.1 -> int 98.
    {
        std::vector<int64_t> v{100, 1, 5};
        REQUIRE(percentile_linear_trunc(v, 99.0) == 98);
    }
    // sorted [10,20,30,40,50]: rank=3.96 -> 40 + 0.96*10 = 49.6 -> int 49.
    {
        std::vector<int64_t> v{50, 40, 30, 20, 10};
        REQUIRE(percentile_linear_trunc(v, 99.0) == 49);
    }
    // Single element: rank 0 -> that element.
    {
        std::vector<int64_t> v{7};
        REQUIRE(percentile_linear_trunc(v, 99.0) == 7);
    }
    // Constant input: every order statistic equal -> that value.
    {
        std::vector<int64_t> v{3, 3, 3, 3};
        REQUIRE(percentile_linear_trunc(v, 99.0) == 3);
    }
    // Empty input -> 0 (caller guards with std::max(..., 1)).
    {
        std::vector<int64_t> v{};
        REQUIRE(percentile_linear_trunc(v, 99.0) == 0);
    }
}

TEST_CASE("percentile_linear_trunc handles median and bounds", "[percentile]") {
    // sorted [1,2,3,4]: q=50 -> rank=1.5 -> 2 + 0.5*(3-2) = 2.5 -> int 2.
    {
        std::vector<int64_t> v{4, 3, 2, 1};
        REQUIRE(percentile_linear_trunc(v, 50.0) == 2);
    }
    // q=0 -> minimum; q=100 -> maximum.
    {
        std::vector<int64_t> v{9, 2, 7, 4};
        REQUIRE(percentile_linear_trunc(v, 0.0) == 2);
        std::vector<int64_t> w{9, 2, 7, 4};
        REQUIRE(percentile_linear_trunc(w, 100.0) == 9);
    }
}
