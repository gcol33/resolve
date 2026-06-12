#include <catch2/catch_test_macros.hpp>
#include "resolve/species_encoding.hpp"

using namespace resolve;

static std::vector<SpeciesRecord> make_test_records() {
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

// ============================================================================
// Feature-hash CPU/CUDA parity
// ============================================================================

// Independent reimplementation of the device hash (cuda/kernels.cu:
// murmur_hash32 applied to (int64)murmur_hash(species, 0)). If hash_species or
// feature_hash_bucket_sign ever diverges from the kernel scheme (the original
// bug: CPU used a single string hash for the bucket and a second string hash's
// parity for the sign, while the kernel double-hashes), this fails.
static void kernel_hash_formula(const std::string& s, int hash_dim,
                                int& bucket_out, float& sign_out) {
    uint64_t h = static_cast<uint64_t>(static_cast<int64_t>(resolve::murmur_hash(s, 0)));
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    int32_t hi = static_cast<int32_t>(h);
    bucket_out = static_cast<int>(
        static_cast<uint32_t>(hi < 0 ? -hi : hi) % static_cast<uint32_t>(hash_dim));
    sign_out = (hi >= 0) ? 1.0f : -1.0f;
}

TEST_CASE("feature_hash_bucket_sign matches the CUDA kernel hash formula", "[hash][parity]") {
    const int hash_dim = 64;
    const std::vector<std::string> species = {
        "Quercus robur", "Fagus sylvatica", "Picea abies", "sp_a", "x", ""
    };
    for (const auto& s : species) {
        int k_bucket; float k_sign;
        kernel_hash_formula(s, hash_dim, k_bucket, k_sign);
        auto bs = resolve::feature_hash_bucket_sign(s, hash_dim);
        REQUIRE(bs.bucket == k_bucket);
        REQUIRE(bs.sign == k_sign);
        REQUIRE(bs.bucket >= 0);
        REQUIRE(bs.bucket < hash_dim);
    }
}

TEST_CASE("hash_species aggregates into the CUDA-parity buckets", "[hash][parity]") {
    const int hash_dim = 64;
    std::vector<std::pair<std::string, float>> sp = {
        {"Quercus robur", 3.0f}, {"Fagus sylvatica", 2.0f}, {"Picea abies", 5.0f}
    };

    std::vector<float> emb(hash_dim, 0.0f);
    hash_species(sp, emb.data(), hash_dim);

    // Reconstruct the expected embedding from the kernel formula.
    std::vector<float> expected(hash_dim, 0.0f);
    bool uses_finalizer_distinct_from_single_hash = false;
    for (const auto& [name, abd] : sp) {
        int bucket; float sign;
        kernel_hash_formula(name, hash_dim, bucket, sign);
        expected[bucket] += sign * abd;
        // The old (buggy) CPU bucket was murmur_hash(name,0) % hash_dim; confirm
        // the finalizer actually changes at least one bucket so this test would
        // have failed against the pre-fix implementation.
        int old_bucket = static_cast<int>(resolve::murmur_hash(name, 0) % static_cast<uint32_t>(hash_dim));
        if (old_bucket != bucket) uses_finalizer_distinct_from_single_hash = true;
    }
    for (int i = 0; i < hash_dim; ++i) {
        REQUIRE(emb[i] == expected[i]);
    }
    REQUIRE(uses_finalizer_distinct_from_single_hash);
}

// ============================================================================
// SpeciesVocab Tests
// ============================================================================

TEST_CASE("SpeciesVocab from_records builds sorted 1-indexed mapping", "[species_vocab]") {
    auto records = make_test_records();
    auto vocab = SpeciesVocab::from_records(records);

    REQUIRE(vocab.size() == 6);  // 5 species + 1 unknown slot
    REQUIRE(vocab.encode("sp_a") >= 1);
    REQUIRE(vocab.encode("sp_b") >= 1);
    REQUIRE(vocab.encode("unknown_species") == 0);

    // Alphabetical order: sp_a < sp_b < sp_c < sp_d < sp_e
    REQUIRE(vocab.encode("sp_a") < vocab.encode("sp_b"));
    REQUIRE(vocab.encode("sp_b") < vocab.encode("sp_c"));
}

TEST_CASE("SpeciesVocab min_count filters rare species", "[species_vocab]") {
    auto records = make_test_records();
    auto vocab = SpeciesVocab::from_records(records, /*min_count=*/2);

    // sp_d and sp_e appear only once → filtered out
    REQUIRE(vocab.encode("sp_a") > 0);  // appears in P1, P2
    REQUIRE(vocab.encode("sp_b") > 0);  // appears in P1, P3
    REQUIRE(vocab.encode("sp_c") > 0);  // appears in P1, P3
    REQUIRE(vocab.encode("sp_d") == 0); // only in P2
    REQUIRE(vocab.encode("sp_e") == 0); // only in P3
}

// ============================================================================
// RankPoolEncoder Tests
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
    REQUIRE(result.species_ids.size(1) >= 2);    // max_species >= 2
    REQUIRE(result.genus_ids.sizes() == result.species_ids.sizes());
    REQUIRE(result.family_ids.sizes() == result.species_ids.sizes());
    REQUIRE(result.weights.sizes() == result.species_ids.sizes());
    REQUIRE(result.mask.sizes() == result.species_ids.sizes());
    REQUIRE(result.has_cover.size(0) == 3);
    REQUIRE(result.unknown_fraction.size(0) == 3);
}

TEST_CASE("RankPoolEncoder mask is correct", "[rank_pool_enc]") {
    auto records = make_test_records();
    std::vector<std::string> plot_ids = {"P1", "P2", "P3"};

    RankPoolEncoder encoder(PoolWeighting::Binary);
    encoder.fit(records);
    auto result = encoder.transform(records, plot_ids);

    // P1 has 3 species, P2 has 2, P3 has 3
    // max_species = 3, so mask should reflect actual species count
    auto mask_a = result.mask.accessor<bool, 2>();
    int p1_count = 0, p2_count = 0;
    for (int64_t j = 0; j < result.mask.size(1); ++j) {
        if (mask_a[0][j]) p1_count++;
        if (mask_a[1][j]) p2_count++;
    }
    REQUIRE(p1_count == 3);
    REQUIRE(p2_count == 2);
}

TEST_CASE("RankPoolEncoder unknown fraction tracks unknown species", "[rank_pool_enc]") {
    // Add an unknown species to P1
    auto records = make_test_records();
    records.push_back({"unknown_sp", "unk_g", "unk_f", 20.0f, "P1"});

    std::vector<std::string> plot_ids = {"P1", "P2", "P3"};

    RankPoolEncoder encoder(PoolWeighting::Abundance);
    encoder.fit(make_test_records());  // fit on original (without unknown_sp)
    auto result = encoder.transform(records, plot_ids);

    // P1 now has unknown_sp with abundance 20 out of total 37
    float uf = result.unknown_fraction.accessor<float, 1>()[0];
    REQUIRE(uf > 0.0f);
    REQUIRE(uf < 1.0f);
}

TEST_CASE("RankPoolEncoder rank weighting assigns 1/rank", "[rank_pool_enc]") {
    auto records = make_test_records();
    std::vector<std::string> plot_ids = {"P1"};

    RankPoolEncoder encoder(PoolWeighting::Rank);
    encoder.fit(records);
    auto result = encoder.transform(records, plot_ids);

    // P1 has sp_a(10), sp_b(5), sp_c(2) → ranks 1,2,3 → weights 1.0, 0.5, 0.333
    auto w_a = result.weights.accessor<float, 2>();
    auto m_a = result.mask.accessor<bool, 2>();
    std::vector<float> weights;
    for (int64_t j = 0; j < result.weights.size(1); ++j) {
        if (m_a[0][j]) {
            weights.push_back(w_a[0][j]);
        }
    }
    REQUIRE(weights.size() == 3);

    // Highest weight should be 1.0 (rank 1)
    float max_w = *std::max_element(weights.begin(), weights.end());
    REQUIRE(max_w == 1.0f);
}

TEST_CASE("RankPoolEncoder transform not fitted throws", "[rank_pool_enc]") {
    RankPoolEncoder encoder;
    std::vector<SpeciesRecord> records;
    std::vector<std::string> plot_ids;
    REQUIRE_THROWS(encoder.transform(records, plot_ids));
}

// ============================================================================
// EmbeddingEncoder Tests
// ============================================================================

TEST_CASE("EmbeddingEncoder fit + transform produces correct shapes", "[embed_enc]") {
    auto records = make_test_records();
    std::vector<std::string> plot_ids = {"P1", "P2", "P3"};

    EmbeddingEncoder encoder(/*top_k_species=*/3, /*top_k_taxonomy=*/2);
    encoder.fit(records);

    REQUIRE(encoder.is_fitted());

    auto result = encoder.transform(records, plot_ids);

    REQUIRE(result.species_ids.size(0) == 3);   // 3 plots
    REQUIRE(result.species_ids.size(1) == 3);    // top_k_species = 3
    REQUIRE(result.genus_ids.size(0) == 3);
    REQUIRE(result.genus_ids.size(1) == 2);      // top_k_taxonomy = 2
    REQUIRE(result.family_ids.size(0) == 3);
    REQUIRE(result.family_ids.size(1) == 2);
    REQUIRE(result.unknown_fraction.size(0) == 3);
}

TEST_CASE("EmbeddingEncoder selects top species by abundance", "[embed_enc]") {
    auto records = make_test_records();
    std::vector<std::string> plot_ids = {"P1"};

    EmbeddingEncoder encoder(/*top_k_species=*/2, /*top_k_taxonomy=*/2);
    encoder.fit(records);
    auto result = encoder.transform(records, plot_ids);

    // P1: sp_a(10), sp_b(5), sp_c(2) → top-2 are sp_a and sp_b
    auto sp_a = result.species_ids.accessor<int64_t, 2>();
    // Both should be non-zero (known species)
    REQUIRE(sp_a[0][0] > 0);
    REQUIRE(sp_a[0][1] > 0);
}

TEST_CASE("EmbeddingEncoder transform not fitted throws", "[embed_enc]") {
    EmbeddingEncoder encoder;
    std::vector<SpeciesRecord> records;
    std::vector<std::string> plot_ids;
    REQUIRE_THROWS(encoder.transform(records, plot_ids));
}

// ============================================================================
// percentile_linear_trunc — numpy int(np.percentile(x, q, "linear")) parity
// ============================================================================

TEST_CASE("percentile_linear_trunc matches int(np.percentile) on skewed input",
          "[percentile]") {
    // The auto species cap (-1) takes the 99th percentile of per-plot lengths.
    // Reference values are int(np.percentile(values, 99)) with numpy's linear
    // interpolation default. The earlier floor-rank index (no interpolation)
    // returned the lower order statistic and over-truncated.

    // sorted [1,5,100]: rank=1.98 -> 5 + 0.98*(100-5) = 98.1 -> int 98.
    // (floor-rank would have returned values[1] = 5.)
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
