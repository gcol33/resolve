#include <catch2/catch_test_macros.hpp>
#include <torch/torch.h>

#include "resolve/hash_core.hpp"
#include "resolve/species_encoding.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#ifdef RESOLVE_HAS_CUDA
#include "resolve/cuda/feature_hash.hpp"
#include <c10/cuda/CUDAFunctions.h>
#endif

using namespace resolve;

namespace {

constexpr uint64_t kFmixC1 = 0xff51afd7ed558ccdULL;
constexpr uint64_t kFmixC2 = 0xc4ceb9fe1a85ec53ULL;

// Independent reference for hash_bucket_sign: fmix64 written out here, |h|
// taken in 64-bit signed arithmetic, so h == INT32_MIN gives 2^31.
HashBucketSign reference_slot(int64_t key, int32_t hash_dim) {
    uint64_t h = static_cast<uint64_t>(key);
    h ^= h >> 33;
    h *= kFmixC1;
    h ^= h >> 33;
    h *= kFmixC2;
    h ^= h >> 33;
    const int32_t hi = static_cast<int32_t>(h);
    const int64_t magnitude = hi < 0 ? -static_cast<int64_t>(hi) : static_cast<int64_t>(hi);
    return {static_cast<int32_t>(magnitude % hash_dim), hi >= 0 ? 1.0f : -1.0f};
}

// Multiplicative inverse of an odd number modulo 2^64 (Newton iteration; each
// step doubles the number of correct low bits, starting from 3).
constexpr uint64_t inverse_mod_2_64(uint64_t a) {
    uint64_t x = a;
    for (int i = 0; i < 6; ++i) x *= 2 - a * x;
    return x;
}

// Inverse of fmix64: x ^= x >> 33 is its own inverse because 2 * 33 >= 64.
uint64_t fmix64_inverse(uint64_t h) {
    h ^= h >> 33;
    h *= inverse_mod_2_64(kFmixC2);
    h ^= h >> 33;
    h *= inverse_mod_2_64(kFmixC1);
    h ^= h >> 33;
    return h;
}

// Keys whose finalized low 32 bits are exactly 0x80000000 (h == INT32_MIN),
// one per choice of the discarded high 32 bits.
std::vector<int64_t> int32_min_keys() {
    std::vector<int64_t> keys;
    for (uint64_t high : {0x0ULL, 0x1ULL, 0xdeadbeefULL, 0xffffffffULL}) {
        keys.push_back(static_cast<int64_t>(fmix64_inverse((high << 32) | 0x80000000ULL)));
    }
    return keys;
}

std::string random_name(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> length(0, 40);
    std::uniform_int_distribution<int> byte(1, 255);
    std::string s(static_cast<size_t>(length(rng)), ' ');
    for (char& c : s) c = static_cast<char>(byte(rng));
    return s;
}

}  // namespace

TEST_CASE("murmur_fmix_i32 reproduces reference values", "[hash]") {
    REQUIRE(murmur_fmix_i32(0) == 0);
    REQUIRE(murmur_fmix_i32(1) == 885181228);
    REQUIRE(murmur_fmix_i32(-1) == 1266835233);
    REQUIRE(murmur_fmix_i32(123456789) == 1669658758);
    REQUIRE(murmur_fmix_i32(INT64_MIN) == -1355701101);
    REQUIRE(murmur_fmix_i32(INT64_MAX) == -1456411158);
}

TEST_CASE("feature_hash_bucket_sign reproduces reference slots", "[hash]") {
    struct Expected { const char* name; uint32_t key; int32_t h;
                      int32_t bucket64; int32_t bucket1000; float sign; };
    const std::vector<Expected> table = {
        {"Quercus robur",    0x782a2fd8u,  141646451, 51, 451,  1.0f},
        {"Fagus sylvatica",  0x67b46b48u, 1172536222, 30, 222,  1.0f},
        {"Picea abies",      0x9e2f591cu, 1055308197, 37, 197,  1.0f},
        {"Ranunculus acris", 0x1148f86cu, -899638961, 49, 961, -1.0f},
        {"",                 0x0u,                 0,  0,   0,  1.0f},
    };
    for (const auto& e : table) {
        CAPTURE(e.name);
        REQUIRE(feature_hash(e.name, 0) == e.key);
        REQUIRE(murmur_fmix_i32(static_cast<int64_t>(e.key)) == e.h);
        const HashBucketSign s64 = feature_hash_bucket_sign(e.name, 64);
        const HashBucketSign s1000 = feature_hash_bucket_sign(e.name, 1000);
        REQUIRE(s64.bucket == e.bucket64);
        REQUIRE(s1000.bucket == e.bucket1000);
        REQUIRE(s64.sign == e.sign);
        REQUIRE(s1000.sign == e.sign);
    }
}

TEST_CASE("hash_bucket_sign maps h == INT32_MIN to 2^31 mod hash_dim", "[hash]") {
    const std::vector<int64_t> keys = int32_min_keys();
    REQUIRE(keys.front() == static_cast<int64_t>(0x3838d7e82a5fee3aULL));

    const uint32_t two_pow_31 = 0x80000000u;
    for (int64_t key : keys) {
        CAPTURE(key);
        REQUIRE(murmur_fmix_i32(key) == INT32_MIN);
        for (int32_t hash_dim : {1, 7, 64, 1000, 1024, INT32_MAX}) {
            CAPTURE(hash_dim);
            const HashBucketSign slot = hash_bucket_sign(key, hash_dim);
            REQUIRE(slot.bucket == static_cast<int32_t>(two_pow_31 % static_cast<uint32_t>(hash_dim)));
            REQUIRE(slot.bucket >= 0);
            REQUIRE(slot.bucket < hash_dim);
            REQUIRE(slot.sign == -1.0f);
        }
    }
    REQUIRE(hash_bucket_sign(keys.front(), 1000).bucket == 648);
}

TEST_CASE("hash_bucket_sign agrees with the reference on random strings and keys", "[hash]") {
    std::mt19937_64 rng(20261023);
    const std::vector<int32_t> dims = {1, 3, 64, 256, 1000, 1024, 65537, INT32_MAX};

    for (int i = 0; i < 200000; ++i) {
        const std::string name = random_name(rng);
        const int64_t key = static_cast<int64_t>(feature_hash(name, 0));
        const int32_t hash_dim = dims[static_cast<size_t>(i) % dims.size()];
        const HashBucketSign got = feature_hash_bucket_sign(name, hash_dim);
        const HashBucketSign want = reference_slot(key, hash_dim);
        if (got.bucket != want.bucket || got.sign != want.sign) {
            CAPTURE(i, key, hash_dim, got.bucket, want.bucket, got.sign, want.sign);
            FAIL("string slot differs from the reference");
        }
    }

    for (int i = 0; i < 1000000; ++i) {
        const int64_t key = static_cast<int64_t>(rng());
        const int32_t hash_dim = dims[static_cast<size_t>(i) % dims.size()];
        const HashBucketSign got = hash_bucket_sign(key, hash_dim);
        const HashBucketSign want = reference_slot(key, hash_dim);
        if (got.bucket != want.bucket || got.sign != want.sign) {
            CAPTURE(i, key, hash_dim, got.bucket, want.bucket, got.sign, want.sign);
            FAIL("key slot differs from the reference");
        }
    }

    for (int64_t key : int32_min_keys()) {
        for (int32_t hash_dim : dims) {
            REQUIRE(hash_bucket_sign(key, hash_dim).bucket == reference_slot(key, hash_dim).bucket);
            REQUIRE(hash_bucket_sign(key, hash_dim).sign == reference_slot(key, hash_dim).sign);
        }
    }
}

TEST_CASE("CUDA hash kernel writes the CPU buckets and signs", "[hash][cuda]") {
#ifdef RESOLVE_HAS_CUDA
    if (c10::cuda::device_count() == 0) {
        SKIP("No CUDA device available");
    }

    std::mt19937_64 rng(7);
    std::vector<int64_t> keys = int32_min_keys();
    keys.push_back(0);
    keys.push_back(INT64_MIN);
    keys.push_back(INT64_MAX);
    for (const char* name : {"Quercus robur", "Ranunculus acris", ""}) {
        keys.push_back(static_cast<int64_t>(feature_hash(name, 0)));
    }
    for (int i = 0; i < 4000; ++i) keys.push_back(static_cast<int64_t>(rng()));
    for (int i = 0; i < 4000; ++i) {
        keys.push_back(static_cast<int64_t>(feature_hash(random_name(rng), 0)));
    }
    const int64_t n = static_cast<int64_t>(keys.size());
    const auto gpu = torch::Device(torch::kCUDA);

    SECTION("one record per plot: each row holds exactly that record's slot") {
        auto species = torch::tensor(keys, torch::kInt64);
        auto offsets = torch::arange(n + 1, torch::kInt64);
        auto weights = torch::ones({n}, torch::kFloat32);
        auto batch = torch::arange(n, torch::kInt64).to(gpu);

        for (int32_t hash_dim : {64, 1000, 1024}) {
            CAPTURE(hash_dim);
            auto expected = torch::zeros({n, hash_dim}, torch::kFloat32);
            auto acc = expected.accessor<float, 2>();
            for (int64_t i = 0; i < n; ++i) {
                const HashBucketSign slot = hash_bucket_sign(keys[static_cast<size_t>(i)], hash_dim);
                acc[i][slot.bucket] = slot.sign;
            }
            auto got = cuda::compute_batch_hash_embedding_cuda(
                batch, species, weights, offsets, hash_dim).to(torch::kCPU);
            REQUIRE(torch::equal(got, expected));
        }
    }

    SECTION("several records per plot: rows equal the CPU aggregation") {
        const int64_t n_plots = 97;
        const int32_t hash_dim = 256;
        std::vector<int64_t> offsets_v = {0};
        std::uniform_int_distribution<int64_t> plot_size(0, 2 * n / n_plots);
        while (offsets_v.size() < static_cast<size_t>(n_plots)) {
            offsets_v.push_back(std::min(n, offsets_v.back() + plot_size(rng)));
        }
        offsets_v.push_back(n);

        // Integer-valued weights keep every float sum exact whatever order the
        // kernel's atomics add in.
        std::uniform_int_distribution<int> weight(1, 9);
        std::vector<float> weights_v(static_cast<size_t>(n));
        for (float& w : weights_v) w = static_cast<float>(weight(rng));

        auto expected = torch::zeros({n_plots, hash_dim}, torch::kFloat32);
        auto acc = expected.accessor<float, 2>();
        for (int64_t p = 0; p < n_plots; ++p) {
            for (int64_t r = offsets_v[static_cast<size_t>(p)]; r < offsets_v[static_cast<size_t>(p) + 1]; ++r) {
                const HashBucketSign slot = hash_bucket_sign(keys[static_cast<size_t>(r)], hash_dim);
                acc[p][slot.bucket] += slot.sign * weights_v[static_cast<size_t>(r)];
            }
        }

        auto got = cuda::compute_batch_hash_embedding_cuda(
            torch::arange(n_plots, torch::kInt64).to(gpu),
            torch::tensor(keys, torch::kInt64), torch::tensor(weights_v, torch::kFloat32),
            torch::tensor(offsets_v, torch::kInt64), hash_dim).to(torch::kCPU);
        REQUIRE(torch::equal(got, expected));
    }
#else
    SKIP("Built without CUDA");
#endif
}
