#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "resolve/trainer.hpp"

using namespace resolve;
using namespace Catch::Matchers;

// ============================================================================
// SpatialBlockSplitter Tests
// ============================================================================

TEST_CASE("SpatialBlockSplitter basic splitting", "[spatial_cv]") {
    SpatialBlockSplitter splitter(/*lat_size=*/1.0f, /*lon_size=*/1.0f,
                                   /*n_splits=*/3, /*seed=*/42);

    // 100 random coordinates
    auto coords = torch::rand({100, 2}) * 10.0;  // 0-10 degrees
    auto folds = splitter.split(coords);

    REQUIRE(folds.size() == 3);

    // All indices covered, no overlap between train and test per fold
    for (int f = 0; f < 3; ++f) {
        auto& [train, test] = folds[f];
        REQUIRE(train.size() + test.size() == 100);

        // No overlap
        std::set<int64_t> train_set(train.begin(), train.end());
        for (auto idx : test) {
            REQUIRE(train_set.count(idx) == 0);
        }
    }
}

TEST_CASE("SpatialBlockSplitter all indices appear in test exactly once across folds", "[spatial_cv]") {
    SpatialBlockSplitter splitter(1.0f, 1.0f, 5, 42);

    auto coords = torch::rand({50, 2}) * 5.0;
    auto folds = splitter.split(coords);

    std::vector<int> test_count(50, 0);
    for (auto& [train, test] : folds) {
        for (auto idx : test) {
            test_count[idx]++;
        }
    }

    // Each plot should appear in exactly one test fold
    for (int i = 0; i < 50; ++i) {
        REQUIRE(test_count[i] == 1);
    }
}

TEST_CASE("SpatialBlockSplitter balance mode produces similar fold sizes", "[spatial_cv]") {
    SpatialBlockSplitter splitter(0.5f, 0.5f, 3, 42, /*balance=*/true);

    auto coords = torch::rand({300, 2}) * 10.0;
    auto folds = splitter.split(coords);

    // With balanced splitting, fold sizes should be within 30% of mean
    float mean_size = 100.0f;  // 300 / 3
    for (auto& [train, test] : folds) {
        float test_size = static_cast<float>(test.size());
        REQUIRE(test_size > mean_size * 0.3f);
        REQUIRE(test_size < mean_size * 2.0f);
    }
}

TEST_CASE("SpatialBlockSplitter deterministic with same seed", "[spatial_cv]") {
    auto coords = torch::rand({100, 2}) * 10.0;

    SpatialBlockSplitter s1(1.0f, 1.0f, 3, 42);
    SpatialBlockSplitter s2(1.0f, 1.0f, 3, 42);

    auto folds1 = s1.split(coords);
    auto folds2 = s2.split(coords);

    for (int f = 0; f < 3; ++f) {
        REQUIRE(folds1[f].first == folds2[f].first);
        REQUIRE(folds1[f].second == folds2[f].second);
    }
}

TEST_CASE("SpatialBlockSplitter spatial coherence", "[spatial_cv]") {
    // Points at the same location should be in the same fold
    SpatialBlockSplitter splitter(1.0f, 1.0f, 3, 42);

    // 10 points at (5.5, 5.5), 10 at (0.5, 0.5)
    auto coords = torch::zeros({20, 2});
    coords.index_put_({torch::indexing::Slice(0, 10), torch::indexing::Slice()}, 5.5f);
    coords.index_put_({torch::indexing::Slice(10, 20), torch::indexing::Slice()}, 0.5f);

    auto folds = splitter.split(coords);

    // All points in same block should be in same test fold
    for (auto& [train, test] : folds) {
        std::set<int64_t> test_set(test.begin(), test.end());
        // If any of first group is in test, all should be
        bool first_in_test = test_set.count(0) > 0;
        if (first_in_test) {
            for (int i = 0; i < 10; ++i) {
                REQUIRE(test_set.count(i) == 1);
            }
        }
        // Same for second group
        bool second_in_test = test_set.count(10) > 0;
        if (second_in_test) {
            for (int i = 10; i < 20; ++i) {
                REQUIRE(test_set.count(i) == 1);
            }
        }
    }
}

TEST_CASE("SpatialBlockSplitter rejects invalid params", "[spatial_cv]") {
    REQUIRE_THROWS(SpatialBlockSplitter(1.0f, 1.0f, 1, 42));   // n_splits < 2
    REQUIRE_THROWS(SpatialBlockSplitter(-1.0f, 1.0f, 3, 42));  // negative size
    REQUIRE_THROWS(SpatialBlockSplitter(1.0f, 0.0f, 3, 42));   // zero size
}
