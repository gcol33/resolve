// Neighbour searches over plots (plot_neighbours.hpp) against brute force: the
// presence-set cosine, the neighbour descent's lists (exact similarities, the
// ranking rule, near-complete recall, the same lists on any thread count) and
// the k-d tree's exact nearest positions with hiding rules.

#include <catch2/catch_test_macros.hpp>

#include "resolve/geo.hpp"
#include "resolve/plot_neighbours.hpp"
#include "resolve/sparse_rows.hpp"

#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

using namespace resolve;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Plots drawn around a few communities: each takes most species of its
// community's pool and a few from anywhere, so neighbours are meaningful and
// many similarities tie.
torch::Tensor community_plots(int64_t n, int64_t n_species, uint32_t seed) {
    std::mt19937 rng(seed);
    const int communities = 12;
    std::vector<std::vector<int64_t>> pools(communities);
    std::uniform_int_distribution<int64_t> any(0, n_species - 1);
    for (auto& pool : pools) {
        for (int j = 0; j < 25; ++j) pool.push_back(any(rng));
    }
    torch::Tensor out = torch::zeros({n, n_species}, torch::kFloat32);
    auto a = out.accessor<float, 2>();
    std::uniform_int_distribution<int> which(0, communities - 1);
    std::uniform_int_distribution<int> pick(0, 24);
    std::uniform_int_distribution<int> size(3, 14);
    for (int64_t i = 0; i < n; ++i) {
        const auto& pool = pools[which(rng)];
        const int m = size(rng);
        for (int j = 0; j < m; ++j) a[i][pool[pick(rng)]] = 1.0f + static_cast<float>(j % 3);
        for (int j = 0; j < 2; ++j) a[i][any(rng)] = 0.5f;
    }
    return out;
}

// Exact cosine of presence between every pair.
torch::Tensor brute_cosine(const torch::Tensor& species) {
    const torch::Tensor p = (species != 0).to(torch::kFloat64);
    const torch::Tensor counts = p.sum(1);
    return p.matmul(p.t()) / torch::sqrt(counts.unsqueeze(1) * counts.unsqueeze(0)).clamp_min(1e-300);
}

// Share of each row's exact top k (among visible candidates of positive
// similarity) that the list matches by score: an entry scoring at least the
// exact k-th score counts, so ties at the k-th score are not misses.
double tie_aware_recall(const torch::Tensor& sim, const SetNeighbours& found, int64_t n_rows,
                        int64_t n_candidates, int64_t k, const std::vector<std::vector<char>>& hidden) {
    const auto s = sim.accessor<double, 2>();
    const auto idx = found.indices.accessor<int64_t, 2>();
    int64_t matched = 0;
    int64_t wanted = 0;
    for (int64_t i = 0; i < n_rows; ++i) {
        std::vector<double> visible;
        for (int64_t j = 0; j < n_candidates; ++j) {
            if (j != i && s[i][j] > 0 && !(hidden.size() && hidden[i][j])) visible.push_back(s[i][j]);
        }
        std::sort(visible.rbegin(), visible.rend());
        const auto expect = static_cast<int64_t>(std::min<std::size_t>(visible.size(), static_cast<std::size_t>(k)));
        if (expect == 0) continue;
        const double kth = visible[static_cast<std::size_t>(expect - 1)];
        int64_t good = 0;
        for (int64_t m = 0; m < k; ++m) {
            if (idx[i][m] >= 0 && s[i][idx[i][m]] >= kth) ++good;
        }
        matched += std::min(good, expect);
        wanted += expect;
    }
    return static_cast<double>(matched) / static_cast<double>(wanted);
}

// Every listed neighbour is a candidate other than the plot, visible, with its
// exact similarity, the list ranked by similarity then row, -1 after it.
void check_lists(const torch::Tensor& sim, const SetNeighbours& found, int64_t n_candidates,
                 const std::vector<std::vector<char>>& hidden) {
    const auto s = sim.accessor<double, 2>();
    const auto idx = found.indices.accessor<int64_t, 2>();
    const auto val = found.similarities.accessor<double, 2>();
    for (int64_t i = 0; i < found.indices.size(0); ++i) {
        bool ended = false;
        for (int64_t m = 0; m < found.indices.size(1); ++m) {
            const int64_t v = idx[i][m];
            if (v < 0) {
                ended = true;
                CHECK(std::isnan(val[i][m]));
                continue;
            }
            REQUIRE_FALSE(ended);
            CHECK(v != i);
            CHECK(v < n_candidates);
            if (hidden.size()) CHECK_FALSE(hidden[i][v]);
            CHECK(std::abs(val[i][m] - s[i][v]) < 1e-12);
            if (m > 0) {
                const int64_t u = idx[i][m - 1];
                CHECK((s[i][u] > s[i][v] + 1e-12 || (std::abs(s[i][u] - s[i][v]) < 1e-12 && u < v)));
            }
        }
    }
}

}  // namespace

TEST_CASE("PresenceSets measures cosine of presence, dense or sparse", "[plot_neighbours]") {
    torch::Tensor species = community_plots(60, 80, 3);
    species[7].zero_();
    const torch::Tensor expected = brute_cosine(species);
    const PresenceSets dense(species);
    const PresenceSets sparse(to_sparse_rows(species));
    const auto e = expected.accessor<double, 2>();
    for (int64_t a = 0; a < 60; ++a) {
        for (int64_t b = 0; b < 60; ++b) {
            CHECK(std::abs(dense.cosine(a, b) - (a == 7 || b == 7 ? 0.0 : e[a][b])) < 1e-12);
            CHECK(dense.cosine(a, b) == sparse.cosine(a, b));
        }
    }
    CHECK(dense.count(7) == 0);
}

TEST_CASE("The neighbour descent finds the nearest candidates by species", "[plot_neighbours]") {
    const int64_t n = 700;
    const int64_t n_candidates = 560;
    const int64_t k = 12;
    const torch::Tensor species = community_plots(n, 300, 11);
    const torch::Tensor sim = brute_cosine(species);
    const PresenceSets sets(species);
    SetGraphConfig config;
    config.k = k;
    SetNeighbourGraph graph(sets, n_candidates, config);
    const SetNeighbours open = graph.open();
    REQUIRE(open.indices.sizes() == torch::IntArrayRef({n, k}));
    check_lists(sim, open, n_candidates, {});
    CHECK(tie_aware_recall(sim, open, n, n_candidates, k, {}) > 0.98);

    // Hiding: each plot skips the plots more similar than its radius.
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> radius(0.3, 0.9);
    std::vector<double> radii(static_cast<std::size_t>(n_candidates));
    for (auto& r : radii) r = radius(rng);
    const auto s = sim.accessor<double, 2>();
    std::vector<std::vector<char>> hidden(static_cast<std::size_t>(n_candidates),
                                          std::vector<char>(static_cast<std::size_t>(n), 0));
    for (int64_t i = 0; i < n_candidates; ++i) {
        for (int64_t j = 0; j < n; ++j) hidden[i][j] = s[i][j] > radii[i] + 1e-12 ? 1 : 0;
    }
    const SetNeighbours visible = graph.hidden(n_candidates, [&](int64_t u, int64_t, double similarity) {
        return similarity > radii[static_cast<std::size_t>(u)];
    });
    check_lists(sim, visible, n_candidates, hidden);
    CHECK(tie_aware_recall(sim, visible, n_candidates, n_candidates, k, hidden) > 0.98);
}

TEST_CASE("The neighbour descent gives the same lists on any thread count", "[plot_neighbours]") {
    const torch::Tensor species = community_plots(900, 400, 23);
    const PresenceSets sets(species);
    SetGraphConfig config;
    config.k = 10;
    auto run = [&](int threads) {
        const int before = at::get_num_threads();
        at::set_num_threads(threads);
        SetNeighbourGraph graph(sets, 800, config);
        const SetNeighbours open = graph.open();
        const SetNeighbours visible =
            graph.hidden(800, [](int64_t u, int64_t, double similarity) { return similarity > 0.4 + (u % 5) * 0.1; });
        at::set_num_threads(before);
        return std::make_pair(open.indices, visible.indices);
    };
    const auto one = run(1);
    const auto many = run(8);
    CHECK(torch::equal(one.first, many.first));
    CHECK(torch::equal(one.second, many.second));
}

TEST_CASE("PlaceIndex is the exact nearest search, with hiding and ties", "[plot_neighbours]") {
    const int64_t n = 600;
    const int64_t n_candidates = 450;
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> lat(44.0, 49.0);
    std::uniform_real_distribution<double> lon(5.0, 15.0);
    torch::Tensor coords = torch::empty({n, 2}, torch::kFloat64);
    auto c = coords.accessor<double, 2>();
    for (int64_t i = 0; i < n; ++i) {
        c[i][0] = lat(rng);
        c[i][1] = lon(rng);
    }
    // Shared positions (ties) and positions without coordinates.
    for (int64_t i = 0; i < 40; ++i) {
        c[i + 100][0] = c[i][0];
        c[i + 100][1] = c[i][1];
    }
    for (int64_t i = 300; i < 320; ++i) c[i][0] = kNaN;

    std::vector<double> radius(static_cast<std::size_t>(n));
    std::uniform_real_distribution<double> r(0.0, 60.0);
    for (auto& x : radius) x = r(rng);
    auto hides = [&](int64_t u, int64_t v, double km) {
        return km < radius[static_cast<std::size_t>(u)] || (u + v) % 7 == 0;
    };
    std::vector<int64_t> queries(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) queries[static_cast<std::size_t>(i)] = i;

    const int64_t k = 9;
    const PlaceIndex index(coords, n_candidates);
    for (const bool hiding : {false, true}) {
        const PlaceNeighbours found =
            hiding ? index.nearest(queries, k, hides, radius) : index.nearest(queries, k);
        const auto idx = found.indices.accessor<int64_t, 2>();
        const auto km = found.km.accessor<double, 2>();
        for (int64_t q = 0; q < n; ++q) {
            std::vector<std::pair<double, int64_t>> all;
            if (std::isfinite(c[q][0])) {
                for (int64_t v = 0; v < n_candidates; ++v) {
                    if (v == q || !std::isfinite(c[v][0])) continue;
                    const double d = great_circle_km(c[q][0], c[q][1], c[v][0], c[v][1]);
                    if (hiding && hides(q, v, d)) continue;
                    all.emplace_back(d, v);
                }
            }
            std::sort(all.begin(), all.end());
            for (int64_t m = 0; m < k; ++m) {
                if (m < static_cast<int64_t>(all.size())) {
                    CHECK(idx[q][m] == all[static_cast<std::size_t>(m)].second);
                    CHECK(km[q][m] == all[static_cast<std::size_t>(m)].first);
                } else {
                    CHECK(idx[q][m] == -1);
                    CHECK(std::isnan(km[q][m]));
                }
            }
        }
    }
}

TEST_CASE("A plot sharing too few species lists the lowest rows sharing none at similarity 0",
          "[plot_neighbours]") {
    torch::Tensor species = community_plots(200, 150, 31);
    species[4].zero_();
    const PresenceSets sets(species);
    SetGraphConfig config;
    config.k = 6;
    SetNeighbourGraph graph(sets, 150, config);
    const SetNeighbours open = graph.open();
    const std::vector<int64_t> expected = {0, 1, 2, 3, 5, 6};
    for (int64_t m = 0; m < 6; ++m) {
        CHECK(open.indices[4][m].item<int64_t>() == expected[static_cast<std::size_t>(m)]);
        CHECK(open.similarities[4][m].item<double>() == 0.0);
    }
    const SetNeighbours visible =
        graph.hidden(150, [](int64_t u, int64_t v, double) { return u == 4 && v < 3; });
    const std::vector<int64_t> beyond = {3, 5, 6, 7, 8, 9};
    for (int64_t m = 0; m < 6; ++m) CHECK(visible.indices[4][m].item<int64_t>() == beyond[static_cast<std::size_t>(m)]);
}
