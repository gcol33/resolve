// Tests for resolve::fuzzy::FuzzyIndex.
//
// Coverage:
//   - Basic single-edit cases (insert / delete / substitute / transpose)
//   - UTF-8 / multi-byte entries
//   - Brute-force DP cross-check on random vocab
//   - Bucket-hint hit + bucket-hint miss with auto fallback to the global trie
//   - case_insensitive round-trip
//   - damerau=true vs damerau=false distinguishes "abc" <-> "bac"
//   - Recovery test: noisy queries recall the true target in top-5

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/fuzzy.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

// --------------------------------------------------------------------------
// Brute-force Damerau-Levenshtein for cross-checking.
// --------------------------------------------------------------------------
int damerau_dp(const std::string& a, const std::string& b) {
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());
    std::vector<std::vector<int>> d(static_cast<std::size_t>(n + 1),
                                    std::vector<int>(static_cast<std::size_t>(m + 1), 0));
    for (int i = 0; i <= n; ++i) d[static_cast<std::size_t>(i)][0] = i;
    for (int j = 0; j <= m; ++j) d[0][static_cast<std::size_t>(j)] = j;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            int cost = (a[static_cast<std::size_t>(i - 1)] == b[static_cast<std::size_t>(j - 1)]) ? 0 : 1;
            int v = d[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(j - 1)] + cost;
            v = std::min(v, d[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(j)] + 1);
            v = std::min(v, d[static_cast<std::size_t>(i)][static_cast<std::size_t>(j - 1)] + 1);
            if (i >= 2 && j >= 2 &&
                a[static_cast<std::size_t>(i - 1)] == b[static_cast<std::size_t>(j - 2)] &&
                a[static_cast<std::size_t>(i - 2)] == b[static_cast<std::size_t>(j - 1)])
            {
                v = std::min(v, d[static_cast<std::size_t>(i - 2)][static_cast<std::size_t>(j - 2)] + 1);
            }
            d[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = v;
        }
    }
    return d[static_cast<std::size_t>(n)][static_cast<std::size_t>(m)];
}

int levenshtein_dp(const std::string& a, const std::string& b) {
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());
    std::vector<std::vector<int>> d(static_cast<std::size_t>(n + 1),
                                    std::vector<int>(static_cast<std::size_t>(m + 1), 0));
    for (int i = 0; i <= n; ++i) d[static_cast<std::size_t>(i)][0] = i;
    for (int j = 0; j <= m; ++j) d[0][static_cast<std::size_t>(j)] = j;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            int cost = (a[static_cast<std::size_t>(i - 1)] == b[static_cast<std::size_t>(j - 1)]) ? 0 : 1;
            int v = d[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(j - 1)] + cost;
            v = std::min(v, d[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(j)] + 1);
            v = std::min(v, d[static_cast<std::size_t>(i)][static_cast<std::size_t>(j - 1)] + 1);
            d[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = v;
        }
    }
    return d[static_cast<std::size_t>(n)][static_cast<std::size_t>(m)];
}

std::string random_word(std::mt19937& rng, int len) {
    std::uniform_int_distribution<int> char_dist('a', 'z');
    std::string s;
    s.reserve(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) s.push_back(static_cast<char>(char_dist(rng)));
    return s;
}

// Apply k random edits (insert/delete/substitute/transpose) to `s`.
std::string add_noise(std::mt19937& rng, std::string s, int k) {
    std::uniform_int_distribution<int> edit_kind(0, 3);
    std::uniform_int_distribution<int> char_dist('a', 'z');
    for (int e = 0; e < k; ++e) {
        if (s.empty()) {
            s.push_back(static_cast<char>(char_dist(rng)));
            continue;
        }
        int kind = edit_kind(rng);
        std::uniform_int_distribution<int> pos_dist(0, static_cast<int>(s.size()) - 1);
        int p = pos_dist(rng);
        if (kind == 0) {                                          // insert
            s.insert(static_cast<std::size_t>(p), 1, static_cast<char>(char_dist(rng)));
        } else if (kind == 1) {                                   // delete
            s.erase(static_cast<std::size_t>(p), 1);
        } else if (kind == 2) {                                   // substitute
            s[static_cast<std::size_t>(p)] = static_cast<char>(char_dist(rng));
        } else if (s.size() >= 2) {                               // transpose
            if (p == static_cast<int>(s.size()) - 1) --p;
            std::swap(s[static_cast<std::size_t>(p)],
                      s[static_cast<std::size_t>(p + 1)]);
        } else {
            s.push_back(static_cast<char>(char_dist(rng)));
        }
    }
    return s;
}

} // namespace

// --------------------------------------------------------------------------
// Basic correctness
// --------------------------------------------------------------------------

TEST_CASE("FuzzyIndex: exact match returns distance 0", "[fuzzy]") {
    std::vector<std::string> entries{"apple", "banana", "cherry"};
    auto idx = resolve::fuzzy::FuzzyIndex::build(entries);
    REQUIRE(idx.size() == 3);

    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 2;
    q.top_n = 5;
    auto out = idx.query("apple", q);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out[0].entry == "apple");
    REQUIRE(out[0].distance == 0);
}

TEST_CASE("FuzzyIndex: single substitution distance 1", "[fuzzy]") {
    auto idx = resolve::fuzzy::FuzzyIndex::build({"apple", "table"});
    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 2;
    auto out = idx.query("apxle", q);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out[0].entry == "apple");
    REQUIRE(out[0].distance == 1);
}

TEST_CASE("FuzzyIndex: single insertion distance 1", "[fuzzy]") {
    auto idx = resolve::fuzzy::FuzzyIndex::build({"apple"});
    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 2;
    auto out = idx.query("appple", q);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out[0].entry == "apple");
    REQUIRE(out[0].distance == 1);
}

TEST_CASE("FuzzyIndex: single deletion distance 1", "[fuzzy]") {
    auto idx = resolve::fuzzy::FuzzyIndex::build({"apple"});
    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 2;
    auto out = idx.query("aple", q);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out[0].entry == "apple");
    REQUIRE(out[0].distance == 1);
}

TEST_CASE("FuzzyIndex: transposition is distance 1 with damerau=true", "[fuzzy][damerau]") {
    resolve::fuzzy::BuildOptions opts;
    opts.damerau = true;
    auto idx = resolve::fuzzy::FuzzyIndex::build({"abc"}, opts);

    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 2;
    auto out = idx.query("bac", q);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out[0].distance == 1);
}

TEST_CASE("FuzzyIndex: transposition is distance 2 with damerau=false", "[fuzzy][damerau]") {
    resolve::fuzzy::BuildOptions opts;
    opts.damerau = false;
    auto idx = resolve::fuzzy::FuzzyIndex::build({"abc"}, opts);

    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 2;
    auto out = idx.query("bac", q);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out[0].distance == 2);
}

TEST_CASE("FuzzyIndex: empty needle, empty entry", "[fuzzy][edge]") {
    auto idx = resolve::fuzzy::FuzzyIndex::build({"", "a", "ab"});
    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 2;
    q.top_n = 10;

    auto out = idx.query("", q);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out[0].entry == "");
    REQUIRE(out[0].distance == 0);
}

TEST_CASE("FuzzyIndex: case-insensitive matching", "[fuzzy][ci]") {
    resolve::fuzzy::BuildOptions opts;
    opts.case_insensitive = true;
    auto idx = resolve::fuzzy::FuzzyIndex::build({"Quercus robur", "Pinus sylvestris"}, opts);

    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 0;
    auto out = idx.query("QUERCUS ROBUR", q);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out[0].entry == "Quercus robur");
    REQUIRE(out[0].distance == 0);
}

TEST_CASE("FuzzyIndex: UTF-8 codepoint matching (one cp, not byte)", "[fuzzy][utf8]") {
    // U+00E9 (e-acute) is two UTF-8 bytes. Substituting it for 'e' should be
    // distance 1, not 2.
    auto idx = resolve::fuzzy::FuzzyIndex::build({"r\xC3\xA9sumé"});
    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 1;
    auto out = idx.query("resumé", q);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out[0].distance == 1);
}

// --------------------------------------------------------------------------
// Cross-check against brute force DP
// --------------------------------------------------------------------------

TEST_CASE("FuzzyIndex: agrees with brute-force DP on random vocab (Damerau)", "[fuzzy][crosscheck]") {
    std::mt19937 rng(123);
    const int vocab_size = 500;
    const int n_queries = 100;
    const int max_k = 2;

    std::vector<std::string> entries;
    entries.reserve(static_cast<std::size_t>(vocab_size));
    std::unordered_set<std::string> seen;
    while (static_cast<int>(entries.size()) < vocab_size) {
        std::uniform_int_distribution<int> len_dist(3, 10);
        std::string w = random_word(rng, len_dist(rng));
        if (seen.insert(w).second) entries.push_back(w);
    }

    resolve::fuzzy::BuildOptions bopts;
    bopts.damerau = true;
    bopts.max_edit_distance = max_k;
    auto idx = resolve::fuzzy::FuzzyIndex::build(entries, bopts);

    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = max_k;
    q.top_n = vocab_size;  // collect all candidates within k

    for (int qi = 0; qi < n_queries; ++qi) {
        std::uniform_int_distribution<int> qlen(3, 10);
        std::string needle = random_word(rng, qlen(rng));

        // Brute force: enumerate every entry, keep those with distance <= k.
        std::vector<std::pair<int, int>> truth;  // (distance, id)
        for (std::size_t i = 0; i < entries.size(); ++i) {
            int d = damerau_dp(needle, entries[i]);
            if (d <= max_k) truth.emplace_back(d, static_cast<int>(i));
        }
        std::sort(truth.begin(), truth.end());

        auto got = idx.query(needle, q);

        REQUIRE(got.size() == truth.size());
        for (std::size_t i = 0; i < got.size(); ++i) {
            REQUIRE(got[i].distance == truth[i].first);
            // Entry IDs may differ when distances tie. Check the multiset of
            // (distance, entry) pairs instead of strict ordering.
        }

        std::vector<std::pair<int, std::string>> got_set, truth_set;
        for (const auto& m : got) got_set.emplace_back(m.distance, m.entry);
        for (const auto& t : truth) truth_set.emplace_back(t.first, entries[static_cast<std::size_t>(t.second)]);
        std::sort(got_set.begin(), got_set.end());
        std::sort(truth_set.begin(), truth_set.end());
        REQUIRE(got_set == truth_set);
    }
}

TEST_CASE("FuzzyIndex: agrees with brute-force DP (plain Levenshtein)", "[fuzzy][crosscheck]") {
    std::mt19937 rng(456);
    const int vocab_size = 300;
    const int n_queries = 60;
    const int max_k = 2;

    std::vector<std::string> entries;
    std::unordered_set<std::string> seen;
    while (static_cast<int>(entries.size()) < vocab_size) {
        std::uniform_int_distribution<int> len_dist(4, 9);
        std::string w = random_word(rng, len_dist(rng));
        if (seen.insert(w).second) entries.push_back(w);
    }

    resolve::fuzzy::BuildOptions bopts;
    bopts.damerau = false;
    bopts.max_edit_distance = max_k;
    auto idx = resolve::fuzzy::FuzzyIndex::build(entries, bopts);

    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = max_k;
    q.top_n = vocab_size;

    for (int qi = 0; qi < n_queries; ++qi) {
        std::uniform_int_distribution<int> qlen(4, 9);
        std::string needle = random_word(rng, qlen(rng));

        std::vector<std::pair<int, std::string>> truth_set;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            int d = levenshtein_dp(needle, entries[i]);
            if (d <= max_k) truth_set.emplace_back(d, entries[i]);
        }
        std::sort(truth_set.begin(), truth_set.end());

        std::vector<std::pair<int, std::string>> got_set;
        for (const auto& m : idx.query(needle, q)) got_set.emplace_back(m.distance, m.entry);
        std::sort(got_set.begin(), got_set.end());
        REQUIRE(got_set == truth_set);
    }
}

// --------------------------------------------------------------------------
// Bucket hint
// --------------------------------------------------------------------------

TEST_CASE("FuzzyIndex: bucket hint hits the right bucket", "[fuzzy][bucket]") {
    std::vector<std::string> entries{
        "Quercus robur", "Quercus alba", "Quercus rubra",
        "Pinus sylvestris", "Pinus pinea",
        "Acer rubrum", "Acer saccharum",
    };
    resolve::fuzzy::BuildOptions opts;
    opts.bucket_fn = [](std::string_view s) {
        auto sp = s.find(' ');
        return sp == std::string_view::npos ? std::string(s) : std::string(s.substr(0, sp));
    };
    auto idx = resolve::fuzzy::FuzzyIndex::build(entries, opts);

    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 2;
    q.top_n = 5;
    q.bucket_hint = std::string("Quercus");

    auto out = idx.query("Quercus rubra", q);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out[0].entry == "Quercus rubra");
    REQUIRE(out[0].distance == 0);

    // Every returned candidate should be in the Quercus bucket.
    for (const auto& m : out) {
        REQUIRE(m.entry.rfind("Quercus", 0) == 0);
    }
}

TEST_CASE("FuzzyIndex: bucket miss falls back to global trie", "[fuzzy][bucket]") {
    // Mistyped genus: "Querucs" instead of "Quercus". Bucket lookup misses,
    // but the global trie should still find "Quercus robur" at distance 1.
    std::vector<std::string> entries{
        "Quercus robur", "Quercus alba",
        "Pinus sylvestris",
    };
    resolve::fuzzy::BuildOptions opts;
    opts.bucket_fn = [](std::string_view s) {
        auto sp = s.find(' ');
        return sp == std::string_view::npos ? std::string(s) : std::string(s.substr(0, sp));
    };
    opts.damerau = true;
    auto idx = resolve::fuzzy::FuzzyIndex::build(entries, opts);

    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 2;
    q.top_n = 5;
    q.bucket_hint = std::string("Querucs");  // wrong bucket key

    auto out = idx.query("Querucs robur", q);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out[0].entry == "Quercus robur");
    REQUIRE(out[0].distance == 1);
}

// --------------------------------------------------------------------------
// Recovery test: noisy queries recall the true target
// --------------------------------------------------------------------------

TEST_CASE("FuzzyIndex: noisy queries recall the true target in top-5", "[fuzzy][recovery]") {
    std::mt19937 rng(789);
    const int n_targets = 100;
    const int queries_per_target = 5;
    const int max_k = 3;

    std::vector<std::string> entries;
    std::unordered_set<std::string> seen;
    while (static_cast<int>(entries.size()) < n_targets) {
        std::uniform_int_distribution<int> len_dist(8, 15);
        std::string w = random_word(rng, len_dist(rng));
        if (seen.insert(w).second) entries.push_back(w);
    }

    // Sprinkle in distractors so the index is non-trivial.
    while (static_cast<int>(entries.size()) < n_targets + 400) {
        std::uniform_int_distribution<int> len_dist(8, 15);
        std::string w = random_word(rng, len_dist(rng));
        if (seen.insert(w).second) entries.push_back(w);
    }

    resolve::fuzzy::BuildOptions bopts;
    bopts.damerau = true;
    bopts.max_edit_distance = max_k;
    auto idx = resolve::fuzzy::FuzzyIndex::build(entries, bopts);

    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = max_k;
    q.top_n = 5;

    int hits = 0, total = 0;
    std::uniform_int_distribution<int> nk(1, max_k);
    for (int t = 0; t < n_targets; ++t) {
        const std::string& target = entries[static_cast<std::size_t>(t)];
        for (int j = 0; j < queries_per_target; ++j) {
            std::string noisy = add_noise(rng, target, nk(rng));
            auto out = idx.query(noisy, q);
            ++total;
            bool found = false;
            for (const auto& m : out) {
                if (m.entry == target) { found = true; break; }
            }
            if (found) ++hits;
        }
    }

    double recall = static_cast<double>(hits) / static_cast<double>(total);
    INFO("recovery recall = " << recall << " (" << hits << "/" << total << ")");
    REQUIRE(recall >= 0.95);
}

// --------------------------------------------------------------------------
// Batch
// --------------------------------------------------------------------------

TEST_CASE("FuzzyIndex: query_batch matches serial query", "[fuzzy][batch]") {
    std::mt19937 rng(321);
    std::vector<std::string> entries;
    std::unordered_set<std::string> seen;
    while (entries.size() < 200) {
        std::uniform_int_distribution<int> len_dist(5, 12);
        std::string w = random_word(rng, len_dist(rng));
        if (seen.insert(w).second) entries.push_back(w);
    }

    resolve::fuzzy::BuildOptions bopts;
    bopts.damerau = true;
    bopts.max_edit_distance = 2;
    auto idx = resolve::fuzzy::FuzzyIndex::build(entries, bopts);

    std::vector<std::string> needles;
    for (int i = 0; i < 50; ++i) {
        const auto& target = entries[static_cast<std::size_t>(i)];
        needles.push_back(add_noise(rng, target, 1));
    }

    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = 2;
    q.top_n = 3;

    auto batch = idx.query_batch(needles, q);
    REQUIRE(batch.size() == needles.size());
    for (std::size_t i = 0; i < needles.size(); ++i) {
        auto serial = idx.query(needles[i], q);
        REQUIRE(batch[i].size() == serial.size());
        for (std::size_t k = 0; k < serial.size(); ++k) {
            REQUIRE(batch[i][k].distance == serial[k].distance);
            REQUIRE(batch[i][k].entry == serial[k].entry);
        }
    }
}

// Issue #69: exercise the adaptive-k tightening path (small top_n) and verify
// the returned matches are exactly the closest top-N by distance, not merely a
// high-recall subset (the other cross-checks use top_n = vocab_size, which never
// tightens effective_k).
TEST_CASE("FuzzyIndex: small top_n returns the exact closest-N by distance",
          "[fuzzy][crosscheck]") {
    std::mt19937 rng(777);
    const int vocab_size = 400;
    const int n_queries = 60;
    const int max_k = 3;
    const int top_n = 5;

    std::vector<std::string> entries;
    std::unordered_set<std::string> seen;
    while (static_cast<int>(entries.size()) < vocab_size) {
        std::uniform_int_distribution<int> len_dist(3, 10);
        std::string w = random_word(rng, len_dist(rng));
        if (seen.insert(w).second) entries.push_back(w);
    }

    resolve::fuzzy::BuildOptions bopts;
    bopts.damerau = true;
    bopts.max_edit_distance = max_k;
    auto idx = resolve::fuzzy::FuzzyIndex::build(entries, bopts);

    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = max_k;
    q.top_n = top_n;

    for (int qi = 0; qi < n_queries; ++qi) {
        std::uniform_int_distribution<int> qlen(3, 10);
        std::string needle = random_word(rng, qlen(rng));

        std::vector<int> truth_dists;
        for (const auto& e : entries) {
            int d = damerau_dp(needle, e);
            if (d <= max_k) truth_dists.push_back(d);
        }
        std::sort(truth_dists.begin(), truth_dists.end());

        auto got = idx.query(needle, q);
        const std::size_t expect =
            std::min<std::size_t>(static_cast<std::size_t>(top_n), truth_dists.size());
        REQUIRE(got.size() == expect);
        // got must be sorted ascending and hold exactly the `expect` smallest
        // distances (ties at the boundary may pick different entries, so we
        // compare the distance profile, which is what "closest N" pins down).
        for (std::size_t i = 0; i < expect; ++i) {
            REQUIRE(got[i].distance == truth_dists[i]);
        }
    }
}
