#pragma once

// Native fuzzy-string index for RESOLVE.
//
// Generic Damerau-Levenshtein top-N matcher over a string vocabulary.
// Trie storage plus DP-row state machine (the "Levenshtein automaton")
// walked simultaneously with the trie. Supports an opt-in bucket hint
// (e.g. WFO's genus prefix) that prunes the trie before the full scan.
//
// All character handling is on UTF-32 codepoints decoded from UTF-8 input;
// a substitution between a Latin letter and its accented variant therefore
// costs one edit instead of one-per-byte.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace resolve::fuzzy {

struct Match {
    std::string entry;   // Matched dictionary entry (original casing).
    int id = -1;         // Index into the original `entries` vector passed at build.
    int distance = 0;    // Damerau-Levenshtein distance from needle.
};

struct BuildOptions {
    // Maximum edit distance the index is sized for. Query-time `max_edit_distance`
    // must be <= this value.
    int max_edit_distance = 3;

    // Optional bucket function. If provided, entries are partitioned by
    // bucket_fn(entry) and stored as per-bucket sub-tries plus a global
    // trie. Query-time bucket_hint then prunes to the matching bucket
    // first, falling back to the global trie on a miss.
    std::function<std::string(std::string_view)> bucket_fn = nullptr;

    // Lowercase entries and queries before matching.
    bool case_insensitive = true;

    // Allow adjacent transpositions ("Querucs" <-> "Quercus" at distance 1).
    bool damerau = true;
};

struct QueryOptions {
    int max_edit_distance = 2;            // Must be <= BuildOptions::max_edit_distance.
    int top_n = 5;
    std::optional<std::string> bucket_hint;  // Try this bucket first; auto-fall-back.
};

class FuzzyIndex {
public:
    FuzzyIndex();
    ~FuzzyIndex();
    FuzzyIndex(FuzzyIndex&&) noexcept;
    FuzzyIndex& operator=(FuzzyIndex&&) noexcept;
    FuzzyIndex(const FuzzyIndex&) = delete;
    FuzzyIndex& operator=(const FuzzyIndex&) = delete;

    static FuzzyIndex build(const std::vector<std::string>& entries,
                            BuildOptions opts = {});

    std::vector<Match> query(std::string_view needle,
                             QueryOptions opts = {}) const;

    // OpenMP-parallel batch query. If `bucket_fn` is provided, it is applied
    // per-needle to derive the bucket hint; otherwise opts.bucket_hint is reused.
    std::vector<std::vector<Match>> query_batch(
        const std::vector<std::string>& needles,
        QueryOptions opts = {},
        std::function<std::string(std::string_view)> bucket_fn = nullptr
    ) const;

    std::size_t size() const noexcept;          // Number of indexed entries.
    std::size_t bucket_count() const noexcept;  // Including the global bucket.
    int max_supported_distance() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace resolve::fuzzy
