#pragma once

// Internal helpers shared by fuzzy_automaton.cpp / fuzzy_index.cpp / fuzzy_search.cpp.
// Not part of the public include/resolve/fuzzy.hpp surface.

#include "resolve/fuzzy.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace resolve::fuzzy::detail {

// Decode a UTF-8 string into a vector of Unicode code points.
// Malformed sequences yield U+FFFD and resync.
std::vector<uint32_t> utf8_to_codepoints(std::string_view s);

// Lowercase a single code point (ASCII A-Z + Latin-1 uppercase C0-DE, excl. D7).
uint32_t to_lower_cp(uint32_t cp);

// Lowercase all code points in-place.
void to_lower_inplace(std::vector<uint32_t>& cps);

// Damerau-Levenshtein automaton driven by row-state DP.
//
// State = (parent_row, grandparent_row, parent_char) where the rows are the
// DP columns at depths d and d-1 of the trie walk. step() advances by one
// character; accepts() reports the distance at the current depth.
class LevenshteinAutomaton {
public:
    LevenshteinAutomaton(std::vector<uint32_t> needle, int max_k, bool damerau);

    int needle_length() const noexcept { return m_; }
    int max_k() const noexcept { return max_k_; }
    bool damerau() const noexcept { return damerau_; }
    const std::vector<uint32_t>& needle() const noexcept { return needle_; }

    // Initialize the row for depth 0 (no candidate chars consumed yet).
    void initial_row(std::vector<int>& row) const;

    // Advance the automaton by one character.
    //   parent_row       : row at depth d
    //   grandparent_row  : row at depth d-1 (nullptr at d=0)
    //   parent_char      : char consumed at depth d (ignored if grandparent null)
    //   c                : new char being consumed (depth d+1)
    //   effective_k      : current upper bound on edits to keep alive
    //   new_row          : output, resized to m+1
    // Returns true if min(new_row) <= effective_k (subtree alive).
    bool step(const std::vector<int>& parent_row,
              const std::vector<int>* grandparent_row,
              uint32_t parent_char,
              uint32_t c,
              int effective_k,
              std::vector<int>& new_row) const;

    // True iff row[m] <= max_k. Distance written to `distance` when true.
    bool accepts(const std::vector<int>& row, int& distance) const;

private:
    std::vector<uint32_t> needle_;
    int max_k_;
    bool damerau_;
    int m_;
};

// Compact trie. Nodes packed into a single vector for cache friendliness.
// Children of a node are a singly linked sibling list rooted at first_child.
// During build we use a child map for O(1) lookup; after build it's frozen.
struct TrieNode {
    uint32_t c = 0;          // Code point on the edge into this node.
    int32_t first_child = -1;
    int32_t next_sibling = -1;
    int32_t entry_id = -1;   // >= 0 if this node terminates an indexed entry.
};

class Trie {
public:
    Trie();
    void insert(const std::vector<uint32_t>& cps, int entry_id);
    void finalize();

    int root() const noexcept { return 0; }
    const TrieNode& node(int idx) const { return nodes_[idx]; }
    std::size_t node_count() const noexcept { return nodes_.size(); }

private:
    int find_or_create_child(int parent, uint32_t c);

    std::vector<TrieNode> nodes_;
    // Build-time child maps: nodes_[i]'s children mapped by codepoint -> node index.
    // Cleared in finalize() to reclaim memory.
    std::vector<std::vector<std::pair<uint32_t, int32_t>>> child_lists_;
    bool finalized_ = false;
};

} // namespace resolve::fuzzy::detail

namespace resolve::fuzzy {

// Implementation detail of FuzzyIndex, declared here so that both
// fuzzy_index.cpp (build) and fuzzy_search.cpp (query) see the same layout.
struct FuzzyIndex::Impl {
    BuildOptions opts;
    int max_supported_k = 0;
    std::vector<std::string> entries;
    detail::Trie global_trie;
    std::unordered_map<std::string, detail::Trie> buckets;
};

} // namespace resolve::fuzzy
