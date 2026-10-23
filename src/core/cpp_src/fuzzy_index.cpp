// FuzzyIndex: trie storage and build pipeline.
// Query/search logic lives in fuzzy_search.cpp.

#include "resolve/fuzzy.hpp"
#include "fuzzy_internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace resolve::fuzzy {

namespace detail {

// ---------------------------------------------------------------------------
// Trie
// ---------------------------------------------------------------------------

Trie::Trie() {
    nodes_.emplace_back();  // root node, index 0
    child_lists_.emplace_back();
}

int Trie::find_or_create_child(int parent, uint32_t c) {
    {
        const auto& list = child_lists_[static_cast<std::size_t>(parent)];
        for (const auto& kv : list) {
            if (kv.first == c) return kv.second;
        }
    }
    int new_idx = static_cast<int>(nodes_.size());
    TrieNode n;
    n.c = c;
    nodes_.push_back(n);
    // child_lists_ grows by one to mirror nodes_; this may reallocate, so the
    // earlier read-only `list` reference must not be used past this point.
    child_lists_.emplace_back();
    child_lists_[static_cast<std::size_t>(parent)].emplace_back(c, new_idx);
    return new_idx;
}

void Trie::insert(const std::vector<uint32_t>& cps, int entry_id) {
    int cur = 0;
    for (uint32_t c : cps) {
        cur = find_or_create_child(cur, c);
    }
    // Every entry ending here is kept, so entries that fold to one key are all
    // returned by a query that reaches the node.
    terminals_.emplace_back(cur, entry_id);
}

void Trie::finalize() {
    if (finalized_) return;
    // Build sibling chains in alphabetic order for deterministic traversal.
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        auto& list = child_lists_[i];
        std::sort(list.begin(), list.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        int32_t prev = -1;
        for (auto it = list.rbegin(); it != list.rend(); ++it) {
            // Set next_sibling for each child to point to the previously seen
            // (alphabetically later) sibling.
            // We walk in reverse so the chain runs in alphabetic order.
            nodes_[static_cast<std::size_t>(it->second)].next_sibling = prev;
            prev = it->second;
        }
        if (!list.empty()) {
            nodes_[i].first_child = list.front().second;
        }
    }
    child_lists_.clear();
    child_lists_.shrink_to_fit();

    std::sort(terminals_.begin(), terminals_.end());
    entry_ids_.reserve(terminals_.size());
    for (const auto& [node, entry] : terminals_) {
        auto& n = nodes_[static_cast<std::size_t>(node)];
        if (n.n_entries == 0) n.first_entry = static_cast<int32_t>(entry_ids_.size());
        ++n.n_entries;
        entry_ids_.push_back(entry);
    }
    terminals_.clear();
    terminals_.shrink_to_fit();
    finalized_ = true;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public ctor / dtor / moves
// ---------------------------------------------------------------------------

FuzzyIndex::FuzzyIndex() : impl_(std::make_unique<Impl>()) {}
FuzzyIndex::~FuzzyIndex() = default;
FuzzyIndex::FuzzyIndex(FuzzyIndex&&) noexcept = default;
FuzzyIndex& FuzzyIndex::operator=(FuzzyIndex&&) noexcept = default;

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------

FuzzyIndex FuzzyIndex::build(const std::vector<std::string>& entries,
                             BuildOptions opts)
{
    if (opts.max_edit_distance < 0) {
        throw std::invalid_argument("FuzzyIndex: max_edit_distance must be >= 0");
    }

    FuzzyIndex idx;
    auto& impl = *idx.impl_;
    impl.opts = std::move(opts);
    impl.max_supported_k = impl.opts.max_edit_distance;
    impl.entries = entries;

    const bool ci = impl.opts.case_insensitive;
    const bool use_buckets = static_cast<bool>(impl.opts.bucket_fn);

    for (std::size_t id = 0; id < impl.entries.size(); ++id) {
        const std::string& raw = impl.entries[id];
        std::vector<uint32_t> cps = detail::utf8_to_codepoints(raw);
        if (ci) detail::to_lower_inplace(cps);

        impl.global_trie.insert(cps, static_cast<int>(id));

        if (use_buckets) {
            std::string key = impl.opts.bucket_fn(raw);
            // Bucket keys fold the same way the entries do.
            if (ci) key = detail::lower_utf8(key);
            impl.buckets[key].insert(cps, static_cast<int>(id));
        }
    }

    impl.global_trie.finalize();
    for (auto& kv : impl.buckets) kv.second.finalize();

    return idx;
}

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------

std::size_t FuzzyIndex::size() const noexcept {
    return impl_ ? impl_->entries.size() : 0;
}

std::size_t FuzzyIndex::bucket_count() const noexcept {
    if (!impl_) return 0;
    // +1 for the global trie that always exists.
    return impl_->buckets.size() + 1;
}

int FuzzyIndex::max_supported_distance() const noexcept {
    return impl_ ? impl_->max_supported_k : 0;
}

} // namespace resolve::fuzzy
