// FuzzyIndex query path: iterative DFS over the trie driven by the
// Damerau-Levenshtein automaton, with top-N collection and adaptive k tightening.

#include "resolve/fuzzy.hpp"
#include "fuzzy_internal.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace resolve::fuzzy {

namespace {

struct MatchInternal {
    int distance;
    int id;
};

// Max-heap of size <= n on `distance`. Keeps the n smallest distances seen.
class TopNCollector {
public:
    TopNCollector(int n, int max_k)
        : n_(n), max_k_(max_k) { heap_.reserve(static_cast<std::size_t>(n + 1)); }

    int effective_k() const {
        if (static_cast<int>(heap_.size()) < n_) return max_k_;
        int top_dist = heap_.front().distance;
        return top_dist < max_k_ ? top_dist : max_k_;
    }

    void offer(int distance, int id) {
        if (static_cast<int>(heap_.size()) < n_) {
            heap_.push_back({distance, id});
            std::push_heap(heap_.begin(), heap_.end(), MaxCmp{});
        } else if (distance < heap_.front().distance) {
            std::pop_heap(heap_.begin(), heap_.end(), MaxCmp{});
            heap_.back() = {distance, id};
            std::push_heap(heap_.begin(), heap_.end(), MaxCmp{});
        }
    }

    std::vector<MatchInternal> drain_sorted() {
        std::sort(heap_.begin(), heap_.end(),
                  [](const MatchInternal& a, const MatchInternal& b) {
                      if (a.distance != b.distance) return a.distance < b.distance;
                      return a.id < b.id;
                  });
        return std::move(heap_);
    }

    std::size_t size() const noexcept { return heap_.size(); }

private:
    struct MaxCmp {
        bool operator()(const MatchInternal& a, const MatchInternal& b) const {
            return a.distance < b.distance;  // top = largest distance
        }
    };

    int n_;
    int max_k_;
    std::vector<MatchInternal> heap_;
};

// Iterative DFS over a trie. Re-uses depth-indexed buffers so we never
// allocate per visited node.
void dfs_trie(const detail::Trie& trie,
              const detail::LevenshteinAutomaton& aut,
              TopNCollector& collector)
{
    const auto& root = trie.node(trie.root());

    std::vector<std::vector<int>> rows;     // rows[d] = DP column at depth d
    std::vector<uint32_t> chars;            // chars[d] = code point consumed at depth d
    rows.reserve(64);
    chars.reserve(64);
    rows.emplace_back();
    aut.initial_row(rows[0]);
    chars.push_back(0);  // unused at depth 0

    // Empty entry: the root itself terminates the empty string. Check it
    // against the initial row before descending.
    if (root.entry_id >= 0) {
        int dist = 0;
        if (aut.accepts(rows[0], dist)) {
            collector.offer(dist, root.entry_id);
        }
    }

    if (root.first_child < 0) return;

    struct Frame { int node; int depth; };
    std::vector<Frame> stack;
    stack.reserve(256);
    stack.push_back({root.first_child, 1});

    while (!stack.empty()) {
        Frame f = stack.back();
        stack.pop_back();

        const auto& tn = trie.node(f.node);

        // Grow per-depth buffers as needed.
        if (static_cast<int>(rows.size()) <= f.depth) {
            rows.resize(static_cast<std::size_t>(f.depth + 1));
            chars.resize(static_cast<std::size_t>(f.depth + 1), 0);
        }

        const int parent_depth = f.depth - 1;
        const std::vector<int>* grandparent =
            (parent_depth >= 1) ? &rows[static_cast<std::size_t>(parent_depth - 1)] : nullptr;
        const uint32_t parent_char =
            (parent_depth >= 1) ? chars[static_cast<std::size_t>(parent_depth)] : 0u;

        const int eff_k = collector.effective_k();
        bool alive = aut.step(rows[static_cast<std::size_t>(parent_depth)],
                              grandparent,
                              parent_char,
                              tn.c,
                              eff_k,
                              rows[static_cast<std::size_t>(f.depth)]);
        chars[static_cast<std::size_t>(f.depth)] = tn.c;

        if (alive && tn.entry_id >= 0) {
            int dist = 0;
            if (aut.accepts(rows[static_cast<std::size_t>(f.depth)], dist)) {
                collector.offer(dist, tn.entry_id);
            }
        }

        // Always queue the next sibling at the same depth: it does not depend
        // on the subtree under the current node.
        if (tn.next_sibling >= 0) {
            stack.push_back({tn.next_sibling, f.depth});
        }
        // Only descend if the current state has hope of reaching an accepting
        // state somewhere in the subtree.
        if (alive && tn.first_child >= 0) {
            stack.push_back({tn.first_child, f.depth + 1});
        }
    }
}

// Lowercase a UTF-8 string and produce its codepoint sequence.
std::string lower_to_utf8(std::string_view s) {
    auto cps = detail::utf8_to_codepoints(s);
    detail::to_lower_inplace(cps);
    std::string out;
    out.reserve(s.size());
    for (uint32_t cp : cps) {
        if (cp < 0x80u) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800u) {
            out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else if (cp < 0x10000u) {
            out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else {
            out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        }
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// query
// ---------------------------------------------------------------------------

std::vector<Match> FuzzyIndex::query(std::string_view needle, QueryOptions opts) const {
    if (!impl_) return {};
    const auto& impl = *impl_;

    if (opts.top_n <= 0) return {};
    if (opts.max_edit_distance < 0) {
        throw std::invalid_argument("FuzzyIndex::query: max_edit_distance must be >= 0");
    }
    int k = opts.max_edit_distance;
    if (k > impl.max_supported_k) k = impl.max_supported_k;

    auto needle_cps = detail::utf8_to_codepoints(needle);
    if (impl.opts.case_insensitive) detail::to_lower_inplace(needle_cps);

    detail::LevenshteinAutomaton aut(std::move(needle_cps), k, impl.opts.damerau);

    TopNCollector collector(opts.top_n, k);

    // Try bucket first if a hint was given and that bucket exists. This is a
    // deliberate speed/completeness tradeoff: when the bucket yields any match
    // we return the bucket-local top-N WITHOUT scanning the global trie, so the
    // result is not guaranteed to be the global best if a strictly closer entry
    // lives in another bucket (e.g. the hint's own key is the misspelled part).
    // Callers that need a global guarantee should omit bucket_hint; the WFO
    // matcher uses the hint as a genus fast-path and falls back to difflib for
    // the misspelled-genus case.
    if (opts.bucket_hint && !impl.buckets.empty()) {
        std::string key = *opts.bucket_hint;
        if (impl.opts.case_insensitive) key = lower_to_utf8(key);
        auto it = impl.buckets.find(key);
        if (it != impl.buckets.end()) {
            dfs_trie(it->second, aut, collector);
            if (collector.size() > 0) {
                auto raw = collector.drain_sorted();
                std::vector<Match> out;
                out.reserve(raw.size());
                for (const auto& m : raw) {
                    out.push_back(Match{impl.entries[static_cast<std::size_t>(m.id)], m.id, m.distance});
                }
                return out;
            }
        }
    }

    // Fall back to (or default to) the global trie.
    dfs_trie(impl.global_trie, aut, collector);
    auto raw = collector.drain_sorted();
    std::vector<Match> out;
    out.reserve(raw.size());
    for (const auto& m : raw) {
        out.push_back(Match{impl.entries[static_cast<std::size_t>(m.id)], m.id, m.distance});
    }
    return out;
}

// ---------------------------------------------------------------------------
// query_batch
// ---------------------------------------------------------------------------

std::vector<std::vector<Match>> FuzzyIndex::query_batch(
    const std::vector<std::string>& needles,
    QueryOptions opts,
    std::function<std::string(std::string_view)> bucket_fn) const
{
    std::vector<std::vector<Match>> results(needles.size());
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(needles.size());

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 32)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        QueryOptions local = opts;
        if (bucket_fn) {
            local.bucket_hint = bucket_fn(needles[static_cast<std::size_t>(i)]);
        }
        results[static_cast<std::size_t>(i)] =
            this->query(needles[static_cast<std::size_t>(i)], local);
    }

    return results;
}

} // namespace resolve::fuzzy
