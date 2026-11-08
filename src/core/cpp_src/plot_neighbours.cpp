#include "resolve/plot_neighbours.hpp"

#include "resolve/geo.hpp"
#include "resolve/sparse_rows.hpp"

#include <ATen/Parallel.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace resolve {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// murmur3's finaliser and a hash of three words built from it: the samples of
// a descent are drawn by sorting on it, so they depend on no thread order.
inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}
inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
    return fmix32(a ^ fmix32(b ^ fmix32(c ^ 0x9e3779b9u)));
}

// What a hash draws for: the start of an initial posting walk, the new entries
// a list joins, and the reverse entries a plot joins.
enum Salt : uint32_t { kSaltStart = 0, kSaltNew = 1, kSaltReverseNew = 2, kSaltReverseOld = 3 };
inline uint32_t salt(Salt what, int round) { return static_cast<uint32_t>(round) * 4u + what; }

// Keep the `keep` entries of `v` with the smallest (hash, entry).
void sample_by_hash(std::vector<int32_t>& v, std::size_t keep, uint32_t owner, uint32_t s) {
    if (v.size() <= keep) return;
    std::vector<std::pair<uint32_t, int32_t>> keyed;
    keyed.reserve(v.size());
    for (const int32_t x : v) keyed.emplace_back(hash3(owner, static_cast<uint32_t>(x), s), x);
    std::partial_sort(keyed.begin(), keyed.begin() + static_cast<std::ptrdiff_t>(keep), keyed.end());
    v.clear();
    for (std::size_t i = 0; i < keep; ++i) v.push_back(keyed[i].second);
}

template <typename F>
void parallel_rows(int64_t n, F f) {
    at::parallel_for(0, n, /*grain_size=*/256, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) f(i);
    });
}

}  // namespace

// ---------------------------------------------------------------------------
// Presence sets
// ---------------------------------------------------------------------------

PresenceSets::PresenceSets(const torch::Tensor& species) {
    if (!species.defined() || species.dim() != 2) {
        throw std::invalid_argument("PresenceSets: species must be a (n_plots, n_species) matrix");
    }
    const int64_t n = species.size(0);
    const int64_t n_columns = species.size(1);
    torch::Tensor rows;
    torch::Tensor columns;
    if (is_sparse_rows(species)) {
        const torch::Tensor s = species.to(torch::kCPU);
        const torch::Tensor present = s.values() != 0;
        const torch::Tensor crow = s.crow_indices();
        const torch::Tensor entry_rows =
            torch::repeat_interleave(torch::arange(n, torch::kInt64), crow.slice(0, 1) - crow.slice(0, 0, n));
        rows = entry_rows.index({present});
        columns = s.col_indices().index({present});
    } else {
        const torch::Tensor nz = (species.to(torch::kCPU) != 0).nonzero();
        rows = nz.select(1, 0).contiguous();
        columns = nz.select(1, 1).contiguous();
    }
    const auto r = rows.accessor<int64_t, 1>();
    const auto c = columns.accessor<int64_t, 1>();
    const int64_t nnz = rows.size(0);

    // Columns ranked rarest first among these plots, ties by column.
    std::vector<int64_t> frequency(static_cast<std::size_t>(n_columns), 0);
    for (int64_t e = 0; e < nnz; ++e) ++frequency[static_cast<std::size_t>(c[e])];
    std::vector<int32_t> by_rarity(static_cast<std::size_t>(n_columns));
    std::iota(by_rarity.begin(), by_rarity.end(), 0);
    std::stable_sort(by_rarity.begin(), by_rarity.end(),
                     [&](int32_t a, int32_t b) { return frequency[a] < frequency[b]; });
    std::vector<int32_t> rank_of(static_cast<std::size_t>(n_columns));
    for (int64_t i = 0; i < n_columns; ++i) rank_of[by_rarity[i]] = static_cast<int32_t>(i);

    offsets_.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int64_t e = 0; e < nnz; ++e) ++offsets_[static_cast<std::size_t>(r[e]) + 1];
    for (int64_t i = 0; i < n; ++i) offsets_[i + 1] += offsets_[i];
    ranks_.resize(static_cast<std::size_t>(nnz));
    {
        std::vector<int64_t> fill(offsets_.begin(), offsets_.end() - 1);
        for (int64_t e = 0; e < nnz; ++e) ranks_[fill[r[e]]++] = rank_of[c[e]];
    }
    for (int64_t i = 0; i < n; ++i) std::sort(ranks_.begin() + offsets_[i], ranks_.begin() + offsets_[i + 1]);

    posting_offsets_.assign(static_cast<std::size_t>(n_columns) + 1, 0);
    for (const int32_t s : ranks_) ++posting_offsets_[static_cast<std::size_t>(s) + 1];
    for (int64_t i = 0; i < n_columns; ++i) posting_offsets_[i + 1] += posting_offsets_[i];
    postings_.resize(ranks_.size());
    std::vector<int64_t> fill(posting_offsets_.begin(), posting_offsets_.end() - 1);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t e = offsets_[i]; e < offsets_[i + 1]; ++e) postings_[fill[ranks_[e]]++] = static_cast<int32_t>(i);
    }
}

int32_t PresenceSets::overlap(int64_t a, int64_t b) const noexcept {
    const int32_t* x = &ranks_[offsets_[a]];
    const int32_t* y = &ranks_[offsets_[b]];
    const int32_t* x_end = x + count(a);
    const int32_t* y_end = y + count(b);
    int32_t shared = 0;
    while (x < x_end && y < y_end) {
        if (*x < *y) {
            ++x;
        } else if (*y < *x) {
            ++y;
        } else {
            ++shared;
            ++x;
            ++y;
        }
    }
    return shared;
}

double PresenceSets::cosine(int64_t a, int64_t b) const noexcept {
    const int32_t shared = overlap(a, b);
    if (shared == 0) return 0.0;
    return shared / std::sqrt(static_cast<double>(count(a)) * static_cast<double>(count(b)));
}

// ---------------------------------------------------------------------------
// Neighbour descent
// ---------------------------------------------------------------------------

void SetGraphConfig::validate() const {
    if (k < 1) throw std::invalid_argument("SetGraphConfig: k must be at least 1");
    if (!(sample_rate > 0.0 && sample_rate <= 1.0)) {
        throw std::invalid_argument("SetGraphConfig: sample_rate must be in (0, 1]");
    }
    if (!(stop_fraction >= 0.0)) throw std::invalid_argument("SetGraphConfig: stop_fraction must be >= 0");
    if (max_rounds < 0) throw std::invalid_argument("SetGraphConfig: max_rounds must be >= 0");
}

// A plot's lists are slots: slot u holds plot u's open list, slot n + u its
// hidden list (plots u < n_hidden). An entry is a candidate plot and the
// number of species it shares with the slot's plot; similarity is ranked
// exactly by comparing overlap^2 / count across entries of one slot (the
// slot's own count cancels).
struct SetNeighbourGraph::State {
    struct Entry {
        int32_t plot;
        int32_t overlap;
    };

    int64_t n = 0;
    int64_t k = 0;
    int64_t n_hidden = 0;
    std::vector<Entry> lists;
    std::vector<int32_t> counts;
    std::vector<uint8_t> fresh;
    std::unique_ptr<std::atomic_flag[]> locks;
    const SimilarityHiding* hides = nullptr;

    [[nodiscard]] int64_t n_slots() const { return n + n_hidden; }
    [[nodiscard]] int64_t plot_of(int64_t slot) const { return slot < n ? slot : slot - n; }

    void resize_slots(int64_t hidden_plots) {
        n_hidden = hidden_plots;
        const auto slots = static_cast<std::size_t>(n_slots());
        lists.resize(slots * static_cast<std::size_t>(k), Entry{-1, 0});
        counts.resize(slots, 0);
        fresh.resize(slots * static_cast<std::size_t>(k), 0);
        locks = std::make_unique<std::atomic_flag[]>(slots);
    }

    const PresenceSets* sets = nullptr;
    int64_t n_candidates = 0;

    [[nodiscard]] double similarity(int64_t u, int64_t v, int32_t shared) const {
        return shared / std::sqrt(static_cast<double>(sets->count(u)) * static_cast<double>(sets->count(v)));
    }

    // Whether a precedes b in a slot of one plot.
    [[nodiscard]] bool better(const Entry& a, const Entry& b) const {
        const int64_t lhs = int64_t{a.overlap} * a.overlap * sets->count(b.plot);
        const int64_t rhs = int64_t{b.overlap} * b.overlap * sets->count(a.plot);
        return lhs > rhs || (lhs == rhs && a.plot < b.plot);
    }

    // The slot keeps e if it is not there yet and ranks within the top k.
    void insert(int64_t slot, Entry e) {
        std::atomic_flag& lock = locks[static_cast<std::size_t>(slot)];
        while (lock.test_and_set(std::memory_order_acquire)) {
        }
        Entry* list = &lists[static_cast<std::size_t>(slot * k)];
        int32_t& c = counts[static_cast<std::size_t>(slot)];
        bool seen = false;
        for (int32_t m = 0; m < c; ++m) {
            if (list[m].plot == e.plot) {
                seen = true;
                break;
            }
        }
        if (!seen && (c < k || better(e, list[c - 1]))) {
            int64_t pos = c < k ? c : k - 1;
            while (pos > 0 && better(e, list[pos - 1])) {
                list[pos] = list[pos - 1];
                --pos;
            }
            list[pos] = e;
            if (c < k) ++c;
        }
        lock.clear(std::memory_order_release);
    }

    // Plot v proposed to plot u, sharing `shared` species: to u's open list,
    // and to its hidden list unless the rules hide v from u.
    void propose(int64_t u, int64_t v, int32_t shared) {
        if (u == v || v >= n_candidates || shared == 0) return;
        const Entry e{static_cast<int32_t>(v), shared};
        insert(u, e);
        if (u < n_hidden && !(*hides)(u, v, similarity(u, v, shared))) insert(n + u, e);
    }
};

namespace {

using Entry = SetNeighbourGraph::State::Entry;

}  // namespace

SetNeighbourGraph::SetNeighbourGraph(const PresenceSets& sets, int64_t n_candidates, SetGraphConfig config)
    : sets_(sets), n_candidates_(n_candidates), config_(config), state_(new State) {
    config_.validate();
    if (n_candidates < 0 || n_candidates > sets.size()) {
        throw std::invalid_argument("SetNeighbourGraph: n_candidates must be within the plots");
    }
    State& st = *state_;
    st.n = sets.size();
    st.k = config_.k;
    st.resize_slots(0);

    st.sets = &sets_;
    st.n_candidates = n_candidates_;
    const int64_t k = st.k;

    // Initial lists: candidates sharing the plot's rarest species, walked
    // from a hashed start through each species' candidate postings.
    const auto& offsets = sets_.posting_offsets_;
    const auto& postings = sets_.postings_;
    parallel_rows(st.n, [&](int64_t q) {
        int64_t taken = 0;
        const int32_t* species = sets_.species(q);
        for (int32_t j = 0; j < sets_.count(q) && taken < 2 * k; ++j) {
            const int32_t s = species[j];
            const int32_t* begin = &postings[static_cast<std::size_t>(offsets[s])];
            const int32_t* end = std::lower_bound(begin, begin + (offsets[s + 1] - offsets[s]),
                                                  static_cast<int32_t>(n_candidates_));
            const int64_t length = end - begin;
            if (length == 0) continue;
            const int64_t start = hash3(static_cast<uint32_t>(q), static_cast<uint32_t>(s),
                                        salt(kSaltStart, 0)) % static_cast<uint64_t>(length);
            for (int64_t m = 0; m < std::min(length, k) && taken < 2 * k; ++m) {
                const int32_t v = begin[(start + m) % length];
                if (v == q) continue;
                st.propose(q, v, sets_.overlap(q, v));
                ++taken;
            }
        }
    });
    std::fill(st.fresh.begin(), st.fresh.end(), uint8_t{1});
    descend();
}

SetNeighbourGraph::~SetNeighbourGraph() { delete state_; }

void SetNeighbourGraph::descend() {
    State& st = *state_;
    const int64_t k = st.k;
    const auto sample = static_cast<std::size_t>(std::max<int64_t>(1, static_cast<int64_t>(config_.sample_rate * k)));
    for (int round = 0; round < config_.max_rounds; ++round, ++rounds_) {
        const int64_t slots = st.n_slots();
        const std::vector<Entry> start_lists = st.lists;
        const std::vector<int32_t> start_counts = st.counts;
        const std::vector<uint8_t> start_fresh = st.fresh;

        // Each slot's new entries (a hashed sample) and old entries.
        std::vector<std::vector<int32_t>> joined_new(static_cast<std::size_t>(slots));
        std::vector<std::vector<int32_t>> joined_old(static_cast<std::size_t>(slots));
        parallel_rows(slots, [&](int64_t slot) {
            const Entry* list = &start_lists[static_cast<std::size_t>(slot * k)];
            auto& fresh_entries = joined_new[static_cast<std::size_t>(slot)];
            for (int32_t m = 0; m < start_counts[slot]; ++m) {
                (start_fresh[static_cast<std::size_t>(slot * k + m)] ? fresh_entries
                                                                      : joined_old[static_cast<std::size_t>(slot)])
                    .push_back(list[m].plot);
            }
            sample_by_hash(fresh_entries, sample, static_cast<uint32_t>(st.plot_of(slot)),
                           salt(kSaltNew, rounds_));
            std::sort(fresh_entries.begin(), fresh_entries.end());
        });
        // Reverse entries: the plots whose lists hold each candidate.
        std::vector<std::vector<int32_t>> reverse_new(static_cast<std::size_t>(st.n));
        std::vector<std::vector<int32_t>> reverse_old(static_cast<std::size_t>(st.n));
        for (int64_t slot = 0; slot < slots; ++slot) {
            const auto owner = static_cast<int32_t>(st.plot_of(slot));
            for (const int32_t v : joined_new[static_cast<std::size_t>(slot)]) reverse_new[v].push_back(owner);
            for (const int32_t v : joined_old[static_cast<std::size_t>(slot)]) reverse_old[v].push_back(owner);
        }
        parallel_rows(st.n, [&](int64_t v) {
            for (auto* list : {&reverse_new[v], &reverse_old[v]}) {
                std::sort(list->begin(), list->end());
                list->erase(std::unique(list->begin(), list->end()), list->end());
            }
            sample_by_hash(reverse_new[v], sample, static_cast<uint32_t>(v), salt(kSaltReverseNew, rounds_));
            sample_by_hash(reverse_old[v], sample, static_cast<uint32_t>(v), salt(kSaltReverseOld, rounds_));
        });

        // Local join: every pair of a plot's new entries, and each new entry
        // with each old one, proposed to each other.
        parallel_rows(st.n, [&](int64_t i) {
            std::vector<int32_t> fresh_plots = joined_new[static_cast<std::size_t>(i)];
            std::vector<int32_t> old_plots = joined_old[static_cast<std::size_t>(i)];
            if (i < st.n_hidden) {
                const auto& h_new = joined_new[static_cast<std::size_t>(st.n + i)];
                const auto& h_old = joined_old[static_cast<std::size_t>(st.n + i)];
                fresh_plots.insert(fresh_plots.end(), h_new.begin(), h_new.end());
                old_plots.insert(old_plots.end(), h_old.begin(), h_old.end());
            }
            fresh_plots.insert(fresh_plots.end(), reverse_new[i].begin(), reverse_new[i].end());
            old_plots.insert(old_plots.end(), reverse_old[i].begin(), reverse_old[i].end());
            std::sort(fresh_plots.begin(), fresh_plots.end());
            fresh_plots.erase(std::unique(fresh_plots.begin(), fresh_plots.end()), fresh_plots.end());
            std::sort(old_plots.begin(), old_plots.end());
            old_plots.erase(std::unique(old_plots.begin(), old_plots.end()), old_plots.end());
            std::vector<int32_t> only_old;
            std::set_difference(old_plots.begin(), old_plots.end(), fresh_plots.begin(), fresh_plots.end(),
                                std::back_inserter(only_old));
            for (std::size_t a = 0; a < fresh_plots.size(); ++a) {
                const int32_t u = fresh_plots[a];
                for (std::size_t b = a + 1; b < fresh_plots.size(); ++b) {
                    const int32_t v = fresh_plots[b];
                    if (u >= n_candidates_ && v >= n_candidates_) continue;
                    const int32_t shared = sets_.overlap(u, v);
                    st.propose(u, v, shared);
                    st.propose(v, u, shared);
                }
                for (const int32_t v : only_old) {
                    if (u >= n_candidates_ && v >= n_candidates_) continue;
                    const int32_t shared = sets_.overlap(u, v);
                    st.propose(u, v, shared);
                    st.propose(v, u, shared);
                }
            }
        });

        // An entry is new when the round brought it, or it was new and not
        // joined; the round's change is the count of entries it brought.
        std::atomic<int64_t> changed{0};
        parallel_rows(slots, [&](int64_t slot) {
            const Entry* before = &start_lists[static_cast<std::size_t>(slot * k)];
            const Entry* now = &st.lists[static_cast<std::size_t>(slot * k)];
            const auto& joined = joined_new[static_cast<std::size_t>(slot)];
            int64_t brought = 0;
            for (int32_t m = 0; m < st.counts[slot]; ++m) {
                int32_t was = -1;
                for (int32_t t = 0; t < start_counts[slot]; ++t) {
                    if (before[t].plot == now[m].plot) {
                        was = t;
                        break;
                    }
                }
                const bool fresh = was < 0 || (start_fresh[static_cast<std::size_t>(slot * k + was)] &&
                                               !std::binary_search(joined.begin(), joined.end(), now[m].plot));
                st.fresh[static_cast<std::size_t>(slot * k + m)] = fresh ? 1 : 0;
                if (was < 0) ++brought;
            }
            changed += brought;
        });
        if (static_cast<double>(changed.load()) < config_.stop_fraction * static_cast<double>(slots * k)) {
            ++rounds_;
            break;
        }
    }
}

SetNeighbours SetNeighbourGraph::lists(int64_t first_slot, int64_t n_rows) const {
    const State& st = *state_;
    const int64_t k = st.k;
    torch::Tensor indices = torch::full({n_rows, k}, -1, torch::kInt64);
    torch::Tensor similarities = torch::full({n_rows, k}, kNaN, torch::kFloat64);
    auto idx = indices.accessor<int64_t, 2>();
    auto sim = similarities.accessor<double, 2>();
    for (int64_t row = 0; row < n_rows; ++row) {
        const int64_t slot = first_slot + row;
        const int64_t plot = st.plot_of(slot);
        const Entry* list = &st.lists[static_cast<std::size_t>(slot * k)];
        int32_t c = st.counts[slot];
        for (int32_t m = 0; m < c; ++m) {
            idx[row][m] = list[m].plot;
            sim[row][m] = st.similarity(plot, list[m].plot, list[m].overlap);
        }
        // Fewer plots sharing a species than k: the lowest-row candidates
        // sharing none follow, at similarity 0, as an exact search ranks them.
        for (int64_t v = 0; v < n_candidates_ && c < k; ++v) {
            if (v == plot || sets_.overlap(plot, v) > 0) continue;
            if (slot >= st.n && (*st.hides)(plot, v, 0.0)) continue;
            idx[row][c] = v;
            sim[row][c] = 0.0;
            ++c;
        }
    }
    return {indices, similarities};
}

SetNeighbours SetNeighbourGraph::open() const { return lists(0, state_->n); }

SetNeighbours SetNeighbourGraph::hidden(int64_t n_queries, const SimilarityHiding& hides) {
    State& st = *state_;
    if (n_queries < 0 || n_queries > st.n) {
        throw std::invalid_argument("SetNeighbourGraph::hidden: n_queries must be within the plots");
    }
    const int64_t k = st.k;
    st.hides = &hides;
    st.resize_slots(0);
    st.resize_slots(n_queries);
    // Each hidden list starts as the visible part of the plot's open list.
    parallel_rows(n_queries, [&](int64_t u) {
        const Entry* open_list = &st.lists[static_cast<std::size_t>(u * k)];
        Entry* list = &st.lists[static_cast<std::size_t>((st.n + u) * k)];
        int32_t c = 0;
        for (int32_t m = 0; m < st.counts[u]; ++m) {
            const Entry e = open_list[m];
            if (!hides(u, e.plot, st.similarity(u, e.plot, e.overlap))) {
                list[c] = e;
                st.fresh[static_cast<std::size_t>((st.n + u) * k + c)] = 1;
                ++c;
            }
        }
        st.counts[static_cast<std::size_t>(st.n + u)] = c;
    });
    descend();
    SetNeighbours out = lists(st.n, n_queries);
    st.hides = nullptr;
    st.resize_slots(0);
    return out;
}

// ---------------------------------------------------------------------------
// Positions
// ---------------------------------------------------------------------------

namespace {

constexpr int64_t kLeafPlaces = 16;
constexpr double kRadians = 3.14159265358979323846 / 180.0;
// Slack on the tree's bounds, in km: a bound converted from a chord may sit
// this far off the haversine distance it bounds.
constexpr double kBoundSlackKm = 1e-7;

double chord_to_km(double chord) { return 2.0 * kEarthRadiusKm * std::asin(std::min(1.0, chord / 2.0)); }

}  // namespace

PlaceIndex::PlaceIndex(const torch::Tensor& coordinates, int64_t n_candidates) : n_candidates_(n_candidates) {
    if (!coordinates.defined() || coordinates.dim() != 2 || coordinates.size(1) != 2) {
        throw std::invalid_argument("PlaceIndex: coordinates must be a (rows, 2) latitude / longitude matrix");
    }
    const int64_t n = coordinates.size(0);
    if (n_candidates < 0 || n_candidates > n) {
        throw std::invalid_argument("PlaceIndex: n_candidates must be within the rows");
    }
    const torch::Tensor c = coordinates.to(torch::kCPU, torch::kFloat64).contiguous();
    const auto a = c.accessor<double, 2>();
    lat_.resize(static_cast<std::size_t>(n));
    lon_.resize(static_cast<std::size_t>(n));
    xyz_.resize(static_cast<std::size_t>(3 * n));
    for (int64_t i = 0; i < n; ++i) {
        lat_[i] = a[i][0];
        lon_[i] = a[i][1];
        const double phi = lat_[i] * kRadians;
        const double lambda = lon_[i] * kRadians;
        xyz_[3 * i] = std::cos(phi) * std::cos(lambda);
        xyz_[3 * i + 1] = std::cos(phi) * std::sin(lambda);
        xyz_[3 * i + 2] = std::sin(phi);
        if (i < n_candidates && std::isfinite(lat_[i]) && std::isfinite(lon_[i])) order_.push_back(i);
    }
    if (!order_.empty()) build(0, static_cast<int64_t>(order_.size()));
}

int32_t PlaceIndex::build(int64_t begin, int64_t end) {
    Node node{};
    for (int d = 0; d < 3; ++d) {
        node.lo[d] = std::numeric_limits<double>::infinity();
        node.hi[d] = -std::numeric_limits<double>::infinity();
    }
    for (int64_t i = begin; i < end; ++i) {
        for (int d = 0; d < 3; ++d) {
            node.lo[d] = std::min(node.lo[d], xyz_[3 * order_[i] + d]);
            node.hi[d] = std::max(node.hi[d], xyz_[3 * order_[i] + d]);
        }
    }
    node.begin = begin;
    node.end = end;
    node.left = node.right = -1;
    const auto id = static_cast<int32_t>(nodes_.size());
    nodes_.push_back(node);
    if (end - begin > kLeafPlaces) {
        int axis = 0;
        for (int d = 1; d < 3; ++d) {
            if (node.hi[d] - node.lo[d] > node.hi[axis] - node.lo[axis]) axis = d;
        }
        const int64_t mid = begin + (end - begin) / 2;
        std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end,
                         [&](int64_t p, int64_t q) {
                             const double a = xyz_[3 * p + axis];
                             const double b = xyz_[3 * q + axis];
                             return a < b || (a == b && p < q);
                         });
        const int32_t left = build(begin, mid);
        const int32_t right = build(mid, end);
        nodes_[id].left = left;
        nodes_[id].right = right;
    }
    return id;
}

PlaceNeighbours PlaceIndex::nearest(const std::vector<int64_t>& queries, int64_t k, const DistanceHiding& hides,
                                   const std::vector<double>& radius) const {
    const auto n_queries = static_cast<int64_t>(queries.size());
    torch::Tensor indices = torch::full({n_queries, k}, -1, torch::kInt64);
    torch::Tensor km = torch::full({n_queries, k}, kNaN, torch::kFloat64);
    if (k < 1 || nodes_.empty()) return {indices, km};
    auto idx = indices.accessor<int64_t, 2>();
    auto dist = km.accessor<double, 2>();

    struct Found {
        double km;
        int64_t row;
    };
    // a ranks before b: nearer, or as near and the lower row.
    auto before = [](const Found& a, const Found& b) { return a.km < b.km || (a.km == b.km && a.row < b.row); };

    parallel_rows(n_queries, [&](int64_t qi) {
        const int64_t q = queries[static_cast<std::size_t>(qi)];
        if (!std::isfinite(lat_[q]) || !std::isfinite(lon_[q])) return;
        const double* p = &xyz_[3 * q];
        const double hidden_within = radius.empty() ? kNaN : radius[static_cast<std::size_t>(qi)];
        std::vector<Found> heap;  // the worst found on top
        heap.reserve(static_cast<std::size_t>(k) + 1);
        auto worst = [&]() { return heap.size() == static_cast<std::size_t>(k) ? heap.front().km : std::numeric_limits<double>::infinity(); };
        auto box_km = [&](const Node& node, bool far) {
            double s = 0.0;
            for (int d = 0; d < 3; ++d) {
                double gap;
                if (far) {
                    gap = std::max(std::abs(p[d] - node.lo[d]), std::abs(p[d] - node.hi[d]));
                } else {
                    gap = p[d] < node.lo[d] ? node.lo[d] - p[d] : (p[d] > node.hi[d] ? p[d] - node.hi[d] : 0.0);
                }
                s += gap * gap;
            }
            return chord_to_km(std::sqrt(s));
        };
        auto visit = [&](auto&& self, int32_t id) -> void {
            const Node& node = nodes_[static_cast<std::size_t>(id)];
            if (box_km(node, false) - kBoundSlackKm > worst()) return;
            if (std::isfinite(hidden_within) && box_km(node, true) + kBoundSlackKm < hidden_within) return;
            if (node.left < 0) {
                for (int64_t i = node.begin; i < node.end; ++i) {
                    const int64_t v = order_[static_cast<std::size_t>(i)];
                    if (v == q) continue;
                    const Found f{great_circle_km(lat_[q], lon_[q], lat_[v], lon_[v]), v};
                    if (heap.size() == static_cast<std::size_t>(k) && !before(f, heap.front())) continue;
                    if (hides && hides(q, v, f.km)) continue;
                    if (heap.size() == static_cast<std::size_t>(k)) {
                        std::pop_heap(heap.begin(), heap.end(), before);
                        heap.back() = f;
                    } else {
                        heap.push_back(f);
                    }
                    std::push_heap(heap.begin(), heap.end(), before);
                }
                return;
            }
            const Node& l = nodes_[static_cast<std::size_t>(node.left)];
            const Node& r = nodes_[static_cast<std::size_t>(node.right)];
            const bool left_first = box_km(l, false) <= box_km(r, false);
            self(self, left_first ? node.left : node.right);
            self(self, left_first ? node.right : node.left);
        };
        visit(visit, 0);
        std::sort_heap(heap.begin(), heap.end(), before);
        for (std::size_t m = 0; m < heap.size(); ++m) {
            idx[qi][static_cast<int64_t>(m)] = heap[m].row;
            dist[qi][static_cast<int64_t>(m)] = heap[m].km;
        }
    });
    return {indices, km};
}

}  // namespace resolve
