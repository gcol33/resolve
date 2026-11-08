#pragma once

// Neighbour searches over plots that scale to a continent's plots: by species
// composition (cosine similarity of presence sets) and by position
// (great-circle distance), each with per-plot hiding rules.
//
// By species, an exact search compares every plot with every other: at the
// similarities plots share in EVA (the 50th neighbour near 0.45) no bound
// prunes enough to change that. SetNeighbourGraph builds the neighbour lists
// by nearest-neighbour descent (Dong, Charikar & Li 2011, "Efficient k-nearest
// neighbor graph construction for generic similarity measures", WWW), whose
// cost grows close to linearly in the number of plots: a plot's neighbours'
// neighbours are compared with it, round after round, until the lists settle.
// The result is deterministic whatever the thread count: each round reads the
// lists as they stood when it began, a plot's list is the top k of every
// plot proposed to it, and samples are drawn by hashing.
//
// By position, PlaceIndex is an exact k-d tree over positions on the unit
// sphere.

#include "resolve/retrieval.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace resolve {

// Per plot, how near another plot may be and still be its neighbour. A plot
// is hidden from plot i when it is more similar than similarity[i], less than
// km[i] kilometres away, or less than years[i] years apart. A NaN radius,
// position or year hides nothing; an undefined tensor applies no rule on its
// axis. Each defined tensor holds one value per plot the rules apply to.
struct HidingRadii {
    torch::Tensor similarity;
    torch::Tensor km;
    torch::Tensor years;
};

// The species of each plot as a set: an entry other than 0 counts once.
class PresenceSets {
public:
    // `species`: (n_plots, n_species), dense or sparse rows (sparse_rows.hpp).
    explicit PresenceSets(const torch::Tensor& species);

    [[nodiscard]] int64_t size() const noexcept { return static_cast<int64_t>(offsets_.size()) - 1; }
    [[nodiscard]] int32_t count(int64_t plot) const noexcept {
        return static_cast<int32_t>(offsets_[plot + 1] - offsets_[plot]);
    }
    // Species the two plots share.
    [[nodiscard]] int32_t overlap(int64_t a, int64_t b) const noexcept;
    // overlap / sqrt(count(a) count(b)); 0 when either plot has no species.
    [[nodiscard]] double cosine(int64_t a, int64_t b) const noexcept;

    // A plot's species as ranks, rarest first among these plots (ties by
    // species column).
    [[nodiscard]] const int32_t* species(int64_t plot) const noexcept { return &ranks_[offsets_[plot]]; }

private:
    friend class SetNeighbourGraph;
    std::vector<int64_t> offsets_;
    std::vector<int32_t> ranks_;
    // The plots holding each species rank, in increasing order.
    std::vector<int64_t> posting_offsets_;
    std::vector<int32_t> postings_;
};

// Neighbour lists: (n, k) int64 plot rows, most similar first, -1 after a
// plot's last neighbour; (n, k) float64 similarities, NaN after it.
struct SetNeighbours {
    torch::Tensor indices;
    torch::Tensor similarities;
};

struct SetGraphConfig {
    // Neighbours per plot.
    int64_t k = 50;
    // Share of a list's new entries (and of its reverse entries) joined each
    // round (Dong et al.'s rho).
    double sample_rate = 0.5;
    // Rounds stop once fewer than this share of all list entries changed.
    double stop_fraction = 0.001;
    int max_rounds = 30;

    void validate() const;
};

// Whether plot `v` is hidden from plot `u`, given their similarity.
using SimilarityHiding = std::function<bool(int64_t u, int64_t v, double similarity)>;

// Neighbour lists of every plot of a PresenceSets among its first
// `n_candidates` plots (the plots with a known value), by cosine similarity.
// A plot is never its own neighbour; ties go to the lower row. A plot with
// fewer than k candidates sharing a species with it lists the lowest-row
// candidates sharing none after them, at similarity 0, as an exact search
// ranks them.
class SetNeighbourGraph {
public:
    SetNeighbourGraph(const PresenceSets& sets, int64_t n_candidates, SetGraphConfig config);
    ~SetNeighbourGraph();
    SetNeighbourGraph(const SetNeighbourGraph&) = delete;
    SetNeighbourGraph& operator=(const SetNeighbourGraph&) = delete;

    // Every plot's neighbours, no plot hidden.
    [[nodiscard]] SetNeighbours open() const;

    // The neighbours of plots [0, n_queries) that `hides` leaves visible,
    // found by continuing the descent with these lists beside the open ones
    // (which keep guiding it through the hidden plots).
    [[nodiscard]] SetNeighbours hidden(int64_t n_queries, const SimilarityHiding& hides);

    [[nodiscard]] int rounds() const noexcept { return rounds_; }

    struct State;

private:
    // Rounds of local joins until the lists settle.
    void descend();
    [[nodiscard]] SetNeighbours lists(int64_t first_slot, int64_t n_rows) const;

    const PresenceSets& sets_;
    int64_t n_candidates_;
    SetGraphConfig config_;
    State* state_;
    int rounds_ = 0;
};

// Whether plot `v` is hidden from plot `u`, given their distance in km.
using DistanceHiding = std::function<bool(int64_t u, int64_t v, double km)>;

// Exact nearest positions among the first `n_candidates` rows of a (n, 2)
// latitude / longitude matrix in degrees, by the haversine distance on a
// sphere of kEarthRadiusKm. A row with a NaN coordinate is never a neighbour
// and has none.
class PlaceIndex {
public:
    PlaceIndex(const torch::Tensor& coordinates, int64_t n_candidates);

    // For every row of `queries` (rows of the same matrix), its k nearest
    // candidates, nearest first, ties to the lower row; never itself. With
    // `hides`, candidates it hides are skipped; `radius` (one per query, NaN
    // for none) is the km within which `hides` hides every candidate, which
    // lets the search pass over them.
    [[nodiscard]] PlaceNeighbours nearest(const std::vector<int64_t>& queries, int64_t k,
                                         const DistanceHiding& hides = {},
                                         const std::vector<double>& radius = {}) const;

private:
    struct Node {
        double lo[3];
        double hi[3];
        int64_t begin;
        int64_t end;
        int32_t left;
        int32_t right;
    };
    std::vector<double> lat_;
    std::vector<double> lon_;
    std::vector<double> xyz_;
    std::vector<int64_t> order_;
    std::vector<Node> nodes_;
    int64_t n_candidates_;

    int32_t build(int64_t begin, int64_t end);
};

}  // namespace resolve
