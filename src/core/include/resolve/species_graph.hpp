#pragma once

#include "resolve/dataset.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <cstdint>

namespace resolve {

// What relation an edge of the species graph stands for. The value is the index
// the encoder's edge-type embedding looks up, so the numbering is part of a
// trained model and must not be renumbered.
enum class SpeciesEdgeType : int64_t {
    SameGenus = 0,
    SameFamily = 1,
    CoOccurrence = 2,
};

// A typed graph over the species vocabulary: node i is species code i, so node
// 0 is the reserved <UNK> species and carries no edges. Edges are directed and
// emitted in both directions, which is what message passing reads.
struct SpeciesGraph {
    torch::Tensor edge_index;  // (2, n_edges) int64: row 0 source, row 1 target
    torch::Tensor edge_type;   // (n_edges,) int64, a SpeciesEdgeType value
    int64_t n_species = 0;

    [[nodiscard]] int64_t n_edges() const {
        return edge_index.defined() ? edge_index.size(1) : 0;
    }
    [[nodiscard]] bool empty() const { return n_edges() == 0; }
};

// Build the species graph a HeterogeneousGNN passes messages on from the
// dataset's own taxonomy and co-occurrence, as HeterogeneousGNNConfig asks for
// it:
//
//   use_taxonomic_edges     two species sharing a genus are joined by a
//                           SameGenus edge, two sharing a family by a
//                           SameFamily edge. Read from the dataset's
//                           species_genus_ids() / species_family_ids().
//   use_cooccurrence_edges  two species recorded in the same plot in at least
//                           `cooccurrence_threshold` of the plots are joined by
//                           a CoOccurrence edge, keeping each species' strongest
//                           `k_cooccurrence` partners. Read from the dataset's
//                           species_vector(), so the sparse species encoding.
//
// A pair that stands in two relations gets one edge of each type. The graph is
// built once from the training data and travels in the checkpoint, so scoring
// reads the same graph the weights were trained on.
//
// Refuses, rather than quietly returning a graph the configuration did not ask
// for: a requested relation whose input the dataset does not carry, both
// relations switched off, and a taxonomic relation so dense that its edge list
// would not fit in memory all raise.
SpeciesGraph build_species_graph(const ResolveDataset& dataset,
                                 const HeterogeneousGNNConfig& config);

// Ceiling on the number of edges either relation may produce. A genus of G
// species contributes G * (G - 1) directed edges, which grows quadratically in
// the largest group, so a vocabulary with one enormous family can ask for an
// edge list far larger than the graph it describes. Hitting this raises with
// the count rather than exhausting memory.
inline constexpr int64_t kMaxSpeciesGraphEdges = 20'000'000;

}  // namespace resolve
