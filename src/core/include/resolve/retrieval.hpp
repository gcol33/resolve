#pragma once

// Retrieval-augmented prediction (Gorishniy et al. 2024, "TabR: Tabular Deep
// Learning Meets Nearest Neighbors", ICLR): a trained model's prediction for a
// plot is blended with the labels of the reference plots nearest to it in the
// model's latent space.
//
// Per plot, the k reference plots most similar to it by cosine similarity of
// their latents are weighted by a softmax of similarity / temperature. For a
// target, only the neighbours labelled for it count: their weights are
// renormalised over them, and a plot none of whose neighbours is labelled keeps
// the model's prediction unchanged. Then, with blend share alpha,
//
//   regression      (1 - alpha) * prediction + alpha * sum_j w_j y_j,
//                   both in the target's original units;
//   classification  (1 - alpha) * softmax(logits) + alpha * the neighbours'
//                   label distribution (sum of w_j over the neighbours of each
//                   class), the predicted class its argmax.
//
// The search is exact: brute force over blocks of the reference set, on the
// device the latents live on, so it needs no index structure and no
// dependency. A reference plot that is also scored finds itself first; score
// the reference set only to study that.

#include "resolve/plot_inputs.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace resolve {

struct RetrievalConfig {
    // Reference plots each prediction reads (at most the reference size).
    int k_neighbors = 96;
    // Share of the retrieved label in the blend, in [0, 1]: 0 is the model
    // alone, 1 the neighbours alone.
    float alpha = 0.3f;
    // Scale of the softmax over the neighbours' cosine similarities (> 0): a
    // small value concentrates the weight on the most similar neighbours.
    float temperature = 1.0f;

    // Throws std::invalid_argument naming the first field out of range.
    void validate() const;
};

// The neighbours of each query row, most similar first.
struct Neighbours {
    torch::Tensor indices;       // (n_queries, k) int64 rows of `reference`
    torch::Tensor similarities;  // (n_queries, k) float32 cosine similarities
};

// The nearest places of each query row, nearest first; -1 and NaN where a row
// has fewer than k (a position without coordinates has none).
struct PlaceNeighbours {
    torch::Tensor indices;  // (n_queries, k) int64 rows of the reference
    torch::Tensor km;       // (n_queries, k) float64 great-circle distances
};

// The k rows of `reference` most similar to each row of `queries` by cosine
// similarity, most similar first. Each is a (rows, dim) matrix of one dim,
// dense in any floating type or sparse rows (sparse_rows.hpp), computed in
// float32. The search runs on the device `reference` is on, and the result
// lives there; `queries` may sit elsewhere (a large query set in host memory)
// and is moved over one block of rows at a time.
//
// With `exclude_self`, queries and reference are the same plots in the same
// order: query row i is reference row i and is never returned as its own
// neighbour, while any other row, one with identical values included (a plot
// with the same species list), is. This is the leave-one-out search over one
// set of plots; pass the same tensor twice.
//
// k is clamped to the reference size (one less under `exclude_self`). Exact:
// the similarities are computed in blocks of at most kQueryBlockRows query rows
// and kBlockElements values, each merged into a running top k, so memory stays
// bounded whatever the size of either set. Throws std::invalid_argument for
// shapes that do not match, k < 1, or `exclude_self` with different row
// counts.
[[nodiscard]] Neighbours nearest_by_cosine(const torch::Tensor& queries,
                                           const torch::Tensor& reference,
                                           int64_t k,
                                           bool exclude_self = false);

// The k rows of `reference` nearest to each row of `queries` by great-circle
// distance, nearest first: the same exact blocked search over (rows, 2)
// latitude / longitude matrices in degrees, distances by the haversine
// formula on a sphere of kEarthRadiusKm. A row with a NaN coordinate is never
// a neighbour and has none. `exclude_self` as for nearest_by_cosine. Runs on
// the CPU.
[[nodiscard]] PlaceNeighbours nearest_by_great_circle(const torch::Tensor& queries,
                                                      const torch::Tensor& reference,
                                                      int64_t k,
                                                      bool exclude_self = false);

// The reference plots of a retrieval: their latents and their labels for each
// target of a model.
class RetrievalIndex {
public:
    // Elements of one block of the similarity matrix (queries x reference).
    static constexpr int64_t kBlockElements = int64_t{1} << 24;
    // Query rows of one block: fewer queries search the reference in one
    // pass, more in successive blocks of this many rows.
    static constexpr int64_t kQueryBlockRows = 4096;

    // `latents` (n_reference, latent_dim); `labels` holds for every target of
    // `targets` its (n_reference,) labels: regression values in original units
    // (NaN where a plot is unlabelled), classification codes (kUnlabelledClass
    // where unlabelled). Throws std::invalid_argument for an invalid config, an
    // empty reference, or a target without labels of the reference's length.
    RetrievalIndex(const torch::Tensor& latents,
                   std::vector<TargetConfig> targets,
                   const TensorMap& labels,
                   RetrievalConfig config);

    // Blend one batch of predictions with the labels of the queries'
    // neighbours. `query_latent` (n, latent_dim); `predictions` holds every
    // target's (n,) prediction -- regression in original units, classification
    // codes -- and `probabilities` every classification target's (n, n_classes)
    // softmax rows. Both are updated in place on the device they are on.
    void blend(const torch::Tensor& query_latent,
               TensorMap& predictions,
               TensorMap& probabilities) const;

    // This index with its tensors on `device`.
    [[nodiscard]] RetrievalIndex to(const torch::Device& device) const;

    [[nodiscard]] int64_t n_reference() const { return latents_.size(0); }
    [[nodiscard]] const RetrievalConfig& config() const noexcept { return config_; }

private:
    RetrievalIndex() = default;

    torch::Tensor latents_;  // (n_reference, latent_dim), rows of unit length
    std::vector<TargetConfig> targets_;
    TensorMap labels_;
    RetrievalConfig config_;
};

}  // namespace resolve
