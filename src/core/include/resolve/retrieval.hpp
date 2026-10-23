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

// The k rows of `reference` most similar to each row of `queries` by cosine
// similarity, most similar first. Both are (rows, dim) on one device; k is
// clamped to the reference size. Exact: the similarities are computed in
// blocks of at most kBlockElements and merged into a running top k, so memory
// stays bounded whatever the reference size.
struct Neighbours {
    torch::Tensor indices;       // (n_queries, k) int64 rows of `reference`
    torch::Tensor similarities;  // (n_queries, k) float32 cosine similarities
};
[[nodiscard]] Neighbours nearest_by_cosine(const torch::Tensor& queries,
                                           const torch::Tensor& reference,
                                           int64_t k);

// The reference plots of a retrieval: their latents and their labels for each
// target of a model.
class RetrievalIndex {
public:
    // Elements of one block of the similarity matrix (queries x reference).
    static constexpr int64_t kBlockElements = int64_t{1} << 24;

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
