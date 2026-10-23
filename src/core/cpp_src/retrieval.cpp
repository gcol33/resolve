#include "resolve/retrieval.hpp"
#include "resolve/target_labels.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace resolve {

namespace {

torch::Tensor unit_rows(const torch::Tensor& rows) {
    return torch::nn::functional::normalize(
        rows.to(torch::kFloat32),
        torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(1).eps(1e-12));
}

// The k largest inner products of each query row with the reference rows,
// computed one block of reference rows at a time and merged into a running
// top k. Both inputs are unit rows on one device.
Neighbours top_k_inner_product(const torch::Tensor& queries,
                               const torch::Tensor& reference,
                               int64_t k) {
    const int64_t n_queries = queries.size(0);
    const int64_t n_reference = reference.size(0);
    k = std::min(k, n_reference);
    const auto index_options = torch::TensorOptions().dtype(torch::kInt64).device(queries.device());
    if (n_queries == 0 || k == 0) {
        return {torch::empty({n_queries, k}, index_options),
                torch::empty({n_queries, k}, queries.options())};
    }
    const int64_t block =
        std::max<int64_t>(1, RetrievalIndex::kBlockElements / n_queries);

    torch::Tensor best_similarities;
    torch::Tensor best_indices;
    for (int64_t start = 0; start < n_reference; start += block) {
        const int64_t end = std::min(start + block, n_reference);
        const torch::Tensor similarities = queries.matmul(reference.slice(0, start, end).t());
        auto [block_similarities, block_indices] =
            similarities.topk(std::min(k, end - start), /*dim=*/1);
        block_indices.add_(start);
        if (!best_similarities.defined()) {
            best_similarities = std::move(block_similarities);
            best_indices = std::move(block_indices);
            continue;
        }
        const torch::Tensor merged_similarities = torch::cat({best_similarities, block_similarities}, 1);
        const torch::Tensor merged_indices = torch::cat({best_indices, block_indices}, 1);
        auto [kept_similarities, position] = merged_similarities.topk(k, /*dim=*/1);
        best_similarities = std::move(kept_similarities);
        best_indices = merged_indices.gather(1, position);
    }
    return {best_indices, best_similarities};
}

}  // namespace

void RetrievalConfig::validate() const {
    if (k_neighbors < 1) {
        throw std::invalid_argument("retrieval: k_neighbors must be at least 1, got " +
                                    std::to_string(k_neighbors));
    }
    if (!(alpha >= 0.0f && alpha <= 1.0f)) {
        throw std::invalid_argument("retrieval: alpha must be in [0, 1], got " +
                                    std::to_string(alpha));
    }
    if (!(temperature > 0.0f)) {
        throw std::invalid_argument("retrieval: temperature must be positive, got " +
                                    std::to_string(temperature));
    }
}

Neighbours nearest_by_cosine(const torch::Tensor& queries,
                             const torch::Tensor& reference,
                             int64_t k) {
    if (queries.dim() != 2 || reference.dim() != 2 || queries.size(1) != reference.size(1)) {
        throw std::invalid_argument(
            "nearest_by_cosine: queries and reference must be (rows, dim) with one dim");
    }
    if (k < 1) throw std::invalid_argument("nearest_by_cosine: k must be at least 1");
    return top_k_inner_product(unit_rows(queries), unit_rows(reference), k);
}

RetrievalIndex::RetrievalIndex(const torch::Tensor& latents,
                               std::vector<TargetConfig> targets,
                               const TensorMap& labels,
                               RetrievalConfig config)
    : targets_(std::move(targets)), config_(config) {
    config_.validate();
    if (!latents.defined() || latents.dim() != 2 || latents.size(0) == 0) {
        throw std::invalid_argument("retrieval: the reference set holds no plot");
    }
    latents_ = unit_rows(latents.detach());
    const int64_t n_reference = latents_.size(0);
    for (const TargetConfig& target : targets_) {
        const auto it = labels.find(target.name);
        if (it == labels.end() || !it->second.defined()) {
            throw std::invalid_argument("retrieval: the reference set carries no labels for target '" +
                                        target.name + "'");
        }
        const torch::Tensor& values = it->second;
        if (values.dim() != 1 || values.size(0) != n_reference) {
            throw std::invalid_argument("retrieval: target '" + target.name + "' has " +
                                        std::to_string(values.numel()) +
                                        " labels for " + std::to_string(n_reference) +
                                        " reference plots");
        }
        torch::Tensor stored;
        if (target.task == TaskType::Classification) {
            stored = values.to(latents_.device(), torch::kInt64);
            const int64_t max_code = stored.numel() > 0 ? stored.max().item<int64_t>() : -1;
            if (max_code >= target.num_classes) {
                throw std::invalid_argument(
                    "retrieval: target '" + target.name + "' has class code " +
                    std::to_string(max_code) + " in the reference set, outside the model's " +
                    std::to_string(target.num_classes) +
                    " classes; build the reference set with the model's vocabularies");
            }
        } else {
            stored = values.to(latents_.device(), torch::kFloat32);
        }
        labels_[target.name] = std::move(stored);
    }
}

RetrievalIndex RetrievalIndex::to(const torch::Device& device) const {
    RetrievalIndex moved;
    moved.latents_ = latents_.to(device);
    moved.targets_ = targets_;
    moved.labels_ = to_device(labels_, device);
    moved.config_ = config_;
    return moved;
}

void RetrievalIndex::blend(const torch::Tensor& query_latent,
                           TensorMap& predictions,
                           TensorMap& probabilities) const {
    const Neighbours neighbours = top_k_inner_product(
        unit_rows(query_latent.to(latents_.device())), latents_, config_.k_neighbors);
    const torch::Tensor logits = neighbours.similarities / config_.temperature;
    const float alpha = config_.alpha;

    for (const TargetConfig& target : targets_) {
        const auto pred_it = predictions.find(target.name);
        if (pred_it == predictions.end()) continue;
        const torch::Tensor neighbour_labels = labels_.at(target.name).index({neighbours.indices});
        const torch::Tensor labelled =
            labelled_mask(neighbour_labels, target.task).reshape(neighbour_labels.sizes());
        const torch::Tensor any_labelled = labelled.any(/*dim=*/1);
        // Softmax over the labelled neighbours only; a row with none of them
        // labelled gets zero weight (its softmax over -inf is NaN).
        const torch::Tensor weights = torch::where(
            any_labelled.unsqueeze(1),
            torch::softmax(logits.masked_fill(labelled.logical_not(),
                                              -std::numeric_limits<float>::infinity()),
                           /*dim=*/1),
            torch::zeros_like(logits));
        const torch::Tensor blend_rows = any_labelled.to(pred_it->second.device());

        if (target.task == TaskType::Classification) {
            torch::Tensor& probs = probabilities.at(target.name);
            const torch::Tensor codes =
                torch::where(labelled, neighbour_labels, torch::zeros_like(neighbour_labels));
            const torch::Tensor retrieved =
                torch::zeros({codes.size(0), target.num_classes}, weights.options())
                    .scatter_add_(1, codes, weights)
                    .to(probs.device(), probs.scalar_type());
            probs = torch::where(blend_rows.unsqueeze(1),
                                 probs * (1.0f - alpha) + retrieved * alpha, probs);
            pred_it->second = probs.argmax(/*dim=*/1);
        } else {
            torch::Tensor& pred = pred_it->second;
            const torch::Tensor values =
                torch::where(labelled, neighbour_labels, torch::zeros_like(neighbour_labels));
            const torch::Tensor retrieved =
                (weights * values).sum(/*dim=*/1).to(pred.device(), pred.scalar_type());
            pred = torch::where(blend_rows, pred * (1.0f - alpha) + retrieved * alpha, pred);
        }
    }
}

}  // namespace resolve
