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

// Full float32 matrix products for the lifetime of the guard. Training turns
// TF32 on for cuBLAS process-wide (TrainConfig::allow_tf32), whose 10-bit
// mantissa moves cosine similarities by about 1e-3 and reorders near ties;
// an exact search must not depend on whether a model was trained earlier in
// the process.
class Float32Matmul {
public:
    Float32Matmul() : previous_(at::globalContext().allowTF32CuBLAS()) {
        at::globalContext().setAllowTF32CuBLAS(false);
    }
    ~Float32Matmul() { at::globalContext().setAllowTF32CuBLAS(previous_); }
    Float32Matmul(const Float32Matmul&) = delete;
    Float32Matmul& operator=(const Float32Matmul&) = delete;

private:
    bool previous_;
};

// Under a leave-one-out search query row i is reference row i: the score of
// that pair, where the block of queries [q0, q1) meets the block of reference
// rows [r0, r1), is set to -inf so the top k never keeps it.
void mask_self_pairs(torch::Tensor& scores, int64_t q0, int64_t q1, int64_t r0, int64_t r1) {
    const int64_t lo = std::max(q0, r0);
    const int64_t hi = std::min(q1, r1);
    if (lo >= hi) return;
    const torch::Tensor rows =
        torch::arange(lo, hi, torch::TensorOptions().dtype(torch::kInt64).device(scores.device()));
    scores.index_put_({rows - q0, rows - r0}, -std::numeric_limits<float>::infinity());
}

// The k largest scores of each query row over the reference rows, most similar
// first. The queries are taken kQueryBlockRows rows at a time
// (`query_block(q0, q1)` returns them ready for scoring) and each block is
// scored against kBlockElements / its rows reference rows at a time
// (`score_block(queries, r0, r1)` returns the (q1 - q0, r1 - r0) scores), each
// block's top k merged into the running top k, so memory stays bounded and the
// result is exact. k is at most the reference size, one less with
// `exclude_self`, so a masked pair is never among the k kept.
template <typename QueryBlock, typename ScoreBlock>
Neighbours blocked_top_k(int64_t n_queries,
                         int64_t n_reference,
                         int64_t k,
                         bool exclude_self,
                         const torch::Device& device,
                         QueryBlock query_block,
                         ScoreBlock score_block) {
    k = std::max<int64_t>(0, std::min(k, exclude_self ? n_reference - 1 : n_reference));
    const auto index_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    const auto score_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    if (n_queries == 0 || k == 0) {
        return {torch::empty({n_queries, k}, index_options),
                torch::empty({n_queries, k}, score_options)};
    }
    const Float32Matmul full_precision;
    const int64_t query_rows = std::min(n_queries, RetrievalIndex::kQueryBlockRows);
    const int64_t reference_rows = std::max<int64_t>(1, RetrievalIndex::kBlockElements / query_rows);

    std::vector<torch::Tensor> indices;
    std::vector<torch::Tensor> similarities;
    for (int64_t q0 = 0; q0 < n_queries; q0 += query_rows) {
        const int64_t q1 = std::min(q0 + query_rows, n_queries);
        const torch::Tensor queries = query_block(q0, q1);
        torch::Tensor best_similarities;
        torch::Tensor best_indices;
        for (int64_t r0 = 0; r0 < n_reference; r0 += reference_rows) {
            const int64_t r1 = std::min(r0 + reference_rows, n_reference);
            torch::Tensor scores = score_block(queries, r0, r1);
            if (exclude_self) mask_self_pairs(scores, q0, q1, r0, r1);
            auto [block_similarities, block_indices] = scores.topk(std::min(k, r1 - r0), /*dim=*/1);
            block_indices.add_(r0);
            if (!best_similarities.defined()) {
                best_similarities = std::move(block_similarities);
                best_indices = std::move(block_indices);
                continue;
            }
            const torch::Tensor merged_similarities =
                torch::cat({best_similarities, block_similarities}, 1);
            const torch::Tensor merged_indices = torch::cat({best_indices, block_indices}, 1);
            auto [kept_similarities, position] = merged_similarities.topk(k, /*dim=*/1);
            best_similarities = std::move(kept_similarities);
            best_indices = merged_indices.gather(1, position);
        }
        indices.push_back(std::move(best_indices));
        similarities.push_back(std::move(best_similarities));
    }
    if (indices.size() == 1) return {indices.front(), similarities.front()};
    return {torch::cat(indices, 0), torch::cat(similarities, 0)};
}

void require_dense_matrix(const torch::Tensor& t, const char* name) {
    if (!t.defined() || t.dim() != 2) {
        throw std::invalid_argument(std::string("nearest_by_cosine: ") + name +
                                    " must be a (rows, dim) matrix");
    }
    if (t.layout() != torch::kStrided) {
        throw std::invalid_argument(std::string("nearest_by_cosine: ") + name +
                                    " must be a dense matrix");
    }
    if (!t.is_floating_point()) {
        throw std::invalid_argument(std::string("nearest_by_cosine: ") + name +
                                    " must hold floating-point values");
    }
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
                             int64_t k,
                             bool exclude_self) {
    require_dense_matrix(queries, "queries");
    require_dense_matrix(reference, "reference");
    if (queries.size(1) != reference.size(1)) {
        throw std::invalid_argument("nearest_by_cosine: queries have " +
                                    std::to_string(queries.size(1)) + " columns, reference " +
                                    std::to_string(reference.size(1)));
    }
    if (k < 1) throw std::invalid_argument("nearest_by_cosine: k must be at least 1");
    if (exclude_self && queries.size(0) != reference.size(0)) {
        throw std::invalid_argument(
            "nearest_by_cosine: exclude_self needs query row i to be reference row i, but there "
            "are " + std::to_string(queries.size(0)) + " queries and " +
            std::to_string(reference.size(0)) + " reference rows");
    }
    const torch::Device device = reference.device();
    // Each block of reference rows is brought to unit length as it is scored,
    // so no normalised copy of either whole set is held.
    return blocked_top_k(
        queries.size(0), reference.size(0), k, exclude_self, device,
        [&](int64_t q0, int64_t q1) {
            return unit_rows(queries.slice(0, q0, q1).to(device, torch::kFloat32));
        },
        [&](const torch::Tensor& query_rows, int64_t r0, int64_t r1) {
            return query_rows.matmul(unit_rows(reference.slice(0, r0, r1)).t());
        });
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
    const torch::Tensor queries = unit_rows(query_latent.to(latents_.device()));
    const Neighbours neighbours = blocked_top_k(
        queries.size(0), n_reference(), config_.k_neighbors, /*exclude_self=*/false,
        latents_.device(),
        [&](int64_t q0, int64_t q1) { return queries.slice(0, q0, q1); },
        [&](const torch::Tensor& query_rows, int64_t r0, int64_t r1) {
            return query_rows.matmul(latents_.slice(0, r0, r1).t());
        });
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
