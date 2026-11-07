#include "resolve/retrieval.hpp"
#include "resolve/geo.hpp"
#include "resolve/sparse_rows.hpp"
#include "resolve/target_labels.hpp"

#include <algorithm>
#include <limits>
#include <optional>
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

constexpr float kHidden = -std::numeric_limits<float>::infinity();

// Under a leave-one-out search query row i is reference row i: the score of
// that pair, where the block of queries [q0, q1) meets the block of reference
// rows [r0, r1), is set to -inf so the top k never keeps it.
void mask_self_pairs(torch::Tensor& scores, int64_t q0, int64_t q1, int64_t r0, int64_t r1) {
    const int64_t lo = std::max(q0, r0);
    const int64_t hi = std::min(q1, r1);
    if (lo >= hi) return;
    const torch::Tensor rows =
        torch::arange(lo, hi, torch::TensorOptions().dtype(torch::kInt64).device(scores.device()));
    scores.index_put_({rows - q0, rows - r0}, kHidden);
}

// A block of scores no hiding rule touches.
struct NoMask {
    void operator()(torch::Tensor&, int64_t, int64_t) const {}
};

// The k largest scores of each query row over the reference rows, largest
// first. The queries are taken at most `query_rows_cap` rows at a time
// (`query_block(q0, q1)` returns them ready for scoring) and each block is
// scored against kBlockElements / its rows reference rows at a time
// (`score_block(queries, r0, r1)` returns the (q1 - q0, r1 - r0) scores, and
// `mask_block(scores, r0, r1)` may set hidden pairs to -inf), each block's top
// k merged into the running top k, so memory stays bounded and the result is
// exact. k is at most the reference size, one less with `exclude_self`, so
// the masked self pair is never among the k kept. A kept score of -inf (a row
// with fewer than k visible reference rows) is returned as index -1 with a
// NaN score.
template <typename QueryBlock, typename ScoreBlock, typename MaskBlock = NoMask>
Neighbours blocked_top_k(int64_t n_queries,
                         int64_t n_reference,
                         int64_t k,
                         bool exclude_self,
                         const torch::Device& device,
                         QueryBlock query_block,
                         ScoreBlock score_block,
                         MaskBlock mask_block = {},
                         int64_t query_rows_cap = RetrievalIndex::kQueryBlockRows) {
    k = std::max<int64_t>(0, std::min(k, exclude_self ? n_reference - 1 : n_reference));
    const auto index_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    const auto score_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    if (n_queries == 0 || k == 0) {
        return {torch::empty({n_queries, k}, index_options),
                torch::empty({n_queries, k}, score_options)};
    }
    const Float32Matmul full_precision;
    const int64_t query_rows = std::clamp<int64_t>(query_rows_cap, 1, n_queries);
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
            mask_block(scores, r0, r1);
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
        const torch::Tensor missing = torch::isneginf(best_similarities);
        best_indices.masked_fill_(missing, -1);
        best_similarities.masked_fill_(missing, std::numeric_limits<float>::quiet_NaN());
        indices.push_back(std::move(best_indices));
        similarities.push_back(std::move(best_similarities));
    }
    if (indices.size() == 1) return {indices.front(), similarities.front()};
    return {torch::cat(indices, 0), torch::cat(similarities, 0)};
}

void require_matrix(const torch::Tensor& t, const char* fn, const char* name) {
    if (!t.defined() || t.dim() != 2) {
        throw std::invalid_argument(std::string(fn) + ": " + name + " must be a (rows, dim) matrix");
    }
    if (t.layout() != torch::kStrided && !is_sparse_rows(t)) {
        throw std::invalid_argument(std::string(fn) + ": " + name +
                                    " must be a dense matrix or sparse rows");
    }
    if (!is_sparse_rows(t) && !t.is_floating_point()) {
        throw std::invalid_argument(std::string(fn) + ": " + name +
                                    " must hold floating-point values");
    }
}

void require_search_shapes(const torch::Tensor& queries, const torch::Tensor& reference, int64_t k,
                           bool exclude_self, const char* fn) {
    require_matrix(queries, fn, "queries");
    require_matrix(reference, fn, "reference");
    if (queries.size(1) != reference.size(1)) {
        throw std::invalid_argument(std::string(fn) + ": queries have " +
                                    std::to_string(queries.size(1)) + " columns, reference " +
                                    std::to_string(reference.size(1)));
    }
    if (k < 1) throw std::invalid_argument(std::string(fn) + ": k must be at least 1");
    if (exclude_self && queries.size(0) != reference.size(0)) {
        throw std::invalid_argument(
            std::string(fn) + ": exclude_self needs query row i to be reference row i, but there "
            "are " + std::to_string(queries.size(0)) + " queries and " +
            std::to_string(reference.size(0)) + " reference rows");
    }
}

// Cosine similarities of a block of query rows against reference rows. The
// query block is held dense at unit length; a dense reference block is brought
// to unit length as it is scored, and a sparse reference set is scaled to unit
// rows once (its entries only) and multiplied entry by entry, so no dense copy
// of a whole set is held.
class CosineScores {
public:
    CosineScores(const torch::Tensor& queries, const torch::Tensor& reference,
                 const torch::Device& device)
        : queries_(queries),
          reference_(is_sparse_rows(reference)
                         ? sparse_rows_unit_length(reference.to(device, torch::kFloat32))
                         : reference),
          device_(device) {}

    // Query rows per block: few enough that the dense block of a wide sparse
    // set stays within kBlockElements values.
    [[nodiscard]] int64_t query_rows_cap() const {
        if (!is_sparse_rows(queries_)) return RetrievalIndex::kQueryBlockRows;
        return std::clamp<int64_t>(
            RetrievalIndex::kBlockElements / std::max<int64_t>(1, queries_.size(1)), 1,
            RetrievalIndex::kQueryBlockRows);
    }

    [[nodiscard]] torch::Tensor query_block(int64_t q0, int64_t q1) const {
        torch::Tensor rows = slice_rows(queries_, q0, q1).to(device_);
        if (is_sparse_rows(rows)) rows = sparse_rows_to_dense(rows);
        return unit_rows(rows);
    }

    [[nodiscard]] torch::Tensor score(const torch::Tensor& query_rows, int64_t r0, int64_t r1) const {
        const torch::Tensor block = slice_rows(reference_, r0, r1);
        if (is_sparse_rows(block)) {
            return sparse_rows_matmul(block, query_rows.t().contiguous()).t();
        }
        return query_rows.matmul(unit_rows(block).t());
    }

private:
    torch::Tensor queries_;
    torch::Tensor reference_;
    torch::Device device_;
};

// Great-circle distances in km, float64, of a block of query positions against
// reference positions, both (rows, 2) latitude / longitude.
torch::Tensor block_km(const torch::Tensor& query_rows, const torch::Tensor& reference_rows) {
    return great_circle_km(query_rows.select(1, 0).unsqueeze(1), query_rows.select(1, 1).unsqueeze(1),
                           reference_rows.select(1, 0).unsqueeze(0),
                           reference_rows.select(1, 1).unsqueeze(0));
}

// The rules of one NeighbourHiding, evaluated block by block. A search hands
// over what it already computed for its own score (the similarities of a
// cosine search, the distances of a great-circle one); every other rule is
// computed here from the hiding's axes.
class HidingMask {
public:
    HidingMask(const NeighbourHiding& hiding, int64_t n_queries, int64_t n_reference,
               const torch::Device& device, bool own_similarity, bool own_km)
        : hiding_(hiding), device_(device) {
        const HidingRadii& r = hiding.radii;
        auto check_radius = [&](const torch::Tensor& radius, const char* axis) {
            if (radius.defined() && (radius.dim() != 1 || radius.size(0) != n_queries)) {
                throw std::invalid_argument(std::string("neighbour hiding: the ") + axis +
                                            " radius must hold one value per query row");
            }
        };
        check_radius(r.similarity, "similarity");
        check_radius(r.km, "km");
        check_radius(r.years, "years");
        auto check_axis = [&](const torch::Tensor& q, const torch::Tensor& ref, const char* axis) {
            if (!q.defined() || !ref.defined() || q.size(0) != n_queries ||
                ref.size(0) != n_reference) {
                throw std::invalid_argument(std::string("neighbour hiding: the ") + axis +
                                            " rule needs the " + axis +
                                            " of every query and reference row");
            }
        };
        if (r.similarity.defined() && !own_similarity) {
            check_axis(hiding.queries.species, hiding.reference.species, "species");
            similarity_.emplace(hiding.queries.species, hiding.reference.species, device_);
        }
        if (r.km.defined() && !own_km) {
            check_axis(hiding.queries.coordinates, hiding.reference.coordinates, "coordinates");
            query_coordinates_ = hiding.queries.coordinates.to(torch::kCPU, torch::kFloat64);
            reference_coordinates_ = hiding.reference.coordinates.to(torch::kCPU, torch::kFloat64);
        }
        if (r.years.defined()) {
            check_axis(hiding.queries.years, hiding.reference.years, "years");
            query_years_ = hiding.queries.years.to(device_, torch::kFloat64);
            reference_years_ = hiding.reference.years.to(device_, torch::kFloat64).unsqueeze(0);
        }
    }

    // Starts a block of query rows [q0, q1).
    void begin_queries(int64_t q0, int64_t q1) {
        const HidingRadii& r = hiding_.radii;
        q0_ = q0;
        q1_ = q1;
        if (similarity_) block_species_ = similarity_->query_block(q0, q1);
        if (r.similarity.defined()) {
            similarity_radius_ = r.similarity.slice(0, q0, q1).to(device_, torch::kFloat32).unsqueeze(1);
        }
        if (r.km.defined()) {
            km_radius_ = r.km.slice(0, q0, q1).to(torch::kCPU, torch::kFloat64).unsqueeze(1);
        }
        if (r.years.defined()) {
            years_radius_ = r.years.slice(0, q0, q1).to(device_, torch::kFloat64).unsqueeze(1);
        }
    }

    // Sets the scores of the pairs (current query block) x [r0, r1) the rules
    // hide to -inf. `similarity` and `km` are the search's own values for the
    // block when it has them, undefined otherwise.
    void apply(torch::Tensor& scores, int64_t r0, int64_t r1, const torch::Tensor& similarity,
               const torch::Tensor& km) const {
        const HidingRadii& r = hiding_.radii;
        torch::Tensor hidden = torch::zeros(scores.sizes(), scores.options().dtype(torch::kBool));
        if (r.similarity.defined()) {
            const torch::Tensor s =
                similarity.defined() ? similarity : similarity_->score(block_species_, r0, r1);
            hidden |= s > similarity_radius_;
        }
        if (r.km.defined()) {
            const torch::Tensor d =
                km.defined() ? km
                             : block_km(query_coordinates_.slice(0, q0_, q1_),
                                        reference_coordinates_.slice(0, r0, r1));
            hidden |= (d < km_radius_).to(scores.device());
        }
        if (r.years.defined()) {
            const torch::Tensor gap = (query_years_.slice(0, q0_, q1_).unsqueeze(1) -
                                       reference_years_.slice(1, r0, r1))
                                          .abs();
            hidden |= gap < years_radius_;
        }
        scores.masked_fill_(hidden, kHidden);
    }

private:
    const NeighbourHiding& hiding_;
    torch::Device device_;
    std::optional<CosineScores> similarity_;
    torch::Tensor query_coordinates_;
    torch::Tensor reference_coordinates_;
    torch::Tensor query_years_;
    torch::Tensor reference_years_;
    int64_t q0_ = 0;
    int64_t q1_ = 0;
    torch::Tensor block_species_;
    torch::Tensor similarity_radius_;
    torch::Tensor km_radius_;
    torch::Tensor years_radius_;
};

void require_coordinates(const torch::Tensor& t, const char* name) {
    if (!t.defined() || t.dim() != 2 || t.size(1) != 2 || !t.is_floating_point()) {
        throw std::invalid_argument(std::string("nearest_by_great_circle: ") + name +
                                    " must be a (rows, 2) floating matrix of latitude, longitude");
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
                             bool exclude_self,
                             const NeighbourHiding* hiding) {
    require_search_shapes(queries, reference, k, exclude_self, "nearest_by_cosine");
    const torch::Device device = reference.device();
    const CosineScores scores(queries, reference, device);
    std::optional<HidingMask> mask;
    if (hiding != nullptr) {
        mask.emplace(*hiding, queries.size(0), reference.size(0), device, /*own_similarity=*/true,
                     /*own_km=*/false);
    }
    return blocked_top_k(
        queries.size(0), reference.size(0), k, exclude_self, device,
        [&](int64_t q0, int64_t q1) {
            if (mask) mask->begin_queries(q0, q1);
            return scores.query_block(q0, q1);
        },
        [&](const torch::Tensor& query_rows, int64_t r0, int64_t r1) {
            return scores.score(query_rows, r0, r1);
        },
        [&](torch::Tensor& block, int64_t r0, int64_t r1) {
            // The similarity rule reads the block before any pair is hidden.
            if (mask) mask->apply(block, r0, r1, /*similarity=*/block.clone(), /*km=*/{});
        },
        scores.query_rows_cap());
}

PlaceNeighbours nearest_by_great_circle(const torch::Tensor& queries,
                                        const torch::Tensor& reference,
                                        int64_t k,
                                        bool exclude_self,
                                        const NeighbourHiding* hiding) {
    require_coordinates(queries, "queries");
    require_coordinates(reference, "reference");
    require_search_shapes(queries, reference, k, exclude_self, "nearest_by_great_circle");
    const torch::Device device = torch::kCPU;
    const torch::Tensor query_positions = queries.to(device, torch::kFloat64);
    const torch::Tensor reference_positions = reference.to(device, torch::kFloat64);
    std::optional<HidingMask> mask;
    if (hiding != nullptr) {
        mask.emplace(*hiding, queries.size(0), reference.size(0), device, /*own_similarity=*/false,
                     /*own_km=*/true);
    }
    // The block's distances, read by the hiding rule; the score is their
    // negative, -inf where a coordinate is missing.
    torch::Tensor block_distances;
    const Neighbours found = blocked_top_k(
        queries.size(0), reference.size(0), k, exclude_self, device,
        [&](int64_t q0, int64_t q1) {
            if (mask) mask->begin_queries(q0, q1);
            return query_positions.slice(0, q0, q1);
        },
        [&](const torch::Tensor& query_rows, int64_t r0, int64_t r1) {
            block_distances = block_km(query_rows, reference_positions.slice(0, r0, r1));
            return torch::nan_to_num(-block_distances, /*nan=*/-std::numeric_limits<double>::infinity())
                .to(torch::kFloat32);
        },
        [&](torch::Tensor& block, int64_t r0, int64_t r1) {
            if (mask) mask->apply(block, r0, r1, /*similarity=*/{}, block_distances);
        });
    // The distances of the neighbours kept, recomputed in float64 from the
    // positions rather than read back from the float32 scores.
    const torch::Tensor has = found.indices >= 0;
    const torch::Tensor picked = reference_positions.index({found.indices.clamp_min(0)});
    torch::Tensor km = great_circle_km(query_positions.select(1, 0).unsqueeze(1),
                                       query_positions.select(1, 1).unsqueeze(1),
                                       picked.select(2, 0), picked.select(2, 1));
    km = torch::where(has, km, torch::full_like(km, std::numeric_limits<double>::quiet_NaN()));
    return {found.indices, std::move(km)};
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
