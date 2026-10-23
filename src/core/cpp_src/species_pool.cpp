#include "resolve/species_pool.hpp"

#include "resolve/types.hpp"

namespace resolve {

namespace {
bool present(const torch::Tensor& t) { return t.defined() && t.numel() > 0; }
}  // namespace

PoolInputs PoolInputs::from(const PlotInputs& inputs) {
    PoolInputs pool;
    pool.species_ids = inputs.species_ids;
    pool.genus_ids = present(inputs.pool_genus_ids) ? inputs.pool_genus_ids : torch::Tensor();
    pool.family_ids = present(inputs.pool_family_ids) ? inputs.pool_family_ids : torch::Tensor();
    const int64_t batch = pool.species_ids.size(0);
    pool.has_cover = present(inputs.pool_has_cover)
        ? inputs.pool_has_cover
        : torch::ones({batch}, torch::TensorOptions().dtype(torch::kFloat32)
                                   .device(pool.species_ids.device()));
    pool.mask = present(inputs.pool_mask) ? inputs.pool_mask : (pool.species_ids != 0);
    pool.weights = present(inputs.pool_weights) ? inputs.pool_weights
                                                : pool.mask.to(torch::kFloat32);
    return pool;
}

void PoolInputs::apply_cover_dropout(bool training, float cover_dropout) {
    if (!training || cover_dropout <= 0.0f) return;
    const auto dropped = torch::rand({species_ids.size(0)},
                                     torch::TensorOptions().device(species_ids.device())) <
                         cover_dropout;
    weights = torch::where(dropped.unsqueeze(1), mask.to(torch::kFloat32), weights);
    has_cover = torch::where(dropped, torch::zeros_like(has_cover), has_cover);
}

torch::Tensor PoolInputs::mask_with_a_key() const {
    const auto valid = mask.to(torch::kBool);
    const auto empty_row = valid.logical_not().all(/*dim=*/1, /*keepdim=*/true);
    const auto first = torch::arange(valid.size(1), torch::TensorOptions()
                                                        .dtype(torch::kLong)
                                                        .device(valid.device()))
                           .eq(0)
                           .unsqueeze(0);
    return valid.logical_or(empty_row.logical_and(first));
}

torch::Tensor PoolInputs::normalized(const torch::Tensor& w) const {
    const auto masked = w * mask.to(torch::kFloat32);
    return masked / masked.sum(/*dim=*/1, /*keepdim=*/true).clamp_min(kEpsilon);
}

torch::Tensor PoolInputs::taxonomy_weights(AggregationMode aggregation,
                                           const torch::Tensor& species_weights) const {
    return aggregation == AggregationMode::Count ? normalized(mask.to(torch::kFloat32))
                                                 : species_weights;
}

torch::Tensor weighted_pool(const torch::Tensor& ids, const torch::nn::Embedding& table,
                            const torch::Tensor& w) {
    namespace F = torch::nn::functional;
    return F::embedding_bag(ids, table->weight,
                            F::EmbeddingBagFuncOptions()
                                .mode(torch::kSum)
                                .per_sample_weights(w)
                                .padding_idx(0));
}

SpeciesTables::SpeciesTables(torch::nn::Module& owner, const SpeciesEncoderSpec& spec,
                             int species_dim, int genus_dim, int family_dim) {
    auto table = [](int64_t rows, int dim) {
        return torch::nn::Embedding(torch::nn::EmbeddingOptions(rows, dim).padding_idx(0));
    };
    species = owner.register_module("species_embedding", table(spec.n_species, species_dim));
    if (spec.has_taxonomy()) {
        genus = owner.register_module("genus_embedding", table(spec.n_genera, genus_dim));
        family = owner.register_module("family_embedding", table(spec.n_families, family_dim));
    }
}

torch::Tensor SpeciesTables::species_weights() const { return species->weight.detach(); }

torch::Tensor SpeciesTables::genus_weights() const {
    return has_taxonomy() ? genus->weight.detach() : torch::Tensor();
}

torch::Tensor SpeciesTables::family_weights() const {
    return has_taxonomy() ? family->weight.detach() : torch::Tensor();
}

void SpeciesTables::append_composition_parameters(std::vector<torch::Tensor>& out) const {
    out.push_back(species->weight);
    if (has_taxonomy()) {
        out.push_back(genus->weight);
        out.push_back(family->weight);
    }
}

}  // namespace resolve
