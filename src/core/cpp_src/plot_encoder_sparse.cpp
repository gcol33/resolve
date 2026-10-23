#include "resolve/plot_encoder_sparse.hpp"

#include "resolve/sparse_rows.hpp"

namespace resolve {

namespace {

class ProjectedSpeciesFeatures final : public SpeciesFeatures {
public:
    ProjectedSpeciesFeatures(torch::nn::Module& owner, const SpeciesEncoderSpec& spec)
        : projection_(owner.register_module(
              "species_projection",
              torch::nn::Linear(spec.n_species, spec.config.species_embed_dim))) {}

    [[nodiscard]] int64_t width() const noexcept override {
        return projection_->weight.size(0);
    }

    torch::Tensor forward(const PlotInputs& inputs, bool) override {
        require_sparse_rows(inputs.species_vector, "PlotEncoderSparse: species_vector");
        return sparse_rows_linear(inputs.species_vector, projection_->weight, projection_->bias);
    }

    // Column j of the projection's weight is species j's vector, so the
    // projection is the sparse encoding's species table.
    void append_composition_parameters(std::vector<torch::Tensor>& out) const override {
        out.push_back(projection_->weight);
        out.push_back(projection_->bias);
    }

private:
    torch::nn::Linear projection_;
};

}  // namespace

std::unique_ptr<SpeciesFeatures> make_sparse_features(torch::nn::Module& owner,
                                                      const SpeciesEncoderSpec& spec) {
    return std::make_unique<ProjectedSpeciesFeatures>(owner, spec);
}

PlotEncoderSparseImpl::PlotEncoderSparseImpl(const SpeciesEncoderSpec& spec) {
    auto features = make_sparse_features(*this, spec);
    auto slots = make_per_rank_slot_taxonomy(*this, spec);
    assemble(std::move(features), std::move(slots), spec.n_continuous, spec);
}

}  // namespace resolve
