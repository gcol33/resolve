#include "resolve/plot_encoder_embed.hpp"

namespace resolve {

namespace {

class FusedSpeciesFeatures final : public SpeciesFeatures {
public:
    FusedSpeciesFeatures(torch::nn::Module& owner, const SpeciesEncoderSpec& spec)
        : table_(owner.register_module(
              "fused_species",
              FusedPositionalEmbedding(spec.n_species, spec.config.top_k_species,
                                       spec.config.species_embed_dim))) {}

    [[nodiscard]] int64_t width() const noexcept override { return table_->total_output_dim(); }

    torch::Tensor forward(const PlotInputs& inputs, bool) override {
        return table_->forward(inputs.species_ids);
    }

    [[nodiscard]] torch::Tensor species_weights() const override {
        return table_->position_average();
    }

    void append_composition_parameters(std::vector<torch::Tensor>& out) const override {
        out.push_back(table_->embedding()->weight);
    }

private:
    FusedPositionalEmbedding table_;
};

}  // namespace

std::unique_ptr<SpeciesFeatures> make_embed_features(torch::nn::Module& owner,
                                                     const SpeciesEncoderSpec& spec) {
    return std::make_unique<FusedSpeciesFeatures>(owner, spec);
}

PlotEncoderEmbedImpl::PlotEncoderEmbedImpl(const SpeciesEncoderSpec& spec) {
    auto features = make_embed_features(*this, spec);
    auto slots = make_fused_slot_taxonomy(*this, spec);
    assemble(std::move(features), std::move(slots), spec.n_continuous, spec);
}

}  // namespace resolve
