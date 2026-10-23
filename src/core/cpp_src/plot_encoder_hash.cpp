#include "resolve/plot_encoder_hash.hpp"

namespace resolve {

namespace {

class HashFeatures final : public SpeciesFeatures {
public:
    [[nodiscard]] int64_t width() const noexcept override { return 0; }
    torch::Tensor forward(const PlotInputs&, bool) override { return {}; }
    // The hash is parameter-free.
    void append_composition_parameters(std::vector<torch::Tensor>&) const override {}
};

}  // namespace

std::unique_ptr<SpeciesFeatures> make_hash_features(torch::nn::Module&,
                                                    const SpeciesEncoderSpec&) {
    return std::make_unique<HashFeatures>();
}

PlotEncoderImpl::PlotEncoderImpl(const SpeciesEncoderSpec& spec) {
    auto features = make_hash_features(*this, spec);
    auto slots = make_per_rank_slot_taxonomy(*this, spec);
    assemble(std::move(features), std::move(slots),
             spec.n_continuous + spec.config.hash_dim, spec);
}

}  // namespace resolve
