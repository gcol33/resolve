#include "resolve/plot_encoder_rank_pool.hpp"

#include "resolve/species_pool.hpp"

namespace resolve {

namespace {

class PooledSpeciesFeatures final : public SpeciesFeatures {
public:
    PooledSpeciesFeatures(torch::nn::Module& owner, const SpeciesEncoderSpec& spec)
        : tables_(owner, spec, spec.config.species_embed_dim, spec.config.genus_emb_dim,
                  spec.config.family_emb_dim),
          aggregation_(spec.aggregation),
          cover_dropout_(spec.config.cover_dropout) {
        width_ = spec.config.species_embed_dim + 1;
        if (tables_.has_taxonomy()) {
            width_ += spec.config.genus_emb_dim + spec.config.family_emb_dim;
        }
    }

    [[nodiscard]] int64_t width() const noexcept override { return width_; }

    torch::Tensor forward(const PlotInputs& inputs, bool training) override {
        auto pool = PoolInputs::from(inputs);
        pool.apply_cover_dropout(training, cover_dropout_);
        const auto w = pool.normalized(pool.weights);

        std::vector<torch::Tensor> parts{weighted_pool(pool.species_ids, tables_.species, w)};
        if (tables_.has_taxonomy()) {
            check_concat_taxonomy(true, pool.genus_ids, pool.family_ids, "PlotEncoderRankPool");
            const auto taxon_w = pool.taxonomy_weights(aggregation_, w);
            parts.push_back(weighted_pool(pool.genus_ids, tables_.genus, taxon_w));
            parts.push_back(weighted_pool(pool.family_ids, tables_.family, taxon_w));
        }
        parts.push_back(pool.has_cover.unsqueeze(-1));
        return torch::cat(parts, /*dim=*/1);
    }

    [[nodiscard]] torch::Tensor species_weights() const override {
        return tables_.species_weights();
    }
    [[nodiscard]] torch::Tensor genus_weights() const override { return tables_.genus_weights(); }
    [[nodiscard]] torch::Tensor family_weights() const override {
        return tables_.family_weights();
    }
    void append_composition_parameters(std::vector<torch::Tensor>& out) const override {
        tables_.append_composition_parameters(out);
    }

private:
    SpeciesTables tables_;
    AggregationMode aggregation_;
    float cover_dropout_;
    int64_t width_ = 0;
};

}  // namespace

std::unique_ptr<SpeciesFeatures> make_rank_pool_features(torch::nn::Module& owner,
                                                         const SpeciesEncoderSpec& spec) {
    return std::make_unique<PooledSpeciesFeatures>(owner, spec);
}

PlotEncoderRankPoolImpl::PlotEncoderRankPoolImpl(const SpeciesEncoderSpec& spec) {
    assemble(make_rank_pool_features(*this, spec), nullptr, spec.n_continuous, spec);
}

}  // namespace resolve
