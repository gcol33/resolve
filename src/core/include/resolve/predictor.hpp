#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "resolve/dataset.hpp"
#include <torch/torch.h>

namespace resolve {

// Forward declaration
class ResolveDataset;

// Predictor for inference with trained models
class Predictor {
public:
    Predictor(
        ResolveModel model,
        Scalers scalers,
        torch::Device device = torch::kCPU
    );

    // Constructor that also carries the categorical vocabulary captured at
    // training time. Used by `load()`.
    //
    // The vocabulary is the model's ID namespace, not a decoder applied to
    // whatever a dataset happens to carry: `predict(ResolveDataset)` consumes
    // `dataset.categorical_ids()` (and the species / taxonomy IDs) as given,
    // because a ResolveDataset retains only encoded codes, never the raw
    // strings they came from. So this vocabulary is used to VALIDATE the
    // dataset: predict() rejects one whose codes mean something else. Build the
    // dataset in the model's namespace to begin with --
    // `ResolveDataset::from_csv_with_vocabs(..., predictor.external_vocabs(), ...)`
    // (or `from_csv_with_schema(..., predictor.schema(), ...)` when the model
    // has no categorical covariates). See gcol33/resolve#102.
    Predictor(
        ResolveModel model,
        Scalers scalers,
        CategoricalVocab categorical_vocab,
        torch::Device device = torch::kCPU
    );

    // Load from saved checkpoint.
    // vram_fraction caps the PyTorch CUDA caching allocator on the target
    // device before model weights are uploaded; matches TrainConfig's default
    // (1.0) so dedicated training/inference jobs on a solo GPU use the full
    // device. Pass an explicit lower value when sharing the GPU with a
    // desktop or other workloads. Ignored when device is CPU.
    static Predictor load(
        const std::string& path,
        torch::Device device = torch::kCPU,
        float vram_fraction = 1.0f
    );

    // Predict on a ResolveDataset (preferred API).
    //
    // The dataset must have been built in the model's integer-code namespace,
    // i.e. with the training vocabularies. Every non-hash encoder indexes an
    // embedding table with a code that is a function of the file the vocab was
    // fitted on (the species vocab is frequency-ranked; the taxonomy and
    // categorical vocabs are functions of their value sets), so a dataset built
    // with a plain `from_csv` on new data carries codes that point at other
    // species' embedding rows. That produces wrong predictions with no error,
    // which is why this rejects such a dataset instead of scoring it:
    //
    //     auto ds = ResolveDataset::from_csv_with_vocabs(
    //         header, species, roles, targets, predictor.external_vocabs(),
    //         dataset_config_from_checkpoint(predictor.schema(),
    //                                        predictor.model()->config()));
    //
    // A vocabulary mismatch throws std::runtime_error naming the offending
    // vocabulary. A checkpoint written before gcol33/resolve#102 carries only
    // the vocabulary SIZES, so only those can be compared; `load()` warns once
    // in that case.
    //
    // `batch_size` controls how the forward pass is chunked along dim 0:
    //   -1  : single forward pass over the whole dataset (legacy behavior).
    //          Maximum throughput, but allocates O(n_plots * hidden) of
    //          device memory in one shot — easy to OOM on large test sets.
    //   >0  : forward over consecutive chunks of `batch_size` plots, with
    //          results concatenated on CPU. Default of 4096 keeps peak VRAM
    //          predictable while still amortising the host->device copy.
    // Only -1 or a positive value is valid; any other value (including 0)
    // raises std::invalid_argument. Use -1, not 0, to opt out of chunking.
    //
    // The default (4096) matches the trainer's default batch size and keeps
    // the predictor safe on 16 GiB-class GPUs at typical hidden sizes
    // (see issue #2 for the OOM symptom that motivated this knob).
    ResolvePredictions predict(
        const ResolveDataset& dataset,
        bool return_latent = false,
        int64_t batch_size = 4096
    );

    // Predict on new data (raw tensor API)
    // Returns predictions for all targets.
    // categorical_ids: (n_samples, n_categoricals) int64 codes produced by
    // CategoricalVocab — undefined/empty when the model has no categoricals
    // (the model layer pads with zeros in that case).
    ResolvePredictions predict(
        torch::Tensor coordinates,
        torch::Tensor covariates,
        torch::Tensor hash_embedding,
        torch::Tensor species_ids,
        torch::Tensor species_vector,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor unknown_fraction,
        torch::Tensor unknown_count,
        // Pool-style fields (rank_pool / transformer)
        torch::Tensor pool_genus_ids = {},
        torch::Tensor pool_family_ids = {},
        torch::Tensor pool_weights = {},
        torch::Tensor pool_mask = {},
        torch::Tensor pool_has_cover = {},
        torch::Tensor categorical_ids = {},
        bool return_latent = false
    );

    // Get latent embeddings
    torch::Tensor get_embeddings(
        torch::Tensor coordinates,
        torch::Tensor covariates,
        torch::Tensor hash_embedding,
        torch::Tensor genus_ids,
        torch::Tensor family_ids
    );

    // Get learned embedding weights (averaged across positions)
    [[nodiscard]] torch::Tensor get_genus_embeddings() const;
    [[nodiscard]] torch::Tensor get_family_embeddings() const;
    [[nodiscard]] torch::Tensor get_species_embeddings() const;

    // Optimize model for inference (fuses BatchNorm into Linear layers)
    void optimize_for_inference();

    // Accessors
    [[nodiscard]] ResolveModel& model() noexcept { return model_; }
    [[nodiscard]] const ResolveModel& model() const noexcept { return model_; }
    [[nodiscard]] const Scalers& scalers() const noexcept { return scalers_; }
    [[nodiscard]] torch::Device device() const noexcept { return device_; }
    // Vocabulary the model was trained with. Empty for models without
    // categorical covariates. Users wanting to score raw CSVs at inference
    // should pass this through CategoricalVocab::encode_batch() to recover
    // training-consistent integer codes.
    [[nodiscard]] const CategoricalVocab& categorical_vocab() const noexcept {
        return categorical_vocab_;
    }

    // The model's schema, i.e. everything the checkpoint recorded about the
    // data it was trained on -- including the fitted species / genus / family
    // vocabularies and the loader knobs needed to rebuild a matching
    // DatasetConfig (issue #102). Pass to
    // ResolveDataset::from_csv_with_schema / from_species_csv_with_schema to
    // build an inference dataset in this model's ID namespace.
    // Not noexcept: the module holder's operator-> checks for an empty holder,
    // so dereferencing model_ is a throwing operation.
    [[nodiscard]] const ResolveSchema& schema() const {
        return model_->schema();
    }

    // Ordered species vocabulary the model was trained with: element i is the
    // name that encodes to code i, index 0 the reserved "<UNK>" slot. Empty for
    // a checkpoint written before issue #102 (which stored only the size).
    [[nodiscard]] const std::vector<std::string>& species_vocab() const {
        return schema().species_vocab;
    }
    // Ordered genus / family vocabularies, same layout as species_vocab().
    [[nodiscard]] const std::vector<std::string>& genus_vocab() const {
        return schema().genus_vocab;
    }
    [[nodiscard]] const std::vector<std::string>& family_vocab() const {
        return schema().family_vocab;
    }

    // Every training vocabulary in the form the ResolveDataset *_with_vocabs
    // loaders accept: the schema's species / taxonomy vocabularies plus the
    // categorical maps, which live on the Predictor rather than the schema.
    // This is the complete carrier -- prefer it over schema() when the model
    // has categorical covariates.
    [[nodiscard]] ExternalVocabs external_vocabs() const;

private:
    // Throw when `dataset` was not built in this model's integer-code
    // namespace. Compares the full ordered vocabularies when the checkpoint
    // carries them, otherwise the vocabulary sizes alone (the pre-issue-#102
    // fallback). Species IDs are only checked for encoders that actually index
    // a species embedding table (hash mode derives its features from the
    // species STRING, so its codes are irrelevant).
    void validate_dataset_vocabs(const ResolveDataset& dataset) const;

    ResolveModel model_;
    Scalers scalers_;
    CategoricalVocab categorical_vocab_;
    torch::Device device_;
};

} // namespace resolve
