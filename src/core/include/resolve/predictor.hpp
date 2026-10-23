#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "resolve/dataset.hpp"
#include "resolve/retrieval.hpp"
#include <torch/torch.h>

#include <optional>

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
    // has no categorical covariates).
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
    // vocabulary. A checkpoint that stores only the vocabulary SIZES allows
    // only those to be compared; `load()` warns once in that case.
    //
    // The result carries one `predictions` entry per target (a regression
    // value on the original scale, or a classification target's argmax code)
    // and, for every classification target, a `probabilities` entry holding
    // the softmax row `(n_plots, n_classes)` the code was taken from, so a
    // caller can read a per-plot confidence (max probability, margin,
    // entropy) off a checkpoint scored on any dataset. With a retrieval set
    // (set_retrieval) both are the blend with the plot's nearest reference
    // plots.
    //
    // `batch_size` controls how the forward pass is chunked along dim 0:
    //   -1  : one forward pass over the whole dataset. Maximum throughput,
    //          but allocates O(n_plots * hidden) of device memory in one
    //          shot, easy to OOM on large test sets.
    //   >0  : forward over consecutive chunks of `batch_size` plots.
    // Only -1 or a positive value is valid; any other value (including 0)
    // raises std::invalid_argument. Use -1, not 0, to opt out of chunking.
    // Either way every output tensor is on the CPU: each chunk's outputs move
    // there before the next chunk runs, so peak device memory is one chunk's.
    //
    // The default (4096) matches the trainer's default batch size and keeps
    // the predictor safe on 16 GiB-class GPUs at typical hidden sizes.
    ResolvePredictions predict(
        const ResolveDataset& dataset,
        bool return_latent = false,
        int64_t batch_size = 4096
    );

    // Predict on tensors: the parts of the continuous block (assembled and
    // standardised as at training time; a missing coordinate or covariate is
    // NaN) and the other model inputs, `inputs.continuous` left empty.
    // unknown_fraction / unknown_count are required when the model reads them
    // (ResolveSchema::track_unknown_fraction / track_unknown_count); a model
    // with categorical columns given no categorical_ids reads every column as
    // UNK. `batch_size` and the output device are as for the dataset overload;
    // plot_ids are the row numbers.
    ResolvePredictions predict(
        const ContinuousInputs& parts,
        const PlotInputs& inputs,
        bool return_latent = false,
        int64_t batch_size = 4096
    );

    // The latent embedding (n_plots, latent_dim) of each plot, on the CPU,
    // from the same inputs and through the same chunked path as predict().
    torch::Tensor get_embeddings(
        const ContinuousInputs& parts,
        const PlotInputs& inputs,
        int64_t batch_size = 4096
    );

    // Get learned embedding weights (averaged across positions)
    [[nodiscard]] torch::Tensor get_genus_embeddings() const;
    [[nodiscard]] torch::Tensor get_family_embeddings() const;
    [[nodiscard]] torch::Tensor get_species_embeddings() const;

    // Retrieval augmentation (retrieval.hpp). Encodes the `reference` plots in
    // chunks of `batch_size`, keeps their latents and labels on the predictor's
    // device, and from then on every predict() blends each plot's prediction
    // with the labels of its nearest reference plots; the latent and
    // get_embeddings() are unchanged. `reference` must be built in the model's
    // namespace (validated as predict() validates a dataset) and carry labels
    // for every target of the model, normally the training plots; a plot
    // unlabelled for a target there is not retrieved for that target. Replaces
    // a retrieval set before.
    void set_retrieval(const ResolveDataset& reference,
                       const RetrievalConfig& config = RetrievalConfig{},
                       int64_t batch_size = 4096);
    // Back to the model's own predictions.
    void clear_retrieval() noexcept { retrieval_.reset(); }
    // The retrieval predict() applies, or nullptr.
    [[nodiscard]] const RetrievalIndex* retrieval() const noexcept {
        return retrieval_ ? &*retrieval_ : nullptr;
    }

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
    // DatasetConfig. Pass to
    // ResolveDataset::from_csv_with_schema / from_species_csv_with_schema to
    // build an inference dataset in this model's ID namespace.
    // Not noexcept: the module holder's operator-> checks for an empty holder,
    // so dereferencing model_ is a throwing operation.
    [[nodiscard]] const ResolveSchema& schema() const {
        return model_->schema();
    }

    // Ordered species vocabulary the model was trained with: element i is the
    // name that encodes to code i, index 0 the reserved "<UNK>" slot. Empty for
    // a checkpoint that stores only the vocabulary size.
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
    // carries them, otherwise the vocabulary sizes alone. Species IDs are only checked for encoders that actually index
    // a species embedding table (hash mode derives its features from the
    // species STRING, so its codes are irrelevant).
    void validate_dataset_vocabs(const ResolveDataset& dataset) const;

    // The one inference path behind predict() and get_embeddings(): slices
    // the inputs into chunks of `batch_size` plots (one chunk for -1 and for
    // an encoder whose output depends on its batch-mates), assembles and
    // standardises each chunk's continuous block, runs it on device_ and
    // moves what was asked for -- the per-target predictions and
    // probabilities, the latent, or both -- to the CPU.
    ResolvePredictions run_inference(const ContinuousInputs& parts,
                                     const PlotInputs& inputs,
                                     bool want_outputs,
                                     bool want_latent,
                                     int64_t batch_size);

    ResolveModel model_;
    Scalers scalers_;
    CategoricalVocab categorical_vocab_;
    torch::Device device_;
    std::optional<RetrievalIndex> retrieval_;
};

} // namespace resolve
