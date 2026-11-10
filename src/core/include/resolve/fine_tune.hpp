#pragma once

// Fine-tuning: a model that starts from a trained one.
//
// A trained model -- a checkpoint, or a member of a released suite -- is the
// PARENT. A fine-tuned model reads the plots exactly as its parent does (the
// parent's species, genus, family and categorical vocabularies, its loader
// settings and its continuous-block scalers), starts from the parent's encoder
// weights, and trains heads for the caller's own targets: a new classification
// scheme, a new regression variable, or a released target refitted on regional
// plots. ModelConfig::freeze_encoder keeps the parent's encoder fixed and
// trains the heads alone; without it the encoder is trained further from the
// parent's weights.
//
//   ParentModel parent = ParentModel::open(suite_dir, {.suite_target = "altitude"});
//   ResolveDataset ds = ResolveDataset::load(header, species, roles, targets,
//                                            parent.dataset_config(), &parent.vocabs());
//   ModelConfig mc = parent.model_config();
//   mc.freeze_encoder = true;
//   Trainer trainer(ResolveModel(ds.schema(), model_config_for_dataset(ds, mc)), tc);
//   trainer.init_from(parent);
//   trainer.prepare_data(ds);
//   trainer.fit();
//   trainer.save(path, &trainer.run_metadata());
//
// The checkpoint then records its parent (RunMetadata::parent: the parent's
// SHA-256, and the suite, target and seed it came from), and
// write_derived_suite (suite.hpp) releases fine-tuned checkpoints as a suite of
// their own that SuitePredictor scores.

#include "resolve/categorical.hpp"
#include "resolve/dataset.hpp"
#include "resolve/scalers.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace resolve {

// Where a warm start reads its parent from, and what it carries over.
struct InitFromConfig {
    InitReuse reuse = InitReuse::Encoder;
    // A suite directory as the source: the target whose member to start from
    // (may be left empty for a suite of one target) and that member's seed
    // (-1: the target's first member). Both stay unset for a checkpoint.
    std::string suite_target;
    int64_t suite_seed = -1;
};

// The ModelConfig fields a fine-tuned model may set apart from its parent's:
// the task heads, the training-time regularisation and what is frozen. Every
// other field shapes or defines the encoder, so Trainer::init_from requires it
// to equal the parent's.
inline constexpr const char* kFineTuneModelFields[] = {
    "head_hidden_dims",   "head_activation",     "head_dropout",
    "dropout",            "transformer_dropout", "cover_dropout",
    "plot_totals_dropout", "moe_noise_std",      "moe_aux_loss_weight",
    "freeze_composition", "freeze_encoder",
};

// The DatasetConfig fields a fine-tuned model's data may set apart from its
// parent's: compute choices that leave every model input as it is.
inline constexpr const char* kFineTuneDatasetFields[] = {"use_cuda_hash"};

// A parent read once: its configuration, vocabularies, scalers and weights,
// and the record a fine-tuned checkpoint keeps of it.
class ParentModel {
public:
    // `source` is a checkpoint file, or a suite directory (holding
    // manifest.json) whose member `config.suite_target` / `config.suite_seed`
    // names; a suite member's size and SHA-256 are checked against the
    // manifest before it is read. Throws std::invalid_argument for a suite
    // setting given with a checkpoint, an unknown target or seed, and a
    // checkpoint that carries no species vocabulary; std::runtime_error for a
    // member that differs from its manifest entry.
    static ParentModel open(const std::string& source, const InitFromConfig& config = {});

    [[nodiscard]] const ModelConfig& model_config() const noexcept { return model_config_; }
    [[nodiscard]] const ResolveSchema& schema() const noexcept { return schema_; }
    [[nodiscard]] const Scalers& scalers() const noexcept { return scalers_; }
    [[nodiscard]] const CategoricalVocab& categorical_vocab() const noexcept {
        return categorical_vocab_;
    }

    // What a fine-tuning dataset is loaded with (ResolveDataset::load): the
    // parent's vocabularies, which also replay the classes of a target the
    // parent shares, and its loader settings.
    [[nodiscard]] ExternalVocabs vocabs() const;
    [[nodiscard]] DatasetConfig dataset_config() const;

    // The checkpoint's path, SHA-256, size and engine version, and the suite
    // fields when it came from a suite; reuse, n_tensors and heads are set by
    // Trainer::init_from.
    [[nodiscard]] const ParentRecord& record() const noexcept { return record_; }

    // Every parameter and buffer of the parent by its name in
    // named_parameters() / named_buffers(), on the CPU.
    [[nodiscard]] const std::unordered_map<std::string, torch::Tensor>& parameters() const noexcept {
        return parameters_;
    }
    [[nodiscard]] const std::unordered_map<std::string, torch::Tensor>& buffers() const noexcept {
        return buffers_;
    }

private:
    ModelConfig model_config_;
    ResolveSchema schema_;
    Scalers scalers_;
    CategoricalVocab categorical_vocab_;
    ParentRecord record_;
    std::unordered_map<std::string, torch::Tensor> parameters_;
    std::unordered_map<std::string, torch::Tensor> buffers_;
};

// Every way `model_config` / `dataset_config` / `schema` (a fine-tuned
// model's) departs from what `parent` fixes: an encoder-shaping config field
// with another value, other loader settings, other vocabularies, covariates,
// categorical columns, coordinates, taxonomy or abundance. Empty when the model
// reads the plots as its parent does. Trainer::init_from refuses a model with
// any.
[[nodiscard]] std::vector<std::string> fine_tune_mismatches(const ParentModel& parent,
                                                           const ModelConfig& model_config,
                                                           const ResolveSchema& schema);

}  // namespace resolve
