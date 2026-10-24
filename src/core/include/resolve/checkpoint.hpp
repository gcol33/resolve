#pragma once

#include "resolve/json.hpp"
#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <utility>

namespace resolve {

// Read the checkpoint at `path` into `archive`, its tensors onto `device`,
// retrying a transient read fault (io::with_retry). A path that is not a
// readable file throws io::IOError, a storage failure, once the retries are
// spent; an unreadable archive (a truncated or foreign file) throws the
// deserializer's error.
void read_checkpoint_archive(torch::serialize::InputArchive& archive, const std::string& path,
                             const torch::Device& device, const char* what);

// Write `archive` to `path`, retrying a transient write fault; a write that
// still fails throws io::IOError.
void write_checkpoint_archive(torch::serialize::OutputArchive& archive, const std::string& path);

// Write `checkpoint_dir`/progress.json for monitoring a run: the epoch, the
// best epoch, its held-out loss and its value of the selection metric the run
// follows (both null until an epoch is scored), the patience count and one
// band accuracy of the best epoch.
void write_progress_file(
    const std::string& checkpoint_dir,
    int epoch,
    int max_epochs,
    int best_epoch,
    float best_loss,
    SelectionMetric selection_metric,
    float best_selected,
    int epochs_without_improvement,
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& metrics
);

// Save model config to archive
void save_model_config(
    torch::serialize::OutputArchive& archive,
    const ModelConfig& config
);

// Load model config from archive
ModelConfig load_model_config(
    torch::serialize::InputArchive& archive
);

// Save training config to archive. `requested_batch_size` is the batch size the
// caller asked for before any CUDA OOM auto-halve; pass -1 (default) to fall back
// to config.batch_size. It is written under `train_batch_size` (so load_train_config
// restores the requested value) while `train_effective_batch_size` records the
// effective config.batch_size, making an OOM fallback detectable.
void save_train_config(
    torch::serialize::OutputArchive& archive,
    const TrainConfig& config,
    int requested_batch_size = -1
);

// Load training config from archive (inverse of save_train_config). Recovers
// the persisted training hyperparameters; fields save_train_config does not
// write keep their TrainConfig defaults.
TrainConfig load_train_config(
    torch::serialize::InputArchive& archive
);

// The batch size that trained the model (`train_effective_batch_size`), or the
// requested one for a checkpoint that does not record it.
int load_effective_batch_size(
    torch::serialize::InputArchive& archive
);

// Save run metadata to archive
void save_run_metadata(
    torch::serialize::OutputArchive& archive,
    const RunMetadata& metadata
);

// Load run metadata from archive (inverse of save_run_metadata).
RunMetadata load_run_metadata(
    torch::serialize::InputArchive& archive
);

// What a checkpoint holds, as one JSON document: the run metadata, the schema
// (its loader settings included), the model config, the persisted train config
// (`batch_size` as requested beside `effective_batch_size`) and the final
// metrics. The sidecar beside a checkpoint is this document, and `resolve info`
// prints it.
json::Value checkpoint_description(
    const ModelConfig& model_config,
    const TrainConfig& train_config,
    int effective_batch_size,
    const RunMetadata& metadata,
    const ResolveSchema& schema
);

// checkpoint_description of a saved checkpoint, read onto the CPU.
json::Value describe_checkpoint(const std::string& path);

// Write checkpoint_description as `<checkpoint>.json` beside the checkpoint
// (`.pt` replaced). `train_config.batch_size` is the effective batch size and
// `requested_batch_size` the requested one; -1 (default) takes them as equal.
void write_metadata_json(
    const std::string& checkpoint_path,
    const ModelConfig& model_config,
    const TrainConfig& train_config,
    const RunMetadata& metadata,
    const ResolveSchema& schema,
    int requested_batch_size = -1
);

// Save scalers to archive
void save_scalers(
    torch::serialize::OutputArchive& archive,
    const Scalers& scalers
);

// Load scalers from archive
Scalers load_scalers(
    torch::serialize::InputArchive& archive
);

// Save schema to archive
void save_schema(
    torch::serialize::OutputArchive& archive,
    const ResolveSchema& schema
);

// Load schema from archive. With `model_config` (the checkpoint's own), the
// loader settings that also size the model are taken from it.
ResolveSchema load_schema(
    torch::serialize::InputArchive& archive,
    const ModelConfig* model_config = nullptr
);

} // namespace resolve
