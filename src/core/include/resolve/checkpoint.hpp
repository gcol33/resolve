#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include <torch/torch.h>
#include <string>
#include <unordered_map>

namespace resolve {

// Write training progress to JSON file for monitoring
void write_progress_file(
    const std::string& checkpoint_dir,
    int epoch,
    int max_epochs,
    int best_epoch,
    float best_loss,
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
// effective config.batch_size, making an OOM fallback detectable (issue #86).
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

// Save run metadata to archive
void save_run_metadata(
    torch::serialize::OutputArchive& archive,
    const RunMetadata& metadata
);

// Load run metadata from archive (inverse of save_run_metadata).
RunMetadata load_run_metadata(
    torch::serialize::InputArchive& archive
);

// Write run metadata as JSON file alongside checkpoint. `requested_batch_size`
// mirrors save_train_config: -1 (default) uses train_config.batch_size; otherwise
// the sidecar's `batch_size` is the requested value and `effective_batch_size` the
// value that actually trained the model (issue #86).
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

// Load schema from archive
ResolveSchema load_schema(
    torch::serialize::InputArchive& archive
);

} // namespace resolve
