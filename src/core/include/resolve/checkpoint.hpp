#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
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

// Save scalers to archive
void save_scalers(
    torch::serialize::OutputArchive& archive,
    const Scalers& scalers
);

// Load scalers from archive
Scalers load_scalers(
    torch::serialize::InputArchive& archive
);

} // namespace resolve
