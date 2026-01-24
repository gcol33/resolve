#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace resolve {

// Task type for prediction heads
enum class TaskType {
    Regression,
    Classification
};

// Target transform type
enum class TransformType {
    None,
    Log1p
};

// Species encoding mode
enum class SpeciesEncodingMode {
    Hash,   // Feature hashing (default)
    Embed,  // Learnable embeddings for top-k species
    Sparse  // Explicit species abundance/presence vector
};

// Configuration for a prediction target
struct TargetConfig {
    std::string name;
    TaskType task;
    TransformType transform = TransformType::None;
    int num_classes = 0;  // For classification
    float weight = 1.0f;  // Loss weight in multi-task
};

// Schema information for a dataset
struct ResolveSchema {
    int64_t n_plots = 0;
    int64_t n_species = 0;          // Number of unique species
    int64_t n_species_vocab = 0;    // Size of species vocabulary (for embed/sparse modes)
    bool has_coordinates = true;
    bool has_abundance = false;
    bool has_taxonomy = false;
    int64_t n_genera = 0;
    int64_t n_families = 0;
    int64_t n_genera_vocab = 0;     // Vocab size for embed mode
    int64_t n_families_vocab = 0;   // Vocab size for embed mode
    std::vector<std::string> covariate_names;
    std::vector<TargetConfig> targets;
    bool track_unknown_fraction = true;
    bool track_unknown_count = false;
};

// Alias for backwards compatibility
using SpaccSchema = ResolveSchema;

// Model configuration
struct ModelConfig {
    SpeciesEncodingMode species_encoding = SpeciesEncodingMode::Hash;
    bool uses_explicit_vector = false;  // For hash mode with selection="all"
    int hash_dim = 32;
    int species_embed_dim = 32;
    int genus_emb_dim = 8;
    int family_emb_dim = 8;
    int top_k = 3;
    int top_k_species = 10;  // For embed mode
    int n_taxonomy_slots = 3;  // May be 2*top_k for top_bottom mode
    std::vector<int64_t> hidden_dims = {2048, 1024, 512, 256, 128, 64};
    float dropout = 0.3f;
};

// Training configuration
struct TrainConfig {
    int batch_size = 4096;
    int max_epochs = 500;
    int patience = 50;
    float lr = 1e-3f;
    float weight_decay = 1e-4f;
    std::pair<int, int> phase_boundaries = {100, 300};
    torch::Device device = torch::kCPU;
};

// Batch of data for training/inference
struct ResolveBatch {
    torch::Tensor continuous;      // (batch, n_continuous)
    torch::Tensor genus_ids;       // (batch, n_taxonomy_slots) or empty
    torch::Tensor family_ids;      // (batch, n_taxonomy_slots) or empty
    torch::Tensor species_ids;     // (batch, top_k_species) for embed mode
    torch::Tensor species_vector;  // (batch, n_species) for sparse mode
    std::unordered_map<std::string, torch::Tensor> targets;  // target_name -> tensor

    ResolveBatch to(torch::Device device) const {
        ResolveBatch batch;
        batch.continuous = continuous.to(device);
        if (genus_ids.defined()) {
            batch.genus_ids = genus_ids.to(device);
        }
        if (family_ids.defined()) {
            batch.family_ids = family_ids.to(device);
        }
        if (species_ids.defined()) {
            batch.species_ids = species_ids.to(device);
        }
        if (species_vector.defined()) {
            batch.species_vector = species_vector.to(device);
        }
        for (const auto& [name, tensor] : targets) {
            batch.targets[name] = tensor.to(device);
        }
        return batch;
    }
};

// Alias for backwards compatibility
using SpaccBatch = ResolveBatch;

// Results from training
struct TrainResult {
    int best_epoch;
    std::unordered_map<std::string, std::unordered_map<std::string, float>> final_metrics;
    std::vector<float> train_loss_history;
    std::vector<float> test_loss_history;
    float train_time_seconds = 0.0f;
    int resumed_from_epoch = 0;
};

// Predictions output
struct ResolvePredictions {
    std::unordered_map<std::string, torch::Tensor> predictions;
    std::vector<std::string> plot_ids;
    torch::Tensor latent;    // optional latent representations
};

// Alias for backwards compatibility
using SpaccPredictions = ResolvePredictions;

} // namespace resolve
