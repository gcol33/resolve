#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace resolve {

// ============================================================================
// Enums
// ============================================================================

enum class TaskType {
    Regression,
    Classification
};

enum class TransformType {
    None,
    Log1p
};

enum class SpeciesEncodingMode {
    Hash,   // Feature hashing (default)
    Embed,  // Learnable embeddings for top-k species
    Sparse  // Explicit species abundance/presence vector
};

enum class SelectionMode {
    Top,        // Top-k by abundance
    Bottom,     // Bottom-k (rare species)
    TopBottom,  // Both top and bottom
    All         // All species (explicit vector)
};

enum class NormalizationMode {
    Raw,    // No normalization
    Norm,   // Sum to 1
    Log1p   // log(1 + x)
};

enum class RepresentationMode {
    Abundance,        // Use abundance values
    PresenceAbsence   // Binary presence/absence
};

// ============================================================================
// Target Configuration
// ============================================================================

struct TargetConfig {
    std::string name;
    std::string column;
    TaskType task = TaskType::Regression;
    TransformType transform = TransformType::None;
    int num_classes = 0;
    float weight = 1.0f;

    static TargetConfig regression(const std::string& name, const std::string& column,
                                    TransformType transform = TransformType::None,
                                    float weight = 1.0f) {
        return TargetConfig{name, column, TaskType::Regression, transform, 0, weight};
    }

    static TargetConfig classification(const std::string& name, const std::string& column,
                                        int num_classes, float weight = 1.0f) {
        return TargetConfig{name, column, TaskType::Classification, TransformType::None, num_classes, weight};
    }
};

// ============================================================================
// Role Mapping (column name mappings)
// ============================================================================

struct RoleMapping {
    // Required
    std::string plot_id;
    std::string species_id;
    std::string species_plot_id;

    // Optional
    std::optional<std::string> coords_lat;
    std::optional<std::string> coords_lon;
    std::optional<std::string> abundance;
    std::optional<std::string> taxonomy_genus;
    std::optional<std::string> taxonomy_family;
    std::vector<std::string> covariates;

    bool has_coordinates() const {
        return coords_lat.has_value() && coords_lon.has_value();
    }

    bool has_abundance() const {
        return abundance.has_value();
    }

    bool has_taxonomy() const {
        return taxonomy_genus.has_value() && taxonomy_family.has_value();
    }

    void validate() const {
        if (plot_id.empty() || species_id.empty() || species_plot_id.empty()) {
            throw std::invalid_argument("plot_id, species_id, and species_plot_id are required");
        }
        if (coords_lat.has_value() != coords_lon.has_value()) {
            throw std::invalid_argument("coords_lat and coords_lon must both be specified or both omitted");
        }
    }
};

// ============================================================================
// Schema (derived from dataset)
// ============================================================================

struct ResolveSchema {
    int64_t n_plots = 0;
    int64_t n_species = 0;
    int64_t n_continuous = 0;
    bool has_coordinates = true;
    bool has_abundance = false;
    bool has_taxonomy = false;
    int64_t n_genera = 0;
    int64_t n_families = 0;
    std::vector<std::string> covariate_names;
    std::vector<TargetConfig> targets;

    // Species encoding config
    NormalizationMode species_normalization = NormalizationMode::Norm;
    bool track_unknown_fraction = true;
    bool track_unknown_count = false;

    // Vocab sizes (for embed/sparse modes)
    int64_t n_species_vocab = 0;
    int64_t n_genera_vocab = 0;
    int64_t n_families_vocab = 0;
};

// ============================================================================
// Model Configuration
// ============================================================================

struct ModelConfig {
    SpeciesEncodingMode species_encoding = SpeciesEncodingMode::Hash;
    bool uses_explicit_vector = false;

    // Hash mode
    int hash_dim = 32;

    // Embed mode
    int species_embed_dim = 32;
    int top_k_species = 10;

    // Taxonomy embeddings
    int genus_emb_dim = 8;
    int family_emb_dim = 8;
    int top_k = 3;
    int n_taxonomy_slots = 3;

    // MLP architecture
    std::vector<int64_t> hidden_dims = {2048, 1024, 512, 256, 128, 64};
    float dropout = 0.3f;
};

// ============================================================================
// Training Configuration
// ============================================================================

struct TrainConfig {
    int batch_size = 4096;
    int max_epochs = 500;
    float lr = 1e-3f;
    float weight_decay = 1e-4f;

    // Early stopping
    int patience = 50;
    float min_delta = 0.0f;

    // Learning rate scheduling
    bool use_lr_scheduler = true;
    float lr_max = 1e-3f;
    float lr_div_factor = 25.0f;
    float lr_final_div_factor = 1e4f;
    float pct_start = 0.3f;

    // Gradient clipping
    float max_grad_norm = 1.0f;

    // Checkpointing
    bool save_checkpoints = true;
    std::string checkpoint_dir = "checkpoints";
    int checkpoint_interval = 10;  // Save every N epochs

    // Caching
    bool use_cache = true;
    std::string cache_dir = ".resolve_cache";

    // Device
    torch::Device device = torch::kCPU;
};

// ============================================================================
// Batch of data
// ============================================================================

struct ResolveBatch {
    torch::Tensor continuous;       // (batch, n_continuous)
    torch::Tensor genus_ids;        // (batch, n_taxonomy_slots)
    torch::Tensor family_ids;       // (batch, n_taxonomy_slots)
    torch::Tensor species_ids;      // (batch, top_k_species) for embed mode
    torch::Tensor species_vector;   // (batch, n_species) for sparse mode
    std::unordered_map<std::string, torch::Tensor> targets;

    ResolveBatch to(torch::Device device) const {
        ResolveBatch batch;
        batch.continuous = continuous.defined() ? continuous.to(device) : continuous;
        batch.genus_ids = genus_ids.defined() ? genus_ids.to(device) : genus_ids;
        batch.family_ids = family_ids.defined() ? family_ids.to(device) : family_ids;
        batch.species_ids = species_ids.defined() ? species_ids.to(device) : species_ids;
        batch.species_vector = species_vector.defined() ? species_vector.to(device) : species_vector;
        for (const auto& [name, tensor] : targets) {
            batch.targets[name] = tensor.to(device);
        }
        return batch;
    }
};

// ============================================================================
// Training Results
// ============================================================================

struct TrainResult {
    int best_epoch = 0;
    int total_epochs = 0;
    float best_loss = std::numeric_limits<float>::max();
    std::unordered_map<std::string, std::unordered_map<std::string, float>> final_metrics;
    std::vector<float> train_loss_history;
    std::vector<float> test_loss_history;
    float train_time_seconds = 0.0f;
    int resumed_from_epoch = 0;
    bool early_stopped = false;
};

// ============================================================================
// Predictions
// ============================================================================

struct ResolvePredictions {
    std::unordered_map<std::string, torch::Tensor> predictions;
    std::vector<std::string> plot_ids;
    torch::Tensor latent;
    std::unordered_map<std::string, torch::Tensor> confidence;
};

} // namespace resolve
