#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <functional>
#include <iostream>

namespace resolve {

// Logging callback for training progress
using LogCallback = std::function<void(const std::string&)>;

// Default logging to stdout
inline void default_log(const std::string& msg) {
    std::cout << msg << std::endl;
}

// Null logger (discards all messages)
inline void null_log(const std::string&) {}

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

// Loss configuration presets
enum class LossConfigMode {
    MAE,       // Pure MAE loss (no SMAPE, no band penalty)
    SMAPE,     // SMAPE as primary loss
    Combined   // Phased: MAE -> MAE+SMAPE -> MAE+SMAPE+band (default)
};

// Learning rate scheduler type
enum class LRSchedulerType {
    None,           // Constant learning rate
    StepLR,         // Step decay every N epochs
    CosineAnnealing // Cosine annealing to min_lr
};

// Species selection mode (which species to include)
enum class SelectionMode {
    Top,        // Top-k by abundance
    Bottom,     // Bottom-k by abundance
    TopBottom,  // Top-k and bottom-k
    All         // All species (explicit vector)
};

// How species are represented
enum class RepresentationMode {
    Abundance,        // Use abundance values
    PresenceAbsence   // Binary presence/absence
};

// Normalization for abundances
enum class NormalizationMode {
    Raw,    // Raw abundance values
    Norm,   // Normalized (sum to 1)
    Log1p   // Log1p transformed
};

// Aggregation mode for taxonomy
enum class AggregationMode {
    Abundance,  // Sum abundances
    Count       // Count species
};

// Configuration for a prediction target
struct TargetConfig {
    std::string name;
    TaskType task;
    TransformType transform = TransformType::None;
    int num_classes = 0;  // For classification
    float weight = 1.0f;  // Loss weight in multi-task
    std::vector<float> class_weights;  // Optional class weights for imbalanced classification
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
    LossConfigMode loss_config = LossConfigMode::Combined;
    torch::Device device = torch::kCPU;

    // Learning rate scheduling
    LRSchedulerType lr_scheduler = LRSchedulerType::None;
    int lr_step_size = 100;      // For StepLR: decay every N epochs
    float lr_gamma = 0.1f;       // For StepLR: multiply LR by gamma
    float lr_min = 1e-6f;        // For CosineAnnealing: minimum LR

    // Band accuracy thresholds for regression metrics
    std::vector<float> band_thresholds = {0.1f, 0.25f, 0.5f};

    // Checkpointing
    std::string checkpoint_dir;   // Directory for checkpoints (empty = disabled)
    int checkpoint_every = 0;     // Save checkpoint every N epochs (0 = only best)

    // Logging callback (defaults to stdout, use null_log to disable)
    LogCallback log = default_log;
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

// Species record for encoding
struct SpeciesRecord {
    std::string species_id;
    std::string genus;
    std::string family;
    float abundance = 1.0f;
    std::string plot_id;
};

// Encoded species data (output of encoding process)
struct EncodedSpecies {
    torch::Tensor hash_embedding;   // (n_plots, hash_dim) for hash mode
    torch::Tensor genus_ids;        // (n_plots, n_taxonomy_slots)
    torch::Tensor family_ids;       // (n_plots, n_taxonomy_slots)
    torch::Tensor unknown_fraction; // (n_plots,)
    torch::Tensor unknown_count;    // (n_plots,)
    torch::Tensor species_vector;   // (n_plots, n_species) for sparse mode
    torch::Tensor species_ids;      // (n_plots, top_k_species) for embed mode
    std::vector<std::string> plot_ids;
};

// Taxonomy vocabulary for encoding genus/family names to IDs
class TaxonomyVocab {
public:
    TaxonomyVocab() = default;

    // Fit vocabulary from species records
    void fit(const std::vector<SpeciesRecord>& records) {
        genus_to_idx_.clear();
        family_to_idx_.clear();

        // Index 0 reserved for unknown
        genus_to_idx_["<UNK>"] = 0;
        family_to_idx_["<UNK>"] = 0;

        for (const auto& rec : records) {
            if (!rec.genus.empty() && genus_to_idx_.find(rec.genus) == genus_to_idx_.end()) {
                genus_to_idx_[rec.genus] = static_cast<int64_t>(genus_to_idx_.size());
            }
            if (!rec.family.empty() && family_to_idx_.find(rec.family) == family_to_idx_.end()) {
                family_to_idx_[rec.family] = static_cast<int64_t>(family_to_idx_.size());
            }
        }
    }

    // Encode genus name to ID
    int64_t encode_genus(const std::string& genus) const {
        auto it = genus_to_idx_.find(genus);
        return it != genus_to_idx_.end() ? it->second : 0;
    }

    // Encode family name to ID
    int64_t encode_family(const std::string& family) const {
        auto it = family_to_idx_.find(family);
        return it != family_to_idx_.end() ? it->second : 0;
    }

    int64_t n_genera() const { return static_cast<int64_t>(genus_to_idx_.size()); }
    int64_t n_families() const { return static_cast<int64_t>(family_to_idx_.size()); }

    // Accessors for serialization
    const std::unordered_map<std::string, int64_t>& genus_map() const { return genus_to_idx_; }
    const std::unordered_map<std::string, int64_t>& family_map() const { return family_to_idx_; }

    // Set from loaded data
    void set_genus_map(const std::unordered_map<std::string, int64_t>& m) { genus_to_idx_ = m; }
    void set_family_map(const std::unordered_map<std::string, int64_t>& m) { family_to_idx_ = m; }

    // Save vocabulary to archive (strings serialized as concatenated bytes with lengths)
    void save(torch::serialize::OutputArchive& archive, const std::string& prefix = "taxonomy_") const {
        // Build ordered lists from maps
        std::vector<std::string> genera(genus_to_idx_.size());
        for (const auto& [name, idx] : genus_to_idx_) {
            genera[idx] = name;
        }
        std::vector<std::string> families(family_to_idx_.size());
        for (const auto& [name, idx] : family_to_idx_) {
            families[idx] = name;
        }

        // Serialize genus vocab: lengths tensor + concatenated bytes tensor
        std::vector<int64_t> genus_lengths;
        std::vector<uint8_t> genus_bytes;
        for (const auto& s : genera) {
            genus_lengths.push_back(static_cast<int64_t>(s.size()));
            genus_bytes.insert(genus_bytes.end(), s.begin(), s.end());
        }
        archive.write(prefix + "genus_lengths", torch::tensor(genus_lengths));
        if (!genus_bytes.empty()) {
            archive.write(prefix + "genus_bytes", torch::from_blob(
                genus_bytes.data(), {static_cast<int64_t>(genus_bytes.size())}, torch::kUInt8).clone());
        } else {
            archive.write(prefix + "genus_bytes", torch::empty({0}, torch::kUInt8));
        }

        // Serialize family vocab
        std::vector<int64_t> family_lengths;
        std::vector<uint8_t> family_bytes;
        for (const auto& s : families) {
            family_lengths.push_back(static_cast<int64_t>(s.size()));
            family_bytes.insert(family_bytes.end(), s.begin(), s.end());
        }
        archive.write(prefix + "family_lengths", torch::tensor(family_lengths));
        if (!family_bytes.empty()) {
            archive.write(prefix + "family_bytes", torch::from_blob(
                family_bytes.data(), {static_cast<int64_t>(family_bytes.size())}, torch::kUInt8).clone());
        } else {
            archive.write(prefix + "family_bytes", torch::empty({0}, torch::kUInt8));
        }
    }

    // Load vocabulary from archive
    static TaxonomyVocab load(torch::serialize::InputArchive& archive, const std::string& prefix = "taxonomy_") {
        TaxonomyVocab vocab;

        // Load genus vocab
        torch::Tensor genus_lengths_t, genus_bytes_t;
        archive.read(prefix + "genus_lengths", genus_lengths_t);
        archive.read(prefix + "genus_bytes", genus_bytes_t);

        auto genus_lengths = genus_lengths_t.accessor<int64_t, 1>();
        auto genus_bytes_ptr = genus_bytes_t.data_ptr<uint8_t>();
        int64_t offset = 0;
        for (int64_t i = 0; i < genus_lengths_t.size(0); ++i) {
            std::string name(reinterpret_cast<const char*>(genus_bytes_ptr + offset), genus_lengths[i]);
            vocab.genus_to_idx_[name] = i;
            offset += genus_lengths[i];
        }

        // Load family vocab
        torch::Tensor family_lengths_t, family_bytes_t;
        archive.read(prefix + "family_lengths", family_lengths_t);
        archive.read(prefix + "family_bytes", family_bytes_t);

        auto family_lengths = family_lengths_t.accessor<int64_t, 1>();
        auto family_bytes_ptr = family_bytes_t.data_ptr<uint8_t>();
        offset = 0;
        for (int64_t i = 0; i < family_lengths_t.size(0); ++i) {
            std::string name(reinterpret_cast<const char*>(family_bytes_ptr + offset), family_lengths[i]);
            vocab.family_to_idx_[name] = i;
            offset += family_lengths[i];
        }

        return vocab;
    }

private:
    std::unordered_map<std::string, int64_t> genus_to_idx_;
    std::unordered_map<std::string, int64_t> family_to_idx_;
};

// Alias for backwards compatibility
using SpaccPredictions = ResolvePredictions;

} // namespace resolve
