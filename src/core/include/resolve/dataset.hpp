#pragma once

#include "resolve/types.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/encoder.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace resolve {

// Configuration for dataset loading
struct DatasetConfig {
    // Species encoding configuration
    SpeciesEncodingMode species_encoding = SpeciesEncodingMode::Hash;
    int hash_dim = 32;
    int top_k = 3;
    int top_k_species = 10;  // For embed mode
    SelectionMode selection = SelectionMode::Top;
    RepresentationMode representation = RepresentationMode::Abundance;
    NormalizationMode normalization = NormalizationMode::Raw;
    AggregationMode aggregation = AggregationMode::Abundance;

    // Unknown species tracking
    bool track_unknown_fraction = true;
    bool track_unknown_count = false;

    // Taxonomy
    bool use_taxonomy = true;  // If available in data
};

// Loaded dataset ready for training
class ResolveDataset {
public:
    // Load from two CSV files: header data (one row per plot) and species data (multiple rows per plot)
    static ResolveDataset from_csv(
        const std::string& header_path,
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Load from single CSV file with species data only (header data inferred)
    static ResolveDataset from_species_csv(
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Accessors for encoded data
    const torch::Tensor& coordinates() const { return coordinates_; }
    const torch::Tensor& covariates() const { return covariates_; }
    const torch::Tensor& hash_embedding() const { return hash_embedding_; }
    const torch::Tensor& species_ids() const { return species_ids_; }
    const torch::Tensor& species_vector() const { return species_vector_; }
    const torch::Tensor& genus_ids() const { return genus_ids_; }
    const torch::Tensor& family_ids() const { return family_ids_; }
    const torch::Tensor& unknown_fraction() const { return unknown_fraction_; }
    const torch::Tensor& unknown_count() const { return unknown_count_; }
    const std::unordered_map<std::string, torch::Tensor>& targets() const { return targets_; }

    // Schema information
    const ResolveSchema& schema() const { return schema_; }

    // Plot IDs for tracking
    const std::vector<std::string>& plot_ids() const { return plot_ids_; }

    // Taxonomy vocabulary (for saving/loading)
    const TaxonomyVocab& taxonomy_vocab() const { return taxonomy_vocab_; }
    TaxonomyVocab& taxonomy_vocab() { return taxonomy_vocab_; }

    // Species vocabulary (for embed mode)
    const std::vector<std::string>& species_vocab() const { return species_vocab_; }

    // Number of plots
    int64_t n_plots() const { return schema_.n_plots; }

    // Dataset configuration
    const DatasetConfig& config() const { return config_; }

private:
    ResolveDataset() = default;

    // Load header CSV data
    void load_header_data(
        const std::string& header_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets
    );

    // Load and encode species data
    void load_species_data(
        const std::string& species_path,
        const RoleMapping& roles
    );

    // Build species vocabulary from data
    void build_species_vocab(
        const std::unordered_map<std::string, std::vector<std::pair<std::string, float>>>& plot_species
    );

    // Encode species data based on mode
    void encode_species(
        const std::unordered_map<std::string, std::vector<SpeciesRecord>>& plot_records
    );

    // Data tensors
    torch::Tensor coordinates_;      // (n_plots, 2)
    torch::Tensor covariates_;       // (n_plots, n_covariates)
    torch::Tensor hash_embedding_;   // (n_plots, hash_dim) for hash mode
    torch::Tensor species_ids_;      // (n_plots, top_k_species) for embed mode
    torch::Tensor species_vector_;   // (n_plots, n_species) for sparse mode
    torch::Tensor genus_ids_;        // (n_plots, n_taxonomy_slots)
    torch::Tensor family_ids_;       // (n_plots, n_taxonomy_slots)
    torch::Tensor unknown_fraction_; // (n_plots,)
    torch::Tensor unknown_count_;    // (n_plots,)
    std::unordered_map<std::string, torch::Tensor> targets_;

    // Metadata
    ResolveSchema schema_;
    DatasetConfig config_;
    std::vector<std::string> plot_ids_;
    std::vector<std::string> species_vocab_;
    std::unordered_map<std::string, int64_t> species_to_idx_;
    TaxonomyVocab taxonomy_vocab_;

    // Target configurations
    std::vector<TargetConfig> target_configs_;
};

} // namespace resolve
