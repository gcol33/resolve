#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include "resolve/loss.hpp"
#include "resolve/dataset.hpp"
#include <torch/torch.h>
#include <chrono>

namespace resolve {

// Forward declaration
class ResolveDataset;

// Data scalers (mean, scale) per feature/target
struct Scalers {
    torch::Tensor continuous_mean;
    torch::Tensor continuous_scale;
    std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>> target_scalers;
};

// Trainer for ResolveModel
// Supports all three encoding modes (hash, embed, sparse)
class Trainer {
public:
    Trainer(
        ResolveModel model,
        const TrainConfig& config = TrainConfig{}
    );

    // Prepare data from a ResolveDataset (preferred API)
    void prepare_data(
        const ResolveDataset& dataset,
        float test_size = 0.2f,
        int seed = 42
    );

    // Prepare data for training (raw tensor API for backwards compatibility)
    // coordinates: (n_plots, 2) or empty if no coords
    // covariates: (n_plots, n_covariates) or empty
    // hash_embedding: (n_plots, hash_dim) for hash mode
    // species_ids: (n_plots, top_k_species) for embed mode
    // species_vector: (n_plots, n_species) for sparse mode
    // genus_ids: (n_plots, n_taxonomy_slots) or empty
    // family_ids: (n_plots, n_taxonomy_slots) or empty
    // unknown_fraction: (n_plots,) optional
    // unknown_count: (n_plots,) optional
    // targets: map of target_name -> (n_plots,) tensor
    void prepare_data(
        torch::Tensor coordinates,
        torch::Tensor covariates,
        torch::Tensor hash_embedding,
        torch::Tensor species_ids,
        torch::Tensor species_vector,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor unknown_fraction,
        torch::Tensor unknown_count,
        const std::unordered_map<std::string, torch::Tensor>& targets,
        float test_size = 0.2f,
        int seed = 42
    );

    // Train the model
    TrainResult fit();

    // Save model and state
    void save(const std::string& path) const;

    // Load model and state
    static std::tuple<ResolveModel, Scalers> load(
        const std::string& path,
        torch::Device device = torch::kCPU
    );

    // Accessors
    ResolveModel& model() { return model_; }
    const Scalers& scalers() const { return scalers_; }
    const TrainConfig& config() const { return config_; }

private:
    // Train one epoch
    float train_epoch(int epoch);

    // Evaluate on test set
    std::pair<float, std::unordered_map<std::string, std::unordered_map<std::string, float>>>
    eval_epoch(int epoch);

    // Create data loaders
    void create_loaders();

    // Compute learning rate for given epoch based on scheduler config
    float get_learning_rate(int epoch) const;

    // Update optimizer learning rate
    void update_learning_rate(float lr);

    ResolveModel model_;
    TrainConfig config_;
    Scalers scalers_;
    MultiTaskLoss loss_fn_;

    // Training data
    torch::Tensor train_continuous_;
    torch::Tensor train_genus_ids_;
    torch::Tensor train_family_ids_;
    torch::Tensor train_species_ids_;     // For embed mode
    torch::Tensor train_species_vector_;  // For sparse mode
    std::unordered_map<std::string, torch::Tensor> train_targets_;

    torch::Tensor test_continuous_;
    torch::Tensor test_genus_ids_;
    torch::Tensor test_family_ids_;
    torch::Tensor test_species_ids_;
    torch::Tensor test_species_vector_;
    std::unordered_map<std::string, torch::Tensor> test_targets_;

    // Best model state for restoring
    std::vector<char> best_model_state_;

    // Optimizer
    std::unique_ptr<torch::optim::AdamW> optimizer_;

    bool data_prepared_ = false;
};

} // namespace resolve
