// Define _USE_MATH_DEFINES before cmath for M_PI on Windows
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

// Fallback definition of M_PI if still not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "resolve/trainer.hpp"
#include "resolve/dataset.hpp"
#include "resolve/utils.hpp"
#include <fstream>
#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <cmath>

namespace resolve {

Trainer::Trainer(
    ResolveModel model,
    const TrainConfig& config
) : model_(model), config_(config), loss_fn_(model->schema().targets, config.phase_boundaries, config.loss_config)
{
    model_->to(config_.device);
}

void Trainer::prepare_data(
    const ResolveDataset& dataset,
    float test_size,
    int seed
) {
    // Delegate to the raw tensor API using data from the dataset
    prepare_data(
        dataset.coordinates(),
        dataset.covariates(),
        dataset.hash_embedding(),
        dataset.species_ids(),
        dataset.species_vector(),
        dataset.genus_ids(),
        dataset.family_ids(),
        dataset.unknown_fraction(),
        dataset.unknown_count(),
        dataset.targets(),
        test_size,
        seed
    );
}

void Trainer::prepare_data(
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
    float test_size,
    int seed
) {
    // Determine n_plots from first defined tensor
    int64_t n_plots = 0;
    if (coordinates.defined() && coordinates.numel() > 0) {
        n_plots = coordinates.size(0);
    } else if (hash_embedding.defined() && hash_embedding.numel() > 0) {
        n_plots = hash_embedding.size(0);
    } else if (species_ids.defined() && species_ids.numel() > 0) {
        n_plots = species_ids.size(0);
    } else if (species_vector.defined() && species_vector.numel() > 0) {
        n_plots = species_vector.size(0);
    } else {
        throw std::runtime_error("No valid input tensors provided");
    }

    // Create indices and shuffle
    std::vector<int64_t> indices(n_plots);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Split indices
    int64_t n_test = static_cast<int64_t>(n_plots * test_size);
    int64_t n_train = n_plots - n_test;

    auto train_idx = torch::tensor(std::vector<int64_t>(indices.begin(), indices.begin() + n_train));
    auto test_idx = torch::tensor(std::vector<int64_t>(indices.begin() + n_train, indices.end()));

    // Build continuous features based on encoding mode
    std::vector<torch::Tensor> continuous_parts;
    push_if_defined(continuous_parts, coordinates);
    push_if_defined(continuous_parts, covariates);
    push_if_defined(continuous_parts, unknown_fraction, 1);
    if (unknown_count.defined() && unknown_count.numel() > 0) {
        continuous_parts.push_back(unknown_count.to(torch::kFloat32).unsqueeze(1));
    }

    // For hash mode, include hash embedding in continuous
    if (model_->species_encoding() == SpeciesEncodingMode::Hash &&
        !model_->uses_explicit_vector()) {
        push_if_defined(continuous_parts, hash_embedding);
    }

    torch::Tensor continuous;
    if (!continuous_parts.empty()) {
        continuous = torch::cat(continuous_parts, /*dim=*/1);
    } else {
        continuous = torch::zeros({n_plots, 0}, torch::kFloat32);
    }

    // Compute scalers on training data
    auto train_continuous = continuous.index_select(0, train_idx);
    if (train_continuous.size(1) > 0) {
        scalers_.continuous_mean = train_continuous.mean(0);
        scalers_.continuous_scale = train_continuous.std(0) + 1e-8f;

        // Scale continuous features
        continuous = (continuous - scalers_.continuous_mean) / scalers_.continuous_scale;
    }

    // Split data
    train_continuous_ = continuous.index_select(0, train_idx);
    test_continuous_ = continuous.index_select(0, test_idx);

    if (genus_ids.defined() && genus_ids.numel() > 0) {
        train_genus_ids_ = genus_ids.index_select(0, train_idx);
        test_genus_ids_ = genus_ids.index_select(0, test_idx);
    }
    if (family_ids.defined() && family_ids.numel() > 0) {
        train_family_ids_ = family_ids.index_select(0, train_idx);
        test_family_ids_ = family_ids.index_select(0, test_idx);
    }
    if (species_ids.defined() && species_ids.numel() > 0) {
        train_species_ids_ = species_ids.index_select(0, train_idx);
        test_species_ids_ = species_ids.index_select(0, test_idx);
    }
    if (species_vector.defined() && species_vector.numel() > 0) {
        train_species_vector_ = species_vector.index_select(0, train_idx);
        test_species_vector_ = species_vector.index_select(0, test_idx);
    }

    // Scale and split targets
    for (const auto& cfg : model_->schema().targets) {
        auto target_it = targets.find(cfg.name);
        if (target_it == targets.end()) continue;

        auto target = target_it->second.clone();

        // Apply transform if needed
        if (cfg.transform == TransformType::Log1p) {
            target = torch::log1p(target);
        }

        // Compute scaler on training data
        auto train_target = target.index_select(0, train_idx);
        auto target_mean = train_target.mean();
        auto target_scale = train_target.std() + 1e-8f;

        scalers_.target_scalers[cfg.name] = {target_mean, target_scale};

        // Scale if regression
        if (cfg.task == TaskType::Regression) {
            target = (target - target_mean) / target_scale;
        }

        train_targets_[cfg.name] = target.index_select(0, train_idx);
        test_targets_[cfg.name] = target.index_select(0, test_idx);
    }

    data_prepared_ = true;
}

void Trainer::create_loaders() {
    // Note: For simplicity, we handle batching manually in train_epoch/eval_epoch
}

float Trainer::train_epoch(int epoch) {
    model_->train();

    int64_t n_train = train_continuous_.size(0);
    int batch_size = config_.batch_size;

    // Shuffle training data
    auto perm = torch::randperm(n_train);

    float total_loss = 0.0f;
    int n_batches = 0;

    for (int64_t start = 0; start < n_train; start += batch_size) {
        int64_t end = std::min(start + batch_size, n_train);
        auto batch_idx = perm.slice(0, start, end);

        // Get batch data
        auto batch_continuous = train_continuous_.index_select(0, batch_idx).to(config_.device);
        auto batch_genus_ids = select_batch(train_genus_ids_, batch_idx, config_.device);
        auto batch_family_ids = select_batch(train_family_ids_, batch_idx, config_.device);
        auto batch_species_ids = select_batch(train_species_ids_, batch_idx, config_.device);
        auto batch_species_vector = select_batch(train_species_vector_, batch_idx, config_.device);

        std::unordered_map<std::string, torch::Tensor> batch_targets;
        for (const auto& [name, tensor] : train_targets_) {
            batch_targets[name] = tensor.index_select(0, batch_idx).to(config_.device);
        }

        // Forward pass
        optimizer_->zero_grad();
        auto predictions = model_->forward(
            batch_continuous, batch_genus_ids, batch_family_ids,
            batch_species_ids, batch_species_vector
        );

        // Compute loss
        std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>> batch_scalers;
        for (const auto& [name, scaler] : scalers_.target_scalers) {
            batch_scalers[name] = {
                scaler.first.to(config_.device),
                scaler.second.to(config_.device)
            };
        }

        auto [loss, _] = loss_fn_.compute(predictions, batch_targets, epoch, batch_scalers);

        // Backward pass
        loss.backward();

        // Gradient clipping
        torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);

        optimizer_->step();

        total_loss += loss.item<float>();
        n_batches++;
    }

    return total_loss / n_batches;
}

std::pair<float, std::unordered_map<std::string, std::unordered_map<std::string, float>>>
Trainer::eval_epoch(int epoch) {
    model_->eval();
    torch::NoGradGuard no_grad;

    // Get all test data
    auto test_continuous = test_continuous_.to(config_.device);
    auto test_genus_ids = to_device_if_defined(test_genus_ids_, config_.device);
    auto test_family_ids = to_device_if_defined(test_family_ids_, config_.device);
    auto test_species_ids = to_device_if_defined(test_species_ids_, config_.device);
    auto test_species_vector = to_device_if_defined(test_species_vector_, config_.device);

    std::unordered_map<std::string, torch::Tensor> test_targets;
    for (const auto& [name, tensor] : test_targets_) {
        test_targets[name] = tensor.to(config_.device);
    }

    // Forward pass
    auto predictions = model_->forward(
        test_continuous, test_genus_ids, test_family_ids,
        test_species_ids, test_species_vector
    );

    // Compute loss
    std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>> batch_scalers;
    for (const auto& [name, scaler] : scalers_.target_scalers) {
        batch_scalers[name] = {
            scaler.first.to(config_.device),
            scaler.second.to(config_.device)
        };
    }

    auto [loss, _] = loss_fn_.compute(predictions, test_targets, epoch, batch_scalers);

    // Compute metrics per target
    std::unordered_map<std::string, std::unordered_map<std::string, float>> all_metrics;

    for (const auto& cfg : model_->schema().targets) {
        auto pred_it = predictions.find(cfg.name);
        auto target_it = test_targets.find(cfg.name);

        if (pred_it != predictions.end() && target_it != test_targets.end()) {
            all_metrics[cfg.name] = Metrics::compute(
                pred_it->second, target_it->second, cfg.task, cfg.transform,
                config_.band_thresholds, cfg.num_classes
            );
        }
    }

    return {loss.item<float>(), all_metrics};
}

float Trainer::get_learning_rate(int epoch) const {
    switch (config_.lr_scheduler) {
        case LRSchedulerType::StepLR: {
            // Step decay: multiply LR by gamma every lr_step_size epochs
            int n_decays = epoch / config_.lr_step_size;
            return config_.lr * std::pow(config_.lr_gamma, static_cast<float>(n_decays));
        }
        case LRSchedulerType::CosineAnnealing: {
            // Cosine annealing from lr to lr_min
            float progress = static_cast<float>(epoch) / config_.max_epochs;
            float cosine = 0.5f * (1.0f + std::cos(M_PI * progress));
            return config_.lr_min + (config_.lr - config_.lr_min) * cosine;
        }
        case LRSchedulerType::None:
        default:
            return config_.lr;
    }
}

void Trainer::update_learning_rate(float lr) {
    for (auto& group : optimizer_->param_groups()) {
        static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
    }
}

// Helper to write progress.json
static void write_progress_file(
    const std::string& checkpoint_dir,
    int epoch,
    int max_epochs,
    int best_epoch,
    float best_loss,
    int epochs_without_improvement,
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& metrics
) {
    namespace fs = std::filesystem;
    fs::create_directories(checkpoint_dir);

    std::string progress_path = checkpoint_dir + "/progress.json";
    std::ofstream file(progress_path);
    if (!file.is_open()) return;

    file << "{\n";
    file << "  \"epoch\": " << epoch << ",\n";
    file << "  \"max_epochs\": " << max_epochs << ",\n";
    file << "  \"best_epoch\": " << best_epoch << ",\n";
    file << "  \"best_loss\": " << best_loss << ",\n";
    file << "  \"epochs_without_improvement\": " << epochs_without_improvement << ",\n";
    file << "  \"progress_pct\": " << (100.0f * epoch / max_epochs) << ",\n";

    // Write best metric (first target's first band metric if available)
    float best_metric = 0.0f;
    for (const auto& [target_name, target_metrics] : metrics) {
        for (const auto& [metric_name, value] : target_metrics) {
            if (metric_name.find("band_") == 0) {
                best_metric = value;
                break;
            }
        }
        break;
    }
    file << "  \"best_metric\": " << best_metric << "\n";
    file << "}\n";
}

TrainResult Trainer::fit() {
    if (!data_prepared_) {
        throw std::runtime_error("Data must be prepared before training");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create checkpoint directory if specified
    bool use_checkpoints = !config_.checkpoint_dir.empty();
    if (use_checkpoints) {
        std::filesystem::create_directories(config_.checkpoint_dir);
    }

    // Create optimizer
    optimizer_ = std::make_unique<torch::optim::AdamW>(
        model_->parameters(),
        torch::optim::AdamWOptions(config_.lr).weight_decay(config_.weight_decay)
    );

    TrainResult result;
    float best_loss = std::numeric_limits<float>::infinity();
    int patience_counter = 0;

    for (int epoch = 0; epoch < config_.max_epochs; ++epoch) {
        // Update learning rate based on scheduler
        float current_lr = get_learning_rate(epoch);
        update_learning_rate(current_lr);

        float train_loss = train_epoch(epoch);
        auto [test_loss, metrics] = eval_epoch(epoch);

        result.train_loss_history.push_back(train_loss);
        result.test_loss_history.push_back(test_loss);

        // Check for improvement
        if (test_loss < best_loss) {
            best_loss = test_loss;
            result.best_epoch = epoch;
            result.final_metrics = metrics;
            patience_counter = 0;

            // Save best model state to memory
            std::ostringstream oss;
            torch::serialize::OutputArchive archive;
            model_->save(archive);
            archive.save_to(oss);
            auto str = oss.str();
            best_model_state_.assign(str.begin(), str.end());

            // Save best checkpoint
            if (use_checkpoints) {
                save(config_.checkpoint_dir + "/best.pt");
            }
        } else {
            patience_counter++;
            if (patience_counter >= config_.patience) {
                config_.log("Early stopping at epoch " + std::to_string(epoch));
                break;
            }
        }

        // Periodic checkpoint saving
        if (use_checkpoints && config_.checkpoint_every > 0 && (epoch + 1) % config_.checkpoint_every == 0) {
            save(config_.checkpoint_dir + "/checkpoint_" + std::to_string(epoch + 1) + ".pt");
        }

        // Write progress file
        if (use_checkpoints) {
            write_progress_file(
                config_.checkpoint_dir, epoch, config_.max_epochs,
                result.best_epoch, best_loss, patience_counter, result.final_metrics
            );
        }

        // Print progress
        // Print progress using log callback
        if (epoch % 10 == 0) {
            std::ostringstream msg;
            msg << "Epoch " << epoch << " - Train: " << train_loss << " Test: " << test_loss;
            if (config_.lr_scheduler != LRSchedulerType::None) {
                msg << " LR: " << current_lr;
            }
            config_.log(msg.str());
        }
    }

    // Restore best model state
    if (!best_model_state_.empty()) {
        std::istringstream iss(std::string(best_model_state_.begin(), best_model_state_.end()));
        torch::serialize::InputArchive archive;
        archive.load_from(iss);
        model_->load(archive);
    }

    // Save final checkpoint
    if (use_checkpoints) {
        save(config_.checkpoint_dir + "/checkpoint.pt");
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.train_time_seconds = std::chrono::duration<float>(end_time - start_time).count();

    return result;
}

void Trainer::save(const std::string& path) const {
    torch::serialize::OutputArchive archive;

    // Save model
    model_->save(archive);

    // Save config
    archive.write("species_encoding", torch::tensor(static_cast<int>(model_->config().species_encoding)));
    archive.write("uses_explicit_vector", torch::tensor(static_cast<int>(model_->config().uses_explicit_vector)));
    archive.write("hash_dim", torch::tensor(model_->config().hash_dim));
    archive.write("species_embed_dim", torch::tensor(model_->config().species_embed_dim));
    archive.write("genus_emb_dim", torch::tensor(model_->config().genus_emb_dim));
    archive.write("family_emb_dim", torch::tensor(model_->config().family_emb_dim));
    archive.write("top_k", torch::tensor(model_->config().top_k));
    archive.write("top_k_species", torch::tensor(model_->config().top_k_species));
    archive.write("n_taxonomy_slots", torch::tensor(model_->config().n_taxonomy_slots));
    archive.write("dropout", torch::tensor(model_->config().dropout));

    // Save hidden dims
    std::vector<int64_t> hidden_dims_vec(model_->config().hidden_dims);
    archive.write("hidden_dims", torch::tensor(hidden_dims_vec));

    // Save scalers
    if (scalers_.continuous_mean.defined()) {
        archive.write("continuous_mean", scalers_.continuous_mean);
        archive.write("continuous_scale", scalers_.continuous_scale);
    }

    // Save target scalers
    archive.write("n_target_scalers", torch::tensor(static_cast<int64_t>(scalers_.target_scalers.size())));
    int idx = 0;
    for (const auto& [name, scaler] : scalers_.target_scalers) {
        archive.write("target_scaler_mean_" + std::to_string(idx), scaler.first);
        archive.write("target_scaler_scale_" + std::to_string(idx), scaler.second);
        idx++;
    }

    archive.save_to(path);
}

std::tuple<ResolveModel, Scalers> Trainer::load(
    const std::string& path,
    torch::Device device
) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);

    // Load config values
    torch::Tensor species_encoding_t, uses_explicit_vector_t;
    torch::Tensor hash_dim_t, species_embed_dim_t;
    torch::Tensor genus_emb_dim_t, family_emb_dim_t;
    torch::Tensor top_k_t, top_k_species_t, n_taxonomy_slots_t;
    torch::Tensor dropout_t, hidden_dims_t;

    archive.read("species_encoding", species_encoding_t);
    archive.read("uses_explicit_vector", uses_explicit_vector_t);
    archive.read("hash_dim", hash_dim_t);
    archive.read("species_embed_dim", species_embed_dim_t);
    archive.read("genus_emb_dim", genus_emb_dim_t);
    archive.read("family_emb_dim", family_emb_dim_t);
    archive.read("top_k", top_k_t);
    archive.read("top_k_species", top_k_species_t);
    archive.read("n_taxonomy_slots", n_taxonomy_slots_t);
    archive.read("dropout", dropout_t);
    archive.read("hidden_dims", hidden_dims_t);

    ModelConfig config;
    config.species_encoding = static_cast<SpeciesEncodingMode>(species_encoding_t.item<int>());
    config.uses_explicit_vector = uses_explicit_vector_t.item<int>() != 0;
    config.hash_dim = hash_dim_t.item<int>();
    config.species_embed_dim = species_embed_dim_t.item<int>();
    config.genus_emb_dim = genus_emb_dim_t.item<int>();
    config.family_emb_dim = family_emb_dim_t.item<int>();
    config.top_k = top_k_t.item<int>();
    config.top_k_species = top_k_species_t.item<int>();
    config.n_taxonomy_slots = n_taxonomy_slots_t.item<int>();
    config.dropout = dropout_t.item<float>();

    std::vector<int64_t> hidden_dims(hidden_dims_t.size(0));
    for (int i = 0; i < hidden_dims_t.size(0); ++i) {
        hidden_dims[i] = hidden_dims_t[i].item<int64_t>();
    }
    config.hidden_dims = hidden_dims;

    // Load scalers
    Scalers scalers;
    try {
        archive.read("continuous_mean", scalers.continuous_mean);
        archive.read("continuous_scale", scalers.continuous_scale);
    } catch (...) {
        // Scalers may not be present
    }

    // Load target scalers
    torch::Tensor n_target_scalers_t;
    try {
        archive.read("n_target_scalers", n_target_scalers_t);
        int64_t n_scalers = n_target_scalers_t.item<int64_t>();
        for (int64_t i = 0; i < n_scalers; ++i) {
            torch::Tensor mean, scale;
            archive.read("target_scaler_mean_" + std::to_string(i), mean);
            archive.read("target_scaler_scale_" + std::to_string(i), scale);
            // Note: target name is lost - would need to save names too for full implementation
        }
    } catch (...) {
        // Target scalers may not be present
    }

    // Create placeholder model (would need full schema)
    ResolveSchema schema;
    ResolveModel model(schema, config);

    // Load model weights
    model->load(archive);
    model->to(device);

    return {model, scalers};
}

} // namespace resolve
