#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include "resolve/plot_encoder.hpp"
#include "resolve/trainer.hpp"

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

namespace resolve {

/**
 * Predictor class for inference with trained RESOLVE models.
 *
 * Loads a trained model checkpoint and provides prediction methods
 * for new data.
 */
class Predictor {
public:
    Predictor() = default;

    /**
     * Load a trained model from checkpoint.
     *
     * @param checkpoint_path Path to the saved checkpoint (.pt file)
     * @param device Device to use for inference ("cpu", "cuda", "cuda:0", etc.)
     */
    void load(const std::string& checkpoint_path, const std::string& device = "cpu") {
        device_ = torch::Device(device);

        // Load checkpoint
        torch::serialize::InputArchive archive;
        archive.load_from(checkpoint_path);

        // Load config
        std::string config_json;
        archive.read("config_json", config_json);
        auto config = nlohmann::json::parse(config_json);

        // Reconstruct model config
        ModelConfig model_config;
        model_config.encoder_dim = config["encoder_dim"].get<int>();
        model_config.hidden_dim = config["hidden_dim"].get<int>();
        model_config.n_encoder_layers = config["n_encoder_layers"].get<int>();
        model_config.dropout = config["dropout"].get<float>();
        model_config.hash_dim = config["hash_dim"].get<int>();
        model_config.genus_vocab_size = config["genus_vocab_size"].get<int>();
        model_config.family_vocab_size = config["family_vocab_size"].get<int>();
        model_config.species_vocab_size = config.value("species_vocab_size", 0);
        model_config.n_species_vector = config.value("n_species_vector", 0);
        model_config.n_continuous = config["n_continuous"].get<int>();
        model_config.top_k = config["top_k"].get<int>();
        model_config.mode = static_cast<SpeciesEncodingMode>(config["mode"].get<int>());

        // Load target configs
        for (const auto& tc : config["targets"]) {
            TargetConfig target;
            target.name = tc["name"].get<std::string>();
            target.task = static_cast<TaskType>(tc["task"].get<int>());
            target.n_classes = tc.value("n_classes", 1);
            target.transform = static_cast<TransformType>(tc["transform"].get<int>());
            target_configs_.push_back(target);
        }

        // Create model
        model_ = ResolveModel(model_config, target_configs_);

        // Load state dict
        torch::serialize::InputArchive model_archive;
        archive.read("model_state", model_archive);
        model_->load_state_dict(model_archive);

        // Load plot encoder
        // TODO: Update checkpoint format to save/load PlotEncoder
        // torch::serialize::InputArchive encoder_archive;
        // archive.read("plot_encoder", encoder_archive);
        // plot_encoder_ = PlotEncoder::load(encoder_archive);

        // Load scalers
        torch::serialize::InputArchive scaler_archive;
        archive.read("continuous_scaler", scaler_archive);
        continuous_scaler_.load(scaler_archive);

        torch::serialize::InputArchive target_scalers_archive;
        archive.read("target_scalers", target_scalers_archive);
        for (const auto& cfg : target_configs_) {
            if (cfg.task == TaskType::Regression) {
                torch::serialize::InputArchive ts_archive;
                target_scalers_archive.read(cfg.name, ts_archive);
                target_scalers_[cfg.name].load(ts_archive);
            }
        }

        model_->to(device_);
        model_->eval();
        is_loaded_ = true;
    }

    /**
     * Check if model is loaded.
     */
    bool is_loaded() const { return is_loaded_; }

    /**
     * Get target configurations.
     */
    const std::vector<TargetConfig>& target_configs() const { return target_configs_; }

    /**
     * Predict from observation records and continuous features.
     *
     * TODO: Migrate to PlotEncoder-based prediction
     * The previous version used SpeciesEncoder which has been removed.
     *
     * @param plot_data Vector of plot records (one per plot)
     * @param obs_data Vector of observation records
     * @param plot_ids Vector of plot IDs (in order of desired output)
     * @param continuous Continuous features tensor (n_plots, n_continuous)
     * @param return_latent Whether to return latent representations
     * @return ResolvePredictions with predictions for each target
     */
    ResolvePredictions predict(
        const std::vector<PlotRecord>& plot_data,
        const std::vector<ObservationRecord>& obs_data,
        const std::vector<std::string>& plot_ids,
        torch::Tensor continuous,
        bool return_latent = false
    ) {
        if (!is_loaded_) {
            throw std::runtime_error("Model not loaded. Call load() first.");
        }

        torch::NoGradGuard no_grad;

        // TODO: Use plot_encoder_.transform(plot_data, obs_data, plot_ids)
        // and build model inputs from EncodedPlotData

        // Scale continuous features
        if (continuous.size(1) > 0) {
            continuous = continuous_scaler_.transform(continuous);
        }

        continuous = continuous.to(device_);

        // TODO: Get encoded features from PlotEncoder and forward through model
        // For now, return empty predictions
        ResolvePredictions result;
        result.plot_ids = plot_ids;

        return result;
    }

    /**
     * Predict from a ResolveDataset.
     *
     * TODO: Migrate to PlotEncoder-based prediction
     * The previous version used SpeciesEncoder which has been removed.
     *
     * @param dataset The dataset to predict on
     * @param batch_size Batch size for prediction
     * @param return_latent Whether to return latent representations
     * @return ResolvePredictions with predictions for each target
     */
    ResolvePredictions predict_dataset(
        const ResolveDataset& dataset,
        int batch_size = 64,
        bool return_latent = false
    ) {
        if (!is_loaded_) {
            throw std::runtime_error("Model not loaded. Call load() first.");
        }

        torch::NoGradGuard no_grad;

        auto plot_ids = dataset.plot_ids();

        // TODO: Convert dataset to PlotRecord/ObservationRecord format
        // and use plot_encoder_ to encode

        ResolvePredictions result;
        result.plot_ids = plot_ids;

        return result;
    }

    /**
     * Get confidence intervals for regression predictions using MC Dropout.
     *
     * TODO: Migrate to PlotEncoder-based prediction
     *
     * @param plot_data Vector of plot records
     * @param obs_data Vector of observation records
     * @param plot_ids Vector of plot IDs
     * @param continuous Continuous features tensor
     * @param n_samples Number of MC samples
     * @param confidence Confidence level (e.g., 0.95 for 95% CI)
     * @return Map of target name to (lower, median, upper) tensors
     */
    std::unordered_map<std::string, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
    predict_with_uncertainty(
        const std::vector<PlotRecord>& plot_data,
        const std::vector<ObservationRecord>& obs_data,
        const std::vector<std::string>& plot_ids,
        torch::Tensor continuous,
        int n_samples = 100,
        float confidence = 0.95f
    ) {
        if (!is_loaded_) {
            throw std::runtime_error("Model not loaded. Call load() first.");
        }

        // Enable dropout for MC sampling
        model_->train();

        std::unordered_map<std::string, std::vector<torch::Tensor>> samples;

        for (int i = 0; i < n_samples; ++i) {
            auto preds = predict(plot_data, obs_data, plot_ids, continuous, false);
            for (const auto& cfg : target_configs_) {
                if (preds.predictions.count(cfg.name) > 0) {
                    samples[cfg.name].push_back(preds.predictions[cfg.name]);
                }
            }
        }

        // Back to eval mode
        model_->eval();

        // Compute statistics
        std::unordered_map<std::string, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result;

        float lower_q = (1.0f - confidence) / 2.0f;
        float upper_q = 1.0f - lower_q;

        for (const auto& cfg : target_configs_) {
            if (samples.count(cfg.name) == 0 || samples[cfg.name].empty()) continue;

            auto stacked = torch::stack(samples[cfg.name], 0);  // (n_samples, n_plots)

            auto sorted = std::get<0>(torch::sort(stacked, 0));
            int lower_idx = static_cast<int>(lower_q * n_samples);
            int upper_idx = static_cast<int>(upper_q * n_samples);
            int median_idx = n_samples / 2;

            auto lower = sorted[lower_idx];
            auto median = sorted[median_idx];
            auto upper = sorted[upper_idx];

            result[cfg.name] = std::make_tuple(lower, median, upper);
        }

        return result;
    }

private:
    ResolveModel model_{nullptr};
    PlotEncoder plot_encoder_;
    StandardScaler continuous_scaler_;
    std::unordered_map<std::string, StandardScaler> target_scalers_;
    std::vector<TargetConfig> target_configs_;
    torch::Device device_{torch::kCPU};
    bool is_loaded_ = false;
};

} // namespace resolve
