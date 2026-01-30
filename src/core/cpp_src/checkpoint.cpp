#include "resolve/checkpoint.hpp"
#include <fstream>
#include <filesystem>

namespace resolve {

void write_progress_file(
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

void save_model_config(
    torch::serialize::OutputArchive& archive,
    const ModelConfig& config
) {
    archive.write("species_encoding", torch::tensor(static_cast<int>(config.species_encoding)));
    archive.write("uses_explicit_vector", torch::tensor(static_cast<int>(config.uses_explicit_vector)));
    archive.write("hash_dim", torch::tensor(config.hash_dim));
    archive.write("species_embed_dim", torch::tensor(config.species_embed_dim));
    archive.write("genus_emb_dim", torch::tensor(config.genus_emb_dim));
    archive.write("family_emb_dim", torch::tensor(config.family_emb_dim));
    archive.write("top_k", torch::tensor(config.top_k));
    archive.write("top_k_species", torch::tensor(config.top_k_species));
    archive.write("n_taxonomy_slots", torch::tensor(config.n_taxonomy_slots));
    archive.write("dropout", torch::tensor(config.dropout));

    // Save hidden dims
    std::vector<int64_t> hidden_dims_vec(config.hidden_dims);
    archive.write("hidden_dims", torch::tensor(hidden_dims_vec));
}

ModelConfig load_model_config(
    torch::serialize::InputArchive& archive
) {
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

    return config;
}

void save_scalers(
    torch::serialize::OutputArchive& archive,
    const Scalers& scalers
) {
    if (scalers.continuous_mean.defined()) {
        archive.write("continuous_mean", scalers.continuous_mean);
        archive.write("continuous_scale", scalers.continuous_scale);
    }

    // Save target scalers
    archive.write("n_target_scalers", torch::tensor(static_cast<int64_t>(scalers.target_scalers.size())));
    int idx = 0;
    for (const auto& [name, scaler] : scalers.target_scalers) {
        archive.write("target_scaler_mean_" + std::to_string(idx), scaler.first);
        archive.write("target_scaler_scale_" + std::to_string(idx), scaler.second);
        idx++;
    }
}

Scalers load_scalers(
    torch::serialize::InputArchive& archive
) {
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

    return scalers;
}

} // namespace resolve
