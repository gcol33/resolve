#include "resolve/checkpoint.hpp"
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace resolve {

std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t);
#else
    localtime_r(&time_t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

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

    // Save MoE configuration
    archive.write("moe_routing", torch::tensor(static_cast<int>(config.moe_routing)));
    archive.write("n_experts", torch::tensor(config.n_experts));
    archive.write("moe_top_k", torch::tensor(config.moe_top_k));
    archive.write("moe_noise_std", torch::tensor(config.moe_noise_std));
    archive.write("moe_aux_loss_weight", torch::tensor(config.moe_aux_loss_weight));

    // Save expert hidden dims
    std::vector<int64_t> expert_dims_vec(config.expert_hidden_dims);
    archive.write("expert_hidden_dims", torch::tensor(expert_dims_vec));

    // Save configurable architecture fields
    archive.write("activation", torch::tensor(static_cast<int>(config.activation)));
    archive.write("normalization", torch::tensor(static_cast<int>(config.normalization)));
    archive.write("norm_groups", torch::tensor(config.norm_groups));
    archive.write("use_residual", torch::tensor(static_cast<int>(config.use_residual)));
    archive.write("leaky_relu_slope", torch::tensor(config.leaky_relu_slope));
    archive.write("elu_alpha", torch::tensor(config.elu_alpha));

    // Save head architecture
    std::vector<int64_t> head_dims_vec(config.head_hidden_dims);
    archive.write("head_hidden_dims", torch::tensor(head_dims_vec));
    archive.write("head_activation", torch::tensor(static_cast<int>(config.head_activation)));
    archive.write("head_dropout", torch::tensor(config.head_dropout));
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

    // Load MoE configuration (with backward compatibility)
    try {
        torch::Tensor moe_routing_t, n_experts_t, moe_top_k_t, moe_noise_std_t, moe_aux_loss_weight_t;
        torch::Tensor expert_hidden_dims_t;

        archive.read("moe_routing", moe_routing_t);
        config.moe_routing = static_cast<MoERoutingType>(moe_routing_t.item<int>());

        archive.read("n_experts", n_experts_t);
        config.n_experts = n_experts_t.item<int>();

        archive.read("moe_top_k", moe_top_k_t);
        config.moe_top_k = moe_top_k_t.item<int>();

        archive.read("moe_noise_std", moe_noise_std_t);
        config.moe_noise_std = moe_noise_std_t.item<float>();

        archive.read("moe_aux_loss_weight", moe_aux_loss_weight_t);
        config.moe_aux_loss_weight = moe_aux_loss_weight_t.item<float>();

        archive.read("expert_hidden_dims", expert_hidden_dims_t);
        config.expert_hidden_dims.clear();
        for (int i = 0; i < expert_hidden_dims_t.size(0); ++i) {
            config.expert_hidden_dims.push_back(expert_hidden_dims_t[i].item<int64_t>());
        }
    } catch (...) {
        // MoE config may not be present in older checkpoints - use defaults
        config.moe_routing = MoERoutingType::None;
    }

    // Load configurable architecture fields (with backward compatibility)
    try {
        torch::Tensor activation_t, normalization_t, norm_groups_t, use_residual_t;
        torch::Tensor leaky_relu_slope_t, elu_alpha_t;
        torch::Tensor head_hidden_dims_t, head_activation_t, head_dropout_t;

        archive.read("activation", activation_t);
        config.activation = static_cast<ActivationType>(activation_t.item<int>());

        archive.read("normalization", normalization_t);
        config.normalization = static_cast<NormLayerType>(normalization_t.item<int>());

        archive.read("norm_groups", norm_groups_t);
        config.norm_groups = norm_groups_t.item<int>();

        archive.read("use_residual", use_residual_t);
        config.use_residual = use_residual_t.item<int>() != 0;

        archive.read("leaky_relu_slope", leaky_relu_slope_t);
        config.leaky_relu_slope = leaky_relu_slope_t.item<float>();

        archive.read("elu_alpha", elu_alpha_t);
        config.elu_alpha = elu_alpha_t.item<float>();

        // Load head architecture
        archive.read("head_hidden_dims", head_hidden_dims_t);
        config.head_hidden_dims.clear();
        for (int i = 0; i < head_hidden_dims_t.size(0); ++i) {
            config.head_hidden_dims.push_back(head_hidden_dims_t[i].item<int64_t>());
        }

        archive.read("head_activation", head_activation_t);
        config.head_activation = static_cast<ActivationType>(head_activation_t.item<int>());

        archive.read("head_dropout", head_dropout_t);
        config.head_dropout = head_dropout_t.item<float>();
    } catch (...) {
        // Architecture config may not be present in older checkpoints - use defaults
        // Defaults match the legacy behavior (GELU + BatchNorm + no residual)
        config.activation = ActivationType::GELU;
        config.normalization = NormLayerType::BatchNorm;
        config.norm_groups = kDefaultNormGroups;
        config.use_residual = false;
        config.leaky_relu_slope = kDefaultLeakyReLUSlope;
        config.elu_alpha = kDefaultELUAlpha;
        config.head_hidden_dims = {};
        config.head_activation = ActivationType::GELU;
        config.head_dropout = 0.0f;
    }

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

void save_schema(
    torch::serialize::OutputArchive& archive,
    const ResolveSchema& schema
) {
    archive.write("schema_n_plots", torch::tensor(schema.n_plots));
    archive.write("schema_n_species", torch::tensor(schema.n_species));
    archive.write("schema_n_species_vocab", torch::tensor(schema.n_species_vocab));
    archive.write("schema_has_coordinates", torch::tensor(static_cast<int>(schema.has_coordinates)));
    archive.write("schema_has_abundance", torch::tensor(static_cast<int>(schema.has_abundance)));
    archive.write("schema_has_taxonomy", torch::tensor(static_cast<int>(schema.has_taxonomy)));
    archive.write("schema_n_genera", torch::tensor(schema.n_genera));
    archive.write("schema_n_families", torch::tensor(schema.n_families));
    archive.write("schema_n_genera_vocab", torch::tensor(schema.n_genera_vocab));
    archive.write("schema_n_families_vocab", torch::tensor(schema.n_families_vocab));
    archive.write("schema_track_unknown_fraction", torch::tensor(static_cast<int>(schema.track_unknown_fraction)));
    archive.write("schema_track_unknown_count", torch::tensor(static_cast<int>(schema.track_unknown_count)));
    archive.write("schema_n_covariates", torch::tensor(static_cast<int64_t>(schema.covariate_names.size())));
    archive.write("schema_n_targets", torch::tensor(static_cast<int64_t>(schema.targets.size())));
    for (size_t i = 0; i < schema.targets.size(); ++i) {
        const auto& target = schema.targets[i];
        std::string prefix = "schema_target_" + std::to_string(i) + "_";
        archive.write(prefix + "task", torch::tensor(static_cast<int>(target.task)));
        archive.write(prefix + "transform", torch::tensor(static_cast<int>(target.transform)));
        archive.write(prefix + "num_classes", torch::tensor(target.num_classes));
        archive.write(prefix + "weight", torch::tensor(target.weight));
    }
}

ResolveSchema load_schema(
    torch::serialize::InputArchive& archive
) {
    ResolveSchema schema;
    torch::Tensor t;
    archive.read("schema_n_plots", t);
    schema.n_plots = t.item<int64_t>();
    archive.read("schema_n_species", t);
    schema.n_species = t.item<int64_t>();
    archive.read("schema_n_species_vocab", t);
    schema.n_species_vocab = t.item<int64_t>();
    archive.read("schema_has_coordinates", t);
    schema.has_coordinates = t.item<int>() != 0;
    archive.read("schema_has_abundance", t);
    schema.has_abundance = t.item<int>() != 0;
    archive.read("schema_has_taxonomy", t);
    schema.has_taxonomy = t.item<int>() != 0;
    archive.read("schema_n_genera", t);
    schema.n_genera = t.item<int64_t>();
    archive.read("schema_n_families", t);
    schema.n_families = t.item<int64_t>();
    archive.read("schema_n_genera_vocab", t);
    schema.n_genera_vocab = t.item<int64_t>();
    archive.read("schema_n_families_vocab", t);
    schema.n_families_vocab = t.item<int64_t>();
    archive.read("schema_track_unknown_fraction", t);
    schema.track_unknown_fraction = t.item<int>() != 0;
    archive.read("schema_track_unknown_count", t);
    schema.track_unknown_count = t.item<int>() != 0;
    archive.read("schema_n_covariates", t);
    int64_t n_covariates = t.item<int64_t>();
    schema.covariate_names.resize(n_covariates);
    archive.read("schema_n_targets", t);
    int64_t n_targets = t.item<int64_t>();
    schema.targets.resize(n_targets);
    for (int64_t i = 0; i < n_targets; ++i) {
        std::string prefix = "schema_target_" + std::to_string(i) + "_";
        archive.read(prefix + "task", t);
        schema.targets[i].task = static_cast<TaskType>(t.item<int>());
        archive.read(prefix + "transform", t);
        schema.targets[i].transform = static_cast<TransformType>(t.item<int>());
        archive.read(prefix + "num_classes", t);
        schema.targets[i].num_classes = t.item<int>();
        archive.read(prefix + "weight", t);
        schema.targets[i].weight = t.item<float>();
    }
    return schema;
}

void save_train_config(
    torch::serialize::OutputArchive& archive,
    const TrainConfig& config
) {
    archive.write("train_batch_size", torch::tensor(config.batch_size));
    archive.write("train_max_epochs", torch::tensor(config.max_epochs));
    archive.write("train_patience", torch::tensor(config.patience));
    archive.write("train_lr", torch::tensor(config.lr));
    archive.write("train_weight_decay", torch::tensor(config.weight_decay));
    archive.write("train_phase_boundary_1", torch::tensor(config.phase_boundaries.first));
    archive.write("train_phase_boundary_2", torch::tensor(config.phase_boundaries.second));
    archive.write("train_loss_config", torch::tensor(static_cast<int>(config.loss_config)));
    archive.write("train_lr_scheduler", torch::tensor(static_cast<int>(config.lr_scheduler)));
    archive.write("train_lr_step_size", torch::tensor(config.lr_step_size));
    archive.write("train_lr_gamma", torch::tensor(config.lr_gamma));
    archive.write("train_lr_min", torch::tensor(config.lr_min));

    // Save band thresholds
    std::vector<float> thresholds(config.band_thresholds);
    archive.write("train_band_thresholds", torch::tensor(thresholds));
}

TrainConfig load_train_config(
    torch::serialize::InputArchive& archive
) {
    TrainConfig config;
    torch::Tensor t;

    try {
        archive.read("train_batch_size", t);
        config.batch_size = t.item<int>();
        archive.read("train_max_epochs", t);
        config.max_epochs = t.item<int>();
        archive.read("train_patience", t);
        config.patience = t.item<int>();
        archive.read("train_lr", t);
        config.lr = t.item<float>();
        archive.read("train_weight_decay", t);
        config.weight_decay = t.item<float>();
        archive.read("train_phase_boundary_1", t);
        config.phase_boundaries.first = t.item<int>();
        archive.read("train_phase_boundary_2", t);
        config.phase_boundaries.second = t.item<int>();
        archive.read("train_loss_config", t);
        config.loss_config = static_cast<LossConfigMode>(t.item<int>());
        archive.read("train_lr_scheduler", t);
        config.lr_scheduler = static_cast<LRSchedulerType>(t.item<int>());
        archive.read("train_lr_step_size", t);
        config.lr_step_size = t.item<int>();
        archive.read("train_lr_gamma", t);
        config.lr_gamma = t.item<float>();
        archive.read("train_lr_min", t);
        config.lr_min = t.item<float>();

        archive.read("train_band_thresholds", t);
        config.band_thresholds.clear();
        for (int i = 0; i < t.size(0); ++i) {
            config.band_thresholds.push_back(t[i].item<float>());
        }
    } catch (...) {
        // TrainConfig may not be present in older checkpoints
    }

    return config;
}

void save_run_metadata(
    torch::serialize::OutputArchive& archive,
    const RunMetadata& metadata
) {
    // Save version as bytes
    std::vector<uint8_t> version_bytes(metadata.resolve_version.begin(), metadata.resolve_version.end());
    archive.write("meta_version_len", torch::tensor(static_cast<int64_t>(version_bytes.size())));
    if (!version_bytes.empty()) {
        archive.write("meta_version", torch::from_blob(
            version_bytes.data(), {static_cast<int64_t>(version_bytes.size())}, torch::kUInt8).clone());
    }

    // Save timestamps as bytes
    std::vector<uint8_t> created_bytes(metadata.created_at.begin(), metadata.created_at.end());
    archive.write("meta_created_len", torch::tensor(static_cast<int64_t>(created_bytes.size())));
    if (!created_bytes.empty()) {
        archive.write("meta_created", torch::from_blob(
            created_bytes.data(), {static_cast<int64_t>(created_bytes.size())}, torch::kUInt8).clone());
    }

    std::vector<uint8_t> completed_bytes(metadata.completed_at.begin(), metadata.completed_at.end());
    archive.write("meta_completed_len", torch::tensor(static_cast<int64_t>(completed_bytes.size())));
    if (!completed_bytes.empty()) {
        archive.write("meta_completed", torch::from_blob(
            completed_bytes.data(), {static_cast<int64_t>(completed_bytes.size())}, torch::kUInt8).clone());
    }

    // Save numeric fields
    archive.write("meta_train_time", torch::tensor(metadata.train_time_seconds));
    archive.write("meta_n_plots_train", torch::tensor(metadata.n_plots_train));
    archive.write("meta_n_plots_test", torch::tensor(metadata.n_plots_test));
    archive.write("meta_best_epoch", torch::tensor(metadata.best_epoch));
    archive.write("meta_total_epochs", torch::tensor(metadata.total_epochs));

    // Save final metrics as flattened tensors
    int64_t n_targets = static_cast<int64_t>(metadata.final_metrics.size());
    archive.write("meta_n_targets", torch::tensor(n_targets));

    int target_idx = 0;
    for (const auto& [target_name, metrics] : metadata.final_metrics) {
        std::string prefix = "meta_target_" + std::to_string(target_idx) + "_";

        // Save target name
        std::vector<uint8_t> name_bytes(target_name.begin(), target_name.end());
        archive.write(prefix + "name_len", torch::tensor(static_cast<int64_t>(name_bytes.size())));
        if (!name_bytes.empty()) {
            archive.write(prefix + "name", torch::from_blob(
                name_bytes.data(), {static_cast<int64_t>(name_bytes.size())}, torch::kUInt8).clone());
        }

        // Save metrics for this target
        int64_t n_metrics = static_cast<int64_t>(metrics.size());
        archive.write(prefix + "n_metrics", torch::tensor(n_metrics));

        int metric_idx = 0;
        for (const auto& [metric_name, value] : metrics) {
            std::string m_prefix = prefix + "metric_" + std::to_string(metric_idx) + "_";

            std::vector<uint8_t> m_name_bytes(metric_name.begin(), metric_name.end());
            archive.write(m_prefix + "name_len", torch::tensor(static_cast<int64_t>(m_name_bytes.size())));
            if (!m_name_bytes.empty()) {
                archive.write(m_prefix + "name", torch::from_blob(
                    m_name_bytes.data(), {static_cast<int64_t>(m_name_bytes.size())}, torch::kUInt8).clone());
            }
            archive.write(m_prefix + "value", torch::tensor(value));
            metric_idx++;
        }
        target_idx++;
    }
}

RunMetadata load_run_metadata(
    torch::serialize::InputArchive& archive
) {
    RunMetadata metadata;
    torch::Tensor t;

    try {
        // Load version
        archive.read("meta_version_len", t);
        int64_t version_len = t.item<int64_t>();
        if (version_len > 0) {
            archive.read("meta_version", t);
            auto ptr = t.data_ptr<uint8_t>();
            metadata.resolve_version = std::string(reinterpret_cast<const char*>(ptr), version_len);
        }

        // Load timestamps
        archive.read("meta_created_len", t);
        int64_t created_len = t.item<int64_t>();
        if (created_len > 0) {
            archive.read("meta_created", t);
            auto ptr = t.data_ptr<uint8_t>();
            metadata.created_at = std::string(reinterpret_cast<const char*>(ptr), created_len);
        }

        archive.read("meta_completed_len", t);
        int64_t completed_len = t.item<int64_t>();
        if (completed_len > 0) {
            archive.read("meta_completed", t);
            auto ptr = t.data_ptr<uint8_t>();
            metadata.completed_at = std::string(reinterpret_cast<const char*>(ptr), completed_len);
        }

        // Load numeric fields
        archive.read("meta_train_time", t);
        metadata.train_time_seconds = t.item<float>();
        archive.read("meta_n_plots_train", t);
        metadata.n_plots_train = t.item<int64_t>();
        archive.read("meta_n_plots_test", t);
        metadata.n_plots_test = t.item<int64_t>();
        archive.read("meta_best_epoch", t);
        metadata.best_epoch = t.item<int>();
        archive.read("meta_total_epochs", t);
        metadata.total_epochs = t.item<int>();

        // Load final metrics
        archive.read("meta_n_targets", t);
        int64_t n_targets = t.item<int64_t>();

        for (int64_t i = 0; i < n_targets; ++i) {
            std::string prefix = "meta_target_" + std::to_string(i) + "_";

            // Load target name
            archive.read(prefix + "name_len", t);
            int64_t name_len = t.item<int64_t>();
            std::string target_name;
            if (name_len > 0) {
                archive.read(prefix + "name", t);
                auto ptr = t.data_ptr<uint8_t>();
                target_name = std::string(reinterpret_cast<const char*>(ptr), name_len);
            }

            // Load metrics
            archive.read(prefix + "n_metrics", t);
            int64_t n_metrics = t.item<int64_t>();

            for (int64_t j = 0; j < n_metrics; ++j) {
                std::string m_prefix = prefix + "metric_" + std::to_string(j) + "_";

                archive.read(m_prefix + "name_len", t);
                int64_t m_name_len = t.item<int64_t>();
                std::string metric_name;
                if (m_name_len > 0) {
                    archive.read(m_prefix + "name", t);
                    auto ptr = t.data_ptr<uint8_t>();
                    metric_name = std::string(reinterpret_cast<const char*>(ptr), m_name_len);
                }

                archive.read(m_prefix + "value", t);
                metadata.final_metrics[target_name][metric_name] = t.item<float>();
            }
        }
    } catch (...) {
        // Metadata may not be present in older checkpoints
    }

    return metadata;
}

void write_metadata_json(
    const std::string& checkpoint_path,
    const ModelConfig& model_config,
    const TrainConfig& train_config,
    const RunMetadata& metadata,
    const ResolveSchema& schema
) {
    // Replace .pt extension with .json
    std::string json_path = checkpoint_path;
    if (json_path.size() >= 3 && json_path.substr(json_path.size() - 3) == ".pt") {
        json_path = json_path.substr(0, json_path.size() - 3) + ".json";
    } else {
        json_path += ".json";
    }

    std::ofstream file(json_path);
    if (!file.is_open()) return;

    file << "{\n";

    // Run metadata
    file << "  \"resolve_version\": \"" << metadata.resolve_version << "\",\n";
    file << "  \"created_at\": \"" << metadata.created_at << "\",\n";
    file << "  \"completed_at\": \"" << metadata.completed_at << "\",\n";
    file << "  \"train_time_seconds\": " << metadata.train_time_seconds << ",\n";
    file << "  \"n_plots_train\": " << metadata.n_plots_train << ",\n";
    file << "  \"n_plots_test\": " << metadata.n_plots_test << ",\n";
    file << "  \"best_epoch\": " << metadata.best_epoch << ",\n";
    file << "  \"total_epochs\": " << metadata.total_epochs << ",\n";

    // Model config
    file << "  \"model_config\": {\n";
    file << "    \"species_encoding\": " << static_cast<int>(model_config.species_encoding) << ",\n";
    file << "    \"hash_dim\": " << model_config.hash_dim << ",\n";
    file << "    \"species_embed_dim\": " << model_config.species_embed_dim << ",\n";
    file << "    \"genus_emb_dim\": " << model_config.genus_emb_dim << ",\n";
    file << "    \"family_emb_dim\": " << model_config.family_emb_dim << ",\n";
    file << "    \"top_k\": " << model_config.top_k << ",\n";
    file << "    \"dropout\": " << model_config.dropout << ",\n";
    file << "    \"hidden_dims\": [";
    for (size_t i = 0; i < model_config.hidden_dims.size(); ++i) {
        file << model_config.hidden_dims[i];
        if (i < model_config.hidden_dims.size() - 1) file << ", ";
    }
    file << "]\n";
    file << "  },\n";

    // Train config
    file << "  \"train_config\": {\n";
    file << "    \"batch_size\": " << train_config.batch_size << ",\n";
    file << "    \"max_epochs\": " << train_config.max_epochs << ",\n";
    file << "    \"patience\": " << train_config.patience << ",\n";
    file << "    \"lr\": " << train_config.lr << ",\n";
    file << "    \"weight_decay\": " << train_config.weight_decay << ",\n";
    file << "    \"loss_config\": " << static_cast<int>(train_config.loss_config) << ",\n";
    file << "    \"lr_scheduler\": " << static_cast<int>(train_config.lr_scheduler) << "\n";
    file << "  },\n";

    // Schema
    file << "  \"schema\": {\n";
    file << "    \"n_plots\": " << schema.n_plots << ",\n";
    file << "    \"n_species\": " << schema.n_species << ",\n";
    file << "    \"has_coordinates\": " << (schema.has_coordinates ? "true" : "false") << ",\n";
    file << "    \"has_taxonomy\": " << (schema.has_taxonomy ? "true" : "false") << ",\n";
    file << "    \"n_genera\": " << schema.n_genera << ",\n";
    file << "    \"n_families\": " << schema.n_families << ",\n";
    file << "    \"n_covariates\": " << schema.covariate_names.size() << ",\n";
    file << "    \"n_targets\": " << schema.targets.size() << "\n";
    file << "  },\n";

    // Final metrics
    file << "  \"final_metrics\": {\n";
    bool first_target = true;
    for (const auto& [target_name, metrics] : metadata.final_metrics) {
        if (!first_target) file << ",\n";
        first_target = false;
        file << "    \"" << target_name << "\": {\n";
        bool first_metric = true;
        for (const auto& [metric_name, value] : metrics) {
            if (!first_metric) file << ",\n";
            first_metric = false;
            file << "      \"" << metric_name << "\": " << value;
        }
        file << "\n    }";
    }
    file << "\n  }\n";

    file << "}\n";
}

} // namespace resolve
