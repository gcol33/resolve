#include "resolve/checkpoint.hpp"
#include "resolve/checkpoint_schema_keys.hpp"
#include "resolve/config_json.hpp"
#include "resolve/config_registry.hpp"
#include "resolve/io_retry.hpp"
#include "resolve/string_list_archive.hpp"
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace resolve {

// ============================================================================
// Checkpoint field marshaling
// ============================================================================
//
// ModelConfig, TrainConfig and the schema's DatasetConfig cross the archive
// through the field registry in resolve/config_registry.hpp: one visitor writes
// a row, its inverse reads it, and both walk the same list, so a field can be
// neither written under one spelling and looked up under another nor written
// and never read.

namespace {

// Writes one registry row into the archive.
struct ArchiveFieldWriter {
    torch::serialize::OutputArchive& archive;
    std::string prefix;  // non-empty only inside a numbered block (parallel branches)

    template <typename T>
    void operator()(const char*, const char* key, const T& value) const {
        if constexpr (std::is_same_v<T, LogCallback> || std::is_same_v<T, torch::Device>) {
            // The log callback has no serialized form, and the device describes
            // the machine a run happened on rather than the recipe. Both rows
            // carry no key.
            (void)key;
        } else if constexpr (resolve::is_registered_config_v<T>) {
            // A nested config's rows carry their own fully-qualified keys.
            for_each_field(value, ArchiveFieldWriter{archive, prefix});
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            if (!has_checkpoint_key(key)) return;
            archive.write(prefix + key, torch::tensor(static_cast<int64_t>(value.size())));
            for (std::size_t i = 0; i < value.size(); ++i) {
                for_each_field(value[i],
                               ArchiveFieldWriter{archive, parallel_branch_prefix(i)});
            }
        } else {
            if (!has_checkpoint_key(key)) return;
            const std::string k = prefix + key;
            if constexpr (std::is_same_v<T, bool>) {
                archive.write(k, torch::tensor(static_cast<int>(value)));
            } else if constexpr (std::is_enum_v<T>) {
                archive.write(k, torch::tensor(static_cast<int>(value)));
            } else if constexpr (std::is_same_v<T, int>) {
                archive.write(k, torch::tensor(value));
            } else if constexpr (std::is_same_v<T, float>) {
                archive.write(k, torch::tensor(value));
            } else if constexpr (std::is_same_v<T, std::string>) {
                write_string(archive, k, value);
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                std::vector<int64_t> tmp(value);
                archive.write(k, torch::tensor(tmp));
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                std::vector<float> tmp(value);
                archive.write(k, torch::tensor(tmp));
            } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
                archive.write(k + "1", torch::tensor(value.first));
                archive.write(k + "2", torch::tensor(value.second));
            } else {
                static_assert(registry_detail::always_false<T>,
                              "config field type has no checkpoint representation; "
                              "add a branch here and to ArchiveFieldReader");
            }
        }
    }
};

// Inverse of ArchiveFieldWriter. A key the archive does not carry leaves the
// field as the caller initialized it, which is how a checkpoint written before
// a field existed keeps loading; `absent`, when given, collects the names of
// those fields.
//
// Every read uses a FRESH tensor: InputArchive::read copies into the tensor it is
// handed, so reusing one across reads of different dtype or size trips a
// setStorage size-mismatch.
struct ArchiveFieldReader {
    torch::serialize::InputArchive& archive;
    std::string prefix;
    std::vector<std::string>* absent = nullptr;

    template <typename T>
    void operator()(const char* name, const char* key, T& value) const {
        if constexpr (std::is_same_v<T, LogCallback> || std::is_same_v<T, torch::Device>) {
            (void)name;
            (void)key;
            (void)value;
        } else if constexpr (resolve::is_registered_config_v<T>) {
            for_each_field(value, ArchiveFieldReader{archive, prefix, nullptr});
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            if (!has_checkpoint_key(key)) return;
            torch::Tensor n_t;
            if (!archive.try_read(prefix + key, n_t)) return note_absent(name);
            const int64_t n = n_t.item<int64_t>();
            value.clear();
            for (int64_t i = 0; i < n; ++i) {
                ParallelBranchConfig branch;
                for_each_field(branch, ArchiveFieldReader{
                    archive, parallel_branch_prefix(static_cast<std::size_t>(i)), nullptr});
                value.push_back(std::move(branch));
            }
        } else {
            if (!has_checkpoint_key(key)) return;
            if (!read_row(prefix + key, value)) note_absent(name);
        }
    }

private:
    void note_absent(const char* name) const {
        if (absent != nullptr) absent->emplace_back(name);
    }

    template <typename T>
    bool read_row(const std::string& k, T& value) const {
        if constexpr (std::is_same_v<T, std::pair<int, int>>) {
            torch::Tensor first_t, second_t;
            if (!archive.try_read(k + "1", first_t) || !archive.try_read(k + "2", second_t)) {
                return false;
            }
            value = {first_t.item<int>(), second_t.item<int>()};
            return true;
        } else if constexpr (std::is_same_v<T, std::string>) {
            return try_read_string(archive, k, value);
        } else if constexpr (std::is_enum_v<T>) {
            // An enum field stored in the string layout (k + "_len" beside the
            // bytes) is read by its name.
            std::string enum_name;
            if (try_read_string(archive, k, enum_name)) {
                value = enum_from_name<T>(enum_name);
                return true;
            }
            torch::Tensor t;
            if (!archive.try_read(k, t)) return false;
            value = static_cast<T>(t.item<int>());
            return true;
        } else {
            torch::Tensor t;
            if (!archive.try_read(k, t)) return false;
            if constexpr (std::is_same_v<T, bool>) {
                value = t.item<int>() != 0;
            } else if constexpr (std::is_same_v<T, int>) {
                value = t.item<int>();
            } else if constexpr (std::is_same_v<T, float>) {
                value = t.item<float>();
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                const auto values = t.to(torch::kCPU).to(torch::kLong).contiguous();
                const int64_t* data = values.numel() > 0 ? values.data_ptr<int64_t>() : nullptr;
                value.assign(data, data + values.numel());
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                const auto values = t.to(torch::kCPU).to(torch::kFloat32).contiguous();
                const float* data = values.numel() > 0 ? values.data_ptr<float>() : nullptr;
                value.assign(data, data + values.numel());
            } else {
                static_assert(registry_detail::always_false<T>,
                              "config field type has no checkpoint representation; "
                              "add a branch here and to ArchiveFieldWriter");
            }
            return true;
        }
    }
};

// A per-target metric tree, targets and metrics in name order.
json::Value metrics_json(
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& metrics) {
    const std::map<std::string, std::unordered_map<std::string, float>> by_target(
        metrics.begin(), metrics.end());
    json::Value out = json::Value::object();
    for (const auto& [target, values] : by_target) {
        const std::map<std::string, float> by_metric(values.begin(), values.end());
        json::Value metric_object = json::Value::object();
        for (const auto& [metric, value] : by_metric) metric_object.set(metric, value);
        out.set(target, std::move(metric_object));
    }
    return out;
}

void write_json_text(const std::string& path, const json::Value& value) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) return;
    file << json::dump(value) << "\n";
}

}  // namespace

void write_progress_file(
    const std::string& checkpoint_dir,
    int epoch,
    int max_epochs,
    int best_epoch,
    float best_loss,
    int epochs_without_improvement,
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& metrics
) {
    std::filesystem::create_directories(checkpoint_dir);

    // The best metric is the band accuracy of the first target (by name) that
    // reports one, its narrowest band by name. Band names order numerically
    // only for equal-width thresholds (band_25/50/75); a band_100 sorts before
    // band_25.
    float best_metric = 0.0f;
    std::string best_metric_name;
    const std::map<std::string, std::unordered_map<std::string, float>> by_target(
        metrics.begin(), metrics.end());
    for (const auto& [target, values] : by_target) {
        std::vector<std::string> bands;
        for (const auto& [metric, unused] : values) {
            if (metric.rfind("band_", 0) == 0) bands.push_back(metric);
        }
        if (!bands.empty()) {
            const std::string band = *std::min_element(bands.begin(), bands.end());
            best_metric = values.at(band);
            best_metric_name = target + "/" + band;
            break;
        }
    }

    json::Value progress = json::Value::object();
    progress.set("epoch", epoch);
    progress.set("max_epochs", max_epochs);
    progress.set("best_epoch", best_epoch);
    // Infinite until an epoch has been scored (a fixed-duration run without a
    // held-out fold never scores one), which JSON writes as null.
    progress.set("best_loss", best_loss);
    progress.set("epochs_without_improvement", epochs_without_improvement);
    progress.set("progress_pct",
                 max_epochs > 0 ? 100.0f * static_cast<float>(epoch) / static_cast<float>(max_epochs)
                                : 0.0f);
    progress.set("best_metric", best_metric);
    progress.set("best_metric_name", best_metric_name);
    write_json_text(checkpoint_dir + "/progress.json", progress);
}

void save_model_config(
    torch::serialize::OutputArchive& archive,
    const ModelConfig& config
) {
    for_each_field(config, ArchiveFieldWriter{archive, std::string()});
}

ModelConfig load_model_config(
    torch::serialize::InputArchive& archive
) {
    // The long-standing keys are demanded rather than defaulted: their absence
    // means a truncated or foreign archive, and substituting a default would
    // build a differently-shaped model whose weights then fail to load with a
    // shape error far from the cause.
    for (const char* key : kRequiredModelConfigKeys) {
        torch::Tensor probe;
        if (!archive.try_read(key, probe)) {
            throw std::runtime_error(
                std::string("checkpoint is missing the required model-config key '") +
                key + "'");
        }
    }

    ModelConfig config;
    for_each_field(config, ArchiveFieldReader{archive, std::string()});
    return config;
}

void save_scalers(
    torch::serialize::OutputArchive& archive,
    const Scalers& scalers
) {
    // An explicit presence flag, so load_scalers knows whether a continuous
    // block was scaled instead of inferring it from a failed read.
    int has_continuous = scalers.continuous_mean.defined() ? 1 : 0;
    archive.write("scalers_has_continuous", torch::tensor(has_continuous));
    if (has_continuous) {
        archive.write("continuous_mean", scalers.continuous_mean);
        archive.write("continuous_scale", scalers.continuous_scale);
        if (scalers.continuous_fill.defined()) {
            archive.write("continuous_fill", scalers.continuous_fill);
        }
    }

    // Each target scaler with its target's name, so the loaded Predictor
    // returns predictions in the target's own units.
    archive.write("n_target_scalers", torch::tensor(static_cast<int64_t>(scalers.target_scalers.size())));
    int idx = 0;
    for (const auto& [name, scaler] : scalers.target_scalers) {
        std::string prefix = "target_scaler_" + std::to_string(idx) + "_";
        write_string(archive, prefix + "name", name);
        archive.write(prefix + "mean", scaler.first);
        archive.write(prefix + "scale", scaler.second);
        idx++;
    }
}

Scalers load_scalers(
    torch::serialize::InputArchive& archive
) {
    Scalers scalers;

    // A checkpoint without the presence flag predates it and always scaled a
    // continuous block when it had one: read whatever is there.
    torch::Tensor has_t;
    bool has_continuous = true;
    if (archive.try_read("scalers_has_continuous", has_t)) {
        has_continuous = (has_t.item<int>() != 0);
    }
    if (has_continuous) {
        archive.try_read("continuous_mean", scalers.continuous_mean);
        archive.try_read("continuous_scale", scalers.continuous_scale);
        // Absent on a checkpoint written before missing values were filled.
        torch::Tensor fill;
        if (archive.try_read("continuous_fill", fill)) {
            scalers.continuous_fill = fill;
        }
    }

    torch::Tensor n_target_scalers_t;
    if (archive.try_read("n_target_scalers", n_target_scalers_t)) {
        int64_t n_scalers = n_target_scalers_t.item<int64_t>();
        for (int64_t i = 0; i < n_scalers; ++i) {
            const std::string prefix = "target_scaler_" + std::to_string(i) + "_";
            std::string name;
            try_read_string(archive, prefix + "name", name);
            torch::Tensor mean, scale;
            if (!name.empty() && archive.try_read(prefix + "mean", mean) &&
                archive.try_read(prefix + "scale", scale)) {
                scalers.target_scalers[name] = {mean, scale};
            }
        }
    }

    return scalers;
}

void save_schema(
    torch::serialize::OutputArchive& archive,
    const ResolveSchema& schema
) {
    namespace k = ckpt_schema_keys;
    archive.write(k::kNPlots, torch::tensor(schema.n_plots));
    archive.write(k::kNSpecies, torch::tensor(schema.n_species));
    archive.write(k::kNSpeciesVocab, torch::tensor(schema.n_species_vocab));
    archive.write(k::kHasCoordinates, torch::tensor(static_cast<int>(schema.has_coordinates)));
    archive.write(k::kHasAbundance, torch::tensor(static_cast<int>(schema.has_abundance)));
    archive.write(k::kHasTaxonomy, torch::tensor(static_cast<int>(schema.has_taxonomy)));
    archive.write(k::kNGenera, torch::tensor(schema.n_genera));
    archive.write(k::kNFamilies, torch::tensor(schema.n_families));
    archive.write(k::kNCovariates, torch::tensor(static_cast<int64_t>(schema.covariate_names.size())));
    for (size_t i = 0; i < schema.covariate_names.size(); ++i) {
        write_string(archive, k::covariate(static_cast<int64_t>(i)), schema.covariate_names[i]);
    }
    archive.write(k::kNTargets, torch::tensor(static_cast<int64_t>(schema.targets.size())));
    for (size_t i = 0; i < schema.targets.size(); ++i) {
        const auto& target = schema.targets[i];
        std::string prefix = k::target_prefix(static_cast<int64_t>(i));
        write_string(archive, prefix + k::kTargetName, target.name);
        archive.write(prefix + k::kTargetTask, torch::tensor(static_cast<int>(target.task)));
        archive.write(prefix + k::kTargetTransform, torch::tensor(static_cast<int>(target.transform)));
        archive.write(prefix + k::kTargetNumClasses, torch::tensor(target.num_classes));
        archive.write(prefix + k::kTargetWeight, torch::tensor(target.weight));

        // The ordered class vocabulary of a classification target: empty for
        // regression and for a column the loader did not factorize.
        archive.write(prefix + k::kTargetNClassNames,
                      torch::tensor(static_cast<int64_t>(target.class_names.size())));
        for (size_t j = 0; j < target.class_names.size(); ++j) {
            write_string(archive, prefix + k::kTargetClassPrefix + std::to_string(j),
                         target.class_names[j]);
        }

        // Per-class loss weights for an imbalanced classification target;
        // empty is unweighted cross-entropy.
        archive.write(prefix + k::kTargetNClassWeights,
                      torch::tensor(static_cast<int64_t>(target.class_weights.size())));
        if (!target.class_weights.empty()) {
            archive.write(prefix + k::kTargetClassWeights,
                          torch::tensor(target.class_weights));
        }

        archive.write(prefix + k::kTargetMissing,
                      torch::tensor(static_cast<int>(target.missing)));
    }

    // Categorical covariates: column count, per-column name and vocab size
    // (the reserved UNK slot at code 0 included, so the column's embedding
    // table has that many rows), and the shared embed_dim.
    archive.write(k::kNCategoricals,
                  torch::tensor(static_cast<int64_t>(schema.categorical_names.size())));
    archive.write(k::kCategoricalEmbedDim,
                  torch::tensor(schema.categorical_embed_dim));
    for (size_t i = 0; i < schema.categorical_names.size(); ++i) {
        const std::string prefix = k::categorical_prefix(static_cast<int64_t>(i));
        write_string(archive, prefix + k::kCategoricalName, schema.categorical_names[i]);
        archive.write(prefix + k::kCategoricalVocabSize,
                      torch::tensor(schema.categorical_vocab_sizes[i]));
    }

    // The loader settings, under the keys their registry rows carry.
    for_each_field(schema.dataset_config, ArchiveFieldWriter{archive, std::string()});

    // The fitted species / genus / family vocabularies, index = integer code.
    // They make a checkpoint self-sufficient for inference: a new file
    // otherwise fits its own codes, and every non-hash encoder then indexes its
    // embedding tables with the wrong rows.
    write_string_list(archive, k::kSpeciesVocab, schema.species_vocab);
    write_string_list(archive, k::kGenusVocab, schema.genus_vocab);
    write_string_list(archive, k::kFamilyVocab, schema.family_vocab);
}

ResolveSchema load_schema(
    torch::serialize::InputArchive& archive,
    const ModelConfig* model_config
) {
    for (const char* key : kRequiredDatasetConfigKeys) {
        torch::Tensor probe;
        if (!archive.try_read(key, probe)) {
            throw std::runtime_error(
                std::string("checkpoint is missing the required schema key '") + key + "'");
        }
    }

    // Every read uses a fresh tensor: InputArchive::read copies into the
    // tensor it is handed, and reading int64 / int32 / float32 entries through
    // one tensor trips a storage-size mismatch.
    auto read_i64 = [&](const std::string& key) {
        torch::Tensor t;
        archive.read(key, t);
        return t.item<int64_t>();
    };
    auto read_i32 = [&](const std::string& key) {
        torch::Tensor t;
        archive.read(key, t);
        return t.item<int>();
    };
    auto read_f32 = [&](const std::string& key) {
        torch::Tensor t;
        archive.read(key, t);
        return t.item<float>();
    };
    auto read_count = [&](const std::string& key) {
        const int64_t n = read_i64(key);
        if (n < 0) {
            throw std::runtime_error("checkpoint entry '" + key + "' is corrupt: negative count");
        }
        return n;
    };

    namespace k = ckpt_schema_keys;
    ResolveSchema schema;
    schema.n_plots = read_i64(k::kNPlots);
    schema.n_species = read_i64(k::kNSpecies);
    schema.n_species_vocab = read_i64(k::kNSpeciesVocab);
    schema.has_coordinates = read_i32(k::kHasCoordinates) != 0;
    schema.has_abundance = read_i32(k::kHasAbundance) != 0;
    schema.has_taxonomy = read_i32(k::kHasTaxonomy) != 0;
    schema.n_genera = read_i64(k::kNGenera);
    schema.n_families = read_i64(k::kNFamilies);
    const int64_t n_covariates = read_count(k::kNCovariates);
    schema.covariate_names.resize(static_cast<size_t>(n_covariates));
    for (int64_t i = 0; i < n_covariates; ++i) {
        // A checkpoint written before covariate names were saved has none; the
        // model indexes covariates by position, so an empty name is safe.
        try_read_string(archive, k::covariate(i), schema.covariate_names[static_cast<size_t>(i)]);
    }
    const int64_t n_targets = read_count(k::kNTargets);
    schema.targets.resize(static_cast<size_t>(n_targets));
    for (int64_t i = 0; i < n_targets; ++i) {
        TargetConfig& target = schema.targets[static_cast<size_t>(i)];
        const std::string prefix = k::target_prefix(i);
        // A checkpoint written before target names were saved has none; the
        // heads are registered by name, so each gets a distinct one.
        try_read_string(archive, prefix + k::kTargetName, target.name);
        if (target.name.empty()) target.name = "target_" + std::to_string(i);
        target.task = static_cast<TaskType>(read_i32(prefix + k::kTargetTask));
        target.transform = static_cast<TransformType>(read_i32(prefix + k::kTargetTransform));
        target.num_classes = read_i32(prefix + k::kTargetNumClasses);
        target.weight = read_f32(prefix + k::kTargetWeight);

        // Absent on a checkpoint written before classification targets carried
        // a class vocabulary, which predicted raw integer codes.
        torch::Tensor n_cn_t;
        if (archive.try_read(prefix + k::kTargetNClassNames, n_cn_t)) {
            const int64_t n_cn = n_cn_t.item<int64_t>();
            if (n_cn < 0) {
                throw std::runtime_error("checkpoint target '" + target.name +
                                         "' is corrupt: negative class count");
            }
            target.class_names.resize(static_cast<size_t>(n_cn));
            for (int64_t j = 0; j < n_cn; ++j) {
                target.class_names[static_cast<size_t>(j)] =
                    read_string(archive, prefix + k::kTargetClassPrefix + std::to_string(j));
            }
        }

        // Absent: unweighted cross-entropy.
        torch::Tensor n_cw_t;
        if (archive.try_read(prefix + k::kTargetNClassWeights, n_cw_t)) {
            const int64_t n_cw = n_cw_t.item<int64_t>();
            if (n_cw > 0) {
                torch::Tensor cw_t;
                archive.read(prefix + k::kTargetClassWeights, cw_t);
                cw_t = cw_t.to(torch::kCPU).to(torch::kFloat32).contiguous();
                if (cw_t.numel() != n_cw) {
                    throw std::runtime_error(
                        "checkpoint target '" + target.name + "' is corrupt: " +
                        std::to_string(n_cw) + " class weights recorded, " +
                        std::to_string(cw_t.numel()) + " stored");
                }
                const float* weights = cw_t.data_ptr<float>();
                target.class_weights.assign(weights, weights + n_cw);
            }
        }

        // Absent: Drop, what a loader did before the policy existed.
        torch::Tensor missing_t;
        if (archive.try_read(prefix + k::kTargetMissing, missing_t)) {
            target.missing = static_cast<TargetMissingPolicy>(missing_t.item<int>());
        }
    }

    // Absent on a checkpoint written before categorical covariates existed:
    // a schema with none.
    torch::Tensor n_cat_t;
    if (archive.try_read(k::kNCategoricals, n_cat_t)) {
        const int64_t n_cat = n_cat_t.item<int64_t>();
        if (n_cat < 0) {
            throw std::runtime_error("checkpoint categorical count is corrupt: negative");
        }
        torch::Tensor embed_dim_t;
        if (archive.try_read(k::kCategoricalEmbedDim, embed_dim_t)) {
            schema.categorical_embed_dim = embed_dim_t.item<int64_t>();
        }
        schema.categorical_names.resize(static_cast<size_t>(n_cat));
        schema.categorical_vocab_sizes.resize(static_cast<size_t>(n_cat));
        for (int64_t i = 0; i < n_cat; ++i) {
            const std::string prefix = k::categorical_prefix(i);
            schema.categorical_names[static_cast<size_t>(i)] =
                read_string(archive, prefix + k::kCategoricalName);
            schema.categorical_vocab_sizes[static_cast<size_t>(i)] =
                read_i64(prefix + k::kCategoricalVocabSize);
        }
    }

    // The loader settings. A setting the checkpoint predates keeps
    // unrecorded_dataset_config()'s value and is listed.
    for_each_field(schema.dataset_config,
                   ArchiveFieldReader{archive, std::string(), &schema.unrecorded_dataset_settings});

    // The settings that also size the model are authoritative in the
    // ModelConfig, which every checkpoint carries.
    if (model_config != nullptr) {
        schema.dataset_config.species_encoding = model_config->species_encoding;
        schema.dataset_config.hash_dim = model_config->hash_dim;
        schema.dataset_config.top_k = model_config->top_k;
        auto& unrecorded = schema.unrecorded_dataset_settings;
        unrecorded.erase(std::remove_if(unrecorded.begin(), unrecorded.end(),
                                        [](const std::string& name) {
                                            return name == "species_encoding" ||
                                                   name == "hash_dim" || name == "top_k";
                                        }),
                         unrecorded.end());
    }

    // Absent on a checkpoint that stores only the vocabulary sizes: the
    // vectors stay empty, has_species_vocab()/has_taxonomy_vocab() report
    // false, and the Predictor checks sizes only, with a warning.
    try_read_string_list(archive, k::kSpeciesVocab, schema.species_vocab);
    try_read_string_list(archive, k::kGenusVocab, schema.genus_vocab);
    try_read_string_list(archive, k::kFamilyVocab, schema.family_vocab);
    return schema;
}

void save_train_config(
    torch::serialize::OutputArchive& archive,
    const TrainConfig& config,
    int requested_batch_size
) {
    // `config.batch_size` is the EFFECTIVE batch size that trained the model
    // (the CUDA out-of-memory retry halves it). `train_batch_size` records the
    // value the caller REQUESTED (restored by load_train_config) and
    // `train_effective_batch_size` the effective one, so a fallback run is
    // detectable when the two differ. A negative requested value means no
    // separate request is known. The registry pass runs on a copy carrying the
    // requested value, so train_batch_size is written once.
    TrainConfig persisted = config;
    persisted.batch_size = requested_batch_size >= 0 ? requested_batch_size : config.batch_size;
    for_each_field(persisted, ArchiveFieldWriter{archive, std::string()});

    // Derived, so not a registry row: the effective size is a property of the
    // run, not of the requested recipe.
    archive.write(kEffectiveBatchSizeKey, torch::tensor(config.batch_size));
}

void save_run_metadata(
    torch::serialize::OutputArchive& archive,
    const RunMetadata& metadata
) {
    write_string(archive, "meta_version", metadata.resolve_version);
    write_string(archive, "meta_created", metadata.created_at);
    write_string(archive, "meta_completed", metadata.completed_at);

    archive.write("meta_train_time", torch::tensor(metadata.train_time_seconds));
    archive.write("meta_n_plots_train", torch::tensor(metadata.n_plots_train));
    archive.write("meta_n_plots_test", torch::tensor(metadata.n_plots_test));

    // The hold-out mode is stored by name, so the key reads the same however
    // the enum is ordered.
    write_string(archive, "meta_hold_out", enum_to_name(metadata.hold_out));
    archive.write("meta_test_size", torch::tensor(metadata.test_size));
    archive.write("meta_split_seed", torch::tensor(static_cast<int64_t>(metadata.split_seed)));
    archive.write("meta_best_epoch", torch::tensor(metadata.best_epoch));
    archive.write("meta_total_epochs", torch::tensor(metadata.total_epochs));

    // The final metrics, flattened per target and metric.
    int64_t n_targets = static_cast<int64_t>(metadata.final_metrics.size());
    archive.write("meta_n_targets", torch::tensor(n_targets));

    int target_idx = 0;
    for (const auto& [target_name, metrics] : metadata.final_metrics) {
        std::string prefix = "meta_target_" + std::to_string(target_idx) + "_";
        write_string(archive, prefix + "name", target_name);
        archive.write(prefix + "n_metrics", torch::tensor(static_cast<int64_t>(metrics.size())));

        int metric_idx = 0;
        for (const auto& [metric_name, value] : metrics) {
            std::string m_prefix = prefix + "metric_" + std::to_string(metric_idx) + "_";
            write_string(archive, m_prefix + "name", metric_name);
            archive.write(m_prefix + "value", torch::tensor(value));
            metric_idx++;
        }
        target_idx++;
    }
}

TrainConfig load_train_config(
    torch::serialize::InputArchive& archive
) {
    // The fields save_train_config does not write (device, checkpoint
    // destination, AMP and cuDNN switches, log callback) and every key a
    // checkpoint predates keep their TrainConfig defaults.
    TrainConfig config;
    for_each_field(config, ArchiveFieldReader{archive, std::string()});
    return config;
}

int load_effective_batch_size(
    torch::serialize::InputArchive& archive
) {
    torch::Tensor t;
    if (archive.try_read(kEffectiveBatchSizeKey, t)) return t.item<int>();
    return load_train_config(archive).batch_size;
}

RunMetadata load_run_metadata(
    torch::serialize::InputArchive& archive
) {
    // A key the checkpoint does not carry keeps the RunMetadata default.
    RunMetadata meta;

    try_read_string(archive, "meta_version", meta.resolve_version);
    try_read_string(archive, "meta_created", meta.created_at);
    try_read_string(archive, "meta_completed", meta.completed_at);

    { torch::Tensor x; if (archive.try_read("meta_train_time", x)) meta.train_time_seconds = x.item<float>(); }
    { torch::Tensor x; if (archive.try_read("meta_n_plots_train", x)) meta.n_plots_train = x.item<int64_t>(); }
    { torch::Tensor x; if (archive.try_read("meta_n_plots_test", x)) meta.n_plots_test = x.item<int64_t>(); }
    {
        std::string hold_out;
        if (try_read_string(archive, "meta_hold_out", hold_out) && !hold_out.empty()) {
            meta.hold_out = enum_from_name<HoldOutMode>(hold_out);
        }
    }
    { torch::Tensor x; if (archive.try_read("meta_test_size", x)) meta.test_size = x.item<float>(); }
    { torch::Tensor x; if (archive.try_read("meta_split_seed", x)) meta.split_seed = static_cast<int>(x.item<int64_t>()); }
    { torch::Tensor x; if (archive.try_read("meta_best_epoch", x)) meta.best_epoch = x.item<int>(); }
    { torch::Tensor x; if (archive.try_read("meta_total_epochs", x)) meta.total_epochs = x.item<int>(); }

    torch::Tensor n_targets_t;
    if (archive.try_read("meta_n_targets", n_targets_t)) {
        int64_t n_targets = n_targets_t.item<int64_t>();
        for (int64_t ti = 0; ti < n_targets; ++ti) {
            std::string prefix = "meta_target_" + std::to_string(ti) + "_";
            std::string target_name;
            try_read_string(archive, prefix + "name", target_name);
            torch::Tensor n_metrics_t;
            if (!archive.try_read(prefix + "n_metrics", n_metrics_t)) continue;
            int64_t n_metrics = n_metrics_t.item<int64_t>();
            std::unordered_map<std::string, float> metrics;
            for (int64_t mi = 0; mi < n_metrics; ++mi) {
                std::string m_prefix = prefix + "metric_" + std::to_string(mi) + "_";
                std::string metric_name;
                try_read_string(archive, m_prefix + "name", metric_name);
                torch::Tensor val_t;
                if (archive.try_read(m_prefix + "value", val_t))
                    metrics[metric_name] = val_t.item<float>();
            }
            meta.final_metrics[target_name] = std::move(metrics);
        }
    }
    return meta;
}

// ============================================================================
// Description
// ============================================================================

namespace {

json::Value schema_description(const ResolveSchema& schema) {
    json::Value out = json::Value::object();
    out.set("n_plots", schema.n_plots);
    out.set("n_species", schema.n_species);
    out.set("n_species_vocab", schema.n_species_vocab);
    out.set("has_coordinates", schema.has_coordinates);
    out.set("has_abundance", schema.has_abundance);
    out.set("has_taxonomy", schema.has_taxonomy);
    out.set("n_genera", schema.n_genera);
    out.set("n_families", schema.n_families);
    out.set("covariates", json::Value::array_of(schema.covariate_names));
    json::Value categoricals = json::Value::array();
    for (size_t i = 0; i < schema.categorical_names.size(); ++i) {
        json::Value column = json::Value::object();
        column.set("name", schema.categorical_names[i]);
        column.set("vocab_size", schema.categorical_vocab_sizes[i]);
        categoricals.push_back(std::move(column));
    }
    out.set("categoricals", std::move(categoricals));
    out.set("categorical_embed_dim", schema.categorical_embed_dim);
    json::Value targets = json::Value::array();
    for (const auto& target : schema.targets) {
        json::Value t = json::Value::object();
        t.set("name", target.name);
        t.set("task", enum_to_name(target.task));
        t.set("transform", enum_to_name(target.transform));
        t.set("num_classes", target.num_classes);
        t.set("weight", target.weight);
        t.set("missing", enum_to_name(target.missing));
        t.set("class_names", json::Value::array_of(target.class_names));
        targets.push_back(std::move(t));
    }
    out.set("targets", std::move(targets));
    out.set("dataset_config", config_to_json(schema.dataset_config));
    if (!schema.unrecorded_dataset_settings.empty()) {
        out.set("unrecorded_dataset_settings",
                json::Value::array_of(schema.unrecorded_dataset_settings));
    }
    return out;
}

}  // namespace

json::Value checkpoint_description(
    const ModelConfig& model_config,
    const TrainConfig& train_config,
    int effective_batch_size,
    const RunMetadata& metadata,
    const ResolveSchema& schema
) {
    json::Value out = json::Value::object();
    out.set("resolve_version", metadata.resolve_version);
    out.set("created_at", metadata.created_at);
    out.set("completed_at", metadata.completed_at);
    out.set("train_time_seconds", metadata.train_time_seconds);
    out.set("n_plots_train", metadata.n_plots_train);
    out.set("n_plots_test", metadata.n_plots_test);
    out.set("hold_out", enum_to_name(metadata.hold_out));
    out.set("test_size", metadata.test_size);
    out.set("split_seed", metadata.split_seed);
    out.set("best_epoch", metadata.best_epoch);
    out.set("total_epochs", metadata.total_epochs);
    out.set("schema", schema_description(schema));
    out.set("model_config", config_to_json(model_config));
    // `batch_size` is the value the caller requested; the CUDA out-of-memory
    // retry may have trained the model at a smaller `effective_batch_size`.
    json::Value train = config_to_json(train_config);
    train.set("effective_batch_size", effective_batch_size);
    out.set("train_config", std::move(train));
    out.set("final_metrics", metrics_json(metadata.final_metrics));
    return out;
}

void read_checkpoint_archive(torch::serialize::InputArchive& archive, const std::string& path,
                             const torch::Device& device, const char* what) {
    io::with_retry(
        [&] {
            std::error_code ec;
            if (!std::filesystem::is_regular_file(path, ec)) {
                throw io::IOError("cannot read checkpoint '" + path + "': no such file");
            }
            archive.load_from(path, device);
        },
        what);
}

void write_checkpoint_archive(torch::serialize::OutputArchive& archive, const std::string& path) {
    io::with_retry(
        [&] {
            try {
                archive.save_to(path);
            } catch (const c10::Error& e) {
                throw io::IOError("cannot write checkpoint '" + path + "': " + e.what_without_backtrace());
            }
        },
        "checkpoint save");
}

json::Value describe_checkpoint(const std::string& path) {
    torch::serialize::InputArchive archive;
    // Read onto the CPU: deserialization restores every tensor, weights
    // included, so a GPU-saved checkpoint would otherwise need a CUDA device.
    read_checkpoint_archive(archive, path, torch::kCPU, "checkpoint load");
    const ModelConfig model_config = load_model_config(archive);
    return checkpoint_description(model_config, load_train_config(archive),
                                  load_effective_batch_size(archive), load_run_metadata(archive),
                                  load_schema(archive, &model_config));
}

void write_metadata_json(
    const std::string& checkpoint_path,
    const ModelConfig& model_config,
    const TrainConfig& train_config,
    const RunMetadata& metadata,
    const ResolveSchema& schema,
    int requested_batch_size
) {
    std::string json_path = checkpoint_path;
    if (json_path.size() >= 3 && json_path.substr(json_path.size() - 3) == ".pt") {
        json_path = json_path.substr(0, json_path.size() - 3) + ".json";
    } else {
        json_path += ".json";
    }
    TrainConfig requested = train_config;
    if (requested_batch_size >= 0) requested.batch_size = requested_batch_size;
    write_json_text(json_path, checkpoint_description(model_config, requested,
                                                      train_config.batch_size, metadata, schema));
}

} // namespace resolve
