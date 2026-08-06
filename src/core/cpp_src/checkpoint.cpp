#include "resolve/checkpoint.hpp"
#include "resolve/checkpoint_schema_keys.hpp"
#include "resolve/config_registry.hpp"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <ostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <cstddef>
#include <cstdio>

namespace resolve {

namespace {
// Escape a string for embedding in a JSON string literal. Target names and
// class labels come from user CSV headers and may contain " or \; emitting them
// raw produces invalid JSON that a monitoring parser chokes on.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
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
    file << "  \"progress_pct\": "
         << (max_epochs > 0 ? 100.0f * epoch / max_epochs : 0.0f) << ",\n";

    // Write a deterministic "best metric": the lexicographically-first band
    // accuracy of the alphabetically-first target. `metrics` is an unordered_map,
    // so select via sorted keys rather than iteration order (which is unspecified
    // and made the reported value vary across runs/builds). Ordering is
    // lexicographic on the band name, which equals numeric order only for
    // equal-width thresholds (band_25/50/75); a band_100 would sort before band_25.
    float best_metric = 0.0f;
    std::string best_metric_name;
    {
        std::vector<std::string> target_names;
        target_names.reserve(metrics.size());
        for (const auto& [name, unused] : metrics) target_names.push_back(name);
        std::sort(target_names.begin(), target_names.end());
        for (const auto& tname : target_names) {
            const auto& tm = metrics.at(tname);
            std::vector<std::string> band_names;
            for (const auto& [mname, unused] : tm) {
                if (mname.rfind("band_", 0) == 0) band_names.push_back(mname);
            }
            if (!band_names.empty()) {
                std::sort(band_names.begin(), band_names.end());
                best_metric = tm.at(band_names.front());
                best_metric_name = tname + "/" + band_names.front();
                break;
            }
        }
    }
    file << "  \"best_metric\": " << best_metric << ",\n";
    file << "  \"best_metric_name\": \"" << json_escape(best_metric_name) << "\"\n";
    file << "}\n";
}

// ============================================================================
// Checkpoint field marshaling
// ============================================================================
//
// ModelConfig and TrainConfig cross the archive through the field registry in
// resolve/config_registry.hpp: one visitor writes a row, its inverse reads it,
// and both walk the same list. A field can therefore not be written under one
// spelling and looked up under another, nor be written and never read -- the
// two failure modes behind issues #37, #91 and #108. The archive key a row
// carries is the spelling earlier releases wrote; the registry reproduces it
// rather than renaming it, so existing checkpoints keep loading.

// Why: libtorch's archive API stores tensors, not strings. A std::string
// round-trips as a length-prefixed UInt8 tensor pair, "<key>_len" + "<key>"
// (the same approach save_run_metadata uses for resolve_version/timestamps).
static void write_string_to_archive(
    torch::serialize::OutputArchive& archive,
    const std::string& prefix,
    const std::string& value
) {
    archive.write(prefix + "_len", torch::tensor(static_cast<int64_t>(value.size())));
    if (!value.empty()) {
        std::vector<uint8_t> bytes(value.begin(), value.end());
        archive.write(prefix, torch::from_blob(
            bytes.data(), {static_cast<int64_t>(bytes.size())}, torch::kUInt8).clone());
    }
}

// Inverse of write_string_to_archive. Leaves `out` untouched (returning false)
// when the key is absent or the stored length is zero, so a checkpoint written
// before the field existed keeps the struct default.
static bool try_read_string_from_archive(
    torch::serialize::InputArchive& archive,
    const std::string& prefix,
    std::string& out
) {
    torch::Tensor len_t;
    if (!archive.try_read(prefix + "_len", len_t)) return false;
    const int64_t len = len_t.item<int64_t>();
    if (len <= 0) return false;
    torch::Tensor bytes_t;
    if (!archive.try_read(prefix, bytes_t)) return false;
    out.assign(reinterpret_cast<const char*>(bytes_t.data_ptr<uint8_t>()),
               static_cast<std::size_t>(len));
    return true;
}

namespace {

// Writes one registry row into the archive.
struct ArchiveFieldWriter {
    torch::serialize::OutputArchive& archive;
    std::string prefix;  // non-empty only inside a numbered block (parallel branches)

    template <typename T>
    void operator()(const char*, const char* key, const T& value) const {
        if constexpr (std::is_same_v<T, LogCallback> || std::is_same_v<T, torch::Device>) {
            // The log callback has no serialized form, and the device describes
            // the machine a run happened on rather than the recipe. Both are
            // deliberately absent from the archive.
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
                write_string_to_archive(archive, k, value);
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

// Inverse of ArchiveFieldWriter. An absent key leaves the struct default in
// place, which is how a checkpoint written before a field existed keeps loading.
//
// Every read uses a FRESH tensor: InputArchive::read copies into the tensor it is
// handed, so reusing one across reads of different dtype or size trips a
// setStorage size-mismatch.
struct ArchiveFieldReader {
    torch::serialize::InputArchive& archive;
    std::string prefix;

    template <typename T>
    void operator()(const char*, const char* key, T& value) const {
        if constexpr (std::is_same_v<T, LogCallback> || std::is_same_v<T, torch::Device>) {
            (void)key;
            (void)value;
        } else if constexpr (resolve::is_registered_config_v<T>) {
            for_each_field(value, ArchiveFieldReader{archive, prefix});
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            if (!has_checkpoint_key(key)) return;
            torch::Tensor n_t;
            if (!archive.try_read(prefix + key, n_t)) return;
            const int64_t n = n_t.item<int64_t>();
            value.clear();
            for (int64_t i = 0; i < n; ++i) {
                ParallelBranchConfig branch;
                for_each_field(branch, ArchiveFieldReader{
                    archive, parallel_branch_prefix(static_cast<std::size_t>(i))});
                value.push_back(std::move(branch));
            }
        } else {
            if (!has_checkpoint_key(key)) return;
            const std::string k = prefix + key;
            if constexpr (std::is_same_v<T, bool>) {
                torch::Tensor t;
                if (archive.try_read(k, t)) value = t.item<int>() != 0;
            } else if constexpr (std::is_enum_v<T>) {
                torch::Tensor t;
                if (archive.try_read(k, t)) value = static_cast<T>(t.item<int>());
            } else if constexpr (std::is_same_v<T, int>) {
                torch::Tensor t;
                if (archive.try_read(k, t)) value = t.item<int>();
            } else if constexpr (std::is_same_v<T, float>) {
                torch::Tensor t;
                if (archive.try_read(k, t)) value = t.item<float>();
            } else if constexpr (std::is_same_v<T, std::string>) {
                try_read_string_from_archive(archive, k, value);
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                torch::Tensor t;
                if (archive.try_read(k, t)) {
                    std::vector<int64_t> out(static_cast<std::size_t>(t.size(0)));
                    for (int64_t i = 0; i < t.size(0); ++i) {
                        out[static_cast<std::size_t>(i)] = t[i].item<int64_t>();
                    }
                    value = std::move(out);
                }
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                torch::Tensor t;
                if (archive.try_read(k, t)) {
                    std::vector<float> out(static_cast<std::size_t>(t.size(0)));
                    for (int64_t i = 0; i < t.size(0); ++i) {
                        out[static_cast<std::size_t>(i)] = t[i].item<float>();
                    }
                    value = std::move(out);
                }
            } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
                torch::Tensor first_t, second_t;
                const bool has_first = archive.try_read(k + "1", first_t);
                const bool has_second = archive.try_read(k + "2", second_t);
                if (has_first && has_second) {
                    value = {first_t.item<int>(), second_t.item<int>()};
                }
            } else {
                static_assert(registry_detail::always_false<T>,
                              "config field type has no checkpoint representation; "
                              "add a branch here and to ArchiveFieldWriter");
            }
        }
    }
};

// Renders one registry row as a JSON object member, for the human-readable
// sidecar written next to a checkpoint. Enums appear as their names (the same
// spellings the CLI accepts) rather than as raw ordinals.
struct JsonFieldWriter {
    std::ostream& out;
    int indent;
    bool first = true;

    void open_member(const char* name) {
        if (!first) out << ",\n";
        first = false;
        out << std::string(static_cast<std::size_t>(indent), ' ')
            << "\"" << json_escape(name) << "\": ";
    }

    template <typename Seq>
    void write_array(const Seq& values) {
        out << "[";
        bool first_item = true;
        for (const auto& v : values) {
            if (!first_item) out << ", ";
            first_item = false;
            out << v;
        }
        out << "]";
    }

    template <typename Cfg>
    void write_object(const Cfg& cfg) {
        out << "{\n";
        JsonFieldWriter nested{out, indent + 2};
        for_each_field(cfg, nested);
        out << "\n" << std::string(static_cast<std::size_t>(indent), ' ') << "}";
    }

    template <typename T>
    void operator()(const char* name, const char*, const T& value) {
        if constexpr (std::is_same_v<T, LogCallback>) {
            (void)name;  // a callback has no printable value
        } else if constexpr (std::is_same_v<T, torch::Device>) {
            open_member(name);
            out << "\"" << (value.is_cuda() ? "cuda" : "cpu") << "\"";
        } else if constexpr (resolve::is_registered_config_v<T>) {
            open_member(name);
            write_object(value);
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            open_member(name);
            out << "[";
            for (std::size_t i = 0; i < value.size(); ++i) {
                if (i > 0) out << ", ";
                write_object(value[i]);
            }
            out << "]";
        } else if constexpr (std::is_same_v<T, bool>) {
            open_member(name);
            out << (value ? "true" : "false");
        } else if constexpr (std::is_enum_v<T>) {
            open_member(name);
            out << "\"" << json_escape(enum_to_name(value)) << "\"";
        } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
            open_member(name);
            out << value;
        } else if constexpr (std::is_same_v<T, std::string>) {
            open_member(name);
            out << "\"" << json_escape(value) << "\"";
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                             std::is_same_v<T, std::vector<float>>) {
            open_member(name);
            write_array(value);
        } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
            open_member(name);
            out << "[" << value.first << ", " << value.second << "]";
        } else {
            static_assert(registry_detail::always_false<T>,
                          "config field type has no JSON representation; "
                          "add a branch to JsonFieldWriter");
        }
    }
};

}  // namespace

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

// Why: a vocabulary is a LIST of strings, and the per-string
// write_string_to_archive pattern above costs two archive entries per element
// -- a 30k-species vocab would become 60k zip members. Serialize the whole list
// as two tensors instead: an int64 lengths vector plus the concatenated UTF-8
// bytes. Same layout TaxonomyVocab::save / CategoricalVocab::save already use,
// so this is the proven round-trip, factored out once (issue #102).
static void write_string_list(
    torch::serialize::OutputArchive& archive,
    const std::string& prefix,
    const std::vector<std::string>& values
) {
    std::vector<int64_t> lengths;
    lengths.reserve(values.size());
    std::vector<uint8_t> bytes;
    for (const auto& s : values) {
        lengths.push_back(static_cast<int64_t>(s.size()));
        bytes.insert(bytes.end(), s.begin(), s.end());
    }
    archive.write(prefix + "_lengths", torch::tensor(lengths));
    if (!bytes.empty()) {
        archive.write(prefix + "_bytes", torch::from_blob(
            bytes.data(), {static_cast<int64_t>(bytes.size())}, torch::kUInt8).clone());
    } else {
        archive.write(prefix + "_bytes", torch::empty({0}, torch::kUInt8));
    }
}

// Inverse of write_string_list. Returns false (leaving `out` untouched) when the
// key is absent, which is how a pre-issue-#102 checkpoint is detected.
static bool try_read_string_list(
    torch::serialize::InputArchive& archive,
    const std::string& prefix,
    std::vector<std::string>& out
) {
    torch::Tensor lengths_t;
    if (!archive.try_read(prefix + "_lengths", lengths_t)) return false;
    torch::Tensor bytes_t;
    if (!archive.try_read(prefix + "_bytes", bytes_t)) return false;

    lengths_t = lengths_t.to(torch::kLong).contiguous();
    bytes_t = bytes_t.to(torch::kUInt8).contiguous();
    const auto lengths = lengths_t.accessor<int64_t, 1>();
    const auto* ptr = bytes_t.data_ptr<uint8_t>();
    const int64_t n_bytes = bytes_t.numel();

    out.clear();
    out.reserve(static_cast<size_t>(lengths_t.size(0)));
    int64_t offset = 0;
    for (int64_t i = 0; i < lengths_t.size(0); ++i) {
        const int64_t len = lengths[i];
        if (len < 0 || offset + len > n_bytes) {
            throw std::runtime_error(
                "checkpoint string list '" + prefix + "' is corrupt: entry " +
                std::to_string(i) + " runs past the byte buffer");
        }
        if (len == 0) {
            // ptr can legitimately be null for a 0-numel byte tensor, so never
            // form std::string(nullptr, 0) even though the length is zero.
            out.emplace_back();
        } else {
            out.emplace_back(reinterpret_cast<const char*>(ptr + offset),
                             static_cast<size_t>(len));
        }
        offset += len;
    }
    return true;
}

void save_scalers(
    torch::serialize::OutputArchive& archive,
    const Scalers& scalers
) {
    // Why: load_scalers previously ate exceptions silently. If the
    // "continuous_mean" archive.read failed for any reason, scalers came
    // back with undefined tensors and downstream predict crashed in
    // (continuous - undefined_tensor). Make presence explicit with a
    // boolean flag so the load path has a clean signal.
    int has_continuous = scalers.continuous_mean.defined() ? 1 : 0;
    archive.write("scalers_has_continuous", torch::tensor(has_continuous));
    if (has_continuous) {
        archive.write("continuous_mean", scalers.continuous_mean);
        archive.write("continuous_scale", scalers.continuous_scale);
    }

    // Save target scalers. Why: the previous format wrote only mean/scale
    // and dropped the target name, so load_scalers couldn't rebuild the
    // {name -> (mean, scale)} map and the loaded Predictor produced
    // predictions in scaled (mean=0, std=1) space instead of the original
    // target scale. Write name alongside each entry.
    archive.write("n_target_scalers", torch::tensor(static_cast<int64_t>(scalers.target_scalers.size())));
    int idx = 0;
    for (const auto& [name, scaler] : scalers.target_scalers) {
        std::string prefix = "target_scaler_" + std::to_string(idx) + "_";
        write_string_to_archive(archive, prefix + "name", name);
        archive.write(prefix + "mean", scaler.first);
        archive.write(prefix + "scale", scaler.second);
        idx++;
    }
}

Scalers load_scalers(
    torch::serialize::InputArchive& archive
) {
    Scalers scalers;

    // Continuous scalers. Prefer the explicit presence flag from the new
    // save format; fall back to try_read on bare keys for older checkpoints.
    torch::Tensor has_t;
    bool has_continuous = false;
    if (archive.try_read("scalers_has_continuous", has_t)) {
        has_continuous = (has_t.item<int>() != 0);
    } else {
        // Legacy checkpoint: just attempt the bare reads.
        has_continuous = true;
    }
    if (has_continuous) {
        // try_read avoids the silent-catch-everything anti-pattern below.
        archive.try_read("continuous_mean", scalers.continuous_mean);
        archive.try_read("continuous_scale", scalers.continuous_scale);
    }

    auto read_string_pair = [&](const std::string& prefix) -> std::string {
        torch::Tensor len_t;
        if (!archive.try_read(prefix + "_len", len_t)) return std::string();
        int64_t len = len_t.item<int64_t>();
        if (len <= 0) return std::string();
        torch::Tensor t;
        if (!archive.try_read(prefix, t)) return std::string();
        auto ptr = t.data_ptr<uint8_t>();
        return std::string(reinterpret_cast<const char*>(ptr), len);
    };

    // Target scalers: restore the {name -> (mean, scale)} map. Try the
    // new naming first; fall back to legacy keys so older checkpoints
    // still partially load (without names, target_scalers stays empty
    // and predictions come back in scaled space — better than crashing).
    torch::Tensor n_target_scalers_t;
    if (archive.try_read("n_target_scalers", n_target_scalers_t)) {
        int64_t n_scalers = n_target_scalers_t.item<int64_t>();
        for (int64_t i = 0; i < n_scalers; ++i) {
            std::string idx_s = std::to_string(i);
            std::string prefix = "target_scaler_" + idx_s + "_";
            std::string name = read_string_pair(prefix + "name");

            torch::Tensor mean, scale;
            if (!archive.try_read(prefix + "mean", mean)) {
                archive.try_read("target_scaler_mean_" + idx_s, mean);
            }
            if (!archive.try_read(prefix + "scale", scale)) {
                archive.try_read("target_scaler_scale_" + idx_s, scale);
            }
            if (!name.empty() && mean.defined() && scale.defined()) {
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
    archive.write(k::kNGeneraVocab, torch::tensor(schema.n_genera_vocab));
    archive.write(k::kNFamiliesVocab, torch::tensor(schema.n_families_vocab));
    archive.write(k::kTrackUnknownFrac, torch::tensor(static_cast<int>(schema.track_unknown_fraction)));
    archive.write(k::kTrackUnknownCount, torch::tensor(static_cast<int>(schema.track_unknown_count)));
    archive.write(k::kNCovariates, torch::tensor(static_cast<int64_t>(schema.covariate_names.size())));
    for (size_t i = 0; i < schema.covariate_names.size(); ++i) {
        write_string_to_archive(archive, k::covariate(static_cast<int64_t>(i)),
                                schema.covariate_names[i]);
    }
    archive.write(k::kNTargets, torch::tensor(static_cast<int64_t>(schema.targets.size())));
    for (size_t i = 0; i < schema.targets.size(); ++i) {
        const auto& target = schema.targets[i];
        std::string prefix = k::target_prefix(static_cast<int64_t>(i));
        write_string_to_archive(archive, prefix + k::kTargetName, target.name);
        archive.write(prefix + k::kTargetTask, torch::tensor(static_cast<int>(target.task)));
        archive.write(prefix + k::kTargetTransform, torch::tensor(static_cast<int>(target.transform)));
        archive.write(prefix + k::kTargetNumClasses, torch::tensor(target.num_classes));
        archive.write(prefix + k::kTargetWeight, torch::tensor(target.weight));

        // Ordered class vocabulary for classification targets. Empty (count
        // 0) for regression. Empty for already-integer-encoded
        // classification targets that the loader didn't auto-factorize.
        // Per-class strings are serialized via the same length-prefix +
        // UInt8 bytes scheme used elsewhere. Back-compat: pre-classification
        // checkpoints won't have this key; the load path treats absent ==
        // empty (see schema load below).
        archive.write(prefix + k::kTargetNClassNames,
                      torch::tensor(static_cast<int64_t>(target.class_names.size())));
        for (size_t j = 0; j < target.class_names.size(); ++j) {
            write_string_to_archive(archive,
                prefix + k::kTargetClassPrefix + std::to_string(j),
                target.class_names[j]);
        }

        // Optional per-class loss weights for imbalanced classification
        // (issue #91). Consumed by MultiTaskLoss via
        // CrossEntropyFuncOptions().weight(...). Length-prefixed float array,
        // matching the value-tree schema path that already carries it.
        // Back-compat: pre-fix checkpoints omit this key; the load path
        // treats absent == empty (unweighted CE).
        archive.write(prefix + k::kTargetNClassWeights,
                      torch::tensor(static_cast<int64_t>(target.class_weights.size())));
        if (!target.class_weights.empty()) {
            archive.write(prefix + k::kTargetClassWeights,
                          torch::tensor(target.class_weights));
        }
    }

    // Categorical covariates: column count + per-column name + per-column
    // vocab size + shared embed_dim. Vocab sizes include the reserved UNK
    // slot at code 0 (so the column's embedding table is size K+1).
    archive.write(k::kNCategoricals,
                  torch::tensor(static_cast<int64_t>(schema.categorical_names.size())));
    archive.write(k::kCategoricalEmbedDim,
                  torch::tensor(schema.categorical_embed_dim));
    for (size_t i = 0; i < schema.categorical_names.size(); ++i) {
        const std::string prefix = k::categorical_prefix(static_cast<int64_t>(i));
        write_string_to_archive(archive, prefix + k::kCategoricalName, schema.categorical_names[i]);
        archive.write(prefix + k::kCategoricalVocabSize,
                      torch::tensor(schema.categorical_vocab_sizes[i]));
    }

    // Rank-pool / transformer pooling scheme + resolved species cap (issue #38),
    // so the predict side rebuilds the same DatasetConfig instead of defaulting
    // to Log1p.
    archive.write(k::kPoolWeighting, torch::tensor(schema.pool_weighting));
    archive.write(k::kPoolSpeciesCap, torch::tensor(schema.pool_species_cap));

    // Remaining DatasetConfig knobs (issue #102). #38 restored pool_weighting
    // only; without these the inference-side DatasetConfig silently reverted to
    // the struct defaults for the selection / representation / normalization /
    // aggregation / taxonomy switches, and to top_k_species = 10 (which changes
    // the species_ids width in embed mode).
    archive.write(k::kTopKSpecies, torch::tensor(schema.top_k_species));
    archive.write(k::kSelection, torch::tensor(static_cast<int>(schema.selection)));
    archive.write(k::kRepresentation, torch::tensor(static_cast<int>(schema.representation)));
    archive.write(k::kNormalization, torch::tensor(static_cast<int>(schema.normalization)));
    archive.write(k::kAggregation, torch::tensor(static_cast<int>(schema.aggregation)));
    archive.write(k::kUseTaxonomy, torch::tensor(static_cast<int>(schema.use_taxonomy)));

    // Fitted species / genus / family vocabularies, index = integer code
    // (issue #102). These are what make a checkpoint self-sufficient for
    // inference: without them a new file re-fits its own frequency-ranked
    // species codes and its own sorted taxonomy codes, and every non-hash
    // encoder then indexes the embedding tables with the wrong rows.
    write_string_list(archive, k::kSpeciesVocab, schema.species_vocab);
    write_string_list(archive, k::kGenusVocab, schema.genus_vocab);
    write_string_list(archive, k::kFamilyVocab, schema.family_vocab);
}

ResolveSchema load_schema(
    torch::serialize::InputArchive& archive
) {
    // Why: each archive.read(key, t) reuses the destination tensor's
    // storage rather than allocating fresh. Reading heterogeneous dtypes
    // (int64 / int32 / float32) into the same tensor then triggers a
    // storage-size mismatch at libtorch's set_storage_offset (storage of
    // size 4 used to satisfy itemsize 8, or vice versa).
    // How to apply: every read uses a fresh local tensor.
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
    auto read_bool = [&](const std::string& key) -> bool {
        return read_i32(key) != 0;
    };
    auto read_f32 = [&](const std::string& key) {
        torch::Tensor t;
        archive.read(key, t);
        return t.item<float>();
    };
    auto read_string = [&](const std::string& prefix) -> std::string {
        int64_t len = read_i64(prefix + "_len");
        if (len <= 0) return std::string();
        torch::Tensor t;
        archive.read(prefix, t);
        auto ptr = t.data_ptr<uint8_t>();
        return std::string(reinterpret_cast<const char*>(ptr), len);
    };

    namespace k = ckpt_schema_keys;
    ResolveSchema schema;
    schema.n_plots = read_i64(k::kNPlots);
    schema.n_species = read_i64(k::kNSpecies);
    schema.n_species_vocab = read_i64(k::kNSpeciesVocab);
    schema.has_coordinates = read_bool(k::kHasCoordinates);
    schema.has_abundance = read_bool(k::kHasAbundance);
    schema.has_taxonomy = read_bool(k::kHasTaxonomy);
    schema.n_genera = read_i64(k::kNGenera);
    schema.n_families = read_i64(k::kNFamilies);
    schema.n_genera_vocab = read_i64(k::kNGeneraVocab);
    schema.n_families_vocab = read_i64(k::kNFamiliesVocab);
    schema.track_unknown_fraction = read_bool(k::kTrackUnknownFrac);
    schema.track_unknown_count = read_bool(k::kTrackUnknownCount);
    int64_t n_covariates = read_i64(k::kNCovariates);
    schema.covariate_names.resize(n_covariates);
    for (int64_t i = 0; i < n_covariates; ++i) {
        // Back-compat: older checkpoints didn't save covariate names.
        // try_read returns false silently when the key is absent, leaving
        // the existing empty string in place. Names aren't load-bearing
        // for model construction (model indexes by count, not name), so
        // empty-string fallback is safe.
        torch::Tensor len_t;
        if (archive.try_read(k::covariate(i) + "_len", len_t)) {
            int64_t len = len_t.item<int64_t>();
            if (len > 0) {
                torch::Tensor name_t;
                archive.read(k::covariate(i), name_t);
                auto ptr = name_t.data_ptr<uint8_t>();
                schema.covariate_names[i] = std::string(reinterpret_cast<const char*>(ptr), len);
            }
        }
    }
    int64_t n_targets = read_i64(k::kNTargets);
    schema.targets.resize(n_targets);
    for (int64_t i = 0; i < n_targets; ++i) {
        std::string prefix = k::target_prefix(i);
        // Back-compat: older checkpoints didn't save target names. Missing
        // names would collide on register_module("head_") for all targets,
        // so synthesize a fallback name when absent.
        torch::Tensor name_len_t;
        if (archive.try_read(prefix + k::kTargetName + "_len", name_len_t)) {
            schema.targets[i].name = read_string(prefix + k::kTargetName);
        }
        if (schema.targets[i].name.empty()) {
            schema.targets[i].name = "target_" + std::to_string(i);
        }
        schema.targets[i].task = static_cast<TaskType>(read_i32(prefix + k::kTargetTask));
        schema.targets[i].transform = static_cast<TransformType>(read_i32(prefix + k::kTargetTransform));
        schema.targets[i].num_classes = read_i32(prefix + k::kTargetNumClasses);
        schema.targets[i].weight = read_f32(prefix + k::kTargetWeight);

        // Class names (back-compat: pre-classification checkpoints omit
        // these keys; treat absent as no class vocab, which matches the
        // original behaviour where classification just used raw int codes).
        torch::Tensor n_cn_t;
        if (archive.try_read(prefix + k::kTargetNClassNames, n_cn_t)) {
            int64_t n_cn = n_cn_t.item<int64_t>();
            schema.targets[i].class_names.resize(static_cast<size_t>(n_cn));
            for (int64_t j = 0; j < n_cn; ++j) {
                schema.targets[i].class_names[j] =
                    read_string(prefix + k::kTargetClassPrefix + std::to_string(j));
            }
        }

        // Class weights (issue #91; back-compat: absent == empty == unweighted CE).
        torch::Tensor n_cw_t;
        if (archive.try_read(prefix + k::kTargetNClassWeights, n_cw_t)) {
            int64_t n_cw = n_cw_t.item<int64_t>();
            if (n_cw > 0) {
                torch::Tensor cw_t;
                archive.read(prefix + k::kTargetClassWeights, cw_t);
                cw_t = cw_t.to(torch::kFloat32).contiguous();
                auto ptr = cw_t.data_ptr<float>();
                schema.targets[i].class_weights.assign(ptr, ptr + n_cw);
            }
        }
    }

    // Categorical covariates (back-compat: pre-categorical-port checkpoints
    // won't have any of these keys; treat as schema with zero categoricals).
    torch::Tensor n_cat_t;
    if (archive.try_read(k::kNCategoricals, n_cat_t)) {
        int64_t n_cat = n_cat_t.item<int64_t>();
        torch::Tensor embed_dim_t;
        if (archive.try_read(k::kCategoricalEmbedDim, embed_dim_t)) {
            schema.categorical_embed_dim = embed_dim_t.item<int64_t>();
        }
        schema.categorical_names.resize(n_cat);
        schema.categorical_vocab_sizes.resize(n_cat);
        for (int64_t i = 0; i < n_cat; ++i) {
            const std::string prefix = k::categorical_prefix(i);
            schema.categorical_names[i] = read_string(prefix + k::kCategoricalName);
            schema.categorical_vocab_sizes[i] = read_i64(prefix + k::kCategoricalVocabSize);
        }
    }

    // Pool weighting scheme + species cap (back-compat: pre-issue-#38
    // checkpoints keep the schema defaults, Log1p / auto).
    torch::Tensor pool_w_t, pool_cap_t;
    if (archive.try_read(k::kPoolWeighting, pool_w_t)) {
        schema.pool_weighting = pool_w_t.item<int>();
    }
    if (archive.try_read(k::kPoolSpeciesCap, pool_cap_t)) {
        schema.pool_species_cap = pool_cap_t.item<int>();
    }

    // Remaining DatasetConfig knobs (issue #102). Each read uses a FRESH tensor
    // (InputArchive::read copies into the passed tensor, so reusing one across
    // differing dtypes trips a setStorage mismatch); try_read keeps the schema
    // default -- which is the DatasetConfig default -- for a pre-fix checkpoint.
    auto rd_i32 = [&](const char* key, auto&& assign) {
        torch::Tensor t;
        if (archive.try_read(key, t)) assign(t.item<int>());
    };
    rd_i32(k::kTopKSpecies, [&](int v) { schema.top_k_species = v; });
    rd_i32(k::kSelection, [&](int v) { schema.selection = static_cast<SelectionMode>(v); });
    rd_i32(k::kRepresentation, [&](int v) { schema.representation = static_cast<RepresentationMode>(v); });
    rd_i32(k::kNormalization, [&](int v) { schema.normalization = static_cast<NormalizationMode>(v); });
    rd_i32(k::kAggregation, [&](int v) { schema.aggregation = static_cast<AggregationMode>(v); });
    rd_i32(k::kUseTaxonomy, [&](int v) { schema.use_taxonomy = (v != 0); });

    // Fitted vocabularies (issue #102). Absent on a pre-fix checkpoint: the
    // vectors stay empty, has_species_vocab()/has_taxonomy_vocab() report
    // false, and the Predictor falls back to the size-only guard plus a loud
    // warning rather than silently trusting re-fitted codes.
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
    // Every persisted field comes from the TrainConfig registry, which also
    // carries the `train_*` key each one has always been written under. The
    // fields with an empty key (device, checkpoint destination, AMP / cuDNN
    // switches, log callback) describe the machine a run happened on rather than
    // the recipe and stay out of the archive; the registry's writer skips them.
    //
    // batch_size semantics in the checkpoint:
    // ---------------------------------------
    // Trainer::fit mutates `config_.batch_size` in place when the CUDA
    // auto-halve-on-OOM loop fires, so `config.batch_size` at save time is the
    // EFFECTIVE batch size that actually trained the model. `train_batch_size`
    // records the value the caller REQUESTED (passed here, restored by
    // load_train_config), and `train_effective_batch_size` records the effective
    // value, so a fallback run is detectable when the two diverge. On a clean run
    // they are equal. A -1 requested value means "no separate request known".
    // The requested value goes through the registry pass on a copy, so
    // train_batch_size is written exactly once.
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

TrainConfig load_train_config(
    torch::serialize::InputArchive& archive
) {
    // Inverse of save_train_config, over the same registry rows and the same
    // `train_*` keys. Fields the writer skips (device, checkpoint_dir /
    // checkpoint_every, AMP and cuDNN flags, log callback) keep their TrainConfig
    // defaults -- the caller sets those for the run. Every read is a try_read, so
    // a checkpoint missing a key falls back to the struct default instead of
    // throwing (forward/backward compatibility across schema additions).
    TrainConfig config;
    for_each_field(config, ArchiveFieldReader{archive, std::string()});
    return config;
}

RunMetadata load_run_metadata(
    torch::serialize::InputArchive& archive
) {
    // Inverse of save_run_metadata. Decodes the byte-stored strings and the
    // flattened per-target metric tree. Missing keys fall back to defaults.
    RunMetadata meta;

    auto read_string_pair = [&](const std::string& prefix) -> std::string {
        torch::Tensor len_t;
        if (!archive.try_read(prefix + "_len", len_t)) return std::string();
        int64_t len = len_t.item<int64_t>();
        if (len <= 0) return std::string();
        torch::Tensor t;
        if (!archive.try_read(prefix, t)) return std::string();
        auto ptr = t.data_ptr<uint8_t>();
        return std::string(reinterpret_cast<const char*>(ptr), static_cast<size_t>(len));
    };

    std::string version = read_string_pair("meta_version");
    if (!version.empty()) meta.resolve_version = version;
    meta.created_at = read_string_pair("meta_created");
    meta.completed_at = read_string_pair("meta_completed");

    // Fresh tensor per read (see load_train_config note on InputArchive::read).
    { torch::Tensor x; if (archive.try_read("meta_train_time", x)) meta.train_time_seconds = x.item<float>(); }
    { torch::Tensor x; if (archive.try_read("meta_n_plots_train", x)) meta.n_plots_train = x.item<int64_t>(); }
    { torch::Tensor x; if (archive.try_read("meta_n_plots_test", x)) meta.n_plots_test = x.item<int64_t>(); }
    { torch::Tensor x; if (archive.try_read("meta_best_epoch", x)) meta.best_epoch = x.item<int>(); }
    { torch::Tensor x; if (archive.try_read("meta_total_epochs", x)) meta.total_epochs = x.item<int>(); }

    torch::Tensor n_targets_t;
    if (archive.try_read("meta_n_targets", n_targets_t)) {
        int64_t n_targets = n_targets_t.item<int64_t>();
        for (int64_t ti = 0; ti < n_targets; ++ti) {
            std::string prefix = "meta_target_" + std::to_string(ti) + "_";
            std::string target_name = read_string_pair(prefix + "name");
            torch::Tensor n_metrics_t;
            if (!archive.try_read(prefix + "n_metrics", n_metrics_t)) continue;
            int64_t n_metrics = n_metrics_t.item<int64_t>();
            std::unordered_map<std::string, float> metrics;
            for (int64_t mi = 0; mi < n_metrics; ++mi) {
                std::string m_prefix = prefix + "metric_" + std::to_string(mi) + "_";
                std::string metric_name = read_string_pair(m_prefix + "name");
                torch::Tensor val_t;
                if (archive.try_read(m_prefix + "value", val_t))
                    metrics[metric_name] = val_t.item<float>();
            }
            meta.final_metrics[target_name] = std::move(metrics);
        }
    }
    return meta;
}

void write_metadata_json(
    const std::string& checkpoint_path,
    const ModelConfig& model_config,
    const TrainConfig& train_config,
    const RunMetadata& metadata,
    const ResolveSchema& schema,
    int requested_batch_size
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
    file << "  \"resolve_version\": \"" << json_escape(metadata.resolve_version) << "\",\n";
    file << "  \"created_at\": \"" << json_escape(metadata.created_at) << "\",\n";
    file << "  \"completed_at\": \"" << json_escape(metadata.completed_at) << "\",\n";
    file << "  \"train_time_seconds\": " << metadata.train_time_seconds << ",\n";
    file << "  \"n_plots_train\": " << metadata.n_plots_train << ",\n";
    file << "  \"n_plots_test\": " << metadata.n_plots_test << ",\n";
    file << "  \"best_epoch\": " << metadata.best_epoch << ",\n";
    file << "  \"total_epochs\": " << metadata.total_epochs << ",\n";

    // Model config. Emitted from the field registry, so every field the struct
    // carries -- including the architecture sub-configs -- reaches the sidecar,
    // and a field added later appears here without another edit. Enums are
    // written as the names the CLI accepts rather than as ordinals.
    file << "  \"model_config\": {\n";
    {
        JsonFieldWriter writer{file, 4};
        for_each_field(model_config, writer);
    }
    file << "\n  },\n";

    // Train config. `batch_size` is the value the caller REQUESTED; the CUDA OOM
    // auto-halve retry may have shrunk the value that actually trained the model,
    // reported under `effective_batch_size`. `batch_size_floor` is the retry lower
    // bound. When the two batch sizes diverge, the run fell back on OOM (issue #86).
    const int req_bs = requested_batch_size >= 0 ? requested_batch_size : train_config.batch_size;
    TrainConfig requested_config = train_config;
    requested_config.batch_size = req_bs;
    file << "  \"train_config\": {\n";
    {
        JsonFieldWriter writer{file, 4};
        for_each_field(requested_config, writer);
        writer.open_member("effective_batch_size");
        file << train_config.batch_size;
    }
    file << "\n  },\n";

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
        file << "    \"" << json_escape(target_name) << "\": {\n";
        bool first_metric = true;
        for (const auto& [metric_name, value] : metrics) {
            if (!first_metric) file << ",\n";
            first_metric = false;
            file << "      \"" << json_escape(metric_name) << "\": " << value;
        }
        file << "\n    }";
    }
    file << "\n  }\n";

    file << "}\n";
}

} // namespace resolve
