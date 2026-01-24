#include "resolve/plot_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace resolve {

// ============================================================================
// CategoryVocab Implementation
// ============================================================================

void CategoryVocab::fit(const std::vector<std::string>& values) {
    // Collect unique values and sort for deterministic ordering
    std::set<std::string> unique_values(values.begin(), values.end());

    // Index 0 reserved for unknown
    unk_idx_ = 0;
    idx_to_value_.clear();
    idx_to_value_.push_back("<UNK>");
    value_to_idx_.clear();
    value_to_idx_["<UNK>"] = 0;

    // Add all unique values
    for (const auto& val : unique_values) {
        if (val.empty()) continue;
        int64_t idx = static_cast<int64_t>(idx_to_value_.size());
        idx_to_value_.push_back(val);
        value_to_idx_[val] = idx;
    }
}

int64_t CategoryVocab::encode(const std::string& value) const {
    auto it = value_to_idx_.find(value);
    if (it != value_to_idx_.end()) {
        return it->second;
    }
    return unk_idx_;
}

bool CategoryVocab::contains(const std::string& value) const {
    return value_to_idx_.find(value) != value_to_idx_.end();
}

void CategoryVocab::save(std::ostream& os) const {
    os << idx_to_value_.size() << "\n";
    for (const auto& val : idx_to_value_) {
        os << val << "\n";
    }
}

CategoryVocab CategoryVocab::load(std::istream& is) {
    CategoryVocab vocab;
    size_t size;
    is >> size;
    is.ignore();  // Skip newline

    vocab.idx_to_value_.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        std::string val;
        std::getline(is, val);
        vocab.idx_to_value_.push_back(val);
        vocab.value_to_idx_[val] = static_cast<int64_t>(i);
    }
    vocab.unk_idx_ = 0;
    return vocab;
}

// ============================================================================
// StandardScaler Implementation
// ============================================================================

void StandardScaler::fit(const std::vector<float>& values) {
    if (values.empty()) {
        mean_ = 0.0f;
        std_ = 1.0f;
        fitted_ = true;
        return;
    }

    // Compute mean
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    mean_ = static_cast<float>(sum / values.size());

    // Compute std
    double sq_sum = 0.0;
    for (float v : values) {
        sq_sum += (v - mean_) * (v - mean_);
    }
    std_ = static_cast<float>(std::sqrt(sq_sum / values.size()));

    // Avoid division by zero
    if (std_ < 1e-8f) {
        std_ = 1.0f;
    }

    fitted_ = true;
}

float StandardScaler::transform(float value) const {
    return (value - mean_) / std_;
}

std::vector<float> StandardScaler::transform(const std::vector<float>& values) const {
    std::vector<float> result;
    result.reserve(values.size());
    for (float v : values) {
        result.push_back(transform(v));
    }
    return result;
}

void StandardScaler::save(std::ostream& os) const {
    os << mean_ << " " << std_ << "\n";
}

StandardScaler StandardScaler::load(std::istream& is) {
    StandardScaler scaler;
    is >> scaler.mean_ >> scaler.std_;
    scaler.fitted_ = true;
    return scaler;
}

// ============================================================================
// EncodedPlotData Implementation
// ============================================================================

torch::Tensor EncodedPlotData::continuous_features() const {
    std::vector<torch::Tensor> tensors;

    for (const auto& col : columns) {
        if (!col.is_embedding_ids) {
            tensors.push_back(col.values);
        }
    }

    if (tensors.empty()) {
        // Return empty tensor with correct batch size
        int64_t n_plots = plot_ids.size();
        return torch::zeros({n_plots, 0}, torch::kFloat32);
    }

    return torch::cat(tensors, /*dim=*/1);
}

torch::Tensor EncodedPlotData::embedding_ids(const std::string& name) const {
    for (const auto& col : columns) {
        if (col.name == name && col.is_embedding_ids) {
            return col.values;
        }
    }
    throw std::runtime_error("Embedding column not found: " + name);
}

std::vector<std::tuple<std::string, int64_t, int, int>> EncodedPlotData::embedding_specs() const {
    std::vector<std::tuple<std::string, int64_t, int, int>> specs;
    for (const auto& col : columns) {
        if (col.is_embedding_ids) {
            specs.emplace_back(col.name, col.vocab_size, col.embed_dim, col.n_slots);
        }
    }
    return specs;
}

// ============================================================================
// PlotEncoder Implementation
// ============================================================================

void PlotEncoder::add_numeric(const std::string& name,
                              const std::vector<std::string>& columns,
                              DataSource source) {
    ColumnSpec spec;
    spec.name = name;
    spec.columns = columns;
    spec.type = EncodingType::Numeric;
    spec.source = source;
    specs_.push_back(spec);
}

void PlotEncoder::add_raw(const std::string& name,
                          const std::vector<std::string>& columns,
                          DataSource source) {
    ColumnSpec spec;
    spec.name = name;
    spec.columns = columns;
    spec.type = EncodingType::Raw;
    spec.source = source;
    specs_.push_back(spec);
}

void PlotEncoder::add_hash(const std::string& name,
                           const std::vector<std::string>& columns,
                           int dim,
                           int top_k,
                           int bottom_k,
                           const std::string& rank_by,
                           DataSource source) {
    ColumnSpec spec;
    spec.name = name;
    spec.columns = columns;
    spec.type = EncodingType::Hash;
    spec.source = source;
    spec.dim = dim;
    spec.top_k = top_k;
    spec.bottom_k = bottom_k;
    spec.rank_by = rank_by;
    specs_.push_back(spec);
}

void PlotEncoder::add_embed(const std::string& name,
                            const std::vector<std::string>& columns,
                            int dim,
                            int top_k,
                            int bottom_k,
                            const std::string& rank_by,
                            DataSource source) {
    ColumnSpec spec;
    spec.name = name;
    spec.columns = columns;
    spec.type = EncodingType::Embed;
    spec.source = source;
    spec.dim = dim;
    spec.top_k = top_k;
    spec.bottom_k = bottom_k;
    spec.rank_by = rank_by;
    specs_.push_back(spec);
}

void PlotEncoder::add_onehot(const std::string& name,
                             const std::vector<std::string>& columns,
                             DataSource source) {
    ColumnSpec spec;
    spec.name = name;
    spec.columns = columns;
    spec.type = EncodingType::OneHot;
    spec.source = source;
    specs_.push_back(spec);
}

void PlotEncoder::fit(const std::vector<PlotRecord>& plot_data,
                      const std::vector<ObservationRecord>& obs_data) {
    for (const auto& spec : specs_) {
        if (spec.type == EncodingType::Numeric) {
            // Fit scaler for each column
            for (const auto& col : spec.columns) {
                std::vector<float> values;
                if (spec.source == DataSource::Plot) {
                    for (const auto& record : plot_data) {
                        auto it = record.numeric.find(col);
                        if (it != record.numeric.end()) {
                            values.push_back(it->second);
                        }
                    }
                } else {
                    for (const auto& record : obs_data) {
                        auto it = record.numeric.find(col);
                        if (it != record.numeric.end()) {
                            values.push_back(it->second);
                        }
                    }
                }

                StandardScaler scaler;
                scaler.fit(values);
                scalers_[spec.name + ":" + col] = scaler;
            }
        } else if (spec.type == EncodingType::Embed || spec.type == EncodingType::OneHot) {
            // Fit vocabulary for each column
            for (const auto& col : spec.columns) {
                std::vector<std::string> values;
                if (spec.source == DataSource::Plot) {
                    for (const auto& record : plot_data) {
                        auto it = record.categorical.find(col);
                        if (it != record.categorical.end()) {
                            values.push_back(it->second);
                        }
                    }
                } else {
                    for (const auto& record : obs_data) {
                        auto it = record.categorical.find(col);
                        if (it != record.categorical.end()) {
                            values.push_back(it->second);
                        }
                    }
                }

                CategoryVocab vocab;
                vocab.fit(values);
                vocabs_[spec.name + ":" + col] = vocab;

                // Track known values for unknown detection
                known_values_[spec.name + ":" + col] =
                    std::unordered_set<std::string>(values.begin(), values.end());
            }
        } else if (spec.type == EncodingType::Hash) {
            // Track known values for unknown detection
            for (const auto& col : spec.columns) {
                std::unordered_set<std::string> values;
                if (spec.source == DataSource::Plot) {
                    for (const auto& record : plot_data) {
                        auto it = record.categorical.find(col);
                        if (it != record.categorical.end()) {
                            values.insert(it->second);
                        }
                    }
                } else {
                    for (const auto& record : obs_data) {
                        auto it = record.categorical.find(col);
                        if (it != record.categorical.end()) {
                            values.insert(it->second);
                        }
                    }
                }
                known_values_[spec.name + ":" + col] = values;
            }
        }
        // Raw type needs no fitting
    }

    fitted_ = true;
}

EncodedPlotData PlotEncoder::transform(
    const std::vector<PlotRecord>& plot_data,
    const std::vector<ObservationRecord>& obs_data,
    const std::vector<std::string>& plot_ids
) const {
    if (!fitted_) {
        throw std::runtime_error("PlotEncoder must be fitted before transform");
    }

    EncodedPlotData result;
    result.plot_ids = plot_ids;

    // Build plot_id to index map
    std::unordered_map<std::string, int64_t> plot_id_to_idx;
    for (size_t i = 0; i < plot_ids.size(); ++i) {
        plot_id_to_idx[plot_ids[i]] = static_cast<int64_t>(i);
    }

    // Build plot_id to records maps
    std::unordered_map<std::string, const PlotRecord*> plot_records_map;
    for (const auto& record : plot_data) {
        plot_records_map[record.plot_id] = &record;
    }

    std::unordered_map<std::string, std::vector<const ObservationRecord*>> obs_records_map;
    for (const auto& record : obs_data) {
        obs_records_map[record.plot_id].push_back(&record);
    }

    // Encode each spec
    for (const auto& spec : specs_) {
        EncodedColumn col;
        col.name = spec.name;
        col.is_embedding_ids = (spec.type == EncodingType::Embed);
        col.vocab_size = 0;
        col.embed_dim = spec.dim;
        col.n_slots = spec.n_slots();

        switch (spec.type) {
            case EncodingType::Numeric:
            case EncodingType::Raw: {
                col.values = encode_numeric(spec, plot_data, plot_ids);
                break;
            }
            case EncodingType::Hash: {
                col.values = encode_hash(spec, obs_data, plot_ids);
                break;
            }
            case EncodingType::Embed: {
                col.values = encode_embed(spec, plot_data, obs_data, plot_ids);
                // Set vocab size (use first column's vocab)
                if (!spec.columns.empty()) {
                    auto it = vocabs_.find(spec.name + ":" + spec.columns[0]);
                    if (it != vocabs_.end()) {
                        col.vocab_size = it->second.size();
                    }
                }
                break;
            }
            case EncodingType::OneHot: {
                col.values = encode_onehot(spec, plot_data, plot_ids);
                break;
            }
        }

        result.columns.push_back(col);
    }

    // Compute unknown fraction (for observation-level data)
    result.unknown_fraction = torch::zeros({static_cast<int64_t>(plot_ids.size())}, torch::kFloat32);
    // TODO: Implement proper unknown tracking if needed

    return result;
}

EncodedPlotData PlotEncoder::fit_transform(
    const std::vector<PlotRecord>& plot_data,
    const std::vector<ObservationRecord>& obs_data,
    const std::vector<std::string>& plot_ids
) {
    fit(plot_data, obs_data);
    return transform(plot_data, obs_data, plot_ids);
}

int64_t PlotEncoder::vocab_size(const std::string& name) const {
    for (const auto& spec : specs_) {
        if (spec.name == name && spec.type == EncodingType::Embed) {
            if (!spec.columns.empty()) {
                auto it = vocabs_.find(spec.name + ":" + spec.columns[0]);
                if (it != vocabs_.end()) {
                    return it->second.size();
                }
            }
        }
    }
    return 0;
}

int PlotEncoder::continuous_dim() const {
    int dim = 0;
    for (const auto& spec : specs_) {
        switch (spec.type) {
            case EncodingType::Numeric:
            case EncodingType::Raw:
                dim += static_cast<int>(spec.columns.size());
                break;
            case EncodingType::Hash:
                dim += spec.dim;
                break;
            case EncodingType::OneHot:
                for (const auto& col : spec.columns) {
                    auto it = vocabs_.find(spec.name + ":" + col);
                    if (it != vocabs_.end()) {
                        dim += static_cast<int>(it->second.size());
                    }
                }
                break;
            case EncodingType::Embed:
                // Embeddings don't contribute to continuous dim directly
                break;
        }
    }
    return dim;
}

std::vector<std::tuple<std::string, int64_t, int, int>> PlotEncoder::embedding_configs() const {
    std::vector<std::tuple<std::string, int64_t, int, int>> configs;
    for (const auto& spec : specs_) {
        if (spec.type == EncodingType::Embed) {
            int64_t vsize = vocab_size(spec.name);
            configs.emplace_back(spec.name, vsize, spec.dim, spec.n_slots());
        }
    }
    return configs;
}

// ============================================================================
// Internal Encoding Methods
// ============================================================================

torch::Tensor PlotEncoder::encode_numeric(
    const ColumnSpec& spec,
    const std::vector<PlotRecord>& plot_data,
    const std::vector<std::string>& plot_ids
) const {
    int64_t n_plots = plot_ids.size();
    int64_t n_cols = spec.columns.size();
    auto result = torch::zeros({n_plots, n_cols}, torch::kFloat32);
    auto accessor = result.accessor<float, 2>();

    // Build plot_id to record map
    std::unordered_map<std::string, const PlotRecord*> records_map;
    for (const auto& record : plot_data) {
        records_map[record.plot_id] = &record;
    }

    for (int64_t i = 0; i < n_plots; ++i) {
        auto it = records_map.find(plot_ids[i]);
        if (it == records_map.end()) continue;

        const auto& record = *it->second;
        for (int64_t j = 0; j < n_cols; ++j) {
            const auto& col = spec.columns[j];
            auto val_it = record.numeric.find(col);
            if (val_it != record.numeric.end()) {
                float value = val_it->second;

                // Apply scaling if Numeric (not Raw)
                if (spec.type == EncodingType::Numeric) {
                    auto scaler_it = scalers_.find(spec.name + ":" + col);
                    if (scaler_it != scalers_.end()) {
                        value = scaler_it->second.transform(value);
                    }
                }

                accessor[i][j] = value;
            }
        }
    }

    return result;
}

torch::Tensor PlotEncoder::encode_hash(
    const ColumnSpec& spec,
    const std::vector<ObservationRecord>& obs_data,
    const std::vector<std::string>& plot_ids
) const {
    int64_t n_plots = plot_ids.size();
    auto result = torch::zeros({n_plots, spec.dim}, torch::kFloat32);

    // Group observations by plot
    std::unordered_map<std::string, std::vector<const ObservationRecord*>> plot_obs;
    for (const auto& record : obs_data) {
        plot_obs[record.plot_id].push_back(&record);
    }

    // Build plot_id to index
    std::unordered_map<std::string, int64_t> plot_idx;
    for (size_t i = 0; i < plot_ids.size(); ++i) {
        plot_idx[plot_ids[i]] = static_cast<int64_t>(i);
    }

    for (const auto& [pid, records] : plot_obs) {
        auto idx_it = plot_idx.find(pid);
        if (idx_it == plot_idx.end()) continue;
        int64_t i = idx_it->second;

        // Aggregate values across columns
        std::unordered_map<std::string, float> weighted_values;
        for (const auto* record : records) {
            // Get weight (from rank_by column or default to 1)
            float weight = 1.0f;
            if (!spec.rank_by.empty()) {
                auto w_it = record->numeric.find(spec.rank_by);
                if (w_it != record->numeric.end()) {
                    weight = w_it->second;
                }
            }

            // Combine column values into a single key
            std::string key;
            for (const auto& col : spec.columns) {
                auto it = record->categorical.find(col);
                if (it != record->categorical.end()) {
                    if (!key.empty()) key += ":";
                    key += it->second;
                }
            }

            if (!key.empty()) {
                weighted_values[key] += weight;
            }
        }

        // Apply top/bottom selection if specified
        if (spec.has_selection()) {
            auto selected = select_top_bottom(weighted_values, spec.top_k, spec.bottom_k);
            std::unordered_map<std::string, float> filtered;
            for (const auto& key : selected) {
                filtered[key] = weighted_values[key];
            }
            weighted_values = filtered;
        }

        // Hash the values
        auto hashed = hash_values(weighted_values, spec.dim);
        result[i] = hashed;
    }

    return result;
}

torch::Tensor PlotEncoder::encode_embed(
    const ColumnSpec& spec,
    const std::vector<PlotRecord>& plot_data,
    const std::vector<ObservationRecord>& obs_data,
    const std::vector<std::string>& plot_ids
) const {
    int64_t n_plots = plot_ids.size();
    int n_slots = spec.n_slots();
    int n_cols = static_cast<int>(spec.columns.size());

    // Output: (n_plots, n_slots * n_cols)
    auto result = torch::zeros({n_plots, n_slots * n_cols}, torch::kInt64);

    if (spec.source == DataSource::Plot) {
        // Plot-level: single value per plot per column
        std::unordered_map<std::string, const PlotRecord*> records_map;
        for (const auto& record : plot_data) {
            records_map[record.plot_id] = &record;
        }

        for (int64_t i = 0; i < n_plots; ++i) {
            auto it = records_map.find(plot_ids[i]);
            if (it == records_map.end()) continue;

            const auto& record = *it->second;
            for (int j = 0; j < n_cols; ++j) {
                const auto& col = spec.columns[j];
                auto val_it = record.categorical.find(col);
                if (val_it != record.categorical.end()) {
                    auto vocab_it = vocabs_.find(spec.name + ":" + col);
                    if (vocab_it != vocabs_.end()) {
                        int64_t idx = vocab_it->second.encode(val_it->second);
                        result[i][j] = idx;
                    }
                }
            }
        }
    } else {
        // Observation-level: aggregate and select top/bottom
        std::unordered_map<std::string, std::vector<const ObservationRecord*>> plot_obs;
        for (const auto& record : obs_data) {
            plot_obs[record.plot_id].push_back(&record);
        }

        std::unordered_map<std::string, int64_t> plot_idx;
        for (size_t i = 0; i < plot_ids.size(); ++i) {
            plot_idx[plot_ids[i]] = static_cast<int64_t>(i);
        }

        for (const auto& [pid, records] : plot_obs) {
            auto idx_it = plot_idx.find(pid);
            if (idx_it == plot_idx.end()) continue;
            int64_t i = idx_it->second;

            // For each column, aggregate weights and select
            for (int j = 0; j < n_cols; ++j) {
                const auto& col = spec.columns[j];
                auto vocab_it = vocabs_.find(spec.name + ":" + col);
                if (vocab_it == vocabs_.end()) continue;

                std::unordered_map<std::string, float> weighted_values;
                for (const auto* record : records) {
                    float weight = 1.0f;
                    if (!spec.rank_by.empty()) {
                        auto w_it = record->numeric.find(spec.rank_by);
                        if (w_it != record->numeric.end()) {
                            weight = w_it->second;
                        }
                    }

                    auto it = record->categorical.find(col);
                    if (it != record->categorical.end()) {
                        weighted_values[it->second] += weight;
                    }
                }

                // Select top/bottom
                auto selected = select_top_bottom(weighted_values, spec.top_k, spec.bottom_k);

                // Encode selected values
                for (int k = 0; k < n_slots && k < static_cast<int>(selected.size()); ++k) {
                    int64_t idx = vocab_it->second.encode(selected[k]);
                    result[i][j * n_slots + k] = idx;
                }
            }
        }
    }

    return result;
}

torch::Tensor PlotEncoder::encode_onehot(
    const ColumnSpec& spec,
    const std::vector<PlotRecord>& plot_data,
    const std::vector<std::string>& plot_ids
) const {
    int64_t n_plots = plot_ids.size();

    // Calculate total dimension
    int total_dim = 0;
    for (const auto& col : spec.columns) {
        auto it = vocabs_.find(spec.name + ":" + col);
        if (it != vocabs_.end()) {
            total_dim += static_cast<int>(it->second.size());
        }
    }

    auto result = torch::zeros({n_plots, total_dim}, torch::kFloat32);

    // Build plot_id to record map
    std::unordered_map<std::string, const PlotRecord*> records_map;
    for (const auto& record : plot_data) {
        records_map[record.plot_id] = &record;
    }

    int offset = 0;
    for (const auto& col : spec.columns) {
        auto vocab_it = vocabs_.find(spec.name + ":" + col);
        if (vocab_it == vocabs_.end()) continue;

        const auto& vocab = vocab_it->second;
        int vocab_size = static_cast<int>(vocab.size());

        for (int64_t i = 0; i < n_plots; ++i) {
            auto it = records_map.find(plot_ids[i]);
            if (it == records_map.end()) continue;

            const auto& record = *it->second;
            auto val_it = record.categorical.find(col);
            if (val_it != record.categorical.end()) {
                int64_t idx = vocab.encode(val_it->second);
                if (idx >= 0 && idx < vocab_size) {
                    result[i][offset + idx] = 1.0f;
                }
            }
        }

        offset += vocab_size;
    }

    return result;
}

torch::Tensor PlotEncoder::hash_values(
    const std::unordered_map<std::string, float>& weighted_values,
    int dim
) const {
    auto result = torch::zeros({dim}, torch::kFloat32);
    auto accessor = result.accessor<float, 1>();

    // Normalize weights
    float total = 0.0f;
    for (const auto& [key, weight] : weighted_values) {
        total += weight;
    }
    if (total < 1e-8f) total = 1.0f;

    for (const auto& [key, weight] : weighted_values) {
        float norm_weight = weight / total;

        // Hash the key
        std::hash<std::string> hasher;
        size_t hash = hasher(key);

        // Determine sign (using second hash)
        size_t sign_hash = hasher(key + "_sign");
        float sign = (sign_hash % 2 == 0) ? 1.0f : -1.0f;

        // Determine bucket
        int bucket = static_cast<int>(hash % dim);

        accessor[bucket] += sign * norm_weight;
    }

    return result;
}

std::vector<std::string> PlotEncoder::select_top_bottom(
    const std::unordered_map<std::string, float>& weighted_values,
    int top_k,
    int bottom_k
) const {
    // Sort by weight descending
    std::vector<std::pair<std::string, float>> sorted_values(
        weighted_values.begin(), weighted_values.end()
    );
    std::sort(sorted_values.begin(), sorted_values.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<std::string> selected;
    std::unordered_set<std::string> added;

    // Add top-k
    for (int k = 0; k < top_k && k < static_cast<int>(sorted_values.size()); ++k) {
        selected.push_back(sorted_values[k].first);
        added.insert(sorted_values[k].first);
    }

    // Add bottom-k (from end, avoiding duplicates)
    for (int k = 0; k < bottom_k; ++k) {
        int idx = static_cast<int>(sorted_values.size()) - 1 - k;
        if (idx < 0) break;
        if (added.find(sorted_values[idx].first) == added.end()) {
            selected.push_back(sorted_values[idx].first);
            added.insert(sorted_values[idx].first);
        }
    }

    return selected;
}

// ============================================================================
// Serialization
// ============================================================================

void PlotEncoder::save(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }

    // Version
    ofs << "PlotEncoder_v1\n";

    // Number of specs
    ofs << specs_.size() << "\n";

    for (const auto& spec : specs_) {
        ofs << spec.name << "\n";
        ofs << static_cast<int>(spec.type) << " " << static_cast<int>(spec.source) << "\n";
        ofs << spec.columns.size();
        for (const auto& col : spec.columns) {
            ofs << " " << col;
        }
        ofs << "\n";
        ofs << spec.dim << " " << spec.top_k << " " << spec.bottom_k << " " << spec.rank_by << "\n";
    }

    // Save vocabs
    ofs << vocabs_.size() << "\n";
    for (const auto& [name, vocab] : vocabs_) {
        ofs << name << "\n";
        vocab.save(ofs);
    }

    // Save scalers
    ofs << scalers_.size() << "\n";
    for (const auto& [name, scaler] : scalers_) {
        ofs << name << "\n";
        scaler.save(ofs);
    }
}

PlotEncoder PlotEncoder::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }

    std::string version;
    std::getline(ifs, version);
    if (version != "PlotEncoder_v1") {
        throw std::runtime_error("Unknown PlotEncoder format: " + version);
    }

    PlotEncoder encoder;

    // Load specs
    size_t n_specs;
    ifs >> n_specs;
    ifs.ignore();

    for (size_t i = 0; i < n_specs; ++i) {
        ColumnSpec spec;
        std::getline(ifs, spec.name);

        int type, source;
        ifs >> type >> source;
        spec.type = static_cast<EncodingType>(type);
        spec.source = static_cast<DataSource>(source);

        size_t n_cols;
        ifs >> n_cols;
        spec.columns.resize(n_cols);
        for (size_t j = 0; j < n_cols; ++j) {
            ifs >> spec.columns[j];
        }

        ifs >> spec.dim >> spec.top_k >> spec.bottom_k;
        ifs.ignore();
        std::getline(ifs, spec.rank_by);

        encoder.specs_.push_back(spec);
    }

    // Load vocabs
    size_t n_vocabs;
    ifs >> n_vocabs;
    ifs.ignore();

    for (size_t i = 0; i < n_vocabs; ++i) {
        std::string name;
        std::getline(ifs, name);
        encoder.vocabs_[name] = CategoryVocab::load(ifs);
    }

    // Load scalers
    size_t n_scalers;
    ifs >> n_scalers;
    ifs.ignore();

    for (size_t i = 0; i < n_scalers; ++i) {
        std::string name;
        std::getline(ifs, name);
        encoder.scalers_[name] = StandardScaler::load(ifs);
    }

    encoder.fitted_ = true;
    return encoder;
}

}  // namespace resolve
