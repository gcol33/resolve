#include "resolve/dataset.hpp"
#include "resolve/csv_reader.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <functional>

namespace resolve {

namespace {

// MurmurHash3 finalizer for feature hashing
uint32_t murmur_hash(const std::string& key, uint32_t seed = 0) {
    uint32_t h = seed;
    for (char c : key) {
        h ^= static_cast<uint32_t>(c);
        h *= 0x5bd1e995;
        h ^= h >> 15;
    }
    return h;
}

// Feature hashing for species
void hash_species(
    const std::vector<std::pair<std::string, float>>& species_abundances,
    float* embedding,
    int hash_dim
) {
    std::fill(embedding, embedding + hash_dim, 0.0f);

    for (const auto& [species, abundance] : species_abundances) {
        uint32_t h1 = murmur_hash(species, 0);
        uint32_t h2 = murmur_hash(species, 1);

        int idx = h1 % hash_dim;
        float sign = (h2 % 2 == 0) ? 1.0f : -1.0f;
        embedding[idx] += sign * abundance;
    }
}

// Select top-k species by abundance
std::vector<std::pair<std::string, float>> select_top_k(
    std::vector<std::pair<std::string, float>> species,
    int k
) {
    if (static_cast<int>(species.size()) <= k) {
        return species;
    }

    std::partial_sort(
        species.begin(),
        species.begin() + k,
        species.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; }
    );

    species.resize(k);
    return species;
}

// Select bottom-k species by abundance
std::vector<std::pair<std::string, float>> select_bottom_k(
    std::vector<std::pair<std::string, float>> species,
    int k
) {
    if (static_cast<int>(species.size()) <= k) {
        return species;
    }

    std::partial_sort(
        species.begin(),
        species.begin() + k,
        species.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    );

    species.resize(k);
    return species;
}

// Parse float safely
float safe_stof(const std::string& s, float default_val = 0.0f) {
    if (s.empty()) return default_val;
    try {
        return std::stof(s);
    } catch (...) {
        return default_val;
    }
}

// Parse int safely
int safe_stoi(const std::string& s, int default_val = 0) {
    if (s.empty()) return default_val;
    try {
        return std::stoi(s);
    } catch (...) {
        return default_val;
    }
}

} // anonymous namespace


ResolveDataset ResolveDataset::from_csv(
    const std::string& header_path,
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    ResolveDataset dataset;
    dataset.config_ = config;

    // Load header data (coordinates, covariates, targets)
    dataset.load_header_data(header_path, roles, targets);

    // Load species data
    dataset.load_species_data(species_path, roles);

    return dataset;
}

ResolveDataset ResolveDataset::from_species_csv(
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    ResolveDataset dataset;
    dataset.config_ = config;

    // Load species data and extract plot-level info
    CSVReader reader(species_path);

    // Find column indices
    int plot_col = reader.column_index(roles.plot_id);
    int species_col = reader.column_index(roles.species_id);

    if (plot_col < 0 || species_col < 0) {
        throw std::runtime_error("Required columns not found: plot_id or species_id");
    }

    int abundance_col = roles.abundance ? reader.column_index(*roles.abundance) : -1;
    int lon_col = roles.longitude ? reader.column_index(*roles.longitude) : -1;
    int lat_col = roles.latitude ? reader.column_index(*roles.latitude) : -1;
    int genus_col = roles.genus ? reader.column_index(*roles.genus) : -1;
    int family_col = roles.family ? reader.column_index(*roles.family) : -1;

    // Find target columns
    std::vector<int> target_cols;
    for (const auto& target : targets) {
        int col = reader.column_index(target.column_name);
        target_cols.push_back(col);
    }

    // First pass: collect all unique plots and their species
    std::unordered_map<std::string, std::vector<SpeciesRecord>> plot_records;
    std::unordered_map<std::string, std::pair<float, float>> plot_coords;
    std::unordered_map<std::string, std::vector<float>> plot_targets;
    std::unordered_set<std::string> seen_plots;

    reader.read_rows([&](size_t, const std::vector<std::string>& row) {
        if (row.size() <= static_cast<size_t>(std::max({plot_col, species_col}))) {
            return;  // Skip malformed rows
        }

        std::string plot_id = row[plot_col];
        std::string species_id = row[species_col];

        SpeciesRecord record;
        record.plot_id = plot_id;
        record.species_id = species_id;
        record.abundance = abundance_col >= 0 && row.size() > static_cast<size_t>(abundance_col)
            ? safe_stof(row[abundance_col], 1.0f) : 1.0f;

        if (genus_col >= 0 && row.size() > static_cast<size_t>(genus_col)) {
            record.genus = row[genus_col];
        }
        if (family_col >= 0 && row.size() > static_cast<size_t>(family_col)) {
            record.family = row[family_col];
        }

        plot_records[plot_id].push_back(record);

        // Extract plot-level data from first occurrence
        if (seen_plots.find(plot_id) == seen_plots.end()) {
            seen_plots.insert(plot_id);
            dataset.plot_ids_.push_back(plot_id);

            // Coordinates
            if (lon_col >= 0 && lat_col >= 0 &&
                row.size() > static_cast<size_t>(std::max(lon_col, lat_col))) {
                plot_coords[plot_id] = {
                    safe_stof(row[lon_col]),
                    safe_stof(row[lat_col])
                };
            }

            // Targets
            std::vector<float> target_values;
            for (size_t i = 0; i < target_cols.size(); ++i) {
                int col = target_cols[i];
                if (col >= 0 && row.size() > static_cast<size_t>(col)) {
                    target_values.push_back(safe_stof(row[col]));
                } else {
                    target_values.push_back(0.0f);
                }
            }
            plot_targets[plot_id] = target_values;
        }
    });

    int64_t n_plots = static_cast<int64_t>(dataset.plot_ids_.size());
    dataset.schema_.n_plots = n_plots;

    // Build coordinates tensor
    if (!plot_coords.empty()) {
        dataset.coordinates_ = torch::zeros({n_plots, 2}, torch::kFloat32);
        auto coords_acc = dataset.coordinates_.accessor<float, 2>();

        for (int64_t i = 0; i < n_plots; ++i) {
            const auto& plot_id = dataset.plot_ids_[i];
            auto it = plot_coords.find(plot_id);
            if (it != plot_coords.end()) {
                coords_acc[i][0] = it->second.first;
                coords_acc[i][1] = it->second.second;
            }
        }
        dataset.schema_.has_coordinates = true;
    }

    // Build target tensors
    for (size_t t = 0; t < targets.size(); ++t) {
        const auto& target_spec = targets[t];
        dataset.target_configs_.push_back({
            target_spec.target_name.empty() ? target_spec.column_name : target_spec.target_name,
            target_spec.task,
            target_spec.transform,
            target_spec.num_classes,
            target_spec.weight
        });

        torch::Tensor target_tensor;
        if (target_spec.task == TaskType::Classification) {
            target_tensor = torch::zeros({n_plots}, torch::kLong);
            auto acc = target_tensor.accessor<int64_t, 1>();
            for (int64_t i = 0; i < n_plots; ++i) {
                const auto& values = plot_targets[dataset.plot_ids_[i]];
                if (t < values.size()) {
                    acc[i] = static_cast<int64_t>(values[t]);
                }
            }
        } else {
            target_tensor = torch::zeros({n_plots}, torch::kFloat32);
            auto acc = target_tensor.accessor<float, 1>();
            for (int64_t i = 0; i < n_plots; ++i) {
                const auto& values = plot_targets[dataset.plot_ids_[i]];
                if (t < values.size()) {
                    acc[i] = values[t];
                }
            }
        }

        std::string name = target_spec.target_name.empty() ? target_spec.column_name : target_spec.target_name;
        dataset.targets_[name] = target_tensor;
    }

    dataset.schema_.targets = dataset.target_configs_;

    // Encode species data
    dataset.encode_species(plot_records);

    return dataset;
}

void ResolveDataset::load_header_data(
    const std::string& header_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets
) {
    CSVReader reader(header_path);

    // Find column indices
    int plot_col = reader.column_index(roles.plot_id);
    if (plot_col < 0) {
        throw std::runtime_error("Plot ID column not found: " + roles.plot_id);
    }

    int lon_col = roles.longitude ? reader.column_index(*roles.longitude) : -1;
    int lat_col = roles.latitude ? reader.column_index(*roles.latitude) : -1;

    std::vector<int> covariate_cols;
    for (const auto& cov : roles.covariates) {
        int col = reader.column_index(cov);
        if (col >= 0) {
            covariate_cols.push_back(col);
            schema_.covariate_names.push_back(cov);
        }
    }

    std::vector<int> target_cols;
    for (const auto& target : targets) {
        int col = reader.column_index(target.column_name);
        target_cols.push_back(col);

        target_configs_.push_back({
            target.target_name.empty() ? target.column_name : target.target_name,
            target.task,
            target.transform,
            target.num_classes,
            target.weight
        });
    }

    // Count rows first
    size_t n_rows = reader.count_rows();
    int64_t n_plots = static_cast<int64_t>(n_rows);
    schema_.n_plots = n_plots;

    // Allocate tensors
    plot_ids_.reserve(n_plots);

    if (lon_col >= 0 && lat_col >= 0) {
        coordinates_ = torch::zeros({n_plots, 2}, torch::kFloat32);
        schema_.has_coordinates = true;
    }

    if (!covariate_cols.empty()) {
        covariates_ = torch::zeros({n_plots, static_cast<int64_t>(covariate_cols.size())}, torch::kFloat32);
    }

    // Initialize target tensors
    for (size_t t = 0; t < targets.size(); ++t) {
        const auto& target = targets[t];
        std::string name = target.target_name.empty() ? target.column_name : target.target_name;

        if (target.task == TaskType::Classification) {
            targets_[name] = torch::zeros({n_plots}, torch::kLong);
        } else {
            targets_[name] = torch::zeros({n_plots}, torch::kFloat32);
        }
    }

    schema_.targets = target_configs_;

    // Read data
    auto coords_acc = coordinates_.defined() ? coordinates_.accessor<float, 2>() : torch::TensorAccessor<float, 2>(nullptr, nullptr, nullptr);
    auto cov_acc = covariates_.defined() ? covariates_.accessor<float, 2>() : torch::TensorAccessor<float, 2>(nullptr, nullptr, nullptr);

    int64_t row_idx = 0;
    reader.read_rows([&](size_t, const std::vector<std::string>& row) {
        if (row.size() <= static_cast<size_t>(plot_col)) {
            return;
        }

        plot_ids_.push_back(row[plot_col]);

        // Coordinates
        if (coordinates_.defined() && lon_col >= 0 && lat_col >= 0) {
            coords_acc[row_idx][0] = safe_stof(row[lon_col]);
            coords_acc[row_idx][1] = safe_stof(row[lat_col]);
        }

        // Covariates
        if (covariates_.defined()) {
            for (size_t i = 0; i < covariate_cols.size(); ++i) {
                int col = covariate_cols[i];
                if (row.size() > static_cast<size_t>(col)) {
                    cov_acc[row_idx][i] = safe_stof(row[col]);
                }
            }
        }

        // Targets
        for (size_t t = 0; t < targets.size(); ++t) {
            const auto& target = targets[t];
            std::string name = target.target_name.empty() ? target.column_name : target.target_name;
            int col = target_cols[t];

            if (col >= 0 && row.size() > static_cast<size_t>(col)) {
                if (target.task == TaskType::Classification) {
                    targets_[name][row_idx] = static_cast<int64_t>(safe_stoi(row[col]));
                } else {
                    targets_[name][row_idx] = safe_stof(row[col]);
                }
            }
        }

        row_idx++;
    });
}

void ResolveDataset::load_species_data(
    const std::string& species_path,
    const RoleMapping& roles
) {
    CSVReader reader(species_path);

    // Find column indices
    int plot_col = reader.column_index(roles.plot_id);
    int species_col = reader.column_index(roles.species_id);

    if (plot_col < 0 || species_col < 0) {
        throw std::runtime_error("Required columns not found: plot_id or species_id");
    }

    int abundance_col = roles.abundance ? reader.column_index(*roles.abundance) : -1;
    int genus_col = roles.genus ? reader.column_index(*roles.genus) : -1;
    int family_col = roles.family ? reader.column_index(*roles.family) : -1;

    // Collect species records by plot
    std::unordered_map<std::string, std::vector<SpeciesRecord>> plot_records;

    reader.read_rows([&](size_t, const std::vector<std::string>& row) {
        if (row.size() <= static_cast<size_t>(std::max(plot_col, species_col))) {
            return;
        }

        std::string plot_id = row[plot_col];
        std::string species_id = row[species_col];

        SpeciesRecord record;
        record.plot_id = plot_id;
        record.species_id = species_id;
        record.abundance = abundance_col >= 0 && row.size() > static_cast<size_t>(abundance_col)
            ? safe_stof(row[abundance_col], 1.0f) : 1.0f;

        if (genus_col >= 0 && row.size() > static_cast<size_t>(genus_col)) {
            record.genus = row[genus_col];
        }
        if (family_col >= 0 && row.size() > static_cast<size_t>(family_col)) {
            record.family = row[family_col];
        }

        plot_records[plot_id].push_back(record);
    });

    // Encode species
    encode_species(plot_records);
}

void ResolveDataset::build_species_vocab(
    const std::unordered_map<std::string, std::vector<std::pair<std::string, float>>>& plot_species
) {
    // Count species frequencies
    std::unordered_map<std::string, int> species_counts;

    for (const auto& [plot_id, species] : plot_species) {
        for (const auto& [sp, abundance] : species) {
            species_counts[sp]++;
        }
    }

    // Sort by frequency
    std::vector<std::pair<std::string, int>> sorted_species(
        species_counts.begin(), species_counts.end()
    );
    std::sort(sorted_species.begin(), sorted_species.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; }
    );

    // Build vocabulary
    species_vocab_.clear();
    species_vocab_.push_back("<UNK>");  // Index 0 for unknown
    species_to_idx_.clear();
    species_to_idx_["<UNK>"] = 0;

    for (const auto& [sp, count] : sorted_species) {
        species_to_idx_[sp] = static_cast<int64_t>(species_vocab_.size());
        species_vocab_.push_back(sp);
    }

    schema_.n_species = static_cast<int64_t>(species_counts.size());
    schema_.n_species_vocab = static_cast<int64_t>(species_vocab_.size());
}

void ResolveDataset::encode_species(
    const std::unordered_map<std::string, std::vector<SpeciesRecord>>& plot_records
) {
    int64_t n_plots = static_cast<int64_t>(plot_ids_.size());

    // Collect all records for taxonomy vocab
    std::vector<SpeciesRecord> all_records;
    for (const auto& [plot_id, records] : plot_records) {
        all_records.insert(all_records.end(), records.begin(), records.end());
    }

    // Check if we have taxonomy data
    bool has_genus = false;
    bool has_family = false;
    for (const auto& rec : all_records) {
        if (!rec.genus.empty()) has_genus = true;
        if (!rec.family.empty()) has_family = true;
        if (has_genus && has_family) break;
    }

    schema_.has_taxonomy = (has_genus || has_family) && config_.use_taxonomy;
    schema_.has_abundance = true;  // We always have abundance (even if defaulted to 1.0)

    // Fit taxonomy vocabulary
    if (schema_.has_taxonomy) {
        taxonomy_vocab_.fit(all_records);
        schema_.n_genera = taxonomy_vocab_.n_genera();
        schema_.n_families = taxonomy_vocab_.n_families();
        schema_.n_genera_vocab = taxonomy_vocab_.n_genera();
        schema_.n_families_vocab = taxonomy_vocab_.n_families();
    }

    // Build species vocabulary for embed/sparse modes
    std::unordered_map<std::string, std::vector<std::pair<std::string, float>>> plot_species;
    for (const auto& [plot_id, records] : plot_records) {
        for (const auto& rec : records) {
            plot_species[plot_id].push_back({rec.species_id, rec.abundance});
        }
    }
    build_species_vocab(plot_species);

    // Determine n_taxonomy_slots
    int n_taxonomy_slots = config_.top_k;
    if (config_.selection == SelectionMode::TopBottom) {
        n_taxonomy_slots = 2 * config_.top_k;
    }

    // Encode based on mode
    if (config_.species_encoding == SpeciesEncodingMode::Hash) {
        // Feature hashing mode
        hash_embedding_ = torch::zeros({n_plots, config_.hash_dim}, torch::kFloat32);
        auto hash_acc = hash_embedding_.accessor<float, 2>();

        for (int64_t i = 0; i < n_plots; ++i) {
            const auto& plot_id = plot_ids_[i];
            auto it = plot_records.find(plot_id);
            if (it == plot_records.end()) continue;

            // Convert to species-abundance pairs
            std::vector<std::pair<std::string, float>> species;
            for (const auto& rec : it->second) {
                species.push_back({rec.species_id, rec.abundance});
            }

            // Apply selection
            std::vector<std::pair<std::string, float>> selected;
            if (config_.selection == SelectionMode::Top) {
                selected = select_top_k(species, config_.top_k);
            } else if (config_.selection == SelectionMode::Bottom) {
                selected = select_bottom_k(species, config_.top_k);
            } else if (config_.selection == SelectionMode::TopBottom) {
                auto top = select_top_k(species, config_.top_k);
                auto bottom = select_bottom_k(species, config_.top_k);
                selected = top;
                for (const auto& s : bottom) {
                    if (std::find_if(selected.begin(), selected.end(),
                            [&s](const auto& x) { return x.first == s.first; }) == selected.end()) {
                        selected.push_back(s);
                    }
                }
            } else {
                selected = species;
            }

            // Apply normalization
            if (config_.normalization == NormalizationMode::Norm) {
                float total = 0.0f;
                for (const auto& [sp, ab] : selected) total += ab;
                if (total > 0) {
                    for (auto& [sp, ab] : selected) ab /= total;
                }
            } else if (config_.normalization == NormalizationMode::Log1p) {
                for (auto& [sp, ab] : selected) ab = std::log1p(ab);
            }

            // Hash
            hash_species(selected, &hash_acc[i][0], config_.hash_dim);
        }

    } else if (config_.species_encoding == SpeciesEncodingMode::Embed) {
        // Learnable embeddings for top-k species
        species_ids_ = torch::zeros({n_plots, config_.top_k_species}, torch::kLong);
        auto ids_acc = species_ids_.accessor<int64_t, 2>();

        for (int64_t i = 0; i < n_plots; ++i) {
            const auto& plot_id = plot_ids_[i];
            auto it = plot_records.find(plot_id);
            if (it == plot_records.end()) continue;

            std::vector<std::pair<std::string, float>> species;
            for (const auto& rec : it->second) {
                species.push_back({rec.species_id, rec.abundance});
            }

            auto selected = select_top_k(species, config_.top_k_species);

            for (size_t j = 0; j < selected.size() && j < static_cast<size_t>(config_.top_k_species); ++j) {
                auto sp_it = species_to_idx_.find(selected[j].first);
                ids_acc[i][j] = sp_it != species_to_idx_.end() ? sp_it->second : 0;
            }
        }

    } else {
        // Sparse/explicit vector mode
        species_vector_ = torch::zeros({n_plots, schema_.n_species_vocab}, torch::kFloat32);
        auto vec_acc = species_vector_.accessor<float, 2>();

        for (int64_t i = 0; i < n_plots; ++i) {
            const auto& plot_id = plot_ids_[i];
            auto it = plot_records.find(plot_id);
            if (it == plot_records.end()) continue;

            for (const auto& rec : it->second) {
                auto sp_it = species_to_idx_.find(rec.species_id);
                if (sp_it != species_to_idx_.end()) {
                    float value = rec.abundance;
                    if (config_.representation == RepresentationMode::PresenceAbsence) {
                        value = 1.0f;
                    }
                    vec_acc[i][sp_it->second] = value;
                }
            }
        }
    }

    // Encode taxonomy
    if (schema_.has_taxonomy) {
        genus_ids_ = torch::zeros({n_plots, n_taxonomy_slots}, torch::kLong);
        family_ids_ = torch::zeros({n_plots, n_taxonomy_slots}, torch::kLong);
        auto genus_acc = genus_ids_.accessor<int64_t, 2>();
        auto family_acc = family_ids_.accessor<int64_t, 2>();

        for (int64_t i = 0; i < n_plots; ++i) {
            const auto& plot_id = plot_ids_[i];
            auto it = plot_records.find(plot_id);
            if (it == plot_records.end()) continue;

            // Get sorted records by abundance
            auto records = it->second;
            std::sort(records.begin(), records.end(),
                [](const auto& a, const auto& b) { return a.abundance > b.abundance; }
            );

            // Fill taxonomy slots
            int slot = 0;
            for (const auto& rec : records) {
                if (slot >= n_taxonomy_slots) break;
                genus_acc[i][slot] = taxonomy_vocab_.encode_genus(rec.genus);
                family_acc[i][slot] = taxonomy_vocab_.encode_family(rec.family);
                slot++;
            }
        }
    }

    // Unknown fraction/count tracking
    if (config_.track_unknown_fraction) {
        unknown_fraction_ = torch::zeros({n_plots}, torch::kFloat32);
        // For now, assume all species are known (would need external vocab to track unknowns)
    }

    if (config_.track_unknown_count) {
        unknown_count_ = torch::zeros({n_plots}, torch::kFloat32);
    }
}

} // namespace resolve
