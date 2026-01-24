#pragma once

#include "types.hpp"
#include "vocab.hpp"

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace resolve {

// Forward declaration for CSV parsing (could use rapidcsv or custom)
namespace csv {

struct DataFrame {
    std::vector<std::string> columns;
    std::unordered_map<std::string, std::vector<std::string>> string_data;
    std::unordered_map<std::string, std::vector<double>> numeric_data;
    size_t n_rows = 0;

    bool has_column(const std::string& col) const {
        return string_data.count(col) > 0 || numeric_data.count(col) > 0;
    }

    const std::vector<std::string>& get_string(const std::string& col) const {
        return string_data.at(col);
    }

    const std::vector<double>& get_numeric(const std::string& col) const {
        return numeric_data.at(col);
    }

    std::vector<double> get_as_numeric(const std::string& col) const {
        if (numeric_data.count(col)) {
            return numeric_data.at(col);
        }
        // Convert string to numeric
        std::vector<double> result;
        result.reserve(n_rows);
        for (const auto& s : string_data.at(col)) {
            if (s.empty()) {
                result.push_back(std::nan(""));
            } else {
                try {
                    result.push_back(std::stod(s));
                } catch (...) {
                    result.push_back(std::nan(""));
                }
            }
        }
        return result;
    }

    static DataFrame from_csv(const std::string& path);
};

} // namespace csv


/**
 * ResolveDataset: Validated container for ecological plot data.
 *
 * Holds header (plot-level) and species (occurrence) data with semantic
 * role mappings. Validates structure and provides train/test splitting.
 *
 * This is the C++ equivalent of Python's ResolveDataset class.
 */
class ResolveDataset {
public:
    /**
     * Construct from raw data.
     */
    ResolveDataset(
        csv::DataFrame header,
        csv::DataFrame species,
        RoleMapping roles,
        std::vector<TargetConfig> targets,
        NormalizationMode species_normalization = NormalizationMode::Norm,
        bool track_unknown_fraction = true,
        bool track_unknown_count = false
    ) : header_(std::move(header)),
        species_(std::move(species)),
        roles_(std::move(roles)),
        targets_(std::move(targets)),
        species_normalization_(species_normalization),
        track_unknown_fraction_(track_unknown_fraction),
        track_unknown_count_(track_unknown_count)
    {
        validate();
        build_index();
    }

    /**
     * Load dataset from CSV files.
     */
    static ResolveDataset from_csv(
        const std::string& header_path,
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetConfig>& targets,
        NormalizationMode species_normalization = NormalizationMode::Norm,
        bool track_unknown_fraction = true,
        bool track_unknown_count = false
    ) {
        auto header = csv::DataFrame::from_csv(header_path);
        auto species = csv::DataFrame::from_csv(species_path);
        return ResolveDataset(
            std::move(header),
            std::move(species),
            roles,
            targets,
            species_normalization,
            track_unknown_fraction,
            track_unknown_count
        );
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    size_t n_plots() const { return header_.n_rows; }

    const RoleMapping& roles() const { return roles_; }

    const std::vector<TargetConfig>& targets() const { return targets_; }

    NormalizationMode species_normalization() const { return species_normalization_; }

    bool track_unknown_fraction() const { return track_unknown_fraction_; }

    bool track_unknown_count() const { return track_unknown_count_; }

    std::vector<std::string> plot_ids() const {
        return header_.get_string(roles_.plot_id);
    }

    /**
     * Get schema derived from dataset.
     */
    ResolveSchema schema() const {
        ResolveSchema s;
        s.n_plots = static_cast<int64_t>(n_plots());
        s.n_species = count_unique_species();
        s.has_coordinates = roles_.has_coordinates();
        s.has_abundance = roles_.has_abundance();
        s.has_taxonomy = roles_.has_taxonomy();

        if (s.has_taxonomy) {
            s.n_genera = count_unique(roles_.taxonomy_genus.value());
            s.n_families = count_unique(roles_.taxonomy_family.value());
        }

        // n_continuous: coordinates (if present) + covariates
        int n_coords = s.has_coordinates ? 2 : 0;
        s.n_continuous = n_coords + static_cast<int64_t>(roles_.covariates.size());
        s.covariate_names = roles_.covariates;
        s.targets = targets_;
        s.species_normalization = species_normalization_;
        s.track_unknown_fraction = track_unknown_fraction_;
        s.track_unknown_count = track_unknown_count_;

        return s;
    }

    // ========================================================================
    // Data extraction
    // ========================================================================

    /**
     * Get coordinates as (n_plots, 2) tensor [lat, lon].
     * Returns empty tensor if no coordinates.
     */
    torch::Tensor get_coordinates() const {
        if (!roles_.has_coordinates()) {
            return torch::Tensor();
        }

        auto lat = header_.get_as_numeric(roles_.coords_lat.value());
        auto lon = header_.get_as_numeric(roles_.coords_lon.value());

        auto tensor = torch::zeros({static_cast<int64_t>(n_plots()), 2}, torch::kFloat32);
        for (size_t i = 0; i < n_plots(); ++i) {
            tensor[i][0] = std::isnan(lat[i]) ? 0.0f : static_cast<float>(lat[i]);
            tensor[i][1] = std::isnan(lon[i]) ? 0.0f : static_cast<float>(lon[i]);
        }
        return tensor;
    }

    /**
     * Get covariates as (n_plots, n_covariates) tensor.
     * Returns empty tensor if no covariates.
     */
    torch::Tensor get_covariates() const {
        if (roles_.covariates.empty()) {
            return torch::Tensor();
        }

        int64_t n_cov = static_cast<int64_t>(roles_.covariates.size());
        auto tensor = torch::zeros({static_cast<int64_t>(n_plots()), n_cov}, torch::kFloat32);

        for (int64_t c = 0; c < n_cov; ++c) {
            auto values = header_.get_as_numeric(roles_.covariates[c]);
            for (size_t i = 0; i < n_plots(); ++i) {
                tensor[i][c] = std::isnan(values[i]) ? 0.0f : static_cast<float>(values[i]);
            }
        }
        return tensor;
    }

    /**
     * Get target array by name.
     * For regression: float tensor with optional log1p transform.
     * For classification: int64 tensor with category codes.
     */
    torch::Tensor get_target(const std::string& name) const {
        const auto* cfg = find_target(name);
        if (!cfg) {
            throw std::runtime_error("Unknown target: " + name);
        }

        auto values = header_.get_as_numeric(cfg->column);

        if (cfg->task == TaskType::Regression) {
            auto tensor = torch::zeros({static_cast<int64_t>(n_plots())}, torch::kFloat32);
            for (size_t i = 0; i < n_plots(); ++i) {
                float val = static_cast<float>(values[i]);
                if (cfg->transform == TransformType::Log1p && !std::isnan(val)) {
                    val = std::log1p(val);
                }
                tensor[i] = val;
            }
            return tensor;
        } else {
            // Classification: encode as integers
            auto str_values = header_.get_string(cfg->column);
            std::unordered_map<std::string, int64_t> encoding;
            int64_t next_code = 0;

            auto tensor = torch::zeros({static_cast<int64_t>(n_plots())}, torch::kInt64);
            for (size_t i = 0; i < n_plots(); ++i) {
                const auto& v = str_values[i];
                if (encoding.find(v) == encoding.end()) {
                    encoding[v] = next_code++;
                }
                tensor[i] = encoding[v];
            }
            return tensor;
        }
    }

    /**
     * Get boolean mask for non-null target values.
     */
    torch::Tensor get_target_mask(const std::string& name) const {
        const auto* cfg = find_target(name);
        if (!cfg) {
            throw std::runtime_error("Unknown target: " + name);
        }

        auto values = header_.get_as_numeric(cfg->column);
        auto tensor = torch::zeros({static_cast<int64_t>(n_plots())}, torch::kBool);
        for (size_t i = 0; i < n_plots(); ++i) {
            tensor[i] = !std::isnan(values[i]);
        }
        return tensor;
    }

    /**
     * Get species data for a specific plot.
     */
    struct PlotSpecies {
        std::vector<std::string> species_ids;
        std::vector<double> abundances;
        std::vector<std::string> genera;
        std::vector<std::string> families;
    };

    PlotSpecies get_species_for_plot(const std::string& plot_id) const {
        PlotSpecies result;
        auto it = plot_species_index_.find(plot_id);
        if (it == plot_species_index_.end()) {
            return result;
        }

        const auto& species_ids_col = species_.get_string(roles_.species_id);
        const std::vector<double>* abundance_col = nullptr;
        if (roles_.has_abundance()) {
            abundance_col = &species_.get_numeric(roles_.abundance.value());
        }
        const std::vector<std::string>* genus_col = nullptr;
        const std::vector<std::string>* family_col = nullptr;
        if (roles_.has_taxonomy()) {
            genus_col = &species_.get_string(roles_.taxonomy_genus.value());
            family_col = &species_.get_string(roles_.taxonomy_family.value());
        }

        for (size_t idx : it->second) {
            result.species_ids.push_back(species_ids_col[idx]);
            if (abundance_col) {
                result.abundances.push_back((*abundance_col)[idx]);
            } else {
                result.abundances.push_back(1.0);  // Presence-absence
            }
            if (genus_col) {
                result.genera.push_back((*genus_col)[idx]);
                result.families.push_back((*family_col)[idx]);
            }
        }
        return result;
    }

    // ========================================================================
    // Splitting
    // ========================================================================

    /**
     * Split into train and test datasets.
     */
    std::pair<ResolveDataset, ResolveDataset> split(
        double test_size = 0.2,
        uint64_t seed = 42
    ) const {
        auto ids = plot_ids();
        std::vector<size_t> indices(ids.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Shuffle
        std::mt19937_64 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);

        // Split
        size_t n_test = static_cast<size_t>(ids.size() * test_size);
        size_t n_train = ids.size() - n_test;

        std::unordered_set<std::string> train_ids, test_ids;
        for (size_t i = 0; i < n_train; ++i) {
            train_ids.insert(ids[indices[i]]);
        }
        for (size_t i = n_train; i < ids.size(); ++i) {
            test_ids.insert(ids[indices[i]]);
        }

        return {
            filter_by_plot_ids(train_ids),
            filter_by_plot_ids(test_ids)
        };
    }

    /**
     * Filter dataset to rows with non-null target values.
     */
    ResolveDataset filter_by_target(const std::string& name) const {
        auto mask = get_target_mask(name);
        auto ids = plot_ids();

        std::unordered_set<std::string> valid_ids;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (mask[i].item<bool>()) {
                valid_ids.insert(ids[i]);
            }
        }

        return filter_by_plot_ids(valid_ids);
    }

private:
    csv::DataFrame header_;
    csv::DataFrame species_;
    RoleMapping roles_;
    std::vector<TargetConfig> targets_;
    NormalizationMode species_normalization_;
    bool track_unknown_fraction_;
    bool track_unknown_count_;

    // Index: plot_id -> row indices in species_
    std::unordered_map<std::string, std::vector<size_t>> plot_species_index_;

    void validate() const {
        roles_.validate();

        // Check required columns in header
        if (!header_.has_column(roles_.plot_id)) {
            throw std::invalid_argument("Missing plot_id column: " + roles_.plot_id);
        }
        if (roles_.has_coordinates()) {
            if (!header_.has_column(roles_.coords_lat.value())) {
                throw std::invalid_argument("Missing coords_lat column");
            }
            if (!header_.has_column(roles_.coords_lon.value())) {
                throw std::invalid_argument("Missing coords_lon column");
            }
        }

        // Check target columns
        for (const auto& cfg : targets_) {
            if (!header_.has_column(cfg.column)) {
                throw std::invalid_argument("Target column not found: " + cfg.column);
            }
        }

        // Check covariates
        for (const auto& cov : roles_.covariates) {
            if (!header_.has_column(cov)) {
                throw std::invalid_argument("Covariate not found: " + cov);
            }
        }

        // Check species columns
        if (!species_.has_column(roles_.species_id)) {
            throw std::invalid_argument("Missing species_id column");
        }
        if (!species_.has_column(roles_.species_plot_id)) {
            throw std::invalid_argument("Missing species_plot_id column");
        }
    }

    void build_index() {
        const auto& plot_ids = species_.get_string(roles_.species_plot_id);
        for (size_t i = 0; i < plot_ids.size(); ++i) {
            plot_species_index_[plot_ids[i]].push_back(i);
        }
    }

    const TargetConfig* find_target(const std::string& name) const {
        for (const auto& cfg : targets_) {
            if (cfg.name == name) {
                return &cfg;
            }
        }
        return nullptr;
    }

    int64_t count_unique_species() const {
        const auto& ids = species_.get_string(roles_.species_id);
        std::unordered_set<std::string> unique(ids.begin(), ids.end());
        return static_cast<int64_t>(unique.size());
    }

    int64_t count_unique(const std::string& col) const {
        const auto& values = species_.get_string(col);
        std::unordered_set<std::string> unique(values.begin(), values.end());
        unique.erase("");  // Don't count empty strings
        return static_cast<int64_t>(unique.size());
    }

    ResolveDataset filter_by_plot_ids(const std::unordered_set<std::string>& valid_ids) const {
        // Filter header
        csv::DataFrame new_header;
        new_header.columns = header_.columns;
        const auto& header_ids = header_.get_string(roles_.plot_id);

        std::vector<size_t> header_indices;
        for (size_t i = 0; i < header_ids.size(); ++i) {
            if (valid_ids.count(header_ids[i])) {
                header_indices.push_back(i);
            }
        }
        new_header.n_rows = header_indices.size();

        for (const auto& col : header_.columns) {
            if (header_.string_data.count(col)) {
                const auto& src = header_.string_data.at(col);
                auto& dst = new_header.string_data[col];
                dst.reserve(header_indices.size());
                for (size_t i : header_indices) {
                    dst.push_back(src[i]);
                }
            }
            if (header_.numeric_data.count(col)) {
                const auto& src = header_.numeric_data.at(col);
                auto& dst = new_header.numeric_data[col];
                dst.reserve(header_indices.size());
                for (size_t i : header_indices) {
                    dst.push_back(src[i]);
                }
            }
        }

        // Filter species
        csv::DataFrame new_species;
        new_species.columns = species_.columns;
        const auto& species_plot_ids = species_.get_string(roles_.species_plot_id);

        std::vector<size_t> species_indices;
        for (size_t i = 0; i < species_plot_ids.size(); ++i) {
            if (valid_ids.count(species_plot_ids[i])) {
                species_indices.push_back(i);
            }
        }
        new_species.n_rows = species_indices.size();

        for (const auto& col : species_.columns) {
            if (species_.string_data.count(col)) {
                const auto& src = species_.string_data.at(col);
                auto& dst = new_species.string_data[col];
                dst.reserve(species_indices.size());
                for (size_t i : species_indices) {
                    dst.push_back(src[i]);
                }
            }
            if (species_.numeric_data.count(col)) {
                const auto& src = species_.numeric_data.at(col);
                auto& dst = new_species.numeric_data[col];
                dst.reserve(species_indices.size());
                for (size_t i : species_indices) {
                    dst.push_back(src[i]);
                }
            }
        }

        return ResolveDataset(
            std::move(new_header),
            std::move(new_species),
            roles_,
            targets_,
            species_normalization_,
            track_unknown_fraction_,
            track_unknown_count_
        );
    }
};

// ============================================================================
// CSV Implementation (simple version, could be replaced with rapidcsv)
// ============================================================================

namespace csv {

inline DataFrame DataFrame::from_csv(const std::string& path) {
    DataFrame df;
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open CSV file: " + path);
    }

    std::string line;

    // Read header
    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty CSV file: " + path);
    }

    std::stringstream header_ss(line);
    std::string col;
    while (std::getline(header_ss, col, ',')) {
        // Remove quotes if present
        if (!col.empty() && col.front() == '"' && col.back() == '"') {
            col = col.substr(1, col.size() - 2);
        }
        df.columns.push_back(col);
        df.string_data[col] = {};
    }

    // Read data rows
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream row_ss(line);
        size_t col_idx = 0;
        std::string value;

        while (std::getline(row_ss, value, ',') && col_idx < df.columns.size()) {
            // Remove quotes if present
            if (!value.empty() && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.size() - 2);
            }
            df.string_data[df.columns[col_idx]].push_back(value);
            col_idx++;
        }

        // Fill remaining columns with empty strings
        while (col_idx < df.columns.size()) {
            df.string_data[df.columns[col_idx]].push_back("");
            col_idx++;
        }

        df.n_rows++;
    }

    // Try to detect numeric columns and convert
    for (const auto& col_name : df.columns) {
        auto& str_col = df.string_data[col_name];
        bool is_numeric = true;

        for (const auto& val : str_col) {
            if (val.empty()) continue;
            try {
                std::stod(val);
            } catch (...) {
                is_numeric = false;
                break;
            }
        }

        if (is_numeric) {
            auto& num_col = df.numeric_data[col_name];
            num_col.reserve(str_col.size());
            for (const auto& val : str_col) {
                if (val.empty()) {
                    num_col.push_back(std::nan(""));
                } else {
                    num_col.push_back(std::stod(val));
                }
            }
        }
    }

    return df;
}

} // namespace csv

} // namespace resolve
