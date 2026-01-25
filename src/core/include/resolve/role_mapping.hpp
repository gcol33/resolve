#pragma once

#include "resolve/types.hpp"
#include <string>
#include <vector>
#include <optional>

namespace resolve {

// Column roles for CSV data
enum class ColumnRole {
    PlotId,         // Unique identifier for plots
    SpeciesId,      // Species identifier (name or code)
    Abundance,      // Species abundance value
    Longitude,      // X coordinate
    Latitude,       // Y coordinate
    Genus,          // Genus name for taxonomy
    Family,         // Family name for taxonomy
    Covariate,      // Additional covariates
    Target,         // Prediction target (area, habitat, etc.)
    Ignore          // Ignore this column
};

// Mapping between column names and their roles
struct RoleMapping {
    // Required columns
    std::string plot_id;
    std::string species_id;

    // Optional columns
    std::optional<std::string> abundance;
    std::optional<std::string> longitude;
    std::optional<std::string> latitude;
    std::optional<std::string> genus;
    std::optional<std::string> family;

    // Multiple covariates
    std::vector<std::string> covariates;

    // Multiple targets
    std::vector<std::string> targets;

    // Helper to check if coordinates are available
    bool has_coordinates() const {
        return longitude.has_value() && latitude.has_value();
    }

    // Helper to check if taxonomy is available
    bool has_taxonomy() const {
        return genus.has_value() || family.has_value();
    }

    // Helper to check if abundance is available
    bool has_abundance() const {
        return abundance.has_value();
    }
};

// Target specification for loading
struct TargetSpec {
    std::string column_name;
    std::string target_name;  // Name to use in model (defaults to column_name)
    TaskType task = TaskType::Regression;
    TransformType transform = TransformType::None;
    int num_classes = 0;  // For classification tasks
    float weight = 1.0f;

    // Convenience constructors
    static TargetSpec regression(const std::string& column, TransformType transform = TransformType::None) {
        TargetSpec spec;
        spec.column_name = column;
        spec.target_name = column;
        spec.task = TaskType::Regression;
        spec.transform = transform;
        return spec;
    }

    static TargetSpec classification(const std::string& column, int num_classes) {
        TargetSpec spec;
        spec.column_name = column;
        spec.target_name = column;
        spec.task = TaskType::Classification;
        spec.num_classes = num_classes;
        return spec;
    }
};

} // namespace resolve
