#pragma once

#include "resolve/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
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

    // Multiple covariates (treated as continuous, standardized at fit time)
    std::vector<std::string> covariates;

    // Multiple categorical covariates. These are columns of strings (or
    // string-like integers) that get auto-factorized at CSV-load time
    // (CategoricalVocab, code 0 reserved for unknown/NA) and embedded via
    // CategoricalEmbedder in the encoder. Disjoint from `covariates` — a
    // column listed here MUST NOT also appear in `covariates`.
    std::vector<std::string> categoricals;

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

    // Helper to check if any categorical covariates were declared
    bool has_categoricals() const {
        return !categoricals.empty();
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

    // Optional explicit string->int class mapping for classification targets.
    // Mirrors the Python POC's `cfg["mapping"]`:
    //   - empty (default): the loader auto-fits the mapping from the data.
    //                      If every unique non-NA value is integer-parseable
    //                      (e.g. "0".."8"), the parsed integers are used as
    //                      codes (keeps already-encoded columns byte-stable).
    //                      Otherwise, sorted-unique non-NA values are
    //                      factorized to 0..K-1.
    //   - non-empty: applied verbatim. Values not present in the mapping
    //                are treated as missing (row dropped, same as NA).
    // NA-like strings ("", "NA", "NaN", ".", "-", "None", ...) always become
    // missing and drive the same row-drop path as a missing regression target.
    // After load, ResolveDataset's TargetConfig.class_names holds the
    // resulting ordered vocab, and the dataset's target tensor is int64.
    std::unordered_map<std::string, int64_t> class_mapping;

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

    // Classification target with an explicit user-provided string->int mapping.
    // `num_classes` is derived from the mapping size (the +0 baseline assumes
    // codes are dense 0..K-1, which is the convention the auto-fit path also
    // produces). Use the regular `classification(column, num_classes)` factory
    // when the column is already encoded as integers.
    static TargetSpec classification_with_mapping(
        const std::string& column,
        const std::unordered_map<std::string, int64_t>& mapping
    ) {
        TargetSpec spec;
        spec.column_name = column;
        spec.target_name = column;
        spec.task = TaskType::Classification;
        spec.num_classes = static_cast<int>(mapping.size());
        spec.class_mapping = mapping;
        return spec;
    }
};

} // namespace resolve
