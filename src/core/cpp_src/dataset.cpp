#include "resolve/dataset.hpp"
#include "resolve/csv_reader.hpp"
#include "resolve/csv_utils.hpp"
#include "resolve/species_encoding.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <numeric>
#include <cmath>
#include <iostream>
#include <functional>
#include <optional>

namespace resolve {

namespace {

// A cell is "missing" if it is empty or matches a common NA marker.
// Match is case-insensitive on the marker words. Mirrors the Python POC's
// `_NA_STRINGS` ({"", "NA", "na", "N/A", "n/a", "NaN", "nan", "NULL", "null",
// "None", "none", ".", "-"}) — kept in sync so the C++ and POC loaders drop
// the same rows.
bool is_missing_cell(const std::string& s) {
    if (s.empty()) return true;
    if (s == "." || s == "-") return true;
    auto eq_ci = [&](const char* tok) {
        if (s.size() != std::strlen(tok)) return false;
        for (size_t i = 0; i < s.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(s[i])) !=
                std::tolower(static_cast<unsigned char>(tok[i]))) return false;
        }
        return true;
    };
    return eq_ci("NA") || eq_ci("N/A") || eq_ci("NaN") ||
           eq_ci("NULL") || eq_ci("None");
}

// Parse a regression target string. Returns nullopt for empty / NA marker /
// unparseable / non-finite values. Distinct from `safe_stof`, which silently
// maps every failure to 0.0f and would conflate "missing" with "actual zero".
std::optional<float> parse_regression_target(const std::string& s) {
    if (is_missing_cell(s)) return std::nullopt;
    try {
        float v = std::stof(s);
        if (!std::isfinite(v)) return std::nullopt;
        return v;
    } catch (...) {
        return std::nullopt;
    }
}

// Try to parse a string as a strict signed integer. Returns nullopt on any
// failure (empty, non-numeric chars, overflow, leading/trailing spaces, ...).
// Used by the classification auto-fit path: if every unique non-NA value of
// a classification column parses as an int, those parsed ints are used as
// codes directly. Keeps already-integer-encoded columns (e.g. "0".."8")
// byte-stable across load/save and matches the POC's behaviour exactly.
std::optional<int64_t> parse_strict_int64(const std::string& s) {
    if (s.empty()) return std::nullopt;
    try {
        size_t pos = 0;
        int64_t v = std::stoll(s, &pos);
        if (pos != s.size()) return std::nullopt;  // trailing garbage
        return v;
    } catch (...) {
        return std::nullopt;
    }
}

// Auto-fit a string->int class mapping over the raw cells of one classification
// column. Mirrors `_encode_categorical(raw, mapping=None)` in the Python POC:
//   - NA-like cells contribute nothing to the mapping.
//   - If every distinct non-NA cell parses as a base-10 integer, the parsed
//     integers are used as the codes (preserves "0".."8" encodings exactly).
//   - Otherwise the sorted-unique non-NA values are factorized to 0..K-1
//     (sorting is lexicographic on the raw strings, matching Python's sorted()).
// Returns:
//   - mapping_out : string -> int64 code
//   - class_names : ordered vocab (class_names[code] == original string)
void fit_classification_mapping(
    const std::vector<std::string>& raw,
    std::unordered_map<std::string, int64_t>& mapping_out,
    std::vector<std::string>& class_names_out
) {
    mapping_out.clear();
    class_names_out.clear();

    std::vector<std::string> uniques;
    {
        std::unordered_set<std::string> seen;
        seen.reserve(raw.size());
        for (const auto& v : raw) {
            if (is_missing_cell(v)) continue;
            if (seen.insert(v).second) uniques.push_back(v);
        }
    }
    if (uniques.empty()) return;

    std::sort(uniques.begin(), uniques.end());

    // Try strict-int path first.
    std::vector<int64_t> parsed;
    parsed.reserve(uniques.size());
    bool all_int = true;
    for (const auto& v : uniques) {
        auto p = parse_strict_int64(v);
        if (!p) { all_int = false; break; }
        parsed.push_back(*p);
    }

    if (all_int) {
        // Use parsed ints as codes. class_names is ordered by code (so the
        // vocab round-trips through save/load), which means we need to sort
        // (code, name) pairs by code and dedupe to a dense vector.
        std::vector<std::pair<int64_t, std::string>> pairs;
        pairs.reserve(uniques.size());
        for (size_t i = 0; i < uniques.size(); ++i) {
            mapping_out[uniques[i]] = parsed[i];
            pairs.emplace_back(parsed[i], uniques[i]);
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        // class_names is sized so that class_names[code] == original string.
        // When codes aren't dense (e.g. {0, 2, 5}), the gaps stay empty.
        int64_t max_code = pairs.back().first;
        class_names_out.resize(static_cast<size_t>(max_code + 1));
        for (const auto& [c, n] : pairs) {
            class_names_out[static_cast<size_t>(c)] = n;
        }
    } else {
        for (size_t i = 0; i < uniques.size(); ++i) {
            mapping_out[uniques[i]] = static_cast<int64_t>(i);
        }
        class_names_out = uniques;  // index = code
    }
}

}  // namespace


// ColumnIndices implementation
ColumnIndices ColumnIndices::from_reader(const CSVReader& reader, const RoleMapping& roles) {
    ColumnIndices idx;
    idx.plot = reader.column_index(roles.plot_id);
    idx.species = reader.column_index(roles.species_id);
    idx.abundance = roles.abundance ? reader.column_index(*roles.abundance) : -1;
    idx.longitude = roles.longitude ? reader.column_index(*roles.longitude) : -1;
    idx.latitude = roles.latitude ? reader.column_index(*roles.latitude) : -1;
    idx.genus = roles.genus ? reader.column_index(*roles.genus) : -1;
    idx.family = roles.family ? reader.column_index(*roles.family) : -1;
    return idx;
}


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

ResolveDataset ResolveDataset::from_csv_with_schema(
    const std::string& header_path,
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const ResolveDataset& schema_source,
    const DatasetConfig& config
) {
    ResolveDataset dataset;
    dataset.config_ = config;
    dataset.use_external_vocabs_ = true;

    // Copy the source's fitted vocabularies into the new dataset. The
    // load_*/encode_species paths see use_external_vocabs_=true and skip
    // their own fit calls, falling back to these pre-populated members for
    // encoding. Any value not in the source's vocab encodes as 0 (UNK) via
    // the existing encode paths.
    dataset.categorical_vocab_ = schema_source.categorical_vocab_;
    dataset.taxonomy_vocab_ = schema_source.taxonomy_vocab_;
    dataset.species_vocab_ = schema_source.species_vocab_;
    dataset.species_to_idx_ = schema_source.species_to_idx_;

    // Classification targets need their class mappings replayed from the
    // source's class_names so the explicit-mapping branch in load_header_data
    // is taken (instead of fit_classification_mapping). The caller is not
    // required to populate TargetSpec.class_mapping; we do it here from
    // schema_source's target configs.
    std::vector<TargetSpec> targets_with_mappings = targets;
    for (auto& spec : targets_with_mappings) {
        if (spec.task != TaskType::Classification) continue;
        if (!spec.class_mapping.empty()) continue;  // caller already set one
        const std::string name = spec.target_name.empty()
            ? spec.column_name
            : spec.target_name;
        for (const auto& src_cfg : schema_source.target_configs_) {
            if (src_cfg.name != name) continue;
            for (size_t i = 0; i < src_cfg.class_names.size(); ++i) {
                if (!src_cfg.class_names[i].empty()) {
                    spec.class_mapping[src_cfg.class_names[i]] = static_cast<int64_t>(i);
                }
            }
            if (spec.num_classes == 0) spec.num_classes = src_cfg.num_classes;
            break;
        }
    }

    dataset.load_header_data(header_path, roles, targets_with_mappings);
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
    auto cols = ColumnIndices::from_reader(reader, roles);
    if (cols.plot < 0 || cols.species < 0) {
        throw std::runtime_error("Required columns not found: plot_id or species_id");
    }

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
        if (row.size() <= static_cast<size_t>(std::max({cols.plot, cols.species}))) {
            return;  // Skip malformed rows
        }

        std::string plot_id = row[cols.plot];
        std::string species_id = row[cols.species];

        SpeciesRecord record;
        record.plot_id = plot_id;
        record.species_id = species_id;
        record.abundance = cols.abundance >= 0 && row.size() > static_cast<size_t>(cols.abundance)
            ? safe_stof(row[cols.abundance], 1.0f) : 1.0f;

        if (cols.genus >= 0 && row.size() > static_cast<size_t>(cols.genus)) {
            record.genus = row[cols.genus];
        }
        if (cols.family >= 0 && row.size() > static_cast<size_t>(cols.family)) {
            record.family = row[cols.family];
        }

        plot_records[plot_id].push_back(record);

        // Extract plot-level data from first occurrence
        if (seen_plots.find(plot_id) == seen_plots.end()) {
            seen_plots.insert(plot_id);
            dataset.plot_ids_.push_back(plot_id);

            // Coordinates
            if (cols.longitude >= 0 && cols.latitude >= 0 &&
                row.size() > static_cast<size_t>(std::max(cols.longitude, cols.latitude))) {
                plot_coords[plot_id] = {
                    safe_stof(row[cols.longitude]),
                    safe_stof(row[cols.latitude])
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

    // ---- Categorical covariates ----
    // Each column listed in roles.categoricals must (a) exist in the header
    // CSV, (b) be disjoint from roles.covariates so the user can't double-
    // count the same column. We collect raw strings during the row scan
    // below, then fit the vocab + encode after the count_rows pass so the
    // tensor allocation matches the actual filtered row count.
    {
        std::unordered_set<std::string> cov_set(roles.covariates.begin(),
                                                roles.covariates.end());
        for (const auto& cat : roles.categoricals) {
            if (cov_set.count(cat)) {
                throw std::runtime_error(
                    "Column '" + cat + "' is listed in both roles.covariates "
                    "and roles.categoricals; pick one role");
            }
        }
    }
    std::vector<int> categorical_cols;
    std::vector<std::string> categorical_names_resolved;
    categorical_cols.reserve(roles.categoricals.size());
    categorical_names_resolved.reserve(roles.categoricals.size());
    for (const auto& cat : roles.categoricals) {
        int col = reader.column_index(cat);
        if (col < 0) {
            throw std::runtime_error(
                "Categorical column not found in header CSV: " + cat);
        }
        categorical_cols.push_back(col);
        categorical_names_resolved.push_back(cat);
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
    float* coords_data = coordinates_.defined() ? coordinates_.data_ptr<float>() : nullptr;
    float* cov_data = covariates_.defined() ? covariates_.data_ptr<float>() : nullptr;
    int64_t cov_cols = covariates_.defined() ? covariates_.size(1) : 0;

    // Raw string buffer for each categorical column. Pre-reserved to n_plots
    // so the row-scan loop only does a push_back (no per-row allocation).
    // Filled with "" (NA) when a row is too short to hold the column.
    std::vector<std::vector<std::string>> categorical_raw(categorical_cols.size());
    for (auto& buf : categorical_raw) {
        buf.reserve(static_cast<size_t>(n_plots));
    }

    // Raw string buffer for each classification target column. Same layout
    // as `categorical_raw` — we collect strings during the scan and factorize
    // them post-scan into int64 codes that get written into targets_[name].
    // Regression targets are still parsed inline (see the row-scan body).
    std::vector<std::vector<std::string>> classification_raw(targets.size());
    for (size_t t = 0; t < targets.size(); ++t) {
        if (targets[t].task == TaskType::Classification) {
            classification_raw[t].reserve(static_cast<size_t>(n_plots));
        }
    }

    // Per-row keep mask. A row is "kept" iff every requested target column
    // produced a usable value (finite numeric for regression, non-missing
    // string for classification). Rows with any missing target get dropped
    // after the scan. Mirrors the POC's `ResolveDataset.from_fast_csv`
    // NaN-target drop semantics, including the classification case (the POC
    // drops nulls produced by `_encode_categorical`).
    std::vector<char> keep_row;
    keep_row.reserve(static_cast<size_t>(n_plots));

    int64_t row_idx = 0;
    reader.read_rows([&](size_t, const std::vector<std::string>& row) {
        if (row.size() <= static_cast<size_t>(plot_col)) {
            return;
        }

        plot_ids_.push_back(row[plot_col]);
        bool row_ok = true;

        // Coordinates
        if (coords_data && lon_col >= 0 && lat_col >= 0) {
            coords_data[row_idx * 2 + 0] = safe_stof(row[lon_col]);
            coords_data[row_idx * 2 + 1] = safe_stof(row[lat_col]);
        }

        // Covariates
        if (cov_data) {
            for (size_t i = 0; i < covariate_cols.size(); ++i) {
                int col = covariate_cols[i];
                if (row.size() > static_cast<size_t>(col)) {
                    cov_data[row_idx * cov_cols + static_cast<int64_t>(i)] = safe_stof(row[col]);
                }
            }
        }

        // Categorical covariates (raw strings; factorized post-loop)
        for (size_t c = 0; c < categorical_cols.size(); ++c) {
            int col = categorical_cols[c];
            if (row.size() > static_cast<size_t>(col)) {
                categorical_raw[c].push_back(row[col]);
            } else {
                categorical_raw[c].emplace_back();  // treated as NA -> code 0
            }
        }

        // Targets — parse with NaN-aware helpers and mark the row for drop
        // if any target is missing/non-finite/out-of-range. We still write a
        // sentinel zero into the tensor so the slot is well-defined; the
        // post-scan compaction step will drop it.
        for (size_t t = 0; t < targets.size(); ++t) {
            const auto& target = targets[t];
            std::string name = target.target_name.empty() ? target.column_name : target.target_name;
            int col = target_cols[t];

            if (col < 0 || row.size() <= static_cast<size_t>(col)) {
                row_ok = false;
                // Still push to classification_raw[t] so the buffer stays
                // aligned with row_idx; the post-scan compaction will drop
                // it. Empty string is treated as missing by the auto-fit
                // path, but we mark the row dead via row_ok anyway.
                if (target.task == TaskType::Classification) {
                    classification_raw[t].emplace_back();
                }
                continue;
            }
            if (target.task == TaskType::Classification) {
                // Defer string->int encoding until post-scan so the auto-fit
                // path sees the full distribution of values. Just stash the
                // raw cell here; row_ok will be flipped to false later if
                // the cell is missing/unmapped under the resolved mapping.
                classification_raw[t].push_back(row[col]);
            } else {
                auto parsed = parse_regression_target(row[col]);
                if (!parsed.has_value()) { row_ok = false; continue; }
                targets_[name][row_idx] = *parsed;
            }
        }

        keep_row.push_back(row_ok ? 1 : 0);
        row_idx++;
    });

    // ---- Fit + encode classification target columns ----
    // For each classification target, either use the explicit class_mapping
    // from the TargetSpec, or auto-fit one from the raw column data. Encode
    // the raw strings into the int64 target tensor. Missing/unmapped cells
    // flip the corresponding `keep_row[i]` to 0 so the compaction below
    // drops those plots — same semantics as a missing regression target.
    //
    // Mirrors the POC's `_apply_categorical_encoding` + `_encode_categorical`
    // pipeline. Loud per-target log line ("Encoded 'Eunis_lvl1' as Int64
    // (auto, 9 classes, 0 null)") matches what `from_fast_csv` prints.
    for (size_t t = 0; t < targets.size(); ++t) {
        const auto& target = targets[t];
        if (target.task != TaskType::Classification) continue;
        std::string name = target.target_name.empty() ? target.column_name : target.target_name;
        const auto& raw = classification_raw[t];

        std::unordered_map<std::string, int64_t> mapping;
        std::vector<std::string> class_names;
        const bool explicit_mapping = !target.class_mapping.empty();

        if (explicit_mapping) {
            mapping = target.class_mapping;
            // Build class_names from the explicit mapping, indexed by code.
            int64_t max_code = -1;
            for (const auto& [_, c] : mapping) if (c > max_code) max_code = c;
            if (max_code >= 0) {
                class_names.assign(static_cast<size_t>(max_code + 1), std::string{});
                for (const auto& [k, c] : mapping) {
                    if (c >= 0) class_names[static_cast<size_t>(c)] = k;
                }
            }
        } else {
            fit_classification_mapping(raw, mapping, class_names);
        }

        // Encode and count nulls. Tensor is preallocated to (n_plots,)
        // kLong; write each row, flip keep_row[i] = 0 for unmapped/NA.
        auto tgt = targets_[name];  // int64 tensor (n_plots,)
        auto acc = tgt.accessor<int64_t, 1>();
        int64_t n_null = 0;
        for (int64_t i = 0; i < row_idx; ++i) {
            const std::string& v = raw[static_cast<size_t>(i)];
            if (is_missing_cell(v)) {
                if (keep_row[static_cast<size_t>(i)]) keep_row[static_cast<size_t>(i)] = 0;
                ++n_null;
                continue;
            }
            auto it = mapping.find(v);
            if (it == mapping.end()) {
                if (keep_row[static_cast<size_t>(i)]) keep_row[static_cast<size_t>(i)] = 0;
                ++n_null;
                continue;
            }
            acc[i] = it->second;
        }

        // Persist the resolved vocab + num_classes on the target config so
        // the schema/checkpoint round-trips it. num_classes is the *count of
        // distinct classes* (= mapping.size()), matching the POC's
        // `cfg["num_classes"] = len(categorical_mappings[col])`. This can be
        // smaller than class_names.size() when the integer-pass path
        // produces sparse codes (e.g. mapping = {"0":0,"2":2,"5":5} →
        // num_classes=3 but class_names is length 6 with gaps). Sparse-int
        // EUNIS-style columns essentially never occur, but stay consistent
        // with the POC anyway. Auto-fill only when the caller passed 0.
        target_configs_[t].class_names = class_names;
        if (target_configs_[t].num_classes == 0) {
            target_configs_[t].num_classes = static_cast<int>(mapping.size());
        }

        const char* src = explicit_mapping ? "explicit" : "auto";
        std::cout << "  Encoded '" << target.column_name << "' as Int64 ("
                  << src << ", " << mapping.size() << " classes, "
                  << n_null << " null)" << std::endl;
    }
    schema_.targets = target_configs_;

    // ---- Filter rows with missing targets ----
    // After the scan, compact every per-plot buffer (plot_ids, coords,
    // covariates, categorical_raw, targets) to only the rows where every
    // target produced a usable value. Loud one-line summary mirrors the
    // POC's "Filtered N species records for invalid plots" log so users
    // see the n_plots drop instead of wondering where their plots went.
    const int64_t n_loaded = row_idx;
    int64_t n_keep = 0;
    for (char k : keep_row) if (k) ++n_keep;

    if (n_keep < n_loaded) {
        const int64_t n_dropped = n_loaded - n_keep;
        std::vector<int64_t> keep_idx;
        keep_idx.reserve(static_cast<size_t>(n_keep));
        for (int64_t i = 0; i < n_loaded; ++i) {
            if (keep_row[i]) keep_idx.push_back(i);
        }

        // Compact plot_ids_ in place via gather.
        std::vector<std::string> new_pids;
        new_pids.reserve(static_cast<size_t>(n_keep));
        for (int64_t i : keep_idx) new_pids.push_back(std::move(plot_ids_[i]));
        plot_ids_ = std::move(new_pids);

        // Compact tensors via index_select. index_select returns a new
        // contiguous tensor — the original buffer is released when the old
        // member is overwritten, so the working set drops immediately.
        auto idx_t = torch::tensor(keep_idx, torch::kInt64);
        if (coordinates_.defined()) coordinates_ = coordinates_.index_select(0, idx_t);
        if (covariates_.defined())  covariates_  = covariates_.index_select(0, idx_t);
        for (auto& [name, tensor] : targets_) {
            tensor = tensor.index_select(0, idx_t);
        }

        // Compact categorical raw-string buffers (factorization runs below).
        for (auto& buf : categorical_raw) {
            std::vector<std::string> new_buf;
            new_buf.reserve(static_cast<size_t>(n_keep));
            for (int64_t i : keep_idx) new_buf.push_back(std::move(buf[i]));
            buf = std::move(new_buf);
        }

        schema_.n_plots = n_keep;

        std::cout << "  Dropped " << n_dropped
                  << " plots with missing/NaN target ("
                  << n_keep << " of " << n_loaded << " kept; ";
        // List the target column names so the user knows which target drove
        // the drop. Multiple targets => any-missing semantics.
        bool first = true;
        for (const auto& target : targets) {
            if (!first) std::cout << ", ";
            std::cout << "'" << target.column_name << "'";
            first = false;
        }
        std::cout << ")" << std::endl;
    }

    // ---- Fit + encode categorical covariates ----
    // After the scan completes we have the full raw-string buffers. Fit each
    // column's vocab (sorted-unique non-NA -> codes 1..K) and encode into a
    // (n_plots, n_categoricals) int64 tensor stored on the dataset. The
    // vocab itself lives on the dataset for save/load.
    if (!categorical_names_resolved.empty()) {
        if (!use_external_vocabs_) {
            categorical_vocab_.fit(categorical_names_resolved, categorical_raw);
        }
        categorical_ids_ = categorical_vocab_.encode_batch(
            categorical_names_resolved, categorical_raw);
        schema_.categorical_names = categorical_names_resolved;
        schema_.categorical_vocab_sizes = categorical_vocab_.vocab_sizes();
        // schema_.categorical_embed_dim is left at its default here; the
        // model constructor will overwrite it from ModelConfig and the
        // updated value is what gets persisted in the checkpoint.
    }
}

void ResolveDataset::load_species_data(
    const std::string& species_path,
    const RoleMapping& roles
) {
    CSVReader reader(species_path);

    // Find column indices
    auto cols = ColumnIndices::from_reader(reader, roles);
    if (cols.plot < 0 || cols.species < 0) {
        throw std::runtime_error("Required columns not found: plot_id or species_id");
    }

    // Collect species records by plot
    std::unordered_map<std::string, std::vector<SpeciesRecord>> plot_records;

    reader.read_rows([&](size_t, const std::vector<std::string>& row) {
        if (row.size() <= static_cast<size_t>(std::max(cols.plot, cols.species))) {
            return;
        }

        std::string plot_id = row[cols.plot];
        std::string species_id = row[cols.species];

        SpeciesRecord record;
        record.plot_id = plot_id;
        record.species_id = species_id;
        record.abundance = cols.abundance >= 0 && row.size() > static_cast<size_t>(cols.abundance)
            ? safe_stof(row[cols.abundance], 1.0f) : 1.0f;

        if (cols.genus >= 0 && row.size() > static_cast<size_t>(cols.genus)) {
            record.genus = row[cols.genus];
        }
        if (cols.family >= 0 && row.size() > static_cast<size_t>(cols.family)) {
            record.family = row[cols.family];
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

    // Copy tracking flags from config to schema
    schema_.track_unknown_fraction = config_.track_unknown_fraction;
    schema_.track_unknown_count = config_.track_unknown_count;

    // Fit taxonomy vocabulary
    if (schema_.has_taxonomy) {
        if (!use_external_vocabs_) {
            taxonomy_vocab_.fit(all_records);
        }
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
    if (!use_external_vocabs_) {
        build_species_vocab(plot_species);
    } else {
        // Vocab pre-populated by from_csv_with_schema; just publish sizes
        // to the schema so downstream model construction reads the right
        // values. n_species reports the count of fitted species (excluding
        // the UNK slot at index 0), matching build_species_vocab() which
        // assigns schema_.n_species = species_counts.size() (also excluding
        // UNK). n_species_vocab includes UNK.
        schema_.n_species = static_cast<int64_t>(species_vocab_.size()) - 1;
        if (schema_.n_species < 0) schema_.n_species = 0;
        schema_.n_species_vocab = static_cast<int64_t>(species_vocab_.size());
    }

    // Determine n_taxonomy_slots
    int n_taxonomy_slots = config_.top_k;
    if (config_.selection == SelectionMode::TopBottom) {
        n_taxonomy_slots = 2 * config_.top_k;
    }

    // Encode based on mode
    if (config_.species_encoding == SpeciesEncodingMode::Hash) {
        if (config_.use_cuda_hash) {
            // CUDA hash mode: store raw species data in COO format for GPU computation
            // This enables on-the-fly hash embedding computation per batch on GPU

            // First pass: count total records and build plot index mapping
            std::unordered_map<std::string, int64_t> plot_id_to_idx;
            for (int64_t i = 0; i < n_plots; ++i) {
                plot_id_to_idx[plot_ids_[i]] = i;
            }

            // Count records per plot after selection
            std::vector<int64_t> records_per_plot(n_plots, 0);
            int64_t total_records = 0;

            for (int64_t i = 0; i < n_plots; ++i) {
                const auto& plot_id = plot_ids_[i];
                auto it = plot_records.find(plot_id);
                if (it == plot_records.end()) continue;

                // Convert to species-abundance pairs
                std::vector<std::pair<std::string, float>> species;
                for (const auto& rec : it->second) {
                    species.push_back({rec.species_id, rec.abundance});
                }

                // Apply selection (but not normalization yet - done at batch time)
                auto selected = apply_selection(std::move(species), config_.selection, config_.top_k);
                records_per_plot[i] = static_cast<int64_t>(selected.size());
                total_records += static_cast<int64_t>(selected.size());
            }

            // Build CSR-style offsets
            plot_offsets_ = torch::zeros({n_plots + 1}, torch::kLong);
            auto offset_acc = plot_offsets_.accessor<int64_t, 1>();
            offset_acc[0] = 0;
            for (int64_t i = 0; i < n_plots; ++i) {
                offset_acc[i + 1] = offset_acc[i] + records_per_plot[i];
            }

            // Allocate COO tensors
            raw_plot_indices_ = torch::zeros({total_records}, torch::kLong);
            raw_species_ids_ = torch::zeros({total_records}, torch::kLong);
            raw_weights_ = torch::zeros({total_records}, torch::kFloat32);

            auto plot_idx_acc = raw_plot_indices_.accessor<int64_t, 1>();
            auto species_id_acc = raw_species_ids_.accessor<int64_t, 1>();
            auto weight_acc = raw_weights_.accessor<float, 1>();

            // Second pass: fill COO data
            int64_t record_idx = 0;
            for (int64_t i = 0; i < n_plots; ++i) {
                const auto& plot_id = plot_ids_[i];
                auto it = plot_records.find(plot_id);
                if (it == plot_records.end()) continue;

                std::vector<std::pair<std::string, float>> species;
                for (const auto& rec : it->second) {
                    species.push_back({rec.species_id, rec.abundance});
                }

                auto selected = apply_selection(std::move(species), config_.selection, config_.top_k);
                apply_normalization(selected, config_.normalization);

                for (const auto& [sp_name, weight] : selected) {
                    plot_idx_acc[record_idx] = i;
                    // Hash species name to int64 using MurmurHash
                    species_id_acc[record_idx] = static_cast<int64_t>(murmur_hash(sp_name));
                    weight_acc[record_idx] = weight;
                    record_idx++;
                }
            }

            // Still create empty hash_embedding tensor to indicate hash mode (but it won't be used)
            // The actual hash embedding is computed on-the-fly during training
            hash_embedding_ = torch::Tensor();

        } else {
            // Standard CPU hash mode: pre-compute hash embeddings
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

                // Apply selection and normalization
                auto selected = apply_selection(std::move(species), config_.selection, config_.top_k);
                apply_normalization(selected, config_.normalization);

                // Hash
                hash_species(selected, &hash_acc[i][0], config_.hash_dim);
            }
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

    } else if (config_.species_encoding == SpeciesEncodingMode::RankPool ||
               config_.species_encoding == SpeciesEncodingMode::Transformer) {
        // Pool-style encoding: per-species taxonomy IDs + per-species species
        // IDs, weights, and masks. Mirrors the Python POC's
        // src/resolve/encode/_pool_base.py + rank_pool.py end-to-end via the
        // standalone RankPoolEncoder (single source of truth for
        // PoolWeighting semantics, vocab build, padding, has_cover flag).
        //
        // Output tensors (all (n_plots, max_species) except has_cover):
        //   species_ids_      : int64  per-species vocab index (lookup into
        //                              PlotEncoderRankPool::species_embedding)
        //   pool_genus_ids_   : int64  per-species genus index
        //   pool_family_ids_  : int64  per-species family index
        //   pool_weights_     : f32    per-species weight (binary/abundance/
        //                              log1p/norm/rank — see PoolWeighting)
        //   pool_mask_        : bool   true where a real species sits, false
        //                              in the padding region
        //   pool_has_cover_   : f32    (n_plots,) 1.0 if plot had real
        //                              abundance values, 0.0 otherwise

        // Flatten the per-plot records into a single vector keyed by plot_id,
        // matching RankPoolEncoder::transform's expected input layout.
        std::vector<SpeciesRecord> all_pool_records;
        all_pool_records.reserve(all_records.size());
        for (int64_t i = 0; i < n_plots; ++i) {
            const auto& plot_id = plot_ids_[i];
            auto it = plot_records.find(plot_id);
            if (it == plot_records.end()) continue;
            for (const auto& rec : it->second) {
                // The encoder reads plot_id off each record (not the outer
                // map key) — copy the plot_id over so missing-plot_id
                // SpeciesRecord rows still slot into the right plot.
                SpeciesRecord r = rec;
                if (r.plot_id.empty()) r.plot_id = plot_id;
                all_pool_records.push_back(std::move(r));
            }
        }

        RankPoolEncoder rp_encoder(config_.pool_weighting, /*min_frequency=*/1);
        rp_encoder.fit(all_pool_records);
        if (use_external_vocabs_) {
            // from_csv_with_schema path: replace the encoder's freshly-fit
            // species + taxonomy vocabs with the training-set vocabs that
            // were copied onto this dataset before load. fit() above is
            // still useful — it builds species_to_genus_/species_to_family_
            // from the test records so transform() can look up each species's
            // genus/family string. Test-only species map to UNK=0 via the
            // reused species_vocab, which is the correct behaviour at
            // inference time.
            // Strip the dataset-level "<UNK>"=>0 sentinel before handing the
            // map to SpeciesVocab: SpeciesVocab is the 1-indexed map with an
            // implicit UNK at 0, so including a literal "<UNK>"=>0 entry would
            // inflate n_species_vocab by one on the test split (encoder reports
            // species_to_id_.size()+1) and desync it from the train split.
            std::unordered_map<std::string, int64_t> sp_map;
            sp_map.reserve(species_to_idx_.size());
            for (const auto& [name, id] : species_to_idx_) {
                if (name == "<UNK>" || id == 0) continue;
                sp_map.emplace(name, id);
            }
            auto sv = SpeciesVocab::from_map(std::move(sp_map));
            rp_encoder.set_vocabs(std::move(sv), taxonomy_vocab_);
        }
        auto encoded = rp_encoder.transform(all_pool_records, plot_ids_,
                                            config_.pool_species_cap);

        species_ids_ = encoded.species_ids;
        pool_genus_ids_ = encoded.genus_ids;
        pool_family_ids_ = encoded.family_ids;
        pool_weights_ = encoded.weights;
        pool_mask_ = encoded.mask;
        // Encoder returns float32 has_cover; downstream APIs accept either,
        // but we expose it as float32 so PlotEncoderRankPool's "default to
        // ones" path (which yields float32) stays consistent.
        pool_has_cover_ = encoded.has_cover;

        // The rank-pool encoder owns its own species vocab. Sync it back
        // onto the dataset so the schema reports the right vocab size for
        // ResolveModel's PlotEncoderRankPool::species_embedding sizing
        // (which must allocate (n_species_vocab, species_embed_dim)).
        const auto& sp_vocab = rp_encoder.species_vocab();
        species_to_idx_ = sp_vocab.species_to_id();
        species_vocab_.clear();
        species_vocab_.push_back("<UNK>");
        // Rebuild ordered vocab from the id map (codes 1..K, sorted by code).
        std::vector<std::pair<std::string, int64_t>> sp_items(
            species_to_idx_.begin(), species_to_idx_.end());
        std::sort(sp_items.begin(), sp_items.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        for (const auto& [name, _id] : sp_items) {
            species_vocab_.push_back(name);
        }
        schema_.n_species = static_cast<int64_t>(sp_items.size());
        schema_.n_species_vocab = encoded.n_species_vocab;
        // Refresh taxonomy schema fields from the encoder's own vocab so the
        // rank-pool encoder lookup tables match what the model is sized for.
        schema_.has_taxonomy = encoded.n_genera_vocab > 1 && config_.use_taxonomy;
        if (schema_.has_taxonomy) {
            schema_.n_genera = encoded.n_genera_vocab;
            schema_.n_families = encoded.n_families_vocab;
            schema_.n_genera_vocab = encoded.n_genera_vocab;
            schema_.n_families_vocab = encoded.n_families_vocab;
            // Replace the dataset's TaxonomyVocab with the encoder's so
            // downstream code that reads taxonomy_vocab() gets the right
            // string-to-id maps.
            taxonomy_vocab_ = rp_encoder.taxonomy_vocab();
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
