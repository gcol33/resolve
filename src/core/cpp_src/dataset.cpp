#include "resolve/dataset.hpp"
#include "resolve/continuous_block.hpp"
#include "resolve/csv_reader.hpp"
#include "resolve/csv_utils.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/io_retry.hpp"
#include "resolve/name_pool.hpp"
#include "resolve/sparse_rows.hpp"
#include "resolve/species_encoder.hpp"
#include "resolve/target_labels.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <unordered_set>

namespace resolve {

namespace {

// A coordinate, covariate or masked regression cell the loader could not read.
// It stays NaN in the dataset's tensors; DatasetConfig::missing_values decides
// how the model reads a coordinate or covariate.
constexpr float kMissingValue = std::numeric_limits<float>::quiet_NaN();

// Parse a numeric cell. Returns nullopt for empty / NA marker / unparseable /
// non-finite values, so "missing" is never conflated with an actual zero.
std::optional<float> parse_numeric_cell(std::string_view s) {
    if (is_na_string(s)) return std::nullopt;
    auto v = parse_float_strict(s);  // locale-free, rejects trailing garbage
    if (!v || !std::isfinite(*v)) return std::nullopt;
    return v;
}

// A 1-D tensor that takes ownership of `values` without copying them.
template <typename T>
torch::Tensor owning_tensor(std::vector<T>&& values, torch::ScalarType dtype) {
    auto* owner = new std::vector<T>(std::move(values));
    const auto n = static_cast<int64_t>(owner->size());
    return torch::from_blob(owner->data(), {n}, [owner](void*) { delete owner; },
                            torch::TensorOptions().dtype(dtype));
}

// Upper bound on a directly-used integer class code. A code at or above this
// would size class_names to ~that many entries (unbounded allocation / OOM) for
// only a handful of classes; a code beyond it falls back to a compact
// factorization (auto path) or is rejected (explicit path).
constexpr int64_t kMaxDirectClassCode = 1 << 20;  // 1,048,575

// Auto-fit a string->int class mapping over the raw cells of one classification
// column:
//   - NA-like cells contribute nothing to the mapping.
//   - If every distinct non-NA cell parses as a base-10 integer, the parsed
//     integers are used as the codes (preserves "0".."8" encodings exactly).
//   - Otherwise the sorted-unique non-NA values are factorized to 0..K-1
//     (sorting is lexicographic on the raw strings).
// mapping_out is string -> code; class_names[code] is the original string.
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
            if (is_na_string(v)) continue;
            if (seen.insert(v).second) uniques.push_back(v);
        }
    }
    if (uniques.empty()) return;

    std::sort(uniques.begin(), uniques.end());

    std::vector<int64_t> parsed;
    parsed.reserve(uniques.size());
    bool all_int = true;
    // A column whose every label is a base-10 integer keeps those integers as
    // its codes, so an integer-encoded column ("0".."8") keeps its codes
    // across load/save.
    for (const auto& v : uniques) {
        auto p = parse_int64_strict(v);
        if (!p) { all_int = false; break; }
        parsed.push_back(*p);
    }

    // Parsed integers serve as class codes only when they are non-negative and
    // reasonably dense: a negative label would index class_names out of
    // bounds, and an ID-like label (9999999999) would size class_names to ~that
    // value for a handful of classes. Either falls back to a compact 0..K-1
    // factorization of the sorted unique labels.
    bool use_direct_codes = false;
    if (all_int) {
        int64_t min_code = *std::min_element(parsed.begin(), parsed.end());
        int64_t max_code = *std::max_element(parsed.begin(), parsed.end());
        use_direct_codes = (min_code >= 0 && max_code < kMaxDirectClassCode);
    }

    if (use_direct_codes) {
        // class_names is ordered by code so the vocab round-trips through
        // save/load; codes that are not dense (e.g. {0, 2, 5}) leave empty gaps.
        std::vector<std::pair<int64_t, std::string>> pairs;
        pairs.reserve(uniques.size());
        for (size_t i = 0; i < uniques.size(); ++i) {
            mapping_out[uniques[i]] = parsed[i];
            pairs.emplace_back(parsed[i], uniques[i]);
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
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

// Resolve a classification target's (mapping, class_names) from the caller's
// explicit class_mapping (class_names indexed by code, gaps for sparse codes),
// or, when there is none, fit it on the raw labels of the rows `keep` marks.
// A class value occurring only in rows dropped because a different target is
// missing never enters class_names: it would inflate num_classes with a
// zero-example class and shift every later code of the lexicographic
// factorization.
void resolve_classification_mapping(
    const std::unordered_map<std::string, int64_t>& explicit_mapping,
    const std::vector<std::string>& raw_labels,
    const std::vector<char>& keep,
    std::unordered_map<std::string, int64_t>& mapping_out,
    std::vector<std::string>& class_names_out
) {
    if (explicit_mapping.empty()) {
        std::vector<std::string> kept;
        kept.reserve(raw_labels.size());
        for (size_t i = 0; i < raw_labels.size(); ++i) {
            if (i < keep.size() && !keep[i]) continue;
            kept.push_back(raw_labels[i]);
        }
        fit_classification_mapping(kept, mapping_out, class_names_out);
        return;
    }
    mapping_out = explicit_mapping;
    int64_t max_code = -1;
    for (const auto& [name, c] : mapping_out) {
        // A negative code would be written into the int64 target tensor and
        // index the classification head out of bounds.
        if (c < 0) {
            throw std::invalid_argument(
                "class_mapping code for '" + name + "' is negative (" +
                std::to_string(c) + "); class codes must be >= 0");
        }
        if (c > max_code) max_code = c;
    }
    if (max_code >= kMaxDirectClassCode) {
        throw std::invalid_argument(
            "class_mapping code " + std::to_string(max_code) +
            " exceeds the maximum direct class code (" +
            std::to_string(kMaxDirectClassCode) +
            "); use dense codes 0..K-1");
    }
    if (max_code >= 0) {
        class_names_out.assign(static_cast<size_t>(max_code + 1), std::string{});
        for (const auto& [k, c] : mapping_out)
            class_names_out[static_cast<size_t>(c)] = k;
    }
}

// Encode one classification target: resolve the mapping (fit on kept rows),
// write the (n,) int64 code tensor, and set + validate cfg.class_names /
// cfg.num_classes. An NA/unmapped cell flips keep[i]=0 under
// TargetMissingPolicy::Drop; under Mask the row stays and its code is
// kUnlabelledClass. Either way the cell contributes nothing to the class list.
//
// The direct-integer path emits the parsed integers verbatim as class codes, so
// class_names.size() == max_code + 1. The classification head is sized from
// num_classes, so a caller-supplied num_classes smaller than that (a 1-indexed
// column 1..9 built with num_classes=9) would let a code index the head out of
// bounds: it is filled when the caller passed 0 and rejected when too small.
torch::Tensor encode_classification_target(
    const TargetSpec& spec,
    const std::vector<std::string>& raw,
    std::vector<char>& keep,
    TargetConfig& cfg,
    const std::string& log_name
) {
    std::unordered_map<std::string, int64_t> mapping;
    std::vector<std::string> class_names;
    const bool explicit_mapping = !spec.class_mapping.empty();
    resolve_classification_mapping(spec.class_mapping, raw, keep, mapping, class_names);

    const bool masked = spec.missing == TargetMissingPolicy::Mask;
    const int64_t n = static_cast<int64_t>(raw.size());
    auto tensor = torch::zeros({n}, torch::kLong);
    auto acc = tensor.accessor<int64_t, 1>();
    int64_t n_null = 0;
    auto mark_missing = [&](int64_t i) {
        if (masked) {
            acc[i] = kUnlabelledClass;
        } else if (i < static_cast<int64_t>(keep.size()) && keep[i]) {
            keep[i] = 0;
        }
        ++n_null;
    };
    for (int64_t i = 0; i < n; ++i) {
        const std::string& v = raw[static_cast<size_t>(i)];
        if (is_na_string(v)) {
            mark_missing(i);
            continue;
        }
        auto it = mapping.find(v);
        if (it == mapping.end()) {
            mark_missing(i);
            continue;
        }
        acc[i] = it->second;
    }

    cfg.class_names = class_names;
    const int n_classes = static_cast<int>(class_names.size());
    if (cfg.num_classes == 0) {
        cfg.num_classes = n_classes;
    } else if (cfg.num_classes < n_classes) {
        throw std::invalid_argument(
            "classification target '" + log_name + "': num_classes=" +
            std::to_string(cfg.num_classes) + " is smaller than the number of class "
            "codes in the data (" + std::to_string(n_classes) + "). Direct integer "
            "labels are used verbatim as class codes, so a 1-indexed column 1.." +
            std::to_string(n_classes - 1) + " needs num_classes >= " +
            std::to_string(n_classes) + " (index 0 reserved). Pass num_classes=0 to "
            "size the head automatically, or use dense 0-based codes.");
    }

    std::cout << "  Encoded '" << spec.column_name << "' as Int64 ("
              << (explicit_mapping ? "explicit" : "auto") << ", "
              << mapping.size() << " classes, " << n_null << " null"
              << (masked ? ", masked" : "") << ")" << std::endl;
    return tensor;
}

// Finalize the keep flags for a classification target WITHOUT fitting the
// vocab, so every classification vocab is fit against the final surviving-row
// set even when another classification target drives additional drops.
// Auto-fit maps every non-NA cell in a kept row, so only NA cells drop; an
// explicit mapping additionally drops non-NA cells absent from the mapping. A
// target whose missing labels are masked drops nothing.
void mark_classification_drops(
    const TargetSpec& spec,
    const std::vector<std::string>& raw,
    std::vector<char>& keep
) {
    if (spec.missing == TargetMissingPolicy::Mask) return;
    const bool explicit_mapping = !spec.class_mapping.empty();
    const int64_t n = static_cast<int64_t>(raw.size());
    for (int64_t i = 0; i < n; ++i) {
        if (i >= static_cast<int64_t>(keep.size()) || !keep[i]) continue;
        const std::string& v = raw[static_cast<size_t>(i)];
        if (is_na_string(v)) { keep[i] = 0; continue; }
        if (explicit_mapping && spec.class_mapping.find(v) == spec.class_mapping.end())
            keep[i] = 0;
    }
}

// The name a target takes in the model: target_name, or the column when unset.
std::string target_name_of(const TargetSpec& spec) {
    return spec.target_name.empty() ? spec.column_name : spec.target_name;
}

// The TargetConfig a loader starts a target from. Classification class names
// and num_classes are filled in later by encode_classification_target.
TargetConfig target_config_of(const TargetSpec& spec) {
    TargetConfig cfg{target_name_of(spec), spec.task, spec.transform,
                     spec.num_classes, spec.weight};
    cfg.missing = spec.missing;
    return cfg;
}

// After every target is encoded and before the rows marked in `keep` are
// compacted away. Only a dataset with a masked target has work here: a Drop
// target's unlabelled rows are already marked, so every kept row is labelled
// for every Drop target.
//
//   - A kept row labelled for no target is dropped: it would train nothing.
//   - Each masked target reports the number of kept plots labelled for it, and
//     a masked target labelled on none of them is an error, since there is
//     nothing to fit its head to.
void finalize_masked_labels(
    const std::vector<TargetSpec>& targets,
    const std::unordered_map<std::string, torch::Tensor>& tensors,
    std::vector<char>& keep
) {
    const bool any_masked = std::any_of(targets.begin(), targets.end(),
        [](const TargetSpec& t) { return t.missing == TargetMissingPolicy::Mask; });
    if (!any_masked || keep.empty()) return;

    const int64_t n = static_cast<int64_t>(keep.size());
    std::vector<torch::Tensor> labelled(targets.size());
    std::vector<char> any_label(keep.size(), 0);
    for (size_t t = 0; t < targets.size(); ++t) {
        labelled[t] = labelled_mask(tensors.at(target_name_of(targets[t])),
                                    targets[t].task).contiguous();
        const auto acc = labelled[t].accessor<bool, 1>();
        for (int64_t i = 0; i < n; ++i) {
            if (acc[i]) any_label[static_cast<size_t>(i)] = 1;
        }
    }

    int64_t n_unlabelled = 0;
    for (size_t i = 0; i < keep.size(); ++i) {
        if (keep[i] && !any_label[i]) {
            keep[i] = 0;
            ++n_unlabelled;
        }
    }
    if (n_unlabelled > 0) {
        std::cout << "  Dropped " << n_unlabelled
                  << " plots labelled for no target" << std::endl;
    }

    int64_t n_kept = 0;
    for (char k : keep) if (k) ++n_kept;
    for (size_t t = 0; t < targets.size(); ++t) {
        if (targets[t].missing != TargetMissingPolicy::Mask) continue;
        const std::string name = target_name_of(targets[t]);
        const auto acc = labelled[t].accessor<bool, 1>();
        int64_t n_labelled = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (acc[i] && keep[static_cast<size_t>(i)]) ++n_labelled;
        }
        std::cout << "  Target '" << name << "' (missing labels masked): labelled on "
                  << n_labelled << " of " << n_kept << " plots" << std::endl;
        if (n_labelled == 0 && n_kept > 0) {
            throw std::invalid_argument(
                "target '" + name + "' (missing = mask) is labelled on none of the " +
                std::to_string(n_kept) + " plots kept, so there is nothing to fit "
                "its head to. Check the column '" + targets[t].column_name + "'.");
        }
    }
}

// The index of a named role column in `source`. A named column the source
// lacks is a configuration error (typically a typo'd role name), never a
// silent "no such feature": it would otherwise train a model without the input
// the caller asked for and bake the wrong feature count into the checkpoint.
int require_column(const RowSource& source, const std::string& name, const char* role,
                   const char* table) {
    const int col = source.column_index(name);
    if (col < 0) {
        throw std::runtime_error(std::string(role) + " column '" + name +
                                 "' not found in the " + table + " table");
    }
    return col;
}

// Index of an optional role, -1 when the role is unset.
int optional_column(const RowSource& source, const std::optional<std::string>& name,
                    const char* role, const char* table) {
    return name ? require_column(source, *name, role, table) : -1;
}

// The cell of `row` at `col`, or nullopt when the row is too short to hold it.
std::optional<std::string_view> cell_at(RowFields row, int col) {
    if (col < 0 || row.size() <= static_cast<size_t>(col)) return std::nullopt;
    return row[static_cast<size_t>(col)];
}

std::optional<float> parse_cell_at(RowFields row, int col) {
    const auto cell = cell_at(row, col);
    return cell ? parse_numeric_cell(*cell) : std::nullopt;
}

const char* missing_value_handling(MissingValuePolicy policy) {
    return policy == MissingValuePolicy::Zero
        ? "the model reads it as 0 (missing_values = zero)"
        : "the model flags it and reads the fitting fold's mean (missing_values = indicate)";
}

void report_missing_covariate(const std::string& column, int64_t n_missing,
                              MissingValuePolicy policy) {
    if (n_missing > 0) {
        std::cerr << "[RESOLVE] covariate column '" << column << "' has " << n_missing
                  << " missing/unparseable cell(s); " << missing_value_handling(policy)
                  << std::endl;
    }
}

void report_missing_coordinates(int64_t n_missing, MissingValuePolicy policy) {
    if (n_missing > 0) {
        std::cerr << "[RESOLVE] " << n_missing
                  << " plot(s) have a missing/unparseable coordinate; "
                  << missing_value_handling(policy) << std::endl;
    }
}

// Positions of the kept rows.
std::vector<int64_t> kept_indices(const std::vector<char>& keep) {
    std::vector<int64_t> idx;
    for (size_t i = 0; i < keep.size(); ++i) {
        if (keep[i]) idx.push_back(static_cast<int64_t>(i));
    }
    return idx;
}

// Reduce `values` to the positions `idx` lists, in that order.
template <typename T>
void keep_rows(std::vector<T>& values, const std::vector<int64_t>& idx) {
    std::vector<T> kept;
    kept.reserve(idx.size());
    for (int64_t i : idx) kept.push_back(std::move(values[static_cast<size_t>(i)]));
    values = std::move(kept);
}

torch::Tensor rows_to_tensor(std::vector<float>& buffer, int64_t n_rows, int64_t width) {
    if (n_rows == 0) {
        return width > 0 ? torch::zeros({0, width}, torch::kFloat32)
                         : torch::zeros({0}, torch::kFloat32);
    }
    if (width > 0) {
        return torch::from_blob(buffer.data(), {n_rows, width}, torch::kFloat32).clone();
    }
    return torch::from_blob(buffer.data(), {n_rows}, torch::kFloat32).clone();
}

// The species-table columns a record is read from. plot_id and species_id are
// required; a named abundance / genus / family role must resolve.
struct ColumnIndices {
    int plot = -1;
    int species = -1;
    int abundance = -1;
    int genus = -1;
    int family = -1;

    ColumnIndices(const RowSource& source, const RoleMapping& roles)
        : plot(require_column(source, roles.plot_id, "plot_id", "species")),
          species(require_column(source, roles.species_id, "species_id", "species")),
          abundance(optional_column(source, roles.abundance_column(), "abundance", "species")),
          genus(optional_column(source, roles.genus_column(), "genus", "species")),
          family(optional_column(source, roles.family_column(), "family", "species")) {}
};

// Reject a zero-abundance reading no weighting can use, before any row is read.
void require_valid_zero_abundance(float zero_abundance_as) {
    if (!std::isfinite(zero_abundance_as) || zero_abundance_as < 0.0f) {
        throw std::invalid_argument(
            "DatasetConfig.zero_abundance_as must be a finite value >= 0; got " +
            std::to_string(zero_abundance_as));
    }
}

// Fixed genus/family slots of the hash and sparse encodings, also the
// embedding encoder's taxonomy width: top_k, doubled under TopBottom.
int taxonomy_slot_count(const DatasetConfig& config) {
    return config.selection == SelectionMode::TopBottom ? 2 * config.top_k : config.top_k;
}

// The species width a schema records as resolved: encode_species overwrites a
// rank-pool schema's pool_species_cap with the width the data was truncated
// to, so a positive value is a width; a percentile request (< 0) or no cap (0)
// that never reached a rank-pool encoder records none.
int resolved_species_width(const ResolveSchema& schema) {
    const int cap = schema.dataset_config.pool_species_cap;
    return cap > 0 ? cap : 0;
}

}  // namespace

// =============================================================================
// Species rows
// =============================================================================

// Reads species records from the rows of a species table into a
// PlotAssemblages::Builder, whose names are interned on arrival. A missing /
// NA / unparseable abundance cell is read as 1.0, a presence, and counted so
// the loader warns about it; a parsed abundance of exactly 0 is read as
// DatasetConfig::zero_abundance_as.
class ResolveDataset::SpeciesRows {
public:
    SpeciesRows(const RowSource& source, const RoleMapping& roles, float zero_abundance_as)
        : cols_(source, roles),
          zero_abundance_as_(zero_abundance_as),
          min_width_(static_cast<size_t>(std::max(cols_.plot, cols_.species)) + 1) {}

    // The plot `row` names, or nullopt for a row too short to name its plot
    // and species.
    [[nodiscard]] std::optional<std::string_view> plot_id(RowFields row) const {
        if (row.size() < min_width_) return std::nullopt;
        return row[static_cast<size_t>(cols_.plot)];
    }

    // Reads the record a row with a plot_id() holds, as a record of plot code
    // `plot`; a code < 0 (a plot the load does not keep) reads the abundance
    // cell for the warning count and keeps nothing.
    void add(RowFields row, int32_t plot) {
        float abundance = 1.0f;
        if (cols_.abundance >= 0) {
            if (auto parsed = parse_cell_at(row, cols_.abundance)) {
                abundance = *parsed == 0.0f ? zero_abundance_as_ : *parsed;
            } else {
                ++n_coerced_;
            }
        }
        if (plot < 0) return;
        builder_.add(plot, row[static_cast<size_t>(cols_.species)], abundance,
                     cell_at(row, cols_.genus).value_or(std::string_view{}),
                     cell_at(row, cols_.family).value_or(std::string_view{}));
    }

    [[nodiscard]] bool has_abundance_column() const noexcept { return cols_.abundance >= 0; }

    // The records grouped into the assemblages of `n_plots` plots (plot_of maps
    // a plot code to its plot, -1 drops it), after warning about coerced
    // abundance cells.
    PlotAssemblages finish(int64_t n_plots, std::span<const int32_t> plot_of,
                           RepresentationMode representation) {
        if (n_coerced_ > 0) {
            std::cerr << "[RESOLVE] warning: " << n_coerced_
                      << " abundance cell(s) were missing/NA/unparseable and "
                         "defaulted to weight 1.0; these are treated as real "
                         "presences, not missing cover"
                      << std::endl;
        }
        return std::move(builder_).finish(n_plots, plot_of, representation);
    }

private:
    ColumnIndices cols_;
    float zero_abundance_as_;
    size_t min_width_;
    int64_t n_coerced_ = 0;
    PlotAssemblages::Builder builder_;
};

// =============================================================================
// Plot-level columns
// =============================================================================

// The plot-level columns of a table, one row per plot: coordinates, numeric
// and categorical covariates, and targets. The two-file loader adds every
// header row, the single-table loader the first row of each plot, and
// absorb_plot_columns turns both into the same tensors. Plot IDs are interned
// in `plots`, so a plot's code is its row here.
struct ResolveDataset::PlotColumns {
    PlotColumns(const RowSource& source, const RoleMapping& roles,
                const std::vector<TargetSpec>& target_specs, bool read_covariates,
                const char* table)
        : targets(target_specs) {
        lon_col = optional_column(source, roles.longitude_column(), "longitude", table);
        lat_col = optional_column(source, roles.latitude_column(), "latitude", table);
        has_coordinates = lon_col >= 0 && lat_col >= 0;

        if (read_covariates) {
            const std::unordered_set<std::string> numeric(roles.covariates.begin(),
                                                          roles.covariates.end());
            for (const auto& name : roles.categoricals) {
                if (numeric.count(name)) {
                    throw std::runtime_error(
                        "Column '" + name + "' is listed in both roles.covariates "
                        "and roles.categoricals; pick one role");
                }
            }
            for (const auto& name : roles.covariates) {
                covariate_cols.push_back(require_column(source, name, "covariate", table));
                covariate_names.push_back(name);
            }
            for (const auto& name : roles.categoricals) {
                categorical_cols.push_back(require_column(source, name, "categorical", table));
                categorical_names.push_back(name);
            }
        }
        covariate_missing.assign(covariate_cols.size(), 0);
        categorical_raw.resize(categorical_cols.size());

        for (const auto& target : targets) {
            target_cols.push_back(require_column(source, target.column_name, "target", table));
        }
        regression.resize(targets.size());
        classification.resize(targets.size());
    }

    // The code of plot `plot_id`, appending the plot `row` describes when the
    // ID is new (the bool). Every buffer grows by exactly one entry per plot
    // (NaN / "" for a missing cell), so the buffers stay aligned with the
    // codes; keep marks a plot whose Drop target is missing or unparseable.
    std::pair<int32_t, bool> add(RowFields row, std::string_view plot_id) {
        const auto [code, inserted] = plots.insert(plot_id);
        if (!inserted) return {code, false};
        bool row_ok = true;

        // A blank / NA / unparseable coordinate stays NaN rather than reading
        // as (0, 0), a real location in the Gulf of Guinea.
        if (has_coordinates) {
            const auto lon = parse_cell_at(row, lon_col);
            const auto lat = parse_cell_at(row, lat_col);
            coordinates.push_back(lon.value_or(kMissingValue));
            coordinates.push_back(lat.value_or(kMissingValue));
            if (!lon || !lat) ++coordinates_missing;
        }
        for (size_t i = 0; i < covariate_cols.size(); ++i) {
            const auto value = parse_cell_at(row, covariate_cols[i]);
            covariates.push_back(value.value_or(kMissingValue));
            if (!value) ++covariate_missing[i];
        }
        for (size_t c = 0; c < categorical_cols.size(); ++c) {
            categorical_raw[c].emplace_back(cell_at(row, categorical_cols[c]).value_or(""));
        }
        // Classification cells are encoded once the whole column is known, so
        // the auto-fit sees every value; an NA cell drops the row then.
        for (size_t t = 0; t < targets.size(); ++t) {
            const bool masked = targets[t].missing == TargetMissingPolicy::Mask;
            const auto cell = cell_at(row, target_cols[t]);
            if (targets[t].task == TaskType::Classification) {
                classification[t].emplace_back(cell.value_or(""));
                if (!cell && !masked) row_ok = false;
            } else {
                const auto value = cell ? parse_numeric_cell(*cell) : std::nullopt;
                regression[t].push_back(value.value_or(masked ? kMissingValue : 0.0f));
                if (!value && !masked) row_ok = false;
            }
        }
        keep.push_back(row_ok ? 1 : 0);
        return {code, true};
    }

    const std::vector<TargetSpec>& targets;
    int lon_col = -1;
    int lat_col = -1;
    std::vector<int> covariate_cols;
    std::vector<int> categorical_cols;
    std::vector<int> target_cols;

    NamePool plots;
    bool has_coordinates = false;
    std::vector<float> coordinates;  // (lon, lat) per row
    int64_t coordinates_missing = 0;
    std::vector<std::string> covariate_names;
    std::vector<float> covariates;  // covariate_cols.size() values per row
    std::vector<int64_t> covariate_missing;
    std::vector<std::string> categorical_names;
    std::vector<std::vector<std::string>> categorical_raw;
    std::vector<std::vector<float>> regression;            // per target, empty for classification
    std::vector<std::vector<std::string>> classification;  // per target, empty for regression
    std::vector<char> keep;
};

std::vector<int64_t> ResolveDataset::absorb_plot_columns(PlotColumns& columns,
                                                        int64_t n_skipped) {
    const auto& targets = columns.targets;
    const auto n_loaded = static_cast<int64_t>(columns.plots.size());

    if (columns.has_coordinates) {
        coordinates_ = rows_to_tensor(columns.coordinates, n_loaded, 2);
        schema_.has_coordinates = true;
    }
    schema_.covariate_names = columns.covariate_names;
    const auto n_covariates = static_cast<int64_t>(columns.covariate_names.size());
    if (n_covariates > 0) {
        covariates_ = rows_to_tensor(columns.covariates, n_loaded, n_covariates);
    }
    for (size_t i = 0; i < columns.covariate_missing.size(); ++i) {
        report_missing_covariate(columns.covariate_names[i], columns.covariate_missing[i],
                                 config_.missing_values);
    }
    report_missing_coordinates(columns.coordinates_missing, config_.missing_values);

    // Regression tensors as read; classification in two passes, so every
    // classification vocab is fit on the final surviving-row set: the first
    // finalizes keep (adding NA / explicit-unmapped drops), the second fits
    // and encodes on the kept rows.
    target_configs_.clear();
    for (size_t t = 0; t < targets.size(); ++t) {
        target_configs_.push_back(target_config_of(targets[t]));
        if (targets[t].task != TaskType::Classification) {
            targets_[target_name_of(targets[t])] =
                rows_to_tensor(columns.regression[t], n_loaded, 0);
        }
    }
    for (size_t t = 0; t < targets.size(); ++t) {
        if (targets[t].task != TaskType::Classification) continue;
        mark_classification_drops(targets[t], columns.classification[t], columns.keep);
    }
    for (size_t t = 0; t < targets.size(); ++t) {
        if (targets[t].task != TaskType::Classification) continue;
        const std::string name = target_name_of(targets[t]);
        targets_[name] = encode_classification_target(
            targets[t], columns.classification[t], columns.keep, target_configs_[t], name);
    }
    finalize_masked_labels(targets, targets_, columns.keep);

    // Compact every per-plot buffer to the kept rows, and say how many went.
    const std::vector<int64_t> kept = kept_indices(columns.keep);
    const auto n_keep = static_cast<int64_t>(kept.size());
    if (n_keep < n_loaded) {
        const auto idx = torch::tensor(kept, torch::kInt64);
        if (coordinates_.defined()) coordinates_ = coordinates_.index_select(0, idx);
        if (covariates_.defined()) covariates_ = covariates_.index_select(0, idx);
        for (auto& [name, tensor] : targets_) tensor = tensor.index_select(0, idx);
        for (auto& raw : columns.categorical_raw) keep_rows(raw, kept);
    }
    const int64_t n_dropped = n_loaded - n_keep;
    if (n_dropped > 0 || n_skipped > 0) {
        std::cout << "  Kept " << n_keep << " of " << (n_loaded + n_skipped) << " plots (";
        if (n_dropped > 0) {
            std::cout << n_dropped << " dropped for missing/NaN target [";
            for (size_t t = 0; t < targets.size(); ++t) {
                std::cout << (t ? ", " : "") << "'" << targets[t].column_name << "'";
            }
            std::cout << "]";
        }
        if (n_skipped > 0) {
            if (n_dropped > 0) std::cout << "; ";
            std::cout << n_skipped << " skipped as too short to contain plot_id";
        }
        std::cout << ")" << std::endl;
    }

    // Categorical covariates: each column's vocab (sorted-unique non-NA ->
    // codes 1..K) fitted on the kept rows, unless adopted, and encoded into a
    // (n_plots, n_categoricals) int64 tensor. schema_.categorical_embed_dim is
    // left at its default; the model overwrites it from ModelConfig.
    if (!columns.categorical_names.empty()) {
        if (!use_external_vocabs_) {
            categorical_vocab_.fit(columns.categorical_names, columns.categorical_raw);
        }
        categorical_ids_ = categorical_vocab_.encode_batch(columns.categorical_names,
                                                           columns.categorical_raw);
        schema_.categorical_names = columns.categorical_names;
        schema_.categorical_vocab_sizes = categorical_vocab_.vocab_sizes();
    }

    schema_.targets = target_configs_;
    schema_.n_plots = n_keep;
    return kept;
}

void ResolveDataset::finish_load(SpeciesRows& rows, PlotColumns& columns,
                                 const std::vector<int64_t>& kept) {
    // Plot code kept[i] is plot i; every other code is dropped.
    std::vector<int32_t> plot_of(static_cast<size_t>(columns.plots.size()), -1);
    for (size_t i = 0; i < kept.size(); ++i) {
        plot_of[static_cast<size_t>(kept[i])] = static_cast<int32_t>(i);
    }
    has_abundance_column_ = rows.has_abundance_column();
    encode_species(rows.finish(schema_.n_plots, plot_of, config_.representation));
    plot_ids_ = columns.plots.release();
    keep_rows(plot_ids_, kept);
}

// =============================================================================
// Loaders
// =============================================================================

std::vector<TargetSpec> ResolveDataset::begin_load(const std::vector<TargetSpec>& targets,
                                                   const ExternalVocabs* vocabs) {
    require_valid_zero_abundance(config_.zero_abundance_as);
    if (vocabs == nullptr) return targets;
    adopt_vocabs(*vocabs);
    return replay_class_mappings(targets, vocabs->targets);
}

ResolveDataset ResolveDataset::load_tables(
    RowSource& header,
    RowSource& species,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config,
    const ExternalVocabs* vocabs
) {
    ResolveDataset dataset;
    dataset.config_ = config;
    const std::vector<TargetSpec> specs = dataset.begin_load(targets, vocabs);

    // Header rows are plots: a duplicate plot_id would give two plot slots
    // the same species records, so it is rejected.
    const int plot_col = require_column(header, roles.plot_id, "plot_id", "header");
    PlotColumns plots(header, roles, specs, /*read_covariates=*/true, "header");
    int64_t n_skipped = 0;
    header.read_rows([&](size_t, RowFields row) {
        const auto plot_id = cell_at(row, plot_col);
        if (!plot_id) {
            ++n_skipped;
            return;
        }
        if (!plots.add(row, *plot_id).second) {
            throw std::runtime_error(
                "Duplicate plot_id '" + std::string(*plot_id) + "' in header data. Plot "
                "IDs must be unique (each header row is one plot).");
        }
    });
    const std::vector<int64_t> kept = dataset.absorb_plot_columns(plots, n_skipped);

    // A record of a plot the header does not list belongs to no plot.
    SpeciesRows rows(species, roles, config.zero_abundance_as);
    species.read_rows([&](size_t, RowFields row) {
        if (const auto plot_id = rows.plot_id(row)) rows.add(row, plots.plots.find(*plot_id));
    });
    dataset.finish_load(rows, plots, kept);
    return dataset;
}

ResolveDataset ResolveDataset::load_table(
    RowSource& source,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config,
    const ExternalVocabs* vocabs
) {
    ResolveDataset dataset;
    dataset.config_ = config;
    const std::vector<TargetSpec> specs = dataset.begin_load(targets, vocabs);

    // The single table carries plot-level coordinates and targets on each
    // plot's first row; covariates belong in a header table. A RoleMapping
    // shared with the two-file loader is a legitimate pattern, so this warns.
    if (!roles.covariates.empty() || !roles.categoricals.empty()) {
        std::cerr << "[RESOLVE] warning: single-table (species-only) loading "
                     "ignores roles.covariates ("
                  << roles.covariates.size() << ") and roles.categoricals ("
                  << roles.categoricals.size()
                  << "); use the header+species loader for plot-level covariates"
                  << std::endl;
    }

    SpeciesRows rows(source, roles, config.zero_abundance_as);
    PlotColumns plots(source, roles, specs, /*read_covariates=*/false, "species");
    source.read_rows([&](size_t, RowFields row) {
        if (const auto plot_id = rows.plot_id(row)) rows.add(row, plots.add(row, *plot_id).first);
    });
    const std::vector<int64_t> kept = dataset.absorb_plot_columns(plots, /*n_skipped=*/0);
    dataset.finish_load(rows, plots, kept);
    return dataset;
}

// Public verbs. A CSV read is wrapped in io::with_retry<io::IOError>, so a
// transient storage fault re-runs the whole load into a fresh dataset; a
// parse/logic error propagates on the first try. A fully in-memory load does
// no I/O and is not retried.

ResolveDataset ResolveDataset::from_csv(
    const std::string& header_path,
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    return io::with_retry<io::IOError>(
        [&] {
            CSVReader header(header_path);
            CSVReader species(species_path);
            return load_tables(header, species, roles, targets, config, nullptr);
        },
        "dataset CSV load");
}

ResolveDataset ResolveDataset::from_csv_with_schema(
    const std::string& header_path,
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const ResolveDataset& schema_source,
    const DatasetConfig& config
) {
    return from_csv_with_vocabs(header_path, species_path, roles, targets,
                                schema_source.external_vocabs(), config);
}

ResolveDataset ResolveDataset::from_csv_with_schema(
    const std::string& header_path,
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const ResolveSchema& schema,
    const DatasetConfig& config
) {
    return from_csv_with_vocabs(header_path, species_path, roles, targets,
                                external_vocabs_from_schema(schema), config);
}

ResolveDataset ResolveDataset::from_csv_with_vocabs(
    const std::string& header_path,
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const ExternalVocabs& vocabs,
    const DatasetConfig& config
) {
    return io::with_retry<io::IOError>(
        [&] {
            CSVReader header(header_path);
            CSVReader species(species_path);
            return load_tables(header, species, roles, targets, config, &vocabs);
        },
        "dataset CSV load (schema)");
}

ResolveDataset ResolveDataset::from_species_csv(
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    return io::with_retry<io::IOError>(
        [&] {
            CSVReader reader(species_path);
            return load_table(reader, roles, targets, config, nullptr);
        },
        "species CSV load");
}

ResolveDataset ResolveDataset::from_species_csv_with_schema(
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const ResolveSchema& schema,
    const DatasetConfig& config
) {
    return from_species_csv_with_vocabs(species_path, roles, targets,
                                        external_vocabs_from_schema(schema), config);
}

ResolveDataset ResolveDataset::from_species_csv_with_vocabs(
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const ExternalVocabs& vocabs,
    const DatasetConfig& config
) {
    return io::with_retry<io::IOError>(
        [&] {
            CSVReader reader(species_path);
            return load_table(reader, roles, targets, config, &vocabs);
        },
        "species CSV load (schema)");
}

ResolveDataset ResolveDataset::from_dataframe(
    const ColumnTable& header,
    const ColumnTable& species,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    InMemoryRowSource header_source(header);
    InMemoryRowSource species_source(species);
    return load_tables(header_source, species_source, roles, targets, config, nullptr);
}

ResolveDataset ResolveDataset::from_dataframe_header(
    const ColumnTable& header,
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    return io::with_retry<io::IOError>(
        [&] {
            InMemoryRowSource header_source(header);
            CSVReader species(species_path);
            return load_tables(header_source, species, roles, targets, config, nullptr);
        },
        "dataset load (DataFrame header + species CSV)");
}

ResolveDataset ResolveDataset::from_dataframe_with_schema(
    const ColumnTable& header,
    const ColumnTable& species,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const ResolveDataset& schema_source,
    const DatasetConfig& config
) {
    return from_dataframe_with_vocabs(
        header, species, roles, targets, schema_source.external_vocabs(), config);
}

ResolveDataset ResolveDataset::from_dataframe_with_vocabs(
    const ColumnTable& header,
    const ColumnTable& species,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const ExternalVocabs& vocabs,
    const DatasetConfig& config
) {
    InMemoryRowSource header_source(header);
    InMemoryRowSource species_source(species);
    return load_tables(header_source, species_source, roles, targets, config, &vocabs);
}

ResolveDataset ResolveDataset::from_species_dataframe(
    const ColumnTable& species,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    InMemoryRowSource source(species);
    return load_table(source, roles, targets, config, nullptr);
}

ResolveDataset ResolveDataset::from_species_dataframe_with_vocabs(
    const ColumnTable& species,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const ExternalVocabs& vocabs,
    const DatasetConfig& config
) {
    InMemoryRowSource source(species);
    return load_table(source, roles, targets, config, &vocabs);
}

// =============================================================================
// Vocabulary reuse
// =============================================================================
//
// Every non-hash encoder indexes an embedding table by a code that is a
// function of the data its vocab was fitted on (the species vocab is ranked by
// plot count; the taxonomy vocab is a function of the name set). Scoring new
// data therefore has to reuse the TRAINING vocabularies, from whichever of the
// two carriers is available: the in-memory training dataset, or a
// ResolveSchema recovered from a checkpoint.

ExternalVocabs ResolveDataset::external_vocabs() const {
    ExternalVocabs v;
    v.species_vocab = vocab_.species.names();
    v.taxonomy = vocab_.taxonomy;
    v.categorical = categorical_vocab_;
    v.targets = target_configs_;
    v.pool_species_cap = resolved_species_width(schema_);
    return v;
}

ExternalVocabs external_vocabs_from_schema(const ResolveSchema& schema) {
    ExternalVocabs v;
    v.species_vocab = schema.species_vocab;
    v.taxonomy = TaxonomyVocab::from_ordered(schema.genus_vocab, schema.family_vocab);
    // The categorical string -> code maps are not on the schema (it carries
    // only column names + vocab sizes). A caller scoring a categorical model
    // supplies them through the *_with_vocabs overload; left empty here, the
    // loader re-fits them and Predictor::predict rejects the dataset rather
    // than scoring it against mismatched codes.
    v.targets = schema.targets;
    v.pool_species_cap = resolved_species_width(schema);
    return v;
}

DatasetConfig dataset_config_from_checkpoint(const ResolveSchema& schema,
                                             const ModelConfig& model_config) {
    DatasetConfig config = schema.dataset_config;
    // The settings that also size the model: the ModelConfig is authoritative,
    // since a disagreement would have failed at training time.
    config.species_encoding = effective_species_encoding(model_config);
    config.hash_dim = model_config.hash_dim;
    config.top_k = model_config.top_k;
    // A training-time compute path (raw CSR species data hashed per batch);
    // Predictor::predict consumes the precomputed hash embedding.
    config.use_cuda_hash = false;
    return config;
}

torch::Tensor ResolveDataset::continuous_block(bool include_hash) const {
    ContinuousInputs parts = continuous_inputs();
    if (!include_hash) parts.hash_embedding = torch::Tensor();
    return assemble_continuous(parts, schema_.dataset_config.missing_values, schema_.n_plots);
}

ContinuousInputs ResolveDataset::continuous_inputs() const {
    return {coordinates_, covariates_, unknown_fraction_, unknown_count_, hash_embedding_};
}

PlotInputs ResolveDataset::plot_inputs() const {
    PlotInputs inputs;
    inputs.genus_ids = genus_ids_;
    inputs.family_ids = family_ids_;
    inputs.species_ids = species_ids_;
    inputs.species_vector = species_vector_;
    inputs.pool_genus_ids = pool_genus_ids_;
    inputs.pool_family_ids = pool_family_ids_;
    inputs.pool_weights = pool_weights_;
    inputs.pool_mask = pool_mask_;
    inputs.pool_has_cover = pool_has_cover_;
    inputs.categorical_ids = categorical_ids_;
    return inputs;
}

SelectionMode effective_selection(const DatasetConfig& config) {
    switch (config.species_encoding) {
        case SpeciesEncodingMode::Hash:
        case SpeciesEncodingMode::Embed:
            // Fixed per-plot width (top_k / top_k_species), so a selection is
            // always applied.
            return config.selection;
        case SpeciesEncodingMode::RankPool:
        case SpeciesEncodingMode::Transformer:
        case SpeciesEncodingMode::Sparse:
            break;
    }
    // Variable width: without a budget every species is encoded, which is All
    // however the config is spelled.
    return config.species_budget > 0 ? config.selection : SelectionMode::All;
}

void ResolveDataset::adopt_vocabs(const ExternalVocabs& vocabs) {
    if (vocabs.species_vocab.empty()) {
        throw std::runtime_error(
            "ResolveDataset: the supplied vocabularies carry no species vocab, so "
            "the dataset would re-fit its own species codes and index the model's "
            "embedding tables with the wrong rows. A checkpoint that stores only "
            "the vocabulary SIZES cannot be scored this way; retrain or re-save the "
            "model with a current build, or pass the training ResolveDataset to "
            "from_csv_with_schema instead.");
    }
    use_external_vocabs_ = true;
    categorical_vocab_ = vocabs.categorical;
    vocab_.species = SpeciesVocab::from_ordered(vocabs.species_vocab);
    vocab_.taxonomy = vocabs.taxonomy;
    // The source's resolved width replaces whatever cap the caller's config
    // asked for, so a plot keeps the species the model was trained to read.
    if (vocabs.pool_species_cap > 0) {
        config_.pool_species_cap = vocabs.pool_species_cap;
    }
}

std::vector<TargetSpec> ResolveDataset::replay_class_mappings(
    const std::vector<TargetSpec>& targets,
    const std::vector<TargetConfig>& source_configs
) {
    // A classification target takes its class mapping from the source's
    // class_names, so labels encode to the training codes instead of a
    // factorization of the new data's own values. A spec that already carries
    // a mapping keeps it.
    std::vector<TargetSpec> out = targets;
    for (auto& spec : out) {
        if (spec.task != TaskType::Classification) continue;
        if (!spec.class_mapping.empty()) continue;
        const std::string name = target_name_of(spec);
        for (const auto& src_cfg : source_configs) {
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
    return out;
}

// =============================================================================
// Species encoding
// =============================================================================

void ResolveDataset::encode_species(const PlotAssemblages& assemblages) {
    record_encoding_settings();
    fit_vocab(assemblages);

    using EncodeFn = void (ResolveDataset::*)(const PlotAssemblages&);
    struct Encoder {
        SpeciesEncodingMode mode;
        EncodeFn encode;
    };
    static constexpr Encoder kEncoders[] = {
        {SpeciesEncodingMode::Hash, &ResolveDataset::encode_hash},
        {SpeciesEncodingMode::Embed, &ResolveDataset::encode_embed},
        {SpeciesEncodingMode::Sparse, &ResolveDataset::encode_sparse},
        {SpeciesEncodingMode::RankPool, &ResolveDataset::encode_pool},
        {SpeciesEncodingMode::Transformer, &ResolveDataset::encode_pool},
    };
    const auto* encoder = std::find_if(std::begin(kEncoders), std::end(kEncoders),
        [&](const Encoder& e) { return e.mode == config_.species_encoding; });
    if (encoder == std::end(kEncoders)) {
        throw std::invalid_argument("ResolveDataset: unsupported species encoding");
    }
    (this->*(encoder->encode))(assemblages);

    publish_species_statistics(assemblages);
    publish_vocabs_to_schema();
}

void ResolveDataset::record_encoding_settings() {
    // The loader settings, read back by dataset_config_from_checkpoint() so an
    // inference run encodes new plots the same way. The selection is the one
    // this load APPLIES (All for a pooled / sparse encoding without a
    // species_budget), so a checkpoint never reports a selection the run did
    // not make; pool_species_cap is refined to the resolved width by the
    // rank-pool encoding.
    schema_.dataset_config = config_;
    schema_.dataset_config.selection = effective_selection(config_);
    schema_.unrecorded_dataset_settings.clear();
}

void ResolveDataset::fit_vocab(const PlotAssemblages& assemblages) {
    // One vocabulary per dataset, fitted here or adopted before the load, and
    // handed to whichever encoding runs. Taxonomy is fitted only when the
    // records name a genus or family and the config asks for it.
    if (!use_external_vocabs_) {
        vocab_ = EncodingVocab::fit(assemblages, /*min_count=*/1,
                                    assemblages.names_taxa() && config_.use_taxonomy);
    }
    schema_.has_abundance = has_abundance_column_;
    schema_.n_species = vocab_.species.size() - 1;
    schema_.n_species_vocab = vocab_.species.size();
    schema_.has_taxonomy = config_.use_taxonomy && vocab_.has_taxonomy();
    if (schema_.has_taxonomy) {
        schema_.n_genera = vocab_.taxonomy.n_genera();
        schema_.n_families = vocab_.taxonomy.n_families();
    }
}

void ResolveDataset::encode_hash(const PlotAssemblages& assemblages) {
    const int64_t n_plots = assemblages.n_plots();
    if (config_.use_cuda_hash) {
        // Raw species data in CSR form for the per-batch GPU hash: plot p's
        // selected species are entries plot_offsets_[p] .. plot_offsets_[p + 1].
        std::vector<int64_t> species_keys(static_cast<size_t>(assemblages.n_species()));
        for (int32_t s = 0; s < assemblages.n_species(); ++s) {
            species_keys[static_cast<size_t>(s)] = feature_hash_key(assemblages.species_name(s));
        }
        std::vector<int64_t> offsets{0};
        offsets.reserve(static_cast<size_t>(n_plots) + 1);
        std::vector<int64_t> keys;
        std::vector<float> weights;
        for (int64_t p = 0; p < n_plots; ++p) {
            const auto species = assemblages.species(p);
            const auto selected = selection_indices(assemblages, p, config_.selection, config_.top_k);
            const auto values = normalized_values(assemblages, p, selected, config_.normalization);
            for (size_t j = 0; j < selected.size(); ++j) {
                keys.push_back(species_keys[static_cast<size_t>(species[selected[j]])]);
                weights.push_back(values[j]);
            }
            offsets.push_back(static_cast<int64_t>(keys.size()));
        }
        plot_offsets_ = owning_tensor(std::move(offsets), torch::kInt64);
        raw_species_ids_ = owning_tensor(std::move(keys), torch::kInt64);
        raw_weights_ = owning_tensor(std::move(weights), torch::kFloat32);
        hash_embedding_ = torch::Tensor();
    } else {
        // Each species' slot computed once, not once per record.
        std::vector<HashBucketSign> slots(static_cast<size_t>(assemblages.n_species()));
        for (int32_t s = 0; s < assemblages.n_species(); ++s) {
            slots[static_cast<size_t>(s)] = feature_hash_bucket_sign(
                assemblages.species_name(s), static_cast<int32_t>(config_.hash_dim));
        }
        hash_embedding_ = torch::zeros({n_plots, config_.hash_dim}, torch::kFloat32);
        auto rows = hash_embedding_.accessor<float, 2>();
        for (int64_t p = 0; p < n_plots; ++p) {
            const auto species = assemblages.species(p);
            const auto selected = selection_indices(assemblages, p, config_.selection, config_.top_k);
            const auto values = normalized_values(assemblages, p, selected, config_.normalization);
            for (size_t j = 0; j < selected.size(); ++j) {
                const HashBucketSign& slot = slots[static_cast<size_t>(species[selected[j]])];
                rows[p][slot.bucket] += slot.sign * values[j];
            }
        }
    }
    encode_fixed_taxonomy(assemblages);
}

void ResolveDataset::encode_sparse(const PlotAssemblages& assemblages) {
    // Each plot's budgeted species as one sparse row: the species codes in
    // increasing order and the values they carry. A species the vocabulary
    // does not hold, and a value of 0, store nothing.
    const int64_t n_plots = assemblages.n_plots();
    const std::vector<int64_t> codes = assemblages.species_codes(vocab_.species);
    std::vector<int64_t> offsets{0};
    offsets.reserve(static_cast<size_t>(n_plots) + 1);
    std::vector<int64_t> columns;
    std::vector<float> values;
    columns.reserve(static_cast<size_t>(assemblages.n_entries()));
    values.reserve(static_cast<size_t>(assemblages.n_entries()));
    std::vector<std::pair<int64_t, float>> row;
    for (int64_t p = 0; p < n_plots; ++p) {
        const auto species = assemblages.species(p);
        const auto ranking = assemblages.ranking(p);
        row.clear();
        for (size_t i : species_budget_indices(assemblages, p, config_.selection,
                                               config_.species_budget)) {
            const int64_t code = codes[static_cast<size_t>(species[i])];
            const float value = assemblages.value(ranking[i]);
            if (code > 0 && value != 0.0f) row.emplace_back(code, value);
        }
        std::sort(row.begin(), row.end());
        for (const auto& [code, value] : row) {
            columns.push_back(code);
            values.push_back(value);
        }
        offsets.push_back(static_cast<int64_t>(columns.size()));
    }
    species_vector_ = make_sparse_rows(owning_tensor(std::move(offsets), torch::kInt64),
                                       owning_tensor(std::move(columns), torch::kInt64),
                                       owning_tensor(std::move(values), torch::kFloat32),
                                       vocab_.species.size());
    encode_fixed_taxonomy(assemblages);
}

void ResolveDataset::encode_embed(const PlotAssemblages& assemblages) {
    // The species slots take top_k_species under the configured selection;
    // the taxonomy slots match the hash / sparse width.
    EmbeddingEncoder encoder(config_.top_k_species, taxonomy_slot_count(config_),
                             config_.selection, config_.aggregation);
    encoder.set_vocab(vocab_);
    auto encoded = encoder.transform(assemblages);
    species_ids_ = std::move(encoded.species_ids);
    if (schema_.has_taxonomy) {
        genus_ids_ = std::move(encoded.genus_ids);
        family_ids_ = std::move(encoded.family_ids);
    }
}

void ResolveDataset::encode_pool(const PlotAssemblages& assemblages) {
    // Per-species IDs, genus / family IDs, weights and mask, padded to the
    // widest plot or the species cap:
    //   species_ids_     int64 (n_plots, max_species)
    //   pool_genus_ids_  int64 (n_plots, max_species)
    //   pool_family_ids_ int64 (n_plots, max_species)
    //   pool_weights_    f32   (n_plots, max_species), see PoolWeighting
    //   pool_mask_       bool  (n_plots, max_species), true where a species sits
    //   pool_has_cover_  f32   (n_plots,)
    RankPoolEncoder encoder(config_.pool_weighting, /*min_frequency=*/1,
                            config_.selection, config_.species_budget);
    encoder.set_vocab(vocab_);
    auto encoded = encoder.transform(assemblages, config_.pool_species_cap,
                                     has_abundance_column_);
    species_ids_ = std::move(encoded.species_ids);
    pool_genus_ids_ = std::move(encoded.genus_ids);
    pool_family_ids_ = std::move(encoded.family_ids);
    pool_weights_ = std::move(encoded.weights);
    pool_mask_ = std::move(encoded.mask);
    pool_has_cover_ = std::move(encoded.has_cover);
    // The resolved width, so inference truncates every plot to the width the
    // model was trained on even when the cap was a percentile.
    schema_.dataset_config.pool_species_cap = static_cast<int>(species_ids_.size(1));
}

void ResolveDataset::encode_fixed_taxonomy(const PlotAssemblages& assemblages) {
    if (!schema_.has_taxonomy) return;
    auto slots = encode_taxonomy_slots(assemblages, vocab_.taxonomy, config_.aggregation,
                                       taxonomy_slot_count(config_));
    genus_ids_ = std::move(slots.genus_ids);
    family_ids_ = std::move(slots.family_ids);
}

void ResolveDataset::publish_species_statistics(const PlotAssemblages& assemblages) {
    // Unknown-species feature columns, measured against the vocabulary this
    // dataset encodes with, in every encoding. A vocabulary fitted on these
    // records covers every name in them, so a training load reports 0: the
    // columns become informative when the vocabulary came from elsewhere.
    if (config_.track_unknown_fraction || config_.track_unknown_count) {
        auto stats = compute_unknown_species_stats(assemblages, vocab_.species);
        if (config_.track_unknown_fraction) unknown_fraction_ = std::move(stats.fraction);
        if (config_.track_unknown_count) unknown_count_ = std::move(stats.count);
    }

    // How much of each plot an adopted vocabulary recognises, reported beside
    // a prediction; a vocabulary fitted on these records recognises them all.
    if (use_external_vocabs_) {
        species_recognition_ = compute_species_recognition(assemblages, vocab_.species);
    }

    // The genus and family each species CODE belongs to: how the species
    // relate to each other, which a species-level graph is built from
    // (species_graph.hpp).
    if (schema_.has_taxonomy && vocab_.species.size() > 1) {
        auto taxonomy = resolve_species_taxonomy(assemblages, vocab_.species, vocab_.taxonomy);
        species_genus_ids_ = torch::tensor(taxonomy.genus, torch::kInt64);
        species_family_ids_ = torch::tensor(taxonomy.family, torch::kInt64);
    }
}

void ResolveDataset::publish_vocabs_to_schema() {
    schema_.species_vocab = vocab_.species.names();
    schema_.genus_vocab = vocab_.taxonomy.genus_names();
    schema_.family_vocab = vocab_.taxonomy.family_names();
}

std::vector<int64_t> ResolveDataset::plot_indices(const std::vector<std::string>& ids) const {
    std::unordered_map<std::string, int64_t> position;
    position.reserve(plot_ids_.size());
    for (size_t i = 0; i < plot_ids_.size(); ++i) {
        position.emplace(plot_ids_[i], static_cast<int64_t>(i));
    }
    std::vector<int64_t> out;
    out.reserve(ids.size());
    for (const auto& id : ids) {
        auto it = position.find(id);
        if (it == position.end()) {
            throw std::invalid_argument(
                "plot '" + id + "' is not among the dataset's " +
                std::to_string(plot_ids_.size()) + " plots (absent from the "
                "data, or dropped because a target label was missing)");
        }
        out.push_back(it->second);
    }
    return out;
}

} // namespace resolve
