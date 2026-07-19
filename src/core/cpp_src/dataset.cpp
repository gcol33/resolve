#include "resolve/dataset.hpp"
#include "resolve/csv_reader.hpp"
#include "resolve/csv_utils.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/io_retry.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <numeric>
#include <cmath>
#include <iostream>
#include <functional>
#include <optional>
#include <unordered_set>

namespace resolve {

namespace {

// Missing-cell detection is the single canonical `resolve::is_na_string`
// (declared in categorical.hpp), so target / covariate / categorical columns
// all classify the same raw cell identically.

// Parse a regression target string. Returns nullopt for empty / NA marker /
// unparseable / non-finite values. Distinct from `safe_stof`, which silently
// maps every failure to 0.0f and would conflate "missing" with "actual zero".
std::optional<float> parse_regression_target(const std::string& s) {
    if (is_na_string(s)) return std::nullopt;
    auto v = parse_float_strict(s);  // locale-free, rejects trailing garbage
    if (!v || !std::isfinite(*v)) return std::nullopt;
    return v;
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
// Upper bound on a directly-used integer class code. A code at or above this
// would size class_names to ~that many entries (unbounded allocation / OOM) for
// only a handful of classes; a code beyond it falls back to a compact
// factorization (auto path) or is rejected (explicit path).
constexpr int64_t kMaxDirectClassCode = 1 << 20;  // 1,048,575

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

    // Try strict-int path first.
    std::vector<int64_t> parsed;
    parsed.reserve(uniques.size());
    bool all_int = true;
    for (const auto& v : uniques) {
        auto p = parse_strict_int64(v);
        if (!p) { all_int = false; break; }
        parsed.push_back(*p);
    }

    // Use the parsed integers directly as class codes ONLY when they are
    // non-negative and reasonably dense. A negative label would index
    // class_names_out at static_cast<size_t>(negative) == a huge value (OOB
    // heap write); a very large label (e.g. an ID like 9999999999) would size
    // class_names_out to ~that value (unbounded allocation / OOM) even for a
    // handful of classes. In either case fall back to a compact 0..K-1
    // factorization of the sorted unique labels.
    bool use_direct_codes = false;
    if (all_int) {
        int64_t min_code = *std::min_element(parsed.begin(), parsed.end());
        int64_t max_code = *std::max_element(parsed.begin(), parsed.end());
        use_direct_codes = (min_code >= 0 && max_code < kMaxDirectClassCode);
    }

    if (use_direct_codes) {
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

// Resolve a classification target's (mapping, class_names): use the caller's
// explicit class_mapping if non-empty (class_names indexed by code, with gaps
// for sparse codes), otherwise fit from the raw string labels. Single source
// for both the single-table and header+species loaders.
void resolve_classification_mapping(
    const std::unordered_map<std::string, int64_t>& explicit_mapping,
    const std::vector<std::string>& raw_labels,
    std::unordered_map<std::string, int64_t>& mapping_out,
    std::vector<std::string>& class_names_out
) {
    if (!explicit_mapping.empty()) {
        mapping_out = explicit_mapping;
        int64_t max_code = -1;
        for (const auto& [name, c] : mapping_out) {
            // A negative code is written verbatim into the int64 target tensor
            // and later indexes the classification head out of bounds
            // (static_cast<size_t> of a negative -> huge). Reject it here rather
            // than only skipping it for class_names (issue #74).
            if (c < 0) {
                throw std::invalid_argument(
                    "class_mapping code for '" + name + "' is negative (" +
                    std::to_string(c) + "); class codes must be >= 0");
            }
            if (c > max_code) max_code = c;
        }
        // Bound the allocation exactly like the auto path: an ID-like code
        // (e.g. 9999999999) would size class_names to ~10^10 entries -> OOM.
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
    } else {
        fit_classification_mapping(raw_labels, mapping_out, class_names_out);
    }
}

// Like resolve_classification_mapping, but for the auto-fit branch it fits the
// vocabulary from KEPT rows only (issue #84): a class value that occurs solely in
// rows that will be dropped because a *different* target is missing must not enter
// class_names, since that both inflates num_classes with a zero-example class and
// (for the lexicographic factorization) shifts every other code. Explicit mappings
// are caller-fixed and therefore unaffected. `keep` is parallel to `raw_labels`.
void resolve_classification_mapping_kept(
    const std::unordered_map<std::string, int64_t>& explicit_mapping,
    const std::vector<std::string>& raw_labels,
    const std::vector<char>& keep,
    std::unordered_map<std::string, int64_t>& mapping_out,
    std::vector<std::string>& class_names_out
) {
    if (!explicit_mapping.empty()) {
        resolve_classification_mapping(explicit_mapping, raw_labels, mapping_out, class_names_out);
        return;
    }
    std::vector<std::string> kept;
    kept.reserve(raw_labels.size());
    for (size_t i = 0; i < raw_labels.size(); ++i) {
        if (i < keep.size() && !keep[i]) continue;
        kept.push_back(raw_labels[i]);
    }
    fit_classification_mapping(kept, mapping_out, class_names_out);
}

// Single source of truth for encoding one classification target, shared by both
// the header (from_csv) and species-file (from_species_csv) loaders. Resolves the
// mapping (fit on kept rows), writes the (n,) int64 code tensor, flips keep[i]=0
// for NA/unmapped cells, and sets + validates cfg.class_names / cfg.num_classes.
//
// num_classes validation (issue #79): the direct-integer path emits the parsed
// integers verbatim as class codes, so class_names.size() == max_code + 1. The
// classification head is sized from num_classes, so a caller-supplied num_classes
// smaller than class_names.size() (e.g. a 1-indexed column 1..9 built with
// num_classes=9) would let a target code index the head out of bounds. Fill it
// when the caller passed 0; reject a positive-but-too-small value loudly.
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
    resolve_classification_mapping_kept(spec.class_mapping, raw, keep, mapping, class_names);

    const int64_t n = static_cast<int64_t>(raw.size());
    auto tensor = torch::zeros({n}, torch::kLong);
    auto acc = tensor.accessor<int64_t, 1>();
    int64_t n_null = 0;
    for (int64_t i = 0; i < n; ++i) {
        const std::string& v = raw[static_cast<size_t>(i)];
        if (is_na_string(v)) {
            if (i < static_cast<int64_t>(keep.size()) && keep[i]) keep[i] = 0;
            ++n_null;
            continue;
        }
        auto it = mapping.find(v);
        if (it == mapping.end()) {
            if (i < static_cast<int64_t>(keep.size()) && keep[i]) keep[i] = 0;
            ++n_null;
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
              << mapping.size() << " classes, " << n_null << " null)" << std::endl;
    return tensor;
}

// Finalize the keep flags for a classification target WITHOUT fitting the vocab
// (issue #84): so that every classification vocab is fit against the final
// surviving-row set even when another classification target drives additional
// drops. Auto-fit maps every non-NA cell in a kept row, so only NA cells drop;
// an explicit mapping additionally drops non-NA cells absent from the mapping.
void mark_classification_drops(
    const TargetSpec& spec,
    const std::vector<std::string>& raw,
    std::vector<char>& keep
) {
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

// Drop species records for plots that were removed (missing/NA target) so the
// species / taxonomy vocab is built only from surviving plots (issue #68):
// species occurring solely in dropped plots must not inflate the embedding
// tables with never-referenced rows. `kept_ids` is the compacted plot_ids_.
void filter_records_to_plots(
    std::unordered_map<std::string, std::vector<SpeciesRecord>>& plot_records,
    const std::vector<std::string>& kept_ids
) {
    // Filter by set membership, not a size comparison: plot_records (keyed from
    // the species file) and kept_ids (surviving header plots) are independent
    // key sets, so a size guard like `plot_records.size() <= kept_ids.size()`
    // wrongly skips filtering when the species file references a plot absent
    // from the kept set while staying within the count (e.g. one phantom plot,
    // no header plots dropped) -- that plot's species then inflate and shift the
    // frequency-sorted vocab (issue #71, the #68 class via the fast-path guard).
    std::unordered_set<std::string> kept(kept_ids.begin(), kept_ids.end());
    if (plot_records.size() == kept.size()) {
        // Same count: a full pass is only skippable when every key is kept.
        bool all_kept = true;
        for (const auto& [id, _] : plot_records) {
            if (kept.find(id) == kept.end()) { all_kept = false; break; }
        }
        if (all_kept) return;
    }
    for (auto it = plot_records.begin(); it != plot_records.end();) {
        if (kept.find(it->first) == kept.end()) it = plot_records.erase(it);
        else ++it;
    }
}

}  // namespace


// ColumnIndices implementation
ColumnIndices ColumnIndices::from_source(const RowSource& source, const RoleMapping& roles,
                                         bool expect_coordinates) {
    ColumnIndices idx;
    idx.plot = source.column_index(roles.plot_id);
    idx.species = source.column_index(roles.species_id);

    // A named optional-role column that the source cannot resolve is a
    // configuration error (typically a typo'd role name), not a silent "no such
    // feature". The header loader (load_header_data) already throws loudly for
    // an absent covariate/coordinate/target column; without the same guard here
    // a species-table role typo (roles.genus = "genuss") would flip
    // has_taxonomy/has_coordinates to false with no signal and bake the wrong
    // feature count into the checkpoint (issue #94). A role left unset
    // (nullopt) still resolves to -1 without throwing.
    auto resolve_role = [&source](const std::optional<std::string>& name,
                                  const char* role) -> int {
        if (!name) return -1;
        int col = source.column_index(*name);
        if (col < 0) {
            throw std::runtime_error(
                std::string("Role column not found in species CSV: ") + role +
                " = \"" + *name + "\"");
        }
        return col;
    };
    // abundance / genus / family live in the species source for BOTH the
    // single-table and two-file loaders, so they are always guarded here.
    idx.abundance = resolve_role(roles.abundance, "abundance");
    idx.genus = resolve_role(roles.genus, "genus");
    idx.family = resolve_role(roles.family, "family");
    // Coordinates live in the species source only for the single-table loader
    // (expect_coordinates); in the two-file loader they are header roles that
    // load_header_data already validated, so looking them up in the species
    // source would wrongly reject the normal case. Left unresolved (-1) there.
    if (expect_coordinates) {
        idx.longitude = resolve_role(roles.longitude, "longitude");
        idx.latitude = resolve_role(roles.latitude, "latitude");
    }
    return idx;
}


// --- Public loader verbs: thin io::with_retry<io::IOError> wrappers over the
// single-attempt *_impl bodies. A transient storage read fault (issue #20) is
// retried with backoff into a fresh dataset; a parse/logic error (not
// io::IOError) propagates on the first try, so a permanent fault never re-reads
// a multi-GB file. ---

ResolveDataset ResolveDataset::from_csv(
    const std::string& header_path,
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    return io::with_retry<io::IOError>(
        [&] { return from_csv_impl(header_path, species_path, roles, targets, config); },
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
    return io::with_retry<io::IOError>(
        [&] {
            return from_csv_with_schema_impl(
                header_path, species_path, roles, targets, schema_source, config);
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
        [&] { return from_species_csv_impl(species_path, roles, targets, config); },
        "species CSV load");
}

// --- In-memory (DataFrame) loaders (issue #22). These share the exact loader
// bodies as the CSV verbs via the RowSource seam; the only difference is the row
// provider (InMemoryRowSource over a ColumnTable instead of a CSVReader). A
// fully in-memory load does no I/O, so it needs no io::with_retry; the mixed
// header-frame + species-CSV verb retries only the CSV read. ---

ResolveDataset ResolveDataset::from_dataframe(
    const ColumnTable& header,
    const ColumnTable& species,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    ResolveDataset dataset;
    dataset.config_ = config;

    InMemoryRowSource header_source(header);
    dataset.load_header_data(header_source, roles, targets);

    InMemoryRowSource species_source(species);
    dataset.load_species_data(species_source, roles);

    return dataset;
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
            ResolveDataset dataset;
            dataset.config_ = config;

            InMemoryRowSource header_source(header);
            dataset.load_header_data(header_source, roles, targets);

            CSVReader species_reader(species_path);
            dataset.load_species_data(species_reader, roles);

            return dataset;
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
    InMemoryRowSource header_source(header);
    InMemoryRowSource species_source(species);
    return load_with_schema(
        header_source, species_source, roles, targets, schema_source, config);
}

ResolveDataset ResolveDataset::from_species_dataframe(
    const ColumnTable& species,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    InMemoryRowSource source(species);
    return from_species_source(source, roles, targets, config);
}

ResolveDataset ResolveDataset::from_csv_impl(
    const std::string& header_path,
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    ResolveDataset dataset;
    dataset.config_ = config;

    // Load header data (coordinates, covariates, targets)
    CSVReader header_reader(header_path);
    dataset.load_header_data(header_reader, roles, targets);

    // Load species data
    CSVReader species_reader(species_path);
    dataset.load_species_data(species_reader, roles);

    return dataset;
}

ResolveDataset ResolveDataset::from_csv_with_schema_impl(
    const std::string& header_path,
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const ResolveDataset& schema_source,
    const DatasetConfig& config
) {
    CSVReader header_reader(header_path);
    CSVReader species_reader(species_path);
    return load_with_schema(
        header_reader, species_reader, roles, targets, schema_source, config);
}

ResolveDataset ResolveDataset::load_with_schema(
    RowSource& header,
    RowSource& species,
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

    dataset.load_header_data(header, roles, targets_with_mappings);
    dataset.load_species_data(species, roles);

    return dataset;
}

ResolveDataset ResolveDataset::from_species_csv_impl(
    const std::string& species_path,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    CSVReader reader(species_path);
    return from_species_source(reader, roles, targets, config);
}

// Build a SpeciesRecord from one table row using the resolved column indices.
// Single source of truth shared by from_species_source and load_species_data.
// The caller must have bounds-checked the plot and species columns already.
// When an abundance column is present but a cell is missing/NA/unparseable, the
// weight defaults to 1.0 and *abundance_coerced (if given) is incremented so the
// caller can warn -- mirroring the loud coordinate/covariate coercion warnings,
// rather than silently conflating missing cover with a real presence (issue #94).
static SpeciesRecord make_species_record(const std::vector<std::string>& row,
                                         const ColumnIndices& cols,
                                         int64_t* abundance_coerced = nullptr) {
    SpeciesRecord record;
    record.plot_id = row[cols.plot];
    record.species_id = row[cols.species];
    if (cols.abundance >= 0 && row.size() > static_cast<size_t>(cols.abundance)) {
        auto parsed = parse_regression_target(row[cols.abundance]);
        if (parsed.has_value()) {
            record.abundance = *parsed;
        } else {
            record.abundance = 1.0f;
            if (abundance_coerced) (*abundance_coerced)++;
        }
    } else {
        record.abundance = 1.0f;
    }
    if (cols.genus >= 0 && row.size() > static_cast<size_t>(cols.genus)) {
        record.genus = row[cols.genus];
    }
    if (cols.family >= 0 && row.size() > static_cast<size_t>(cols.family)) {
        record.family = row[cols.family];
    }
    return record;
}

// Emit the abundance-coercion warning shared by both species loaders.
static void warn_abundance_coercions(int64_t abundance_coerced) {
    if (abundance_coerced > 0) {
        std::cerr << "[RESOLVE] warning: " << abundance_coerced
                  << " abundance cell(s) were missing/NA/unparseable and "
                     "defaulted to weight 1.0; these are treated as real "
                     "presences, not missing cover"
                  << std::endl;
    }
}

ResolveDataset ResolveDataset::from_species_source(
    RowSource& reader,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets,
    const DatasetConfig& config
) {
    ResolveDataset dataset;
    dataset.config_ = config;

    // Find column indices. The single-table loader reads coordinates from the
    // species source, so a named longitude/latitude role must resolve here too.
    auto cols = ColumnIndices::from_source(reader, roles, /*expect_coordinates=*/true);
    if (cols.plot < 0 || cols.species < 0) {
        throw std::runtime_error("Required columns not found: plot_id or species_id");
    }

    // The single-long-table loader reads only plot-level coordinates/targets
    // from each plot's first occurrence; it does not populate covariates or
    // categorical covariates. Warn rather than silently discard requested roles
    // (a shared RoleMapping reused from the two-file loader is a legitimate
    // pattern, so this is a warning, not an error). Use from_csv (separate
    // header + species tables) when plot-level covariates are needed.
    if (!roles.covariates.empty() || !roles.categoricals.empty()) {
        std::cerr << "[RESOLVE] warning: single-table (species-only) loading "
                     "ignores roles.covariates ("
                  << roles.covariates.size() << ") and roles.categoricals ("
                  << roles.categoricals.size()
                  << "); use the header+species loader for plot-level covariates"
                  << std::endl;
    }

    // Find target columns. A missing target column must fail loudly here too;
    // col == -1 would otherwise silently drop every plot's target value.
    std::vector<int> target_cols;
    for (const auto& target : targets) {
        int col = reader.column_index(target.column_name);
        if (col < 0) {
            throw std::runtime_error("Target column not found: " + target.column_name);
        }
        target_cols.push_back(col);
    }

    // First pass: collect all unique plots and their species
    std::unordered_map<std::string, std::vector<SpeciesRecord>> plot_records;
    std::unordered_map<std::string, std::pair<float, float>> plot_coords;
    std::unordered_map<std::string, std::vector<std::string>> plot_targets;
    std::unordered_set<std::string> seen_plots;
    int64_t coord_na_count = 0;  // missing/unparseable coords coerced to (0,0)
    int64_t abundance_coerced = 0;  // missing/unparseable abundance coerced to 1.0

    reader.read_rows([&](size_t, const std::vector<std::string>& row) {
        if (row.size() <= static_cast<size_t>(std::max({cols.plot, cols.species}))) {
            return;  // Skip malformed rows
        }

        std::string plot_id = row[cols.plot];

        SpeciesRecord record = make_species_record(row, cols, &abundance_coerced);
        plot_records[plot_id].push_back(record);

        // Extract plot-level data from first occurrence
        if (seen_plots.find(plot_id) == seen_plots.end()) {
            seen_plots.insert(plot_id);
            dataset.plot_ids_.push_back(plot_id);

            // Coordinates. NA-aware parse so a missing/unparseable cell is not
            // silently read as a real (0, 0) location (issue #46).
            if (cols.longitude >= 0 && cols.latitude >= 0 &&
                row.size() > static_cast<size_t>(std::max(cols.longitude, cols.latitude))) {
                auto lon = parse_regression_target(row[cols.longitude]);
                auto lat = parse_regression_target(row[cols.latitude]);
                plot_coords[plot_id] = {lon.value_or(0.0f), lat.value_or(0.0f)};
                if (!lon.has_value() || !lat.has_value()) {
                    coord_na_count++;
                }
            }

            // Targets — stash the raw cell strings; classification
            // factorization and regression NaN-parsing happen post-scan
            // (mirrors load_header_data) so string-coded classes are encoded
            // against the full value distribution instead of collapsing to 0.
            std::vector<std::string> target_values;
            target_values.reserve(target_cols.size());
            for (int col : target_cols) {
                // target_cols entries are validated >= 0 (throw above), so the
                // only miss here is a row too short to hold the column.
                if (row.size() > static_cast<size_t>(col)) {
                    target_values.push_back(row[col]);
                } else {
                    target_values.emplace_back();  // missing -> dropped post-scan
                }
            }
            plot_targets[plot_id] = std::move(target_values);
        }
    });

    if (coord_na_count > 0) {
        std::cerr << "[RESOLVE] warning: " << coord_na_count
                  << " plot(s) had a missing/unparseable coordinate coerced to "
                     "(0, 0); spatial models will treat these as a real location"
                  << std::endl;
    }
    warn_abundance_coercions(abundance_coerced);

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

    // Build target tensors with the same semantics as load_header_data:
    // classification targets are factorized against the full value distribution
    // (explicit class_mapping or auto-fit -> codes + class_names), regression
    // targets are NaN-parsed, and any plot whose target is missing/unmapped is
    // dropped post-build (any-target semantics).
    std::vector<char> keep(static_cast<size_t>(n_plots), 1);

    // Raw cell for target t at plot i, in plot order.
    auto raw_at = [&](size_t t, int64_t i) -> const std::string& {
        static const std::string kEmpty;
        const auto& vals = plot_targets[dataset.plot_ids_[i]];
        return t < vals.size() ? vals[t] : kEmpty;
    };

    // Build target configs (in target order) + regression tensors; collect the
    // raw classification cells. Classification encoding is deferred to a second
    // pass so every vocab is fit from the final surviving-row set (issue #84),
    // through the same encode_classification_target used by the header loader
    // (single source of truth + num_classes validation, issues #79/#86).
    std::vector<std::vector<std::string>> cls_raw(targets.size());
    for (size_t t = 0; t < targets.size(); ++t) {
        const auto& target_spec = targets[t];
        std::string name = target_spec.target_name.empty()
            ? target_spec.column_name : target_spec.target_name;

        dataset.target_configs_.push_back({
            name,
            target_spec.task,
            target_spec.transform,
            target_spec.num_classes,
            target_spec.weight
        });

        if (target_spec.task == TaskType::Classification) {
            cls_raw[t].reserve(static_cast<size_t>(n_plots));
            for (int64_t i = 0; i < n_plots; ++i) cls_raw[t].push_back(raw_at(t, i));
        } else {
            auto target_tensor = torch::zeros({n_plots}, torch::kFloat32);
            auto acc = target_tensor.accessor<float, 1>();
            for (int64_t i = 0; i < n_plots; ++i) {
                auto parsed = parse_regression_target(raw_at(t, i));
                if (!parsed.has_value()) { keep[i] = 0; continue; }
                acc[i] = *parsed;
            }
            dataset.targets_[name] = target_tensor;
        }
    }

    // Pass 1: finalize keep for classification NA / explicit-unmapped drops.
    for (size_t t = 0; t < targets.size(); ++t) {
        if (targets[t].task != TaskType::Classification) continue;
        mark_classification_drops(targets[t], cls_raw[t], keep);
    }
    // Pass 2: fit each classification vocab on kept rows only, encode, validate.
    for (size_t t = 0; t < targets.size(); ++t) {
        if (targets[t].task != TaskType::Classification) continue;
        std::string name = targets[t].target_name.empty()
            ? targets[t].column_name : targets[t].target_name;
        dataset.targets_[name] = encode_classification_target(
            targets[t], cls_raw[t], keep, dataset.target_configs_[t], name);
    }

    // Drop plots whose target was missing/unmapped, compacting plot_ids_,
    // coordinates, and every target tensor in lockstep. encode_species (below)
    // derives n_plots from plot_ids_, so dropping here is sufficient.
    {
        int64_t n_keep = 0;
        for (char k : keep) if (k) ++n_keep;
        if (n_keep < n_plots) {
            std::vector<int64_t> keep_idx;
            keep_idx.reserve(static_cast<size_t>(n_keep));
            for (int64_t i = 0; i < n_plots; ++i) if (keep[i]) keep_idx.push_back(i);

            std::vector<std::string> new_pids;
            new_pids.reserve(static_cast<size_t>(n_keep));
            for (int64_t i : keep_idx) new_pids.push_back(std::move(dataset.plot_ids_[i]));
            dataset.plot_ids_ = std::move(new_pids);

            auto idx_t = torch::tensor(keep_idx, torch::kInt64);
            if (dataset.coordinates_.defined())
                dataset.coordinates_ = dataset.coordinates_.index_select(0, idx_t);
            for (auto& [nm, tensor] : dataset.targets_)
                tensor = tensor.index_select(0, idx_t);

            const int64_t n_dropped = n_plots - n_keep;
            n_plots = n_keep;
            dataset.schema_.n_plots = n_plots;
            std::cout << "  Dropped " << n_dropped
                      << " plots with missing/NaN target (" << n_keep
                      << " of " << (n_keep + n_dropped) << " kept)" << std::endl;
        }
    }

    dataset.schema_.targets = dataset.target_configs_;

    dataset.has_abundance_column_ = (cols.abundance >= 0);

    // Build the vocab only from kept plots (issue #68).
    filter_records_to_plots(plot_records, dataset.plot_ids_);

    // Encode species data
    dataset.encode_species(plot_records);

    return dataset;
}

void ResolveDataset::load_header_data(
    RowSource& reader,
    const RoleMapping& roles,
    const std::vector<TargetSpec>& targets
) {
    // Find column indices
    int plot_col = reader.column_index(roles.plot_id);
    if (plot_col < 0) {
        throw std::runtime_error("Plot ID column not found: " + roles.plot_id);
    }

    // A named coordinate column that is absent from the header is a
    // configuration error, not a silent "no coordinates": failing loudly here
    // avoids training a model that quietly dropped the coordinates the user
    // asked for.
    int lon_col = -1, lat_col = -1;
    if (roles.longitude) {
        lon_col = reader.column_index(*roles.longitude);
        if (lon_col < 0) {
            throw std::runtime_error("Longitude column not found in header CSV: " + *roles.longitude);
        }
    }
    if (roles.latitude) {
        lat_col = reader.column_index(*roles.latitude);
        if (lat_col < 0) {
            throw std::runtime_error("Latitude column not found in header CSV: " + *roles.latitude);
        }
    }

    // A missing covariate column must fail loudly, not be silently dropped: a
    // typo'd name would otherwise train a model with fewer features than the
    // user requested and bake the wrong feature count into the checkpoint.
    std::vector<int> covariate_cols;
    for (const auto& cov : roles.covariates) {
        int col = reader.column_index(cov);
        if (col < 0) {
            throw std::runtime_error("Covariate column not found in header CSV: " + cov);
        }
        covariate_cols.push_back(col);
        schema_.covariate_names.push_back(cov);
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

    // A missing target column must fail loudly. Left unchecked, col == -1 makes
    // every row fail row_ok below -> "Kept 0 of N plots" with no exception, i.e.
    // the entire dataset is silently discarded far from the actual cause.
    std::vector<int> target_cols;
    for (const auto& target : targets) {
        int col = reader.column_index(target.column_name);
        if (col < 0) {
            throw std::runtime_error("Target column not found in header CSV: " + target.column_name);
        }
        target_cols.push_back(col);

        target_configs_.push_back({
            target.target_name.empty() ? target.column_name : target.target_name,
            target.task,
            target.transform,
            target.num_classes,
            target.weight
        });
    }

    // Single streaming pass: accumulate into growable buffers rather than a
    // count_rows() prepass followed by preallocated tensors written by row index.
    // The old two-pass approach read the whole (~1.9M-row) header file twice; here
    // the file is read exactly once and tensors are materialized from the buffers
    // afterward, at the true loaded-row count.
    const bool has_coords = (lon_col >= 0 && lat_col >= 0);
    if (has_coords) schema_.has_coordinates = true;
    const int64_t cov_cols = static_cast<int64_t>(covariate_cols.size());

    std::vector<float> coords_buf;   // 2 values per loaded row (lon, lat)
    std::vector<float> cov_buf;      // cov_cols values per loaded row
    // One growable float buffer per regression target (classification targets are
    // built post-scan by encode_classification_target from classification_raw).
    std::vector<std::vector<float>> reg_bufs(targets.size());
    std::vector<std::string> target_names(targets.size());
    for (size_t t = 0; t < targets.size(); ++t) {
        target_names[t] = targets[t].target_name.empty()
                              ? targets[t].column_name : targets[t].target_name;
    }

    // Per-column count of covariate cells that were missing/unparseable and
    // coerced to 0.0 during the scan (issue #32); warned about afterwards.
    std::vector<int64_t> cov_na_counts(covariate_cols.size(), 0);
    // Count of rows whose longitude/latitude was missing/unparseable and
    // coerced to 0.0 -- a real location (Gulf of Guinea) that silently corrupts
    // spatial-graph neighbours. Warned about after the scan.
    int64_t coord_na_count = 0;
    // Rows too short to even hold plot_id are skipped during the scan; count them
    // for the post-scan summary (they never enter any buffer).
    int64_t n_ragged_skipped = 0;

    schema_.targets = target_configs_;

    // Raw string buffer for each categorical column. Grows via push_back during
    // the single scan (no count prepass to reserve against); filled with "" (NA)
    // when a row is too short to hold the column.
    std::vector<std::vector<std::string>> categorical_raw(categorical_cols.size());

    // Raw string buffer for each classification target column. Same layout as
    // `categorical_raw` — we collect strings during the scan and factorize them
    // post-scan into int64 codes that get written into targets_[name].
    // Regression targets are accumulated into reg_bufs (see the row-scan body).
    std::vector<std::vector<std::string>> classification_raw(targets.size());

    // Per-row keep mask. A row is "kept" iff every requested target column
    // produced a usable value (finite numeric for regression, non-missing
    // string for classification). Rows with any missing target get dropped
    // after the scan. Mirrors the POC's `ResolveDataset.from_fast_csv`
    // NaN-target drop semantics, including the classification case (the POC
    // drops nulls produced by `_encode_categorical`).
    std::vector<char> keep_row;

    // Header rows are plot-level: plot_id must be unique. A duplicate would
    // create two plot slots that both look up the same species records, so
    // reject it (consistent with the strictness applied to duplicate columns).
    std::unordered_set<std::string> seen_plot_ids;

    int64_t row_idx = 0;
    reader.read_rows([&](size_t, const std::vector<std::string>& row) {
        if (row.size() <= static_cast<size_t>(plot_col)) {
            ++n_ragged_skipped;
            return;
        }

        const std::string& pid = row[plot_col];
        if (!seen_plot_ids.insert(pid).second) {
            throw std::runtime_error(
                "Duplicate plot_id '" + pid + "' in header data. Plot IDs must "
                "be unique (each header row is one plot).");
        }

        plot_ids_.push_back(pid);
        bool row_ok = true;

        // Coordinates. Parse with the NA-aware helper so a blank / "NA" /
        // unparseable cell is not silently read as a real (0, 0) location; count
        // the coercions and warn after the scan (issue #46). A genuine "0"
        // parses to 0.0 and is not counted.
        if (has_coords) {
            auto lon = (row.size() > static_cast<size_t>(lon_col))
                           ? parse_regression_target(row[lon_col]) : std::nullopt;
            auto lat = (row.size() > static_cast<size_t>(lat_col))
                           ? parse_regression_target(row[lat_col]) : std::nullopt;
            coords_buf.push_back(lon.value_or(0.0f));
            coords_buf.push_back(lat.value_or(0.0f));
            if (!lon.has_value() || !lat.has_value()) {
                coord_na_count++;
            }
        }

        // Covariates. Parse with the NaN-aware helper so a blank / "NA" /
        // unparseable cell is not silently read as a real 0.0 (which would bias
        // standardization). We still write a well-defined 0.0 into the slot, but
        // count the coercions per column and warn after the scan so the missing
        // values are visible rather than silent (issue #32).
        if (cov_cols > 0) {
            for (size_t i = 0; i < covariate_cols.size(); ++i) {
                int col = covariate_cols[i];
                float val = 0.0f;
                if (row.size() > static_cast<size_t>(col)) {
                    auto parsed = parse_regression_target(row[col]);
                    if (parsed.has_value()) val = *parsed;
                    else cov_na_counts[i]++;
                } else {
                    cov_na_counts[i]++;
                }
                cov_buf.push_back(val);
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

        // Targets — parse with NaN-aware helpers and mark the row for drop if
        // any target is missing/non-finite/out-of-range. Every scanned row pushes
        // exactly one value per target (a sentinel for missing) so the per-target
        // buffers stay aligned with row_idx; the post-scan compaction drops the
        // marked rows. target_cols entries are validated >= 0 above, so the only
        // miss here is a row too short to hold the column.
        for (size_t t = 0; t < targets.size(); ++t) {
            const auto& target = targets[t];
            const int col = target_cols[t];
            const bool have_cell = row.size() > static_cast<size_t>(col);

            if (target.task == TaskType::Classification) {
                // Defer string->int encoding until post-scan so the auto-fit
                // path sees the full distribution of values. A short row pushes
                // "" (missing) and drops the row.
                if (have_cell) {
                    classification_raw[t].push_back(row[col]);
                } else {
                    classification_raw[t].emplace_back();
                    row_ok = false;
                }
            } else {
                float val = 0.0f;
                if (have_cell) {
                    auto parsed = parse_regression_target(row[col]);
                    if (parsed.has_value()) val = *parsed;
                    else row_ok = false;
                } else {
                    row_ok = false;
                }
                reg_bufs[t].push_back(val);
            }
        }

        keep_row.push_back(row_ok ? 1 : 0);
        row_idx++;
    });

    const int64_t n_loaded = row_idx;   // rows actually appended during the scan

    // Materialize tensors from the streaming buffers at the true loaded-row count.
    // from_blob + clone copies out of the std::vector storage into tensor-owned
    // memory before the buffers go out of scope. Ragged rows were never appended,
    // so there are no trailing phantom zero rows to compact away (the old prealloc
    // path sized to count_rows and had to trim them).
    if (has_coords) {
        coordinates_ = (n_loaded > 0)
            ? torch::from_blob(coords_buf.data(), {n_loaded, 2}, torch::kFloat32).clone()
            : torch::zeros({0, 2}, torch::kFloat32);
    }
    if (cov_cols > 0) {
        covariates_ = (n_loaded > 0)
            ? torch::from_blob(cov_buf.data(), {n_loaded, cov_cols}, torch::kFloat32).clone()
            : torch::zeros({0, cov_cols}, torch::kFloat32);
    }
    for (size_t t = 0; t < targets.size(); ++t) {
        if (targets[t].task == TaskType::Classification) continue;
        targets_[target_names[t]] = (n_loaded > 0)
            ? torch::from_blob(reg_bufs[t].data(), {n_loaded}, torch::kFloat32).clone()
            : torch::zeros({0}, torch::kFloat32);
    }
    schema_.n_plots = n_loaded;

    // Surface covariate missingness: coercing NA/blank cells to 0.0 injects a
    // real, extreme value into standardization, so make it visible rather than
    // silent. Rows are NOT dropped here (covariates don't gate row validity the
    // way targets do); the researcher decides how to handle them upstream.
    for (size_t i = 0; i < cov_na_counts.size(); ++i) {
        if (cov_na_counts[i] > 0) {
            std::cerr << "[RESOLVE] warning: covariate column '"
                      << schema_.covariate_names[i] << "' had " << cov_na_counts[i]
                      << " missing/unparseable cell(s) coerced to 0.0" << std::endl;
        }
    }
    if (coord_na_count > 0) {
        std::cerr << "[RESOLVE] warning: " << coord_na_count
                  << " plot(s) had a missing/unparseable coordinate coerced to "
                     "(0, 0); spatial models will treat these as a real location"
                  << std::endl;
    }

    // ---- Fit + encode classification target columns ----
    // Two passes so every classification vocab is fit from the final surviving-row
    // set (issue #84): pass 1 finalizes keep_row (regression drops are already in
    // it from the scan; here we add classification NA / explicit-unmapped drops),
    // pass 2 fits each vocab on kept rows only and encodes. The shared
    // encode_classification_target also validates num_classes (issue #79) and is
    // the single source of truth with the from_species_csv loader (issue #86).
    for (size_t t = 0; t < targets.size(); ++t) {
        if (targets[t].task != TaskType::Classification) continue;
        mark_classification_drops(targets[t], classification_raw[t], keep_row);
    }
    for (size_t t = 0; t < targets.size(); ++t) {
        const auto& target = targets[t];
        if (target.task != TaskType::Classification) continue;
        std::string name = target.target_name.empty() ? target.column_name : target.target_name;
        targets_[name] = encode_classification_target(
            target, classification_raw[t], keep_row, target_configs_[t], name);
    }
    schema_.targets = target_configs_;

    // ---- Filter rows with missing targets ----
    // Compact every per-plot buffer (plot_ids, coords, covariates,
    // categorical_raw, targets) to the rows where every target produced a usable
    // value. With the single streaming pass the tensors are already sized to
    // n_loaded (ragged rows were never appended), so only target-dropped rows
    // (n_keep < n_loaded) need index_select. Loud one-line summary mirrors the
    // POC's "Filtered N species records for invalid plots" log so users see the
    // plot-count drop instead of wondering where their plots went.
    int64_t n_keep = 0;
    for (char k : keep_row) if (k) ++n_keep;
    const int64_t n_target_dropped = n_loaded - n_keep;
    const int64_t n_total = n_loaded + n_ragged_skipped;   // physical data rows

    if (n_keep < n_loaded) {
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
    }

    // Report any physical row that was dropped (missing target) or skipped
    // (ragged) so the plot-count change is never silent.
    if (n_target_dropped > 0 || n_ragged_skipped > 0) {
        std::cout << "  Kept " << n_keep << " of " << n_total << " plots (";
        bool need_sep = false;
        if (n_target_dropped > 0) {
            std::cout << n_target_dropped << " dropped for missing/NaN target ";
            // List the target column names so the user knows which target drove
            // the drop. Multiple targets => any-missing semantics.
            bool first = true;
            std::cout << "[";
            for (const auto& target : targets) {
                if (!first) std::cout << ", ";
                std::cout << "'" << target.column_name << "'";
                first = false;
            }
            std::cout << "]";
            need_sep = true;
        }
        if (n_ragged_skipped > 0) {
            if (need_sep) std::cout << "; ";
            std::cout << n_ragged_skipped << " skipped as too short to contain plot_id";
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
    RowSource& reader,
    const RoleMapping& roles
) {
    // Find column indices
    auto cols = ColumnIndices::from_source(reader, roles);
    if (cols.plot < 0 || cols.species < 0) {
        throw std::runtime_error("Required columns not found: plot_id or species_id");
    }

    // Collect species records by plot
    std::unordered_map<std::string, std::vector<SpeciesRecord>> plot_records;
    int64_t abundance_coerced = 0;

    reader.read_rows([&](size_t, const std::vector<std::string>& row) {
        if (row.size() <= static_cast<size_t>(std::max(cols.plot, cols.species))) {
            return;
        }

        std::string plot_id = row[cols.plot];
        plot_records[plot_id].push_back(make_species_record(row, cols, &abundance_coerced));
    });
    warn_abundance_coercions(abundance_coerced);

    has_abundance_column_ = (cols.abundance >= 0);

    // Build the vocab only from kept plots (issue #68): load_header_data has
    // already dropped missing-target plots and compacted plot_ids_.
    filter_records_to_plots(plot_records, plot_ids_);

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
    // Sort by frequency descending, breaking ties by species name ascending so
    // the ID assignment is a pure function of the data (independent of
    // unordered_map iteration order). Without the name tie-break, tied-frequency
    // species get IDs in nondeterministic hash order, so a checkpoint and a
    // separately-rebuilt dataset can disagree on embedding rows (cf. #5 for the
    // taxonomy vocab; SpeciesVocab::from_records already sorts by name).
    std::sort(sorted_species.begin(), sorted_species.end(),
        [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second > b.second : a.first < b.first;
        }
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

    // Record the pool weighting scheme so a checkpoint can rebuild the matching
    // inference-side DatasetConfig (issue #38). pool_species_cap is refined to
    // the resolved max-species width in the rank_pool block below.
    schema_.pool_weighting = static_cast<int>(config_.pool_weighting);
    schema_.pool_species_cap = config_.pool_species_cap;

    // Fit taxonomy vocabulary. Rank-pool / transformer modes rebuild taxonomy
    // from the RankPoolEncoder's own vocab further down (and overwrite these
    // schema fields), so skip the O(n_records) fit here for them.
    const bool pool_taxonomy_mode =
        (config_.species_encoding == SpeciesEncodingMode::RankPool ||
         config_.species_encoding == SpeciesEncodingMode::Transformer);
    if (schema_.has_taxonomy && !pool_taxonomy_mode) {
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
                    // Feature-hash the species name to an int64 seed; the GPU
                    // kernel applies the MurmurHash3 finalizer to this per batch.
                    species_id_acc[record_idx] = static_cast<int64_t>(feature_hash(sp_name));
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
        // Learnable embeddings for top-k species + fixed-slot taxonomy. Route
        // through the standalone EmbeddingEncoder (single source of truth for the
        // top-k species selection and the top-k-by-abundance taxonomy selection),
        // mirroring how RankPool/Transformer route through RankPoolEncoder so the
        // two copies cannot diverge. The encoder adopts the dataset's own species
        // vocab (frequency-ranked, built by build_species_vocab above) and
        // taxonomy vocab via set_vocabs, so the emitted IDs match the tables the
        // model is sized against (both for training and the from_csv_with_schema
        // inference path, where species_to_idx_ / taxonomy_vocab_ are the reused
        // training-set vocabs).

        // Ensure each record carries its plot_id so the encoder groups correctly
        // (the two-file loader sets it, but a bare SpeciesRecord may not).
        std::vector<SpeciesRecord> embed_records;
        embed_records.reserve(all_records.size());
        for (int64_t i = 0; i < n_plots; ++i) {
            const auto& plot_id = plot_ids_[i];
            auto it = plot_records.find(plot_id);
            if (it == plot_records.end()) continue;
            for (const auto& rec : it->second) {
                SpeciesRecord r = rec;
                if (r.plot_id.empty()) r.plot_id = plot_id;
                embed_records.push_back(std::move(r));
            }
        }

        // top_k_taxonomy = n_taxonomy_slots preserves the fixed-slot width (2*top_k
        // under TopBottom selection); SelectionMode::Top preserves the embed
        // contract of encoding the top-k most-abundant species per plot.
        EmbeddingEncoder emb_encoder(config_.top_k_species, n_taxonomy_slots,
                                     SelectionMode::Top);
        emb_encoder.fit(embed_records);  // builds species_to_genus_/_family_ maps

        // Strip the dataset "<UNK>"=>0 sentinel (SpeciesVocab reserves code 0 for
        // UNK implicitly); hand the freq-ranked species IDs + taxonomy vocab over.
        std::unordered_map<std::string, int64_t> sp_map;
        sp_map.reserve(species_to_idx_.size());
        for (const auto& [name, id] : species_to_idx_) {
            if (name == "<UNK>" || id == 0) continue;
            sp_map.emplace(name, id);
        }
        emb_encoder.set_vocabs(SpeciesVocab::from_map(std::move(sp_map)),
                               taxonomy_vocab_);

        auto encoded = emb_encoder.transform(embed_records, plot_ids_);
        species_ids_ = encoded.species_ids;
        // Taxonomy fixed slots come from the encoder here; the shared fixed-slot
        // block below is skipped for embed mode. Only publish when the schema
        // reports taxonomy (matches the prior fixed-slot gate).
        if (schema_.has_taxonomy) {
            genus_ids_ = encoded.genus_ids;
            family_ids_ = encoded.family_ids;
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
                                            config_.pool_species_cap,
                                            has_abundance_column_);

        species_ids_ = encoded.species_ids;
        pool_genus_ids_ = encoded.genus_ids;
        pool_family_ids_ = encoded.family_ids;
        pool_weights_ = encoded.weights;
        pool_mask_ = encoded.mask;
        // Record the resolved species-cap width so inference truncates each plot
        // to the same max-species the model was trained on, even when the cap
        // was auto (p99) and would resolve differently on the inference data.
        schema_.pool_species_cap = static_cast<int>(encoded.species_ids.size(1));
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
        // Gate on genus OR family having real (non-UNK) entries so a family-only
        // dataset keeps its family embeddings, matching the encoder's own gate
        // (species_encoding.cpp: n_genera() > 1 || n_families() > 1) and the
        // coarse loader gate at load_header_data. Genus-only vocab is 1 (UNK).
        schema_.has_taxonomy =
            (encoded.n_genera_vocab > 1 || encoded.n_families_vocab > 1)
            && config_.use_taxonomy;
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

    // Encode taxonomy into fixed slots for hash / sparse modes. Skipped for
    // rank_pool / transformer (which consume the per-species pool_genus_ids_ /
    // pool_family_ids_ populated above) and for embed (whose fixed-slot taxonomy
    // is produced by the EmbeddingEncoder above); allocating + per-plot sorting
    // them again would be wasted work and a second copy of the same selection.
    const bool fixed_slot_taxonomy =
        config_.species_encoding != SpeciesEncodingMode::RankPool &&
        config_.species_encoding != SpeciesEncodingMode::Transformer &&
        config_.species_encoding != SpeciesEncodingMode::Embed;
    if (schema_.has_taxonomy && fixed_slot_taxonomy) {
        genus_ids_ = torch::zeros({n_plots, n_taxonomy_slots}, torch::kLong);
        family_ids_ = torch::zeros({n_plots, n_taxonomy_slots}, torch::kLong);
        auto genus_acc = genus_ids_.accessor<int64_t, 2>();
        auto family_acc = family_ids_.accessor<int64_t, 2>();

        for (int64_t i = 0; i < n_plots; ++i) {
            const auto& plot_id = plot_ids_[i];
            auto it = plot_records.find(plot_id);
            if (it == plot_records.end()) continue;

            // Canonical embed-mode taxonomy: aggregate abundance per DISTINCT
            // genus and per DISTINCT family, then take the top-k of each
            // independently (shared topk_by_abundance; matches EmbeddingEncoder,
            // resolving the former per-species-slot divergence, issue #60).
            std::unordered_map<std::string, float> genus_abd, family_abd;
            for (const auto& rec : it->second) {
                if (!rec.genus.empty())  genus_abd[rec.genus]   += rec.abundance;
                if (!rec.family.empty()) family_abd[rec.family] += rec.abundance;
            }

            auto top_genera = topk_by_abundance(genus_abd, n_taxonomy_slots);
            for (size_t slot = 0; slot < top_genera.size(); ++slot) {
                genus_acc[i][static_cast<int64_t>(slot)] =
                    taxonomy_vocab_.encode_genus(top_genera[slot]);
            }
            auto top_families = topk_by_abundance(family_abd, n_taxonomy_slots);
            for (size_t slot = 0; slot < top_families.size(); ++slot) {
                family_acc[i][static_cast<int64_t>(slot)] =
                    taxonomy_vocab_.encode_family(top_families[slot]);
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
