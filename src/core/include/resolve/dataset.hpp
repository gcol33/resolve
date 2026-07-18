#pragma once

#include "resolve/types.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/encoder.hpp"
#include "resolve/categorical.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/row_source.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace resolve {

// Column indices resolved from RoleMapping - reduces repeated column_index() calls
struct ColumnIndices {
    int plot = -1;
    int species = -1;
    int abundance = -1;
    int longitude = -1;
    int latitude = -1;
    int genus = -1;
    int family = -1;

    // Factory method to resolve all indices from any RowSource and RoleMapping
    // (a CSVReader or an in-memory ColumnTable, via InMemoryRowSource). A named
    // role column that cannot be resolved throws (issue #94). expect_coordinates
    // must be true only for the single-table species loader, where longitude/
    // latitude live in the species source; in the two-file loader they are header
    // roles absent from the species source, so they are not looked up here.
    static ColumnIndices from_source(const RowSource& source, const RoleMapping& roles,
                                     bool expect_coordinates = false);
};

// Configuration for dataset loading
struct DatasetConfig {
    // Species encoding configuration
    SpeciesEncodingMode species_encoding = SpeciesEncodingMode::Hash;
    int hash_dim = 32;
    int top_k = 3;
    int top_k_species = 10;  // For embed mode
    SelectionMode selection = SelectionMode::Top;
    RepresentationMode representation = RepresentationMode::Abundance;
    NormalizationMode normalization = NormalizationMode::Raw;
    AggregationMode aggregation = AggregationMode::Abundance;

    // Unknown species tracking
    bool track_unknown_fraction = true;
    bool track_unknown_count = false;

    // Taxonomy
    bool use_taxonomy = true;  // If available in data

    // CUDA hash computation (for hash mode)
    // When true, stores raw species data in COO format for on-the-fly GPU hash computation
    // This avoids pre-computing hash embeddings and allows dynamic batch processing
    bool use_cuda_hash = false;

    // Per-species weight scheme for the rank_pool / transformer encoders.
    // Mirrors the Python POC's BasePoolEncoder weighting modes
    // (binary, abundance, log1p, norm, rank). Defaults to Log1p which is the
    // v7 paper headline (`rank_log1p_big`). Ignored for hash / embed / sparse
    // encodings.
    PoolWeighting pool_weighting = PoolWeighting::Log1p;

    // Cap on species-per-plot for rank_pool / transformer encoders. Caps the
    // padded `max_species` dimension to avoid one outlier plot inflating the
    // padding for the whole dataset. Mirrors the Python POC's
    // `rank_pool_species_cap` (auto p99 by default in the POC; opt-in here).
    //
    //   0 (default) : no cap. Pad to the global per-plot max. Matches the
    //                 untrimmed behaviour; the longest plot in the dataset
    //                 dictates the width of every row's pool tensors.
    //  -1           : auto p99. Compute the 99th percentile of per-plot
    //                 species counts and truncate longer plots to that
    //                 length. Matches the POC's default behaviour and prints
    //                 a one-line summary so users see the drop.
    //  >0           : manual cap. Truncate longer plots to this many species
    //                 (kept in original CSV row order, matching the POC's
    //                 `a[:cap]` slice).
    //
    // Plots shorter than the cap are unaffected. Truncation is a hard slice
    // (not top-k by abundance); the rank-pool weighting still applies to
    // whatever survives the slice.
    int pool_species_cap = 0;
};

// Loaded dataset ready for training
class ResolveDataset {
public:
    // Load from two CSV files: header data (one row per plot) and species data (multiple rows per plot)
    static ResolveDataset from_csv(
        const std::string& header_path,
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Load from two CSV files, reusing the vocabularies + classification class
    // mappings from `schema_source` instead of fitting fresh ones. Any string
    // value not seen during the source's fit is mapped to the reserved UNK
    // slot (code 0) by the existing encode paths (CategoricalVocab::encode_batch,
    // TaxonomyVocab::encode_genus/encode_family, species_to_idx_ fallback). The
    // resulting dataset is therefore safe to feed to a Predictor that was
    // trained on `schema_source` — its species_ids / categorical_ids / pool_*
    // tensors live in the same vocab namespace as the training data.
    //
    // Required for cross-split workflows (leave-one-dataset-out, sample
    // efficiency, transfer): the training set's vocab must be reused when
    // building the held-out evaluation set, otherwise the model's lookup
    // tables are indexed with the wrong namespace.
    //
    // Classification target class mappings come from `schema_source` too —
    // the caller does not need to populate TargetSpec.class_mapping.
    static ResolveDataset from_csv_with_schema(
        const std::string& header_path,
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const ResolveDataset& schema_source,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Load from single CSV file with species data only (header data inferred)
    static ResolveDataset from_species_csv(
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config = DatasetConfig{}
    );

    // --- In-memory (DataFrame) loaders (issue #22) ---
    // The on-disk from_csv* verbs above force a write-to-temp-CSV / re-read
    // round-trip whenever the header must be filtered/subset before a fit. These
    // accept the same data already in RAM as ColumnTable(s) (the cross-binding
    // carrier built from a pandas DataFrame or an R data.frame). They share the
    // exact loader bodies as the CSV path, so the result is identical to loading
    // the equivalent CSV — only the disk I/O is elided.

    // Header (one row per plot) and species (multiple rows per plot) both in
    // memory. The DataFrame analog of from_csv.
    static ResolveDataset from_dataframe(
        const ColumnTable& header,
        const ColumnTable& species,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Header in memory, species streamed from a CSV path. Targets the canonical
    // pain point directly: the header is the per-fit-filtered frame, while the
    // (large, unfiltered) species table is still read once from disk.
    static ResolveDataset from_dataframe_header(
        const ColumnTable& header,
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config = DatasetConfig{}
    );

    // In-memory analog of from_csv_with_schema: reuse the categorical / taxonomy
    // / species vocabularies and classification class mappings from
    // `schema_source` instead of fitting fresh ones. Required for cross-split
    // workflows (held-out eval sets) when the data is already a filtered frame.
    static ResolveDataset from_dataframe_with_schema(
        const ColumnTable& header,
        const ColumnTable& species,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const ResolveDataset& schema_source,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Single in-memory long table with species data only. The DataFrame analog
    // of from_species_csv (header data inferred from first occurrence per plot).
    static ResolveDataset from_species_dataframe(
        const ColumnTable& species,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Accessors for encoded data
    const torch::Tensor& coordinates() const { return coordinates_; }
    const torch::Tensor& covariates() const { return covariates_; }
    const torch::Tensor& hash_embedding() const { return hash_embedding_; }
    const torch::Tensor& species_ids() const { return species_ids_; }
    const torch::Tensor& species_vector() const { return species_vector_; }
    const torch::Tensor& genus_ids() const { return genus_ids_; }
    const torch::Tensor& family_ids() const { return family_ids_; }
    const torch::Tensor& unknown_fraction() const { return unknown_fraction_; }
    const torch::Tensor& unknown_count() const { return unknown_count_; }
    // Categorical covariate codes. Shape (n_plots, n_categoricals) int64,
    // values produced by CategoricalVocab (0 = UNK). Empty (undefined or 0
    // columns) when the schema declares no categoricals.
    const torch::Tensor& categorical_ids() const { return categorical_ids_; }
    // The fitted vocabularies for each categorical column (string -> code).
    // Saved/loaded as part of the checkpoint so inference on new data uses
    // the same encoding as training.
    const CategoricalVocab& categorical_vocab() const { return categorical_vocab_; }
    CategoricalVocab& categorical_vocab() { return categorical_vocab_; }
    const std::unordered_map<std::string, torch::Tensor>& targets() const { return targets_; }

    // Accessors for pool-style encoder fields (rank_pool / transformer modes)
    const torch::Tensor& pool_genus_ids() const { return pool_genus_ids_; }
    const torch::Tensor& pool_family_ids() const { return pool_family_ids_; }
    const torch::Tensor& pool_weights() const { return pool_weights_; }
    const torch::Tensor& pool_mask() const { return pool_mask_; }
    const torch::Tensor& pool_has_cover() const { return pool_has_cover_; }
    bool has_pool_data() const { return pool_mask_.defined() && pool_mask_.numel() > 0; }

    // Accessors for raw species data (CUDA hash computation)
    const torch::Tensor& raw_plot_indices() const { return raw_plot_indices_; }
    const torch::Tensor& raw_species_ids() const { return raw_species_ids_; }
    const torch::Tensor& raw_weights() const { return raw_weights_; }
    const torch::Tensor& plot_offsets() const { return plot_offsets_; }
    bool has_raw_species_data() const { return raw_plot_indices_.defined() && raw_plot_indices_.numel() > 0; }

    // Schema information
    const ResolveSchema& schema() const { return schema_; }

    // Plot IDs for tracking
    const std::vector<std::string>& plot_ids() const { return plot_ids_; }

    // Taxonomy vocabulary (for saving/loading)
    const TaxonomyVocab& taxonomy_vocab() const { return taxonomy_vocab_; }
    TaxonomyVocab& taxonomy_vocab() { return taxonomy_vocab_; }

    // Species vocabulary (for embed mode)
    const std::vector<std::string>& species_vocab() const { return species_vocab_; }

    // Number of plots
    int64_t n_plots() const { return schema_.n_plots; }

    // Dataset configuration
    const DatasetConfig& config() const { return config_; }

    // Default constructor creates an empty dataset
    ResolveDataset() = default;

private:
    // Single-attempt loader bodies. The public from_* verbs wrap these in
    // io::with_retry<io::IOError> so a transient storage fault re-runs the
    // whole load into a fresh dataset rather than aborting the run (issue #20).
    static ResolveDataset from_csv_impl(
        const std::string& header_path,
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config
    );
    static ResolveDataset from_csv_with_schema_impl(
        const std::string& header_path,
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const ResolveDataset& schema_source,
        const DatasetConfig& config
    );
    static ResolveDataset from_species_csv_impl(
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config
    );

    // Build a dataset from a single long-format row source (the shared body of
    // from_species_csv / from_species_dataframe). The caller owns the source's
    // lifetime and any I/O-retry wrapping.
    static ResolveDataset from_species_source(
        RowSource& source,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config
    );

    // Shared body of from_csv_with_schema / from_dataframe_with_schema: copy the
    // source's vocabularies, replay classification class mappings, then load the
    // header and species from the given row sources. The caller owns the
    // sources' lifetimes and any I/O-retry wrapping.
    static ResolveDataset load_with_schema(
        RowSource& header,
        RowSource& species,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const ResolveDataset& schema_source,
        const DatasetConfig& config
    );

    // Load header data (one row per plot) from any row source.
    void load_header_data(
        RowSource& source,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets
    );

    // Load and encode species data from any row source.
    void load_species_data(
        RowSource& source,
        const RoleMapping& roles
    );

    // Build species vocabulary from data
    void build_species_vocab(
        const std::unordered_map<std::string, std::vector<std::pair<std::string, float>>>& plot_species
    );

    // Encode species data based on mode
    void encode_species(
        const std::unordered_map<std::string, std::vector<SpeciesRecord>>& plot_records
    );

    // Data tensors
    torch::Tensor coordinates_;      // (n_plots, 2)
    torch::Tensor covariates_;       // (n_plots, n_covariates)
    torch::Tensor hash_embedding_;   // (n_plots, hash_dim) for hash mode
    torch::Tensor species_ids_;      // (n_plots, top_k_species) for embed mode
    torch::Tensor species_vector_;   // (n_plots, n_species) for sparse mode
    torch::Tensor genus_ids_;        // (n_plots, n_taxonomy_slots)
    torch::Tensor family_ids_;       // (n_plots, n_taxonomy_slots)
    torch::Tensor unknown_fraction_; // (n_plots,)
    torch::Tensor unknown_count_;    // (n_plots,)
    torch::Tensor categorical_ids_;  // (n_plots, n_categoricals) int64
    CategoricalVocab categorical_vocab_;  // per-column string->code maps
    std::unordered_map<std::string, torch::Tensor> targets_;

    // Pool-style encoder fields for rank_pool / transformer modes
    // Each species in a plot gets its own slot (padded to max_species across all plots)
    torch::Tensor pool_genus_ids_;   // (n_plots, max_species) int64
    torch::Tensor pool_family_ids_;  // (n_plots, max_species) int64
    torch::Tensor pool_weights_;     // (n_plots, max_species) float32 - abundance/weight per species
    torch::Tensor pool_mask_;        // (n_plots, max_species) bool - true where species exists
    torch::Tensor pool_has_cover_;   // (n_plots,) float32 - 1.0 if plot has abundance data, 0.0 otherwise

    // Whether an abundance/cover column was actually mapped in the roles. Drives
    // the rank-pool has_cover flag by column presence (POC semantics), not by
    // inspecting the values (a plot whose covers are all 1.0 still has cover).
    bool has_abundance_column_ = false;

    // Raw species data in COO format for CUDA hash computation
    // Stored when use_cuda_hash=true in config, enables on-the-fly GPU hash computation
    torch::Tensor raw_plot_indices_;   // (n_records,) int64 - which plot each record belongs to
    torch::Tensor raw_species_ids_;    // (n_records,) int64 - hashed species IDs (MurmurHash of string)
    torch::Tensor raw_weights_;        // (n_records,) float32 - abundance/weight values
    torch::Tensor plot_offsets_;       // (n_plots+1,) int64 - CSR-style offsets for fast batch slicing

    // Metadata
    ResolveSchema schema_;
    DatasetConfig config_;
    std::vector<std::string> plot_ids_;
    std::vector<std::string> species_vocab_;
    std::unordered_map<std::string, int64_t> species_to_idx_;
    TaxonomyVocab taxonomy_vocab_;

    // When true, the load_*/encode_species paths skip vocab fitting and use
    // the pre-populated categorical_vocab_ / taxonomy_vocab_ / species_vocab_
    // members. Set by from_csv_with_schema; never set by from_csv.
    bool use_external_vocabs_ = false;

    // Target configurations
    std::vector<TargetConfig> target_configs_;
};

} // namespace resolve
