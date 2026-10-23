#pragma once

#include "resolve/types.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/encoder.hpp"
#include "resolve/categorical.hpp"
#include "resolve/continuous_block.hpp"
#include "resolve/plot_inputs.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/row_source.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace resolve {

// Configuration for dataset loading
struct DatasetConfig {
    // Species encoding configuration
    SpeciesEncodingMode species_encoding = SpeciesEncodingMode::Hash;
    int hash_dim = 32;
    int top_k = 3;
    int top_k_species = 10;  // For embed mode
    // Which species of a plot survive its per-plot budget. Each encoding takes
    // its budget from the knob that also fixes its width:
    //   hash          -> top_k
    //   embed         -> top_k_species (All is rejected: a fixed number of
    //                    ranked slots cannot encode every species)
    //   rank_pool /
    //   transformer /
    //   sparse        -> species_budget below, 0 by default = no budget
    // The selection ranks a plot's species by their recorded abundance (summed
    // over a species' records in the plot), ties by name.
    SelectionMode selection = SelectionMode::Top;
    // The value each species carries into the encoding: its abundance, or 1.0
    // under PresenceAbsence. Applies to every encoding: the hash weights, the
    // sparse vector, the rank-pool / transformer weights (whose has_cover flag
    // is then 0), the abundance-ranked taxonomy slots and the unknown-species
    // fraction. The selection still ranks by recorded abundance.
    RepresentationMode representation = RepresentationMode::Abundance;
    // Rescales the hash weights of a plot's selected species: Raw, Norm (divide
    // by their sum) or Log1p.
    NormalizationMode normalization = NormalizationMode::Raw;
    // How the fixed genus / family slots of the hash, sparse and embed
    // encodings rank a plot's taxa: Abundance by the summed value of their
    // species, Count by how many of the plot's species they hold. rank_pool /
    // transformer carry each species' own taxonomy and reject Count.
    AggregationMode aggregation = AggregationMode::Abundance;

    // Unknown species tracking
    bool track_unknown_fraction = true;
    bool track_unknown_count = false;

    // Taxonomy
    bool use_taxonomy = true;  // If available in data

    // CUDA hash computation (for hash mode)
    // When true, stores raw species data in CSR form for on-the-fly GPU hash
    // computation per batch instead of a precomputed hash embedding.
    bool use_cuda_hash = false;

    // Per-species weight scheme for the rank_pool / transformer encoders:
    // binary, abundance, log1p, norm or rank. Ignored for hash / embed /
    // sparse encodings.
    PoolWeighting pool_weighting = PoolWeighting::Log1p;

    // Cap on species-per-plot for rank_pool / transformer encoders. Caps the
    // padded `max_species` dimension to avoid one outlier plot inflating the
    // padding for the whole dataset.
    //
    //   0 (default) : no cap. The longest plot in the dataset dictates the
    //                 width of every row's pool tensors.
    //  -x           : a percentile cap, read as 100 - x: -1 is the 99th
    //                 percentile, -5 the 95th; x runs 1..99, and -100 or
    //                 below is rejected. The percentile of per-plot species
    //                 counts is computed on the data being loaded and longer
    //                 plots are truncated to it, printing a one-line summary
    //                 so users see the drop. It is a cap, not "no cap": that
    //                 is 0.
    //  >0           : manual cap. Truncate longer plots to this many species,
    //                 kept in plot order (the order of their first record).
    //
    // Plots shorter than the cap are unaffected. Truncation is a hard slice
    // (not top-k by abundance); the rank-pool weighting still applies to
    // whatever survives the slice.
    int pool_species_cap = 0;

    // Per-plot species budget for the encodings whose width is not otherwise
    // fixed: rank_pool, transformer and sparse. `selection` picks
    // which species survive it.
    //
    //   0 (default) : no budget. Every species a plot records is encoded,
    //                 whatever `selection` says.
    //   >0          : keep this many species per plot, taken from the end of
    //                 the abundance ranking `selection` names (up to 2x this
    //                 many under TopBottom, as everywhere else). This is what
    //                 makes a top-versus-bottom species ablation reachable on
    //                 the pooled encoders.
    //
    // SelectionMode::All names no rule for picking a subset, so it encodes
    // every species regardless of the budget. For rank_pool / transformer the
    // budget narrows the plot BEFORE pool_species_cap slices it, so a cap still
    // means what it documents: the first `cap` survivors in plot order.
    //
    // The species vocabulary is fitted over every species, budget or not, so
    // the integer codes are identical across settings and an ablation's arms
    // stay comparable.
    int species_budget = 0;

    // How a missing numeric covariate or coordinate enters the model:
    //
    //   Indicate (default) : the loader keeps the cell as NaN; the model reads
    //                        the fitting fold's mean in its place and a 0/1
    //                        column beside the covariates that marks it (one per
    //                        covariate, one for the coordinate pair).
    //   Zero               : the cell is read as 0.0 and nothing marks it, so a
    //                        recorded 0 and a missing value are one input.
    MissingValuePolicy missing_values = MissingValuePolicy::Indicate;

    // The abundance a record whose recorded abundance is exactly 0 is read at.
    // Vegetation archives write a record that states presence without a cover
    // estimate as 0% (EVA converts every cover scale to a percentage), and a
    // weighting such as log1p then gives the species no weight at all, so a
    // presence-only survey reaches the model with no species signal. Setting
    // this to 1.0 reads such a record at the weight a record with a missing
    // cover already gets. The default 0.0 reads a 0 as 0, i.e. leaves the
    // recorded value alone. Negative values are rejected at load.
    float zero_abundance_as = 0.0f;
};

// Forward declaration: the vocab-carrying loaders take one of these.
class ResolveDataset;

// Vocabularies fitted at training time, handed to a loader so the dataset it
// builds lives in the SAME integer-code namespace as the trained model.
// Every non-hash encoder indexes an embedding table by a code
// that is a function of the file it was fitted on, so an inference dataset that
// re-fits its own codes looks up the wrong embedding rows.
//
// Two sources produce one of these:
//   * an in-memory training ResolveDataset (the from_csv_with_schema(const
//     ResolveDataset&) path), and
//   * a ResolveSchema recovered from a checkpoint (external_vocabs_from_schema),
//     which is the path that needs no training data on disk at all.
struct ExternalVocabs {
    // Ordered species vocabulary; index = integer code, [0] = "<UNK>".
    std::vector<std::string> species_vocab;
    // Genus/family name -> code encoder.
    TaxonomyVocab taxonomy;
    // Per-column categorical string -> code maps.
    CategoricalVocab categorical;
    // Training-time target configs, used to replay each classification
    // target's class -> code mapping so the new dataset encodes labels the
    // same way (the caller need not populate TargetSpec::class_mapping).
    std::vector<TargetConfig> targets;
    // The rank-pool width the source's species records were truncated to,
    // resolved (> 0), or 0 where the source recorded none. A dataset encoded
    // against these vocabularies is truncated to the same width in place of
    // the config's pool_species_cap, so a plot keeps the records the model was
    // trained to read however long the new data's plots run; a percentile cap
    // re-resolved on the new data would read a different number of them.
    int pool_species_cap = 0;
};

// Rebuild the vocabularies a checkpoint's schema carries. `schema.species_vocab`
// / `genus_vocab` / `family_vocab` are empty on a checkpoint that stores only the sizes; the
// result is then an empty ExternalVocabs, and the loaders that receive it throw
// rather than silently re-fitting (the failure mode this whole path exists to
// prevent). The categorical maps are NOT on the schema -- only the column names
// and vocab sizes are -- so pass the Predictor's `categorical_vocab()` through
// `categorical` when scoring a categorical model.
ExternalVocabs external_vocabs_from_schema(const ResolveSchema& schema);

// Reassemble the loading-side DatasetConfig a checkpoint was built with.
// Single source of truth for "what DatasetConfig does this checkpoint imply",
// shared by the CLI predict path and any caller scoring new data.
//
// `model_config` supplies the three knobs that also SIZE the model
// (species_encoding, hash_dim, top_k) and are therefore authoritative there;
// `schema` supplies everything else the dataset loader consumed
// (top_k_species, selection, representation, normalization, aggregation,
// track_unknown_fraction, track_unknown_count, use_taxonomy, species_budget,
// pool_weighting and the resolved pool_species_cap).
//
// `use_cuda_hash` is deliberately NOT restored: it swaps the precomputed hash
// embedding for raw CSR species data, which is a training-time compute choice,
// and Predictor::predict needs the precomputed embedding. It stays at the
// DatasetConfig default (false).
DatasetConfig dataset_config_from_checkpoint(const ResolveSchema& schema,
                                             const ModelConfig& model_config);

// The selection a dataset built under `config` actually applies. Hash and embed
// always select, because their widths ARE per-plot budgets (top_k and
// top_k_species). The rank_pool / transformer / sparse encodings select only
// when species_budget gives them a budget to select down to; with no budget
// they encode every species, which is SelectionMode::All whatever was asked for.
//
// The dataset publishes this, not the requested value, onto its schema, so a
// checkpoint (and everything that reads one back) cannot report a species
// selection the run never made.
SelectionMode effective_selection(const DatasetConfig& config);

// Loaded dataset ready for training.
//
// Every loader reads plot-level columns (coordinates, covariates, targets) and
// species records, groups the records by plot into PlotAssemblages (which
// merges the records of one species in one plot, species_encoding.hpp), fits
// or adopts one EncodingVocab, and runs the configured species encoding.
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
    // TaxonomyVocab::encode_genus/encode_family, SpeciesVocab::encode). The
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

    // Same verb, sourcing the vocabularies from a CHECKPOINT's schema instead of
    // an in-memory training dataset. This is the path for a user
    // who has only a saved model and a new CSV: the training CSVs need not
    // exist, and no dataset has to be rebuilt to recover the vocab.
    //
    //   Predictor p = Predictor::load(ckpt);
    //   auto cfg = dataset_config_from_checkpoint(p.schema(), p.model()->config());
    //   auto ds  = ResolveDataset::from_csv_with_schema(
    //                  header, species, roles, targets, p.schema(), cfg);
    //   auto out = p.predict(ds);
    //
    // Throws when the schema carries no species vocabulary (a sizes-only
    // checkpoint): silently re-fitting is what this overload exists to
    // prevent, so the caller is told to retrain / re-save instead.
    //
    // Categorical covariates: the schema carries column names and vocab sizes
    // but not the string -> code maps, so pass the Predictor's
    // `categorical_vocab()` via the ExternalVocabs overload below when the
    // model has categoricals. Without it the categorical columns are re-fit and
    // Predictor::predict rejects the dataset.
    static ResolveDataset from_csv_with_schema(
        const std::string& header_path,
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const ResolveSchema& schema,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Fully-explicit form: the caller supplies every vocabulary. Used when the
    // model has categorical covariates (whose maps live on the Predictor, not
    // on the schema) and by the two overloads above, which both funnel here.
    static ResolveDataset from_csv_with_vocabs(
        const std::string& header_path,
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const ExternalVocabs& vocabs,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Load from single CSV file with species data only (header data inferred)
    static ResolveDataset from_species_csv(
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Single-table counterpart of from_csv_with_schema: reuse a checkpoint's
    // vocabularies when scoring a species-only CSV (`resolve predict
    // --species-only`).
    static ResolveDataset from_species_csv_with_schema(
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const ResolveSchema& schema,
        const DatasetConfig& config = DatasetConfig{}
    );

    static ResolveDataset from_species_csv_with_vocabs(
        const std::string& species_path,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const ExternalVocabs& vocabs,
        const DatasetConfig& config = DatasetConfig{}
    );

    // --- In-memory (DataFrame) loaders ---
    // The same loads over data already in RAM as ColumnTable(s) (the
    // cross-binding carrier built from a pandas DataFrame or an R data.frame),
    // for a header filtered or subset before a fit. They run the same loader
    // body as the CSV path through the RowSource seam, so the result is
    // identical to loading the equivalent CSV.

    // Header (one row per plot) and species (multiple rows per plot) both in
    // memory. The DataFrame analog of from_csv.
    static ResolveDataset from_dataframe(
        const ColumnTable& header,
        const ColumnTable& species,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Header in memory, species streamed from a CSV path: the header is the
    // per-fit-filtered frame, while the (large, unfiltered) species table is
    // still read once from disk.
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

    static ResolveDataset from_dataframe_with_vocabs(
        const ColumnTable& header,
        const ColumnTable& species,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const ExternalVocabs& vocabs,
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

    static ResolveDataset from_species_dataframe_with_vocabs(
        const ColumnTable& species,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const ExternalVocabs& vocabs,
        const DatasetConfig& config = DatasetConfig{}
    );

    // Accessors for encoded data
    const torch::Tensor& coordinates() const { return coordinates_; }
    const torch::Tensor& covariates() const { return covariates_; }
    const torch::Tensor& hash_embedding() const { return hash_embedding_; }
    const torch::Tensor& species_ids() const { return species_ids_; }
    const torch::Tensor& species_vector() const { return species_vector_; }

    // The taxonomy of the species VOCABULARY (not of a plot): index = species
    // code, value = the genus / family code that species belongs to, 0 where
    // the records name none. Empty when the dataset carries no taxonomy. This
    // is the species-to-species relation a HeterogeneousGNN builds its graph
    // from; genus_ids() / family_ids() above are per-plot slots instead.
    const torch::Tensor& species_genus_ids() const { return species_genus_ids_; }
    const torch::Tensor& species_family_ids() const { return species_family_ids_; }
    const torch::Tensor& genus_ids() const { return genus_ids_; }
    const torch::Tensor& family_ids() const { return family_ids_; }
    const torch::Tensor& unknown_fraction() const { return unknown_fraction_; }
    const torch::Tensor& unknown_count() const { return unknown_count_; }
    // How much of each plot's assemblage the dataset's vocabulary recognises
    // (compute_species_recognition). Filled by the loaders that adopt a
    // vocabulary fitted elsewhere (*_with_schema / *_with_vocabs); undefined
    // tensors otherwise, where every species is recognised by construction.
    const SpeciesRecognition& species_recognition() const { return species_recognition_; }
    // The unstandardised continuous block a model built from this dataset's
    // schema reads (continuous_block.hpp): coordinates, covariates, their
    // missingness flags, the unknown-species columns and, when include_hash is
    // true, the hash embedding. Missing cells are NaN under Indicate.
    [[nodiscard]] torch::Tensor continuous_block(bool include_hash) const;
    // The parts that block is assembled from, hash embedding included; a
    // trainer or predictor assembles and standardises them with its scalers.
    [[nodiscard]] ContinuousInputs continuous_inputs() const;
    // Every per-plot model input the dataset carries, except `continuous`,
    // which comes from continuous_inputs().
    [[nodiscard]] PlotInputs plot_inputs() const;
    // Categorical covariate codes. Shape (n_plots, n_categoricals) int64,
    // values produced by CategoricalVocab (0 = UNK). Empty (undefined or 0
    // columns) when the schema declares no categoricals.
    const torch::Tensor& categorical_ids() const { return categorical_ids_; }
    // The fitted vocabularies for each categorical column (string -> code).
    // Saved/loaded as part of the checkpoint so inference on new data uses
    // the same encoding as training.
    const CategoricalVocab& categorical_vocab() const { return categorical_vocab_; }
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

    // Position of each of `ids` in plot_ids(), in the order given: the plot
    // indices Trainer::prepare_data takes for an explicit stopping fold. An ID
    // the dataset does not hold (absent from the data, or dropped for a missing
    // label) throws std::invalid_argument naming it. A repeated ID yields a
    // repeated index, which prepare_data then rejects.
    [[nodiscard]] std::vector<int64_t> plot_indices(const std::vector<std::string>& ids) const;

    // Taxonomy vocabulary (for saving/loading)
    const TaxonomyVocab& taxonomy_vocab() const { return vocab_.taxonomy; }

    // Species vocabulary. Ordered: index = integer code, [0] = "<UNK>".
    // Mirrored onto schema().species_vocab so it survives into the checkpoint.
    const std::vector<std::string>& species_vocab() const { return vocab_.species.names(); }

    // The vocabularies this dataset fitted, in the form the *_with_vocabs
    // loaders accept. Equivalent to passing the dataset itself to
    // from_csv_with_schema, but usable after the dataset's source files are
    // gone and serializable through the checkpoint schema.
    [[nodiscard]] ExternalVocabs external_vocabs() const;

    // Number of plots
    int64_t n_plots() const { return schema_.n_plots; }

    // Dataset configuration
    const DatasetConfig& config() const { return config_; }

    // Default constructor creates an empty dataset
    ResolveDataset() = default;

private:
    // The plot-level columns of a table (dataset.cpp).
    struct PlotColumns;

    // The one loader body behind every two-table verb (header rows are plots,
    // species rows are records) and every single-table verb (each plot's first
    // row carries its plot-level columns). The caller owns the sources'
    // lifetimes and any I/O-retry wrapping. `vocabs` is null for a fresh fit;
    // non-null adopts the supplied vocabularies and replays classification
    // class mappings from them.
    static ResolveDataset load_tables(
        RowSource& header,
        RowSource& species,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config,
        const ExternalVocabs* vocabs
    );
    static ResolveDataset load_table(
        RowSource& source,
        const RoleMapping& roles,
        const std::vector<TargetSpec>& targets,
        const DatasetConfig& config,
        const ExternalVocabs* vocabs
    );

    // Validates the config, adopts `vocabs` when given, and returns the target
    // specs the load encodes (class mappings replayed from `vocabs`).
    std::vector<TargetSpec> begin_load(const std::vector<TargetSpec>& targets,
                                       const ExternalVocabs* vocabs);

    // Pre-populate the vocab members and flip use_external_vocabs_, so the
    // load encodes against these instead of fitting. Any value not in the
    // supplied vocab encodes as the reserved UNK code 0.
    void adopt_vocabs(const ExternalVocabs& vocabs);

    // Fill each classification TargetSpec's class_mapping from `source_configs`
    // so labels encode to the training-time codes instead of being auto-fit
    // against the new file's value distribution. Specs that already carry a
    // mapping are left alone.
    static std::vector<TargetSpec> replay_class_mappings(
        const std::vector<TargetSpec>& targets,
        const std::vector<TargetConfig>& source_configs
    );

    // Turn the plot-level columns into tensors: targets encoded, plots
    // without a usable label dropped, categoricals fitted / encoded.
    // `n_skipped` counts table rows too short to name their plot.
    void absorb_plot_columns(PlotColumns&& columns, int64_t n_skipped);

    // Group the records into PlotAssemblages, fit or adopt the vocabulary, run
    // the configured encoding, then publish the unknown-species statistics,
    // the recognition report, the vocabulary's taxonomy and the vocabularies.
    void encode_species(std::vector<SpeciesRecord> records);
    void record_encoding_settings();
    void fit_vocab(const PlotAssemblages& assemblages);
    void encode_hash(const PlotAssemblages& assemblages);
    void encode_embed(const PlotAssemblages& assemblages);
    void encode_sparse(const PlotAssemblages& assemblages);
    void encode_pool(const PlotAssemblages& assemblages);
    void encode_fixed_taxonomy(const PlotAssemblages& assemblages);
    void publish_species_statistics(const PlotAssemblages& assemblages);
    // Copy the settled species / genus / family vocabularies onto schema_ so
    // they travel into the checkpoint.
    void publish_vocabs_to_schema();

    // Data tensors
    torch::Tensor coordinates_;      // (n_plots, 2)
    torch::Tensor covariates_;       // (n_plots, n_covariates)
    torch::Tensor hash_embedding_;   // (n_plots, hash_dim) for hash mode
    torch::Tensor species_ids_;      // (n_plots, top_k_species) embed, (n_plots, max_species) pool
    torch::Tensor species_vector_;   // (n_plots, n_species_vocab) for sparse mode
    torch::Tensor genus_ids_;        // (n_plots, n_taxonomy_slots)
    torch::Tensor family_ids_;       // (n_plots, n_taxonomy_slots)
    torch::Tensor unknown_fraction_; // (n_plots,)
    torch::Tensor unknown_count_;    // (n_plots,)
    SpeciesRecognition species_recognition_;
    torch::Tensor categorical_ids_; // (n_plots, n_categoricals) int64
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
    // the rank-pool has_cover flag by column presence, not by inspecting the
    // values (a plot whose covers are all 1.0 still has cover).
    bool has_abundance_column_ = false;

    // Raw species data in CSR form for CUDA hash computation, stored when
    // use_cuda_hash=true in config.
    torch::Tensor raw_plot_indices_;   // (n_records,) int64 - which plot each record belongs to
    torch::Tensor raw_species_ids_;    // (n_records,) int64 - feature_hash_key of the species name
    torch::Tensor raw_weights_;        // (n_records,) float32 - abundance/weight values
    torch::Tensor plot_offsets_;       // (n_plots+1,) int64 - CSR offsets for batch slicing

    // Metadata
    ResolveSchema schema_;
    DatasetConfig config_;
    std::vector<std::string> plot_ids_;
    // The species and taxonomy vocabularies every encoding codes with.
    EncodingVocab vocab_;
    // Taxonomy of the vocabulary: (n_species_vocab,) int64 each, index =
    // species code.
    torch::Tensor species_genus_ids_;
    torch::Tensor species_family_ids_;

    // When true, the load encodes against the pre-populated categorical_vocab_
    // and vocab_ instead of fitting them. Set by the *_with_schema /
    // *_with_vocabs loaders.
    bool use_external_vocabs_ = false;

    // Target configurations
    std::vector<TargetConfig> target_configs_;
};

} // namespace resolve
