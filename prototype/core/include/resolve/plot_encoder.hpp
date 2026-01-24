#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>

namespace resolve {

// ============================================================================
// Encoding Types
// ============================================================================

enum class EncodingType {
    Numeric,    // Bare variable, auto-scaled
    Raw,        // raw() - passthrough, no scaling
    Hash,       // hash() - feature hashing
    Embed,      // embed() - learned embeddings (returns IDs)
    OneHot      // onehot() - one-hot encoding
};

enum class DataSource {
    Plot,        // From plot-level data (one value per plot)
    Observation  // From observation-level data (aggregated per plot)
};

// ============================================================================
// Column Specification
// ============================================================================

/**
 * Specification for encoding a column or group of columns.
 */
struct ColumnSpec {
    std::string name;                    // Unique identifier for this spec
    std::vector<std::string> columns;    // Column names to encode
    EncodingType type;                   // Encoding type
    DataSource source;                   // Where data comes from

    // Parameters
    int dim = 32;                        // Output dimension (hash/embed)
    int top_k = 0;                       // Top-k selection (0 = all)
    int bottom_k = 0;                    // Bottom-k selection (0 = none)
    std::string rank_by;                 // Column to rank by for top/bottom

    // Helpers
    int n_slots() const { return (top_k > 0 || bottom_k > 0) ? top_k + bottom_k : 1; }
    bool has_selection() const { return top_k > 0 || bottom_k > 0; }
};

// ============================================================================
// Encoded Output
// ============================================================================

/**
 * Encoded output for a single column spec.
 */
struct EncodedColumn {
    std::string name;
    torch::Tensor values;      // Shape depends on encoding type
    bool is_embedding_ids;     // True if values are IDs for embedding layer
    int64_t vocab_size;        // For embed: vocabulary size (for embedding layer)
    int embed_dim;             // For embed: embedding dimension
    int n_slots;               // For embed: number of slots per sample
};

/**
 * Full encoded result from PlotEncoder.
 */
struct EncodedPlotData {
    std::vector<EncodedColumn> columns;
    torch::Tensor unknown_fraction;      // (n_plots,) fraction unknown for obs-level
    std::vector<std::string> plot_ids;

    /**
     * Concatenate all continuous features (numeric, raw, hash, onehot).
     * Returns (n_plots, total_continuous_dim).
     */
    torch::Tensor continuous_features() const;

    /**
     * Get embedding IDs by spec name.
     * Returns (n_plots, n_slots) int64 tensor.
     */
    torch::Tensor embedding_ids(const std::string& name) const;

    /**
     * Get all embedding specs for model construction.
     */
    std::vector<std::tuple<std::string, int64_t, int, int>> embedding_specs() const;
};

// ============================================================================
// Data Records
// ============================================================================

/**
 * Plot-level record with named fields.
 */
struct PlotRecord {
    std::string plot_id;
    std::unordered_map<std::string, std::string> categorical;
    std::unordered_map<std::string, float> numeric;
};

/**
 * Observation-level record (e.g., species occurrence).
 */
struct ObservationRecord {
    std::string plot_id;
    std::unordered_map<std::string, std::string> categorical;
    std::unordered_map<std::string, float> numeric;
};

// ============================================================================
// Vocabulary for categorical columns
// ============================================================================

class CategoryVocab {
public:
    CategoryVocab() = default;

    void fit(const std::vector<std::string>& values);
    int64_t encode(const std::string& value) const;
    int64_t size() const { return idx_to_value_.size(); }
    bool contains(const std::string& value) const;

    // Serialization
    void save(std::ostream& os) const;
    static CategoryVocab load(std::istream& is);

private:
    std::unordered_map<std::string, int64_t> value_to_idx_;
    std::vector<std::string> idx_to_value_;
    int64_t unk_idx_ = 0;  // Index for unknown values
};

// ============================================================================
// Standard Scaler for numeric columns
// ============================================================================

class StandardScaler {
public:
    StandardScaler() = default;

    void fit(const std::vector<float>& values);
    float transform(float value) const;
    std::vector<float> transform(const std::vector<float>& values) const;

    float mean() const { return mean_; }
    float std() const { return std_; }

    // Serialization
    void save(std::ostream& os) const;
    static StandardScaler load(std::istream& is);

private:
    float mean_ = 0.0f;
    float std_ = 1.0f;
    bool fitted_ = false;
};

// ============================================================================
// PlotEncoder
// ============================================================================

/**
 * General-purpose encoder for plot data.
 *
 * Supports:
 * - Numeric variables (auto-scaled or raw)
 * - Hash encoding for high-cardinality categoricals
 * - Embedding IDs for learned embeddings
 * - One-hot encoding for small categoricals
 *
 * Can handle both plot-level and observation-level data.
 */
class PlotEncoder {
public:
    PlotEncoder() = default;

    // -------------------------------------------------------------------------
    // Add column specifications
    // -------------------------------------------------------------------------

    /**
     * Add numeric column(s) with auto-scaling.
     */
    void add_numeric(const std::string& name,
                     const std::vector<std::string>& columns,
                     DataSource source = DataSource::Plot);

    /**
     * Add raw column(s) without scaling.
     */
    void add_raw(const std::string& name,
                 const std::vector<std::string>& columns,
                 DataSource source = DataSource::Plot);

    /**
     * Add hash-encoded column(s).
     */
    void add_hash(const std::string& name,
                  const std::vector<std::string>& columns,
                  int dim = 32,
                  int top_k = 0,
                  int bottom_k = 0,
                  const std::string& rank_by = "",
                  DataSource source = DataSource::Observation);

    /**
     * Add embedding column(s) - returns IDs for embedding layer.
     */
    void add_embed(const std::string& name,
                   const std::vector<std::string>& columns,
                   int dim = 16,
                   int top_k = 0,
                   int bottom_k = 0,
                   const std::string& rank_by = "",
                   DataSource source = DataSource::Observation);

    /**
     * Add one-hot encoded column(s).
     */
    void add_onehot(const std::string& name,
                    const std::vector<std::string>& columns,
                    DataSource source = DataSource::Plot);

    // -------------------------------------------------------------------------
    // Fit and Transform
    // -------------------------------------------------------------------------

    /**
     * Fit vocabularies and scalers from data.
     */
    void fit(const std::vector<PlotRecord>& plot_data,
             const std::vector<ObservationRecord>& obs_data = {});

    /**
     * Transform data using fitted encoder.
     */
    EncodedPlotData transform(
        const std::vector<PlotRecord>& plot_data,
        const std::vector<ObservationRecord>& obs_data,
        const std::vector<std::string>& plot_ids
    ) const;

    /**
     * Fit and transform in one call.
     */
    EncodedPlotData fit_transform(
        const std::vector<PlotRecord>& plot_data,
        const std::vector<ObservationRecord>& obs_data,
        const std::vector<std::string>& plot_ids
    );

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    const std::vector<ColumnSpec>& specs() const { return specs_; }
    bool is_fitted() const { return fitted_; }

    /**
     * Get vocabulary size for an embed column.
     */
    int64_t vocab_size(const std::string& name) const;

    /**
     * Get total continuous output dimension.
     */
    int continuous_dim() const;

    /**
     * Get embedding configurations for model construction.
     * Returns: [(name, vocab_size, embed_dim, n_slots), ...]
     */
    std::vector<std::tuple<std::string, int64_t, int, int>> embedding_configs() const;

    // -------------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------------

    void save(const std::string& path) const;
    static PlotEncoder load(const std::string& path);

private:
    // Column specifications
    std::vector<ColumnSpec> specs_;

    // Fitted state
    std::unordered_map<std::string, CategoryVocab> vocabs_;      // For embed/onehot
    std::unordered_map<std::string, StandardScaler> scalers_;    // For numeric
    std::unordered_map<std::string, std::unordered_set<std::string>> known_values_;  // For unknown tracking

    bool fitted_ = false;

    // -------------------------------------------------------------------------
    // Internal encoding methods
    // -------------------------------------------------------------------------

    torch::Tensor encode_numeric(
        const ColumnSpec& spec,
        const std::vector<PlotRecord>& plot_data,
        const std::vector<std::string>& plot_ids
    ) const;

    torch::Tensor encode_hash(
        const ColumnSpec& spec,
        const std::vector<ObservationRecord>& obs_data,
        const std::vector<std::string>& plot_ids
    ) const;

    torch::Tensor encode_embed(
        const ColumnSpec& spec,
        const std::vector<PlotRecord>& plot_data,
        const std::vector<ObservationRecord>& obs_data,
        const std::vector<std::string>& plot_ids
    ) const;

    torch::Tensor encode_onehot(
        const ColumnSpec& spec,
        const std::vector<PlotRecord>& plot_data,
        const std::vector<std::string>& plot_ids
    ) const;

    /**
     * Feature hashing for a weighted map of values.
     */
    torch::Tensor hash_values(
        const std::unordered_map<std::string, float>& weighted_values,
        int dim
    ) const;

    /**
     * Get top-k and/or bottom-k values by weight.
     */
    std::vector<std::string> select_top_bottom(
        const std::unordered_map<std::string, float>& weighted_values,
        int top_k,
        int bottom_k
    ) const;
};

}  // namespace resolve
