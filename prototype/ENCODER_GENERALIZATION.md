# Encoder Generalization Plan

## RESOLVE

**R**elational **E**ncoding via **S**tructured **O**bservation **L**earning with **V**ector **E**mbeddings

## Vision

Transform the current domain-specific `SpeciesEncoder` into a **general-purpose column encoder** that can handle any combination of variables with any encoding strategy. The formula interface should feel natural to R users while mapping cleanly to a flexible C++ backend.

## Target R API

```r
# Core pattern: data (one row per unit) + obs (many rows per unit) + by (join key)
fit <- resolve(
  ph ~ elevation + latitude + hash(species, top = 5, by = "cover") + embed(genus),
  data = plots,
  obs = species_obs,
  by = "plot_id"
)

# Multiple hash/embed groups
fit <- resolve(
  ph ~ elevation + hash(species, dim = 64) + embed(genus, family) + onehot(treatment),
  data = plots,
  obs = species_obs,
  by = "plot_id"
)

# E-commerce example
fit <- resolve(
  ltv ~ age + income + hash(product_id, top = 10, by = "price") + embed(category, brand),
  data = customers,
  obs = purchases,
  by = "customer_id"
)

# Healthcare example
fit <- resolve(
  readmit ~ age + bmi + hash(diagnosis, top = 5) + embed(drug_class),
  data = patients,
  obs = encounters,
  by = "patient_id"
)

# Multiple targets with transformations
fit <- resolve(
  cbind(ph, nitrogen) ~ elevation + hash(species, top = 5, by = "cover") + embed(genus),
  data = plots,
  obs = species_obs,
  by = "plot_id",
  transform = list(nitrogen = "log1p")
)
```

**Key principles:**
- `data` = one row per unit (what you're predicting about)
- `obs` = many rows per unit (observations/events/items to aggregate)
- `by` = join key linking the two tables
- Bare variable names = numeric from `data`, auto-scaled
- `hash()`/`embed()` variables = from `obs`
- `raw()` = opt-out of auto-scaling

## Encoding Types

### 1. Bare variables (numeric, auto-scaled)
```r
ph ~ elevation + latitude + ...
```
- Continuous variables, written directly in formula (R convention)
- **Auto-scaled** (z-score standardization) during training
- Variables come from `data` (plot-level)
- **Error if categorical:** If a bare variable is `character` or `factor`, throw an error requiring explicit encoding

### 2. `raw(...)` (numeric, no scaling)
```r
ph ~ raw(elevation_prescaled) + latitude + ...
```
- Opt-out of auto-scaling for pre-scaled or special variables
- Use when you've already normalized your data
- Passthrough with no transformation

### 3. `hash(..., dim = 32, top = NULL, bottom = NULL, by = NULL)`
```r
ph ~ hash(species, top = 5, by = "cover", dim = 32)
```
- Feature hashing for high-cardinality categoricals
- Supports top/bottom selection when `by` is specified
- Variables typically come from `species` (observation-level)
- Aggregates across observations within each plot
- Output: continuous vector (hash buckets)

### 4. `embed(..., dim = 16, top = NULL, bottom = NULL, by = NULL)`
```r
ph ~ embed(genus, family, dim = 16)
```
- Learned embeddings for categorical variables
- Builds vocabulary during fit, returns integer IDs to embedding layer
- Supports top/bottom selection for observation-level data
- For plot-level categoricals, returns single ID per plot
- Output: integer IDs → embedding layer → continuous vectors

### 5. `onehot(...)`
```r
ph ~ onehot(treatment, soil_type)
```
- One-hot encoding for small categorical vocabularies
- Mainly for plot-level variables with few levels
- Output: binary vector

## Categorical vs Numeric: Why It Matters

**Numeric variables:**
- Direct input to neural network (after scaling)
- The network learns continuous relationships
- Example: `elevation` → scaled → dense layers

**Categorical variables:**
- Cannot be input directly (no meaningful numeric relationship)
- Must be encoded: hash, embed, or onehot
- Example: `genus="Quercus"` → embedding lookup → dense vector

This distinction is fundamental in ML - categorical data needs representation learning (embeddings) while numeric data just needs normalization.

## Validation & Errors

**Categorical as bare variable → Error:**
```r
ph ~ elevation + soil_type + hash(species)
#                ^^^^^^^^^
# Error: 'soil_type' is character/factor but used as numeric variable.
# Use embed(soil_type), hash(soil_type), or onehot(soil_type) for categorical data.
```

**Numeric in categorical function → Allowed:**
```r
ph ~ elevation + embed(site_id) + onehot(treatment_code)
# OK - R users often have numeric codes that are actually categorical
# (site IDs, years, treatment codes like 1/2/3)
# If user wraps it in embed/hash/onehot, trust them.
```

**Variable not found → Error:**
```r
ph ~ elevation + hash(speices)  # typo
# Error: 'speices' not found in data or species. Did you mean 'species'?
```

This validation prevents the main silent error: categorical data getting nonsense numeric treatment. The reverse (numeric as categorical) is a user choice we respect.

## Data Model

Two data sources:
1. **Plot data** (`data` argument) - one row per plot
   - Continuous covariates: elevation, latitude, climate vars
   - Categorical covariates: soil_type, biome, land_use
   - Target variables: ph, nitrogen, carbon

2. **Observation data** (`species` argument) - multiple rows per plot
   - Species identifiers: species_id, genus, family
   - Abundance/cover values for weighting
   - Functional traits (optional)

Variables in the formula are looked up in both data sources:
- If found in `data` → plot-level (one value per plot)
- If found in `species` → observation-level (aggregated per plot)

## C++ Architecture

### Core Types

```cpp
namespace resolve {

// Encoding specification for a single column or group of columns
struct ColumnSpec {
    std::string name;                    // Unique identifier
    std::vector<std::string> columns;    // Column names to encode
    EncodingType type;                   // Hash, Embed, Numeric, OneHot
    int dim = 32;                        // Output dimension (for hash/embed)
    int top_k = 0;                       // Top-k selection (0 = disabled)
    int bottom_k = 0;                    // Bottom-k selection (0 = disabled)
    std::string rank_by;                 // Column to rank by for top/bottom
    bool scale = false;                  // For numeric: standardize?
    DataSource source;                   // Plot or Observation level
};

enum class EncodingType {
    Numeric,    // Passthrough (optionally scaled)
    Hash,       // Feature hashing
    Embed,      // Learned embeddings (returns IDs)
    OneHot      // One-hot encoding
};

enum class DataSource {
    Plot,        // From plot-level data
    Observation  // From observation-level data (e.g., species)
};

// Encoded output for a single column spec
struct EncodedColumn {
    std::string name;
    torch::Tensor values;      // (n_plots, dim) for hash/numeric/onehot
                               // (n_plots, n_slots) int64 for embed
    bool is_embedding_ids;     // True if values are IDs for embedding layer
};

// Full encoded result
struct EncodedData {
    std::vector<EncodedColumn> columns;
    torch::Tensor unknown_fraction;  // (n_plots,) for observation-level unknowns
    std::vector<std::string> plot_ids;

    // Convenience: concatenate all non-embedding columns
    torch::Tensor continuous_features() const;

    // Get embedding IDs by name
    torch::Tensor embedding_ids(const std::string& name) const;
};

}  // namespace resolve
```

### PlotEncoder Class

```cpp
class PlotEncoder {
public:
    PlotEncoder() = default;

    // Add encoding specifications
    void add_numeric(const std::string& name,
                     const std::vector<std::string>& columns,
                     bool scale = false);

    void add_hash(const std::string& name,
                  const std::vector<std::string>& columns,
                  int dim = 32,
                  int top_k = 0,
                  int bottom_k = 0,
                  const std::string& rank_by = "");

    void add_embed(const std::string& name,
                   const std::vector<std::string>& columns,
                   int dim = 16,
                   int top_k = 0,
                   int bottom_k = 0,
                   const std::string& rank_by = "");

    void add_onehot(const std::string& name,
                    const std::vector<std::string>& columns);

    // Fit vocabularies and scalers
    void fit(const std::vector<PlotRecord>& plot_data,
             const std::vector<ObservationRecord>& obs_data);

    // Transform data
    EncodedData transform(const std::vector<PlotRecord>& plot_data,
                          const std::vector<ObservationRecord>& obs_data,
                          const std::vector<std::string>& plot_ids) const;

    // Accessors
    const std::vector<ColumnSpec>& specs() const;
    int64_t vocab_size(const std::string& name) const;  // For embed columns
    int output_dim() const;  // Total continuous output dimension

    // Serialization
    void save(const std::string& path) const;
    static PlotEncoder load(const std::string& path);

private:
    std::vector<ColumnSpec> specs_;

    // Vocabularies for embed columns
    std::unordered_map<std::string, CategoryVocab> vocabs_;

    // Scalers for numeric columns
    std::unordered_map<std::string, StandardScaler> scalers_;

    // Known values for tracking unknowns
    std::unordered_map<std::string, std::unordered_set<std::string>> known_values_;

    bool fitted_ = false;
};
```

### Data Records

```cpp
// Generic record with string key-value pairs
struct PlotRecord {
    std::string plot_id;
    std::unordered_map<std::string, std::string> categorical;
    std::unordered_map<std::string, float> numeric;
};

struct ObservationRecord {
    std::string plot_id;
    std::unordered_map<std::string, std::string> categorical;  // species, genus, family, etc.
    std::unordered_map<std::string, float> numeric;            // abundance, cover, traits, etc.
};
```

## Model Changes

The `ResolveModel` needs to accept dynamic embedding configurations:

```cpp
struct EmbeddingConfig {
    std::string name;
    int64_t vocab_size;
    int embed_dim;
    int n_slots;  // How many IDs per plot (for top-k selection)
};

struct ModelConfig {
    int encoder_dim = 128;
    int hidden_dim = 256;
    int n_encoder_layers = 3;
    float dropout = 0.1;

    int n_continuous;  // Total continuous input dim (from encoder)
    std::vector<EmbeddingConfig> embeddings;  // Dynamic embedding layers
};
```

The model creates embedding layers dynamically based on config:

```cpp
class ResolveModel : public torch::nn::Module {
    // ...
    std::unordered_map<std::string, torch::nn::Embedding> embeddings_;

    torch::Tensor forward(
        torch::Tensor continuous,
        const std::unordered_map<std::string, torch::Tensor>& embedding_ids
    );
};
```

## R Formula Parsing

Update `parse_resolve_formula()` to handle the generalized syntax:

```r
parse_resolve_formula <- function(formula, data, species) {
  # Extract terms: numeric(...), hash(...), embed(...), onehot(...)
  # For each term, determine:
  #   - columns involved
  #   - parameters (dim, top, bottom, by, scale)
  #   - data source (plot vs observation)

  # Return list of column specs that map to C++ ColumnSpec
}
```

## Migration Path

### Phase 1: C++ Generalization
1. Create new `PlotEncoder` class with generic column specs
2. Create `PlotRecord` and `ObservationRecord` types
3. Implement encoding logic for each type (hash, embed, numeric, onehot)
4. Update serialization format

### Phase 2: Backward Compatibility [SKIPPED]
SpeciesEncoder was removed directly since this is local development.

### Phase 3: R Bindings [DONE]
1. ✅ Updated R bindings to expose `PlotEncoder`
2. ✅ Updated `resolve()` to use new formula parsing with data/obs pattern
3. ✅ Updated `predict.resolve()` accordingly

### Phase 4: Python Bindings [DONE]
1. ✅ Mirrored R changes in Python bindings
2. ✅ Updated Python examples

### Phase 5: Cleanup [DONE]
1. ✅ Removed `SpeciesEncoder` completely
2. ✅ Updated all tests to use PlotEncoder
3. ✅ Updated documentation

## File Changes

### New Files
- `core/include/resolve/plot_encoder.hpp`
- `core/src/plot_encoder.cpp`
- `core/include/resolve/column_spec.hpp`
- `core/tests/test_plot_encoder.cpp`

### Modified Files
- `core/include/resolve/types.hpp` - Add new types
- `core/include/resolve/model.hpp` - Dynamic embeddings
- `core/src/model.cpp` - Dynamic embedding creation
- `bindings/r/src/bindings.cpp` - New encoder bindings
- `bindings/r/R/resolve.R` - Updated formula parsing
- `bindings/python/src/bindings.cpp` - New encoder bindings

### Deleted (Phase 5)
- ~~`core/include/resolve/species_encoder.hpp`~~ (removed)
- ~~`core/src/species_encoder.cpp`~~ (removed)

## Example Usage After Generalization

### R
```r
# Ecology example (current use case)
fit <- resolve(
  ph ~ elevation + latitude + hash(species, top = 5, by = "cover") + embed(genus, family),
  data = plots,
  species = species_df
)

# Adding climate zone
fit <- resolve(
  ph ~ elevation + hash(species, top = 5, by = "cover") + embed(genus) + embed(climate_zone),
  data = plots,
  species = species_df
)

# Functional traits
fit <- resolve(
  ph ~ elevation + hash(species) + embed(genus) + numeric(leaf_area, sla, scale = TRUE),
  data = plots,
  species = species_df  # species_df now includes trait columns
)
```

### Python
```python
encoder = PlotEncoder()
encoder.add_numeric("coords", ["latitude", "longitude"], scale=True)
encoder.add_hash("species", ["species_id"], dim=32, top_k=5, rank_by="cover")
encoder.add_embed("taxonomy", ["genus", "family"], dim=16, top_k=3, rank_by="cover")
encoder.add_embed("climate", ["climate_zone"], dim=8)

encoder.fit(plot_records, species_records)
encoded = encoder.transform(plot_records, species_records, plot_ids)

model = ResolveModel(config, target_configs)
outputs = model(encoded.continuous_features(), {
    "taxonomy": encoded.embedding_ids("taxonomy"),
    "climate": encoded.embedding_ids("climate"),
})
```

## Open Questions

1. **Aggregation for observation-level numeric variables**: If species_df contains numeric traits (SLA, leaf area), how do we aggregate to plot level? Options:
   - Weighted mean by abundance
   - Top-k mean
   - Separate aggregation spec

2. **Multiple observation tables**: What if user has species AND functional groups as separate tables? For now, assume single `species` argument.

3. **Interactions**: Should we support `hash(species:site)` for interaction hashing? Defer for now.

4. **Default `by` column**: If `top`/`bottom` specified but no `by`, should we error or use a default (e.g., count)?

## Timeline

- Phase 1: 2-3 days (C++ core)
- Phase 2: 1 day (backward compat)
- Phase 3: 1-2 days (R bindings)
- Phase 4: 1 day (Python bindings)
- Phase 5: 1 day (cleanup)

Total: ~1 week for full generalization
