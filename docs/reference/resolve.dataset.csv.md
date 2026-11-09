# Load a Dataset From CSV Files

Read plots and their species records from CSV files and encode them for
a model: the plot-level columns (coordinates, covariates, categorical
covariates, targets) of a header file, one row per plot, and the species
records of a species file, one row per record.

## Usage

``` r
resolve.dataset.csv(
  header,
  species,
  roles = list(),
  targets = list(),
  config = list(),
  schemaSource = NULL,
  vocabs = NULL,
  names = NULL,
  maxEditDistance = 1L
)
```

## Arguments

- header:

  Path to the header CSV file (one row per plot, with the targets).

- species:

  Path to the species CSV file (one row per species record).

- roles:

  Named list mapping column roles:

  - plot_id: Column name for plot ID (default "plot_id")

  - species_id: Column name for species ID (default "species_id")

  - abundance: Column name for abundance (optional)

  - longitude: Column name for longitude (optional)

  - latitude: Column name for latitude (optional)

  - genus: Column name for genus (optional)

  - family: Column name for family (optional)

  - covariates: Vector of covariate column names (optional)

  - categoricals: Vector of categorical covariate column names
    (optional)

  An optional role is unset by omitting it, or by giving it the empty
  string, which is the spelling to use when a role has to be cleared
  from a list that already carries it. A non-empty column name the file
  does not have is an error, so a typo fails loudly, and so is a key
  outside this list.

- targets:

  Named list of target specifications, the name naming the target. Each
  is a list of:

  - column: Column name in the header file (default: the target's name)

  - task: "regression" (default) or "classification"

  - transform: "none" (default) or "log1p"

  - num_classes: Number of classes for classification

  - weight: Loss weight (default 1.0)

  - class_mapping: Named integer vector pinning each class label to its
    code (default: codes fitted from the data, labels in sorted order)

  - missing: "drop" (default) or "mask". What happens to a plot with no
    label for this target (an empty or NA cell, or a class outside
    `class_mapping`). "drop" leaves the plot out of the dataset. "mask"
    keeps it: the plot trains and is scored on the targets it is
    labelled for, and this target's loss, scaler and metrics use only
    its labelled plots. A plot labelled for no target is left out either
    way.

  A key outside this list, two targets with one name, or an unnamed
  target is an error. A dataset for training needs at least one target;
  one built to be scored needs none.

- config:

  Named list of dataset configuration options; a field left out takes
  the engine's default (`resolve.default_config("dataset")`), and a key
  outside these fields is an error:

  - species_encoding: "hash", "embed", "sparse", "rank_pool" or
    "transformer". hash folds each plot's species into a fixed-width
    vector of `hash_dim` buckets; embed gives each of the plot's
    `top_k_species` ranked species slots a learned embedding; sparse is
    the plot's full species-by-abundance vector; rank_pool pools learned
    species, genus and family embeddings over every species the plot
    records; transformer attends over them.

  - hash_dim: Hash buckets of the hash encoding

  - top_k: Species hashed per plot and fixed genus/family slots

  - top_k_species: Ranked species slots of the embed encoding

  - selection: "top", "bottom", "top_bottom", or "all". Which species of
    a plot survive its per-plot budget: `top_k` for the hash encoding,
    `top_k_species` for embed (which rejects "all", having a fixed
    number of ranked slots), and `species_budget` for rank_pool,
    transformer and sparse.

  - species_budget: species kept per plot by `selection` for the
    rank_pool, transformer and sparse encodings; 0 = no budget, so every
    species a plot records is encoded and `selection` has nothing to
    narrow.

  - representation: "abundance" or "presence_absence". The value each
    species carries into every encoding: its abundance, or 1 under
    "presence_absence" (a pooled dataset then reports no cover). The
    selection still ranks by recorded abundance.

  - aggregation: "abundance" or "count". How the fixed genus/family
    slots of the hash, sparse and embed encodings rank a plot's taxa: by
    the summed value of their species, or by how many of the plot's
    species they hold. rank_pool and transformer pool each species'
    genus and family embeddings with the species weights, or under
    "count" with one weight per species.

  - normalization: "raw", "norm", or "log1p": the abundance values as
    read, scaled to sum to one per plot, or log1p-transformed

  - track_unknown_fraction, track_unknown_count: add the share and the
    number of a plot's species the vocabulary does not know as columns

  - use_taxonomy: encode the genus/family roles

  - pool_weighting: rank_pool weighting "binary", "abundance", "log1p",
    "norm", or "rank"

  - pool_species_cap: rank_pool per-plot species cap; 0 = no cap, -x =
    the (100 - x)th percentile of species per plot, computed on the data
    loaded (-1 = p99, -5 = p95; x in 1..99), \>0 = manual cap

  - missing_values: "indicate" or "zero": a missing covariate or
    coordinate cell is flagged and filled with the fitting fold's mean,
    or read as 0

  - zero_abundance_as: the abundance a recorded 0 is read at

- schemaSource:

  Optional ResolveDataset (from a previous `resolve.dataset.csv()`
  call). When supplied, this dataset is encoded against that dataset's
  species / taxonomy / categorical vocabularies and classification class
  mappings instead of fitting its own, so a held-out or transfer set
  lines up with the training set's embedding namespaces. Default `NULL`
  fits fresh vocabularies.

- vocabs:

  Optional vocabulary list from `predictor$vocabs()` (or
  `dataset$vocabs()`). Same effect as `schemaSource`, sourced from a
  CHECKPOINT rather than an in-memory training dataset, so the training
  CSVs need not exist. This is the path for scoring new data: every
  non-hash encoder indexes an embedding table with a code that is a
  function of the file the vocabulary was fitted on, so a dataset built
  without the training vocabularies looks up the wrong rows and
  `predictor$predict_dataset()` rejects it. Pair with
  `config = predictor$dataset_config()`. Mutually exclusive with
  `schemaSource`. A vocabulary list without a species vocabulary (a
  checkpoint that stores only the vocabulary sizes) is refused.

- names:

  Resolve the species names against the vocabulary of `vocabs` (or
  `schemaSource`) before encoding: the path of the release's
  `names.json`, which maps entered names and synonyms to the names the
  model learnt, or `TRUE` for the vocabulary alone. A name is taken as
  written where the vocabulary holds it, else through the table, else as
  the nearest known name within `maxEditDistance` edits;
  `dataset$species_names()` then lists the names not found as written,
  with the route each took. `NULL` (default) reads every name as
  written.

- maxEditDistance:

  With `names`: the edits a misspelled name may be away from a known
  name (default 1; 0 turns the spelling step off; 2 can also rename
  species the models never saw).

## Value

A ResolveDataset object (an Rcpp module class) with methods:

- coordinates(): Get coordinate matrix

- covariates(): Get covariate matrix

- hash_embedding(): Get hash embedding matrix

- genus_ids(), family_ids(): Get taxonomy ID matrices

- targets(): Get target values as named list

- schema(): Get dataset schema

- plot_ids(): Get plot IDs

- n_plots(): Get number of plots

- species_names(): with `names`, the names not found as written (a list
  of name, token, route, source, via, distance, n_records)

## Examples

``` r
plots <- system.file("extdata", "plots.csv", package = "resolveR")
species <- system.file("extdata", "species.csv", package = "resolveR")
roles <- list(plot_id = "plot_id", species_id = "species", abundance = "cover",
              genus = "genus", family = "family", latitude = "lat", longitude = "lon")
targets <- list(elevation = list(column = "elevation", task = "regression"),
                habitat = list(column = "habitat", task = "classification"))
dataset <- resolve.dataset.csv(
  plots, species, roles = roles, targets = targets,
  config = list(species_encoding = "hash", hash_dim = 64, top_k = 5)
)
dataset$n_plots()
dataset$schema()$n_species
```
