# RESOLVE Changelog

## v1.0.1 (2026-10-24)

### Added

- **Epochs selected by held-out accuracy.** `TrainConfig.selection_metric`
  (`SelectionMetric.Loss`, the default, or `SelectionMetric.Accuracy`) sets
  what early stopping, the plateau scheduler and the kept epoch follow. With
  `Accuracy`, an epoch is kept only when the held-out accuracy of the single
  classification target rises. `TrainResult.selection_history` records the
  selected value per epoch, and `progress.json` records the metric and the
  best value.

### Fixed

- **Transformer species encoding under mixed precision.** Masked attention
  and attention pooling run in float32 inside the autocast region, so
  `use_amp` no longer fails with `masked_fill not implemented for 'Half'`.
- **Heterogeneous GNN under mixed precision.** Messages are summed in their
  own type, so `index_add_` no longer mixes Half and Float.
- **Early stopping writes its own checkpoint.** The epoch early stopping
  fires at now saves its `checkpoint_<N>.pt` and `progress.json`, like every
  other epoch; resuming from it trains nothing more.

## v1.0.0 (2026-10-24)

### Added

- **Retrieval augmentation (TabR) in the engine.** `Predictor::set_retrieval`
  keeps the latents and labels of reference plots (normally the training
  plots) and blends every later prediction with the labels of the plot's k
  nearest reference plots by cosine similarity, weighted by a softmax over
  similarity / temperature: `(1 - alpha) * prediction + alpha * retrieved`,
  regression in original units, classification as class probabilities. The
  search is exact and needs no dependency. Python `Predictor.set_retrieval`
  with `RetrievalConfig`, R `resolve.retrieval()`, `resolve predict
  --retrieval-header / --retrieval-species / --retrieval-k / --retrieval-alpha
  / --retrieval-temperature`. The Python-only `resolve_core.tabr` (which
  averaged class codes into logits and needed faiss) is gone.
- **R reaches the whole engine.** `resolve.cross_validate()` (random or
  spatial-block folds), `modelConfig` / `trainConfig` on
  `resolve.train.dataset()` for every model and training field (any
  `encoder_architecture`, the heterogeneous GNN with its species graph, the
  normalization, activation and head settings), `pretrain = list(method =
  "jepa" | "scarf" | "masked_species", ...)` and `resolve.pretrain_vae()`,
  `resolve.fuzzy_index()` / `resolve.fuzzy_match()`,
  `resolve.set_vram_fraction()` and `resolve.default_config()`.
- **R examples run.** The package ships a small example survey
  (`system.file("extdata", "plots.csv", package = "resolveR")` and
  `species.csv`, 150 synthetic Alpine plots with elevation and habitat), and
  every help-page example trains, predicts or builds a suite on it whenever the
  engine is installed (`@examplesIf resolve.available()`), instead of sitting
  in `\dontrun{}`.
- `model_config_for_dataset(dataset, config)` (C++, Python, and behind the R
  verbs and `resolve train`) sets the fields that size a model to its data from
  the dataset; `target_specs_from_schema(schema)` lists a checkpoint's targets
  for loading labelled data.
- `ResolveDataset::load(header, species, ...)` takes each table as a CSV path or
  an in-memory table, with or without a model's vocabularies; every loader verb
  is a form of it, and the C ABI has the one `resolve_dataset_load`.

### Changed

- **The `resolve-core` project root is `src/core`.** Build from a checkout
  with `pip install ./src/core` (was `./src/core/python`). The published sdist
  now carries the engine sources the extension compiles, so `pip install
  resolve-core` builds from it on platforms without a wheel; CI builds a wheel
  from the sdist on every push. `requires-python` is `>=3.10`, matching the
  wheels (CPython 3.10 to 3.13).
- The R package's pkgdown reference is published at
  <https://gillescolling.com/resolve/r/>, under the documentation site, and the
  site build (`build_docs.py`, `r/build_site.R`) lives in the repository.
- **Model inputs travel as one value.** In C++ every model method takes a
  `PlotInputs` (`forward`, `forward_with_aux`, `forward_single`, `get_latent`,
  `encode_with_activations`, `get_gate_probs`, `Trainer::predict`), and the
  tensor overloads of `Trainer::prepare_data`, `Predictor::predict` and
  `Predictor::get_embeddings` take the continuous-block parts as a
  `ContinuousInputs` beside it. In Python the model methods and
  `Trainer.predict` keep their arguments; `Predictor.predict`,
  `Predictor.get_embeddings` and `Trainer.prepare_data_raw` take every tensor
  by keyword, and an unknown keyword raises `TypeError` on every such method.
  The `Spacc*` aliases are gone.
- `Predictor.predict` on tensors chunks the forward like `predict_dataset`
  (`batch_size`, default 4096) and, like it, returns CPU tensors whatever the
  device; a one-chunk prediction on CUDA used to return device tensors.
- **A species recorded several times in one plot is one entry.** Its records
  (one per vegetation layer) are merged before any encoding: abundances
  summed, placed where the first record stood, and every entry of a species
  carries the alphabetically first genus and family its records name. Every
  encoding, the vocabulary, the unknown-species columns (the count is now of
  species) and the recognition report read the merged entries. Retrain before
  comparing numbers on data with such records.
- **One species vocabulary policy.** Every encoding fits its species codes by
  the number of plots recording a species, ties by name; rank_pool and
  transformer used to rank by name. Saved checkpoints keep scoring, since
  their vocabularies travel by name. `SpeciesVocab.from_ordered`,
  `SpeciesVocab.names` and `EncodingVocab` are new; the standalone encoders
  adopt a vocabulary with `set_vocab` and no longer return unknown-species
  statistics (call `compute_unknown_species_stats`, which takes a
  `representation`).
- **Every species encoding reaches every tabular architecture.** rank_pool and
  transformer feed FT-Transformer, TabNet, SAINT, GNN and ExcelFormer their
  pooled species embedding (genus and family pooled in) and the `has_cover`
  flag; they used to fail with a shape error. A pair the encoding cannot feed
  is refused when the model is built: HeterogeneousGNN and the co-occurrence
  GNN graph need `sparse`, the taxonomic GNN graph the fixed taxonomy slots
  (not rank_pool / transformer). `uses_explicit_vector` turns hash into sparse
  everywhere (model, adapter, the inference-side dataset config) and is refused
  with embed, rank_pool and transformer.
- **`aggregation = count` works under rank_pool and transformer** (it was
  refused): the genus and family embeddings are pooled with one weight per
  species instead of the species weights; under transformer they leave the
  tokens and are added to the pooled vector.
- `ModelConfig.transformer_pooling` is the enum `TransformerPooling`
  (`Attention`, `Cls`) in Python and C++; R and the CLI keep the names
  `"attention"` / `"cls"`, and checkpoints that stored the name still load.
- **The TabM tail honours `activation`, `normalization` and `use_residual`**
  (it always ran LayerNorm and GELU). Under the default `normalization =
  BatchNorm` a TabM model is built differently: retrain TabM checkpoints
  trained without `normalization = LayerNorm`.
- `Trainer.compute_diagnostics` reports every hidden layer of every species
  encoder's tail (plain, residual, mixture backbone, TabM, parallel branches);
  it used to report "not available" for everything but hash.
- In C++ the species encoders (`PlotEncoder`, `PlotEncoderEmbed`,
  `PlotEncoderSparse`, `PlotEncoderRankPool`, `PlotEncoderTransformer`) are
  built from a `SpeciesEncoderSpec` and read a `PlotInputs`; their weight
  getters are `species_weights` / `genus_weights` / `family_weights`. The
  `RESOLVE_FP32_NORM` switch is gone (normalization always runs in fp32 under
  AMP), and `MixtureOfExperts` refuses `MoERoutingType::None`.
- `ResolveSchema.n_genera_vocab` / `n_families_vocab` are removed: they always
  equalled `n_genera` / `n_families`, which size the taxonomy tables (also in
  the C ABI / R schema list).
- **The schema's loader settings are one `DatasetConfig`.** The flat schema
  attributes `track_unknown_fraction`, `track_unknown_count`, `top_k_species`,
  `selection`, `species_budget`, `representation`, `normalization`,
  `aggregation`, `use_taxonomy`, `missing_values`, `zero_abundance_as`,
  `pool_weighting` (an integer) and `pool_species_cap` moved to
  `schema.dataset_config` (Python), `schema()$dataset_config` (R, C ABI), which
  holds every `DatasetConfig` field, `pool_weighting` as the enum.
  `schema.unrecorded_dataset_settings` lists the settings a loaded checkpoint
  predates; they read as the engine read them before the setting existed.
- `resolve info` prints the checkpoint's description, the document the JSON
  sidecar holds, so both show the same fields: the run, the schema (targets
  with their class names, covariates, categoricals), the model config, the
  loader settings and the persisted training config. The sidecar no longer
  lists the device, AMP / cuDNN switches or checkpoint destination, which the
  checkpoint does not carry. Non-finite numbers are `null`, and every number
  is written at full precision in the C locale (also in `progress.json` and
  suite manifests).
- **Every `resolve train` setting is a flag, generated from the config
  structs.** New flags reach the ModelConfig fields the CLI could not set
  (`--species-embed-dim`, `--genus-emb-dim`, `--family-emb-dim`,
  `--categorical-embed-dim`, `--activation`, `--norm-layer`, `--norm-groups`,
  `--residual`, `--leaky-relu-slope`, `--elu-alpha`, `--head-hidden-dims`,
  `--head-activation`, `--head-dropout`) and the TrainConfig fields
  (`--phase-boundaries`, `--amp-init-scale`, ...); every boolean has its
  `--no-` form. Defaults are the structs': `--n-attention-layers` defaults to
  0 (it said 2). A flag for a setting that does not apply is refused
  (`--d-model` without `--encoding transformer`, the mixture's hyperparameters
  without `--moe-routing`, the NCA, step-schedule and cosine-schedule knobs
  outside their loss or schedule, `--cover-dropout` outside rank_pool /
  transformer, `--checkpoint-every` without `--checkpoint-dir`). Numbers parse
  whole (`256x`, `1,5` are errors). `--cuda` on a machine without a CUDA
  device is an error rather than a CPU run.
- A suite's predictions as a table come from the engine
  (`resolve_core.suite_prediction_columns`): `SuitePredictions.to_pandas`, R's
  `as.data.frame()` and `resolve predict --suite` share its columns and its
  one rule for a class code without a label (the code itself). `resolve help`
  lists the spherical-mean columns.
- Suite manifest format 3: a member may state the `zero_abundance_as` its
  training data was read at, for a checkpoint that predates the setting
  (`SuiteMember.zero_abundance_as`). Manifests that do not use it are written
  and read as before. `SuiteManifest.verify(dir, targets=)` checks the named
  targets only.

- **Devices are named `cpu`, `cuda` or `cuda:N` everywhere** (Python, R, the
  C ABI, configs). Any other spelling (`"CUDA"`, `"gpu"`, `"cuda:x"`) is an
  error; it used to run on the CPU. The CLI's `--cuda` is now `--device DEVICE`.
- **Configuration lists are read strictly.** A key the engine does not know, at
  any level of nesting, a key given twice, and a value of the wrong kind are
  errors naming the field and the accepted keys; `config = list(species_encodng
  = "embed")` used to train with the default.
- **The R verbs take the engine's defaults.** `resolve.train.dataset()`'s
  arguments default to `NULL`, the engine's value: in particular `lossConfig`
  is now `"combined"` unless given (it was `"mae"` in R alone). The R package
  holds no copy of a default, a key list or an enum spelling.
- A storage failure (a checkpoint or CSV that cannot be read, a checkpoint that
  cannot be written) is `resolve_core.StorageError`, an `OSError`, after the
  engine's own retries; `resolve_core.retry_io` and the Python retry wrappers
  are gone.
- Every tensor a Python accessor returns is a detached CPU tensor, or `None`;
  the model's forward outputs and parameter accessors stay live on its device.
- Long Python calls release the GIL (CSV loads, save / load / load_state /
  resume, `Predictor.load`, `SuiteManifest.read`, the `compute_*` evaluators).
- `PYTORCH_CUDA_ALLOC_CONF`'s default has one definition in the engine
  (`resolve_core.default_cuda_alloc_conf()`).
- `resolve_core.species_graph` (an unused Python copy of
  `build_species_graph` with other edge codes) is gone.

### Fixed

- R accessors of 64-bit integers (`raw_species_ids()`, `plot_offsets()`, any
  int64 result) return doubles, exact to 2^53, when a value exceeds R's
  integer range; they used to narrow with one warning per element.
- A Python exception raised by a fuzzy index's `bucket_fn` inside
  `query_batch` is raised to the caller after the batch; it used to terminate
  the process.
- The C ABI's value builders return NULL / -1 on failure instead of
  terminating, refuse a NULL map, and free a partly built result tree.
- `Trainer.load` returns `(ResolveModel, Scalers, CategoricalVocab)` in Python
  (it raised `TypeError`).

- `representation = presence_absence` was honoured by the sparse encoding
  only. It now sets every species' value to 1.0 in every encoding (hash
  weights, pool weights with `has_cover` 0, the abundance-ranked taxonomy
  slots, the unknown fraction), while the selection still ranks by recorded
  abundance.
- `aggregation` was stored and never read. It now ranks the fixed genus /
  family slots of the hash, sparse and embed encodings (`abundance`: summed
  value of a taxon's species, `count`: how many species it holds); rank_pool
  and transformer reject `count`.
- Embed filled the slots of a plot holding no more than `top_k_species`
  species in file order; the slots now follow the abundance ranking in every
  plot. Retrain embed models before comparing numbers.
- The hash and sparse taxonomy slots aggregated by each record's own genus
  rather than the species' resolved genus the embed slots and the vocabulary
  use.
- A fuzzy index returned only the first of several entries that fold to the
  same key (case-insensitive duplicates, repeated entries); it returns all of
  them.
- `RankPoolEncodedData`, `EmbeddingEncodedData` and `UnknownSpeciesStats`
  tensor fields raised a conversion error in Python.
- Loading a checkpoint onto a CUDA device read its strings, class weights and
  categorical codes from device memory and crashed.
- `Predictor.get_embeddings` read only the coordinates, covariates, hash
  embedding and taxonomy slots, so it threw or returned a wrong latent for the
  embed, sparse, rank_pool and transformer encoders and ignored categorical
  columns. It takes every model input now and is the latent `predict(...,
  return_latent=True)` returns, for every encoder.
- `forward_single` and `get_gate_probs` take the species IDs, species vector
  and pool tensors, so they work for every species encoding instead of
  refusing the encoders that read composition from them.
- A suite member trained reading a recorded abundance of 0 as another value
  than the input contract states was accepted when it read 0 (and then scored
  at the contract's value); it is refused, as are a member reading genus or
  family the contract does not name and one trained on an abundance column
  the contract does not read.
- A circular mean of bearings whose unit vectors cancel returned an arbitrary
  bearing with a huge spread; it returns NaN for both, as the spherical mean
  does. Every combine rule reports a one-member dispersion as NaN (the
  circular mean said 0).
- Prediction CSVs wrote six significant digits (`1.23457e+06`, `47.1235`);
  they write the shortest exact decimal.
- A checkpoint string or class-weight list whose stored length disagrees with
  its data threw nothing and read out of bounds; it is refused.
- `ResolveSchema.has_abundance` was always true; it records whether the
  species records had an abundance column.
- The Catch2 suite force-includes a prelude that keeps `CHECK` Catch2's
  assertion: in files that included libtorch after Catch2, libtorch's own
  `CHECK` macro had replaced it, so those checks were neither counted nor
  reported.

## v0.12.0 (2026-10-11)

### Added

- **Each target of a model suite can read its own part of the input
  contract.** A suite target may declare `inputs` (`coordinates`,
  `covariates`, `categoricals`): the columns its members read, a subset of the
  suite's input contract. A model of the survey year cannot read the year, and
  a model of the plot location cannot read the coordinates, so a suite reading
  species, coordinates and date can now release both beside its other
  targets. The contract must be the union of what the targets read. Members
  are checked at load against their own target's inputs, and the input plots
  are encoded once per vocabulary and set of columns read. A target that
  declares nothing reads the whole contract, as before: the released suites
  predict bit-identically to 0.11.1.
- **`spherical_mean` combine rule for a location.** A target predicting
  latitude and longitude combines its members as the mean of their unit
  vectors on the sphere (`combine_spherical`), so seeds either side of the
  antimeridian meet there rather than on the far side of the globe. Its
  dispersion is the median great-circle distance of the members from the
  combined point, in km. `value` is a two-column latitude / longitude array;
  every front door writes `<target>_latitude`, `<target>_longitude` and
  `<target>_spread_km`. Every suite prediction now names its dispersion's
  units (`dispersion_units`).
- `resolve info --suite` lists what each declaring target reads.

### Changed

- **Suite manifest format version 2.** A manifest that declares per-target
  inputs or uses a spherical mean is written as version 2; any other manifest
  is still written as version 1 and stays readable by 0.11 engines. A
  manifest stating version 1 while using a version-2 feature is refused.

## v0.11.1 (2026-09-29)

### Fixed

- **A column constant in the fitting rows no longer saturates a prediction**
  (#118). The standardisation scale is the standard deviation plus 1e-8, so a
  continuous column that never varied in the fitting rows was stored with scale
  1e-8. The unknown-species fraction is such a column, since the species
  vocabulary is built on the fitting rows: a plot holding one unseen species at
  1% cover entered the network at 1e6, and every such plot received the same
  prediction, in evaluation on test plots as in later prediction. Such a column
  now standardises to 0, the value it held throughout training, whatever its
  value. Checkpoints store the scale that identifies it, so models trained by
  earlier versions are corrected without retraining; predictions change only
  for plots in which such a column differs from its fitted value.

## v0.11.0 (2026-09-29)

### Added

- **Model suites: released weights scored as one ensemble.** A suite is a
  directory of checkpoints and a `manifest.json` (format `resolve-suite` v1)
  holding the input contract and, per target, its members with SHA-256 and
  size, the combine rule (vote, mean, or circular mean from a bearing or a
  sine-cosine pair), units, released or experimental status with its stated
  limit, training and validation records, licence and provenance.
  `SuitePredictor` verifies every checksum, checks each member against the
  contract, encodes the input once per vocabulary and reports per plot the
  combined prediction, seed agreement or dispersion, and species recognition
  by count and by abundance; it reports no combined support level. A tied vote
  goes to the class with the higher mean probability, and only an exact
  probability tie to the lowest code. `DatasetConfig::zero_abundance_as` is
  recorded in the checkpoint so a suite applies the training data's reading of
  a zero cover. Reached through `resolve predict --suite` / `info --suite`,
  the C ABI's `resolve_suite_*`, nanobind `SuitePredictor` /
  `SuitePredictions.to_pandas` and R `resolve.load_suite`,
  `resolve.predict.suite`, `resolve.verify_suite` and `resolve.seal_suite`.

- **Fixed-duration training with nothing held out.**
  `TrainConfig::fixed_epochs` runs exactly that many epochs of the
  `max_epochs` schedule, with no early stopping, and returns the final
  weights; the learning-rate schedule stays laid out over `max_epochs`. It is
  the one mode that trains with `test_size = 0`: a validated configuration
  refitted on every labelled plot for a duration fixed beforehand. `fit()`
  refuses `test_size = 0` without it, a negative value and a value past
  `max_epochs`. CLI `--fixed-epochs`, R `fixedEpochs`.

- **Every architecture hyperparameter is now a CLI flag.** `resolve train`
  could select an encoder architecture and then left every field of its
  sub-config at the default: no flag existed for any of them, so a standalone
  run could not train what the bindings could -- the gap issue #104 closed for
  covariate columns and the mixture, one level down. The rows and the reads
  come from the same field registry that drives the checkpoint, the C-ABI value
  tree, nanobind and `resolve info` (`cli/config_flags.hpp`), so a field added
  to any sub-config gets its flag, its help line and its read in the edit that
  adds the field. The flag is the field's checkpoint key with dashes --
  `--ft-d-model`, `--tabnet-virtual-batch-size`, `--gnn-graph-mode`,
  `--hgnn-k-cooccurrence`, `--trait-interaction`, `--parallel-aggregation` --
  because that key already carries the struct prefix, and nine sub-configs
  repeat member names. A boolean is the CLI's usual pair of presence flags
  (`--tabnet-use-sparsemax` / `--no-tabnet-use-sparsemax`), an enum lists its
  accepted spellings in the generated help, and every default column shows the
  value the struct actually carries. The parallel block's branches are
  variable-length, so they keep a grammar of their own:
  `--parallel-branch DIMS[:ACTIVATION[:NORMALIZATION[:DROPOUT[:WEIGHT]]]]`,
  repeated per branch. `--parallel-enabled` with no branch is refused rather
  than ignored.

- **`heterogeneous_gnn` passes messages on a species graph the engine builds.**
  The architecture reads a typed graph over the species vocabulary, and
  `HeterogeneousGNNConfig` carried four fields describing how to build one --
  `use_taxonomic_edges`, `use_cooccurrence_edges`, `k_cooccurrence`,
  `cooccurrence_threshold`. No engine code read any of them, and no public
  surface could hand a graph in either: `set_species_graph` existed on the
  adapter alone, which nothing exposes, so selecting the architecture built a
  model whose every forward threw "Species graph not set." `build_species_graph`
  (`species_graph.hpp`) now joins species by shared genus, by shared family and
  by co-occurrence, as the configuration asks: a pair standing in two relations
  gets one edge of each type, and the numbering (same genus 0, same family 1,
  co-occurrence 2) is the index the encoder's edge-type embedding looks up.
  Co-occurrence counts how often two species share a plot, as a share of all
  plots, keeping each species' strongest `k_cooccurrence` partners above
  `cooccurrence_threshold`; it reads the per-plot species vector, so it needs
  the sparse encoding, and the count runs in blocks of 512 species so its memory
  does not grow with the vocabulary. `Trainer::prepare_data` is where the graph
  comes from -- the one place with both the model and the data -- and
  `Trainer::save` writes it into the checkpoint, because the graph is part of
  the trained model and the training data is not around at scoring time. A
  requested relation the dataset cannot supply, both relations switched off, an
  edge type numbered past `n_edge_types` and an edge list too large to hold are
  each refused by name. Along the way `ResolveDataset` gained the taxonomy of
  its own vocabulary, `species_genus_ids()` / `species_family_ids()` (index by
  species code, read the genus or family code), which is what the taxonomic
  relation is built from. Surfaces: `resolve_core.SpeciesGraph`,
  `SpeciesEdgeType`, `build_species_graph`, `ResolveModel.set_species_graph` /
  `.has_species_graph` / `.requires_species_graph` / `.species_graph_edge_index`
  / `.species_graph_edge_type`, the dataset accessors; C-ABI
  `resolve_build_species_graph`, `resolve_model_set_species_graph` and the
  matching `resolve_model_get` / `resolve_dataset_get` keys; R
  `resolve.species_graph()`, `dataset$species_genus_ids()` and the
  `Model$set_species_graph()` family.

- **`PretrainConfig::mixup_alpha` mixes the contrastive view.** The second view
  of a row is mixed with another row of the batch at a weight drawn from
  Beta(alpha, alpha), the augmentation SAINT pairs with feature corruption
  (Somepalli et al., arXiv:2106.01342). A pretext task has no label to mix, so
  the row keeps the larger share of itself and stays the positive pair of view
  1. The draw goes through `PretrainRng::beta_symmetric`, so a mixed run
  reproduces from its seed like every other pretraining draw. 0, the default,
  switches it off, which is what SCARF alone does.

- **`ModelConfig::freeze_composition` keeps the composition tables at their
  initialisation.** A fixed-representation control (the pooled encoder with
  random, untrained species embeddings) previously had to switch gradients off
  from the calling code, which no checkpoint recorded. The engine now does it:
  with the knob on, `ResolveModel` stops the gradient of the tables its species
  encoder reads composition through, and AdamW, which skips a parameter without
  a gradient, leaves them untouched by weight decay as well. Those tables are
  the species, genus and family embeddings of the embed, rank_pool and
  transformer encoders, the species projection and per-rank taxonomy
  embeddings of the sparse encoder, and the per-rank taxonomy embeddings of
  the hash encoder; `ResolveModel::composition_parameters()` returns them. A
  model with none (an adapter architecture, TraitNet, hash without taxonomy)
  refuses the knob. The field is a config-registry row, so it round-trips
  through checkpoints, the JSON sidecar, the C-ABI config tree and nanobind;
  CLI `resolve train --freeze-composition`, R
  `resolve.train.dataset(freezeComposition = TRUE)`, and
  `composition_parameters()` on the nanobind model, the C-ABI model getter and
  the R module.

### Changed

- **The R package is `resolveR`.** Bioconductor already carries a package
  named RESOLVE and CRAN checks names case-insensitively across both, so the
  R client is renamed. The `resolve.` function prefix, the `resolve_c` backend
  and the release assets are unchanged; the user data directory is now
  `R_user_dir("resolveR")`. The metric functions (`resolve_mae`,
  `resolve_rmse`, `resolve_smape`, `resolve_r_squared`,
  `resolve_band_accuracy`, `resolve_accuracy`) are exported under one help
  page.

- **Any percentile species cap.** `pool_species_cap = -x` is the
  (100 - x)th percentile of records per plot for x in 1..99; -100 or below is
  refused. Before, only -1 (p99) was a percentile and any other negative value
  silently meant no cap. -1 resolves as it did.

- **A missing covariate or coordinate is flagged and filled instead of read
  as zero.** The loader wrote a blank covariate cell as 0.0 and a blank
  coordinate as (0, 0), so a model could not tell a recorded 0 from a missing
  value, and a plot without coordinates sat in the Gulf of Guinea. The loader
  now keeps such a cell as `NaN` in `coordinates()` / `covariates()`, and
  `DatasetConfig::missing_values` decides how the model reads it. Under the
  new default, `MissingValuePolicy::Indicate`, the continuous block carries a
  0/1 column per covariate and one for the coordinate pair, and the value is
  filled with the mean of that column's recorded values on the fitting rows
  before standardisation. `MissingValuePolicy::Zero` keeps the earlier
  behaviour. The block is assembled, filled and standardised in one module,
  `continuous_block.{hpp,cpp}`, which the Trainer, the cross-validation
  folds, `Trainer::predict`, `Predictor::predict` and
  `Predictor::get_embeddings` all call, so the column layout cannot differ
  between fitting and scoring. Cross-validation marks a filled cell missing
  again before each fold refits its fill. `SpatialBlockSplitter` keeps a
  plot without coordinates in every training fold and out of every test
  fold. The policy is persisted on `ResolveSchema` (`schema_missing_values`)
  and the fill on `Scalers` (`continuous_fill`); a checkpoint written before
  either key existed loads as `Zero` with no fill and predicts exactly as
  before. Surfaces: `resolve_core.MissingValuePolicy`,
  `DatasetConfig.missing_values`, `ResolveSchema.missing_values` /
  `missing_flag_width()`, `Scalers.continuous_fill`,
  `ResolveDataset.continuous_block(include_hash)`; the C-ABI config and
  schema trees and the scalers value; R `config = list(missing_values =
  "indicate")`; CLI `resolve train --missing-values {indicate,zero}`.
  **Retrain to use it**: a model trained under `Zero` keeps reading missing
  values as zero.

### Fixed

- **Data encoded against a training schema keeps the training species
  width.** `from_csv_with_schema` and every other loader that reuses a
  training dataset's or a checkpoint's vocabularies took only the
  vocabularies, so a percentile `pool_species_cap` was re-resolved on the new
  data: a held-out half was truncated at its own 99th percentile of records per
  plot, while checkpoint inference and the suite path truncated at the width
  the model trained on. The two paths scored the same plots differently
  wherever the new data's plots ran longer than the training data's. A
  withheld database whose 99th percentile was 72 records against a training
  width of 61 scored 180.59 m altitude error one way and 180.32 m the other.
  `ExternalVocabs` now carries the resolved width and the loader applies it in
  place of the config's cap; a source that recorded none leaves the caller's
  cap in force. The width round-trips through the C ABI and is readable from
  Python as `ExternalVocabs.pool_species_cap`.

- **Unlabelled plots can be scored.** `resolve predict --model` required the
  target columns and dropped plots without them, the R loaders refused
  `targets = list()`, and the C ABI read a list of target names as none.

- **A standardization scale that cannot be estimated is 1, not NaN.** The
  sample standard deviation of a single row is NaN, and dividing by it
  standardized every value -- and from there every prediction and every weight
  -- into NaN in silence. A one-row fitting fold comes out of an ordinary
  `test_size` on a small dataset. `standardization_scale`
  (`continuous_block.hpp`) is the one definition, used for the continuous block
  and for each regression target: the sample standard deviation, offset so a
  constant column does not divide by zero, and 1 where it cannot be estimated
  at all.

- **An unknown `transformer_pooling` is refused rather than read as CLS
  pooling.** The pooled vector is chosen by name and the forward treated
  anything that is not `"attention"` as CLS, so a misspelling selected a
  different architecture from the one asked for without a word.
  `PlotEncoderTransformer` now accepts `"attention"` and `"cls"` and names both
  when refusing anything else, the way `TabMConfig::aggregation` already did.

- **Fifteen architecture fields that reached no engine code now shape the
  model.** The field registry makes every configuration field round-trip
  through the checkpoint, the C ABI, nanobind, R and `resolve info`
  automatically; what it cannot check is whether the engine READS the field,
  and a sweep found fifteen that did not. A run could set one, save it, print
  it back, and train a model shaped by none of it -- the defect class of
  `TabNetConfig::use_sparsemax` and of `SelectionMode` outside the hash
  encoding.

  - `FTTransformerConfig::ffn_dropout` appeared in no source but its own
    declaration: the adapter passed `attention_dropout` as the block's single
    dropout rate, so the feed-forward layer and the residual branches ran at
    the attention rate. `TransformerBlock` now takes a `TransformerBlockConfig`
    carrying the three rates the transformer literature separates, and
    FT-Transformer feeds each to the place it names. SAINT and ExcelFormer
    configure one rate, which `uniform` puts at all three, so they are
    unchanged.
  - `TabNetConfig::virtual_batch_size` was documented as the ghost batch norm
    size with no ghost batch norm behind it. `apply_ghost_batch_norm`
    normalizes each slice of the batch against its own statistics through the
    block's own `BatchNorm1d`, so the affine parameters, the running statistics
    and the parameter names are unchanged, and every block of the feature and
    attentive transformers runs through it with the input normalization left
    full-batch as Arik & Pfister have it.
  - `ExcelFormerConfig::pre_norm` was hardcoded to pre-norm in the encoder,
    along with the trailing LayerNorm that belongs to that arrangement.
  - `GNNConfig::graph_mode` selected nothing: every mode built the spatial
    graph. `build_knn_adjacency` now takes the features and the metric, so
    `Taxonomic` measures cosine distance between per-plot genus/family
    composition and `CoOccurrence` between species vectors, and a mode whose
    input the dataset does not carry is refused by name rather than measuring
    distance on whatever sits in the first two columns.
    `GNNConfig::use_edge_features` carries each edge's similarity as its
    adjacency weight -- a Gaussian kernel of the distance for coordinates, the
    cosine similarity for composition -- instead of a plain 1.
  - `TraitNetConfig::interaction_dim` is the width the environment-trait
    combination produces, which the head reads; it was pinned to the width of
    the environment encoding. `TraitNetConfig::interaction` selects that
    combination (`Bilinear`, `MLP`, `Attention`), where bilinear was hardcoded
    and the enum had no consumer at all.
    `TraitNetConfig::shared_trait_encoder = false` gives every species its own
    trait encoder, a grouped matrix per layer, where the shared encoder applies
    one matrix to every species.
  - `ModelConfig::parallel_layers` was read by nothing, so `ParallelBlock` --
    branches, five aggregations, the residual path, all implemented -- was
    never constructed, and the whole architecture was unreachable while
    `resolve info` reported its configuration. The block is now the encoder's
    tail for all five species encodings, and
    `ParallelBranchConfig::branch_weight` is what a branch contributes to the
    aggregation. TabM, a tail-placed mixture and a parallel block all replace
    the same MLP tail, so at most one may be enabled.
  - `SAINTConfig::use_contrastive_pretrain` and `SAINTConfig::mixup_alpha` are
    gone. Both described a self-supervised pre-training stage, which a model
    configuration cannot run: a pretext task is a training stage, and the
    engine runs those through the pretraining API. The augmentation they named
    is now `PretrainConfig::mixup_alpha`, where it acts (see Added). A
    checkpoint carrying the two keys still loads; they are simply not read.

  **Behaviour changes without a shape change, so every checkpoint still
  loads**: retrain an FT-Transformer model to train at the configured FFN rate,
  and a TabNet model to train with ghost batch normalization. A TraitNet
  checkpoint DOES change shape, because its interaction width was the
  environment width and is now `interaction_dim` -- **retrain a TraitNet
  model**, or set `interaction_dim = 0` to keep the old width.

- **A missing covariate no longer reaches a pretext task as NaN.** The loaders
  keep a missing covariate or coordinate as NaN under
  `MissingValuePolicy::Indicate`, the default, and `Trainer::prepare_data`
  fills it from the fitting rows before any forward. A pretrainer takes the
  continuous block straight from the dataset and has no fitted scalers, so an
  unfilled block put NaN through the objective and from there into every
  weight, with no error and no warning. `run_pretrain_loop`, the one place all
  four pretext tasks pass through, now fills each missing cell with its column
  mean -- the same fill training computes for the same rows, extracted as
  `continuous_column_fill` / `fill_missing_continuous` so the two cannot drift
  -- and logs that it did.

- **`Predictor::get_embeddings` assembled its continuous block in a different
  column order from training** (coordinates, hash embedding, covariates,
  where the Trainer places the hash last) and never appended the
  unknown-species columns, so a hash model's embeddings read shuffled inputs
  and a model tracking unknown species could not be embedded at all. It now
  calls the shared block assembly and takes `unknown_fraction` /
  `unknown_count`, which it requires when the model reads them (nanobind
  keywords, C-ABI keys, and an R method overload with the two extra
  arguments).

- **Early stopping no longer waits for a loss phase that changes nothing.**
  Patience counted only once training reached the phase the last epoch is
  in, so the SMAPE and band terms of the combined regression loss get to
  train before a run can stop. That gate applied to every run, including
  classification-only fits, whose loss has no phases, and so a EUNIS fit at
  the default phase boundaries trained about 300 epochs before patience could
  start: one run whose best validation loss came at epoch 12 stopped at epoch
  349 rather than near epoch 62. `MultiTaskLoss::objective_settled` now
  decides when patience counts: from the first epoch when there is no
  regression target or the preset's phased terms carry no weight (MAE,
  SMAPE), from the final phase otherwise. The kept weights are still the
  best by validation loss among the epochs run, but such a run now ends
  `patience` epochs after that best instead of after the curriculum, so a
  validation loss that would have improved again later is no longer reached.

## v0.10.0 (2026-09-08)

### Changed

- **The R package is `resolveR`.** Bioconductor already carries a package
  named RESOLVE, and CRAN checks names case-insensitively across both
  repositories, so the R client is published as `resolveR`. The functions
  keep their `resolve.` prefix, the backend stays `resolve_c`, and the
  install path is `library(resolveR)`. The four exported removal stubs
  (`resolve.encoder()`, `resolve.dataset()`, `resolve.train()`,
  `resolve.predict()`) are gone rather than erroring, and the engine's
  metric functions (`resolve_mae()`, `resolve_rmse()`, `resolve_smape()`,
  `resolve_r_squared()`, `resolve_band_accuracy()`, `resolve_accuracy()`)
  are exported and documented.

### Added

- **Class probabilities from the Predictor (#117).** `Predictor::predict`
  returned a classification target's argmax code and discarded the softmax
  row behind it, so a checkpoint scored on a separately staged test set had no
  per-plot confidence; only `Trainer::compute_classification_predictions`
  exposed probabilities, and only for the trainer's own held-out fold.
  `ResolvePredictions` now carries `probabilities`: one float
  `(n_plots, n_classes)` tensor per classification target, each row the
  softmax over the classes, column `j` = P(class code `j`), row-wise argmax
  equal to `predictions`. Both the one-shot and the chunked predict path fill
  it; regression targets have no entry. Surfaces: nanobind
  `ResolvePredictions.probabilities`; the C-ABI predictions tree gains a
  `probabilities` map of row-major double matrices; R
  `resolve.predict.dataset()` returns it under `$probabilities`, with the
  columns named by the checkpoint's class labels; CLI
  `resolve predict --probabilities` writes one column per class,
  `<target>_prob_<class>` in code order, after `<target>_code`. Contract in
  `tests/test_predictor.cpp`, `tests/core/test_predictor.py`,
  `r/tests/testthat/test-roundtrip.R`, and a `cli-e2e` step. Not a cutover:
  no checkpoint field or tensor shape changed, and every existing accessor
  is unchanged.

## v0.9.1 (2026-09-01)

### Fixed

- **A structured argument that names nothing is rejected, not discarded.**
  `targets = list(list(column = "area", task = "regression"))` -- an unnamed R
  list -- built a dataset carrying zero targets and reported nothing wrong. The
  run then died at `loss.backward()` with `element 0 of tensors does not require
  grad and does not have a grad_fn`, an autograd message pointing nowhere near
  the target specification that caused it. Three layers were each discarding
  what they could not read, and each now says so.

  - `r_list_to_value_map()`, the R client's boundary, normalized every non-map
    value to an empty map so that the zero-length case would work, which
    discarded a populated unnamed list along with it. It now normalizes only
    `NULL` and a zero-length list, and throws for a non-empty list that produced
    no keys, naming the argument.
  - `parse_targets()` in the C ABI returned an empty vector for any tree that
    was not a map. Absent or `NULL` still means "no targets"; any other keyless
    kind now throws, as do an empty target name and a non-map specification. A
    shared `reject_unknown_keys()` also makes `parse_targets()` and
    `parse_roles()` reject a key they do not read, naming it and listing the
    accepted spellings, so `name` / `type` written for `column` / `task` is
    reported rather than ignored.
  - `Trainer::prepare_data()` throws when the data carries no targets. Both
    overloads pass through one body, so the invariant is stated once and holds
    for the CLI, Python and R alike. It sits there rather than at the loader
    because a target-less dataset is legitimate: that is what an inference set
    is, and `Predictor::predict()` never prepares a split.

  R validates at the front door as well, where the caller can still see what
  they typed: `resolve.dataset.csv()` and `resolve.dataset.frame()` require
  every target to be named and uniquely named, and reject an unrecognized key in
  a target specification or in `roles`, against the same key sets the C-ABI
  parsers read.

  No tensor shape, parameter or archive key changed, and every well-formed call
  behaves exactly as before.

## v0.9.0 (2026-08-31)

### Added

- **A mixture of experts is available to every species encoding, and
  `moe_routing` means one thing.** The knob used to select two different
  architectures depending on `species_encoding`: hash built a dedicated encoder
  whose mixture REPLACED the last MLP stages, embed / sparse / rank_pool /
  transformer got a dim-preserving mixture bolted onto the finished latent, and
  the adapter architectures were refused outright. Nothing in any suite
  constructed a model with routing on, so none of it was observable.

  Placement is now explicit, as **`ModelConfig.moe_placement`**:

  - `tail` (the default) makes the experts the encoder's final stage --
    `hidden_dims` minus its last two widths becomes the backbone and the
    mixture projects that to `hidden_dims.back()`, which stays the latent. This
    is what hash mode always did, and it is now open to all five species
    encodings.
  - `post` runs the mixture over the finished latent, preserving its width.
    This is the placement for an encoder with no MLP tail to give up, so the
    adapter architectures (FT-Transformer, TabNet, SAINT, GNN, ExcelFormer,
    HeterogeneousGNN) gain a mixture where they previously got a refusal.

  Asking a tail-less encoder for `tail` now raises an error naming `post`, and
  asking for TabM and a mixture together raises rather than dropping TabM in
  silence, which is what the dedicated MoE encoder did (it took no `TabMConfig`
  at all).

  The mixture reaches every encoding because the encoder tail is now one shared
  thing (`EncoderTail` in `encoder.hpp`) rather than a per-encoder copy: all
  five encoders build it through `build_encoder_tail` and run it through
  `forward_encoder_tail`, so an MLP, a TabM ensemble and a backbone-plus-mixture
  are three settings of one seam. `PlotEncoderMoE`, which duplicated the hash
  encoder's featurisation to bolt a mixture on the end, is gone, and the three
  copies of the encoder-dispatch if-else chain in `model.cpp` collapse into one
  `encode_all`.

- **`resolve train --moe-routing / --moe-placement / --n-experts /
  --expert-hidden-dims / --moe-top-k / --moe-noise-std /
  --moe-aux-loss-weight`.** The CLI could not reach the mixture at all, so a
  standalone run could not train the architecture the bindings could. `resolve
  info` prints the placement alongside the routing.

- **R: `resolve.train.dataset(moeRouting =, moePlacement =, nExperts =,
  expertHiddenDims =, moeTopK =, moeNoiseStd =, moeAuxLossWeight =)`**, and
  Python `resolve_core.MoEPlacement` with `ModelConfig.moe_placement`.

### Fixed

- **A malformed target specification was accepted in silence and failed much
  later as an autograd error.** In R, `targets = list(list(column = "area",
  task = "regression"))` -- an UNNAMED list -- built a dataset carrying ZERO
  targets and reported no problem. Training it then died at `loss.backward()`
  with `element 0 of tensors does not require grad and does not have a
  grad_fn`, a message that names nothing about targets and sends the reader
  into autograd rather than to what they typed.

  A target's name is the key the engine reads (and, absent `column`, the name
  of its column), so an unnamed list carries no targets at all. Three layers
  each stop discarding what they cannot read:

  - `Trainer::prepare_data` raises when the data carries no targets. Both
    overloads pass through one body, so this states the invariant once and it
    holds for the CLI, Python and R alike. Inference is untouched: a
    target-less dataset is exactly what an inference set is, and
    `Predictor::predict` never prepares a split.
  - The C-ABI `parse_targets` raises on a keyless targets tree instead of
    returning an empty vector, and both it and `parse_roles` now reject an
    unknown key, naming it and listing the accepted spellings -- so `name` /
    `type` written for `column` / `task` is reported rather than ignored. An
    empty target name and a non-map specification are rejected too.
  - `r_list_to_value_map` on the R client raises when a NON-EMPTY list produces
    no keys, naming the argument. It previously normalized every non-map value
    to an empty map to handle the zero-length case, which silently discarded
    populated unnamed lists along with it; only NULL and a zero-length list
    normalize now.

  On the R side `resolve.dataset.csv()` / `resolve.dataset.frame()` also
  validate at the front door, where the caller can still see what they wrote:
  every target must be named and uniquely named, and an unrecognized key in a
  target specification or in `roles` is rejected with the accepted set in the
  message.

- **`ResolveModel::get_gate_probs` reported nothing for every non-hash model.**
  It returned an undefined tensor unless the hash-mode MoE encoder was built,
  which was indistinguishable from "MoE is off" even when a mixture was
  actively routing. It now returns the real probabilities for the encoders its
  three-argument signature can drive (hash, TraitNet, the adapter
  architectures) at either placement, and raises for embed / sparse /
  rank_pool / transformer -- whose species inputs that signature does not
  carry -- pointing at `forward_with_aux`, which does.

### Changed

- **Retrain a mixture-of-experts checkpoint on a non-hash encoding.** Under the
  `tail` default an embed / sparse / rank_pool / transformer model with
  `moe_routing` set now puts the experts in the encoder tail (`encoder.backbone`
  + `encoder.moe`) where it previously appended a block to the latent
  (`post_moe`). Set `moe_placement = post` to keep the old architecture.
  **Hash-mode MoE checkpoints are unaffected** -- `tail` is exactly what they
  already were, down to the parameter names -- and **a checkpoint with
  `moe_routing = none`, which is every checkpoint anyone has trained through
  the paper pipeline, is untouched.**

- **`resolve info`'s Model Configuration block is registry-driven**, like the
  Data Encoding and Training Configuration blocks below it. Its rows now carry
  the field name under a `Model` label (`Model species_encoding: rank_pool`)
  rather than a curated one, which is what makes a hyperparameter added to
  `ModelConfig` appear in `info` in the edit that adds it -- `moe_placement`
  needed a line written by hand. A row is hidden only when another field
  switches its feature off: the six architecture sub-configs this checkpoint
  did not select, the mixture's hyperparameters when `moe_routing` is `none`
  (which now prints as `none` rather than going silent), TabM and the parallel
  branches when disabled, and the head's shape when it has no hidden layers.

### Tests

- `src/core/tests/test_moe_placement.cpp` (17 cases): a tail mixture builds,
  runs and trains on all five encodings; the tail's parameters are named
  `backbone` + `moe` and a plain run still writes `mlp`; the backbone/mixture
  split follows `hidden_dims`; the load-balancing loss reaches the optimizer
  (weighting it moves the trained gate); post preserves the latent width and
  covers the tail-less encoders; both refusals; gate probabilities; and
  checkpoint round-trips at either placement, including a parameter-by-parameter
  equality check on reload. `tests/core/test_moe.py` (27) covers the Python
  surface, `r/tests/testthat/` (33) the R one, and `tests.yml` gains a CLI
  end-to-end step asserting the flags change the model and the refusal names
  the placement that works.

- `src/core/tests/test_info_report.cpp` (7 cases): every `ModelConfig` field
  reaches the report unless a rule hides it, every name those rules key on is
  still a registry row, only the selected architecture's sub-config prints, a
  nested row is not gated by the outer struct's rules (both `ModelConfig` and
  `FTTransformerConfig` carry `n_heads`), and each switch -- the mixture, TabM,
  the parallel branches, the head -- governs its own rows.

## v0.8.2 (2026-08-31)

Two fixes that share a root: a guard that was compile-time where it needed to
be run-time, and a seed that covered less than the tests assumed.

### Fixed

- **A CPU run no longer touches the CUDA runtime (#114).** `Trainer::train_epoch`
  acquired its two CUDA streams before checking whether the run was on CUDA at
  all. `RESOLVE_HAS_CUDA` is a COMPILE-time guard, so a CUDA-enabled build ran
  those lines on a `device="cpu"` run too, initializing the CUDA runtime and
  killing the first epoch on a host with no usable driver -- a CPU queue node, a
  CI runner, a laptop. The streams are now held in `std::optional` and acquired
  inside the branch that already gated every USE of them, so the GPU prefetch
  path is unchanged and the CPU path is driver-free. The decision is the pure
  `resolve::use_hash_prefetch()` (`gpu.hpp`), unit-tested without a CUDA device
  the way `decide_oom_retry` is.

- **A seeded run now means what the tests assumed (#115).** The seed passed to
  `prepare_data`, `cross_validate` and `cross_validate_spatial` governs the
  SPLIT. Model weight initialisation draws from the process-global torch RNG,
  as any PyTorch module does, and nothing on the library path seeds it -- so two
  runs with the same seed started from different weights. Measured: six seeded
  `cross_validate_spatial` runs give six different fold losses at one thread as
  at twenty-four, while seeding the global RNG first makes five bit-identical.
  Two test suites were asserting the reproducibility this does not provide and
  getting it only by luck, which is why CI went red at random on unchanged
  code.

  The engine is deliberately unchanged: seeding a global stream from inside
  `fit()` is the side effect issue #107 avoided for pretraining, and weight
  init following the global RNG is the ordinary PyTorch contract. Instead the
  contract is documented (`docs/api/trainer.md` shows the two-seed form) and
  pinned from both sides, and the 17 parameter-recovery cases in
  `test_recovery.cpp` now fix their starting weights, so a correlation
  threshold is no longer evaluated on a fresh random draw each run.

**No library behaviour changed by #115** -- it is a test and documentation fix.
If you rely on reproducible fits, seed `torch.manual_seed()` before
constructing the model; the CLI's `--seed` already does.

## v0.8.1 (2026-08-30)

Three reported defects, all of the same shape: a value the API accepts and
persists, and then does not act on.

### Fixed

- **`SelectionMode` is honoured outside the hash encoding (#113).**
  `apply_selection` was called only inside the hash branch of the loader, so a
  `rank_pool` / `transformer` / `sparse` dataset recorded the selection it was
  given on its schema and encoded every species anyway, and the embed branch
  hardcoded `Top` whatever it was asked for. Each encoding now takes its
  per-plot species budget from the knob that also fixes its width: `top_k` for
  hash, `top_k_species` for embed, and the new **`DatasetConfig.species_budget`**
  for the variable-width encodings. The new knob defaults to `0` -- no budget --
  so every existing configuration encodes exactly what it encoded before;
  setting it makes a top-versus-bottom species ablation reachable on the pooled
  encoders for the first time. The species vocabulary is still fitted over every
  record, so the arms of an ablation share one integer-code namespace and stay
  comparable. The schema now records the selection the run APPLIED, which is
  `All` for a pooled or sparse load with no budget, so a checkpoint can no
  longer report a selection that never happened. Threaded through the schema,
  the checkpoint (`schema_species_budget`, absent on older checkpoints and read
  as `0`), `dataset_config_from_checkpoint`, the C ABI, nanobind, R
  (`config = list(species_budget = ...)`) and the CLI (`--species-budget N`).

- **`Predictor.load(device="cpu")` works on a machine with no CUDA device
  (#112).** `Trainer::load` called `InputArchive::load_from(path)` without the
  requested device, so the unpickler restored every tensor to the device the
  checkpoint was SAVED on and the `model->to(device)` that follows never got the
  chance -- reading a GPU-trained checkpoint on a GPU-less node threw "No CUDA
  GPUs are available" from inside deserialization. The device is now passed to
  the unpickler, in `Trainer::load`, `Trainer::load_state`, and (forced to CPU,
  since they return only scalars) `load_train_config` / `load_run_metadata`.

- **An optional role can be cleared (#111).** `roles.latitude = None` raised
  `TypeError` on the Python bindings: `def_rw` on a `std::optional` member gave
  a getter that read back `None` and a setter that refused it, because a
  nanobind function with no argument annotations takes a fast dispatch path that
  rejects every `None` argument before any caster runs. The five optional role
  columns are now bound with an explicit `str | None` setter. The empty string
  also means unset engine-wide (`RoleMapping::as_column`), so the sentinel
  downstream code already uses keeps working instead of failing with `column not
  found: ""`. A non-empty column name the file does not carry is still the loud
  configuration error it has been since #94.

- **A checkpoint saved before `fit()` recorded `train_batch_size` as 0.**
  `Trainer`'s requested-batch-size tracker started at `0`, which
  `save_train_config` reads as a genuine request, rather than at the `-1` that
  means "no separate request known" and persists the configured size.

### Added

- `resolve info` prints a **Data Encoding** block: the loading-side
  `DatasetConfig` the checkpoint implies, which is the one `resolve predict`
  rebuilds. Driven by the shared field registry, like the Training Configuration
  block beside it.
- `resolve_core.effective_selection(config)` reports the selection a dataset
  built under a config will actually apply.

## v0.8.0 (2026-08-07)

A sweep of issues #102-#110. The engine is the only implementation, the CLI
covers what the bindings cover, and four knobs that were persisted but wired to
nothing now do what they say.

### Breaking

- **The Python POC is gone.** `src/resolve/` (55 files) is deleted; `import
  resolve` no longer resolves. `resolve_core` is the Python surface, and the
  root `pyproject.toml` no longer declares a package. It was kept in-tree for
  one stated reason, the unported `rank_pool` and `transformer` encoders, and
  both have been wired end to end in C++ for some time.

### Retrain before comparing numbers

Existing checkpoints all load. These three change what a loaded model predicts:

- **HeterogeneousGNN attention was single-head.** `HeterogeneousGNNConfig::n_heads`
  reached `TypedMessagePassingLayerImpl` and was dropped on the floor. It is now
  real multi-head attention over disjoint `out_features / n_heads` slices. The
  default is 4, so parameter shapes are unchanged and predictions are not.
- **TabNet checkpoints recording `use_sparsemax = false`** now genuinely run
  1.5-entmax where they previously ran sparsemax regardless.
- **`unknown_fraction` / `unknown_count` carry values when scoring.** Training
  through the plain `from_csv` path still reads 0.0 for every plot, which is the
  correct value there, so training is bit-for-bit unchanged. Scoring through the
  vocabulary-reusing loaders now feeds real values through a weight that only
  ever saw zeros.

### Correctness

- **Checkpoints carry the fitted vocabularies** (#102). A checkpoint stored only
  the *sizes* of the species and taxonomy vocabularies, so scoring new data from
  a checkpoint alone re-fitted the codes and every non-hash encoder looked up
  other species' embedding rows: wrong predictions, no error. `ResolveSchema`
  now carries the ordered species, genus and family vocabularies, and
  `Predictor` rejects a dataset whose vocabularies are not the model's rather
  than silently scoring it. A pre-0.8.0 checkpoint still loads, with a warning.
- **1.5-entmax is the published operator** (#103). `entmax15` dropped the
  `(alpha - 1)` factor of Eq. 13 in Peters, Niculae and Martins (ACL 2019), so
  it ran at a different temperature and collapsed onto sparsemax's support. It
  is now that paper's exact sort-based Algorithm 2, with the closed-form
  Proposition 1 backward.
- **`LossConfigMode::NCA` trained something else.** `PhasedLoss::from_config`
  had no NCA case and fell through to `Combined`, while `NCALossImpl` had zero
  call sites. The preset is live, and its three hyperparameters are now
  `TrainConfig` fields instead of unreachable constants. R's
  `resolve.train.dataset()` also rejected `lossConfig = "nca"` outright.
- **The effective batch size was unreadable after `fit()`** (#105). The OOM
  auto-halve report compared a value `fit()` restores before returning, so it
  was unreachable, and a `save()` after `fit()` recorded the requested batch size
  as the effective one.
- **Uninitialized read in the GNN adapter.** An out-of-range `GNNType` from a
  newer checkpoint or the C ABI read an uninitialized enum.
- **The CLI silently dropped every covariate** (#104). `resolve train` read
  `--header` but nothing populated `RoleMapping::covariates` or `categoricals`,
  so a CLI-trained model was structurally different from the same configuration
  trained through `resolve_core` or R.

### Reproducibility

- **Pretraining is seeded** (#107). `PretrainConfig`, `MLMPretrainConfig` and
  `VAEConfig` gain a `seed`, and every shuffle, mask, corruption and
  reparameterization draw goes through one `PretrainRng` seam. A pretraining run
  no longer advances the global RNG stream, so it cannot shift the dropout draws
  of the finetuning that follows. Module dropout is the exception and is
  documented as such: `torch::nn::Dropout` takes no generator.
- **`resolve train --seed N`** seeds weight initialization, the split and the
  cross-validation folds. Two identical invocations previously produced
  different models.

### CLI

- `--covariate` and `--categorical` (repeatable) on `train` and `predict`,
  `--seed`, cross-validation (`--cv-folds`, `--cv-spatial`, ...), and roughly
  thirty `TrainConfig` / `ModelConfig` / `DatasetConfig` flags the bindings
  already exposed.
- A declarative flag table per subcommand generates the usage text and **rejects
  unknown flags**, naming the near miss. `--maxepochs 10` was previously ignored
  and the default used, with no diagnostic.
- `resolve info` prints every architecture sub-config and the training
  configuration.
- `resolve predict` writes a classification target as the original label plus a
  `<target>_code` column.

### Maintainability

- **One field registry per config struct** (#108). An X-macro list gives each
  field its name and checkpoint key exactly once, and the checkpoint reader and
  writer, the C ABI value tree, the nanobind bindings, the JSON sidecar and
  `resolve info` are all visitors over it. A member added without a registry row
  fails a `static_assert`. Every archive key spelling is unchanged.
- **Compiler warnings are on** (#109). `-Wall -Wextra -Wshadow
  -Wnon-virtual-dtor` on GCC/Clang and `/W4 /permissive-` on MSVC, applied to
  the engine, the C ABI, the CLI and the tests but not to vendored
  dependencies. 619 MSVC and 56 GCC warnings fixed; `-Werror` is armed on the
  Linux CI job.
- Four pretraining loops that each carried their own copy of the epoch scaffold
  now share one, so a fifth pretext task is one loss function.

### Testing and CI

- The Catch2 suite goes from 281 to 375 cases (2999 to 4563 assertions).
- New `tests/core/`, a pytest suite over `resolve_core` including
  parameter-recovery cases that fit to convergence and assert held-out
  correlation and accuracy. The 8-job `python-tests` matrix it replaces was
  installing and exercising the deleted POC, and the production Python surface
  had no automated test at all.
- A CLI end-to-end job trains, inspects and predicts over a committed fixture,
  asserting covariates reach the model, that `--seed` reproduces, and that every
  rejection path exits non-zero. CI previously ran `resolve version` and
  `resolve help`.
- A mechanical check that every public nanobind name is re-exported from
  `resolve_core`. Twelve types and the `fuzzy` submodule were reachable only
  through the private module.

### Removed

- `EncodedSpecies`, a struct with no producer and no caller.
- Documentation claims that the build fetches CLI11 and fast-cpp-csv-parser.
  Neither is fetched anywhere; the argument parser and the CSV reader are both
  hand-rolled.

## v0.7.3 (2026-08-04)

### R package

- **GPU training from R.** `resolve.install_backend()` gains CUDA variants:
  `variant = "cu128"`, `"cu130"`, or `"cuda"` (auto-selects the line from the
  installed NVIDIA driver -- `cu130` for CUDA >= 13, else `cu128`). The CUDA
  builds ship only the small `resolve_c` library on the GitHub release and fetch
  the matching official libtorch from `download.pytorch.org` on first install,
  pinned to the exact version `resolve_c` was built against so the ABI matches;
  GPU training then runs through the ordinary `device = "cuda"` path. A
  backend-variant registry (`{os, arch, variant} -> {asset, libtorch}`) drives
  the downloader, so adding a CUDA line later is one table row.
- **GPU nudge.** On attach, if an NVIDIA GPU is detected but the CPU backend is
  loaded (or none is), the package points to the GPU build.
- The backend is loaded at runtime (`dlopen`/`LoadLibrary`) rather than linked,
  so the package installs and `R CMD check`s with no backend present; libtorch
  threads default to all cores (`RESOLVE_R_TORCH_THREADS=N` to pin/cap).

## v0.7.2 (2026-07-19)

A review sweep of the whole engine, issues #37-#100. Highlights below; each
commit carries the per-issue detail.

### Correctness

- **Checkpoints round-trip the full architecture.** `save_model_config` /
  `load_model_config` serialize every architecture sub-config (FT-Transformer,
  TabNet, SAINT, GNN, TraitNet, ExcelFormer, heterogeneous GNN, parallel
  branches), and weight loading now throws on a missing parameter instead of
  silently leaving it at random init (#37). Classification `class_weights` and
  the rank-pool weighting scheme + species cap are persisted too, so a reloaded
  model keeps its loss and its pooling semantics (#38, #91).
- **Gradients reach the objectives they belong to.** The phase-3 band penalty is
  a differentiable hinge, ExcelFormer's semi-permeable mask is a soft gate with
  an additive log-bias, and BERT-style MLM feeds the 10%-random / 10%-keep ids
  through to the encoder (#42, #43, #44).
- **Self-supervised views hide the answer.** JEPA and SCARF mask the species and
  taxonomy side of each view, so the pretext task cannot be solved by species
  identity alone (#44, #93).
- **Cross-validation starts each fold from the untrained weights** and restores
  the trainer's split afterwards, so CV after `fit()` no longer warm-starts from
  weights that saw the held-out rows (#45, #97).
- **Loaders fail loudly.** A named role column that cannot be resolved throws
  rather than dropping the feature; coordinates parse NA-aware; ranking is dense;
  `num_classes` follows the class list (#40, #46, #47, #94).
- **Determinism knobs are honored.** `cudnn_benchmark = false` survives the
  training loop instead of being re-enabled inside `cache_data_to_gpu` (#92).

### Reported metrics

- **SMAPE** uses the standard `(|p|+|t|)/2` denominator (range 0-2, matches
  sklearn); values previously came out at half scale, and the phased-loss SMAPE
  term shifts by the same constant factor (#95).
- **The VAE ELBO** sums KL over the latent dimension and means over the batch, so
  `kl_weight = 1` is beta = 1 (#96).

### Retraining required

- Taxonomy embedding tables for the hash / sparse / MoE / adapter encoders lose
  the one over-allocated row (`n_genera` already counts `<UNK>`), and the
  coordinate-kNN GNN embeds taxonomy ids instead of concatenating them as
  magnitudes and trains full-batch. Checkpoints from before these changes cannot
  be loaded (#73, #99).

### Tooling

- CI gains a vendored-header drift guard for the R C facade; `resolve info`
  prints the transformer / rank-pool hyperparameters; pretraining configs
  validate their batch size, mask ratio and corruption rate; the header loader
  reads the file in a single streaming pass instead of a count prepass (#100).

## v0.7.1 (2026-06-19)

### Packaging

- **PyPI wheel build repaired across all platforms.** The `resolve-core` wheel
  pipeline (broken since 0.6.x) now builds cleanly on Linux, Windows, and both
  macOS architectures. Linux/Windows install `cmake` and `ninja` explicitly
  because the no-isolation build asks scikit-build-core for `ninja>=1.5`, which
  the manylinux container does not ship; the pip self-upgrade uses `python -m
  pip` so the Windows step no longer aborts. The macOS extension links with
  `-Wl,-undefined,dynamic_lookup` so the Python C-API symbols pulled in via
  `libtorch_python` (absent from nanobind's restricted macOS symbol list)
  resolve from the host interpreter at load time instead of failing the arm64
  link.

## v0.7.0 (2026-06-19)

### New features

- **In-memory (DataFrame) dataset loaders (#22).** Build a dataset directly from
  frames already in RAM, eliminating the write-to-temp-CSV / re-read round-trip
  the CSV loaders force when the header is filtered or subset per fit. Python:
  `ResolveDataset.from_pandas(header, species=, roles, targets, config=,
  schema_source=)` (alias `.from_dataframe`), where `species` may be a
  DataFrame, a CSV path (the large species table is read once from disk while
  the header stays in memory), or `None` (single long frame). R:
  `resolve.dataset.frame(header, species=, ...)`. C++: `from_dataframe` /
  `from_dataframe_header` / `from_species_dataframe` / `from_dataframe_with_schema`.
  A shared `RowSource` seam (implemented by both `CSVReader` and an in-memory
  `ColumnTable`) makes `from_dataframe` byte-identical to `from_csv` on the
  equivalent CSV by construction (an empty cell is a missing value). The
  previously-missing `categorical_ids` R accessor was also registered.

### Bug fixes

- **AMP fp32-normalization guard (#21).** `run_norm_fp32` / `Fp32Norm` force
  every normalization layer to compute in fp32 inside a CUDA autocast region
  while the surrounding Linear/embedding matmuls stay fp16, guarding against
  fp16 BatchNorm-statistic corruption/overflow (a running variance saturating to
  `inf` collapses eval-mode normalization to mean-prediction). On the current
  libtorch build autocast already promotes `batch_norm` to fp32, so the guard is
  defensive there; it removes the dependency on that implicit, version-dependent
  autocast policy. Toggle with `RESOLVE_FP32_NORM=0`; diagnose with
  `RESOLVE_AMP_DEBUG=1`.

## v0.6.2 (2026-06-14)

### Bug fixes

- **Bounded retry on transient storage I/O (#20).** The engine now retries its
  explicit, idempotent file I/O on a transient storage fault instead of aborting
  the run, the complement of #19. `resolve::io::with_retry` (header-only) backs a
  new `io::IOError` thrown by the CSV reader on a failed open or a mid-read
  stream error. The dataset loaders (`from_csv` / `from_csv_with_schema` /
  `from_species_csv`) restart the whole load into a fresh dataset on a transient
  read, while a CSV *parse* error propagates immediately and never re-reads a
  multi-GB file; checkpoint save/load (`Trainer::save` / `load` / `load_state`
  and `Predictor::load`) retry the archive read/write. Tunable via
  `RESOLVE_IO_RETRY_ATTEMPTS` (3) and `RESOLVE_IO_RETRY_BACKOFF_MS` (100).
  mmap-backed page-ins and DLL code-page faults remain out of scope (they cannot
  be resumed at app level; that is #19's fail-fast domain).

## v0.6.1 (2026-06-14)

### Bug fixes

- **Windows process-crash hardening (#18, #19).** A native fault in a headless
  training worker no longer hangs forever on the Windows JIT debugger
  (`vsjitdebugger`): the engine installs an unhandled-exception filter plus a
  first-in-line vectored handler that terminate via `TerminateProcess`, so the
  worker fails fast with the fault's exit code and the orchestrator can record
  and skip it instead of waiting on the AeDebug handshake. The R bindings arm an
  on-exit finalizer alongside the crash handler, mitigating the libtorch teardown
  access violation that could crash the `Rscript.exe` launcher. libtorch's thread
  pools are left at their multi-threaded default so training and prediction use
  all cores; set `RESOLVE_R_TORCH_THREADS=N` (a positive integer) to pin both
  pools to N threads -- to cap CPU use on a shared machine, or as a workaround
  (`N=1`) if a Windows environment still hits the teardown crash.

### Internal

- New `resolve::process` engine module (`process.{hpp,cpp}`) with the
  `install_crash_handler` / `signal_work_complete` / `set_thread_pools` surface,
  wired through the C ABI facade, nanobind (`resolve_core.install_crash_handler`,
  `set_thread_pools`; armed at import with an `atexit` hook), Rcpp + `zzz.R`
  `.onLoad`, and the CLI. Catch2 `test_process.cpp` plus a crash-handler smoke.
- `tests/test_cuda_allocator_config.py` skips cleanly when the compiled
  `resolve_core` extension is not built.

## v0.5.0 (2026-05-18)

### New Features

- **Native FuzzyIndex backbone for `WFOBackbone`**: When `resolve_core` is
  installed, `WFOBackbone` now builds a C++ `FuzzyIndex` (Damerau-Levenshtein,
  genus-bucketed, case-insensitive) over the WFO names at construction time
  and routes `_match_fuzzy` through it. The stdlib `difflib` path remains the
  silent fallback when the native backend is unavailable. Reported
  `fuzzy_dist` is an integer edit distance on the native path; the legacy
  `1 - SequenceMatcher.ratio()` semantic is preserved on the difflib path.
- **Auto-categorical encoding in `from_fast_csv`**: Classification target columns
  whose values are non-numeric (e.g. EUNIS letters `M..V`) are now loaded as
  strings and automatically encoded to nullable `Int64` codes. Integer-string
  values (e.g. `"0".."8"`) are preserved verbatim; non-numeric values are
  factorized in sorted order. `num_classes` is auto-filled from the resulting
  mapping size, so it can be omitted from the target config.
- **`categorical_covariates` kwarg on `from_fast_csv`**: Pass a `{column: mapping}`
  dict to encode covariates with non-numeric values (e.g. `{"ReSurvey (Y/N)":
  {"Y": 1, "N": 0}}`). Use `None` for the mapping to auto-encode by sorted unique
  value. The encoded mappings are accessible via the new `dataset.categorical_mappings`
  property.

### Internal

- `Trainer.predict()` now batches the forward pass via `_batched_forward`,
  removing the OOM on large held-out sets for rank-pool / hash / embed modes.
- "Training complete" is a first-class checkpoint state: `save_checkpoint`
  takes a `completed: bool` kwarg; resumes that find a completed checkpoint
  fast-return instead of raising `UnboundLocalError` on an empty epoch range.
- `_pretrain.py` rebuilt for the pre-padded tuple layout produced by
  `_build_tensors` (the v3 cache refactor). Adds `MaskedSpeciesCollateWrapper`
  for pre-padded batches and fixes a latent categoricals slot off-by-one.
- Dead code removed: `_RankPoolPreparedData` / `RankPoolBatchDataset` /
  `_rank_pool_collate_fn`, deprecated `track_unknown_count` kwarg, the
  `ext/wfo.py` rapidfuzz fallback (single algorithm: native FuzzyIndex when
  available, else difflib).
- `Trainer._best_state`, `_ema_state`, `_using_gpu_loader` initialized in
  `__init__`; defensive `hasattr`/`getattr` at the seven call sites removed.
- `_cv.py` block_size deprecation now uses `warnings.warn(DeprecationWarning)`.

## v0.4.0 (2025-01-25)

### New Features

- **R² metric**: Coefficient of determination for regression evaluation (computed on original scale)
- **Class weights**: Support for imbalanced classification via `class_weights` in target config
- **LR scheduling**: StepLR and CosineAnnealing scheduler options

### Testing & CI

- Comprehensive test suite: Catch2 (C++), pytest (Python), testthat (R)
- GitHub Actions workflows for automated testing and releases

### Packaging

- Python: `resolve-core` (C++ bindings) + `resolve` (high-level wrapper)
- R: Full testthat integration, CRAN-ready structure

## v0.1.0 (2025-01-19)

Initial release of RESOLVE (Representation Encoding for Structured Observation Learning with Vector Embeddings).

### Features

- **Hybrid species encoding**: Feature hashing for full species lists + learned embeddings for dominant taxa
- **Multi-target prediction**: Single shared encoder, multiple task heads (regression and classification)
- **Phased training**: MAE -> SMAPE -> band accuracy optimization for regression targets
- **Semantic role mapping**: Flexible column naming with strict structural requirements
- **Unknown species tracking**: Detects and quantifies novel species at inference time
- **Abundance normalization**: Raw, relative (per-plot), or log-scaled modes
- **CPU-first design**: Works without GPU, scales with CUDA when available

### Core Components

- `ResolveDataset`: Data loading with semantic role mapping
- `ResolveModel`: Neural network architecture with shared encoder and task-specific heads
- `Trainer`: Training loop with phased optimization and early stopping
- `Predictor`: Inference interface with embedding extraction

### Architecture

- Linear compositional pooling: Species effects aggregated linearly before nonlinear mixing
- Taxonomy-aware embeddings: Learned representations for genera and families
- Feature hashing: Scalable species encoding via locality-sensitive hashing
