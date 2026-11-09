# Python reference

The Python package’s classes, functions and values, under the sections
of [the R
reference](https://gillescolling.com/resolve/reference/index.md) and in
its order. `import resolve_core as rc` reaches every name below.
[Imputing plot attributes from species
lists](https://gillescolling.com/resolve/articles/resolveR.html?lang=python)
runs them on the survey that ships with the R package.

## Installation

``` r

# install.packages("pak")
pak::pak("gcol33/resolve/r")

# The engine library, downloaded once
library(resolveR)
resolve.install_backend()
```

``` bash
pip install resolve-core
```

## [Loading data](https://gillescolling.com/resolve/articles/python-data.md)

Plots and the species records observed in them, from CSV files or data
frames, into an engine dataset.

- [`RoleMapping`](https://gillescolling.com/resolve/articles/python-data.html#RoleMapping):
  Names the columns of the two tables by the part they play in a
  dataset.

- [`TargetSpec`](https://gillescolling.com/resolve/articles/python-data.html#TargetSpec):
  One target, a plot-table column the model learns to impute, and how
  its values are read.

- [`TaskType`](https://gillescolling.com/resolve/articles/python-data.html#TaskType):
  The kind of a target: regression or classification.

- [`TransformType`](https://gillescolling.com/resolve/articles/python-data.html#TransformType):
  The transform a regression target is stored under.

- [`TargetMissingPolicy`](https://gillescolling.com/resolve/articles/python-data.html#TargetMissingPolicy):
  What a plot with no value for a target does.

- [`DatasetConfig`](https://gillescolling.com/resolve/articles/python-data.html#DatasetConfig):
  How the species recorded on a plot become model input.

- [`SpeciesEncodingMode`](https://gillescolling.com/resolve/articles/python-data.html#SpeciesEncodingMode):
  The encoding of a plot’s species set.

- [`SelectionMode`](https://gillescolling.com/resolve/articles/python-data.html#SelectionMode):
  Which species of a plot fill its budget.

- [`RepresentationMode`](https://gillescolling.com/resolve/articles/python-data.html#RepresentationMode):
  The value a species carries into the encoding: its abundance, or 1.

- [`NormalizationMode`](https://gillescolling.com/resolve/articles/python-data.html#NormalizationMode):
  How the hash weights of a plot’s selected species are rescaled.

- [`AggregationMode`](https://gillescolling.com/resolve/articles/python-data.html#AggregationMode):
  How genera and families are ranked and weighted.

- [`PoolWeighting`](https://gillescolling.com/resolve/articles/python-data.html#PoolWeighting):
  The weight a species pools with under the rank-pool and transformer
  encodings.

- [`MissingValuePolicy`](https://gillescolling.com/resolve/articles/python-data.html#MissingValuePolicy):
  How a blank covariate or coordinate cell enters the model.

- [`effective_selection()`](https://gillescolling.com/resolve/articles/python-data.html#effective_selection):
  The species selection a dataset loaded under a configuration applies.

- [`ResolveDataset`](https://gillescolling.com/resolve/articles/python-data.html#ResolveDataset):
  The loaded and encoded plots, and the loaders that build them.

- [`ResolveSchema`](https://gillescolling.com/resolve/articles/python-data.html#ResolveSchema):
  The structure of a loaded dataset: the input widths, the targets and
  the vocabularies.

- [`TargetConfig`](https://gillescolling.com/resolve/articles/python-data.html#TargetConfig):
  A target as the dataset loaded it.

- [`ExternalVocabs`](https://gillescolling.com/resolve/articles/python-data.html#ExternalVocabs):
  Every vocabulary a dataset fitted, in the form the loaders that reuse
  vocabularies take.

- [`external_vocabs_from_schema()`](https://gillescolling.com/resolve/articles/python-data.html#external_vocabs_from_schema):
  The vocabularies a schema carries.

- [`EncodingVocab`](https://gillescolling.com/resolve/articles/python-data.html#EncodingVocab):
  The species and taxonomy vocabularies one encoding codes a plot with.

- [`SpeciesVocab`](https://gillescolling.com/resolve/articles/python-data.html#SpeciesVocab):
  Each species name to an integer code, 0 for an unknown species.

- [`TaxonomyVocab`](https://gillescolling.com/resolve/articles/python-data.html#TaxonomyVocab):
  Each genus and family name to an integer code.

- [`CategoricalVocab`](https://gillescolling.com/resolve/articles/python-data.html#CategoricalVocab):
  The integer codes of each categorical column.

- [`SpeciesRecord`](https://gillescolling.com/resolve/articles/python-data.html#SpeciesRecord):
  One species recorded on one plot.

- [`EmbeddingEncoder`](https://gillescolling.com/resolve/articles/python-data.html#EmbeddingEncoder):
  The embed encoding on its own, outside a dataset.

- [`EmbeddingEncodedData`](https://gillescolling.com/resolve/articles/python-data.html#EmbeddingEncodedData):

  The species and taxonomy slots `EmbeddingEncoder` returns.

- [`RankPoolEncoder`](https://gillescolling.com/resolve/articles/python-data.html#RankPoolEncoder):
  The rank-pool encoding on its own, outside a dataset.

- [`RankPoolEncodedData`](https://gillescolling.com/resolve/articles/python-data.html#RankPoolEncodedData):

  The pooled species, weights and mask `RankPoolEncoder` returns.

- [`build_species_graph()`](https://gillescolling.com/resolve/articles/python-data.html#build_species_graph):
  The graph over the species vocabulary a heterogeneous GNN passes
  messages on.

- [`SpeciesGraph`](https://gillescolling.com/resolve/articles/python-data.html#SpeciesGraph):
  The edges of a species graph and their types.

- [`SpeciesEdgeType`](https://gillescolling.com/resolve/articles/python-data.html#SpeciesEdgeType):
  The kind of a species-graph edge: same genus, same family or
  co-occurrence.

## [Training](https://gillescolling.com/resolve/articles/python-training.md)

Fitting a model, cross-validating it and pretraining its encoder without
labels.

- [`ResolveModel`](https://gillescolling.com/resolve/articles/python-training.html#ResolveModel):
  The network: a species encoder, a shared encoder and one output head
  per target.
- [`ModelForwardResult`](https://gillescolling.com/resolve/articles/python-training.html#ModelForwardResult):
  A forward pass’s outputs with its auxiliary loss terms.
- [`model_config_for_dataset()`](https://gillescolling.com/resolve/articles/python-training.html#model_config_for_dataset):
  A model configuration with the fields that size a model taken from its
  dataset.
- [`ModelConfig`](https://gillescolling.com/resolve/articles/python-training.html#ModelConfig):
  The architecture of a model.
- [`FTTransformerConfig`](https://gillescolling.com/resolve/articles/python-training.html#FTTransformerConfig):
  The settings of the FT-Transformer architecture.
- [`TabNetConfig`](https://gillescolling.com/resolve/articles/python-training.html#TabNetConfig):
  The settings of the TabNet architecture.
- [`SAINTConfig`](https://gillescolling.com/resolve/articles/python-training.html#SAINTConfig):
  The settings of the SAINT architecture.
- [`ExcelFormerConfig`](https://gillescolling.com/resolve/articles/python-training.html#ExcelFormerConfig):
  The settings of the ExcelFormer architecture.
- [`GNNConfig`](https://gillescolling.com/resolve/articles/python-training.html#GNNConfig):
  The settings of the graph neural network over plots.
- [`HeterogeneousGNNConfig`](https://gillescolling.com/resolve/articles/python-training.html#HeterogeneousGNNConfig):
  The settings of the graph neural network over species.
- [`TraitNetConfig`](https://gillescolling.com/resolve/articles/python-training.html#TraitNetConfig):
  The settings of the TraitNet architecture.
- [`TabMConfig`](https://gillescolling.com/resolve/articles/python-training.html#TabMConfig):
  TabM as the encoder’s final stage.
- [`ParallelLayersConfig`](https://gillescolling.com/resolve/articles/python-training.html#ParallelLayersConfig):
  A block of parallel branches as the encoder’s final stage.
- [`ParallelBranchConfig`](https://gillescolling.com/resolve/articles/python-training.html#ParallelBranchConfig):
  One branch of a parallel block.
- [`Trainer`](https://gillescolling.com/resolve/articles/python-training.html#Trainer):
  Fits a model: the split, the training loop, cross-validation, the
  held-out evaluations and checkpoints.
- [`TrainConfig`](https://gillescolling.com/resolve/articles/python-training.html#TrainConfig):
  The settings of a training run.
- [`TrainResult`](https://gillescolling.com/resolve/articles/python-training.html#TrainResult):
  The metrics, loss histories and baselines a fit returns.
- [`BaselineMetrics`](https://gillescolling.com/resolve/articles/python-training.html#BaselineMetrics):
  A target’s naive baseline beside the model.
- [`CrossValidationResult`](https://gillescolling.com/resolve/articles/python-training.html#CrossValidationResult):
  The metrics of a cross-validation, per fold and across folds.
- [`SpatialBlockConfig`](https://gillescolling.com/resolve/articles/python-training.html#SpatialBlockConfig):
  The latitude and longitude blocks of a spatial cross-validation.
- [`ResidualAnalysis`](https://gillescolling.com/resolve/articles/python-training.html#ResidualAnalysis):
  The residuals of a regression target on the held-out fold.
- [`ClassificationPredictions`](https://gillescolling.com/resolve/articles/python-training.html#ClassificationPredictions):
  The imputed classes and probabilities of a classification target on
  the held-out fold.
- [`CalibrationResult`](https://gillescolling.com/resolve/articles/python-training.html#CalibrationResult):
  How well a classifier’s probabilities match how often it is right.
- [`CalibrationBin`](https://gillescolling.com/resolve/articles/python-training.html#CalibrationBin):
  One bin of the probability given to the imputed class.
- [`NetworkDiagnostics`](https://gillescolling.com/resolve/articles/python-training.html#NetworkDiagnostics):
  The activations of the species encoder’s hidden layers.
- [`LayerDiagnostics`](https://gillescolling.com/resolve/articles/python-training.html#LayerDiagnostics):
  The activations of one hidden layer.
- [`PretrainConfig`](https://gillescolling.com/resolve/articles/python-training.html#PretrainConfig):
  The settings of JEPA and SCARF pretraining.
- [`JEPAPretrainer`](https://gillescolling.com/resolve/articles/python-training.html#JEPAPretrainer):
  Pretrains an encoder by predicting the latent of a plot from a masked
  view of it.
- [`SCARFPretrainer`](https://gillescolling.com/resolve/articles/python-training.html#SCARFPretrainer):
  Pretrains an encoder by matching each plot with a corrupted copy of
  it.
- [`PretrainResult`](https://gillescolling.com/resolve/articles/python-training.html#PretrainResult):
  The loss history of a pretraining run.
- [`MaskStrategy`](https://gillescolling.com/resolve/articles/python-training.html#MaskStrategy):
  Which features JEPA masks.
- [`MLMPretrainConfig`](https://gillescolling.com/resolve/articles/python-training.html#MLMPretrainConfig):
  The settings of masked-species pretraining.
- [`MaskedSpeciesPretrainer`](https://gillescolling.com/resolve/articles/python-training.html#MaskedSpeciesPretrainer):
  Pretrains a transformer species encoder by naming the species hidden
  from it.
- [`VAEConfig`](https://gillescolling.com/resolve/articles/python-training.html#VAEConfig):
  The settings of the variational autoencoder.
- [`VAEPretrainer`](https://gillescolling.com/resolve/articles/python-training.html#VAEPretrainer):
  A variational autoencoder of the plots’ species vectors.
- [`VAEPretrainResult`](https://gillescolling.com/resolve/articles/python-training.html#VAEPretrainResult):
  The loss histories of a variational autoencoder run.
- [`LossConfigMode`](https://gillescolling.com/resolve/articles/python-training.html#LossConfigMode):
  The regression loss.
- [`LRSchedulerType`](https://gillescolling.com/resolve/articles/python-training.html#LRSchedulerType):
  How the learning rate changes over a run.
- [`SelectionMetric`](https://gillescolling.com/resolve/articles/python-training.html#SelectionMetric):
  The held-out value a run selects its epoch by.
- [`ActivationType`](https://gillescolling.com/resolve/articles/python-training.html#ActivationType):
  The activation functions.
- [`NormLayerType`](https://gillescolling.com/resolve/articles/python-training.html#NormLayerType):
  The normalisation layers.
- [`EncoderArchitecture`](https://gillescolling.com/resolve/articles/python-training.html#EncoderArchitecture):
  The architecture of the shared encoder.
- [`MoERoutingType`](https://gillescolling.com/resolve/articles/python-training.html#MoERoutingType):
  How a mixture of experts routes a plot.
- [`MoEPlacement`](https://gillescolling.com/resolve/articles/python-training.html#MoEPlacement):
  Where the mixture of experts sits in the encoder.
- [`GNNType`](https://gillescolling.com/resolve/articles/python-training.html#GNNType):
  The graph layer of the graph neural network over plots.
- [`GraphConstructionMode`](https://gillescolling.com/resolve/articles/python-training.html#GraphConstructionMode):
  What the plot graph measures neighbourhood on.
- [`TraitInteractionMode`](https://gillescolling.com/resolve/articles/python-training.html#TraitInteractionMode):
  How TraitNet combines the environment and a species’ traits.
- [`TabMAggregation`](https://gillescolling.com/resolve/articles/python-training.html#TabMAggregation):
  How TabM combines its ensemble.
- [`TransformerPooling`](https://gillescolling.com/resolve/articles/python-training.html#TransformerPooling):
  How the transformer encoding pools its species tokens.
- [`ParallelAggregation`](https://gillescolling.com/resolve/articles/python-training.html#ParallelAggregation):
  How a parallel block combines its branches.

## [Saving and imputing](https://gillescolling.com/resolve/articles/python-prediction.md)

Checkpoints, what they record about the run, and imputed values for new
plots.

- [`Predictor`](https://gillescolling.com/resolve/articles/python-prediction.html#Predictor):
  Imputed values, latent representations and embedding tables from a
  trained model.
- [`ResolvePredictions`](https://gillescolling.com/resolve/articles/python-prediction.html#ResolvePredictions):
  The imputed values and class probabilities for every plot.
- [`RetrievalConfig`](https://gillescolling.com/resolve/articles/python-prediction.html#RetrievalConfig):
  The settings of retrieval augmentation.
- [`nearest_by_cosine()`](https://gillescolling.com/resolve/articles/python-prediction.html#nearest_by_cosine):
  The rows of a reference set most similar to each query row by cosine
  similarity.
- [`Scalers`](https://gillescolling.com/resolve/articles/python-prediction.html#Scalers):
  The input and target scaling fitted on the training plots.
- [`RunMetadata`](https://gillescolling.com/resolve/articles/python-prediction.html#RunMetadata):
  The run as a checkpoint records it: split, timing, epochs and final
  metrics.
- [`HoldOutMode`](https://gillescolling.com/resolve/articles/python-prediction.html#HoldOutMode):
  How a run’s held-out fold was drawn.
- [`dataset_config_from_checkpoint()`](https://gillescolling.com/resolve/articles/python-prediction.html#dataset_config_from_checkpoint):
  The loader settings a checkpoint was trained under.
- [`target_specs_from_schema()`](https://gillescolling.com/resolve/articles/python-prediction.html#target_specs_from_schema):
  The targets a checkpoint was trained on.

## [Model suites](https://gillescolling.com/resolve/articles/python-suites.md)

Several trained models released as one directory and scored together.

- [`SuiteManifest`](https://gillescolling.com/resolve/articles/python-suites.html#SuiteManifest):
  A suite’s manifest: the input contract, the targets, their members and
  the provenance.
- [`SuiteInputContract`](https://gillescolling.com/resolve/articles/python-suites.html#SuiteInputContract):
  The columns a suite reads and how their values are read.
- [`SuiteTarget`](https://gillescolling.com/resolve/articles/python-suites.html#SuiteTarget):
  One released target of a suite.
- [`SuiteTargetInputs`](https://gillescolling.com/resolve/articles/python-suites.html#SuiteTargetInputs):
  The part of the input contract a target’s members read.
- [`SuiteMember`](https://gillescolling.com/resolve/articles/python-suites.html#SuiteMember):
  One checkpoint of a suite target.
- [`SuiteCombine`](https://gillescolling.com/resolve/articles/python-suites.html#SuiteCombine):
  How a suite target combines its members.
- [`SuiteTargetStatus`](https://gillescolling.com/resolve/articles/python-suites.html#SuiteTargetStatus):
  Released or experimental.
- [`SuitePredictor`](https://gillescolling.com/resolve/articles/python-suites.html#SuitePredictor):
  A suite loaded from its directory, scored as one model.
- [`SuiteColumns`](https://gillescolling.com/resolve/articles/python-suites.html#SuiteColumns):
  Renames the manifest’s input columns at imputation time.
- [`SuitePredictions`](https://gillescolling.com/resolve/articles/python-suites.html#SuitePredictions):
  Every target’s imputed values, and the table they are written as.
- [`SuiteTargetPrediction`](https://gillescolling.com/resolve/articles/python-suites.html#SuiteTargetPrediction):
  One target’s combined imputed value, the members’ agreement or spread,
  and the species recognition.
- [`suite_prediction_columns()`](https://gillescolling.com/resolve/articles/python-suites.html#suite_prediction_columns):
  The imputed values as named columns, one value per plot.
- [`CombinedPrediction`](https://gillescolling.com/resolve/articles/python-suites.html#CombinedPrediction):
  A combined imputed value with its agreement or dispersion.
- [`combine_vote()`](https://gillescolling.com/resolve/articles/python-suites.html#combine_vote):
  The class most members impute.
- [`combine_mean()`](https://gillescolling.com/resolve/articles/python-suites.html#combine_mean):
  The members’ mean and standard deviation.
- [`combine_circular()`](https://gillescolling.com/resolve/articles/python-suites.html#combine_circular):
  The members’ circular mean and circular standard deviation.
- [`combine_spherical()`](https://gillescolling.com/resolve/articles/python-suites.html#combine_spherical):
  The members’ mean point on the sphere and their spread in km.
- [`bearing_from_components()`](https://gillescolling.com/resolve/articles/python-suites.html#bearing_from_components):
  A bearing from its sine and cosine.
- [`EARTH_RADIUS_KM`](https://gillescolling.com/resolve/articles/python-suites.html#EARTH_RADIUS_KM):
  The Earth radius the spherical mean measures distances with.
- [`sha256_file()`](https://gillescolling.com/resolve/articles/python-suites.html#sha256_file):
  The SHA-256 digest of a file.

## [Gap fitting](https://gillescolling.com/resolve/articles/python-gap-fitting.md)

Filling a dataset’s missing values from its own complete plots, beside
the raw imputed value.

- [`fit_gaps()`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#fit_gaps):
  One target’s gaps from the raw imputed value, the recorded values and
  the plots’ composition, positions and years.
- [`fit_prediction_gaps()`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#fit_prediction_gaps):
  A single model’s gaps from its imputed values and the dataset’s
  tables.
- [`read_gap_tables()`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#read_gap_tables):
  What gap fitting reads about a dataset’s plots, from DataFrames or CSV
  paths.
- [`GapFittingConfig`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#GapFittingConfig):
  The candidates, the neighbour counts, spatial neighbours on or off and
  the matching seed.
- [`GapTarget`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#GapTarget):
  A target’s name, scale and period.
- [`GapScale`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#GapScale):
  How a target’s values lie: linear, log, circular or geographic.
- [`GapMethod`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#GapMethod):
  The methods a gap can be imputed by.
- [`GapFit`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#GapFit):
  The fitted values, the method per plot and the report.
- [`matched_gap_neighbours()`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#matched_gap_neighbours):
  The neighbours the choice is tested with.
- [`GapColumns`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#GapColumns),
  [`GapTables`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#GapTables),
  [`GapFitRequest`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#GapFitRequest)
  : The low-level columns, tables and requests the functions above
  build.

## [Terrain correction](https://gillescolling.com/resolve/articles/python-terrain-correction.md)

Altitude and slope combined with an elevation model read at the plots,
and the elevation model a suite ships.

- [`TerrainGrids`, `open_terrain_field()`, `TerrainCodedField`,
  `TerrainField`](https://gillescolling.com/resolve/articles/python-terrain-correction.html#sources):
  Elevation grids on disk and terrain field files, as sources of
  elevations.
- [`terrain_descriptors()`](https://gillescolling.com/resolve/articles/python-terrain-correction.html#terrain_descriptors):
  The elevation, the 1 km mean, the slope and the 1 km inclination at
  coordinates.
- [`encode_terrain_grids()`](https://gillescolling.com/resolve/articles/python-terrain-correction.html#encode_terrain_grids):
  Elevation grids coded into a terrain field with a bounded error per
  cell.
- [`fit_terrain_tile()`,
  `assemble_terrain_field()`](https://gillescolling.com/resolve/articles/python-terrain-correction.html#fit_terrain_tile):
  The neural terrain field format, tile by tile.
- [`fit_terrain_correction()`](https://gillescolling.com/resolve/articles/python-terrain-correction.html#fit_terrain_correction):
  The gated correction of one target, fitted on plots with recorded
  values.
- [`TerrainCorrection`](https://gillescolling.com/resolve/articles/python-terrain-correction.html#TerrainCorrection):
  A fitted correction: apply, save, load.
- [`nearest_by_great_circle()`](https://gillescolling.com/resolve/articles/python-gap-fitting.html#nearest_by_great_circle):
  The nearest rows of a reference set by great-circle distance.

## [Species names](https://gillescolling.com/resolve/articles/python-names.md)

Species names read as the names a model learnt.

- [`resolve_names()`](https://gillescolling.com/resolve/articles/python-names.html#resolve_names):
  Resolve species names against a model’s vocabulary, with the route
  each takes.
- [`NameTable`](https://gillescolling.com/resolve/articles/python-names.html#NameTable):
  The release’s name table: entered names and synonyms with the token
  each stands for.
- [`SpeciesNames`](https://gillescolling.com/resolve/articles/python-names.html#SpeciesNames):
  How a load or a suite imputation resolves species names.
- [`NameResolver`](https://gillescolling.com/resolve/articles/python-names.html#NameResolver):
  Resolves names against one vocabulary.
- [`ResolvedName`](https://gillescolling.com/resolve/articles/python-names.html#ResolvedName):
  What the resolver gives for one name.
- [`NameRoute`](https://gillescolling.com/resolve/articles/python-names.html#NameRoute):
  Exact, alias, spelling or unresolved.
- [`SpeciesNameReport`](https://gillescolling.com/resolve/articles/python-names.html#SpeciesNameReport):
  The names a load or a suite target did not find as written.

## [Scoring](https://gillescolling.com/resolve/articles/python-scoring.md)

The metrics training reports, for values imputed anywhere.

- [`Metrics`](https://gillescolling.com/resolve/articles/python-scoring.html#Metrics):
  The regression and classification metrics as functions of two tensors.
- [`ClassificationMetrics`](https://gillescolling.com/resolve/articles/python-scoring.html#ClassificationMetrics):
  Accuracy, F1 scores and the confusion matrix of a classification.
- [`ConfidenceMetrics`](https://gillescolling.com/resolve/articles/python-scoring.html#ConfidenceMetrics):
  The accuracy and coverage of the imputed values above a confidence
  threshold.
- [`compute_species_recognition()`](https://gillescolling.com/resolve/articles/python-scoring.html#compute_species_recognition):
  How much of each plot’s species list a vocabulary recognises.
- [`SpeciesRecognition`](https://gillescolling.com/resolve/articles/python-scoring.html#SpeciesRecognition):
  The recognised species of each plot, by count and by abundance.
- [`compute_unknown_species_stats()`](https://gillescolling.com/resolve/articles/python-scoring.html#compute_unknown_species_stats):
  The unknown-species columns of each plot against a vocabulary.
- [`UnknownSpeciesStats`](https://gillescolling.com/resolve/articles/python-scoring.html#UnknownSpeciesStats):
  Each plot’s unknown share of abundance and number of unknown species.

## [The engine](https://gillescolling.com/resolve/articles/python-engine.md)

The engine loaded, its memory and thread settings, its errors, and the
index that matches names within a few edits.

- [`__version__`](https://gillescolling.com/resolve/articles/python-engine.html#__version__):
  The version of the engine loaded.
- [`set_vram_fraction()`](https://gillescolling.com/resolve/articles/python-engine.html#set_vram_fraction):
  Caps the share of GPU memory the engine may use.
- [`configure_cuda_allocator()`](https://gillescolling.com/resolve/articles/python-engine.html#configure_cuda_allocator):
  Sets the CUDA allocator’s configuration to the engine’s default.
- [`default_cuda_alloc_conf()`](https://gillescolling.com/resolve/articles/python-engine.html#default_cuda_alloc_conf):
  The engine’s default CUDA allocator configuration.
- [`set_thread_pools()`](https://gillescolling.com/resolve/articles/python-engine.html#set_thread_pools):
  Sets the number of threads libtorch uses.
- [`install_crash_handler()`](https://gillescolling.com/resolve/articles/python-engine.html#install_crash_handler):
  Makes a native fault end the process at once on Windows.
- [`StorageError`](https://gillescolling.com/resolve/articles/python-engine.html#StorageError):
  The exception for a file the engine cannot read or write.
- [`fuzzy`](https://gillescolling.com/resolve/articles/python-engine.html#fuzzy):
  The submodule of the name index.
- [`fuzzy.FuzzyIndex`](https://gillescolling.com/resolve/articles/python-engine.html#fuzzy.FuzzyIndex):
  An index that returns the entries within a few edits of a query.
- [`fuzzy.Match`](https://gillescolling.com/resolve/articles/python-engine.html#fuzzy.Match):
  One match a query returns.
