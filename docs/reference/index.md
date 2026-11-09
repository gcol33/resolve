# Package index

## Loading data

Plots and the species records observed in them, from CSV files or data
frames, into a RESOLVE engine dataset.

- [`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md)
  : Load a Dataset From CSV Files
- [`resolve.dataset.frame()`](https://gillescolling.com/resolve/reference/resolve.dataset.frame.md)
  : Load a ResolveDataset from in-memory data frames
- [`resolve.species_graph()`](https://gillescolling.com/resolve/reference/resolve.species_graph.md)
  : Build the Species Graph a Heterogeneous GNN Passes Messages On

## Training

Fitting a model, cross-validating it and pretraining its encoder without
labels.

- [`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md)
  : Train a Model on a Dataset
- [`resolve.cross_validate()`](https://gillescolling.com/resolve/reference/resolve.cross_validate.md)
  : Cross-Validate a Model on a Dataset
- [`resolve.pretrain_vae()`](https://gillescolling.com/resolve/reference/resolve.pretrain_vae.md)
  : Pretrain a Variational Autoencoder on Species Composition
- [`resolve.default_config()`](https://gillescolling.com/resolve/reference/resolve.default_config.md)
  : Default Configuration Values
- [`resolve.progress()`](https://gillescolling.com/resolve/reference/resolve.progress.md)
  : Check Training Progress

## Saving and imputing

Checkpoints, what they record about the run, and imputed values for new
plots.

- [`resolve.save()`](https://gillescolling.com/resolve/reference/resolve.save.md)
  : Save a Trained RESOLVE Model
- [`resolve.load()`](https://gillescolling.com/resolve/reference/resolve.load.md)
  : Load a Trained RESOLVE Model
- [`resolve.load_train_config()`](https://gillescolling.com/resolve/reference/resolve.load_train_config.md)
  : Recover the Training Configuration From a Checkpoint
- [`resolve.load_run_metadata()`](https://gillescolling.com/resolve/reference/resolve.load_run_metadata.md)
  : Recover the Run Metadata From a Checkpoint
- [`resolve.predict.dataset()`](https://gillescolling.com/resolve/reference/resolve.predict.dataset.md)
  : Predict with Dataset (C++ API)
- [`resolve.nearest_by_cosine()`](https://gillescolling.com/resolve/reference/resolve.nearest_by_cosine.md)
  : Nearest Rows by Cosine Similarity
- [`resolve.retrieval()`](https://gillescolling.com/resolve/reference/resolve.retrieval.md)
  : Blend Predictions With Their Nearest Reference Plots
- [`resolve.plot_totals()`](https://gillescolling.com/resolve/reference/resolve.plot_totals.md)
  : Give a model each plot's totals, or withhold them

## Model suites

Several trained models released as one directory and scored together.

- [`resolve.load_suite()`](https://gillescolling.com/resolve/reference/resolve.load_suite.md)
  : Load a Model Suite
- [`resolve.verify_suite()`](https://gillescolling.com/resolve/reference/resolve.verify_suite.md)
  : Check a Model Suite Against Its Manifest
- [`resolve.seal_suite()`](https://gillescolling.com/resolve/reference/resolve.seal_suite.md)
  : Seal a Model Suite Being Built
- [`resolve.predict.suite()`](https://gillescolling.com/resolve/reference/resolve.predict.suite.md)
  : Predict With a Model Suite
- [`as.data.frame(`*`<resolve_suite_predictions>`*`)`](https://gillescolling.com/resolve/reference/as.data.frame.resolve_suite_predictions.md)
  : Flatten Suite Predictions to One Row per Plot

## Gap fitting

Imputing a dataset’s missing values from its own complete plots, beside
the raw imputed value.

- [`resolve.fit_gaps()`](https://gillescolling.com/resolve/reference/resolve.fit_gaps.md)
  : Fit a Dataset's Gaps From Its Own Complete Plots
- [`resolve.fit_prediction_gaps()`](https://gillescolling.com/resolve/reference/resolve.fit_prediction_gaps.md)
  : Fit a Single Model's Gaps From Its Predictions and the Dataset's
  Tables

## Terrain correction

Altitude and slope combined with an elevation model read at the plots,
and the compressed elevation model a suite ships.

- [`resolve.terrain_descriptors()`](https://gillescolling.com/resolve/reference/resolve.terrain_descriptors.md)
  : Read Elevation-Model Descriptors at Plot Coordinates
- [`resolve.terrain_grid()`](https://gillescolling.com/resolve/reference/resolve.terrain_grid.md)
  : Read One Tile of an Elevation Model
- [`resolve.fit_terrain_correction()`](https://gillescolling.com/resolve/reference/resolve.fit_terrain_correction.md)
  : Fit a Terrain Correction
- [`resolve.terrain_correction()`](https://gillescolling.com/resolve/reference/resolve.terrain_correction.md)
  : Apply a Saved Terrain Correction
- [`resolve.encode_terrain_grids()`](https://gillescolling.com/resolve/reference/resolve.encode_terrain_grids.md)
  : Code Elevation Grids Into a Coded Terrain Field
- [`resolve.fit_terrain_tile()`](https://gillescolling.com/resolve/reference/resolve.fit_terrain_tile.md)
  [`resolve.assemble_terrain_field()`](https://gillescolling.com/resolve/reference/resolve.fit_terrain_tile.md)
  : Fit and Assemble a Terrain Field

## Scoring

The metrics training reports, for values imputed anywhere.

- [`resolve_band_accuracy()`](https://gillescolling.com/resolve/reference/resolve_metrics.md)
  [`resolve_mae()`](https://gillescolling.com/resolve/reference/resolve_metrics.md)
  [`resolve_rmse()`](https://gillescolling.com/resolve/reference/resolve_metrics.md)
  [`resolve_smape()`](https://gillescolling.com/resolve/reference/resolve_metrics.md)
  [`resolve_accuracy()`](https://gillescolling.com/resolve/reference/resolve_metrics.md)
  [`resolve_r_squared()`](https://gillescolling.com/resolve/reference/resolve_metrics.md)
  : Prediction metrics

## Name matching

Species names read as the names a model learnt, and the entries of a
list of names within a few edits of a query.

- [`resolve.names()`](https://gillescolling.com/resolve/reference/resolve.names.md)
  : Resolve Species Names Against a Model's Vocabulary
- [`resolve.fuzzy_index()`](https://gillescolling.com/resolve/reference/resolve.fuzzy_index.md)
  : Fuzzy String Index
- [`resolve.fuzzy_match()`](https://gillescolling.com/resolve/reference/resolve.fuzzy_match.md)
  : Match Strings Against a Fuzzy Index

## The RESOLVE engine

Installing the RESOLVE engine library, checking which one is loaded,
seeding its random numbers, and the GPU memory settings.

- [`resolve.install_backend()`](https://gillescolling.com/resolve/reference/resolve.install_backend.md)
  : Download and install the resolve_c backend
- [`resolve.available()`](https://gillescolling.com/resolve/reference/resolve.available.md)
  : Is the resolve_c backend available?
- [`resolve.version()`](https://gillescolling.com/resolve/reference/resolve.version.md)
  : Get RESOLVE version
- [`resolve.manual_seed()`](https://gillescolling.com/resolve/reference/resolve.manual_seed.md)
  : Seed the RESOLVE Engine's Random Numbers
- [`resolve.set_vram_fraction()`](https://gillescolling.com/resolve/reference/resolve.set_vram_fraction.md)
  : Cap the GPU Memory the RESOLVE Engine May Use
- [`resolve_configure_cuda_allocator()`](https://gillescolling.com/resolve/reference/resolve_configure_cuda_allocator.md)
  : Configure the PyTorch CUDA caching allocator
