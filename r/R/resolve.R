#' Create a SpeciesEncoder (removed)
#'
#' The standalone `resolve.encoder()` wrapper bound a unified
#' `resolve::SpeciesEncoder` C++ class that has since been split in the
#' C++ engine into `resolve::RankPoolEncoder` (variable-length pool
#' encoding for rank-pool / transformer modes) and
#' `resolve::EmbeddingEncoder` (fixed top-k IDs for embed mode). Neither
#' is a drop-in replacement for the old unified API: the hash-embedding
#' output is gone, save/load is not implemented on the new C++ encoders,
#' and the `aggregation` / `representation` parameters no longer apply.
#'
#' Calling `resolve.encoder()` therefore raises an error. The modern
#' canonical path is [resolve.dataset.csv()], which dispatches to
#' `resolve::ResolveDataset::from_csv()` and performs all encoding
#' inside the C++ engine — no separate fit/transform step is needed.
#'
#' @param ... Ignored; preserved for argument compatibility with the
#'   pre-removal signature so existing call sites surface a clear error
#'   instead of an "unused argument" message.
#'
#' @return Never returns; always errors.
#'
#' @export
resolve.encoder <- function(...) {
  stop(
    "resolve.encoder() has been removed.\n",
    "The unified C++ resolve::SpeciesEncoder class was split into ",
    "resolve::RankPoolEncoder and resolve::EmbeddingEncoder, and the new ",
    "C++ encoders do not provide hash_embedding output, save/load, or the ",
    "aggregation/representation parameters of the old API.\n",
    "Use resolve.dataset.csv() instead: it calls ResolveDataset::from_csv() ",
    "in the C++ engine and handles encoding internally."
  )
}


#' Create a RESOLVE Dataset (removed)
#'
#' The legacy `resolve.dataset()` facade ran encoding in R via the unified
#' `resolve.encoder()` C++ wrapper, which has been removed (see
#' [resolve.encoder()] for the underlying engine change).
#'
#' Calling `resolve.dataset()` therefore raises an error. Use
#' [resolve.dataset.csv()] instead — it dispatches to the C++
#' `ResolveDataset::from_csv()` pipeline directly, performs all encoding
#' inside the engine, and is the input format expected by
#' [resolve.train.dataset()].
#'
#' @param ... Ignored; preserved for argument compatibility with the
#'   pre-removal signature so existing call sites surface a clear error
#'   instead of an "unused argument" message.
#'
#' @return Never returns; always errors.
#'
#' @export
resolve.dataset <- function(...) {
  stop(
    "resolve.dataset() has been removed.\n",
    "It used resolve.encoder() under the hood, which no longer maps to the ",
    "current C++ engine (the unified resolve::SpeciesEncoder was split into ",
    "RankPoolEncoder and EmbeddingEncoder; the new encoders do not expose ",
    "save/load or a hash_embedding output).\n",
    "Use resolve.dataset.csv() instead: it calls ResolveDataset::from_csv() ",
    "in the C++ engine and pairs with resolve.train.dataset() for training."
  )
}



#' Train a RESOLVE Model (legacy facade — removed)
#'
#' The legacy `resolve.train()` facade trained from a `resolve.dataset`
#' object produced by the removed `resolve.dataset()` function, which
#' depended on the equally removed `resolve.encoder()` (see those
#' functions' help pages for the underlying engine change).
#'
#' Calling `resolve.train()` therefore raises an error. Use
#' [resolve.train.dataset()] together with [resolve.dataset.csv()] —
#' that pair uses the C++ engine end-to-end (no R-side encoding step)
#' and is the supported training path.
#'
#' @param ... Ignored; preserved for argument compatibility with the
#'   pre-removal signature so existing call sites surface a clear error
#'   instead of an "unused argument" message.
#'
#' @return Never returns; always errors.
#'
#' @export
resolve.train <- function(...) {
  stop(
    "resolve.train() has been removed.\n",
    "It consumed the legacy resolve.dataset() output, which is no longer ",
    "supported (see ?resolve.encoder for the underlying C++ refactor).\n",
    "Use resolve.dataset.csv() to load and encode in the C++ engine, then ",
    "train with resolve.train.dataset()."
  )
}


#' Predict with a RESOLVE Model (legacy facade — removed)
#'
#' The legacy `resolve.predict()` facade consumed `resolve.dataset()`
#' output (a plain R list with `hashEmbedding` / `genusIds` etc.) which
#' is no longer produced — see [resolve.encoder()] for the underlying
#' engine change.
#'
#' Calling `resolve.predict()` therefore raises an error. Use
#' [resolve.predict.dataset()] together with [resolve.dataset.csv()]
#' and a `Predictor` loaded via [resolve.load()].
#'
#' @param ... Ignored; preserved for argument compatibility with the
#'   pre-removal signature so existing call sites surface a clear error
#'   instead of an "unused argument" message.
#'
#' @return Never returns; always errors.
#'
#' @export
resolve.predict <- function(...) {
  stop(
    "resolve.predict() has been removed.\n",
    "It consumed the legacy resolve.dataset() output, which is no longer ",
    "supported (see ?resolve.encoder for the underlying C++ refactor).\n",
    "Use resolve.dataset.csv() + resolve.predict.dataset() with a Predictor ",
    "loaded via resolve.load()."
  )
}


#' Load a Trained RESOLVE Model
#'
#' Load a model from a saved checkpoint.
#'
#' @param path Path to checkpoint file
#' @param device Device: "cpu" or "cuda" (default "cpu")
#'
#' @return A Predictor object
#'
#' @examples
#' \dontrun{
#' predictor <- resolve.load("model.pt")
#' preds <- resolve.predict(predictor, newData)
#' }
#'
#' @export
resolve.load <- function(path, device = "cpu") {
  # Input validation
  if (!is.character(path) || length(path) != 1) {
    stop("path must be a single file path string")
  }
  if (!file.exists(path)) {
    stop(sprintf("checkpoint file does not exist: %s", path))
  }
  if (!device %in% c("cpu", "cuda")) {
    stop("device must be 'cpu' or 'cuda'")
  }

  .resolve_module$Predictor_load(path, device)
}


#' Recover the Training Configuration From a Checkpoint
#'
#' Reads back the training hyperparameters that were persisted in a RESOLVE
#' checkpoint (batch size, learning rate, weight decay, phase boundaries,
#' learning-rate schedule, band thresholds, VRAM fraction, ...) without loading
#' the model weights. Fields the checkpoint does not persist (device, AMP and
#' cuDNN flags) come back at their defaults.
#'
#' @param path Path to a `.pt` checkpoint written by [resolve.save()] or a
#'   training run.
#' @return A named list of training-configuration fields. `loss_config` and
#'   `lr_scheduler` are integer enum codes.
#' @examples
#' \dontrun{
#' cfg <- resolve.load_train_config("model.pt")
#' cfg$batch_size
#' cfg$band_thresholds
#' }
#' @export
resolve.load_train_config <- function(path) {
  if (!is.character(path) || length(path) != 1) {
    stop("path must be a single file path string")
  }
  if (!file.exists(path)) {
    stop(sprintf("checkpoint file does not exist: %s", path))
  }
  .resolve_module$Trainer_load_train_config(path)
}


#' Recover the Run Metadata From a Checkpoint
#'
#' Reads back the run metadata persisted in a RESOLVE checkpoint: training time,
#' train/test plot counts, best and total epochs, the RESOLVE version and
#' timestamps, and the per-target final-metric tree.
#'
#' @param path Path to a `.pt` checkpoint.
#' @return A named list of run-metadata fields, including `final_metrics`, a
#'   nested list keyed by target then metric (e.g. `$final_metrics$area$rmse`).
#' @examples
#' \dontrun{
#' meta <- resolve.load_run_metadata("model.pt")
#' meta$best_epoch
#' meta$final_metrics$area$rmse
#' }
#' @export
resolve.load_run_metadata <- function(path) {
  if (!is.character(path) || length(path) != 1) {
    stop("path must be a single file path string")
  }
  if (!file.exists(path)) {
    stop(sprintf("checkpoint file does not exist: %s", path))
  }
  .resolve_module$Trainer_load_run_metadata(path)
}


#' Save a Trained RESOLVE Model
#'
#' Save model checkpoint.
#'
#' @param trainer A trained Trainer object (from resolve.train())
#' @param path Path to save checkpoint
#'
#' @examples
#' \dontrun{
#' result <- resolve.train(dataset)
#' resolve.save(result, "model_checkpoint.pt")
#' }
#'
#' @export
resolve.save <- function(trainer, path) {
  if (is.list(trainer) && !is.null(trainer$trainer)) {
    trainer$trainer$save(path)
  } else if (inherits(trainer, "Rcpp_Trainer") || inherits(trainer, "Rcpp_RTrainer")) {
    trainer$save(path)
  } else {
    stop("trainer must be a Trainer object or result from resolve.train()")
  }
}


#' Check Training Progress
#'
#' Read progress from a checkpoint directory.
#'
#' @param checkpointDir Path to checkpoint directory
#'
#' @return A list with progress information, or NULL if no checkpoint exists
#'
#' @examples
#' \dontrun{
#' progress <- resolve.progress("checkpoints/my_model")
#' if (!is.null(progress)) {
#'   cat(sprintf("Epoch %d/%d (%.1f%%)\n",
#'     progress$epoch, progress$maxEpochs,
#'     progress$progressPct))
#' }
#' }
#'
#' @export
resolve.progress <- function(checkpointDir) {
  progressFile <- file.path(checkpointDir, "progress.json")
  if (!file.exists(progressFile)) {
    return(NULL)
  }
  jsonlite::fromJSON(progressFile)
}


#' Load Dataset from CSV Files (C++ Implementation)
#'
#' Load a dataset directly using the C++ ResolveDataset class.
#' This mirrors the Python `ResolveDataset.from_csv()` API exactly.
#'
#' @param header Path to header CSV file (one row per plot with targets)
#' @param species Path to species CSV file (one row per species-plot occurrence)
#' @param roles Named list mapping column roles:
#'   - plot_id: Column name for plot ID (default "plot_id")
#'   - species_id: Column name for species ID (default "species_id")
#'   - abundance: Column name for abundance (optional)
#'   - longitude: Column name for longitude (optional)
#'   - latitude: Column name for latitude (optional)
#'   - genus: Column name for genus (optional)
#'   - family: Column name for family (optional)
#'   - covariates: Vector of covariate column names (optional)
#' @param targets Named list of target configurations. Each target should have:
#'   - column: Column name in header file
#'   - task: "regression" or "classification"
#'   - transform: "none" or "log1p" (optional, default "none")
#'   - num_classes: Number of classes for classification (required for classification)
#'   - weight: Loss weight (optional, default 1.0)
#' @param config Named list of dataset configuration options:
#'   - species_encoding: "hash" (default), "embed", or "sparse"
#'   - hash_dim: Hash dimension (default 32)
#'   - top_k: Top-k genera/families (default 5)
#'   - top_k_species: Top-k species for embed mode (default 10)
#'   - selection: "top", "bottom", "top_bottom", or "all" (default "top")
#'   - representation: "abundance" or "presence_absence" (default "abundance")
#'   - normalization: "raw", "norm", or "log1p" (default "norm")
#'   - track_unknown_fraction: Track unknown species fraction (default TRUE)
#'   - track_unknown_count: Track unknown species count (default FALSE)
#'   - use_taxonomy: Use taxonomy embeddings (default TRUE if genus/family provided)
#'   - pool_weighting: rank_pool weighting "binary", "abundance", "log1p",
#'     "norm", or "rank" (default "log1p")
#'   - pool_species_cap: rank_pool per-plot species cap; 0 = no cap (default),
#'     -1 = auto p99, >0 = manual cap
#' @param schemaSource Optional ResolveDataset (from a previous
#'   \code{resolve.dataset.csv()} call). When supplied, this dataset is encoded
#'   against that dataset's species / taxonomy / categorical vocabularies and
#'   classification class mappings instead of fitting its own, so a held-out or
#'   transfer set lines up with the training set's embedding namespaces. Default
#'   \code{NULL} fits fresh vocabularies.
#'
#' @return A ResolveDataset object (C++ class) with methods:
#'   - coordinates(): Get coordinate matrix
#'   - covariates(): Get covariate matrix
#'   - hash_embedding(): Get hash embedding matrix
#'   - genus_ids(), family_ids(): Get taxonomy ID matrices
#'   - targets(): Get target values as named list
#'   - schema(): Get dataset schema
#'   - plot_ids(): Get plot IDs
#'   - n_plots(): Get number of plots
#'
#' @examples
#' \dontrun{
#' # Load dataset using C++ implementation
#' dataset <- resolve.dataset.csv(
#'   header = "plots.csv",
#'   species = "species.csv",
#'   roles = list(
#'     plot_id = "plot_id",
#'     species_id = "species",
#'     abundance = "cover",
#'     longitude = "lon",
#'     latitude = "lat"
#'   ),
#'   targets = list(
#'     area = list(column = "area", task = "regression", transform = "log1p"),
#'     habitat = list(column = "habitat", task = "classification", num_classes = 9)
#'   ),
#'   config = list(
#'     species_encoding = "hash",
#'     hash_dim = 64,
#'     top_k = 5
#'   )
#' )
#'
#' # Access data
#' print(dataset$schema())
#' print(dataset$n_plots())
#' }
#'
#' @export
resolve.dataset.csv <- function(header,
                                species,
                                roles = list(),
                                targets = list(),
                                config = list(),
                                schemaSource = NULL) {
  # Input validation
  if (!is.character(header) || length(header) != 1) {
    stop("header must be a single file path string")
  }
  if (!file.exists(header)) {
    stop(sprintf("header file does not exist: %s", header))
  }
  if (!is.character(species) || length(species) != 1) {
    stop("species must be a single file path string")
  }
  if (!file.exists(species)) {
    stop(sprintf("species file does not exist: %s", species))
  }

  # Shared roles/targets validation + role defaults (single source of truth
  # with resolve.dataset.frame()).
  roles <- .resolve_normalize_roles_targets(roles, targets)

  # When schemaSource is supplied, encode this dataset against that dataset's
  # species / taxonomy / categorical vocabularies and classification class
  # mappings (leave-one-dataset-out / transfer evaluation), so the model's
  # lookup tables are indexed with the right namespace.
  if (!is.null(schemaSource)) {
    if (!inherits(schemaSource, "Rcpp_ResolveDataset")) {
      stop("schemaSource must be a ResolveDataset from resolve.dataset.csv()")
    }
    return(.resolve_module$ResolveDataset_from_csv_with_schema(
      header_path = header,
      species_path = species,
      roles_list = roles,
      targets_list = targets,
      schema_source = schemaSource,
      config_list = config
    ))
  }

  # Call C++ implementation
  .resolve_module$ResolveDataset_from_csv(
    header_path = header,
    species_path = species,
    roles_list = roles,
    targets_list = targets,
    config_list = config
  )
}

# Shared roles/targets validation + role defaults for the dataset loaders.
.resolve_normalize_roles_targets <- function(roles, targets) {
  if (!is.list(roles)) {
    stop("roles must be a named list")
  }
  if (!is.list(targets)) {
    stop("targets must be a named list")
  }
  if (length(targets) == 0) {
    stop("targets must not be empty - at least one target is required")
  }
  for (name in names(targets)) {
    tgt <- targets[[name]]
    if (!is.list(tgt)) {
      stop(sprintf("target '%s' must be a list with 'column' and 'task'", name))
    }
    if (is.null(tgt$column)) {
      stop(sprintf("target '%s' must have 'column' specified", name))
    }
    if (!is.null(tgt$task) && !tgt$task %in% c("regression", "classification")) {
      stop(sprintf("target '%s' task must be 'regression' or 'classification'", name))
    }
  }
  if (is.null(roles$plot_id)) roles$plot_id <- "plot_id"
  if (is.null(roles$species_id)) roles$species_id <- "species_id"
  roles
}

# Coerce a data.frame to a named list of character columns with NA -> "" (CSV
# missing-value semantics), preserving column order. The cross-binding carrier
# for the in-memory dataset loaders.
.resolve_df_to_columns <- function(df, what) {
  if (!is.data.frame(df)) {
    stop(sprintf("%s must be a data.frame", what))
  }
  cols <- lapply(df, function(col) {
    x <- as.character(col)
    x[is.na(x)] <- ""
    x
  })
  names(cols) <- names(df)
  cols
}

#' Load a ResolveDataset from in-memory data frames
#'
#' In-memory analog of [resolve.dataset.csv()] (issue #22). Builds and encodes a
#' dataset directly from data frames already in R, eliminating the
#' write-to-temp-CSV / re-read round-trip that the file-based loader forces when
#' the header must be filtered or subset before a fit. The result is identical to
#' [resolve.dataset.csv()] on the equivalent CSV; only the disk I/O is elided.
#'
#' @param header A data.frame of header data (one row per plot). In single-table
#'   mode (\code{species = NULL}) this is instead the long-format species frame,
#'   matching [resolve.dataset.csv()]'s single-file behaviour.
#' @param species A data.frame of species data (long format), a single file path
#'   string (the large species table is then read once from disk while the
#'   header stays in memory), or \code{NULL} for single-table mode.
#' @param roles Named list mapping roles to column names (see
#'   [resolve.dataset.csv()]).
#' @param targets Named list of target specifications.
#' @param config Named list of dataset configuration options.
#' @param schemaSource Optional ResolveDataset whose vocabularies / class
#'   mappings are reused (the in-memory analog of \code{schemaSource} in
#'   [resolve.dataset.csv()]). Only valid when \code{species} is a data.frame.
#'
#' @return A ResolveDataset object.
#' @seealso [resolve.dataset.csv()]
#' @export
resolve.dataset.frame <- function(header,
                                  species = NULL,
                                  roles = list(),
                                  targets = list(),
                                  config = list(),
                                  schemaSource = NULL) {
  roles <- .resolve_normalize_roles_targets(roles, targets)

  # Single-table mode: `header` is the long-format species frame.
  if (is.null(species)) {
    if (!is.null(schemaSource)) {
      stop("schemaSource is not supported in single-table mode (pass a separate species frame)")
    }
    cols <- .resolve_df_to_columns(header, "header/species frame")
    return(.resolve_module$ResolveDataset_from_species_dataframe(
      species_cols = cols,
      roles_list = roles,
      targets_list = targets,
      config_list = config
    ))
  }

  header_cols <- .resolve_df_to_columns(header, "header")

  # Header frame + species CSV path.
  if (is.character(species)) {
    if (length(species) != 1) {
      stop("species path must be a single string")
    }
    if (!is.null(schemaSource)) {
      stop("schemaSource is not supported with a species CSV path; pass both as data frames")
    }
    if (!file.exists(species)) {
      stop(sprintf("species file does not exist: %s", species))
    }
    return(.resolve_module$ResolveDataset_from_dataframe_header(
      header_cols = header_cols,
      species_path = species,
      roles_list = roles,
      targets_list = targets,
      config_list = config
    ))
  }

  # Both frames in memory.
  species_cols <- .resolve_df_to_columns(species, "species")
  if (!is.null(schemaSource)) {
    if (!inherits(schemaSource, "Rcpp_ResolveDataset")) {
      stop("schemaSource must be a ResolveDataset from resolve.dataset.csv()/frame()")
    }
    return(.resolve_module$ResolveDataset_from_dataframe_with_schema(
      header_cols = header_cols,
      species_cols = species_cols,
      roles_list = roles,
      targets_list = targets,
      schema_source = schemaSource,
      config_list = config
    ))
  }
  .resolve_module$ResolveDataset_from_dataframe(
    header_cols = header_cols,
    species_cols = species_cols,
    roles_list = roles,
    targets_list = targets,
    config_list = config
  )
}


#' Train with Dataset (C++ API)
#'
#' Train a model using a ResolveDataset object loaded via resolve.dataset.csv().
#' This provides the cleanest API matching Python exactly.
#'
#' @param dataset A ResolveDataset object from resolve.dataset.csv()
#' @param hiddenDims Hidden layer dimensions (default c(2048, 1024, 512, 256, 128, 64))
#' @param maxEpochs Maximum training epochs (default 500)
#' @param patience Early stopping patience (default 50)
#' @param lr Learning rate (default 0.001)
#' @param batchSize Batch size (default 4096)
#' @param device Device: "cpu" or "cuda" (default "cpu")
#' @param testSize Fraction of data for testing (default 0.2)
#' @param seed Random seed (default 42)
#' @param savePath Path to save model checkpoint (optional)
#' @param lossConfig Loss configuration: "mae", "smape", or "combined" (default "mae")
#' @param coverDropout Cover-dropout rate applied to species cover values
#'   in rank-pool / transformer encoding modes (default 0.0, no dropout).
#' @param dModel Model dimension for the transformer / rank-pool encoder
#'   (default 128).
#' @param nHeads Number of attention heads in the transformer encoder
#'   (default 4). Ignored unless `nAttentionLayers > 0`.
#' @param nAttentionLayers Number of self-attention layers in the
#'   transformer encoder (default 0, i.e. rank-pool mean-only path).
#' @param transformerFfDim Feed-forward hidden dimension inside each
#'   transformer block (default 256).
#' @param transformerPooling Pooling strategy used to collapse the per-
#'   species token sequence into a plot embedding: `"attention"` or
#'   `"cls"` (default `"attention"`).
#' @param transformerDropout Dropout rate applied inside transformer
#'   blocks (default 0.1).
#' @param verbose Print training progress (default TRUE)
#'
#' @return A list with trainer, result, and dataset
#'
#' @examples
#' \dontrun{
#' dataset <- resolve.dataset.csv(...)
#' result <- resolve.train.dataset(dataset, maxEpochs = 100)
#' }
#'
#' @export
resolve.train.dataset <- function(dataset,
                                  hiddenDims = NULL,
                                  maxEpochs = 500L,
                                  patience = 50L,
                                  lr = 1e-3,
                                  batchSize = 4096L,
                                  device = "cpu",
                                  testSize = 0.2,
                                  seed = 42L,
                                  savePath = NULL,
                                  lossConfig = "mae",
                                  # RankPool / Transformer options
                                  coverDropout = 0.0,
                                  dModel = 128L,
                                  nHeads = 4L,
                                  nAttentionLayers = 0L,
                                  transformerFfDim = 256L,
                                  transformerPooling = "attention",
                                  transformerDropout = 0.1,
                                  verbose = TRUE) {
  # Input validation
  if (!inherits(dataset, "Rcpp_ResolveDataset")) {
    stop("dataset must be created with resolve.dataset.csv()")
  }
  if (!is.numeric(maxEpochs) || maxEpochs < 1) {
    stop("maxEpochs must be a positive integer")
  }
  if (!is.numeric(patience) || patience < 1) {
    stop("patience must be a positive integer")
  }
  if (!is.numeric(lr) || lr <= 0) {
    stop("lr must be a positive number")
  }
  if (!is.numeric(batchSize) || batchSize < 1) {
    stop("batchSize must be a positive integer")
  }
  if (!device %in% c("cpu", "cuda")) {
    stop("device must be 'cpu' or 'cuda'")
  }
  if (!is.numeric(testSize) || testSize <= 0 || testSize >= 1) {
    stop("testSize must be between 0 and 1 (exclusive)")
  }
  if (!lossConfig %in% c("mae", "smape", "combined")) {
    stop("lossConfig must be 'mae', 'smape', or 'combined'")
  }

  # Get schema from dataset
  schema <- dataset$schema()

  # Default hidden dims
  if (is.null(hiddenDims)) {
    hiddenDims <- c(2048L, 1024L, 512L, 256L, 128L, 64L)
  } else {
    hiddenDims <- as.integer(hiddenDims)
  }

  # Get config from dataset
  datasetConfig <- dataset$config()

  # Build model config from dataset
  modelConfig <- list(
    species_encoding = datasetConfig$species_encoding,
    hash_dim = datasetConfig$hash_dim,
    top_k = datasetConfig$top_k,
    top_k_species = datasetConfig$top_k_species,
    hidden_dims = hiddenDims,
    dropout = 0.3,
    cover_dropout = coverDropout,
    d_model = as.integer(dModel),
    n_heads = as.integer(nHeads),
    n_attention_layers = as.integer(nAttentionLayers),
    transformer_ff_dim = as.integer(transformerFfDim),
    transformer_pooling = transformerPooling,
    transformer_dropout = transformerDropout
  )

  # Create model
  model <- new(.resolve_module$ResolveModel, schema, modelConfig)

  # Build train config
  trainConfig <- list(
    batch_size = as.integer(batchSize),
    max_epochs = as.integer(maxEpochs),
    patience = as.integer(patience),
    lr = lr,
    device = device,
    loss_config = lossConfig
  )

  # Create trainer
  trainer <- new(.resolve_module$Trainer, model, trainConfig)

  # Prepare data from dataset (C++ API)
  trainer$prepare_data_from_dataset(dataset, testSize, as.integer(seed))

  # Train
  if (verbose) {
    cat("Training RESOLVE model...\n")
  }

  result <- trainer$fit()

  if (verbose) {
    cat(sprintf("Training complete. Best epoch: %d\n", result$best_epoch))
    cat(sprintf("Training time: %.1f seconds\n", result$train_time_seconds))
    for (targetName in names(result$final_metrics)) {
      metrics <- result$final_metrics[[targetName]]
      cat(sprintf("  %s: ", targetName))
      metricStrs <- sapply(names(metrics), function(m) {
        sprintf("%s=%.4f", m, metrics[[m]])
      })
      cat(paste(metricStrs, collapse = ", "), "\n")
    }
  }

  # Save if requested
  if (!is.null(savePath)) {
    trainer$save(savePath)
    if (verbose) {
      cat(sprintf("Model saved to: %s\n", savePath))
    }
  }

  list(
    trainer = trainer,
    result = result,
    dataset = dataset
  )
}


#' Predict with Dataset (C++ API)
#'
#' Make predictions on a ResolveDataset using a trained model.
#'
#' @param predictor A Predictor object from resolve.load()
#' @param dataset A ResolveDataset object from resolve.dataset.csv()
#' @param returnLatent Return latent representations (default FALSE)
#'
#' @return Named list of prediction arrays
#'
#' @examples
#' \dontrun{
#' predictor <- resolve.load("model.pt")
#' dataset <- resolve.dataset.csv(...)
#' preds <- resolve.predict.dataset(predictor, dataset)
#' }
#'
#' @export
resolve.predict.dataset <- function(predictor, dataset, returnLatent = FALSE) {
  if (!inherits(predictor, "Rcpp_Predictor")) {
    stop("predictor must be loaded with resolve.load()")
  }
  if (!inherits(dataset, "Rcpp_ResolveDataset")) {
    stop("dataset must be created with resolve.dataset.csv()")
  }

  predictor$predict_dataset(dataset, returnLatent)
}
