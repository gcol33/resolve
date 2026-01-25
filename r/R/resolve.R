#' Create a SpeciesEncoder
#'
#' Create a species encoder that transforms species composition data into
#' fixed-dimension embeddings for model input.
#'
#' @param hashDim Dimension of species hash embedding (default 32)
#' @param topK Number of top/bottom genera/families to track (default 3)
#' @param aggregation Aggregation mode: "abundance" or "count" (default "abundance")
#' @param normalization Normalization mode: "raw", "norm", or "log1p" (default "norm")
#' @param trackUnknownCount Track count of unknown species (default FALSE)
#' @param selection Selection mode: "top", "bottom", "top_bottom", or "all" (default "top")
#' @param representation Representation mode: "abundance" or "presence_absence" (default "abundance")
#' @param minSpeciesFrequency Minimum frequency for a species to be included (default 1)
#'
#' @return A SpeciesEncoder object
#'
#' @examples
#' \dontrun{
#' encoder <- resolve.encoder(
#'   hashDim = 32,
#'   topK = 5,
#'   selection = "top_bottom"
#' )
#' encoder$fit(speciesData)
#' encoded <- encoder$transform(speciesData, plotIds)
#' }
#'
#' @export
resolve.encoder <- function(hashDim = 32L,
                            topK = 3L,
                            aggregation = "abundance",
                            normalization = "norm",
                            trackUnknownCount = FALSE,
                            selection = "top",
                            representation = "abundance",
                            minSpeciesFrequency = 1L) {
  new(.resolve_module$SpeciesEncoder,
      as.integer(hashDim),
      as.integer(topK),
      aggregation,
      normalization,
      trackUnknownCount,
      selection,
      representation,
      as.integer(minSpeciesFrequency))
}


#' Create a RESOLVE Dataset
#'
#' Load and configure a dataset for species composition prediction.
#' This is a convenience function that reads CSV files and prepares data for training.
#'
#' @param header Path to plot-level CSV (one row per plot)
#' @param species Path to species CSV (one row per species-plot occurrence)
#' @param roles Named list mapping semantic roles to column names:
#'   - plotId: Column with plot IDs in header file
#'   - speciesId: Column with species names in species file
#'   - speciesPlotId: Column with plot IDs in species file
#'   - abundance: Column with abundance values (optional)
#'   - genus: Column with genus names (optional)
#'   - family: Column with family names (optional)
#'   - x: Column with x coordinates (optional)
#'   - y: Column with y coordinates (optional)
#' @param targets Named list of target configurations. Each target should have:
#'   - column: Column name in header file
#'   - task: "regression" or "classification"
#'   - transform: "none" or "log1p" (optional)
#' @param speciesNormalization Normalization mode: "raw", "norm", or "log1p" (default "norm")
#' @param trackUnknownFraction Track fraction of unknown species (default TRUE)
#' @param trackUnknownCount Track count of unknown species (default FALSE)
#' @param hashDim Dimension of species hash embedding (default 32)
#' @param topK Number of top genera/families to track (default 5)
#' @param topKSpecies Number of top species to track for embed mode (default 10)
#' @param selection Selection mode: "top", "bottom", "top_bottom", "all" (default "top")
#' @param representation Representation mode: "abundance", "presence_absence" (default "abundance")
#'
#' @return A list containing prepared dataset components:
#'   - encoder: Fitted SpeciesEncoder
#'   - coordinates: Matrix of plot coordinates
#'   - covariates: Matrix of covariate values
#'   - hashEmbedding: Matrix of species hash embeddings
#'   - speciesIds: Matrix of top-k species IDs per plot (for embed mode)
#'   - genusIds: Matrix of genus IDs (if hasTaxonomy)
#'   - familyIds: Matrix of family IDs (if hasTaxonomy)
#'   - unknownFraction: Vector of unknown fractions (if trackUnknownFraction)
#'   - targets: Named list of target vectors
#'   - schema: Schema information
#'
#' @examples
#' \dontrun{
#' dataset <- resolve.dataset(
#'   header = "plots.csv",
#'   species = "species.csv",
#'   roles = list(
#'     plotId = "plot_id",
#'     speciesId = "species",
#'     speciesPlotId = "plot_id",
#'     abundance = "cover"
#'   ),
#'   targets = list(
#'     biomass = list(column = "agb", task = "regression")
#'   )
#' )
#' }
#'
#' @export
resolve.dataset <- function(header,
                            species,
                            roles,
                            targets,
                            speciesNormalization = "norm",
                            trackUnknownFraction = TRUE,
                            trackUnknownCount = FALSE,
                            hashDim = 32L,
                            topK = 5L,
                            topKSpecies = 10L,
                            selection = "top",
                            representation = "abundance") {
  # Read CSV files
  headerDf <- read.csv(header, stringsAsFactors = FALSE)
  speciesDf <- read.csv(species, stringsAsFactors = FALSE)

  # Extract plot IDs
  plotIdCol <- roles$plotId
  plotIds <- as.character(headerDf[[plotIdCol]])

  # Prepare species data frame with required columns
  speciesData <- data.frame(
    species_id = as.character(speciesDf[[roles$speciesId]]),
    plot_id = as.character(speciesDf[[roles$speciesPlotId]]),
    stringsAsFactors = FALSE
  )

  # Add abundance (default to 1 if not provided)
  if (!is.null(roles$abundance) && roles$abundance %in% names(speciesDf)) {
    speciesData$abundance <- as.numeric(speciesDf[[roles$abundance]])
  } else {
    speciesData$abundance <- rep(1, nrow(speciesDf))
  }

  # Add taxonomy if available
  hasTaxonomy <- !is.null(roles$genus) && !is.null(roles$family)
  if (hasTaxonomy) {
    speciesData$genus <- as.character(speciesDf[[roles$genus]])
    speciesData$family <- as.character(speciesDf[[roles$family]])
  } else {
    # Fill with empty strings if no taxonomy
    speciesData$genus <- rep("", nrow(speciesDf))
    speciesData$family <- rep("", nrow(speciesDf))
  }

  # Create and fit encoder
  encoder <- resolve.encoder(
    hashDim = as.integer(hashDim),
    topK = as.integer(topK),
    aggregation = "abundance",
    normalization = speciesNormalization,
    trackUnknownCount = trackUnknownCount,
    selection = selection,
    representation = representation
  )
  encoder$fit(speciesData)

  # Transform species data
  encoded <- encoder$transform(speciesData, plotIds)

  # Extract coordinates if available
  hasCoordinates <- !is.null(roles$x) && !is.null(roles$y)
  if (hasCoordinates) {
    coordinates <- cbind(
      as.numeric(headerDf[[roles$x]]),
      as.numeric(headerDf[[roles$y]])
    )
  } else {
    coordinates <- matrix(0, nrow = length(plotIds), ncol = 2)
  }

  # Extract covariates (columns not used for other purposes)
  usedCols <- c(
    plotIdCol,
    roles$x, roles$y,
    sapply(targets, function(t) t$column)
  )
  covariateCols <- setdiff(names(headerDf), usedCols)
  if (length(covariateCols) > 0) {
    covariates <- as.matrix(headerDf[, covariateCols, drop = FALSE])
    storage.mode(covariates) <- "double"
  } else {
    covariates <- matrix(0, nrow = length(plotIds), ncol = 1)
  }

  # Extract target values
  targetValues <- list()
  targetConfigs <- list()
  for (name in names(targets)) {
    cfg <- targets[[name]]
    targetValues[[name]] <- as.numeric(headerDf[[cfg$column]])
    targetConfigs[[name]] <- cfg
  }

  # Prepare speciesIds for embed mode (top-k species per plot)
  # Build species vocabulary
  uniqueSpecies <- sort(unique(speciesData$species_id))
  speciesToIdx <- setNames(seq_along(uniqueSpecies), uniqueSpecies)
  nSpeciesVocab <- length(uniqueSpecies) + 1L  # +1 for padding/unknown (index 0)

  # Compute top-k species per plot by abundance
  speciesIdsMatrix <- matrix(0L, nrow = length(plotIds), ncol = as.integer(topKSpecies))
  for (i in seq_along(plotIds)) {
    pid <- plotIds[i]
    plotSpecies <- speciesData[speciesData$plot_id == pid, , drop = FALSE]
    if (nrow(plotSpecies) > 0) {
      # Sort by abundance descending
      plotSpecies <- plotSpecies[order(-plotSpecies$abundance), , drop = FALSE]
      # Take top-k species
      topSpecies <- head(plotSpecies$species_id, topKSpecies)
      # Map to indices (1-based, 0 is padding)
      for (j in seq_along(topSpecies)) {
        speciesIdsMatrix[i, j] <- speciesToIdx[topSpecies[j]]
      }
    }
  }

  # Build schema
  schema <- list(
    nPlots = length(plotIds),
    nSpecies = encoder$n_known_species(),
    nSpeciesVocab = nSpeciesVocab,
    hasCoordinates = hasCoordinates,
    hasAbundance = TRUE,
    hasTaxonomy = hasTaxonomy,
    nGenera = encoder$n_genera(),
    nFamilies = encoder$n_families(),
    covariateNames = covariateCols,
    targets = targetConfigs,
    trackUnknownFraction = trackUnknownFraction,
    trackUnknownCount = trackUnknownCount,
    topKSpecies = as.integer(topKSpecies)
  )

  # Return dataset components
  result <- list(
    encoder = encoder,
    coordinates = coordinates,
    covariates = covariates,
    targets = targetValues,
    schema = schema,
    plotIds = plotIds
  )

  # Add encoded species features
  if (!is.null(encoded$hash_embedding)) {
    result$hashEmbedding <- encoded$hash_embedding
  }
  if (!is.null(encoded$genus_ids)) {
    result$genusIds <- encoded$genus_ids
  }
  if (!is.null(encoded$family_ids)) {
    result$familyIds <- encoded$family_ids
  }
  if (!is.null(encoded$unknown_fraction) && trackUnknownFraction) {
    result$unknownFraction <- encoded$unknown_fraction
  }
  if (!is.null(encoded$unknown_count) && trackUnknownCount) {
    result$unknownCount <- encoded$unknown_count
  }
  if (!is.null(encoded$species_vector)) {
    result$speciesVector <- encoded$species_vector
  }

  # Always add speciesIds for embed mode support
  result$speciesIds <- speciesIdsMatrix
  result$speciesVocab <- uniqueSpecies

  class(result) <- "resolve.dataset"
  result
}


#' Train a RESOLVE Model
#'
#' Train a model on a dataset with sensible defaults.
#'
#' @param dataset A resolve.dataset object (from resolve.dataset())
#' @param speciesEncoding How to encode species: "hash" (default) or "embed".
#' @param hiddenDims Hidden layer dimensions (default c(2048, 1024, 512, 256, 128, 64))
#' @param maxEpochs Maximum training epochs (default 500)
#' @param patience Early stopping patience (default 50)
#' @param lr Learning rate (default 0.001)
#' @param batchSize Batch size (default 4096)
#' @param device Device: "cpu" or "cuda" (default "cpu")
#' @param testSize Fraction of data to use for testing (default 0.2)
#' @param seed Random seed (default 42)
#' @param savePath Path to save final model (optional)
#' @param lossConfig Loss configuration preset: "mae" (default), "smape", or "combined".
#'   - "mae": Pure MAE loss (no SMAPE, no band penalty)
#'   - "smape": SMAPE as primary loss from start
#'   - "combined": Phased training (MAE -> MAE+SMAPE -> MAE+SMAPE+band)
#' @param verbose Print training progress (default TRUE)
#'
#' @return A trained model object with fit results
#'
#' @examples
#' \dontrun{
#' dataset <- resolve.dataset(...)
#' result <- resolve.train(dataset)
#' print(result$metrics)
#' }
#'
#' @export
resolve.train <- function(dataset,
                          speciesEncoding = "hash",
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
                          verbose = TRUE) {
  if (!inherits(dataset, "resolve.dataset")) {
    stop("dataset must be created with resolve.dataset()")
  }

  # Validate speciesEncoding
  if (!speciesEncoding %in% c("hash", "embed")) {
    stop("speciesEncoding must be 'hash' or 'embed'")
  }

  # For embed mode, verify dataset has speciesIds
  if (speciesEncoding == "embed" && is.null(dataset$speciesIds)) {
    stop("speciesEncoding='embed' requires dataset with speciesIds")
  }

  # Validate lossConfig
  if (!lossConfig %in% c("mae", "smape", "combined")) {
    stop("lossConfig must be 'mae', 'smape', or 'combined'")
  }

  # Default hidden dims
  if (is.null(hiddenDims)) {
    hiddenDims <- c(2048L, 1024L, 512L, 256L, 128L, 64L)
  } else {
    hiddenDims <- as.integer(hiddenDims)
  }

  # Build model config
  modelConfig <- list(
    species_encoding = speciesEncoding,
    hash_dim = dataset$encoder$hash_dim(),
    top_k = dataset$encoder$top_k(),
    top_k_species = if (!is.null(dataset$schema$topKSpecies)) dataset$schema$topKSpecies else 10L,
    n_taxonomy_slots = dataset$encoder$n_taxonomy_slots(),
    hidden_dims = hiddenDims,
    dropout = 0.3
  )

  # Create model
  model <- new(.resolve_module$ResolveModel, dataset$schema, modelConfig)

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

  # Prepare hash embedding (required for hash mode)
  hashEmb <- if (!is.null(dataset$hashEmbedding)) {
    dataset$hashEmbedding
  } else {
    matrix(0, nrow = nrow(dataset$coordinates), ncol = 1)
  }

  # Prepare data
  trainer$prepare_data(
    coordinates = dataset$coordinates,
    covariates = dataset$covariates,
    hash_embedding = hashEmb,
    species_ids = if (!is.null(dataset$speciesIds)) dataset$speciesIds else NULL,
    species_vector = if (!is.null(dataset$speciesVector)) dataset$speciesVector else NULL,
    genus_ids = if (!is.null(dataset$genusIds)) as.integer(dataset$genusIds) else NULL,
    family_ids = if (!is.null(dataset$familyIds)) as.integer(dataset$familyIds) else NULL,
    unknown_fraction = dataset$unknownFraction,
    unknown_count = dataset$unknownCount,
    targets = dataset$targets,
    test_size = testSize,
    seed = as.integer(seed)
  )

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

  # Return trainer with results
  list(
    trainer = trainer,
    result = result,
    dataset = dataset
  )
}


#' Predict with a RESOLVE Model
#'
#' Make predictions on a dataset using a trained model.
#'
#' @param model A trained model (from resolve.train() or resolve.load())
#' @param dataset A resolve.dataset to predict on
#' @param returnLatent Return latent representations (default FALSE)
#' @param outputSpace Output space for regression predictions:
#'   "raw" (default): inverse-transform predictions to original scale
#'   "transformed": keep predictions in transformed space (e.g., log1p)
#' @param confidenceThreshold Minimum confidence for predictions (0-1).
#'   Predictions below threshold are set to NA. Default 0 keeps all predictions.
#'   Confidence is based on 1 - unknownFraction (species coverage).
#'
#' @return Named list of prediction arrays, plus 'confidence' for each target
#'
#' @examples
#' \dontrun{
#' preds <- resolve.predict(trainedModel, newDataset)
#' preds <- resolve.predict(trainedModel, newDataset, confidenceThreshold = 0.5)
#' }
#'
#' @export
resolve.predict <- function(model,
                            dataset,
                            returnLatent = FALSE,
                            outputSpace = "raw",
                            confidenceThreshold = 0.0) {
  # Validate outputSpace
  if (!outputSpace %in% c("raw", "transformed")) {
    stop("outputSpace must be 'raw' or 'transformed'")
  }

  # Handle both trainer objects and predictor objects
  if (inherits(model, "Rcpp_Predictor") || inherits(model, "Rcpp_RPredictor")) {
    predictor <- model
    hashEmb <- if (!is.null(dataset$hashEmbedding)) {
      dataset$hashEmbedding
    } else {
      matrix(0, nrow = nrow(dataset$coordinates), ncol = 1)
    }

    result <- predictor$predict(
      coordinates = dataset$coordinates,
      covariates = dataset$covariates,
      hash_embedding = hashEmb,
      genus_ids = if (!is.null(dataset$genusIds)) as.integer(dataset$genusIds) else NULL,
      family_ids = if (!is.null(dataset$familyIds)) as.integer(dataset$familyIds) else NULL,
      return_latent = returnLatent
    )

    # Compute confidence from unknownFraction (1 - unknownFraction)
    if (!is.null(dataset$unknownFraction)) {
      confidence <- 1.0 - dataset$unknownFraction
    } else {
      # If no unknown tracking, assume full confidence
      confidence <- rep(1.0, nrow(dataset$coordinates))
    }

    # Post-process each target
    targetNames <- names(dataset$schema$targets)
    for (name in targetNames) {
      if (!is.null(result[[name]])) {
        cfg <- dataset$schema$targets[[name]]

        # Apply inverse transform for log1p targets if outputSpace == "raw"
        if (!is.null(cfg$transform) && cfg$transform == "log1p" && outputSpace == "raw") {
          result[[name]] <- expm1(result[[name]])
        }

        # Apply confidence threshold
        if (confidenceThreshold > 0) {
          result[[name]][confidence < confidenceThreshold] <- NA
        }

        # Add confidence for this target
        result[[paste0(name, "_confidence")]] <- confidence
      }
    }

    result$confidence <- confidence
    result

  } else if (is.list(model) && !is.null(model$trainer)) {
    # Trainer from resolve.train() - use temp file workaround
    tempPath <- tempfile(fileext = ".pt")
    on.exit(unlink(tempPath), add = TRUE)
    model$trainer$save(tempPath)
    predictor <- resolve.load(tempPath)

    # Recurse with the predictor
    resolve.predict(predictor, dataset, returnLatent, outputSpace, confidenceThreshold)
  } else {
    stop("model must be a Predictor object or result from resolve.train()")
  }
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
  .resolve_module$Predictor_load(path, device)
}


#' Save a Trained RESOLVE Model
#'
#' Save model checkpoint.
#'
#' @param trainer A trained Trainer object (from resolve.train())
#' @param path Path to save checkpoint
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
