#' Create a SpeciesEncoder
#'
#' Create a species encoder that transforms species composition data into
#' fixed-dimension embeddings for model input.
#'
#' @param hash_dim Dimension of species hash embedding (default 32)
#' @param top_k Number of top/bottom genera/families to track (default 3)
#' @param aggregation Aggregation mode: "abundance" or "count" (default "abundance")
#' @param normalization Normalization mode: "raw", "norm", or "log1p" (default "norm")
#' @param track_unknown_count Track count of unknown species (default FALSE)
#' @param selection Selection mode: "top", "bottom", "top_bottom", or "all" (default "top")
#' @param representation Representation mode: "abundance" or "presence_absence" (default "abundance")
#' @param min_species_frequency Minimum frequency for a species to be included (default 1)
#'
#' @return A SpeciesEncoder object
#'
#' @examples
#' \dontrun{
#' encoder <- resolve_species_encoder(
#'   hash_dim = 32,
#'   top_k = 5,
#'   selection = "top_bottom"
#' )
#' encoder$fit(species_data)
#' encoded <- encoder$transform(species_data, plot_ids)
#' }
#'
#' @export
resolve_species_encoder <- function(hash_dim = 32L,
                                    top_k = 3L,
                                    aggregation = "abundance",
                                    normalization = "norm",
                                    track_unknown_count = FALSE,
                                    selection = "top",
                                    representation = "abundance",
                                    min_species_frequency = 1L) {
  new(.resolve_module$SpeciesEncoder,
      as.integer(hash_dim),
      as.integer(top_k),
      aggregation,
      normalization,
      track_unknown_count,
      selection,
      representation,
      as.integer(min_species_frequency))
}


#' Create a RESOLVE Dataset
#'
#' Load and configure a dataset for species composition prediction.
#' This is a convenience function that reads CSV files and prepares data for training.
#'
#' @param header Path to plot-level CSV (one row per plot)
#' @param species Path to species CSV (one row per species-plot occurrence)
#' @param roles Named list mapping semantic roles to column names:
#'   - plot_id: Column with plot IDs in header file
#'   - species_id: Column with species names in species file
#'   - species_plot_id: Column with plot IDs in species file
#'   - abundance: Column with abundance values (optional)
#'   - genus: Column with genus names (optional)
#'   - family: Column with family names (optional)
#'   - x: Column with x coordinates (optional)
#'   - y: Column with y coordinates (optional)
#' @param targets Named list of target configurations. Each target should have:
#'   - column: Column name in header file
#'   - task: "regression" or "classification"
#'   - transform: "none" or "log1p" (optional)
#' @param species_normalization Normalization mode: "raw", "norm", or "log1p" (default "norm")
#' @param track_unknown_fraction Track fraction of unknown species (default TRUE)
#' @param track_unknown_count Track count of unknown species (default FALSE)
#' @param hash_dim Dimension of species hash embedding (default 32)
#' @param top_k Number of top genera/families to track (default 5)
#' @param top_k_species Number of top species to track for embed mode (default 10)
#' @param selection Selection mode: "top", "bottom", "top_bottom", "all" (default "top")
#' @param representation Representation mode: "abundance", "presence_absence" (default "abundance")
#'
#' @return A list containing prepared dataset components:
#'   - encoder: Fitted SpeciesEncoder
#'   - coordinates: Matrix of plot coordinates
#'   - covariates: Matrix of covariate values
#'   - hash_embedding: Matrix of species hash embeddings
#'   - species_ids: Matrix of top-k species IDs per plot (for embed mode)
#'   - genus_ids: Matrix of genus IDs (if has_taxonomy)
#'   - family_ids: Matrix of family IDs (if has_taxonomy)
#'   - unknown_fraction: Vector of unknown fractions (if track_unknown_fraction)
#'   - targets: Named list of target vectors
#'   - schema: Schema information
#'
#' @examples
#' \dontrun{
#' dataset <- resolve_dataset(
#'   header = "plots.csv",
#'   species = "species.csv",
#'   roles = list(
#'     plot_id = "plot_id",
#'     species_id = "species",
#'     species_plot_id = "plot_id",
#'     abundance = "cover"
#'   ),
#'   targets = list(
#'     biomass = list(column = "agb", task = "regression")
#'   )
#' )
#' }
#'
#' @export
resolve_dataset <- function(header,
                            species,
                            roles,
                            targets,
                            species_normalization = "norm",
                            track_unknown_fraction = TRUE,
                            track_unknown_count = FALSE,
                            hash_dim = 32L,
                            top_k = 5L,
                            top_k_species = 10L,
                            selection = "top",
                            representation = "abundance") {
  # Read CSV files
  header_df <- read.csv(header, stringsAsFactors = FALSE)
  species_df <- read.csv(species, stringsAsFactors = FALSE)

  # Extract plot IDs
  plot_id_col <- roles$plot_id
  plot_ids <- as.character(header_df[[plot_id_col]])

  # Prepare species data frame with required columns
  species_data <- data.frame(
    species_id = as.character(species_df[[roles$species_id]]),
    plot_id = as.character(species_df[[roles$species_plot_id]]),
    stringsAsFactors = FALSE
  )

  # Add abundance (default to 1 if not provided)
  if (!is.null(roles$abundance) && roles$abundance %in% names(species_df)) {
    species_data$abundance <- as.numeric(species_df[[roles$abundance]])
  } else {
    species_data$abundance <- rep(1, nrow(species_df))
  }

  # Add taxonomy if available
  has_taxonomy <- !is.null(roles$genus) && !is.null(roles$family)
  if (has_taxonomy) {
    species_data$genus <- as.character(species_df[[roles$genus]])
    species_data$family <- as.character(species_df[[roles$family]])
  } else {
    # Fill with empty strings if no taxonomy
    species_data$genus <- rep("", nrow(species_df))
    species_data$family <- rep("", nrow(species_df))
  }

  # Create and fit encoder
  encoder <- resolve_species_encoder(
    hash_dim = as.integer(hash_dim),
    top_k = as.integer(top_k),
    aggregation = "abundance",
    normalization = species_normalization,
    track_unknown_count = track_unknown_count,
    selection = selection,
    representation = representation
  )
  encoder$fit(species_data)

  # Transform species data
  encoded <- encoder$transform(species_data, plot_ids)

  # Extract coordinates if available
  has_coordinates <- !is.null(roles$x) && !is.null(roles$y)
  if (has_coordinates) {
    coordinates <- cbind(
      as.numeric(header_df[[roles$x]]),
      as.numeric(header_df[[roles$y]])
    )
  } else {
    coordinates <- matrix(0, nrow = length(plot_ids), ncol = 2)
  }

  # Extract covariates (columns not used for other purposes)
  used_cols <- c(
    plot_id_col,
    roles$x, roles$y,
    sapply(targets, function(t) t$column)
  )
  covariate_cols <- setdiff(names(header_df), used_cols)
  if (length(covariate_cols) > 0) {
    covariates <- as.matrix(header_df[, covariate_cols, drop = FALSE])
    storage.mode(covariates) <- "double"
  } else {
    covariates <- matrix(0, nrow = length(plot_ids), ncol = 1)
  }

  # Extract target values
  target_values <- list()
  target_configs <- list()
  for (name in names(targets)) {
    cfg <- targets[[name]]
    target_values[[name]] <- as.numeric(header_df[[cfg$column]])
    target_configs[[name]] <- cfg
  }

  # Prepare species_ids for embed mode (top-k species per plot)
  # Build species vocabulary
  unique_species <- sort(unique(species_data$species_id))
  species_to_idx <- setNames(seq_along(unique_species), unique_species)
  n_species_vocab <- length(unique_species) + 1L  # +1 for padding/unknown (index 0)

  # Compute top-k species per plot by abundance
  species_ids_matrix <- matrix(0L, nrow = length(plot_ids), ncol = as.integer(top_k_species))
  for (i in seq_along(plot_ids)) {
    pid <- plot_ids[i]
    plot_species <- species_data[species_data$plot_id == pid, , drop = FALSE]
    if (nrow(plot_species) > 0) {
      # Sort by abundance descending
      plot_species <- plot_species[order(-plot_species$abundance), , drop = FALSE]
      # Take top-k species
      top_species <- head(plot_species$species_id, top_k_species)
      # Map to indices (1-based, 0 is padding)
      for (j in seq_along(top_species)) {
        species_ids_matrix[i, j] <- species_to_idx[top_species[j]]
      }
    }
  }

  # Build schema
  schema <- list(
    n_plots = length(plot_ids),
    n_species = encoder$n_known_species(),
    n_species_vocab = n_species_vocab,
    has_coordinates = has_coordinates,
    has_abundance = TRUE,
    has_taxonomy = has_taxonomy,
    n_genera = encoder$n_genera(),
    n_families = encoder$n_families(),
    covariate_names = covariate_cols,
    targets = target_configs,
    track_unknown_fraction = track_unknown_fraction,
    track_unknown_count = track_unknown_count,
    top_k_species = as.integer(top_k_species)
  )

  # Return dataset components
  result <- list(
    encoder = encoder,
    coordinates = coordinates,
    covariates = covariates,
    targets = target_values,
    schema = schema,
    plot_ids = plot_ids
  )

  # Add encoded species features
  if (!is.null(encoded$hash_embedding)) {
    result$hash_embedding <- encoded$hash_embedding
  }
  if (!is.null(encoded$genus_ids)) {
    result$genus_ids <- encoded$genus_ids
  }
  if (!is.null(encoded$family_ids)) {
    result$family_ids <- encoded$family_ids
  }
  if (!is.null(encoded$unknown_fraction) && track_unknown_fraction) {
    result$unknown_fraction <- encoded$unknown_fraction
  }
  if (!is.null(encoded$unknown_count) && track_unknown_count) {
    result$unknown_count <- encoded$unknown_count
  }
  if (!is.null(encoded$species_vector)) {
    result$species_vector <- encoded$species_vector
  }

  # Always add species_ids for embed mode support
  result$species_ids <- species_ids_matrix
  result$species_vocab <- unique_species

  class(result) <- "resolve_dataset"
  result
}


#' Train a RESOLVE Model
#'
#' Train a model on a dataset with sensible defaults.
#'
#' @param dataset A resolve_dataset object (from resolve_dataset())
#' @param species_encoding How to encode species: "hash" (default) or "embed".
#'   Note: "embed" mode is not yet fully supported in R bindings.
#' @param hidden_dims Hidden layer dimensions (default c(2048, 1024, 512, 256, 128, 64))
#' @param max_epochs Maximum training epochs (default 500)
#' @param patience Early stopping patience (default 50)
#' @param lr Learning rate (default 0.001)
#' @param batch_size Batch size (default 4096)
#' @param device Device: "cpu" or "cuda" (default "cpu")
#' @param test_size Fraction of data to use for testing (default 0.2)
#' @param seed Random seed (default 42)
#' @param save_path Path to save final model (optional)
#' @param loss_config Loss configuration preset: "mae" (default), "smape", or "combined"
#' @param verbose Print training progress (default TRUE)
#'
#' @return A trained model object with fit results
#'
#' @examples
#' \dontrun{
#' dataset <- resolve_dataset(...)
#' result <- resolve_train(dataset)
#' print(result$metrics)
#' }
#'
#' @export
resolve_train <- function(dataset,
                          species_encoding = "hash",
                          hidden_dims = NULL,
                          max_epochs = 500L,
                          patience = 50L,
                          lr = 1e-3,
                          batch_size = 4096L,
                          device = "cpu",
                          test_size = 0.2,
                          seed = 42L,
                          save_path = NULL,
                          loss_config = "mae",
                          verbose = TRUE) {
  if (!inherits(dataset, "resolve_dataset")) {
    stop("dataset must be created with resolve_dataset()")
  }

  # Validate species_encoding
  if (!species_encoding %in% c("hash", "embed")) {
    stop("species_encoding must be 'hash' or 'embed'")
  }

  # For embed mode, verify dataset has species_ids
  if (species_encoding == "embed" && is.null(dataset$species_ids)) {
    stop("species_encoding='embed' requires dataset with species_ids")
  }

  # Validate loss_config
  if (!loss_config %in% c("mae", "smape", "combined")) {
    stop("loss_config must be 'mae', 'smape', or 'combined'")
  }
  if (loss_config != "mae" && verbose) {
    message("Note: loss_config='", loss_config, "' is not yet fully supported in R bindings. ",
            "Using default loss (MAE + band penalty). For full loss_config support, use Python API.")
  }

  # Default hidden dims
  if (is.null(hidden_dims)) {
    hidden_dims <- c(2048L, 1024L, 512L, 256L, 128L, 64L)
  } else {
    hidden_dims <- as.integer(hidden_dims)
  }

  # Build model config
  model_config <- list(
    species_encoding = species_encoding,
    hash_dim = dataset$encoder$hash_dim(),
    top_k = dataset$encoder$top_k(),
    top_k_species = if (!is.null(dataset$schema$top_k_species)) dataset$schema$top_k_species else 10L,
    n_taxonomy_slots = dataset$encoder$n_taxonomy_slots(),
    hidden_dims = hidden_dims,
    dropout = 0.3
  )

  # Create model
  model <- new(.resolve_module$ResolveModel, dataset$schema, model_config)

  # Build train config
  train_config <- list(
    batch_size = as.integer(batch_size),
    max_epochs = as.integer(max_epochs),
    patience = as.integer(patience),
    lr = lr,
    device = device
  )

  # Create trainer
  trainer <- new(.resolve_module$Trainer, model, train_config)

  # Prepare hash embedding (required for hash mode)
  hash_emb <- if (!is.null(dataset$hash_embedding)) {
    dataset$hash_embedding
  } else {
    matrix(0, nrow = nrow(dataset$coordinates), ncol = 1)
  }

  # Prepare data
  trainer$prepare_data(
    coordinates = dataset$coordinates,
    covariates = dataset$covariates,
    hash_embedding = hash_emb,
    species_ids = if (!is.null(dataset$species_ids)) dataset$species_ids else NULL,
    species_vector = if (!is.null(dataset$species_vector)) dataset$species_vector else NULL,
    genus_ids = if (!is.null(dataset$genus_ids)) as.integer(dataset$genus_ids) else NULL,
    family_ids = if (!is.null(dataset$family_ids)) as.integer(dataset$family_ids) else NULL,
    unknown_fraction = dataset$unknown_fraction,
    unknown_count = dataset$unknown_count,
    targets = dataset$targets,
    test_size = test_size,
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
    for (target_name in names(result$final_metrics)) {
      metrics <- result$final_metrics[[target_name]]
      cat(sprintf("  %s: ", target_name))
      metric_strs <- sapply(names(metrics), function(m) {
        sprintf("%s=%.4f", m, metrics[[m]])
      })
      cat(paste(metric_strs, collapse = ", "), "\n")
    }
  }

  # Save if requested
  if (!is.null(save_path)) {
    trainer$save(save_path)
    if (verbose) {
      cat(sprintf("Model saved to: %s\n", save_path))
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
#' @param model A trained model (from resolve_train() or resolve_load())
#' @param dataset A resolve_dataset to predict on
#' @param return_latent Return latent representations (default FALSE)
#' @param output_space Output space for regression predictions:
#'   "raw" (default): inverse-transform predictions to original scale
#'   "transformed": keep predictions in transformed space (e.g., log1p)
#' @param confidence_threshold Minimum confidence for predictions (0-1).
#'   Predictions below threshold are set to NA. Default 0 keeps all predictions.
#'   Confidence is based on 1 - unknown_fraction (species coverage).
#'
#' @return Named list of prediction arrays, plus 'confidence' for each target
#'
#' @examples
#' \dontrun{
#' preds <- resolve_predict(trained_model, new_dataset)
#' preds <- resolve_predict(trained_model, new_dataset, confidence_threshold = 0.5)
#' }
#'
#' @export
resolve_predict <- function(model,
                            dataset,
                            return_latent = FALSE,
                            output_space = "raw",
                            confidence_threshold = 0.0) {
  # Validate output_space
  if (!output_space %in% c("raw", "transformed")) {
    stop("output_space must be 'raw' or 'transformed'")
  }

  # Handle both trainer objects and predictor objects
  if (inherits(model, "Rcpp_Predictor") || inherits(model, "Rcpp_RPredictor")) {
    predictor <- model
    hash_emb <- if (!is.null(dataset$hash_embedding)) {
      dataset$hash_embedding
    } else {
      matrix(0, nrow = nrow(dataset$coordinates), ncol = 1)
    }

    result <- predictor$predict(
      coordinates = dataset$coordinates,
      covariates = dataset$covariates,
      hash_embedding = hash_emb,
      genus_ids = if (!is.null(dataset$genus_ids)) as.integer(dataset$genus_ids) else NULL,
      family_ids = if (!is.null(dataset$family_ids)) as.integer(dataset$family_ids) else NULL,
      return_latent = return_latent
    )

    # Compute confidence from unknown_fraction (1 - unknown_fraction)
    if (!is.null(dataset$unknown_fraction)) {
      confidence <- 1.0 - dataset$unknown_fraction
    } else {
      # If no unknown tracking, assume full confidence
      confidence <- rep(1.0, nrow(dataset$coordinates))
    }

    # Post-process each target
    target_names <- names(dataset$schema$targets)
    for (name in target_names) {
      if (!is.null(result[[name]])) {
        cfg <- dataset$schema$targets[[name]]

        # Apply inverse transform for log1p targets if output_space == "raw"
        if (!is.null(cfg$transform) && cfg$transform == "log1p" && output_space == "raw") {
          result[[name]] <- expm1(result[[name]])
        }

        # Apply confidence threshold
        if (confidence_threshold > 0) {
          result[[name]][confidence < confidence_threshold] <- NA
        }

        # Add confidence for this target
        result[[paste0(name, "_confidence")]] <- confidence
      }
    }

    result$confidence <- confidence
    result

  } else if (is.list(model) && !is.null(model$trainer)) {
    # Trainer from resolve_train() - use temp file workaround
    temp_path <- tempfile(fileext = ".pt")
    on.exit(unlink(temp_path), add = TRUE)
    model$trainer$save(temp_path)
    predictor <- resolve_load(temp_path)

    # Recurse with the predictor
    resolve_predict(predictor, dataset, return_latent, output_space, confidence_threshold)
  } else {
    stop("model must be a Predictor object or result from resolve_train()")
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
#' predictor <- resolve_load("model.pt")
#' preds <- resolve_predict(predictor, new_data)
#' }
#'
#' @export
resolve_load <- function(path, device = "cpu") {
  .resolve_module$Predictor_load(path, device)
}


#' Save a Trained RESOLVE Model
#'
#' Save model checkpoint.
#'
#' @param trainer A trained Trainer object (from resolve_train())
#' @param path Path to save checkpoint
#'
#' @export
resolve_save <- function(trainer, path) {
  if (is.list(trainer) && !is.null(trainer$trainer)) {
    trainer$trainer$save(path)
  } else if (inherits(trainer, "Rcpp_Trainer") || inherits(trainer, "Rcpp_RTrainer")) {
    trainer$save(path)
  } else {
    stop("trainer must be a Trainer object or result from resolve_train()")
  }
}


#' Check Training Progress
#'
#' Read progress from a checkpoint directory.
#'
#' @param checkpoint_dir Path to checkpoint directory
#'
#' @return A list with progress information, or NULL if no checkpoint exists
#'
#' @examples
#' \dontrun{
#' progress <- resolve_progress("checkpoints/my_model")
#' if (!is.null(progress)) {
#'   cat(sprintf("Epoch %d/%d (%.1f%%)\n",
#'     progress$epoch, progress$max_epochs,
#'     progress$progress_pct))
#' }
#' }
#'
#' @export
resolve_progress <- function(checkpoint_dir) {
  progress_file <- file.path(checkpoint_dir, "progress.json")
  if (!file.exists(progress_file)) {
    return(NULL)
  }
  jsonlite::fromJSON(progress_file)
}
