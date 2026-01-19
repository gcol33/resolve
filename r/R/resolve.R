#' Create a RESOLVE Dataset
#'
#' Load and configure a dataset for species composition prediction.
#'
#' @param header Path to plot-level CSV (one row per plot)
#' @param species Path to species CSV (one row per species-plot occurrence)
#' @param roles Named list mapping semantic roles to column names
#' @param targets Named list of target configurations
#' @param species_normalization Normalization mode: "raw", "norm", or "log1p"
#' @param track_unknown_fraction Track fraction of unknown species (default TRUE)
#' @param track_unknown_count Track count of unknown species (default FALSE
#'
#' @return A ResolveDataset object
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
                            track_unknown_count = FALSE) {
  .resolve$ResolveDataset$from_csv(
    header = header,
    species = species,
    roles = roles,
    targets = targets,
    species_normalization = species_normalization,
    track_unknown_fraction = track_unknown_fraction,
    track_unknown_count = track_unknown_count
  )
}


#' Train a RESOLVE Model
#'
#' Train a model on a dataset with sensible defaults.
#'
#' @param dataset A ResolveDataset object
#' @param hash_dim Dimension of species hash embedding (default 32)
#' @param top_k Number of top genera/families to track (default 5)
#' @param hidden_dims Hidden layer dimensions (default c(256, 128, 64))
#' @param max_epochs Maximum training epochs (default 500)
#' @param patience Early stopping patience (default 50)
#' @param lr Learning rate (default 0.001)
#' @param batch_size Batch size (default 512)
#' @param device Device: "auto", "cpu", or "cuda"
#' @param checkpoint_path Path to save model checkpoint (optional)
#' @param verbose Print training progress
#'
#' @return A trained Trainer object
#'
#' @examples
#' \dontrun{
#' trainer <- resolve_train(dataset)
#' trainer <- resolve_train(dataset, max_epochs = 100, checkpoint_path = "model.pt")
#' }
#'
#' @export
resolve_train <- function(dataset,
                          hash_dim = 32L,
                          top_k = 5L,
                          hidden_dims = NULL,
                          max_epochs = 500L,
                          patience = 50L,
                          lr = 1e-3,
                          batch_size = 512L,
                          device = "auto",
                          checkpoint_path = NULL,
                          verbose = TRUE) {
  # Convert R vector to Python list if provided
  if (!is.null(hidden_dims)) {
    hidden_dims <- as.list(as.integer(hidden_dims))
  }

  trainer <- .resolve$Trainer(
    dataset = dataset,
    hash_dim = as.integer(hash_dim),
    top_k = as.integer(top_k),
    hidden_dims = hidden_dims,
    max_epochs = as.integer(max_epochs),
    patience = as.integer(patience),
    lr = lr,
    batch_size = as.integer(batch_size),
    device = device
  )

  trainer$fit()

  if (!is.null(checkpoint_path)) {
    trainer$save(checkpoint_path)
  }

  trainer
}


#' Predict with a RESOLVE Model
#'
#' Make predictions on a dataset using a trained model.
#'
#' @param trainer A trained Trainer object (or loaded Predictor)
#' @param dataset A ResolveDataset to predict on
#' @param output_space Output space: "raw" or "transformed"
#' @param confidence_threshold Minimum confidence for predictions (0-1).
#'   Predictions below threshold are set to NA.
#'   Default 0 means all predictions are kept (gap-fill everything).
#'
#' @return Named list of prediction arrays
#'
#' @examples
#' \dontrun{
#' # Gap-fill all predictions
#' preds <- resolve_predict(trainer, dataset)
#'
#' # Only keep predictions with >= 80% confidence
#' preds <- resolve_predict(trainer, dataset, confidence_threshold = 0.8)
#' }
#'
#' @export
resolve_predict <- function(trainer, dataset, output_space = "raw", confidence_threshold = 0) {
  trainer$predict(dataset, output_space = output_space, confidence_threshold = confidence_threshold)
}


#' Load a Trained RESOLVE Model
#'
#' Load a model from a saved checkpoint.
#'
#' @param path Path to checkpoint file
#' @param device Device: "auto", "cpu", or "cuda"
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
resolve_load <- function(path, device = "auto") {
  .resolve$Predictor$load(path, device = device)
}


#' Save a Trained RESOLVE Model
#'
#' Save model checkpoint.
#'
#' @param trainer A trained Trainer object
#' @param path Path to save checkpoint
#'
#' @export
resolve_save <- function(trainer, path) {
  trainer$save(path)
}
