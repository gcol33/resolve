#' @title RESOLVE Predictor
#' @description
#' R6 class for inference with trained RESOLVE models.
#'
#' Loads a trained checkpoint and provides prediction methods.
#'
#' @export
ResolvePredictor <- R6::R6Class(
  "ResolvePredictor",

  public = list(

    #' @description
    #' Create a new ResolvePredictor and load checkpoint.
    #'
    #' @param checkpoint_path Path to the saved checkpoint (.pt file)
    #' @param device Device to use for inference ("cpu", "cuda", "cuda:0", etc.)
    #' @return A new ResolvePredictor object
    initialize = function(checkpoint_path, device = "cpu") {
      private$.ptr <- predictor_load(checkpoint_path, device)
      private$.checkpoint_path <- checkpoint_path
      private$.device <- device
    },

    #' @description
    #' Check if model is loaded.
    #' @return Logical indicating if model is ready
    is_loaded = function() {
      predictor_is_loaded(private$.ptr)
    },

    #' @description
    #' Get target configurations.
    #' @return List of target configurations
    target_configs = function() {
      predictor_target_configs(private$.ptr)
    },

    #' @description
    #' Predict from species data.
    #'
    #' @param species_df Data frame with species occurrences
    #' @param species_col Column name for species IDs
    #' @param plot_col Column name for plot IDs
    #' @param plot_ids Vector of plot IDs (in order of desired output)
    #' @param continuous Continuous features matrix (n_plots x n_continuous)
    #' @param genus_col Column name for genus (optional)
    #' @param family_col Column name for family (optional)
    #' @param abundance_col Column name for abundance (optional)
    #' @param return_latent Whether to return latent representations
    #' @return List with predictions for each target
    predict = function(species_df, species_col, plot_col, plot_ids, continuous,
                       genus_col = NULL, family_col = NULL, abundance_col = NULL,
                       return_latent = FALSE) {
      predictor_predict(
        private$.ptr,
        species_df,
        species_col,
        plot_col,
        plot_ids,
        as.matrix(continuous),
        genus_col %||% "",
        family_col %||% "",
        abundance_col %||% "",
        return_latent
      )
    },

    #' @description
    #' Print predictor summary.
    print = function() {
      cat("ResolvePredictor\n")
      cat(sprintf("  checkpoint: %s\n", private$.checkpoint_path))
      cat(sprintf("  device: %s\n", private$.device))
      cat(sprintf("  loaded: %s\n", self$is_loaded()))

      if (self$is_loaded()) {
        targets <- self$target_configs()
        cat(sprintf("  targets: %d\n", length(targets)))
        for (tc in targets) {
          cat(sprintf("    - %s (%s)\n", tc$name, tc$task))
        }
      }

      invisible(self)
    }
  ),

  private = list(
    .ptr = NULL,
    .checkpoint_path = NULL,
    .device = NULL
  )
)

#' @title Load RESOLVE Predictor
#' @description
#' Convenience function to load a trained RESOLVE model for inference.
#'
#' @param checkpoint_path Path to the saved checkpoint
#' @param device Device to use ("cpu", "cuda", etc.)
#' @return ResolvePredictor object
#' @export
load_resolve <- function(checkpoint_path, device = "cpu") {
  ResolvePredictor$new(checkpoint_path, device)
}

# Helper for NULL default
`%||%` <- function(x, y) if (is.null(x)) y else x
