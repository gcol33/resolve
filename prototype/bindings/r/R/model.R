#' @title RESOLVE Model
#' @description
#' R6 class for the RESOLVE neural network model.
#'
#' @export
ResolveModel <- R6::R6Class(
  "ResolveModel",

  public = list(

    #' @description
    #' Create a new ResolveModel.
    #'
    #' @param target_configs List of target configurations
    #' @param model_config List of model configuration options
    #' @return A new ResolveModel object
    initialize = function(target_configs, model_config = list()) {
      private$.ptr <- resolve_model_new(target_configs, model_config)
      private$.target_configs <- target_configs
      private$.model_config <- model_config
    },

    #' @description
    #' Forward pass through the model.
    #'
    #' @param continuous Continuous features matrix (n_samples x n_continuous)
    #' @param genus_ids Genus ID matrix (optional)
    #' @param family_ids Family ID matrix (optional)
    #' @param species_ids Species ID matrix (optional, for embed mode)
    #' @param species_vector Species vector matrix (optional, for sparse mode)
    #' @return List with predictions for each target
    forward = function(continuous, genus_ids = NULL, family_ids = NULL,
                       species_ids = NULL, species_vector = NULL) {
      resolve_model_forward(
        private$.ptr,
        as.matrix(continuous),
        genus_ids,
        family_ids,
        species_ids,
        species_vector
      )
    },

    #' @description
    #' Get latent representation.
    #'
    #' @param continuous Continuous features matrix
    #' @param genus_ids Genus ID matrix (optional)
    #' @param family_ids Family ID matrix (optional)
    #' @param species_ids Species ID matrix (optional)
    #' @param species_vector Species vector matrix (optional)
    #' @return Latent representation matrix
    get_latent = function(continuous, genus_ids = NULL, family_ids = NULL,
                          species_ids = NULL, species_vector = NULL) {
      resolve_model_get_latent(
        private$.ptr,
        as.matrix(continuous),
        genus_ids,
        family_ids,
        species_ids,
        species_vector
      )
    },

    #' @description
    #' Set model to training mode.
    train = function() {
      resolve_model_train(private$.ptr)
      invisible(self)
    },

    #' @description
    #' Set model to evaluation mode.
    eval = function() {
      resolve_model_eval(private$.ptr)
      invisible(self)
    },

    #' @description
    #' Move model to device.
    #' @param device Device string ("cpu", "cuda", "cuda:0", etc.)
    to = function(device) {
      resolve_model_to(private$.ptr, device)
      invisible(self)
    },

    #' @description
    #' Save model to file.
    #' @param path File path
    save = function(path) {
      resolve_model_save(private$.ptr, path)
      invisible(self)
    },

    #' @description
    #' Load model from file.
    #' @param path File path
    load = function(path) {
      resolve_model_load(private$.ptr, path)
      invisible(self)
    },

    #' @description
    #' Print model summary.
    print = function() {
      cat("ResolveModel\n")
      cat(sprintf("  encoder_dim: %d\n", private$.model_config$encoder_dim %||% 256L))
      cat(sprintf("  hidden_dim: %d\n", private$.model_config$hidden_dim %||% 512L))
      cat(sprintf("  n_encoder_layers: %d\n", private$.model_config$n_encoder_layers %||% 3L))
      cat(sprintf("  dropout: %.2f\n", private$.model_config$dropout %||% 0.1))
      cat(sprintf("  mode: %s\n", private$.model_config$mode %||% "hash"))
      cat(sprintf("  targets: %d\n", length(private$.target_configs)))
      for (tc in private$.target_configs) {
        cat(sprintf("    - %s (%s)\n", tc$name, tc$task))
      }
      invisible(self)
    }
  ),

  active = list(
    #' @field ptr Internal pointer
    ptr = function() private$.ptr,

    #' @field target_configs Target configurations
    target_configs = function() private$.target_configs,

    #' @field model_config Model configuration
    model_config = function() private$.model_config
  ),

  private = list(
    .ptr = NULL,
    .target_configs = NULL,
    .model_config = NULL
  )
)

# Helper for NULL default
`%||%` <- function(x, y) if (is.null(x)) y else x
