#' @importFrom Rcpp sourceCpp
#' @importFrom R6 R6Class
#' @useDynLib resolve, .registration = TRUE
NULL

#' ResolveModel R6 Class
#'
#' Multi-task neural network for predicting plot attributes from species composition.
#'
#' @export
ResolveModel <- R6::R6Class(
    "ResolveModel",
    public = list(
        #' @description Create a new ResolveModel
        #' @param schema List describing the data schema
        #' @param config List of model configuration (optional)
        initialize = function(schema, config = list()) {
            private$.ptr <- resolve_model_new(schema, config)
            private$.schema <- schema
            private$.config <- config
        },

        #' @description Forward pass through model
        #' @param continuous Numeric matrix of continuous features
        #' @param genus_ids Integer matrix of genus IDs (optional)
        #' @param family_ids Integer matrix of family IDs (optional)
        #' @return List of predictions for each target
        forward = function(continuous, genus_ids = NULL, family_ids = NULL) {
            resolve_model_forward(private$.ptr, continuous, genus_ids, family_ids)
        },

        #' @description Get latent representations
        #' @param continuous Numeric matrix of continuous features
        #' @param genus_ids Integer matrix of genus IDs (optional)
        #' @param family_ids Integer matrix of family IDs (optional)
        #' @return Numeric matrix of latent embeddings
        get_latent = function(continuous, genus_ids = NULL, family_ids = NULL) {
            resolve_model_get_latent(private$.ptr, continuous, genus_ids, family_ids)
        }
    ),
    active = list(
        #' @field schema Data schema
        schema = function() private$.schema,
        #' @field config Model configuration
        config = function() private$.config,
        #' @field ptr External pointer (for internal use)
        ptr = function() private$.ptr
    ),
    private = list(
        .ptr = NULL,
        .schema = NULL,
        .config = NULL
    )
)

#' ResolveTrainer R6 Class
#'
#' Trainer for ResolveModel with automatic data scaling and early stopping.
#'
#' @export
ResolveTrainer <- R6::R6Class(
    "ResolveTrainer",
    public = list(
        #' @description Create a new Trainer
        #' @param model ResolveModel instance
        #' @param config List of training config (batch_size, max_epochs, patience, lr, weight_decay)
        initialize = function(model, config = list()) {
            if (!inherits(model, "ResolveModel")) {
                stop("model must be a ResolveModel instance")
            }
            private$.ptr <- resolve_trainer_new(model$ptr, config)
            private$.model <- model
            private$.config <- config
        },

        #' @description Prepare data for training
        #' @param continuous Numeric matrix of continuous features
        #' @param targets Named list of numeric vectors (target values)
        #' @param genus_ids Integer matrix of genus IDs (optional)
        #' @param family_ids Integer matrix of family IDs (optional)
        #' @param test_size Fraction for test set (default 0.2)
        #' @param seed Random seed (default 42)
        prepare_data = function(continuous, targets, genus_ids = NULL,
                                family_ids = NULL, test_size = 0.2, seed = 42) {
            resolve_trainer_prepare_data(private$.ptr, continuous, targets,
                                         genus_ids, family_ids, test_size, seed)
            invisible(self)
        },

        #' @description Train the model
        #' @return List with training results (best_epoch, final_metrics, loss histories)
        fit = function() {
            result <- resolve_trainer_fit(private$.ptr)
            private$.result <- result
            result
        },

        #' @description Save model checkpoint
        #' @param path File path to save
        save = function(path) {
            resolve_trainer_save(private$.ptr, path)
            invisible(self)
        }
    ),
    active = list(
        #' @field model The ResolveModel being trained
        model = function() private$.model,
        #' @field config Training configuration
        config = function() private$.config,
        #' @field result Training result (after fit)
        result = function() private$.result
    ),
    private = list(
        .ptr = NULL,
        .model = NULL,
        .config = NULL,
        .result = NULL
    )
)

#' Load trained model from checkpoint
#'
#' @param path File path to checkpoint
#' @return List with model (ResolveModel) and scalers
#' @export
resolve_load_checkpoint <- function(path) {
    result <- resolve_trainer_load(path)
    # Note: model and scalers are external pointers
    result
}

#' ResolvePredictor R6 Class
#'
#' Predictor for inference with trained ResolveModel.
#'
#' @export
ResolvePredictor <- R6::R6Class(
    "ResolvePredictor",
    public = list(
        #' @description Create a new Predictor
        #' @param model_ptr External pointer to ResolveModel
        #' @param scalers_ptr External pointer to Scalers
        initialize = function(model_ptr, scalers_ptr) {
            private$.ptr <- resolve_predictor_new(model_ptr, scalers_ptr)
        },

        #' @description Predict on new data
        #' @param continuous Numeric matrix of continuous features
        #' @param genus_ids Integer matrix of genus IDs (optional)
        #' @param family_ids Integer matrix of family IDs (optional)
        #' @param return_latent Whether to return latent embeddings
        #' @return List with predictions (and optionally latent)
        predict = function(continuous, genus_ids = NULL, family_ids = NULL,
                           return_latent = FALSE) {
            resolve_predictor_predict(private$.ptr, continuous, genus_ids,
                                      family_ids, return_latent)
        },

        #' @description Get latent embeddings
        #' @param continuous Numeric matrix of continuous features
        #' @param genus_ids Integer matrix of genus IDs (optional)
        #' @param family_ids Integer matrix of family IDs (optional)
        #' @return Numeric matrix of latent embeddings
        get_embeddings = function(continuous, genus_ids = NULL, family_ids = NULL) {
            resolve_predictor_get_embeddings(private$.ptr, continuous,
                                             genus_ids, family_ids)
        },

        #' @description Get learned genus embeddings
        #' @return Numeric matrix of genus embeddings
        get_genus_embeddings = function() {
            resolve_predictor_get_genus_embeddings(private$.ptr)
        },

        #' @description Get learned family embeddings
        #' @return Numeric matrix of family embeddings
        get_family_embeddings = function() {
            resolve_predictor_get_family_embeddings(private$.ptr)
        }
    ),
    private = list(
        .ptr = NULL
    )
)

#' Load predictor from checkpoint
#'
#' @param path File path to checkpoint
#' @return ResolvePredictor instance
#' @export
resolve_load_predictor <- function(path) {
    ptr <- resolve_predictor_load(path)
    # Create a minimal predictor wrapper
    predictor <- list(ptr = ptr)
    class(predictor) <- "ResolvePredictor_raw"

    # Add methods
    predictor$predict <- function(continuous, genus_ids = NULL, family_ids = NULL,
                                  return_latent = FALSE) {
        resolve_predictor_predict(ptr, continuous, genus_ids, family_ids, return_latent)
    }
    predictor$get_embeddings <- function(continuous, genus_ids = NULL, family_ids = NULL) {
        resolve_predictor_get_embeddings(ptr, continuous, genus_ids, family_ids)
    }
    predictor$get_genus_embeddings <- function() {
        resolve_predictor_get_genus_embeddings(ptr)
    }
    predictor$get_family_embeddings <- function() {
        resolve_predictor_get_family_embeddings(ptr)
    }

    predictor
}

#' Compute Resolve Metrics
#'
#' Convenience function to compute all metrics at once.
#'
#' @param pred Numeric vector of predictions
#' @param target Numeric vector of targets
#' @param threshold Band accuracy threshold (default 0.25)
#' @return Named list of metrics (mae, rmse, smape, band_accuracy)
#' @export
resolve_metrics <- function(pred, target, threshold = 0.25) {
    list(
        mae = resolve_mae(pred, target),
        rmse = resolve_rmse(pred, target),
        smape = resolve_smape(pred, target),
        band_accuracy = resolve_band_accuracy(pred, target, threshold)
    )
}

#' Check CUDA Availability
#'
#' @return TRUE if CUDA is available for GPU acceleration
#' @export
cuda_available <- function() {
    resolve_cuda_available()
}

#' Get CUDA Device Count
#'
#' @return Number of available CUDA devices
#' @export
cuda_device_count <- function() {
    resolve_cuda_device_count()
}
