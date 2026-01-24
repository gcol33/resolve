#' @importFrom Rcpp sourceCpp
#' @importFrom R6 R6Class
#' @useDynLib spacc, .registration = TRUE
NULL

#' TaxonomyVocab Class
#'
#' Vocabulary for encoding genus and family names to integer IDs.
#'
#' @export
TaxonomyVocab <- R6::R6Class(
    "TaxonomyVocab",
    public = list(
        #' @description Create a new TaxonomyVocab
        initialize = function() {
            private$.ptr <- cpp_taxonomy_vocab_new()
        },

        #' @description Fit vocabulary on genera and families
        #' @param genera Character vector of genus names
        #' @param families Character vector of family names
        fit = function(genera, families) {
            cpp_taxonomy_vocab_fit(private$.ptr, genera, families)
            invisible(self)
        },

        #' @description Encode a genus name to ID
        #' @param genus Genus name
        #' @return Integer ID (0 for unknown)
        encode_genus = function(genus) {
            cpp_taxonomy_vocab_encode_genus(private$.ptr, genus)
        },

        #' @description Encode a family name to ID
        #' @param family Family name
        #' @return Integer ID (0 for unknown)
        encode_family = function(family) {
            cpp_taxonomy_vocab_encode_family(private$.ptr, family)
        },

        #' @description Number of unique genera in vocabulary
        n_genera = function() {
            cpp_taxonomy_vocab_n_genera(private$.ptr)
        },

        #' @description Number of unique families in vocabulary
        n_families = function() {
            cpp_taxonomy_vocab_n_families(private$.ptr)
        }
    ),
    private = list(
        .ptr = NULL
    )
)

#' SpeciesEncoder Class
#'
#' Encode species composition data using feature hashing and taxonomy.
#'
#' @export
SpeciesEncoder <- R6::R6Class(
    "SpeciesEncoder",
    public = list(
        #' @description Create a new SpeciesEncoder
        #' @param hash_dim Dimension of feature hash embedding (default 32)
        #' @param top_k Number of top taxa to track (default 3)
        initialize = function(hash_dim = 32, top_k = 3) {
            private$.ptr <- cpp_species_encoder_new(hash_dim, top_k)
            private$.hash_dim <- hash_dim
            private$.top_k <- top_k
        },

        #' @description Fit encoder on species data
        #' @param species_data List of data frames, each with columns:
        #'   species, genus, family, abundance
        fit = function(species_data) {
            cpp_species_encoder_fit(private$.ptr, species_data)
            private$.fitted <- TRUE
            invisible(self)
        },

        #' @description Transform species data to encoded features
        #' @param species_data List of data frames (same format as fit)
        #' @return List with hash_embedding, genus_ids, family_ids matrices
        transform = function(species_data) {
            if (!private$.fitted) {
                stop("Encoder must be fitted before transform")
            }
            cpp_species_encoder_transform(private$.ptr, species_data)
        },

        #' @description Fit and transform in one step
        #' @param species_data List of data frames
        #' @return List with encoded features
        fit_transform = function(species_data) {
            self$fit(species_data)
            self$transform(species_data)
        }
    ),
    active = list(
        #' @field hash_dim Hash embedding dimension
        hash_dim = function() private$.hash_dim,
        #' @field top_k Number of top taxa tracked
        top_k = function() private$.top_k,
        #' @field is_fitted Whether encoder has been fitted
        is_fitted = function() private$.fitted
    ),
    private = list(
        .ptr = NULL,
        .hash_dim = 32,
        .top_k = 3,
        .fitted = FALSE
    )
)

#' SpaccModel Class
#'
#' Multi-task neural network for predicting plot attributes from species composition.
#'
#' @export
SpaccModel <- R6::R6Class(
    "SpaccModel",
    public = list(
        #' @description Create a new SpaccModel
        #' @param schema List describing the data schema (n_plots, n_species, targets, etc.)
        #' @param config List of model configuration (hash_dim, hidden_dims, etc.)
        initialize = function(schema, config = list()) {
            private$.ptr <- cpp_spacc_model_new(schema, config)
            private$.schema <- schema
            private$.config <- config
        },

        #' @description Forward pass through model
        #' @param continuous Numeric matrix of continuous features (n_plots x n_features)
        #' @param genus_ids Integer matrix of genus IDs (optional)
        #' @param family_ids Integer matrix of family IDs (optional)
        #' @return List of predictions for each target
        forward = function(continuous, genus_ids = NULL, family_ids = NULL) {
            cpp_spacc_model_forward(private$.ptr, continuous, genus_ids, family_ids)
        },

        #' @description Get latent representations
        #' @param continuous Numeric matrix of continuous features
        #' @param genus_ids Integer matrix of genus IDs (optional)
        #' @param family_ids Integer matrix of family IDs (optional)
        #' @return Numeric matrix of latent embeddings
        get_latent = function(continuous, genus_ids = NULL, family_ids = NULL) {
            cpp_spacc_model_get_latent(private$.ptr, continuous, genus_ids, family_ids)
        }
    ),
    active = list(
        #' @field schema Data schema
        schema = function() private$.schema,
        #' @field config Model configuration
        config = function() private$.config
    ),
    private = list(
        .ptr = NULL,
        .schema = NULL,
        .config = NULL
    )
)

#' Compute Spacc Metrics
#'
#' @param pred Numeric vector of predictions
#' @param target Numeric vector of targets
#' @param threshold Band accuracy threshold (default 0.25)
#' @return Named list of metrics
#' @export
spacc_metrics <- function(pred, target, threshold = 0.25) {
    list(
        mae = cpp_mae(pred, target),
        rmse = cpp_rmse(pred, target),
        smape = cpp_smape(pred, target),
        band_accuracy = cpp_band_accuracy(pred, target, threshold)
    )
}
