#' @title Fit a RESOLVE Model
#' @description
#' Fit a deep learning model for prediction from relational data.
#'
#' **RESOLVE**: Relational Encoding via Structured Observation Learning
#' with Vector Embeddings
#'
#' The model combines features from a primary data frame with aggregated
#' features from an observation-level data frame (many-to-one relationship).
#'
#' @param formula A formula specifying the model structure.
#'
#'   **Response variables:**
#'   \itemize{
#'     \item Single target: \code{y ~ ...}
#'     \item Multiple targets: \code{cbind(y1, y2) ~ ...}
#'   }
#'
#'   **Numeric features (from data, auto-scaled):**
#'   \itemize{
#'     \item Bare variable names: \code{~ elevation + latitude + ...}
#'     \item Use \code{raw()} to skip scaling: \code{~ raw(prescaled_var) + ...}
#'   }
#'
#'   **Categorical encodings (from obs):**
#'   \itemize{
#'     \item \code{hash(var)} - feature hashing for high-cardinality categoricals
#'     \item \code{hash(var, dim = 64)} - custom hash dimension
#'     \item \code{hash(var, top = 5, by = "weight")} - top 5 by weight column
#'     \item \code{hash(var, top = 3, bottom = 3, by = "weight")} - top + bottom
#'     \item \code{embed(var1, var2)} - learned embeddings
#'     \item \code{embed(var, dim = 16)} - custom embedding dimension
#'     \item \code{onehot(var)} - one-hot encoding for small categoricals
#'   }
#'
#' @param data A data frame with one row per unit. Contains response variable(s)
#'   and any unit-level numeric features.
#' @param obs A data frame with multiple rows per unit (observations/events/items).
#'   Contains categorical variables for \code{hash()}/\code{embed()}/\code{onehot()}
#'   and optional weighting columns.
#' @param by Column name linking \code{data} and \code{obs} (the join key).
#'   Must exist in both data frames.
#' @param transform Named list of transformations for response variables.
#'   Options: "none", "log1p", "sqrt". E.g., \code{list(y = "log1p")}
#' @param epochs Number of training epochs (default: 100)
#' @param batch_size Training batch size (default: 32)
#' @param learning_rate Learning rate (default: 0.001)
#' @param validation_split Fraction of data for validation (default: 0.1)
#' @param early_stopping Stop if validation loss doesn't improve for this many
#'   epochs (default: 10, set to 0 to disable)
#' @param verbose Print training progress (default: TRUE)
#' @param ... Additional arguments passed to model configuration
#'
#' @return An object of class "resolve" containing the fitted model
#'
#' @examples
#' \dontrun{
#' # Ecology: predict soil pH from species composition
#' fit <- resolve(
#'   ph ~ elevation + latitude + hash(species, top = 5, by = "cover") + embed(genus),
#'   data = plots,
#'   obs = species_obs,
#'   by = "plot_id"
#' )
#'
#' # E-commerce: predict customer LTV from purchase history
#' fit <- resolve(
#'   ltv ~ age + income + hash(product_id, top = 10, by = "price") + embed(category),
#'   data = customers,
#'   obs = purchases,
#'   by = "customer_id"
#' )
#'
#' # Healthcare: predict readmission from patient encounters
#' fit <- resolve(
#'   readmit ~ age + bmi + hash(diagnosis, top = 5) + embed(drug_class),
#'   data = patients,
#'   obs = encounters,
#'   by = "patient_id"
#' )
#'
#' # Multiple targets with transformations
#' fit <- resolve(
#'   cbind(ph, nitrogen) ~ elevation + hash(species) + embed(genus),
#'   data = plots,
#'   obs = species_obs,
#'   by = "plot_id",
#'   transform = list(nitrogen = "log1p")
#' )
#'
#' # Predictions
#' pred <- predict(fit, newdata = new_plots, obs = new_species_obs)
#' }
#'
#' @export
resolve <- function(
    formula,
    data,
    obs,
    by,
    transform = list(),
    epochs = 100L,
    batch_size = 32L,
    learning_rate = 0.001,
    validation_split = 0.1,
    early_stopping = 10L,
    verbose = TRUE,
    ...
) {
  # Validate join key

if (missing(by)) {
    stop("'by' argument is required. Specify the column linking data and obs.")
  }
  if (!by %in% names(data)) {
    stop(sprintf("Column '%s' not found in data", by))
  }
  if (!by %in% names(obs)) {
    stop(sprintf("Column '%s' not found in obs", by))
  }

  # Parse formula
  formula_parts <- parse_resolve_formula(formula, data, obs)
  response_vars <- formula_parts$response
  plot_numeric <- formula_parts$plot_numeric
  encoding_specs <- formula_parts$encodings

  # Check response variables exist in data
 for (rv in response_vars) {
    if (!rv %in% names(data)) {
      stop(sprintf("Response variable '%s' not found in data", rv))
    }
  }

  # Validate encoding columns exist in obs
  for (spec in encoding_specs) {
    for (var in spec$vars) {
      if (!var %in% names(obs)) {
        stop(sprintf("Variable '%s' used in %s() not found in obs", var, spec$type))
      }
    }
    # Validate 'by' column if specified
    if (!is.null(spec$by) && !spec$by %in% names(obs)) {
      stop(sprintf("Column '%s' specified in by = ... not found in obs", spec$by))
    }
  }

  # Get unit IDs
  unit_ids <- as.character(data[[by]])

  # Filter obs to only include units in data
  obs_filtered <- obs[obs[[by]] %in% unit_ids, ]

  if (verbose) {
    cat("RESOLVE: Fitting model\n")
    cat(sprintf("  Units: %d\n", length(unit_ids)))
    cat(sprintf("  Observations: %d\n", nrow(obs_filtered)))
    cat(sprintf("  Targets: %s\n", paste(response_vars, collapse = ", ")))

    if (length(plot_numeric) > 0) {
      cat(sprintf("  Numeric (auto-scaled): %s\n", paste(plot_numeric, collapse = ", ")))
    }

    if (length(encoding_specs) > 0) {
      cat("  Encodings:\n")
      for (spec in encoding_specs) {
        vars_str <- paste(spec$vars, collapse = ", ")
        extra_parts <- character(0)
        if (!is.null(spec$dim)) extra_parts <- c(extra_parts, sprintf("dim=%d", spec$dim))
        if (!is.null(spec$top)) extra_parts <- c(extra_parts, sprintf("top=%d", spec$top))
        if (!is.null(spec$bottom)) extra_parts <- c(extra_parts, sprintf("bottom=%d", spec$bottom))
        if (!is.null(spec$by)) extra_parts <- c(extra_parts, sprintf("by=%s", spec$by))
        extra_str <- if (length(extra_parts) > 0) paste0(", ", paste(extra_parts, collapse = ", ")) else ""
        cat(sprintf("    %s(%s%s)\n", spec$type, vars_str, extra_str))
      }
    }
  }

  # Build and fit PlotEncoder
  encoder <- build_plot_encoder(formula_parts, data, obs)

  # Determine which columns are categorical vs numeric in each data frame
  obs_categorical <- character(0)
  obs_numeric <- character(0)
  for (spec in encoding_specs) {
    obs_categorical <- c(obs_categorical, spec$vars)
    if (!is.null(spec$by)) {
      obs_numeric <- c(obs_numeric, spec$by)
    }
  }
  obs_categorical <- unique(obs_categorical)
  obs_numeric <- unique(obs_numeric)

  plot_categorical <- character(0)  # Bare categoricals would have errored in parsing

  # Fit encoder
  plot_encoder_fit(
    encoder,
    plot_df = data,
    plot_id_col = by,
    plot_categorical_cols = plot_categorical,
    plot_numeric_cols = plot_numeric,
    obs_df = obs_filtered,
    obs_plot_id_col = by,
    obs_categorical_cols = obs_categorical,
    obs_numeric_cols = obs_numeric
  )

  if (verbose) {
    info <- plot_encoder_info(encoder)
    cat(sprintf("  Continuous features: %d\n", info$continuous_dim))
    if (length(info$embeddings) > 0) {
      for (emb in info$embeddings) {
        cat(sprintf("  Embedding '%s': vocab=%d, dim=%d, slots=%d\n",
                    emb$name, emb$vocab_size, emb$dim, emb$n_slots))
      }
    }
  }

  # Transform data
  encoded <- plot_encoder_transform(
    encoder,
    plot_df = data,
    plot_id_col = by,
    plot_ids = unit_ids,
    plot_categorical_cols = plot_categorical,
    plot_numeric_cols = plot_numeric,
    obs_df = obs_filtered,
    obs_plot_id_col = by,
    obs_categorical_cols = obs_categorical,
    obs_numeric_cols = obs_numeric
  )

  # Prepare target configs
  target_configs <- lapply(response_vars, function(name) {
    trans <- transform[[name]] %||% "none"
    list(
      name = name,
      task = "regression",
      n_classes = 1L,
      transform = trans
    )
  })

  # Model configuration
  extra_args <- list(...)
  encoder_info <- plot_encoder_info(encoder)

  model_config <- list(
    encoder_dim = extra_args$encoder_dim %||% 128L,
    hidden_dim = extra_args$hidden_dim %||% 256L,
    n_encoder_layers = extra_args$n_encoder_layers %||% 3L,
    dropout = extra_args$dropout %||% 0.1,
    n_continuous = as.integer(encoder_info$continuous_dim),
    embeddings = encoder_info$embeddings
  )

  # Create model
  model <- ResolveModel$new(target_configs, model_config)

  # Training
  if (verbose) {
    cat(sprintf("  Training for %d epochs...\n", epochs))
  }

  # TODO: Implement actual training loop via C++ Trainer

  # Build result object
  result <- list(
    model = model,
    encoder = encoder,
    formula = formula,
    response_vars = response_vars,
    plot_numeric = plot_numeric,
    encoding_specs = encoding_specs,
    by = by,
    transform = transform,
    n_units = length(unit_ids),
    n_obs = nrow(obs_filtered),
    call = match.call()
  )

  class(result) <- "resolve"
  result
}


#' @title Predict Method for RESOLVE Models
#' @description
#' Generate predictions from a fitted RESOLVE model.
#'
#' @param object A fitted resolve model from \code{resolve()}
#' @param newdata Data frame with unit-level features for prediction
#' @param obs Data frame with observations for new units
#' @param interval Type of interval: "none", "confidence", or "prediction"
#' @param level Confidence level for intervals (default: 0.95)
#' @param n_samples Number of MC samples for uncertainty (default: 100)
#' @param ... Additional arguments (ignored)
#'
#' @return A data frame with predictions. If \code{interval != "none"},
#'   includes lower and upper bounds for each target.
#'
#' @export
predict.resolve <- function(
    object,
    newdata,
    obs,
    interval = c("none", "confidence", "prediction"),
    level = 0.95,
    n_samples = 100L,
    ...
) {
  interval <- match.arg(interval)

  by <- object$by

  # Get unit IDs
  unit_ids <- as.character(newdata[[by]])

  # Filter obs
  obs_filtered <- obs[obs[[by]] %in% unit_ids, ]

  # Determine columns (same logic as in resolve())
  obs_categorical <- character(0)
  obs_numeric <- character(0)
  for (spec in object$encoding_specs) {
    obs_categorical <- c(obs_categorical, spec$vars)
    if (!is.null(spec$by)) {
      obs_numeric <- c(obs_numeric, spec$by)
    }
  }
  obs_categorical <- unique(obs_categorical)
  obs_numeric <- unique(obs_numeric)
  plot_categorical <- character(0)

  # Transform using fitted encoder
  encoded <- plot_encoder_transform(
    object$encoder,
    plot_df = newdata,
    plot_id_col = by,
    plot_ids = unit_ids,
    plot_categorical_cols = plot_categorical,
    plot_numeric_cols = object$plot_numeric,
    obs_df = obs_filtered,
    obs_plot_id_col = by,
    obs_categorical_cols = obs_categorical,
    obs_numeric_cols = obs_numeric
  )

  # Set model to eval mode
  object$model$eval()

  # Get predictions
  outputs <- object$model$forward(
    encoded$continuous,
    embedding_ids = encoded$embedding_ids
  )

  # Build result data frame
  result <- data.frame(id = unit_ids)
  names(result)[1] <- by

  for (name in object$response_vars) {
    pred <- as.vector(outputs[[name]])

    # Inverse transform if needed
    trans <- object$transform[[name]] %||% "none"
    if (trans == "log1p") {
      pred <- expm1(pred)
    } else if (trans == "sqrt") {
      pred <- pred^2
    }

    result[[name]] <- pred
  }

  # TODO: Add interval computation via MC Dropout
  if (interval != "none") {
    warning("Confidence intervals not yet implemented, returning point estimates only")
  }

  result
}


#' @title Summary Method for RESOLVE Models
#' @export
summary.resolve <- function(object, ...) {
  cat("\nCall:\n")
  print(object$call)

  cat("\nModel:\n")
  cat(sprintf("  Units: %d\n", object$n_units))
  cat(sprintf("  Observations: %d\n", object$n_obs))
  cat(sprintf("  Join key: %s\n", object$by))

  cat("\nTargets:\n")
  for (name in object$response_vars) {
    trans <- object$transform[[name]] %||% "none"
    cat(sprintf("  %s (transform: %s)\n", name, trans))
  }

  cat("\nNumeric features (auto-scaled):\n")
  if (length(object$plot_numeric) > 0) {
    cat(sprintf("  %s\n", paste(object$plot_numeric, collapse = ", ")))
  } else {
    cat("  (none)\n")
  }

  if (length(object$encoding_specs) > 0) {
    cat("\nEncodings:\n")
    for (spec in object$encoding_specs) {
      vars_str <- paste(spec$vars, collapse = ", ")
      extra_parts <- character(0)
      if (!is.null(spec$dim)) extra_parts <- c(extra_parts, sprintf("dim=%d", spec$dim))
      if (!is.null(spec$top)) extra_parts <- c(extra_parts, sprintf("top=%d", spec$top))
      if (!is.null(spec$bottom)) extra_parts <- c(extra_parts, sprintf("bottom=%d", spec$bottom))
      extra_str <- if (length(extra_parts) > 0) paste0(", ", paste(extra_parts, collapse = ", ")) else ""
      cat(sprintf("  %s(%s%s)\n", spec$type, vars_str, extra_str))
    }
  }

  encoder_info <- plot_encoder_info(object$encoder)
  cat("\nEncoder:\n")
  cat(sprintf("  Continuous dim: %d\n", encoder_info$continuous_dim))
  if (length(encoder_info$embeddings) > 0) {
    for (emb in encoder_info$embeddings) {
      cat(sprintf("  %s: vocab=%d, dim=%d\n", emb$name, emb$vocab_size, emb$dim))
    }
  }

  invisible(object)
}


#' @title Print Method for RESOLVE Models
#' @export
print.resolve <- function(x, ...) {
  cat("RESOLVE model\n")
  cat(sprintf("  Targets: %s\n", paste(x$response_vars, collapse = ", ")))
  cat(sprintf("  Units: %d, Observations: %d\n", x$n_units, x$n_obs))
  if (length(x$embedding_specs) > 0) {
    emb_summary <- sapply(x$embedding_specs, function(s) {
      sprintf("%s(%s)", s$type, paste(s$vars, collapse=","))
    })
    cat(sprintf("  Embeddings: %s\n", paste(emb_summary, collapse = " + ")))
  }
  invisible(x)
}


# ==============================================================================
# Formula parsing
# ==============================================================================

#' Parse resolve formula with hash/embed/onehot/raw syntax
#'
#' Supports:
#' - Bare variable names: auto-scaled numeric (like elevation, latitude)
#' - raw(var): passthrough numeric without scaling
#' - hash(var, ...): feature hashing for high-cardinality categoricals
#' - embed(var, ...): learned embeddings for categoricals
#' - onehot(var): one-hot encoding for small categoricals
#'
#' Validates that categorical variables use explicit encoding (error if not).
#'
#' @keywords internal
parse_resolve_formula <- function(formula, data, obs) {
  # Get response side
  response_term <- formula[[2]]

  # Check for cbind() for multiple responses
  if (is.call(response_term) && as.character(response_term[[1]]) == "cbind") {
    response_vars <- as.character(response_term[-1])
  } else {
    response_vars <- as.character(response_term)
  }

  # Get predictor side as string
  predictor_term <- formula[[3]]
  formula_str <- deparse(predictor_term, width.cutoff = 500)

  # Extract encoding specifications: hash(...), embed(...), onehot(...), raw(...)
  encoding_specs <- list()

  # Pattern for encoding functions with optional parameters
  enc_pattern <- "(hash|embed|onehot|raw)\\s*\\(([^)]+)\\)"

  matches <- gregexpr(enc_pattern, formula_str, perl = TRUE)

  if (matches[[1]][1] != -1) {
    match_data <- regmatches(formula_str, matches)[[1]]

    for (match_str in match_data) {
      # Parse the encoding call
      spec <- parse_encoding_call(match_str)
      encoding_specs <- c(encoding_specs, list(spec))
    }
  }

  # Remove encoding terms from formula to get bare variable names (numeric)
  covariate_str <- formula_str
  covariate_str <- gsub("(hash|embed|onehot|raw)\\s*\\([^)]+\\)", "", covariate_str, perl = TRUE)
  covariate_str <- gsub("\\s*\\+\\s*\\+\\s*", " + ", covariate_str)  # fix double +
  covariate_str <- gsub("^\\s*\\+\\s*", "", covariate_str)  # leading +
  covariate_str <- gsub("\\s*\\+\\s*$", "", covariate_str)  # trailing +
  covariate_str <- trimws(covariate_str)

  # Parse remaining covariates (bare variable names = numeric, auto-scaled)
  numeric_vars <- character(0)
  if (covariate_str != "" && covariate_str != "1" && covariate_str != "0") {
    temp_formula <- tryCatch(
      as.formula(paste("~", covariate_str)),
      error = function(e) NULL
    )
    if (!is.null(temp_formula)) {
      bare_vars <- all.vars(temp_formula)

      # Validate: bare variables must be numeric, not categorical
      for (var in bare_vars) {
        # Check in data (bare variables must be in data, not obs)
        if (var %in% names(data)) {
          col <- data[[var]]
          if (is.character(col) || is.factor(col)) {
            stop(sprintf(
              "'%s' is character/factor but used as numeric variable.\n  Use embed(%s), hash(%s), or onehot(%s) for categorical data.",
              var, var, var, var
            ))
          }
          numeric_vars <- c(numeric_vars, var)
        }
        # If not found, will be caught later in validation
      }
    }
  }

  # All bare numeric variables come from data
  plot_numeric_vars <- numeric_vars
  obs_numeric_vars <- character(0)

  list(
    response = response_vars,
    plot_numeric = plot_numeric_vars,
    obs_numeric = obs_numeric_vars,
    encodings = encoding_specs
  )
}


#' Parse a single encoding call like hash(species, dim = 32)
#' @keywords internal
parse_encoding_call <- function(call_str) {
  # Extract function name
  type_match <- regexpr("^(hash|embed|onehot|raw)", call_str, perl = TRUE)
  type <- regmatches(call_str, type_match)

  # Extract arguments inside parentheses
  args_match <- regexpr("\\(([^)]+)\\)", call_str, perl = TRUE)
  args_str <- regmatches(call_str, args_match)
  args_str <- gsub("^\\(|\\)$", "", args_str)

  # Split by comma, being careful about nested stuff
  args <- strsplit(args_str, ",\\s*")[[1]]

  # Separate variable names from parameters
  vars <- character(0)
  params <- list()

  for (arg in args) {
    arg <- trimws(arg)
    if (grepl("=", arg)) {
      # It's a parameter
      parts <- strsplit(arg, "\\s*=\\s*")[[1]]
      param_name <- trimws(parts[1])
      param_value <- trimws(parts[2])

      # Try to convert to numeric
      numeric_val <- suppressWarnings(as.numeric(param_value))
      if (!is.na(numeric_val)) {
        params[[param_name]] <- numeric_val
      } else {
        # Remove quotes if string
        param_value <- gsub("^['\"]|['\"]$", "", param_value)
        params[[param_name]] <- param_value
      }
    } else {
      # It's a variable name
      vars <- c(vars, arg)
    }
  }

  list(
    type = type,
    vars = vars,
    dim = params$dim,
    top = params$top,
    bottom = params$bottom,
    by = params$by,
    params = params
  )
}


#' Build PlotEncoder from parsed formula specs
#' @keywords internal
build_plot_encoder <- function(formula_parts, data, obs) {
  encoder <- plot_encoder_new()

  # Add auto-scaled numeric variables from data
  if (length(formula_parts$plot_numeric) > 0) {
    plot_encoder_add_numeric(
      encoder,
      name = "plot_numeric",
      columns = formula_parts$plot_numeric,
      source = "plot"
    )
  }

  # Add each encoding spec
  for (spec in formula_parts$encodings) {
    # Determine data source for the variables
    vars_in_obs <- spec$vars[spec$vars %in% names(obs)]
    vars_in_data <- spec$vars[spec$vars %in% names(data)]

    # Default: hash/embed from observation, onehot/raw from data
    source <- if (spec$type %in% c("hash", "embed") && length(vars_in_obs) > 0) {
      "observation"
    } else {
      "plot"
    }

    # Generate a unique name for this spec
    spec_name <- paste0(spec$type, "_", paste(spec$vars, collapse = "_"))

    if (spec$type == "hash") {
      plot_encoder_add_hash(
        encoder,
        name = spec_name,
        columns = spec$vars,
        dim = as.integer(spec$dim %||% 32L),
        top_k = as.integer(spec$top %||% 0L),
        bottom_k = as.integer(spec$bottom %||% 0L),
        rank_by = spec$by %||% "",
        source = source
      )
    } else if (spec$type == "embed") {
      plot_encoder_add_embed(
        encoder,
        name = spec_name,
        columns = spec$vars,
        dim = as.integer(spec$dim %||% 16L),
        top_k = as.integer(spec$top %||% 0L),
        bottom_k = as.integer(spec$bottom %||% 0L),
        rank_by = spec$by %||% "",
        source = source
      )
    } else if (spec$type == "onehot") {
      plot_encoder_add_onehot(
        encoder,
        name = spec_name,
        columns = spec$vars,
        source = source
      )
    } else if (spec$type == "raw") {
      plot_encoder_add_raw(
        encoder,
        name = spec_name,
        columns = spec$vars,
        source = source
      )
    }
  }

  encoder
}


# ==============================================================================
# Special formula functions (for documentation/autocomplete)
# ==============================================================================

#' @title Feature Hashing for High-Cardinality Categoricals
#' @description
#' Use in \code{resolve()} formula to specify feature hashing for categorical
#' variables from observation data. This encoding scales to any vocabulary size.
#'
#' @param ... Column names from obs to hash
#' @param dim Dimension of hash embedding (default: 32)
#' @param top Number of top entries (by weight) to use per unit
#' @param bottom Number of bottom entries (rarest) to use per unit
#' @param by Column name to rank by (e.g., "cover", "price", "count").
#'   Required when using \code{top} or \code{bottom}.
#'
#' @details
#' Selection allows you to focus on the most (or least) frequent/weighted
#' observations per unit. This can reduce noise from rare items.
#'
#' @examples
#' \dontrun{
#' # Ecology: hash species
#' resolve(ph ~ elevation + hash(species, top = 5, by = "cover"),
#'         data = plots, obs = species_obs, by = "plot_id")
#'
#' # E-commerce: hash products
#' resolve(ltv ~ age + hash(product_id, top = 10, by = "price"),
#'         data = customers, obs = purchases, by = "customer_id")
#'
#' # Custom dimension
#' resolve(y ~ x + hash(item, dim = 64), data = units, obs = events, by = "id")
#'
#' # Top + bottom (most and least frequent)
#' resolve(y ~ hash(category, top = 3, bottom = 3, by = "count"), ...)
#' }
#'
#' @export
hash <- function(..., dim = 32, top = NULL, bottom = NULL, by = NULL) {
  # This function is just for documentation/IDE support
  # Actual parsing happens in parse_resolve_formula
  stop("hash() should only be used inside a resolve() formula")
}


#' @title Learned Embeddings for Categorical Variables
#' @description
#' Use in \code{resolve()} formula to specify learned embeddings for
#' categorical variables. Each unique value gets its own learned vector.
#'
#' @param ... Column names from obs to embed
#' @param dim Dimension of embeddings (default: 16)
#' @param top Number of top entries (by weight) to embed per unit
#' @param bottom Number of bottom entries (rarest) to embed per unit
#' @param by Column name to rank by
#'
#' @examples
#' \dontrun{
#' # Ecology: embed taxonomy
#' resolve(ph ~ elevation + hash(species) + embed(genus, family),
#'         data = plots, obs = species_obs, by = "plot_id")
#'
#' # E-commerce: embed categories
#' resolve(ltv ~ age + hash(product_id) + embed(category, brand),
#'         data = customers, obs = purchases, by = "customer_id")
#' }
#'
#' @export
embed <- function(..., dim = 16, top = NULL, bottom = NULL, by = NULL) {
  stop("embed() should only be used inside a resolve() formula")
}


#' @title One-Hot Encoding for Categorical Variables
#' @description
#' Use in \code{resolve()} formula to specify one-hot encoding for
#' categorical variables with small vocabularies. Typically used for
#' data-level (unit-level) categoricals.
#'
#' @param ... Column names to one-hot encode
#'
#' @examples
#' \dontrun{
#' resolve(y ~ x + hash(item) + onehot(treatment),
#'         data = units, obs = events, by = "id")
#' }
#'
#' @export
onehot <- function(...) {
  stop("onehot() should only be used inside a resolve() formula")
}


#' @title Raw Numeric Variables (No Scaling)
#' @description
#' Use in \code{resolve()} formula to include numeric variables without
#' auto-scaling. Use this for pre-scaled data or when you want to preserve
#' the original scale.
#'
#' By default, bare numeric variables in the formula are auto-scaled
#' (z-score standardization). Use \code{raw()} to opt out.
#'
#' @param ... Column names to include without scaling
#'
#' @examples
#' \dontrun{
#' # Default: auto-scaled
#' resolve(y ~ x1 + x2 + hash(item), data = units, obs = events, by = "id")
#'
#' # Use raw() for pre-scaled variables
#' resolve(y ~ raw(x1_scaled) + x2 + hash(item), data = units, obs = events, by = "id")
#' }
#'
#' @export
raw <- function(...) {
  stop("raw() should only be used inside a resolve() formula")
}


# ==============================================================================
# Helpers
# ==============================================================================

#' @keywords internal
`%||%` <- function(x, y) if (is.null(x)) y else x
