#' Is the resolve_c backend available?
#'
#' The R package is a thin client over the `resolve_c` shared library (the C ABI
#' over the RESOLVE engine, which bundles libtorch). That library is loaded at
#' runtime rather than linked at build time, so the package installs and checks
#' without it. This returns `TRUE` once the backend has been located and bound
#' (at load, or after [resolve.install_backend()]). The dataset / training /
#' prediction verbs require it; metric helpers and input validation do not.
#'
#' @return `TRUE` if the resolve_c backend is loaded, else `FALSE`.
#' @seealso [resolve.install_backend()]
#' @examples
#' resolve.available()
#' @export
resolve.available <- function() {
  isTRUE(tryCatch(resolve_capi_is_available(), error = function(e) FALSE))
}

# Stop with a consistent, actionable error when an engine verb is called with no
# backend loaded. Called after a verb's pure-R input validation so argument
# errors still surface first.
.resolve_require_backend <- function() {
  if (!resolve.available()) {
    stop(
      "The resolve_c backend is not installed.\n",
      "Install it with resolve.install_backend(), or set RESOLVE_C_HOME to a ",
      "directory containing the resolve_c shared library.",
      call. = FALSE
    )
  }
}

#' Download and install the resolve_c backend
#'
#' Fetch a prebuilt `resolve_c` shared library (with its bundled libtorch
#' runtime) and load it, so the dataset / training / prediction verbs become
#' available. This mirrors `torch::install_torch()`: the CRAN package is small
#' and backend-free, and the heavy binary is fetched on first use into the
#' user's data directory (`tools::R_user_dir("resolveR", "data")`).
#'
#' The default download URL points at a per-platform asset on the package's
#' GitHub Releases; supply `url` to install from a local or private mirror.
#'
#' The `cpu` variant is self-contained (its libtorch is bundled in the download).
#' The CUDA variants ship only the small `resolve_c` library on the GitHub
#' release and fetch the matching official libtorch (2-4 GB) from
#' `download.pytorch.org` on first install, pinned to the exact version
#' `resolve_c` was built against so the ABI matches. GPU training then runs
#' through the ordinary `device = "cuda"` path of [resolve.train.dataset()] /
#' [resolve.load()].
#'
#' @param version Release version to fetch (default: the installed package
#'   version). Ignored when `url` is given.
#' @param variant One of `"cpu"` (default), `"cu128"`, `"cu130"`, or `"cuda"`.
#'   `"cuda"` auto-selects the CUDA line from the installed NVIDIA driver
#'   (`cu130` for CUDA >= 13, else `cu128`). CUDA variants are Windows/Linux
#'   x86_64 only (no CUDA on macOS).
#' @param url Explicit download URL for a `.zip` containing the library and its
#'   runtime dependencies. Overrides `version` / `variant`.
#' @param dir Destination directory (default: the package's user data dir).
#' @param quiet Suppress progress messages (default `FALSE`).
#' @param force Reinstall even if the backend is already present (default
#'   `FALSE`).
#' @return (Invisibly) the directory the backend was installed into.
#' @seealso [resolve.available()]
#' @examples
#' \dontrun{
#' resolve.install_backend()
#' resolve.install_backend(variant = "cuda")
#' resolve.install_backend(url = "file:///path/to/resolve_c-linux-x86_64-cpu.zip")
#' }
#' @export
resolve.install_backend <- function(version = NULL,
                                     variant = "cpu",
                                     url = NULL,
                                     dir = NULL,
                                     quiet = FALSE,
                                     force = FALSE) {
  # "cuda" -> a concrete cu128/cu130 line from the installed driver.
  variant <- .resolve_normalize_variant(variant)
  oa <- .resolve_os_arch()
  entry <- .resolve_backend_registry(variant, oa$os, oa$arch)
  if (isTRUE(entry$cuda) && oa$os == "macos") {
    stop("CUDA backends are not available on macOS (no CUDA on Apple hardware); ",
         "use variant = 'cpu'.", call. = FALSE)
  }
  if (is.null(dir)) {
    dir <- file.path(tools::R_user_dir("resolveR", "data"), "resolve_c")
  }
  libname <- .resolve_backend_libname()
  dest_lib <- file.path(dir, libname)

  # Already installed: just (re)bind it unless a reinstall was requested.
  if (file.exists(dest_lib) && !force) {
    if (!quiet) {
      message("resolve_c already installed at ", dir,
              " (use force = TRUE to reinstall).")
    }
    .resolve_prepend_loader_path(dir)
    invisible(tryCatch(resolve_capi_load_lib(dest_lib), error = function(e) FALSE))
    return(invisible(dir))
  }

  # resolve_c itself (small): explicit url override, else the GitHub release asset.
  if (is.null(url)) {
    if (is.null(version)) version <- as.character(utils::packageVersion("resolveR"))
    url <- sprintf(
      "https://github.com/gcol33/resolve/releases/download/v%s/%s",
      version, entry$github_asset
    )
  }

  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  if (force) unlink(list.files(dir, full.names = TRUE), recursive = TRUE)

  if (!quiet) message("Installing resolve_c backend '", variant, "' into ", dir)
  .resolve_download_unzip(url, dir, quiet)

  # CUDA variants ship only resolve_c on GitHub (the >2 GB libtorch cannot be a
  # release asset). Fetch the matching official libtorch from PyTorch's CDN,
  # pinned to the exact version resolve_c was built against so the C++ ABI
  # matches, and flatten its runtime libraries next to resolve_c.
  if (isTRUE(entry$cuda)) {
    if (!quiet) {
      message("Fetching matching libtorch (", variant, ", ~2-4 GB, one-time) ",
              "from download.pytorch.org ...")
    }
    lt_dir <- file.path(dir, ".libtorch_tmp")
    unlink(lt_dir, recursive = TRUE)
    dir.create(lt_dir, showWarnings = FALSE)
    on.exit(unlink(lt_dir, recursive = TRUE), add = TRUE)
    .resolve_download_unzip(entry$libtorch_url, lt_dir, quiet)
    lt_lib <- file.path(lt_dir, "libtorch", "lib")
    if (!dir.exists(lt_lib)) {
      stop("the downloaded libtorch archive had no libtorch/lib directory.")
    }
    if (!all(file.copy(list.files(lt_lib, full.names = TRUE), dir, overwrite = TRUE))) {
      stop("failed to place the libtorch runtime libraries next to resolve_c.")
    }
  }

  # Linux CUDA: PyTorch's libtorch links several NVIDIA math libraries by
  # standard soname without bundling them (cudart / cusparse / cufft / cusolver
  # / curand / cublas / nvJitLink). Fetch them from NVIDIA's redistributable CDN,
  # version-pinned to the CUDA line, and flatten beside resolve_c. Windows
  # libtorch bundles the standard-named DLLs, so this runs on Linux only.
  if (isTRUE(entry$cuda) && length(entry$cuda_redist)) {
    if (!quiet) {
      message("Fetching CUDA runtime libraries from NVIDIA (",
              length(entry$cuda_redist), " components, ~1.4 GB, one-time) ...")
    }
    for (u in entry$cuda_redist) .resolve_fetch_cuda_lib(u, dir, quiet)
  }

  if (!file.exists(dest_lib)) {
    stop("The downloaded archive did not contain ", libname, " under ", dir, ".")
  }

  .resolve_prepend_loader_path(dir)
  ok <- isTRUE(tryCatch(resolve_capi_load_lib(dest_lib), error = function(e) FALSE))
  if (!ok) {
    stop("Downloaded resolve_c but could not load it: ",
         tryCatch(resolve_capi_load_error(), error = function(e) ""))
  }
  if (!quiet) message("resolve_c backend '", variant, "' installed and loaded from ", dir, ".")
  invisible(dir)
}


#' Load a Trained RESOLVE Model
#'
#' Load a model from a saved checkpoint.
#'
#' @param path Path to checkpoint file
#' @param device Device to load the model onto: `"cpu"` (default), `"cuda"`
#'   (the current CUDA device) or `"cuda:N"` (CUDA device N).
#' @param vram_fraction Fraction of GPU VRAM the caching allocator may use when
#'   loading onto CUDA (default 1.0). Pass a lower value (e.g. 0.80) when sharing
#'   the GPU with a desktop / GUI to leave headroom.
#'
#' @return A `Predictor` object (an Rcpp module class) exposing the loaded
#'   model. Pass it to [resolve.predict.dataset()] to score a dataset built
#'   from its vocabularies (`predictor$vocabs()`), or read its `schema()` /
#'   `dataset_config()` to see what the checkpoint expects.
#'
#' @examples
#' \dontrun{
#' predictor <- resolve.load("model.pt")
#' dataset <- resolve.dataset.csv(
#'   header = "new_plots.csv", species = "new_species.csv",
#'   roles = list(plot_id = "plot_id", species_id = "species"),
#'   targets = list(),
#'   vocabs = predictor$vocabs(), config = predictor$dataset_config()
#' )
#' preds <- resolve.predict.dataset(predictor, dataset)
#' }
#'
#' @export
resolve.load <- function(path, device = "cpu", vram_fraction = 1.0) {
  .resolve_check_checkpoint_path(path)
  if (!is.numeric(vram_fraction) || length(vram_fraction) != 1 ||
      vram_fraction <= 0 || vram_fraction > 1) {
    stop("vram_fraction must be a single number in (0, 1]")
  }
  .resolve_require_backend()
  .resolve_module()$Predictor_load(path, device, vram_fraction)
}

# A path argument naming one existing checkpoint file.
.resolve_check_checkpoint_path <- function(path) {
  if (!is.character(path) || length(path) != 1) {
    stop("path must be a single file path string")
  }
  if (!file.exists(path)) {
    stop(sprintf("checkpoint file does not exist: %s", path))
  }
  invisible(path)
}

# A checkpoint's persisted record, "train_config" or "run_metadata", read
# without loading the model.
.resolve_read_checkpoint <- function(what, path) {
  .resolve_check_checkpoint_path(path)
  .resolve_require_backend()
  .resolve_module()$Trainer_read_checkpoint(what, path)
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
#' @return A named list of training-configuration fields, the keys and
#'   spellings `trainConfig` in [resolve.train.dataset()] takes (enums such as
#'   `loss_config` and `lr_scheduler` by name).
#' @examples
#' \dontrun{
#' cfg <- resolve.load_train_config("model.pt")
#' cfg$batch_size
#' cfg$band_thresholds
#' }
#' @export
resolve.load_train_config <- function(path) {
  .resolve_read_checkpoint("train_config", path)
}


#' Recover the Run Metadata From a Checkpoint
#'
#' Reads back the run metadata persisted in a RESOLVE checkpoint: training time,
#' train/test plot counts, how the held-out fold was chosen (`hold_out`:
#' `"random"`, a `test_size` fraction drawn with `split_seed`, or `"explicit"`,
#' the plots named by `stoppingPlots`), best and total epochs, the RESOLVE
#' version and timestamps, and the per-target final-metric tree.
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
  .resolve_read_checkpoint("run_metadata", path)
}


#' Default Configuration Values
#'
#' The engine's default for every field of a configuration: the values a field
#' takes when a config list or a `NULL` argument leaves it out. Every verb that
#' takes a config list reads it against these fields, so a key outside them is
#' refused, at any level of nesting.
#'
#' @param which `"dataset"` (the `config` of [resolve.dataset.csv()]),
#'   `"model"` and `"train"` (`modelConfig` and `trainConfig` of
#'   [resolve.train.dataset()]), `"spatial_block"` ([resolve.cross_validate()]),
#'   `"retrieval"` ([resolve.retrieval()]), `"pretrain"` and
#'   `"masked_species"` (the `pretrain` argument of [resolve.train.dataset()]),
#'   or `"vae"` ([resolve.pretrain_vae()]).
#' @return A named list, one element per field.
#' @examples
#' \dontrun{
#' resolve.default_config("train")$max_epochs
#' names(resolve.default_config("model"))
#' }
#' @export
resolve.default_config <- function(which) {
  if (!is.character(which) || length(which) != 1) {
    stop("which must be a single config name")
  }
  .resolve_require_backend()
  .resolve_module()$default_config(which)
}


#' Build the Species Graph a Heterogeneous GNN Passes Messages On
#'
#' Joins the species of a dataset's vocabulary by the relations
#' `heterogeneous_gnn` reads: a same-genus and a same-family edge for species
#' sharing a taxon, and a co-occurrence edge between species recorded in the
#' same plot often enough. `resolve.train.dataset()` builds this itself for a
#' heterogeneous-GNN model (`modelConfig = list(encoder_architecture =
#' "heterogeneous_gnn")`) and the checkpoint carries the result, so call it
#' directly only to inspect the graph or to build a variant by hand.
#'
#' @param dataset A `ResolveDataset`, as returned by [resolve.dataset.csv()].
#'   Co-occurrence edges need the per-plot species vector, so
#'   `species_encoding = "sparse"`; taxonomic edges need genus/family roles.
#' @param kCooccurrence How many co-occurrence partners to keep per species.
#' @param cooccurrenceThreshold Smallest share of plots two species must share
#'   for a co-occurrence edge.
#' @param taxonomicEdges Join species sharing a genus or a family.
#' @param cooccurrenceEdges Join species recorded together.
#'
#' @return A list with `edge_index` (a 2-row matrix of edge endpoints, source
#'   in row 1 and target in row 2, as 0-based species codes), `edge_type` (0
#'   same genus, 1 same family, 2 co-occurrence), `n_species` and `n_edges`.
#'
#' @examples
#' \dontrun{
#' ds <- resolve.dataset.csv(header, species, roles, targets,
#'                           config = list(species_encoding = "sparse"))
#' graph <- resolve.species_graph(ds)
#' graph$n_edges
#' }
#'
#' @export
resolve.species_graph <- function(dataset,
                                  kCooccurrence = NULL,
                                  cooccurrenceThreshold = NULL,
                                  taxonomicEdges = NULL,
                                  cooccurrenceEdges = NULL) {
  .resolve_check_dataset(dataset)
  .resolve_require_backend()
  dataset$species_graph(.resolve_named_settings(list(
    k_cooccurrence = kCooccurrence,
    cooccurrence_threshold = cooccurrenceThreshold,
    use_taxonomic_edges = taxonomicEdges,
    use_cooccurrence_edges = cooccurrenceEdges
  )))
}

# The settings of a list that were given: NULL leaves a field at the engine's
# default.
.resolve_named_settings <- function(settings) {
  settings[!vapply(settings, is.null, logical(1))]
}


#' Save a Trained RESOLVE Model
#'
#' Write a trained model, its scalers and its vocabularies to a checkpoint
#' file that [resolve.load()] reads back.
#'
#' @param trainer The list returned by [resolve.train.dataset()], or the
#'   `Trainer` object it carries.
#' @param path Path of the checkpoint file to write.
#'
#' @return No return value, called for its side effect of writing the
#'   checkpoint to `path`.
#'
#' @examples
#' \dontrun{
#' fit <- resolve.train.dataset(dataset)
#' resolve.save(fit, file.path(tempdir(), "model.pt"))
#' }
#'
#' @export
resolve.save <- function(trainer, path) {
  # Pass the optional `metadata` arg explicitly as NULL: Rcpp module methods
  # register the full C++ arity and do not honor the `= R_NilValue` default, so
  # a one-argument `$save(path)` fails to match ("could not find valid method").
  if (is.list(trainer) && !is.null(trainer$trainer)) {
    trainer$trainer$save(path, NULL)
  } else if (inherits(trainer, "Rcpp_Trainer")) {
    trainer$save(path, NULL)
  } else {
    stop("trainer must be a Trainer object or the result of resolve.train.dataset()")
  }
  invisible(NULL)
}


#' Check Training Progress
#'
#' Read the `progress.json` file that a training run with a checkpoint
#' directory (`checkpointDir` in [resolve.train.dataset()]) updates after
#' every checkpoint, so a long run can be followed from another R session.
#'
#' @param checkpointDir Path to the checkpoint directory of a training run.
#'
#' @return A named list with the fields of the progress file (the current
#'   `epoch`, `maxEpochs`, the percentage `progressPct`, and the latest
#'   losses), or `NULL` when the directory holds no progress file yet.
#'
#' @examples
#' # No run has written to this directory, so there is no progress to report.
#' resolve.progress(tempdir())
#'
#' # A progress file as a training run writes it.
#' dir <- file.path(tempdir(), "resolve-progress-example")
#' dir.create(dir, showWarnings = FALSE)
#' writeLines('{"epoch": 12, "maxEpochs": 100, "progressPct": 12.0}',
#'            file.path(dir, "progress.json"))
#' progress <- resolve.progress(dir)
#' progress$epoch
#' unlink(dir, recursive = TRUE)
#'
#' @export
resolve.progress <- function(checkpointDir) {
  progressFile <- file.path(checkpointDir, "progress.json")
  if (!file.exists(progressFile)) {
    return(NULL)
  }
  jsonlite::fromJSON(progressFile)
}


#' Load a Dataset From CSV Files
#'
#' Read plots and their species records from CSV files and encode them for a
#' model: the plot-level columns (coordinates, covariates, categorical
#' covariates, targets) of a header file, one row per plot, and the species
#' records of a species file, one row per record.
#'
#' @param header Path to the header CSV file (one row per plot, with the
#'   targets).
#' @param species Path to the species CSV file (one row per species record).
#' @param roles Named list mapping column roles:
#'   - plot_id: Column name for plot ID (default "plot_id")
#'   - species_id: Column name for species ID (default "species_id")
#'   - abundance: Column name for abundance (optional)
#'   - longitude: Column name for longitude (optional)
#'   - latitude: Column name for latitude (optional)
#'   - genus: Column name for genus (optional)
#'   - family: Column name for family (optional)
#'   - covariates: Vector of covariate column names (optional)
#'   - categoricals: Vector of categorical covariate column names (optional)
#'
#'   An optional role is unset by omitting it, or by giving it the empty
#'   string, which is the spelling to use when a role has to be cleared from a
#'   list that already carries it. A non-empty column name the file does not
#'   have is an error, so a typo fails loudly, and so is a key outside this
#'   list.
#' @param targets Named list of target specifications, the name naming the
#'   target. Each is a list of:
#'   - column: Column name in the header file (default: the target's name)
#'   - task: "regression" (default) or "classification"
#'   - transform: "none" (default) or "log1p"
#'   - num_classes: Number of classes for classification
#'   - weight: Loss weight (default 1.0)
#'   - class_mapping: Named integer vector pinning each class label to its code
#'     (default: codes fitted from the data, labels in sorted order)
#'   - missing: "drop" (default) or "mask". What happens to a plot with no
#'     label for this target (an empty or NA cell, or a class outside
#'     `class_mapping`). "drop" leaves the plot out of the dataset. "mask"
#'     keeps it: the plot trains and is scored on the targets it is labelled
#'     for, and this target's loss, scaler and metrics use only its labelled
#'     plots. A plot labelled for no target is left out either way.
#'
#'   A key outside this list, two targets with one name, or an unnamed target
#'   is an error. A dataset for training needs at least one target; one built
#'   to be scored needs none.
#' @param config Named list of dataset configuration options; a field left out
#'   takes the engine's default (`resolve.default_config("dataset")`), and a key
#'   outside these fields is an error:
#'   - species_encoding: "hash", "embed", "sparse", "rank_pool" or
#'     "transformer". hash folds each plot's species into a fixed-width vector
#'     of `hash_dim` buckets; embed gives each of the plot's `top_k_species`
#'     ranked species slots a learned embedding; sparse is the plot's full
#'     species-by-abundance vector; rank_pool pools learned species, genus and
#'     family embeddings over every species the plot records; transformer
#'     attends over them.
#'   - hash_dim: Hash buckets of the hash encoding
#'   - top_k: Species hashed per plot and fixed genus/family slots
#'   - top_k_species: Ranked species slots of the embed encoding
#'   - selection: "top", "bottom", "top_bottom", or "all". Which species of a
#'     plot survive its per-plot budget: `top_k` for the hash encoding,
#'     `top_k_species` for embed (which rejects "all", having a fixed number of
#'     ranked slots), and `species_budget` for rank_pool, transformer and
#'     sparse.
#'   - species_budget: species kept per plot by `selection` for the rank_pool,
#'     transformer and sparse encodings; 0 = no budget, so every species a plot
#'     records is encoded and `selection` has nothing to narrow.
#'   - representation: "abundance" or "presence_absence". The value each
#'     species carries into every encoding: its abundance, or 1 under
#'     "presence_absence" (a pooled dataset then reports no cover). The
#'     selection still ranks by recorded abundance.
#'   - aggregation: "abundance" or "count". How the fixed genus/family slots
#'     of the hash, sparse and embed encodings rank a plot's taxa: by the
#'     summed value of their species, or by how many of the plot's species
#'     they hold. rank_pool and transformer pool each species' genus and family
#'     embeddings with the species weights, or under "count" with one weight
#'     per species.
#'   - normalization: "raw", "norm", or "log1p": the abundance values as read,
#'     scaled to sum to one per plot, or log1p-transformed
#'   - track_unknown_fraction, track_unknown_count: add the share and the
#'     number of a plot's species the vocabulary does not know as columns
#'   - use_taxonomy: encode the genus/family roles
#'   - pool_weighting: rank_pool weighting "binary", "abundance", "log1p",
#'     "norm", or "rank"
#'   - pool_species_cap: rank_pool per-plot species cap; 0 = no cap, -x = the
#'     (100 - x)th percentile of species per plot, computed on the data loaded
#'     (-1 = p99, -5 = p95; x in 1..99), >0 = manual cap
#'   - missing_values: "indicate" or "zero": a missing covariate or coordinate
#'     cell is flagged and filled with the fitting fold's mean, or read as 0
#'   - zero_abundance_as: the abundance a recorded 0 is read at
#' @param schemaSource Optional ResolveDataset (from a previous
#'   \code{resolve.dataset.csv()} call). When supplied, this dataset is encoded
#'   against that dataset's species / taxonomy / categorical vocabularies and
#'   classification class mappings instead of fitting its own, so a held-out or
#'   transfer set lines up with the training set's embedding namespaces. Default
#'   \code{NULL} fits fresh vocabularies.
#' @param vocabs Optional vocabulary list from \code{predictor$vocabs()} (or
#'   \code{dataset$vocabs()}). Same effect as \code{schemaSource}, sourced from a
#'   CHECKPOINT rather than an in-memory training dataset, so the training CSVs
#'   need not exist. This is the path for scoring new data: every non-hash
#'   encoder indexes an embedding table with a code that is a function of the
#'   file the vocabulary was fitted on, so a dataset built without the training
#'   vocabularies looks up the wrong rows and \code{predictor$predict_dataset()}
#'   rejects it. Pair with \code{config = predictor$dataset_config()}. Mutually
#'   exclusive with \code{schemaSource}. A vocabulary list without a species
#'   vocabulary (a checkpoint that stores only the vocabulary sizes) is
#'   refused.
#'
#' @return A ResolveDataset object (an Rcpp module class) with methods:
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
                                schemaSource = NULL,
                                vocabs = NULL) {
  for (arg in c("header", "species")) {
    path <- get(arg)
    if (!is.character(path) || length(path) != 1) {
      stop(sprintf("%s must be a single file path string", arg))
    }
    if (!file.exists(path)) {
      stop(sprintf("%s file does not exist: %s", arg, path))
    }
  }
  .resolve_load_dataset(header, species, roles, targets, config, schemaSource, vocabs)
}

# The one loader behind resolve.dataset.csv() and resolve.dataset.frame():
# `header` and `species` are each NULL, a CSV path, or a named list of
# character columns. The engine validates the roles, targets and config, and
# the vocabulary source.
.resolve_load_dataset <- function(header, species, roles, targets, config,
                                  schemaSource, vocabs) {
  for (arg in c("roles", "targets", "config")) {
    if (!is.list(get(arg))) stop(sprintf("%s must be a named list", arg))
  }
  if (!is.null(schemaSource) && !inherits(schemaSource, "Rcpp_ResolveDataset")) {
    stop("schemaSource must be a ResolveDataset from resolve.dataset.csv()/frame()")
  }
  if (!is.null(vocabs) && !is.list(vocabs)) {
    stop("vocabs must be the list returned by predictor$vocabs() or dataset$vocabs()")
  }
  # The plot and species columns default to these names.
  if (is.null(roles$plot_id)) roles$plot_id <- "plot_id"
  if (is.null(roles$species_id)) roles$species_id <- "species_id"
  .resolve_require_backend()
  .resolve_module()$ResolveDataset_load(header, species, roles, targets, config,
                                        schemaSource, vocabs)
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
#' In-memory analog of [resolve.dataset.csv()]. Builds and encodes a dataset
#' directly from data frames already in R, without writing them to a CSV
#' first, for a header filtered or subset before a fit. The result is identical
#' to [resolve.dataset.csv()] on the equivalent CSV.
#'
#' @param header A data.frame of header data (one row per plot). In single-table
#'   mode (\code{species = NULL}) this is instead the long-format species frame,
#'   whose first row of each plot carries the plot's coordinates and targets
#'   (covariates need a header).
#' @param species A data.frame of species data (long format), a single file path
#'   string (the large species table is then read once from disk while the
#'   header stays in memory), or \code{NULL} for single-table mode.
#' @param roles Named list mapping roles to column names (see
#'   [resolve.dataset.csv()]).
#' @param targets Named list of target specifications.
#' @param config Named list of dataset configuration options.
#' @param schemaSource Optional ResolveDataset whose vocabularies / class
#'   mappings are reused (see [resolve.dataset.csv()]).
#' @param vocabs Optional vocabulary list from \code{predictor$vocabs()} (see
#'   [resolve.dataset.csv()]). Mutually exclusive with \code{schemaSource}.
#'
#' @return A ResolveDataset object.
#' @seealso [resolve.dataset.csv()]
#' @export
resolve.dataset.frame <- function(header,
                                  species = NULL,
                                  roles = list(),
                                  targets = list(),
                                  config = list(),
                                  schemaSource = NULL,
                                  vocabs = NULL) {
  if (is.null(species)) {
    # Single-table mode: `header` is the long-format species frame.
    return(.resolve_load_dataset(NULL, .resolve_df_to_columns(header, "species frame"),
                                 roles, targets, config, schemaSource, vocabs))
  }
  if (is.character(species)) {
    if (length(species) != 1) {
      stop("species path must be a single string")
    }
    if (!file.exists(species)) {
      stop(sprintf("species file does not exist: %s", species))
    }
  } else {
    species <- .resolve_df_to_columns(species, "species")
  }
  .resolve_load_dataset(.resolve_df_to_columns(header, "header"), species, roles, targets,
                        config, schemaSource, vocabs)
}

# A ResolveDataset argument.
.resolve_check_dataset <- function(dataset) {
  if (!inherits(dataset, "Rcpp_ResolveDataset")) {
    stop("dataset must be created with resolve.dataset.csv()")
  }
  invisible(dataset)
}


# The model and training arguments of resolve.train.dataset() and
# resolve.cross_validate(), by the ModelConfig / TrainConfig field each sets.
.resolve_model_args <- c(
  hiddenDims = "hidden_dims", moeRouting = "moe_routing", moePlacement = "moe_placement",
  nExperts = "n_experts", expertHiddenDims = "expert_hidden_dims", moeTopK = "moe_top_k",
  moeNoiseStd = "moe_noise_std", moeAuxLossWeight = "moe_aux_loss_weight",
  coverDropout = "cover_dropout", freezeComposition = "freeze_composition",
  dModel = "d_model", nHeads = "n_heads", nAttentionLayers = "n_attention_layers",
  transformerFfDim = "transformer_ff_dim", transformerPooling = "transformer_pooling",
  transformerDropout = "transformer_dropout"
)
.resolve_train_args <- c(
  maxEpochs = "max_epochs", patience = "patience", fixedEpochs = "fixed_epochs",
  lr = "lr", lrScheduler = "lr_scheduler", lrStepSize = "lr_step_size",
  lrGamma = "lr_gamma", lrMin = "lr_min", lrPlateauFactor = "lr_plateau_factor",
  lrPlateauPatience = "lr_plateau_patience", lrPlateauThreshold = "lr_plateau_threshold",
  lrPlateauCooldown = "lr_plateau_cooldown", batchSize = "batch_size", device = "device",
  lossConfig = "loss_config", ncaTemperature = "nca_temperature",
  ncaNeighbors = "nca_neighbors", ncaWeight = "nca_weight",
  checkpointDir = "checkpoint_dir", checkpointEvery = "checkpoint_every"
)

# Range checks on the numeric arguments that were given. The engine checks
# every enum name and every config key.
.resolve_check_numeric_args <- function(args) {
  positive <- c("maxEpochs", "patience", "lr", "batchSize", "ncaTemperature")
  non_negative <- c("fixedEpochs", "checkpointEvery", "ncaWeight", "moeNoiseStd",
                    "moeAuxLossWeight")
  for (arg in intersect(names(args), c(positive, non_negative, "ncaNeighbors"))) {
    value <- args[[arg]]
    if (is.null(value)) next
    if (!is.numeric(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("%s must be a single number", arg))
    }
    if (arg %in% positive && value <= 0) stop(sprintf("%s must be positive", arg))
    if (arg %in% non_negative && value < 0) stop(sprintf("%s must not be negative", arg))
  }
  if (!is.null(args$fixedEpochs) && !is.null(args$maxEpochs) &&
      args$fixedEpochs > args$maxEpochs) {
    stop("fixedEpochs must be at most maxEpochs")
  }
  if (!is.null(args$nExperts) && (!is.numeric(args$nExperts) || args$nExperts < 2)) {
    stop("nExperts must be at least 2; leave moeRouting at \"none\" for no mixture")
  }
  if (!is.null(args$moeTopK) && (!is.numeric(args$moeTopK) || args$moeTopK < 1)) {
    stop("moeTopK must be a positive integer")
  }
  if (!is.null(args$expertHiddenDims) &&
      (!is.numeric(args$expertHiddenDims) || length(args$expertHiddenDims) < 1)) {
    stop("expertHiddenDims must be a non-empty numeric vector")
  }
  invisible(TRUE)
}

# `config` with every given argument of `args` (named as in `map`) set on the
# field it names. A field given both ways is refused rather than one silently
# winning.
.resolve_merge_config <- function(config, args, map, what) {
  if (!is.list(config) || (length(config) > 0 && is.null(names(config)))) {
    stop(sprintf("%s must be a named list", what))
  }
  for (arg in intersect(names(args), names(map))) {
    value <- args[[arg]]
    if (is.null(value)) next
    field <- map[[arg]]
    if (!is.null(config[[field]])) {
      stop(sprintf("%s is given twice: as %s and as %s$%s", field, arg, what, field))
    }
    config[[field]] <- value
  }
  config
}

# The model, created for `dataset`, and the trainer of a run, from the model
# and training arguments in `args` (named as in .resolve_model_args /
# .resolve_train_args, plus modelConfig, trainConfig and pretrain). Pretrains
# the model first when `args$pretrain` asks for it.
.resolve_build_trainer <- function(dataset, args) {
  .resolve_check_numeric_args(args)
  modelConfig <- .resolve_merge_config(if (is.null(args$modelConfig)) list() else args$modelConfig,
                                       args, .resolve_model_args, "modelConfig")
  trainConfig <- .resolve_merge_config(if (is.null(args$trainConfig)) list() else args$trainConfig,
                                       args, .resolve_train_args, "trainConfig")
  .resolve_require_backend()
  model <- .resolve_module()$ResolveModel_for_dataset(dataset, modelConfig)
  pretrained <- NULL
  if (!is.null(args$pretrain)) {
    pretrain <- args$pretrain
    if (!is.list(pretrain) || !is.character(pretrain$method) || length(pretrain$method) != 1) {
      stop("pretrain must be a list with a method (\"jepa\", \"scarf\" or \"masked_species\") ",
           "and the method's settings")
    }
    settings <- pretrain[names(pretrain) != "method"]
    pretrained <- model$pretrain(pretrain$method, dataset, settings)
  }
  list(model = model,
       trainer = new(.resolve_module()$Trainer, model, trainConfig),
       pretrain = pretrained)
}


#' Train a Model on a Dataset
#'
#' Build a model for a dataset loaded with [resolve.dataset.csv()], hold out a
#' fold of its plots for early stopping, and train.
#'
#' Every model and training setting has the engine's default
#' ([resolve.default_config()] `"model"` / `"train"`). The named arguments below
#' set the common ones; `modelConfig` and `trainConfig` reach every field of
#' the engine's `ModelConfig` and `TrainConfig`, nested architecture configs
#' included (`encoder_architecture`, `ft_transformer`, `tabnet`, `saint`, `gnn`,
#' `trait_net`, `excelformer`, `heterogeneous_gnn`, `tabm`, `parallel_layers`,
#' and the activation, normalization and head settings). A key the engine does
#' not know is an error, and so is a field set both by a named argument and in
#' the list. The fields that size a model to its data (the species encoding,
#' hash width, `top_k` and the taxonomy slots) come from the dataset.
#'
#' @param dataset A ResolveDataset object from resolve.dataset.csv()
#' @param hiddenDims Hidden layer widths of the encoder MLP.
#' @param maxEpochs Maximum training epochs.
#' @param patience Early stopping patience in epochs.
#' @param fixedEpochs Train exactly this many epochs of the `maxEpochs`
#'   schedule, with no early stopping, and keep the final weights (0, the
#'   default, is early stopping). The learning-rate schedule is still laid out
#'   over `maxEpochs`. The only mode that accepts `testSize = 0`, for a refit
#'   on every plot.
#' @param lr Learning rate.
#' @param lrScheduler Learning-rate schedule: `"none"` (constant), `"step"`
#'   multiplies the rate by `lrGamma` every `lrStepSize` epochs, `"cosine"`
#'   anneals it from `lr` to `lrMin` over `maxEpochs`, and `"plateau"`
#'   multiplies it by `lrPlateauFactor` when the held-out loss that drives
#'   early stopping has not improved for more than `lrPlateauPatience` epochs
#'   (PyTorch's `ReduceLROnPlateau` in mode `"min"`). `"plateau"` needs a
#'   held-out fold, so it is refused with `testSize = 0`.
#' @param lrStepSize Epochs between decays under `lrScheduler = "step"`.
#' @param lrGamma Factor of each decay under `lrScheduler = "step"`.
#' @param lrMin The rate the cosine schedule ends at, and the floor a plateau
#'   never decays below.
#' @param lrPlateauFactor Factor a plateau multiplies the rate by, in (0, 1).
#' @param lrPlateauPatience Epochs without improvement of the held-out loss a
#'   plateau allows; the next one decays the rate. Like `patience`, an epoch
#'   counts only once the loss objective has reached its final phase.
#' @param lrPlateauThreshold Relative improvement on the best held-out loss
#'   an epoch must make to count as one, in \[0, 1) (the default 0 is any
#'   decrease, the rule early stopping and `best_epoch` use; PyTorch's default
#'   is 1e-4).
#' @param lrPlateauCooldown Epochs after a decay during which no epoch counts
#'   against the rate.
#' @param batchSize Batch size.
#' @param device Device to train on: `"cpu"`, `"cuda"` (the current CUDA
#'   device) or `"cuda:N"`.
#' @param testSize Fraction of the data held out for early stopping (default
#'   0.2), drawn at random with `seed`. 0 holds nothing out and needs
#'   `fixedEpochs`.
#' @param stoppingPlots Plot IDs to hold out for early stopping instead of a
#'   random `testSize` fraction (default `NULL`): a withheld spatial block or
#'   database, or the fold a published protocol selects its epoch on. Early
#'   stopping, `best_epoch`, `final_metrics` and the baselines read exactly
#'   these plots; every other plot trains. Each ID must be a plot of
#'   `dataset`, listed once, and at least one plot must be left to train. With
#'   a masked target, the fold needs at least one plot labelled for it. Not
#'   combined with `testSize`.
#' @param seed Random seed of the split and the epoch shuffle (default 42).
#' @param savePath Path to save model checkpoint (optional)
#' @param checkpointDir Directory the run writes checkpoints to while it
#'   trains (optional): `best.pt` at every new best epoch,
#'   `checkpoint_<N>.pt` every `checkpointEvery` epochs, `progress.json`
#'   after every epoch (see [resolve.progress()]) and `checkpoint.pt` at the
#'   end. `best.pt` and `checkpoint_<N>.pt` carry the run's progress, so
#'   `resumeFrom` can continue from them.
#' @param checkpointEvery Also write `checkpoint_<N>.pt` every this many
#'   epochs (0, the default, writes `best.pt` only). Needs `checkpointDir`.
#' @param resumeFrom A `best.pt` or `checkpoint_<N>.pt` an interrupted run
#'   wrote (optional). The run continues at the epoch after it, with its
#'   optimizer, early-stopping and random-number state, as if it had never
#'   stopped; `result$resumed_from_epoch` names that epoch. Pass the
#'   interrupted run's dataset, model and training arguments and `seed`, so
#'   the split and recipe are the ones it trained on.
#' @param lossConfig Loss configuration: "mae", "smape", "combined" (the
#'   default: MAE, then MAE and SMAPE, then the band penalty, by phase), or
#'   "nca", which takes the "combined" regression schedule and adds the
#'   Neighbourhood Components Analysis term to classification targets.
#' @param ncaTemperature Scale of the stochastic-neighbour softmax in the NCA
#'   term, i.e. the effective number of neighbours each sample spreads over.
#'   Acts only when `lossConfig = "nca"`.
#' @param ncaNeighbors Neighbours per sample the NCA term sums over, its most
#'   similar in-batch samples. Zero or less keeps the whole batch. Acts only
#'   when `lossConfig = "nca"`.
#' @param ncaWeight Weight of the NCA term against the cross-entropy it is
#'   added to. Acts only when `lossConfig = "nca"`.
#' @param moeRouting Mixture-of-experts routing: `"none"` (a single dense
#'   encoder), `"soft"` (every expert contributes, weighted by the gate) or
#'   `"topk"` (only the `moeTopK` highest-gated experts run per plot).
#' @param moePlacement Where the mixture sits: `"tail"` makes it the encoder's
#'   final stage, so `hiddenDims` minus its last two widths becomes the
#'   backbone and the experts produce the latent; `"post"` runs it over the
#'   finished latent instead, preserving the width. `"post"` is the only
#'   placement open to an encoder with no MLP tail. Acts only when
#'   `moeRouting` is not `"none"`.
#' @param nExperts Expert networks in the mixture (at least 2).
#' @param expertHiddenDims Hidden widths inside each expert.
#' @param moeTopK Experts activated per plot under `"topk"` routing.
#' @param moeNoiseStd Standard deviation of the noise added to the gate logits
#'   during training, which spreads load across experts.
#' @param moeAuxLossWeight Weight on the mixture's load-balancing loss, added
#'   to the task loss.
#' @param coverDropout Cover-dropout rate applied to species cover values
#'   in rank-pool / transformer encoding modes.
#' @param freezeComposition Keep the species, genus and family embedding tables
#'   at their initialisation instead of learning them.
#' @param dModel Model dimension for the transformer / rank-pool encoder.
#' @param nHeads Number of attention heads in the transformer encoder.
#'   Ignored unless `nAttentionLayers > 0`.
#' @param nAttentionLayers Number of self-attention layers in the
#'   transformer encoder (0 pools the token embeddings without attention).
#' @param transformerFfDim Feed-forward hidden dimension inside each
#'   transformer block.
#' @param transformerPooling Pooling strategy used to collapse the per-
#'   species token sequence into a plot embedding: `"attention"` or `"cls"`.
#' @param transformerDropout Dropout rate applied inside transformer blocks.
#' @param modelConfig Named list of any further `ModelConfig` fields, for
#'   example `list(encoder_architecture = "ft_transformer", dropout = 0.1)`
#'   or `list(encoder_architecture = "heterogeneous_gnn", heterogeneous_gnn =
#'   list(k_cooccurrence = 10))` (the species graph is built from the dataset).
#' @param trainConfig Named list of any further `TrainConfig` fields, for
#'   example `list(weight_decay = 1e-4, use_amp = TRUE)`.
#' @param pretrain Self-supervised pretraining of the encoder on the
#'   dataset's plots before training (optional): a list with `method`,
#'   `"jepa"`, `"scarf"` or `"masked_species"` (a transformer-encoded model),
#'   and any of that method's settings ([resolve.default_config()]
#'   `"pretrain"`, or `"masked_species"`), e.g. `list(method = "scarf",
#'   pretrain_epochs = 20)`.
#' @param verbose Print training progress (default TRUE)
#'
#' @return A list with `trainer`, `result`, `dataset` and, with `pretrain`,
#'   `pretrain` (the pretraining's `loss_history`, `total_time_seconds` and
#'   `epochs_completed`). `result` carries `best_epoch`, `train_time_seconds`,
#'   `final_metrics`, the loss histories `train_loss` and `test_loss` (one
#'   value per epoch), the learning rate each epoch trained at `lr`, the loss
#'   histories' split by target `train_loss_by_target` and
#'   `test_loss_by_target` (named lists of per-epoch vectors: each target's
#'   weighted term of the total, so per epoch they add up to it),
#'   `train_aux_loss` (a named list of the auxiliary terms of `train_loss`,
#'   which belong to no target, each already weighted: `moe`, the
#'   mixture-of-experts load balancing, and `tabnet_sparsity`, TabNet's
#'   feature-mask sparsity; empty for a model with neither), the per-target
#'   `baselines` (the model against the training mean or mode, regression
#'   values in original units), `diagnostics`, `effective_batch_size` -- the
#'   batch size the run actually trained at, which is smaller than `batchSize`
#'   when the CUDA auto-halve-on-OOM retry fired -- and `resumed_from_epoch`
#'   (0 unless `resumeFrom` continued the run).
#'
#' @examples
#' \dontrun{
#' dataset <- resolve.dataset.csv(...)
#' result <- resolve.train.dataset(dataset, maxEpochs = 100)
#' ft <- resolve.train.dataset(dataset,
#'                             modelConfig = list(encoder_architecture = "ft_transformer"))
#' }
#'
#' @export
resolve.train.dataset <- function(dataset,
                                  hiddenDims = NULL,
                                  maxEpochs = NULL,
                                  patience = NULL,
                                  fixedEpochs = NULL,
                                  lr = NULL,
                                  lrScheduler = NULL,
                                  lrStepSize = NULL,
                                  lrGamma = NULL,
                                  lrMin = NULL,
                                  lrPlateauFactor = NULL,
                                  lrPlateauPatience = NULL,
                                  lrPlateauThreshold = NULL,
                                  lrPlateauCooldown = NULL,
                                  batchSize = NULL,
                                  device = NULL,
                                  testSize = 0.2,
                                  stoppingPlots = NULL,
                                  seed = 42L,
                                  savePath = NULL,
                                  checkpointDir = NULL,
                                  checkpointEvery = NULL,
                                  resumeFrom = NULL,
                                  lossConfig = NULL,
                                  ncaTemperature = NULL,
                                  ncaNeighbors = NULL,
                                  ncaWeight = NULL,
                                  moeRouting = NULL,
                                  moePlacement = NULL,
                                  nExperts = NULL,
                                  expertHiddenDims = NULL,
                                  moeTopK = NULL,
                                  moeNoiseStd = NULL,
                                  moeAuxLossWeight = NULL,
                                  coverDropout = NULL,
                                  freezeComposition = NULL,
                                  dModel = NULL,
                                  nHeads = NULL,
                                  nAttentionLayers = NULL,
                                  transformerFfDim = NULL,
                                  transformerPooling = NULL,
                                  transformerDropout = NULL,
                                  modelConfig = list(),
                                  trainConfig = list(),
                                  pretrain = NULL,
                                  verbose = TRUE) {
  .resolve_check_dataset(dataset)
  args <- mget(c(names(.resolve_model_args), names(.resolve_train_args),
                 "modelConfig", "trainConfig", "pretrain"))
  if (!is.null(checkpointEvery) && checkpointEvery > 0 && is.null(checkpointDir) &&
      is.null(trainConfig$checkpoint_dir)) {
    stop("checkpointEvery needs checkpointDir")
  }
  if (!is.null(resumeFrom) && (!is.character(resumeFrom) || length(resumeFrom) != 1)) {
    stop("resumeFrom must be the path of one checkpoint")
  }
  if (!is.null(stoppingPlots)) {
    if (!missing(testSize)) {
      stop("pass testSize or stoppingPlots, not both")
    }
    if (!is.character(stoppingPlots) || length(stoppingPlots) < 1 ||
        anyNA(stoppingPlots)) {
      stop("stoppingPlots must be a non-empty character vector of plot IDs")
    }
  } else {
    if (!is.numeric(testSize) || length(testSize) != 1 || testSize < 0 || testSize >= 1) {
      stop("testSize must be at least 0 and below 1")
    }
    # Early stopping, the default mode, needs a held-out fold.
    fixed <- if (!is.null(fixedEpochs)) fixedEpochs else trainConfig$fixed_epochs
    if (testSize == 0 && (is.null(fixed) || fixed == 0)) {
      stop("testSize = 0 leaves early stopping nothing to watch; set fixedEpochs")
    }
  }

  built <- .resolve_build_trainer(dataset, args)
  trainer <- built$trainer

  if (is.null(stoppingPlots)) {
    trainer$prepare_data_from_dataset(dataset, testSize, as.integer(seed))
  } else {
    trainer$prepare_data_stopping_fold(dataset, stoppingPlots, as.integer(seed))
  }

  # Train, or continue the run resumeFrom stopped
  if (is.null(resumeFrom)) {
    if (verbose) {
      cat("Training RESOLVE model...\n")
    }
    result <- trainer$fit()
  } else {
    if (verbose) {
      cat(sprintf("Resuming RESOLVE run from %s...\n", resumeFrom))
    }
    result <- trainer$resume(resumeFrom)
  }

  if (verbose) {
    requested <- trainer$get_config()$batch_size
    cat(sprintf("Training complete. Best epoch: %d\n", result$best_epoch))
    cat(sprintf("Training time: %.1f seconds\n", result$train_time_seconds))
    if (!is.null(result$effective_batch_size) && result$effective_batch_size != requested) {
      cat(sprintf(
        "Effective batch size: %d (requested %d) -- OOM auto-halve fired\n",
        result$effective_batch_size, requested
      ))
    }
    for (targetName in names(result$final_metrics)) {
      metrics <- result$final_metrics[[targetName]]
      cat(sprintf("  %s: ", targetName))
      metricStrs <- sapply(names(metrics), function(m) {
        sprintf("%s=%.4f", m, metrics[[m]])
      })
      cat(paste(metricStrs, collapse = ", "), "\n")
    }
  }

  # Save if requested, with the run metadata, so the checkpoint records how the
  # plots were split (resolve.load_run_metadata reads it back).
  if (!is.null(savePath)) {
    trainer$save(savePath, trainer$run_metadata())
    if (verbose) {
      cat(sprintf("Model saved to: %s\n", savePath))
    }
  }

  out <- list(
    trainer = trainer,
    result = result,
    dataset = dataset
  )
  if (!is.null(built$pretrain)) out$pretrain <- built$pretrain
  out
}


#' Cross-Validate a Model on a Dataset
#'
#' Train and score the same model recipe on k folds of a dataset's plots: each
#' fold is held out once while the model, restarted from the same initial
#' weights, trains on the others. Folds are drawn at random with `seed`, or,
#' with `spatial`, as blocks of a latitude / longitude grid, so a held-out fold
#' is a region the model never saw.
#'
#' @param dataset A ResolveDataset object from [resolve.dataset.csv()].
#' @param nFolds Number of folds (default 5).
#' @param seed Random seed of the folds (default 42).
#' @param spatial `NULL` (default) for random folds, or a named list of the
#'   spatial block settings: `lat_size` and `lon_size` (block size in degrees)
#'   and `balance` (`TRUE` assigns blocks to folds by greedy bin-packing for
#'   folds of similar size, `FALSE` round-robin); a field left out takes the
#'   engine's default ([resolve.default_config()] `"spatial_block"`). Needs
#'   coordinates.
#' @param ... The model and training arguments of [resolve.train.dataset()]
#'   (`hiddenDims`, `maxEpochs`, `lr`, `modelConfig`, `trainConfig`,
#'   `pretrain`, ...).
#'
#' @return A list with `n_folds`, `mean_metrics` and `std_metrics` (per target
#'   and metric, over the folds), `fold_results` (each fold's training result,
#'   as `result` of [resolve.train.dataset()]) and `total_time_seconds`.
#'
#' @examples
#' \dontrun{
#' cv <- resolve.cross_validate(dataset, nFolds = 5, maxEpochs = 100)
#' cv$mean_metrics$area$r2
#' blocks <- resolve.cross_validate(dataset, spatial = list(lat_size = 2, lon_size = 2))
#' }
#'
#' @export
resolve.cross_validate <- function(dataset, nFolds = 5L, seed = 42L, spatial = NULL, ...) {
  .resolve_check_dataset(dataset)
  args <- list(...)
  known <- c(names(.resolve_model_args), names(.resolve_train_args),
             "modelConfig", "trainConfig", "pretrain")
  unknown <- setdiff(names(args), known)
  if (length(args) > 0 && (is.null(names(args)) || any(!nzchar(names(args))))) {
    stop("the model and training arguments in ... must be named")
  }
  if (length(unknown) > 0) {
    stop(sprintf("unknown argument(s) %s; resolve.cross_validate takes the model and training ",
                 paste0("'", unknown, "'", collapse = ", ")),
         "arguments of resolve.train.dataset()")
  }
  if (!is.numeric(nFolds) || length(nFolds) != 1 || nFolds < 2) {
    stop("nFolds must be at least 2")
  }
  if (!is.null(spatial) && !is.list(spatial)) {
    stop("spatial must be NULL or a named list of spatial block settings")
  }
  trainer <- .resolve_build_trainer(dataset, args)$trainer
  trainer$prepare_data_from_dataset(dataset, 0.2, as.integer(seed))
  if (is.null(spatial)) {
    trainer$cross_validate(as.integer(nFolds), as.integer(seed))
  } else {
    trainer$cross_validate_spatial(spatial, as.integer(nFolds), as.integer(seed))
  }
}


#' Pretrain a Variational Autoencoder on Species Composition
#'
#' Train a variational autoencoder to reconstruct each plot's species vector,
#' without labels: a compressed representation of composition learned from the
#' plots alone.
#'
#' @param dataset A ResolveDataset loaded with `config = list(species_encoding
#'   = "sparse")`, which carries each plot's species vector.
#' @param config Named list of VAE settings (`latent_dim`, `encoder_dims`,
#'   `decoder_dims`, `dropout`, `kl_weight`, `kl_anneal_epochs`,
#'   `pretrain_epochs`, `pretrain_lr`, `pretrain_weight_decay`, `batch_size`,
#'   `device`, `seed`); a field left out takes the engine's default
#'   ([resolve.default_config()] `"vae"`).
#'
#' @return A list with the per-epoch `loss_history`, `recon_loss_history` and
#'   `kl_loss_history`, `total_time_seconds`, `epochs_completed`, `latent_dim`
#'   and `projection_weights`, the encoder's first layer (one row per unit of
#'   its first hidden layer, one column per species of the vocabulary).
#'
#' @examples
#' \dontrun{
#' ds <- resolve.dataset.csv(header, species, roles, targets,
#'                           config = list(species_encoding = "sparse"))
#' vae <- resolve.pretrain_vae(ds, list(latent_dim = 16, pretrain_epochs = 50))
#' plot(vae$loss_history, type = "l")
#' }
#'
#' @export
resolve.pretrain_vae <- function(dataset, config = list()) {
  .resolve_check_dataset(dataset)
  if (!is.list(config)) stop("config must be a named list")
  .resolve_require_backend()
  dataset$pretrain_vae(config)
}


#' Blend Predictions With Their Nearest Reference Plots
#'
#' Retrieval augmentation (Gorishniy et al. 2024, TabR): from now on every
#' prediction of `predictor` is blended with the labels of the plot's k nearest
#' reference plots in the model's latent space (cosine similarity, weighted by
#' a softmax of similarity over `temperature`): `(1 - alpha) * prediction +
#' alpha * retrieved`, a regression target in original units, a classification
#' target as class probabilities (the neighbours' label distribution), its
#' class the most probable one. Only the neighbours labelled for a target
#' count for it; a plot with none keeps the model's prediction.
#'
#' @param predictor A `Predictor` from [resolve.load()].
#' @param reference The reference plots, normally the training plots: a
#'   ResolveDataset built in the model's namespace (`vocabs =
#'   predictor$vocabs()`) that carries every target of the model. `NULL`
#'   removes a retrieval set before.
#' @param kNeighbors Reference plots each prediction reads.
#' @param alpha Share of the neighbours' labels in the blend, in \[0, 1\].
#' @param temperature Softmax temperature over the neighbours' cosine
#'   similarities (smaller puts the weight on the nearest).
#' @param batchSize Chunk size of the forward pass that encodes the reference
#'   plots (default 4096).
#'
#' @details Settings left `NULL` take the engine's defaults
#'   ([resolve.default_config()] `"retrieval"`). A reference plot that is also
#'   scored retrieves itself first.
#'
#' @return The predictor, invisibly.
#'
#' @examples
#' \dontrun{
#' predictor <- resolve.load("model.pt")
#' reference <- resolve.dataset.csv("train_plots.csv", "train_species.csv", roles,
#'                                  targets, vocabs = predictor$vocabs(),
#'                                  config = predictor$dataset_config())
#' resolve.retrieval(predictor, reference, kNeighbors = 32, alpha = 0.3)
#' preds <- resolve.predict.dataset(predictor, new_plots)
#' }
#'
#' @export
resolve.retrieval <- function(predictor, reference, kNeighbors = NULL, alpha = NULL,
                              temperature = NULL, batchSize = 4096L) {
  if (!inherits(predictor, "Rcpp_Predictor")) {
    stop("predictor must be loaded with resolve.load()")
  }
  .resolve_require_backend()
  if (is.null(reference)) {
    predictor$clear_retrieval()
    return(invisible(predictor))
  }
  .resolve_check_dataset(reference)
  predictor$set_retrieval(reference, .resolve_named_settings(list(
    k_neighbors = kNeighbors, alpha = alpha, temperature = temperature
  )), as.numeric(batchSize))
  invisible(predictor)
}


#' Fuzzy String Index
#'
#' An index over a vocabulary of strings (species names, say) that finds the
#' entries within a few edits of a query: Damerau-Levenshtein distance, where
#' an insertion, a deletion, a substitution and, with `damerau`, a swap of two
#' adjacent characters each cost one edit. Query it with [resolve.fuzzy_match()].
#'
#' @param entries Character vector, the vocabulary.
#' @param maxEditDistance The largest edit distance a query may ask for.
#' @param caseInsensitive Fold case of entries and queries before matching.
#' @param damerau Count a swap of two adjacent characters as one edit.
#' @param buckets Optional character vector, one bucket per entry (the genus of
#'   a species name, say): a query given a bucket hint searches that bucket
#'   first and falls back to the whole vocabulary when it finds nothing there.
#'
#' @details Settings left `NULL` take the engine's defaults.
#'
#' @return A `FuzzyIndex` object (an Rcpp module class) with methods `size()`,
#'   `bucket_count()` and `max_supported_distance()`.
#'
#' @examples
#' \dontrun{
#' index <- resolve.fuzzy_index(c("Quercus robur", "Quercus petraea", "Fagus sylvatica"))
#' resolve.fuzzy_match(index, c("Quercus robr", "Fagus silvatica"))
#' }
#'
#' @export
resolve.fuzzy_index <- function(entries, maxEditDistance = NULL, caseInsensitive = NULL,
                                damerau = NULL, buckets = NULL) {
  if (!is.character(entries)) stop("entries must be a character vector")
  if (!is.null(buckets) && (!is.character(buckets) || length(buckets) != length(entries))) {
    stop("buckets must be a character vector with one bucket per entry")
  }
  .resolve_require_backend()
  .resolve_module()$FuzzyIndex_build(unname(entries), .resolve_named_settings(list(
    max_edit_distance = maxEditDistance, case_insensitive = caseInsensitive,
    damerau = damerau, buckets = if (is.null(buckets)) NULL else as.list(unname(buckets))
  )))
}


#' Match Strings Against a Fuzzy Index
#'
#' The entries of a [resolve.fuzzy_index()] within `maxEditDistance` edits of
#' each needle, nearest first (ties by entry position), at most `topN` per
#' needle.
#'
#' @param index A `FuzzyIndex` from [resolve.fuzzy_index()].
#' @param needles Character vector of strings to match.
#' @param maxEditDistance Largest edit distance of a match (at most the
#'   index's `maxEditDistance`).
#' @param topN Matches kept per needle.
#' @param bucketHint One bucket to search first for every needle.
#' @param bucketHints A bucket per needle to search first (a character vector
#'   as long as `needles`); not combined with `bucketHint`.
#'
#' @details Settings left `NULL` take the engine's defaults.
#'
#' @return A data.frame with one row per match: `needle` (its position in
#'   `needles`), `query` (the needle), `entry` (the matched entry), `id` (its
#'   position in the index's entries) and `distance`. A needle with no match
#'   has no row.
#'
#' @examples
#' \dontrun{
#' index <- resolve.fuzzy_index(c("Quercus robur", "Fagus sylvatica"))
#' resolve.fuzzy_match(index, "Quercus robr", topN = 1)
#' }
#'
#' @export
resolve.fuzzy_match <- function(index, needles, maxEditDistance = NULL, topN = NULL,
                                bucketHint = NULL, bucketHints = NULL) {
  if (!inherits(index, "Rcpp_FuzzyIndex")) {
    stop("index must be built with resolve.fuzzy_index()")
  }
  if (!is.character(needles)) stop("needles must be a character vector")
  .resolve_require_backend()
  raw <- index$query(unname(needles), .resolve_named_settings(list(
    max_edit_distance = maxEditDistance, top_n = topN, bucket_hint = bucketHint,
    bucket_hints = if (is.null(bucketHints)) NULL else as.list(unname(bucketHints))
  )))
  position <- as.integer(raw$query) + 1L
  data.frame(needle = position,
             query = needles[position],
             entry = as.character(raw$entry),
             id = as.integer(raw$id) + 1L,
             distance = as.integer(raw$distance),
             stringsAsFactors = FALSE)
}


#' Cap the GPU Memory the Engine May Use
#'
#' Cap the PyTorch CUDA caching allocator at a fraction of a device's VRAM, so
#' a run leaves headroom for the desktop or other programs on a shared GPU.
#' [resolve.load()] and the `vram_fraction` training setting apply the same cap
#' per call; this sets it for the session.
#'
#' @param fraction Fraction of the device's VRAM, in (0, 1]; 1 removes the cap.
#' @param deviceIndex CUDA device; -1 (default) is the current device.
#' @return No return value. Nothing happens on a CPU-only backend or a machine
#'   without a CUDA device.
#' @examples
#' \dontrun{
#' resolve.set_vram_fraction(0.8)
#' }
#' @export
resolve.set_vram_fraction <- function(fraction, deviceIndex = -1L) {
  if (!is.numeric(fraction) || length(fraction) != 1 || fraction <= 0 || fraction > 1) {
    stop("fraction must be a single number in (0, 1]")
  }
  .resolve_require_backend()
  resolve_set_vram_fraction(fraction, as.integer(deviceIndex))
  invisible(NULL)
}


#' Predict with Dataset (C++ API)
#'
#' Make predictions on a ResolveDataset using a trained model.
#'
#' @param predictor A Predictor object from resolve.load()
#' @param dataset A ResolveDataset object from resolve.dataset.csv()
#' @param returnLatent Return latent representations (default FALSE)
#' @param batchSize Forward-pass chunk size along dim 0 (default 4096). Use -1
#'   for a single forward over the whole dataset (can OOM on large inputs); a
#'   positive value chunks the forward and concatenates results on CPU to bound
#'   peak VRAM. Matches the Python `batch_size` argument.
#'
#' @return A list with \code{predictions} (one numeric vector per target: a
#'   regression value on the original scale, or a classification target's
#'   predicted class code), \code{probabilities} (one numeric matrix per
#'   classification target, \code{n_plots x n_classes}, each row the softmax
#'   over the classes and summing to one; the columns are named by the class
#'   labels the checkpoint carries, in code order), \code{targets} (the
#'   dataset's own target values, when it carried them), \code{plot_ids}, and
#'   \code{latent} when \code{returnLatent} is \code{TRUE}. With a retrieval
#'   set ([resolve.retrieval()]) the predictions and probabilities are the
#'   blend.
#'
#' @examples
#' \dontrun{
#' predictor <- resolve.load("model.pt")
#' dataset <- resolve.dataset.csv(...)
#' preds <- resolve.predict.dataset(predictor, dataset)
#' preds$predictions$habitat            # predicted class code per plot
#' preds$probabilities$habitat[1, ]     # that plot's probability over classes
#' }
#'
#' @export
resolve.predict.dataset <- function(predictor, dataset, returnLatent = FALSE,
                                    batchSize = 4096L) {
  if (!inherits(predictor, "Rcpp_Predictor")) {
    stop("predictor must be loaded with resolve.load()")
  }
  .resolve_check_dataset(dataset)

  .resolve_require_backend()
  preds <- predictor$predict_dataset(dataset, returnLatent, as.integer(batchSize))

  # Column j of a probability matrix is P(class code j - 1); name the columns
  # by the labels the checkpoint carries so a row reads without the schema.
  schema_targets <- predictor$schema()$targets
  for (target in names(preds$probabilities)) {
    labels <- schema_targets[[target]]$class_names
    if (length(labels) == ncol(preds$probabilities[[target]])) {
      colnames(preds$probabilities[[target]]) <- labels
    }
  }
  preds
}
