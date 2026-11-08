# The two-target suite the suite and terrain tests build on disk: a class vote
# over three seeds and a mean over two, trained on synthetic plots.

.suite_plots <- function(n, with_targets, extra = FALSE) {
  i <- seq_len(n) - 1L
  header <- data.frame(
    plot = paste0("P", i),
    lat = 46 + 0.05 * i,
    lon = 9 + 0.03 * i,
    elev = 100 + 5 * i,
    stringsAsFactors = FALSE
  )
  if (with_targets) {
    header$hab <- c("A", "B", "C")[i %% 3L + 1L]
    header$y <- 1 + 0.2 * i
  }
  species <- do.call(rbind, lapply(i, function(p) {
    taxa <- paste0("sp", (p + 0:2) %% 6L)
    if (p >= 30L) taxa <- c(taxa, "sp_late")
    data.frame(plot = paste0("P", p), taxon = taxa,
               cover = c(1, 2, 0, 4)[seq_along(taxa)], genus = paste0("g", substr(taxa, 3, 3)),
               family = "fam", stringsAsFactors = FALSE)
  }))
  if (extra) {
    header <- rbind(header, data.frame(plot = "NEW", lat = 46.5, lon = 9.5, elev = 150))
    species <- rbind(species, data.frame(plot = "NEW", taxon = c("sp1", "sp_unseen"),
                                         cover = c(3, 1), genus = c("g1", "gz"),
                                         family = "fam", stringsAsFactors = FALSE))
  }
  list(header = header, species = species)
}

.suite_roles <- list(plot_id = "plot", species_id = "taxon", abundance = "cover",
                     genus = "genus", family = "family", latitude = "lat",
                     longitude = "lon", covariates = "elev")

.suite_config <- list(species_encoding = "rank_pool", pool_weighting = "log1p",
                      zero_abundance_as = 1.0)

.build_suite <- function(root) {
  data <- .suite_plots(40L, with_targets = TRUE)
  early <- list(header = data$header[1:30, ],
                species = data$species[data$species$plot %in% data$header$plot[1:30], ])

  train <- function(frames, targets, seed, file) {
    ds <- resolve.dataset.frame(frames$header, frames$species, roles = .suite_roles,
                                targets = targets, config = .suite_config)
    dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    resolve.train.dataset(ds, hiddenDims = c(16L, 8L), maxEpochs = 2L, batchSize = 16L,
                          testSize = 0.25, seed = seed, savePath = file, verbose = FALSE)
  }
  targets <- list(
    list(name = "hab", task = "classification", combine = "vote", outputs = list("hab"),
         status = "released", frames = early, seeds = 0:2,
         spec = list(hab = list(column = "hab", task = "classification"))),
    list(name = "y", task = "regression", combine = "mean", outputs = list("y"),
         units = "m", status = "experimental", limit = "synthetic plots only",
         frames = data, seeds = 3:4,
         spec = list(y = list(column = "y", task = "regression")))
  )
  manifest_targets <- lapply(targets, function(t) {
    members <- lapply(t$seeds, function(seed) {
      file <- sprintf("%s/seed_%d/model_final.pt", t$name, seed)
      train(t$frames, t$spec, seed, file.path(root, file))
      list(seed = seed, file = file)
    })
    entry <- t[setdiff(names(t), c("frames", "seeds", "spec"))]
    entry$members <- members
    entry$validation <- list(canonical = list(metric = 0.5))
    entry
  })
  manifest <- list(
    format = "resolve-suite", format_version = 1L, name = "r_suite",
    description = "two targets, two vocabularies", licence = "CC-BY-4.0",
    engine_version = resolve.version(), taxonomy = "synthetic",
    training_scope = "40 synthetic plots",
    inputs = list(plot_id = "plot", species = "taxon", abundance = "cover",
                  genus = "genus", family = "family", latitude = "lat", longitude = "lon",
                  covariates = list("elev"), abundance_units = "percent cover",
                  zero_abundance_as = 1.0),
    targets = manifest_targets
  )
  jsonlite::write_json(manifest, file.path(root, "manifest.json"), auto_unbox = TRUE,
                       pretty = TRUE, digits = NA)
  resolve.seal_suite(root)
}
