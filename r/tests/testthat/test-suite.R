# Model suites through the R front door: build a two-target suite on disk (a
# class vote over three seeds, a mean over two), seal it, load it, and score
# data.frames and CSV paths. The combined predictions are checked against the
# members scored one at a time through resolve.predict.dataset().

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

test_that("a suite combines its members and reports recognition per target", {
  skip_if_no_backend()
  skip_on_cran()

  root <- tempfile("suite_")
  dir.create(root)
  on.exit(unlink(root, recursive = TRUE), add = TRUE)
  sealed <- .build_suite(root)
  expect_equal(length(sealed$targets[[1]]$members), 3L)
  expect_match(sealed$targets[[1]]$members[[1]]$sha256, "^[0-9a-f]{64}$")

  check <- resolve.verify_suite(root)
  expect_length(check$problems, 0)

  suite <- resolve.load_suite(root)
  expect_equal(suite$targets(), c("hab", "y"))
  expect_equal(suite$n_encodings(), 2L)

  score <- .suite_plots(40L, with_targets = FALSE, extra = TRUE)
  expect_message(
    preds <- resolve.predict.suite(suite, species = score$species, header = score$header,
                                   keepMembers = TRUE),
    "'y' is released as experimental: synthetic plots only"
  )
  expect_s3_class(preds, "resolve_suite_predictions")
  expect_equal(preds$plot_ids, score$header$plot)

  hab <- preds$targets$hab
  expect_equal(colnames(hab$probabilities), c("A", "B", "C"))
  expect_equal(unname(rowSums(hab$probabilities)), rep(1, nrow(score$header)),
               tolerance = 1e-5)
  expect_true(all(hab$label %in% c("A", "B", "C")))
  expect_equal(colnames(hab$members), paste0("seed", 0:2))
  # Vote and agreement, recomputed from the members' own labels: among the
  # classes holding the most votes, the one with the higher mean probability.
  votes <- vapply(seq_len(nrow(hab$members)), function(i) {
    counts <- table(factor(hab$members[i, ], levels = c("A", "B", "C")))
    leading <- names(counts)[counts == max(counts)]
    leading[which.max(hab$probabilities[i, leading])]
  }, character(1))
  agreement <- apply(hab$members, 1, function(row) max(table(row)) / 3)
  expect_equal(hab$label, votes)
  expect_equal(hab$agreement, unname(agreement), tolerance = 1e-6)

  # The mean target against its members scored one at a time.
  y <- preds$targets$y
  manifest <- suite$manifest()
  member_values <- sapply(manifest$targets[[2]]$members, function(m) {
    predictor <- resolve.load(file.path(root, m$file))
    cfg <- predictor$dataset_config()
    cfg$zero_abundance_as <- 1.0
    ds <- resolve.dataset.frame(score$header, score$species, roles = .suite_roles,
                                targets = list(), vocabs = predictor$vocabs(), config = cfg)
    p <- resolve.predict.dataset(predictor, ds)
    p$predictions$y[match(preds$plot_ids, p$plot_ids)]
  })
  expect_equal(y$value, rowMeans(member_values), tolerance = 1e-5)
  expect_equal(y$dispersion, apply(member_values, 1, sd), tolerance = 1e-4)
  expect_equal(y$units, "m")

  # sp_late is missing from hab's vocabulary only; sp_unseen from both.
  late <- match("P35", preds$plot_ids)
  expect_equal(hab$n_recognised[late], hab$n_species[late] - 1)
  expect_equal(y$recognised_share[late], 1)
  fresh <- match("NEW", preds$plot_ids)
  expect_equal(y$recognised_share[fresh], 0.5)
  expect_equal(y$recognised_abundance_share[fresh], 0.75, tolerance = 1e-6)

  flat <- as.data.frame(preds, probabilities = TRUE)
  expect_equal(nrow(flat), nrow(score$header))
  for (col in c("plot_id", "hab", "hab_code", "hab_agreement", "hab_prob_A", "y", "y_sd",
                "y_recognised_share", "y_seed3")) {
    expect_true(col %in% names(flat), info = col)
  }

  # CSV paths score like the data.frames.
  h_file <- file.path(root, "score_header.csv")
  s_file <- file.path(root, "score_species.csv")
  write.csv(score$header, h_file, row.names = FALSE)
  write.csv(score$species, s_file, row.names = FALSE)
  from_csv <- suppressMessages(resolve.predict.suite(suite, species = s_file, header = h_file))
  expect_equal(from_csv$targets$y$value, y$value, tolerance = 1e-6)
  expect_equal(from_csv$targets$hab$label, hab$label)
})

test_that("the suite front doors reject what the manifest does not allow", {
  skip_if_no_backend()
  skip_on_cran()

  root <- tempfile("suite_")
  dir.create(root)
  on.exit(unlink(root, recursive = TRUE), add = TRUE)
  .build_suite(root)
  score <- .suite_plots(10L, with_targets = FALSE)

  suite <- resolve.load_suite(root, targets = "y")
  expect_equal(suite$targets(), "y")
  expect_error(resolve.predict.suite(suite, species = score$species),
               "header table is required")
  expect_error(resolve.predict.suite(suite, species = score$species, header = score$header,
                                     columns = list(species = "taxon")),
               "unknown key 'species'")
  renamed <- suppressMessages(resolve.predict.suite(
    suite, species = score$species,
    header = setNames(score$header, c("plot", "lat", "lon", "altitude")),
    columns = list(covariates = "altitude")))
  base <- suppressMessages(resolve.predict.suite(suite, species = score$species,
                                                 header = score$header))
  expect_equal(renamed$targets$y$value, base$targets$y$value)

  expect_error(resolve.load_suite(root, targets = "nope"), "has no target 'nope'")
  expect_error(resolve.load_suite(file.path(root, "missing")), "does not exist")

  member <- file.path(root, "y", "seed_3", "model_final.pt")
  bytes <- readBin(member, "raw", file.size(member))
  bytes[200] <- as.raw(bitwXor(as.integer(bytes[200]), 255L))
  writeBin(bytes, member)
  expect_length(resolve.verify_suite(root)$problems, 1)
  expect_error(resolve.load_suite(root), "SHA-256")
  expect_s4_class(resolve.load_suite(root, verify = FALSE, targets = "hab"), "Rcpp_Suite")
})
