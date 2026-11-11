# Model suites through the R front door: build a two-target suite on disk (a
# class vote over three seeds, a mean over two), seal it, load it, and score
# data.frames and CSV paths. The combined predictions are checked against the
# members scored one at a time through resolve.predict.dataset().

test_that("a suite combines its members and reports recognition per target", {
  skip_if_no_backend()
  skip_on_cran()

  root <- local_dir("suite_")
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

  root <- local_dir("suite_")
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

# A suite whose targets read different parts of one contract: "y" reads the
# species and the coordinates, "where" the species and elev, and predicts the
# plot's latitude / longitude, combined as a spherical mean.
.build_split_suite <- function(root) {
  data <- .suite_plots(40L, with_targets = TRUE)
  coords_only <- .suite_roles[setdiff(names(.suite_roles), "covariates")]
  elev_only <- .suite_roles[setdiff(names(.suite_roles), c("latitude", "longitude"))]
  train <- function(roles, targets, seed, file) {
    ds <- resolve.dataset.frame(data$header, data$species, roles = roles,
                                targets = targets, config = .suite_config)
    dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    resolve.train.dataset(ds, hiddenDims = c(16L, 8L), maxEpochs = 2L, batchSize = 16L,
                          testSize = 0.25, seed = seed, savePath = file, verbose = FALSE)
  }
  member <- function(name, roles, targets, seed) {
    file <- sprintf("%s/seed_%d/model_final.pt", name, seed)
    train(roles, targets, seed, file.path(root, file))
    list(seed = seed, file = file)
  }
  manifest <- list(
    format = "resolve-suite", format_version = 2L, name = "r_split_suite",
    licence = "CC-BY-4.0", engine_version = resolve.version(), taxonomy = "synthetic",
    training_scope = "40 synthetic plots",
    inputs = list(plot_id = "plot", species = "taxon", abundance = "cover",
                  genus = "genus", family = "family", latitude = "lat", longitude = "lon",
                  covariates = list("elev"), zero_abundance_as = 1.0),
    targets = list(
      list(name = "y", task = "regression", combine = "mean", outputs = list("y"),
           units = "m", inputs = list(coordinates = TRUE, covariates = list()),
           status = "released",
           members = lapply(0:1, function(s) member("y", coords_only,
             list(y = list(column = "y", task = "regression")), s))),
      list(name = "where", task = "regression", combine = "spherical_mean",
           outputs = list("lat", "lon"), units = "degrees",
           inputs = list(coordinates = FALSE, covariates = list("elev")),
           status = "released",
           members = lapply(2:4, function(s) member("where", elev_only,
             list(lat = list(column = "lat", task = "regression"),
                  lon = list(column = "lon", task = "regression")), s)))
    )
  )
  jsonlite::write_json(manifest, file.path(root, "manifest.json"), auto_unbox = TRUE,
                       pretty = TRUE, digits = NA)
  resolve.seal_suite(root)
  list(coords_only = coords_only, elev_only = elev_only)
}

test_that("each target of a suite reads its own columns; a location is a spherical mean", {
  skip_if_no_backend()
  skip_on_cran()

  root <- local_dir("suite_")
  roles <- .build_split_suite(root)

  suite <- resolve.load_suite(root)
  expect_equal(suite$n_encodings(), 2L)
  manifest <- suite$manifest()
  expect_equal(manifest$format_version, 2)
  expect_false(manifest$targets[[2]]$inputs$coordinates)

  score <- .suite_plots(40L, with_targets = FALSE, extra = TRUE)
  preds <- resolve.predict.suite(suite, species = score$species, header = score$header,
                                 keepMembers = TRUE)
  where <- preds$targets$where
  expect_equal(colnames(where$value), c("latitude", "longitude"))
  expect_equal(nrow(where$value), nrow(score$header))
  expect_equal(where$dispersion_units, "km")
  expect_equal(names(where$members), c("latitude", "longitude"))
  expect_equal(colnames(where$members$latitude), paste0("seed", 2:4))

  # The combined point against the members scored one at a time, each through
  # the columns it was trained on, averaged as unit vectors.
  members <- lapply(manifest$targets[[2]]$members, function(m) {
    predictor <- resolve.load(file.path(root, m$file))
    cfg <- predictor$dataset_config()
    cfg$zero_abundance_as <- 1.0
    ds <- resolve.dataset.frame(score$header, score$species, roles = roles$elev_only,
                                targets = list(), vocabs = predictor$vocabs(), config = cfg)
    p <- resolve.predict.dataset(predictor, ds)
    i <- match(preds$plot_ids, p$plot_ids)
    cbind(p$predictions$lat[i], p$predictions$lon[i])
  })
  rad <- pi / 180
  xyz <- Reduce(`+`, lapply(members, function(m) {
    cbind(cos(m[, 1] * rad) * cos(m[, 2] * rad), cos(m[, 1] * rad) * sin(m[, 2] * rad),
          sin(m[, 1] * rad))
  }))
  expect_equal(unname(where$value[, "latitude"]),
               atan2(xyz[, 3], sqrt(xyz[, 1]^2 + xyz[, 2]^2)) / rad, tolerance = 1e-4)
  expect_equal(unname(where$value[, "longitude"]), atan2(xyz[, 2], xyz[, 1]) / rad,
               tolerance = 1e-4)
  expect_equal(unname(where$members$latitude[, "seed3"]), members[[2]][, 1], tolerance = 1e-5)

  flat <- as.data.frame(preds)
  for (col in c("where_latitude", "where_longitude", "where_spread_km",
                "where_seed2_latitude", "where_seed4_longitude", "y", "y_sd")) {
    expect_true(col %in% names(flat), info = col)
  }
})

test_that("suite gap fitting adds fitted columns and a report per fitted target", {
  skip_if_no_backend()
  skip_on_cran()

  root <- local_dir("suite_gaps_")
  .build_suite(root)
  suite <- resolve.load_suite(root)
  data <- .suite_plots(40L, with_targets = TRUE)
  header <- data$header
  header$y_rec <- ifelse(seq_len(nrow(header)) %% 4L == 1L, NA, header$y)
  header$year <- 1980 + seq_len(nrow(header))
  header$hab <- NULL
  header$y <- NULL

  preds <- suppressMessages(resolve.predict.suite(
    suite, species = data$species, header = header,
    columns = list(recorded = list(y = "y_rec"), survey_year = "year"), gapFitting = TRUE))
  fit <- preds$targets$y$gap_fit
  expect_s3_class(fit, "resolve_gap_fit")
  expect_equal(fit$report$n_gaps, 10)
  expect_true(fit$report$axes$years$matched)
  expect_null(preds$targets$hab$gap_fit)
  frame <- as.data.frame(preds)
  expect_true(all(c("y_fitted", "y_fit_method") %in% names(frame)))
  expect_equal(frame$y_fit_method[2], "recorded")
  expect_false(frame$y_fit_method[1] == "recorded")
  expect_equal(frame$y_fitted[2], header$y_rec[2], tolerance = 1e-4)

  restricted <- suppressMessages(resolve.predict.suite(
    suite, species = data$species, header = header,
    columns = list(recorded = list(y = "y_rec")),
    gapFitting = list(candidates = c("local_constant", "offset_slope"), k_grid = c(1, 3))))
  expect_true(restricted$targets$y$gap_fit$report$groups$located$method %in%
                c("local_constant", "offset_slope"))
  expect_error(suppressMessages(resolve.predict.suite(
    suite, species = data$species, header = header,
    columns = list(recorded = list(hab = "y_rec")), gapFitting = TRUE)), "regression")
  expect_false("y_recorded" %in% names(frame))

  # Without gap fitting, a recorded column flags the plots that record it.
  flagged <- suppressMessages(resolve.predict.suite(
    suite, species = data$species, header = header,
    columns = list(recorded = list(y = "y_rec"))))
  expect_null(flagged$targets$y$gap_fit)
  expect_null(flagged$targets$hab$recorded)
  expect_identical(flagged$targets$y$recorded, !is.na(header$y_rec))
  flat <- as.data.frame(flagged)
  expect_type(flat$y_recorded, "logical")
  expect_identical(flat$y_recorded, !is.na(header$y_rec))
  expect_identical(match("y_recorded", names(flat)), match("y_sd", names(flat)) + 1L)

  # The single-model path over one member's predictions.
  member <- file.path(root, suite$manifest()$targets[[2]]$members[[1]]$file)
  predictor <- resolve.load(member)
  cfg <- predictor$dataset_config()
  cfg$zero_abundance_as <- 1.0
  ds <- resolve.dataset.frame(header, data$species, roles = .suite_roles, targets = list(),
                              vocabs = predictor$vocabs(), config = cfg)
  single <- resolve.fit_prediction_gaps(
    resolve.predict.dataset(predictor, ds), data$species, header,
    columns = list(plot_id = "plot", species_id = "taxon", latitude = "lat", longitude = "lon",
                   recorded = list(y = "y_rec")),
    targets = list(y = list(scale = "linear")))
  expect_equal(single$y$report$n_gaps, 10)
  expect_equal(sum(single$y$method == "recorded"), 30)
})

test_that("a plot with no recognised species is not imputed", {
  skip_if_no_backend()
  skip_on_cran()

  root <- local_dir("suite_alien_")
  .build_suite(root)
  suite <- resolve.load_suite(root)
  data <- .suite_plots(40L, with_targets = FALSE)
  header <- rbind(data$header, data.frame(plot = "ALIEN", lat = 46.3, lon = 9.2, elev = 140))
  header$y_rec <- c(ifelse(seq_len(40L) %% 4L == 1L, NA, 1 + 0.2 * (seq_len(40L) - 1L)), NA)
  species <- rbind(data$species, data.frame(plot = "ALIEN", taxon = c("sp_q", "sp_r"),
                                            cover = c(2, 1), genus = c("gq", "gr"),
                                            family = "fam", stringsAsFactors = FALSE))

  preds <- suppressMessages(resolve.predict.suite(suite, species = species, header = header,
                                                  keepMembers = TRUE))
  a <- match("ALIEN", preds$plot_ids)
  for (name in names(preds$targets)) {
    imputed <- preds$targets[[name]]$imputed
    expect_type(imputed, "logical")
    expect_false(imputed[a], info = name)
    expect_equal(sum(!imputed), 1L, info = name)
  }
  expect_true(is.na(preds$targets$hab$label[a]))
  expect_equal(preds$targets$hab$value[a], -1L)
  expect_true(is.na(preds$targets$y$value[a]))
  expect_true(all(is.na(preds$targets$y$members[a, ])))
  flat <- as.data.frame(preds)
  expect_true(is.na(flat$hab[a]))
  expect_false(flat$hab_imputed[a])
  expect_true(flat$y_imputed[1])

  fitted <- suppressMessages(resolve.predict.suite(
    suite, species = species, header = header,
    columns = list(recorded = list(y = "y_rec")), gapFitting = TRUE))
  fit <- fitted$targets$y$gap_fit
  expect_equal(fit$report$n_not_imputed, 1)
  expect_equal(fit$report$n_gaps, 10)
  expect_equal(fit$method[a], "not_imputed")
  expect_true(is.na(fit$fitted[a]))
})

test_that("suite and dataset names resolve per vocabulary and predict as the tokens", {
  skip_if_no_backend()
  skip_on_cran()

  root <- local_dir("suite_names_")
  .build_suite(root)
  suite <- resolve.load_suite(root)
  score <- .suite_plots(40L, with_targets = FALSE, extra = TRUE)
  written <- score$species
  written$taxon[written$taxon == "sp1"] <- "Species one entered"
  written$taxon[written$taxon == "sp_late"] <- "sp_latte"
  table <- local_csv("")
  writeLines('{"source": "test", "aliases": {"Species one entered": ["sp1", "eva"]}}', table)

  expected <- suppressMessages(resolve.predict.suite(suite, score$species, score$header))
  resolved <- suppressMessages(resolve.predict.suite(suite, written, score$header,
                                                     names = table))
  plain <- suppressMessages(resolve.predict.suite(suite, written, score$header))
  for (target in c("hab", "y")) {
    expect_equal(resolved$targets[[target]]$value, expected$targets[[target]]$value)
    expect_null(plain$targets[[target]]$species_names)
  }
  hab <- resolved$targets$hab$species_names
  expect_equal(hab$name, c("Species one entered", "sp_latte", "sp_unseen"))
  expect_equal(hab$route, c("alias", "unresolved", "unresolved"))
  y <- resolved$targets$y$species_names
  expect_equal(y$token, c("sp1", "sp_late", "sp_unseen"))
  expect_equal(y$route, c("alias", "spelling", "unresolved"))
  expect_equal(y$n_records[2], sum(score$species$taxon == "sp_late"))
  expect_equal(resolve.names("sp_latte", suite, target = "y")$token, "sp_late")
  expect_equal(resolve.names("sp_latte", suite, target = "hab")$route, "unresolved")
  expect_error(resolve.names("sp_latte", suite), "target")

  member <- file.path(root, suite$manifest()$targets[[2]]$members[[1]]$file)
  predictor <- resolve.load(member)
  cfg <- predictor$dataset_config()
  cfg$zero_abundance_as <- 1.0
  ds <- resolve.dataset.frame(score$header, written, roles = .suite_roles, targets = list(),
                              vocabs = predictor$vocabs(), config = cfg, names = table)
  report <- ds$species_names()
  expect_equal(unlist(report$route), c("alias", "spelling", "unresolved"))
  expect_equal(resolve.names("sp_latte", predictor)$token, "sp_late")
  expect_error(resolve.dataset.frame(score$header, written, roles = .suite_roles,
                                     targets = list(), config = cfg, names = table), "vocab")
})
