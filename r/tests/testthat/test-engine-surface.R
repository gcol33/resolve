# The engine surface the R verbs reach: strict configuration lists, devices,
# engine defaults, the full model / training configuration, cross-validation,
# pretraining, retrieval, the fuzzy index, the VRAM cap and int64 accessors.

signal_dataset <- function(n = 60L, config = small_hash_config(), env = parent.frame()) {
  load_tables(signal_tables(n), signal_roles(), regression_target(), config = config, env = env)
}

train_quietly <- function(dataset, ...) {
  resolve.train.dataset(dataset, hiddenDims = c(16L, 8L), maxEpochs = 2L, batchSize = 16L,
                        verbose = FALSE, ...)
}

test_that("an unknown model or training key is refused at any depth", {
  skip_if_no_backend()
  dataset <- signal_dataset()
  expect_error(train_quietly(dataset, modelConfig = list(hidden_dimz = 8L)), "'hidden_dimz'")
  expect_error(train_quietly(dataset, trainConfig = list(max_epoch = 2L)), "'max_epoch'")
  expect_error(
    train_quietly(dataset, modelConfig = list(encoder_architecture = "ft_transformer",
                                              ft_transformer = list(d_modle = 16L))),
    "model config.ft_transformer")
  expect_error(train_quietly(dataset, lossConfig = "mse"), "loss_config")
  expect_error(train_quietly(dataset, moeRouting = "dense"), "moe_routing")
  expect_error(train_quietly(dataset, moeRouting = "soft", moePlacement = "middle"),
               "moe_placement")
  expect_error(train_quietly(dataset, device = "gpu"), "cuda:N")
  expect_error(train_quietly(dataset, modelConfig = list(hidden_dims = c(8L))),
               "given twice")
  # A list concatenated with c() repeats a key; only one value could be read.
  expect_error(signal_dataset(config = c(small_hash_config(), list(hash_dim = 8))),
               "'hash_dim' is given twice")
})

test_that("unset arguments take the engine's defaults", {
  skip_if_no_backend()
  train <- resolve.default_config("train")
  model <- resolve.default_config("model")
  expect_true(train$max_epochs > 0)
  expect_equal(train$device, "cpu")
  expect_equal(model$species_encoding, "hash")
  for (which in c("dataset", "spatial_block", "retrieval", "pretrain", "masked_species", "vae")) {
    expect_type(resolve.default_config(which), "list")
  }
  expect_error(resolve.default_config("modle"), "model")

  dataset <- signal_dataset()
  fit <- resolve.train.dataset(dataset, maxEpochs = 1L, verbose = FALSE)
  config <- fit$trainer$get_config()
  expect_equal(config$lr, train$lr)
  expect_equal(config$patience, train$patience)
  expect_equal(config$batch_size, train$batch_size)
  expect_equal(config$loss_config, train$loss_config)
})

test_that("the model is sized for its dataset", {
  skip_if_no_backend()
  dataset <- signal_dataset(config = small_hash_config(hash_dim = 6, top_k = 3))
  fit <- train_quietly(dataset, modelConfig = list(dropout = 0.1))
  model <- resolveR:::.resolve_module()$ResolveModel_for_dataset(dataset, list(hash_dim = 99L))
  expect_equal(model$config()$hash_dim, 6L)
  expect_equal(model$config()$top_k, 3L)
})

test_that("modelConfig reaches every architecture, the heterogeneous GNN included", {
  skip_if_no_backend()
  skip_on_cran()
  dataset <- signal_dataset()
  ft <- train_quietly(dataset, modelConfig = list(
    encoder_architecture = "ft_transformer",
    ft_transformer = list(d_model = 16L, n_heads = 2L, n_layers = 1L)))
  expect_length(ft$result$train_loss, 2L)

  # Three species per plot, so species share plots and the graph has
  # co-occurrence edges.
  tables <- signal_tables(40L)
  i <- rep(seq_len(40L), each = 3L)
  tables$species <- data.frame(plot_id = paste0("P", i),
                               species_id = paste0("sp", (i + rep(0:2, 40L)) %% 6L),
                               cover = 1.0, stringsAsFactors = FALSE)
  sparse <- load_tables(tables, signal_roles(), regression_target(),
                        config = list(species_encoding = "sparse"))
  gnn <- train_quietly(sparse, modelConfig = list(
    encoder_architecture = "heterogeneous_gnn",
    heterogeneous_gnn = list(hidden_dim = 8L, output_dim = 8L, n_layers = 1L,
                             n_heads = 2L, use_taxonomic_edges = FALSE,
                             cooccurrence_threshold = 0)))
  expect_length(gnn$result$train_loss, 2L)
})

test_that("cross-validation runs random and spatial folds", {
  skip_if_no_backend()
  skip_on_cran()
  dataset <- signal_dataset(80L)
  cv <- resolve.cross_validate(dataset, nFolds = 3L, hiddenDims = c(8L), maxEpochs = 2L,
                               batchSize = 16L)
  expect_equal(cv$n_folds, 3L)
  expect_length(cv$fold_results, 3L)
  expect_true(is.finite(cv$mean_metrics$y$mae))

  located <- load_tables(signal_tables(80L),
                         signal_roles(latitude = "lat", longitude = "lon"),
                         regression_target())
  # Layer norm: a spatial fold's last batch may hold a single plot.
  spatial <- resolve.cross_validate(located, nFolds = 2L, hiddenDims = c(8L), maxEpochs = 2L,
                                    batchSize = 16L,
                                    modelConfig = list(normalization = "layer_norm"),
                                    spatial = list(lat_size = 0.01, lon_size = 0.01))
  expect_equal(spatial$n_folds, 2L)
  expect_error(resolve.cross_validate(dataset, nFolds = 2L, spatial = list(lat = 1)), "'lat'")
  expect_error(resolve.cross_validate(dataset, maxEpoch = 2L), "maxEpoch")
})

test_that("pretraining runs before training, and the VAE trains on species vectors", {
  skip_if_no_backend()
  skip_on_cran()
  dataset <- signal_dataset()
  fit <- train_quietly(dataset, pretrain = list(method = "scarf", pretrain_epochs = 2L,
                                                batch_size = 16L))
  expect_length(fit$pretrain$loss_history, 2L)
  expect_error(train_quietly(dataset, pretrain = list(method = "bert")), "jepa")
  expect_error(train_quietly(dataset, pretrain = list(method = "jepa", mask_ratioo = 0.2)),
               "'mask_ratioo'")

  sparse <- load_tables(signal_tables(40L), signal_roles(), regression_target(),
                        config = list(species_encoding = "sparse"))
  vae <- resolve.pretrain_vae(sparse, list(latent_dim = 4L, encoder_dims = c(8L),
                                           decoder_dims = c(8L), pretrain_epochs = 2L))
  expect_length(vae$loss_history, 2L)
  expect_equal(vae$latent_dim, 4L)
  expect_equal(nrow(vae$projection_weights), 8L)
  expect_error(resolve.pretrain_vae(dataset), "sparse")
})

test_that("retrieval with k = 1 and alpha = 1 returns the reference plots' labels", {
  skip_if_no_backend()
  skip_on_cran()
  dataset <- signal_dataset()
  model_file <- local_model_file()
  train_quietly(dataset, savePath = model_file)
  predictor <- resolve.load(model_file)
  expect_null(predictor$retrieval())

  resolve.retrieval(predictor, dataset, kNeighbors = 1L, alpha = 1)
  expect_equal(predictor$retrieval()$n_reference, 60L)
  preds <- resolve.predict.dataset(predictor, dataset)
  expect_equal(preds$predictions$y, dataset$targets()$y, tolerance = 1e-4)

  resolve.retrieval(predictor, NULL)
  expect_null(predictor$retrieval())
  expect_error(resolve.retrieval(predictor, dataset, kNeighbors = 0L), "k_neighbors")
})

test_that("the fuzzy index matches by edit distance, with buckets", {
  skip_if_no_backend()
  index <- resolve.fuzzy_index(c("Quercus robur", "Quercus petraea", "Fagus sylvatica"),
                               buckets = c("Quercus", "Quercus", "Fagus"))
  expect_equal(index$size(), 3L)
  matches <- resolve.fuzzy_match(index, c("Quercus robr", "Fagus silvatica", "Zzz"),
                                 topN = 1L, bucketHints = c("Quercus", "Fagus", "Zzz"))
  expect_equal(matches$needle, c(1L, 2L))
  expect_equal(matches$entry, c("Quercus robur", "Fagus sylvatica"))
  expect_equal(matches$id, c(1L, 3L))
  expect_equal(matches$distance, c(1L, 1L))
  expect_error(resolve.fuzzy_index(c("a", "b"), buckets = "x"), "one bucket per entry")
  expect_error(resolve.fuzzy_match(index, "x", topn = 1L))
})

test_that("set_vram_fraction is reachable and checks its range", {
  skip_if_no_backend()
  expect_silent(resolve.set_vram_fraction(1))
  expect_error(resolve.set_vram_fraction(0), "fraction")
  expect_error(resolve.set_vram_fraction(1.5), "fraction")
})

test_that("int64 values beyond R's integer range come back as exact doubles", {
  skip_if_no_backend()
  dataset <- signal_dataset(config = small_hash_config(use_cuda_hash = TRUE))
  expect_true(dataset$has_raw_species_data())
  ids <- dataset$raw_species_ids()
  offsets <- dataset$plot_offsets()
  expect_length(offsets, 61L)
  # The raw IDs are 64-bit hash keys: past R's integer range they arrive as
  # doubles, unrounded and without a warning per element.
  expect_type(ids, "double")
  expect_true(any(abs(ids) > .Machine$integer.max))
  expect_true(all(ids == round(ids)))
  expect_type(offsets, "integer")
})
