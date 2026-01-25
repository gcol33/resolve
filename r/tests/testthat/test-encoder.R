# Tests for SpeciesEncoder

test_that("SpeciesEncoder can be created with defaults", {
  encoder <- resolve.encoder()

  expect_equal(encoder$hash_dim(), 32L)
  expect_equal(encoder$top_k(), 3L)
  expect_false(encoder$is_fitted())
})

test_that("SpeciesEncoder can be created with custom parameters", {
  encoder <- resolve.encoder(
    hashDim = 64L,
    topK = 5L,
    selection = "top_bottom"
  )

  expect_equal(encoder$hash_dim(), 64L)
  expect_equal(encoder$top_k(), 5L)
})

test_that("SpeciesEncoder fit and transform work", {
  skip_on_cran()

  encoder <- resolve.encoder(hashDim = 16L, topK = 2L)

  # Create test species data
  species_data <- data.frame(
    plot_id = c("p1", "p1", "p2", "p2", "p3"),
    species_id = c("sp1", "sp2", "sp1", "sp3", "sp2"),
    abundance = c(0.5, 0.5, 0.8, 0.2, 1.0),
    genus = c("Quercus", "Fagus", "Quercus", "Pinus", "Fagus"),
    family = c("Fagaceae", "Fagaceae", "Fagaceae", "Pinaceae", "Fagaceae")
  )

  # Fit
  encoder$fit(species_data)

  expect_true(encoder$is_fitted())
  expect_true(encoder$n_genera() > 0)
  expect_true(encoder$n_families() > 0)

  # Transform
  result <- encoder$transform(species_data, unique(species_data$plot_id))

  expect_equal(nrow(result$hashEmbedding), 3)  # 3 plots
  expect_equal(ncol(result$hashEmbedding), 16)  # hash_dim
})
