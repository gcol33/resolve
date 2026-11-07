# Gap fitting through the R front door: known relations between the raw
# prediction, the neighbours and the truth, recovered by resolve.fit_gaps().

test_that("offset_slope recovers the slope of the local relation", {
  skip_if_no_backend()
  for (seed in 1:3) {
    for (slope in c(0, 0.4, 1)) {
      set.seed(seed)
      pred <- runif(3000, 1950, 2020)
      truth <- 1990 + slope * (pred - mean(pred)) + 8 + rnorm(3000, 0, 3)
      recorded <- truth
      recorded[2401:3000] <- NA
      fit <- resolve.fit_gaps(pred, recorded, matrix(0, 3000, 1), name = "year",
                              gapFitting = list(candidates = "offset_slope"))
      group <- fit$report$groups$unlocated
      expect_equal(group$method, "offset_slope")
      expect_equal(group$coefficients$resolve, slope, tolerance = 0.02)
      expect_equal(fit$method[1:2400], rep("recorded", 2400))
      expect_equal(fit$fitted[1:2400], truth[1:2400])
    }
  }
})

test_that("a turn of bearings is recovered as a complex coefficient", {
  skip_if_no_backend()
  set.seed(2)
  truth <- runif(2000, 0, 360)
  pred <- (truth - 40 + rnorm(2000, 0, 5)) %% 360
  recorded <- truth
  recorded[1501:2000] <- NA
  fit <- resolve.fit_gaps(pred, recorded, matrix(0, 2000, 1), scale = "circular", period = 360,
                          gapFitting = list(candidates = "offset_slope"))
  turn <- fit$report$groups$unlocated$coefficients$resolve
  expect_equal(turn$angle_deg, 40, tolerance = 1 / 40)
  expect_equal(turn$modulus, 1, tolerance = 0.05)
})

test_that("spatial neighbours are chosen where the gaps are near", {
  skip_if_no_backend()
  set.seed(3)
  n <- 3000
  lat <- runif(n, 45, 50)
  lon <- runif(n, 14, 14.5)
  truth <- 1985 + 30 * sin(2 * pi * lat / 0.5) + rnorm(n)
  pred <- truth + rnorm(n, 0, 6)
  recorded <- truth
  gaps <- runif(n) < 0.2
  recorded[gaps] <- NA
  fit <- resolve.fit_gaps(pred, recorded, matrix(0, n, 1), coordinates = cbind(lat, lon),
                          name = "altitude")
  expect_true(fit$report$groups$located$method %in% c("knn_spatial", "stack_species_spatial"))
  expect_lt(mean(abs(fit$fitted[gaps] - truth[gaps])), mean(abs(pred[gaps] - truth[gaps])))
})

test_that("bad arguments are refused before the engine", {
  expect_error(resolve.fit_gaps(1:3, 1:3, 1:3), "matrix")
  expect_error(resolve.fit_gaps(1:3, 1:3, matrix(0, 3, 1), gapFitting = list(1)), "named list")
})
