# Test metrics R bindings

test_that("band_accuracy computes correctly", {
  pred <- c(100, 200, 300, 400)
  target <- c(100, 180, 400, 600)

  # 25% band: |pred - target| / |target| <= 0.25
  # 100 vs 100: 0% error - in band
  # 200 vs 180: 11% error - in band
  # 300 vs 400: 25% error - in band
  # 400 vs 600: 33% error - out of band
  result <- band_accuracy(pred, target, threshold = 0.25)
  expect_equal(result, 0.75, tolerance = 0.01)
})

test_that("band_accuracy handles NA values", {
  pred <- c(100, NA, 300)
  target <- c(100, 200, 300)

  # Only 2 valid pairs
  result <- band_accuracy(pred, target, threshold = 0.25)
  expect_equal(result, 1.0)  # Both valid pairs are within 25%
})

test_that("mae computes correctly", {
  pred <- c(1, 2, 3)
  target <- c(1, 3, 5)

  # MAE = (0 + 1 + 2) / 3 = 1.0
  result <- mae(pred, target)
  expect_equal(result, 1.0)
})

test_that("rmse computes correctly", {
  pred <- c(1, 2, 3)
  target <- c(1, 3, 5)

  # RMSE = sqrt((0 + 1 + 4) / 3) = sqrt(5/3) ≈ 1.29
  result <- rmse(pred, target)
  expect_equal(result, sqrt(5/3), tolerance = 0.01)
})

test_that("smape computes correctly", {
  pred <- c(100, 200)
  target <- c(110, 180)

  result <- smape(pred, target)

  # SMAPE should be between 0 and 2
  expect_gte(result, 0)
  expect_lte(result, 2)
})

test_that("compute_metrics returns all metrics", {
  pred <- c(1, 2, 3, 4, 5)
  target <- c(1.1, 2.2, 2.8, 4.5, 4.8)

  result <- compute_metrics(pred, target)

  expect_true("mae" %in% names(result))
  expect_true("rmse" %in% names(result))
  expect_true("smape" %in% names(result))
  expect_true("band_10" %in% names(result))
  expect_true("band_25" %in% names(result))
  expect_true("band_50" %in% names(result))

  # All values should be numeric
  expect_type(result$mae, "double")
  expect_type(result$rmse, "double")
  expect_type(result$smape, "double")
})

test_that("metrics handle equal vectors", {
  pred <- c(1, 2, 3)
  target <- c(1, 2, 3)

  expect_equal(mae(pred, target), 0)
  expect_equal(rmse(pred, target), 0)
  expect_equal(band_accuracy(pred, target, 0.01), 1.0)
})
