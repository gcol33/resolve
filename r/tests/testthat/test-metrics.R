# Tests for RESOLVE metrics functions

test_that("resolve_mae computes mean absolute error", {
  pred <- c(1.0, 2.0, 3.0, 4.0, 5.0)
  target <- c(1.5, 2.0, 2.5, 4.5, 5.0)

  mae <- resolve_mae(pred, target)

  expect_equal(mae, 0.3, tolerance = 1e-5)
})

test_that("resolve_rmse computes root mean squared error", {
  pred <- c(1.0, 2.0, 3.0)
  target <- c(2.0, 2.0, 4.0)

  rmse <- resolve_rmse(pred, target)

  expect_equal(rmse, 0.8165, tolerance = 0.001)
})

test_that("resolve_r_squared returns 1 for perfect fit", {
  vals <- c(1.0, 2.0, 3.0, 4.0, 5.0)

  r2 <- resolve_r_squared(vals, vals)

  expect_equal(r2, 1.0)
})

test_that("resolve_r_squared computes correct value for good fit", {
  pred <- c(1.0, 2.0, 3.0, 4.0, 5.0)
  target <- c(1.1, 2.0, 2.9, 4.1, 5.0)

  r2 <- resolve_r_squared(pred, target)

  expect_true(r2 > 0.95)
})

test_that("resolve_smape computes symmetric mean absolute percentage error", {
  pred <- c(100.0, 200.0, 300.0)
  target <- c(110.0, 200.0, 280.0)

  smape <- resolve_smape(pred, target)

  expect_true(smape > 0.0)
  expect_true(smape < 0.1)
})

test_that("resolve_band_accuracy computes band accuracy", {
  pred <- c(100.0, 200.0, 300.0, 400.0)
  target <- c(100.0, 180.0, 250.0, 500.0)

  acc_25 <- resolve_band_accuracy(pred, target, 0.25)
  acc_10 <- resolve_band_accuracy(pred, target, 0.10)

  # Wider band should have >= accuracy
  expect_true(acc_25 >= acc_10)
})
