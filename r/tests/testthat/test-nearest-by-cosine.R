# resolve.nearest_by_cosine(): the engine's exact cosine search against a brute
# force in R, the leave-one-out mode with duplicate rows, and argument checks.

cosine_all <- function(queries, reference) {
  unit <- function(x) x / sqrt(rowSums(x^2))
  unit(queries) %*% t(unit(reference))
}

test_that("the search returns the k most similar reference rows, most similar first", {
  skip_if_no_backend()
  set.seed(3)
  queries <- matrix(rnorm(40 * 7), nrow = 40)
  reference <- matrix(rnorm(300 * 7), nrow = 300)
  k <- 6L

  found <- resolve.nearest_by_cosine(queries, reference, k)
  expect_true(is.integer(found$indices))
  expect_equal(dim(found$indices), c(40L, k))
  expect_equal(dim(found$similarities), c(40L, k))

  all <- cosine_all(queries, reference)
  expected <- t(apply(all, 1, function(row) order(row, decreasing = TRUE)[seq_len(k)]))
  expect_equal(found$indices, expected)
  expect_equal(found$similarities, t(apply(all, 1, sort, decreasing = TRUE))[, seq_len(k)],
               tolerance = 1e-5)
})

test_that("excludeSelf leaves each row out of its own neighbours but keeps its duplicates", {
  skip_if_no_backend()
  set.seed(5)
  plots <- matrix(rbinom(60 * 15, 1, 0.4), nrow = 60)
  plots[2, ] <- plots[1, ]
  plots <- plots[rowSums(plots) > 0, , drop = FALSE]
  k <- 4L

  found <- resolve.nearest_by_cosine(plots, plots, k, excludeSelf = TRUE)
  expect_false(any(found$indices == seq_len(nrow(plots))))
  expect_equal(found$indices[1, 1], 2L)
  expect_equal(found$indices[2, 1], 1L)
  expect_equal(found$similarities[1, 1], 1, tolerance = 1e-6)

  all <- cosine_all(plots, plots)
  diag(all) <- -Inf
  # Presence rows tie often, so compare the similarities each returned row
  # carries and the k largest, not one order of tied rows.
  carried <- t(sapply(seq_len(nrow(plots)), function(i) all[i, found$indices[i, ]]))
  expect_equal(found$similarities, carried, tolerance = 1e-5)
  expect_equal(found$similarities, t(apply(all, 1, sort, decreasing = TRUE))[, seq_len(k)],
               tolerance = 1e-5)

  expect_equal(ncol(resolve.nearest_by_cosine(plots[1:3, ], plots[1:3, ], 10L,
                                              excludeSelf = TRUE)$indices), 2L)
  expect_error(resolve.nearest_by_cosine(plots[1:3, ], plots, 2L, excludeSelf = TRUE),
               "exclude_self")
})

test_that("the arguments are checked", {
  m <- matrix(1, 2, 2)
  expect_error(resolve.nearest_by_cosine(1:4, m, 1L), "queries must be a numeric matrix")
  expect_error(resolve.nearest_by_cosine(m, "a", 1L), "reference must be a numeric matrix")
  expect_error(resolve.nearest_by_cosine(m, m, 1.5), "whole number")
  expect_error(resolve.nearest_by_cosine(m, m, 1L, excludeSelf = NA), "TRUE or FALSE")
  skip_if_no_backend()
  expect_error(resolve.nearest_by_cosine(m, matrix(1, 2, 3), 1L), "columns")
  expect_error(resolve.nearest_by_cosine(m, m, 0L), "at least 1")
  expect_error(resolve.nearest_by_cosine(m, m, 1L, device = "gpu"), "cuda")
})
