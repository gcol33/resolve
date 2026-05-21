# Tests for removed standalone encoder facade.
#
# The unified resolve::SpeciesEncoder C++ class was split into
# RankPoolEncoder + EmbeddingEncoder, and the standalone resolve.encoder()
# wrapper has been removed (see r/R/resolve.R). The tests below pin the
# replacement contract: calling resolve.encoder() must raise a clear error
# that points users at the modern resolve.dataset.csv() pipeline, rather
# than silently no-op or fail with an unused-argument message.

test_that("resolve.encoder() raises a clear removal error", {
  expect_error(resolve.encoder(), "resolve.encoder\\(\\) has been removed")
  expect_error(resolve.encoder(), "resolve.dataset.csv\\(\\)")
})

test_that("resolve.encoder() error survives extra arguments", {
  # Pre-removal signature took hashDim/topK/etc.; the new stub accepts
  # ... so existing call sites surface the removal message rather than
  # an "unused argument" message.
  expect_error(
    resolve.encoder(hashDim = 32L, topK = 5L, selection = "top"),
    "resolve.encoder\\(\\) has been removed"
  )
})

test_that("resolve.dataset() raises a clear removal error", {
  # resolve.dataset() depended on resolve.encoder() and is removed in
  # lockstep.
  expect_error(resolve.dataset(), "resolve.dataset\\(\\) has been removed")
  expect_error(resolve.dataset(), "resolve.dataset.csv\\(\\)")
})

test_that("resolve.train() raises a clear removal error", {
  # resolve.train() consumed resolve.dataset() output and is removed
  # together with that pair.
  expect_error(resolve.train(), "resolve.train\\(\\) has been removed")
  expect_error(resolve.train(), "resolve.train.dataset\\(\\)")
})

test_that("resolve.predict() raises a clear removal error", {
  # resolve.predict() consumed resolve.dataset() output and is removed
  # together with that pair.
  expect_error(resolve.predict(), "resolve.predict\\(\\) has been removed")
  expect_error(resolve.predict(), "resolve.predict.dataset\\(\\)")
})
