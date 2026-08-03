# Skip a test when the resolve_c backend is not installed, so the suite passes
# on CRAN and on any machine without the prebuilt engine. Engine-touching tests
# (dataset / trainer / predict / metrics) call this; pure-R validation and
# stub-error tests do not need it and run everywhere.
skip_if_no_backend <- function() {
  testthat::skip_if_not(resolve.available(), "resolve_c backend not installed")
}
