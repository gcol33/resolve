test_that("a missing backend is reported as not installed", {
  local_mocked_bindings(.resolve_find_backend = function() "")
  msg <- .resolve_backend_message()
  expect_match(msg, "is not installed", fixed = TRUE)
  expect_match(msg, "resolve.install_backend()", fixed = TRUE)
})

test_that("a backend lacking C ABI functions is reported with its path and the missing names", {
  local_mocked_bindings(
    .resolve_find_backend = function() "/opt/resolve_c/libresolve_c.so",
    resolve_capi_load_error = function() {
      "resolve_capi_load: '/opt/resolve_c/libresolve_c.so' is missing symbols: resolve_fit_gaps"
    }
  )
  msg <- .resolve_backend_message()
  expect_no_match(msg, "not installed", fixed = TRUE)
  expect_match(msg, "/opt/resolve_c/libresolve_c.so could not be loaded", fixed = TRUE)
  expect_match(msg, "resolve_fit_gaps", fixed = TRUE)
  expect_match(msg, paste0("built before resolveR ", packageVersion("resolveR")), fixed = TRUE)
  expect_match(msg, "resolve.install_backend(force = TRUE)", fixed = TRUE)
})

test_that("a backend that fails to open is reported with the loader's reason only", {
  local_mocked_bindings(
    .resolve_find_backend = function() "C:/resolve_c/resolve_c.dll",
    resolve_capi_load_error = function() "resolve_capi_load: could not open 'C:/resolve_c/resolve_c.dll' (WinError 126)"
  )
  msg <- .resolve_backend_message()
  expect_match(msg, "WinError 126", fixed = TRUE)
  expect_no_match(msg, "built before", fixed = TRUE)
  expect_match(msg, "resolve.install_backend(force = TRUE)", fixed = TRUE)
})
