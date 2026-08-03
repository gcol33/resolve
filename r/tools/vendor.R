#!/usr/bin/env Rscript
# vendor.R - Refresh the vendored C facade headers.
#
# Usage: Rscript tools/vendor.R
#
# Issue #17: the R package is a thin client over the prebuilt `resolve_c` shared
# library and no longer compiles the C++ engine. The engine headers it needs are
# the pure-C facade resolve/resolve_capi.h and its single-source symbol list
# resolve/resolve_capi_symbols.inc (the X-macro list that drives the runtime
# loader in src/resolve_capi_dynload.*). Both are vendored (and tracked) under
# src/resolve/. This copies the canonical copies from the monorepo core so the
# two never drift; the r-cmd-check `vendor-drift` job diffs them in CI.

VENDORED_FILES <- c("resolve_capi.h", "resolve_capi_symbols.inc")

vendor <- function(pkg_root = here::here()) {
  src_dir  <- file.path(pkg_root, "..", "src", "core", "include", "resolve")
  dest_dir <- file.path(pkg_root, "src", "resolve")
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)

  for (f in VENDORED_FILES) {
    src <- file.path(src_dir, f)
    if (!file.exists(src)) {
      stop("Canonical facade file not found at: ", src,
           "\nRun from the r/ directory of the RESOLVE monorepo.")
    }
    file.copy(src, file.path(dest_dir, f), overwrite = TRUE)
    message("Vendored ", f, " to src/resolve/")
  }
}

if (!interactive()) vendor()
