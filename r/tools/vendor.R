#!/usr/bin/env Rscript
# vendor.R - Refresh the vendored C facade header.
#
# Usage: Rscript tools/vendor.R
#
# Issue #17: the R package is a thin client over the prebuilt `resolve_c` shared
# library and no longer compiles the C++ engine. The only engine header it needs
# is the pure-C facade resolve/resolve_capi.h, which is vendored (and tracked)
# under src/resolve/. This copies the canonical copy from the monorepo core so
# the two never drift.

vendor <- function(pkg_root = here::here()) {
  src_header <- file.path(pkg_root, "..", "src", "core", "include",
                          "resolve", "resolve_capi.h")
  dest_dir <- file.path(pkg_root, "src", "resolve")
  dest_header <- file.path(dest_dir, "resolve_capi.h")

  if (!file.exists(src_header)) {
    stop("C facade header not found at: ", src_header,
         "\nRun from the r/ directory of the RESOLVE monorepo.")
  }

  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  file.copy(src_header, dest_header, overwrite = TRUE)
  message("Vendored resolve_capi.h to src/resolve/")
}

if (!interactive()) vendor()
