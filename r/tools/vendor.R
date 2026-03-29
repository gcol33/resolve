#!/usr/bin/env Rscript
# vendor.R — Bundle C++ core into R package for CRAN release
#
# Usage: Rscript tools/vendor.R
#
# Copies C++ headers and source files from the monorepo root into
# r/src/ so the R package can compile standalone (required for CRAN).
# The vendored files are .gitignored — this is only for release builds.

vendor <- function(pkg_root = here::here()) {
  core_include <- file.path(pkg_root, "..", "src", "core", "include", "resolve")
  core_src <- file.path(pkg_root, "..", "src", "core", "cpp_src")
  dest_include <- file.path(pkg_root, "src", "resolve")
  dest_src <- file.path(pkg_root, "src")

  if (!dir.exists(core_include)) {
    stop("C++ core headers not found at: ", core_include,
         "\nRun from the r/ directory of the RESOLVE monorepo.")
  }

  # Clean previous vendor
  if (dir.exists(dest_include)) unlink(dest_include, recursive = TRUE)

  # Copy headers
  dir.create(dest_include, recursive = TRUE, showWarnings = FALSE)
  headers <- list.files(core_include, pattern = "\\.hpp$", full.names = TRUE)
  file.copy(headers, dest_include)
  message("Vendored ", length(headers), " headers to src/resolve/")

  # Copy source files
  sources <- list.files(core_src, pattern = "\\.cpp$", full.names = TRUE)
  file.copy(sources, dest_src)
  message("Vendored ", length(sources), " source files to src/")

  message("Done. Package is now self-contained for CRAN.")
}

if (!interactive()) vendor()
