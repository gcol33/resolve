#!/usr/bin/env Rscript
# build_cran.R — Build CRAN-ready source tarball
#
# Usage: Rscript tools/build_cran.R
#
# Steps:
# 1. Vendor C++ core files
# 2. Run devtools::document()
# 3. Run R CMD check
# 4. Build source tarball

build_cran <- function() {
  pkg_root <- here::here()

  message("=== Step 1: Vendor C++ core ===")
  source(file.path(pkg_root, "tools", "vendor.R"))
  vendor(pkg_root)

  message("\n=== Step 2: Generate docs ===")
  devtools::document(pkg_root)

  message("\n=== Step 3: R CMD check ===")
  rcmdcheck::rcmdcheck(pkg_root, args = c("--no-manual", "--as-cran"))

  message("\n=== Step 4: Build tarball ===")
  devtools::build(pkg_root)

  message("\nDone!")
}

if (!interactive()) build_cran()
