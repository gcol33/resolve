# libtorch calls the BLAS of the MKL it links statically. R puts its own BLAS
# in the process's global symbol scope first, so the backend has to be loaded
# such that libtorch's BLAS calls still bind inside libtorch. A preloaded
# library that defines the single-precision GEMM entries libtorch calls, and
# exits when one is reached, stands in for a conflicting system BLAS (OpenBLAS
# 0.3.32 exports cblas_sgemm_batch and crashes in it): a fit that reaches it
# ends with that exit instead of finishing.

test_that("libtorch's BLAS calls bind to libtorch, not to a BLAS already loaded", {
  skip_if_no_backend()
  skip_on_cran()
  skip_on_os(c("windows", "mac", "solaris"))
  lib <- dirname(system.file(package = "resolveR"))
  skip_if_not(file.exists(file.path(lib, "resolveR", "Meta", "package.rds")),
              "resolveR is not installed (load_all), so a child R process cannot load it")

  dir <- tempfile("blas_shim_")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  writeLines(c(
    "#include <stdio.h>",
    "#include <stdlib.h>",
    "static void reached(const char* name) {",
    "  fprintf(stderr, \"interposed BLAS reached: %s\\n\", name);",
    "  fflush(stderr);",
    "  _Exit(86);",
    "}",
    "void sgemm_(void) { reached(\"sgemm_\"); }",
    "void cblas_sgemm(void) { reached(\"cblas_sgemm\"); }",
    "void cblas_sgemm_batch(void) { reached(\"cblas_sgemm_batch\"); }"
  ), file.path(dir, "shim.c"))
  r_bin <- file.path(R.home("bin"), "R")
  old_wd <- setwd(dir)
  built <- suppressWarnings(system2(r_bin, c("CMD", "SHLIB", "shim.c"), stdout = TRUE, stderr = TRUE))
  setwd(old_wd)
  shim <- file.path(dir, paste0("shim", .Platform$dynlib.ext))
  skip_if_not(file.exists(shim), paste(c("cannot compile the BLAS shim:", built), collapse = "\n"))

  script <- file.path(dir, "fit.R")
  writeLines(c(
    "library(resolveR)",
    "plots <- system.file('extdata', 'plots.csv', package = 'resolveR')",
    "species <- system.file('extdata', 'species.csv', package = 'resolveR')",
    "roles <- list(plot_id = 'plot_id', species_id = 'species', abundance = 'cover',",
    "              genus = 'genus', family = 'family', latitude = 'lat', longitude = 'lon')",
    "targets <- list(elevation = list(column = 'elevation', task = 'regression'))",
    "dataset <- resolve.dataset.csv(plots, species, roles = roles, targets = targets)",
    "fit <- resolve.train.dataset(dataset, maxEpochs = 1L, verbose = FALSE,",
    "  modelConfig = list(encoder_architecture = 'ft_transformer'))",
    "cat('fit finished\\n')"
  ), script)
  libs <- paste(unique(c(lib, .libPaths())), collapse = .Platform$path.sep)
  out <- suppressWarnings(system2(
    file.path(R.home("bin"), "Rscript"), c("--vanilla", shQuote(script)),
    stdout = TRUE, stderr = TRUE,
    env = c(paste0("R_LIBS=", shQuote(libs)), paste0("LD_PRELOAD=", shQuote(shim)))))
  expect_false(any(grepl("interposed BLAS reached", out, fixed = TRUE)))
  expect(is.null(attr(out, "status")) && any(out == "fit finished"),
         paste(c("the child R process did not finish its fit:", out), collapse = "\n"))
})
