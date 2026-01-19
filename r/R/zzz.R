# Package initialization

.resolve <- NULL

.onLoad <- function(libname, pkgname) {
  .resolve <<- reticulate::import("resolve", delay_load = TRUE)
}

#' Install RESOLVE Python package
#'
#' @param method Installation method (auto, virtualenv, conda)
#' @param conda Path to conda executable
#' @param envname Name of virtual environment
#' @param pip Use pip for installation
#' @param ... Additional arguments passed to py_install
#'
#' @export
install_resolve <- function(method = "auto",
                            conda = "auto",
                            envname = NULL,
                            pip = TRUE,
                            ...) {
  reticulate::py_install(
    "resolve",
    method = method,
    conda = conda,
    envname = envname,
    pip = pip,
    ...
  )
}
