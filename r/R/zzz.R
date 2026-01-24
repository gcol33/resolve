# Package initialization for resolve
# Uses Rcpp modules to expose C++ classes

#' @import Rcpp
#' @useDynLib resolve, .registration = TRUE
NULL

# Rcpp module reference
.resolve_module <- NULL

.onLoad <- function(libname, pkgname) {
  # Load the Rcpp module
  .resolve_module <<- Rcpp::Module("resolve_module", PACKAGE = "resolve")
}

#' Get RESOLVE version
#'
#' @return Version string from C++ core
#' @export
resolve_version <- function() {
  .Call("_resolve_resolve_version", PACKAGE = "resolve")
}
