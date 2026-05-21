# Package initialization for resolve
# Uses Rcpp modules to expose C++ classes

#' @importFrom Rcpp evalCpp
#' @useDynLib resolve, .registration = TRUE
NULL

# Rcpp module reference
.resolve_module <- NULL

.onLoad <- function(libname, pkgname) {
  # Load the Rcpp module
  .resolve_module <<- Rcpp::Module("resolve_module", PACKAGE = "resolve")
}

# Internal: enumerate every class and free function registered in the
# resolve Rcpp module.
#
# Rcpp's Module S4 object does NOT define a `names()` method, so
# `names(.resolve_module)` falls through to the default that lists the
# bindings of the underlying `.xData` environment — i.e. Rcpp internals
# (`pointer`, `packageName`, `moduleName`) plus whatever has been
# lazily cached by previous `$` accesses. It does NOT enumerate the
# RCPP_MODULE(resolve_module) registration. The authoritative answer
# lives in Rcpp's unexported `Module__classes_info` /
# `Module__functions_names` helpers, which read directly from the
# module pointer.
#
# This helper returns the union of registered class names and free
# function names so callers can answer "is X registered?" by
# `X %in% .resolve_module_registered()`. It is also resilient to
# Rcpp internal API tweaks: if either helper disappears it falls back
# to the other.
.resolve_module_registered <- function() {
  mod <- .resolve_module
  if (is.null(mod)) return(character())
  ptr <- tryCatch(mod@pointer, error = function(e) NULL)
  if (is.null(ptr)) return(character())
  classes <- tryCatch({
    info <- Rcpp:::Module__classes_info(ptr)
    if (is.data.frame(info)) info$name else vapply(info, function(x) x$name, character(1))
  }, error = function(e) character())
  funcs <- tryCatch(
    Rcpp:::Module__functions_names(ptr),
    error = function(e) character()
  )
  unique(c(classes, funcs))
}

#' Get RESOLVE version
#'
#' @return Version string from C++ core
#' @export
resolve.version <- function() {
  .Call("_resolve_resolve_version", PACKAGE = "resolve")
}
