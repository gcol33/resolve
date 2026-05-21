# Package initialization for resolve
# Uses Rcpp modules to expose C++ classes

#' @importFrom Rcpp evalCpp
#' @useDynLib resolve, .registration = TRUE
NULL

# Rcpp module reference
.resolve_module <- NULL

.onLoad <- function(libname, pkgname) {
  # Load the Rcpp module reference. We do NOT pass `mustStart = TRUE`
  # here: the underlying boot symbol pulls in libtorch (300+ MB of
  # shared libs) and we want the cost paid only when the user
  # actually exercises a C++ class. Lazy init is triggered on the
  # first `$` access via Rcpp's `.getModulePointer`.
  .resolve_module <<- Rcpp::Module("resolve_module", PACKAGE = "resolve")
}

# Internal: enumerate every class and free function registered in the
# resolve Rcpp module.
#
# Rcpp's `Module` S4 object does NOT define a `names()` method, so
# `names(.resolve_module)` falls through to the default that lists the
# bindings of the underlying `.xData` environment — i.e. Rcpp internals
# (`pointer`, `packageName`, `moduleName`) plus whatever has been lazily
# cached by previous `$` accesses. It does NOT enumerate the
# `RCPP_MODULE(resolve_module)` registration.
#
# The canonical enumeration lives in the `storage` environment that
# `Rcpp::Module(..., mustStart = TRUE)` writes into `mod@.xData`, with
# one binding per registered class (demangled name) and one per free
# function. The `$.Module` operator reads from exactly this env, so
# using it here makes "is X registered?" identical to "does `mod$X`
# resolve?" without instantiating anything. We force `mustStart = TRUE`
# here so the helper works regardless of whether `.onLoad` did it.
.resolve_module_registered <- function() {
  mod <- .resolve_module
  if (is.null(mod)) return(character())
  # First-call after .onLoad: `storage` does not yet exist in
  # mod@.xData because `.onLoad` deferred lazy init. Run Module()
  # with mustStart = TRUE to populate it. This is idempotent on
  # subsequent calls (Module() returns the same S4 object after
  # repopulating storage).
  ready <- tryCatch(
    Rcpp::Module(mod, mustStart = TRUE),
    error = function(e) NULL
  )
  if (is.null(ready)) return(character())
  storage <- tryCatch(get("storage", envir = ready@.xData, inherits = FALSE),
                      error = function(e) NULL)
  if (is.null(storage)) return(character())
  ls(storage, all.names = FALSE)
}

#' Get RESOLVE version
#'
#' @return Version string from C++ core
#' @export
resolve.version <- function() {
  .Call("_resolve_resolve_version", PACKAGE = "resolve")
}
