# Package initialization for resolve.
#
# Issue #17: the package DLL (resolve.dll / resolve.so) links the prebuilt
# `resolve_c` shared library, which links the libtorch runtime libraries. Those
# must be on the loader path (PATH on Windows, LD_LIBRARY_PATH / DYLD_LIBRARY_PATH
# on Unix) when the package DLL is loaded. The expected mechanism is to put the
# resolve_c directory (which also holds the libtorch DLLs) on that path before
# starting R -- set the RESOLVE_C_HOME environment variable to it; the CI and the
# install/check tooling add it to the loader path. As a convenience .onLoad also
# best-effort prepends RESOLVE_C_HOME, which covers re-loads and interactive use
# when the variable is set but the path was not exported.

#' @importFrom Rcpp evalCpp
#' @importFrom methods new
#' @useDynLib resolve, .registration = TRUE
NULL

# Rcpp module reference
.resolve_module <- NULL

# Best-effort: prepend a dev DLL directory (RESOLVE_C_HOME) to the loader path.
# Only useful for source-tree workflows where the runtime DLLs are not staged
# in libs/<arch>; for an installed package the altered-search-path covers it.
.resolve_setup_libpath <- function() {
  home <- Sys.getenv("RESOLVE_C_HOME", "")
  if (!nzchar(home) || !dir.exists(home)) return(invisible())
  var <- if (.Platform$OS.type == "windows") "PATH"
         else if (Sys.info()[["sysname"]] == "Darwin") "DYLD_LIBRARY_PATH"
         else "LD_LIBRARY_PATH"
  cur <- Sys.getenv(var, "")
  newval <- paste(c(home, if (nzchar(cur)) cur), collapse = .Platform$path.sep)
  args <- list(newval)
  names(args) <- var
  do.call(Sys.setenv, args)
  invisible()
}

.onLoad <- function(libname, pkgname) {
  .resolve_setup_libpath()
  # Lazy module init: the boot symbol pulls in resolve_c + libtorch, so defer it
  # to the first `$` access rather than forcing mustStart here.
  .resolve_module <<- Rcpp::Module("resolve_module", PACKAGE = "resolve")
}

# Internal: enumerate every class and free function registered in the resolve
# Rcpp module. The canonical enumeration lives in the `storage` environment that
# `Rcpp::Module(..., mustStart = TRUE)` writes into `mod@.xData`; the
# `$.Module` operator reads from exactly this env, so using it here makes
# "is X registered?" identical to "does `mod$X` resolve?".
.resolve_module_registered <- function() {
  mod <- .resolve_module
  if (is.null(mod)) return(character())
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
