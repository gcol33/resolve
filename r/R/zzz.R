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

# Token whose on-exit finalizer marks engine work complete (see
# .resolve_harden_process). A namespace-level environment so it lives for the
# whole session and its finalizer runs during R's shutdown.
.resolve_exit_token <- new.env(parent = emptyenv())

# Issue #18 / #19: harden the process against a native fault becoming a hang or
# a launcher teardown crash.
#
#   * install_crash_handler() (always; no-op off Windows, no throughput cost):
#     converts an otherwise-unhandled native fault into an immediate
#     TerminateProcess instead of a Windows JIT-debugger handshake that hangs a
#     headless run forever (issue #19) or a teardown access violation that
#     crashes the Rscript.exe launcher (issue #18).
#   * On Windows, pin libtorch's host thread pools to 1 so there are no worker
#     threads to join during process exit -- the join is the suspected source of
#     the Rscript.exe teardown crash (the at::set_num_threads(1) mitigation #18
#     points at). This is the only part with a throughput trade-off, rare for
#     the thin-client metrics / small-CPU workloads that run through the R
#     bindings; set RESOLVE_R_NO_THREAD_PIN to keep libtorch's default threading.
#   * An on-exit finalizer marks work complete so a teardown fault after a
#     finished session is treated as a benign artifact (exit with code 0).
.resolve_harden_process <- function() {
  try(.Call("_resolve_resolve_install_crash_handler", 0L, PACKAGE = "resolve"),
      silent = TRUE)
  if (.Platform$OS.type == "windows" &&
      !nzchar(Sys.getenv("RESOLVE_R_NO_THREAD_PIN"))) {
    try(.Call("_resolve_resolve_set_thread_pools", 1L, 1L, PACKAGE = "resolve"),
        silent = TRUE)
  }
  reg.finalizer(
    .resolve_exit_token,
    function(e) {
      tryCatch(.Call("_resolve_resolve_signal_work_complete", PACKAGE = "resolve"),
               error = function(...) NULL)
    },
    onexit = TRUE
  )
  invisible()
}

.onLoad <- function(libname, pkgname) {
  .resolve_setup_libpath()
  # Lazy module init: the boot symbol pulls in resolve_c + libtorch, so defer it
  # to the first `$` access rather than forcing mustStart here.
  .resolve_module <<- Rcpp::Module("resolve_module", PACKAGE = "resolve")
  .resolve_harden_process()
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
