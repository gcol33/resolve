# Package initialization for resolve.
#
# The package is a thin client over the `resolve_c` shared library (a flat C ABI
# over the RESOLVE engine, which bundles libtorch). resolve_c is NOT linked at
# build time; it is loaded at RUNTIME via dlopen/LoadLibrary
# (src/resolve_capi_dynload.*), the way the mlverse/torch R package loads
# libtorch. This lets the package install and R CMD check with no backend
# present. The engine verbs gate on resolve.available(); when the backend is
# absent they raise a clear "install it" error instead of crashing.
#
# Backend discovery order (.resolve_find_backend): $RESOLVE_C_HOME, then the
# user data dir where resolve.install_backend() puts it, then a packaged copy.
# When found, its directory is prepended to the loader path so resolve_c's
# sibling libtorch libraries resolve, and the library is bound.

#' @importFrom Rcpp evalCpp
#' @importFrom methods new
#' @useDynLib resolve, .registration = TRUE
NULL

# Rcpp module reference
.resolve_module <- NULL

# Platform file name of the resolve_c shared library.
.resolve_backend_libname <- function() {
  if (.Platform$OS.type == "windows") {
    "resolve_c.dll"
  } else if (Sys.info()[["sysname"]] == "Darwin") {
    "libresolve_c.dylib"
  } else {
    "libresolve_c.so"
  }
}

# Directories to search for the backend, in priority order.
.resolve_backend_dirs <- function() {
  dirs <- character(0)
  home <- Sys.getenv("RESOLVE_C_HOME", "")
  if (nzchar(home)) dirs <- c(dirs, home)
  data_dir <- tryCatch(tools::R_user_dir("resolve", "data"), error = function(e) "")
  if (nzchar(data_dir)) dirs <- c(dirs, file.path(data_dir, "resolve_c"))
  pkg_dir <- system.file("resolve_c", package = "resolve")
  if (nzchar(pkg_dir)) dirs <- c(dirs, pkg_dir)
  dirs
}

# Full path to the backend library if present, else "".
.resolve_find_backend <- function() {
  libname <- .resolve_backend_libname()
  for (d in .resolve_backend_dirs()) {
    p <- file.path(d, libname)
    if (file.exists(p)) return(p)
  }
  ""
}

# Prepend `dir` to the OS loader path so resolve_c's sibling runtime libraries
# (the libtorch DLLs / shared objects that live next to it) resolve at load.
.resolve_prepend_loader_path <- function(dir) {
  if (!nzchar(dir) || !dir.exists(dir)) return(invisible())
  var <- if (.Platform$OS.type == "windows") "PATH"
         else if (Sys.info()[["sysname"]] == "Darwin") "DYLD_LIBRARY_PATH"
         else "LD_LIBRARY_PATH"
  cur <- Sys.getenv(var, "")
  parts <- if (nzchar(cur)) c(dir, cur) else dir
  args <- list(paste(parts, collapse = .Platform$path.sep))
  names(args) <- var
  do.call(Sys.setenv, args)
  invisible()
}

# Locate and bind the resolve_c backend. Returns TRUE on success. Idempotent:
# resolve_capi_load_lib() is a no-op once the library is already bound.
.resolve_load_backend <- function() {
  if (isTRUE(tryCatch(resolve_capi_is_available(), error = function(e) FALSE))) {
    return(TRUE)
  }
  path <- .resolve_find_backend()
  if (!nzchar(path)) return(FALSE)
  .resolve_prepend_loader_path(dirname(path))
  isTRUE(tryCatch(resolve_capi_load_lib(path), error = function(e) FALSE))
}

# Token whose on-exit finalizer marks engine work complete (see
# .resolve_harden_process). A namespace-level environment so it lives for the
# whole session and its finalizer runs during R's shutdown.
.resolve_exit_token <- new.env(parent = emptyenv())

# Issue #18 / #19: harden the process against a native fault becoming a hang or
# a launcher teardown crash. Only called once the backend is bound, since the
# handlers are engine (resolve_c) calls.
#
#   * install_crash_handler() (no-op off Windows, no throughput cost): converts
#     an otherwise-unhandled native fault into an immediate TerminateProcess
#     instead of a Windows JIT-debugger hang (issue #19) or a teardown access
#     violation that crashes the Rscript.exe launcher (issue #18).
#   * An on-exit finalizer marks work complete so a teardown fault after a
#     finished session is treated as benign (exit code 0).
#
# libtorch thread pools are left at libtorch's own default (all cores) so
# training and prediction are multi-threaded -- that is where the CPU throughput
# comes from. Issue #18 previously pinned Windows to a single thread as a
# teardown-crash mitigation (no worker threads to join at process exit), but the
# crash handler installed above already turns a teardown fault into an immediate
# TerminateProcess, which never joins the pools, so full threading is safe by
# default. Set RESOLVE_R_TORCH_THREADS=N (a positive integer) to pin both the
# intra- and inter-op pools to N threads -- to cap CPU use on a shared machine,
# or as a workaround (N=1 restores the old single-threaded behaviour) if a
# specific Windows environment still hits the #18 teardown crash.
.resolve_harden_process <- function() {
  try(.Call("_resolve_resolve_install_crash_handler", 0L, PACKAGE = "resolve"),
      silent = TRUE)
  n_threads <- suppressWarnings(as.integer(Sys.getenv("RESOLVE_R_TORCH_THREADS", "")))
  if (!is.na(n_threads) && n_threads >= 1L) {
    try(.Call("_resolve_resolve_set_thread_pools", n_threads, n_threads,
              PACKAGE = "resolve"),
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
  # Bind resolve_c if it is installed; the package loads fine either way.
  loaded <- .resolve_load_backend()
  # Lazy module init: the boot symbol only registers class/method pointers (no
  # engine call), so it is safe with or without the backend; the actual engine
  # work happens when a method is invoked, gated by resolve.available().
  .resolve_module <<- Rcpp::Module("resolve_module", PACKAGE = "resolve")
  # Hardening is an engine call, so only when the backend is bound.
  if (loaded) .resolve_harden_process()
}

.onAttach <- function(libname, pkgname) {
  if (!isTRUE(tryCatch(resolve_capi_is_available(), error = function(e) FALSE))) {
    packageStartupMessage(
      "resolve: the resolve_c backend is not installed, so training / dataset / ",
      "prediction verbs are unavailable.\n",
      "Install it with resolve.install_backend(), or set RESOLVE_C_HOME to a ",
      "directory containing the resolve_c shared library."
    )
  }
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
#' @return Version string from the C++ core.
#' @export
resolve.version <- function() {
  .resolve_require_backend()
  .Call("_resolve_resolve_version", PACKAGE = "resolve")
}
