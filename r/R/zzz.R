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

# --- backend variant registry + GPU detection --------------------------------

# Pinned libtorch version the CUDA resolve_c binaries are built against. The
# release CI MUST build resolve_c against this same version: a resolve_c built
# against libtorch X only loads against libtorch X's DLLs (no cross-version C++
# ABI), so this constant is the single source both the downloader and CI share.
.RESOLVE_LIBTORCH_CUDA_VERSION <- "2.9.0"

# Host OS ("windows"/"macos"/"linux") and CPU arch ("x86_64"/"arm64").
.resolve_os_arch <- function() {
  os <- if (.Platform$OS.type == "windows") "windows"
        else if (Sys.info()[["sysname"]] == "Darwin") "macos" else "linux"
  arch <- switch(R.version$arch, x86_64 = "x86_64", aarch64 = "arm64", R.version$arch)
  list(os = os, arch = arch)
}

# Fetch plan for a backend variant: the small resolve_c asset on the package's
# GitHub release, plus (CUDA only) the matching official libtorch on PyTorch's
# CDN that resolve_c was built against. Adding a CUDA line later is one row here
# plus one CI matrix entry -- no other code changes.
.resolve_backend_registry <- function(variant, os = NULL, arch = NULL) {
  oa <- .resolve_os_arch()
  if (is.null(os)) os <- oa$os
  if (is.null(arch)) arch <- oa$arch
  ver <- .RESOLVE_LIBTORCH_CUDA_VERSION
  libtorch_url <- function(cu) {
    # Linux CUDA libtorch is the pre-cxx11 ABI build; Windows is win-shared. The
    # C ABI boundary is plain C types, so resolve_c's internal libstdc++ ABI does
    # not reach the R client either way.
    stem <- if (os == "windows") "libtorch-win-shared-with-deps"
            else "libtorch-shared-with-deps"
    sprintf("https://download.pytorch.org/libtorch/%s/%s-%s%%2B%s.zip",
            cu, stem, ver, cu)
  }
  reg <- list(
    cpu   = list(cuda = FALSE, libtorch_url = NULL),
    cu128 = list(cuda = TRUE,  libtorch_url = libtorch_url("cu128")),
    cu130 = list(cuda = TRUE,  libtorch_url = libtorch_url("cu130"))
  )
  entry <- reg[[variant]]
  if (is.null(entry)) {
    stop(sprintf("unknown backend variant '%s' (known: %s)",
                 variant, paste(names(reg), collapse = ", ")), call. = FALSE)
  }
  entry$variant <- variant
  entry$github_asset <- sprintf("resolve_c-%s-%s-%s.zip", os, arch, variant)
  entry
}

# TRUE if an NVIDIA GPU is visible (nvidia-smi lists at least one device).
.resolve_has_nvidia_gpu <- function() {
  smi <- Sys.which("nvidia-smi")
  if (!nzchar(smi)) return(FALSE)
  out <- tryCatch(
    suppressWarnings(system2(smi, "--query-gpu=name --format=csv,noheader",
                             stdout = TRUE, stderr = FALSE)),
    error = function(e) character())
  length(out) > 0 && any(nzchar(out))
}

# Max CUDA version the installed driver supports (from nvidia-smi's banner), as a
# numeric like 13.1, or NA if unavailable.
.resolve_driver_cuda_version <- function() {
  smi <- Sys.which("nvidia-smi")
  if (!nzchar(smi)) return(NA_real_)
  out <- tryCatch(suppressWarnings(system2(smi, stdout = TRUE, stderr = FALSE)),
                  error = function(e) character())
  # Linux/older banners say "CUDA Version: 12.6"; recent Windows drivers say
  # "CUDA UMD Version: 13.3" (the driver's max supported CUDA). Match both.
  hit <- grep("CUDA[^:]*Version:", out, value = TRUE)
  if (!length(hit)) return(NA_real_)
  m <- regmatches(hit[1], regexpr("CUDA[^:]*Version:\\s*[0-9]+\\.[0-9]+", hit[1]))
  if (!length(m)) return(NA_real_)
  as.numeric(sub(".*Version:\\s*", "", m))
}

# Resolve variant="cuda" to a concrete line from the driver's CUDA version.
# NULL if no GPU is present; errors if the driver is too old for any line.
.resolve_auto_cuda_variant <- function() {
  if (!.resolve_has_nvidia_gpu()) return(NULL)
  cv <- .resolve_driver_cuda_version()
  if (is.na(cv)) return("cu128")            # GPU present, version unknown -> broadest
  if (cv >= 13.0) return("cu130")
  if (cv >= 12.8) return("cu128")
  stop(sprintf(paste0("the NVIDIA driver supports only CUDA %.1f, but resolve's ",
                      "CUDA backends need >= 12.8; update the driver or use ",
                      "variant='cpu'"), cv), call. = FALSE)
}

# Normalize a user variant to a concrete one ("cuda" -> cu130/cu128 by driver).
.resolve_normalize_variant <- function(variant) {
  variant <- variant[1]
  if (variant %in% c("cpu", "cu128", "cu130")) return(variant)
  if (identical(variant, "cuda")) {
    v <- .resolve_auto_cuda_variant()
    if (is.null(v)) {
      stop("variant='cuda' but no NVIDIA GPU was detected; use variant='cpu', ",
           "or name a line explicitly (cu128 / cu130)", call. = FALSE)
    }
    return(v)
  }
  stop("variant must be one of 'cpu', 'cu128', 'cu130', or 'cuda'", call. = FALSE)
}

# TRUE if the installed backend is a CUDA build (torch_cuda ships beside
# resolve_c). Used to nudge CPU-backend users who actually have a GPU.
.resolve_backend_is_cuda <- function() {
  cuda_lib <- if (.Platform$OS.type == "windows") "torch_cuda.dll" else "libtorch_cuda.so"
  for (d in .resolve_backend_dirs()) {
    if (file.exists(file.path(d, cuda_lib))) return(TRUE)
  }
  FALSE
}

# Download a .zip from `url` and unzip it into `dir`. Raises the download timeout
# for the multi-GB libtorch archives (R's 60s default would abort them).
.resolve_download_unzip <- function(url, dir, quiet = FALSE) {
  old <- options(timeout = max(3600, getOption("timeout", 60)))
  on.exit(options(old), add = TRUE)
  tmp <- tempfile(fileext = ".zip")
  on.exit(unlink(tmp), add = TRUE)
  if (!quiet) message("Downloading ", url)
  utils::download.file(url, tmp, mode = "wb", quiet = quiet)
  utils::unzip(tmp, exdir = dir)
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
  have <- isTRUE(tryCatch(resolve_capi_is_available(), error = function(e) FALSE))
  gpu <- .resolve_has_nvidia_gpu()
  if (!have) {
    hint <- if (gpu) "resolve.install_backend(variant = \"cuda\")  # GPU build for your NVIDIA card"
            else     "resolve.install_backend()"
    packageStartupMessage(
      "resolve: the resolve_c backend is not installed, so training / dataset / ",
      "prediction verbs are unavailable.\n",
      "Install it with ", hint, ", or set RESOLVE_C_HOME to a directory ",
      "containing the resolve_c shared library."
    )
  } else if (gpu && !.resolve_backend_is_cuda()) {
    # Backend works, but it's the CPU build on a machine that has a GPU.
    packageStartupMessage(
      "resolve: an NVIDIA GPU was detected but the CPU backend is loaded. ",
      "For GPU-accelerated training install the GPU build:\n",
      "  resolve.install_backend(variant = \"cuda\", force = TRUE)"
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
