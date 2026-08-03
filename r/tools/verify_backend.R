# Smoke-verify a packaged resolve_c backend zip.
#
# With RESOLVE_C_HOME pointing at an unzipped backend bundle (and the build-tree
# libtorch moved aside so a stale absolute rpath cannot satisfy the load), load
# the installed resolve package and confirm the backend binds and answers. This
# proves the release zip is self-contained -- every libtorch runtime dependency
# travels next to resolve_c -- which is exactly the install.packages("resolve")
# + resolve.install_backend() path a user takes. Run: Rscript r/tools/verify_backend.R
home <- Sys.getenv("RESOLVE_C_HOME", "")
if (!nzchar(home)) stop("RESOLVE_C_HOME is not set")

libname <- if (.Platform$OS.type == "windows") "resolve_c.dll"
           else if (Sys.info()[["sysname"]] == "Darwin") "libresolve_c.dylib"
           else "libresolve_c.so"
lib <- file.path(home, libname)
if (!file.exists(lib)) {
  stop("bundle at ", home, " does not contain ", libname,
       " at its top level (unzip must place the library flat).")
}

suppressPackageStartupMessages(library(resolve))

if (!resolve.available()) {
  err <- tryCatch(resolve:::resolve_capi_load_error(),
                  error = function(e) "<no load-error accessor>")
  stop("resolve_c backend failed to load from ", home, ": ", err)
}

cat("OK: resolve_c backend loaded from", home, "\n")
cat("engine version:", resolve.version(), "\n")
