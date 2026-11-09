# Builds the RESOLVE site, gillescolling.com/resolve/, from this package into ../docs.
#
#   Rscript build_site.R            run from r/
#
# The articles run their code: the R chunks against the engine in RESOLVE_C_HOME (or the backend
# resolve.install_backend() installed), the Python blocks against the resolve_core that PYTHON
# names. Both have to be the version DESCRIPTION carries.
#
# One build runs at a time: a second build would clean docs/ while the first writes it. The lock
# is the folder pkgdown/_cache/build_site.lock holding the builder's process id; a lock whose
# process has ended is taken over. Everything the build prints goes to the console and to
# pkgdown/_cache/build_site.log, and a failed build names the article it was rendering.

source("~/.R/build_pkgdown.R")

cache_dir <- "pkgdown/_cache"
lock_dir <- file.path(cache_dir, "build_site.lock")
log_path <- file.path(cache_dir, "build_site.log")

process_alive <- function(pid) {
  if (is.na(pid)) return(FALSE)
  if (.Platform$OS.type == "windows") {
    out <- suppressWarnings(system2("tasklist", c("/FI", shQuote(paste("PID eq", pid)), "/NH", "/FO", "CSV"),
                                    stdout = TRUE, stderr = FALSE))
    any(grepl(sprintf('"%d"', pid), out, fixed = TRUE))
  } else {
    suppressWarnings(system2("kill", c("-0", pid), stdout = FALSE, stderr = FALSE)) == 0L
  }
}

acquire_lock <- function() {
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  pid_file <- file.path(lock_dir, "pid")
  for (attempt in 1:2) {
    if (dir.create(lock_dir, showWarnings = FALSE)) {
      writeLines(c(as.character(Sys.getpid()), format(Sys.time(), "%Y-%m-%d %H:%M:%S")), pid_file)
      return(invisible())
    }
    held <- suppressWarnings(readLines(pid_file, warn = FALSE))
    pid <- suppressWarnings(as.integer(held[1]))
    if (process_alive(pid)) {
      stop("Another site build (process ", pid, ", started ", held[2], ") is writing docs/. ",
           "Wait for it to finish; two builds delete each other's output.", call. = FALSE)
    }
    unlink(lock_dir, recursive = TRUE)
  }
  stop("Cannot take the build lock ", lock_dir, call. = FALSE)
}

# Output is teed into the log by sink(); messages and warnings are written to that output
# instead of stderr, so each line reaches both once.
log_line <- function(text) cat(sub("\n$", "", text), "\n", sep = "")

# A child process's output passes through R so that it reaches the log too.
run_logged <- function(command, args) {
  out <- suppressWarnings(system2(command, args, stdout = TRUE, stderr = TRUE))
  if (length(out)) cat(out, sep = "\n")
  status <- attr(out, "status")
  if (is.null(status)) 0L else status
}

# The article pkgdown announced last, read back from the log.
article_in_flight <- function() {
  lines <- readLines(log_path, warn = FALSE, encoding = "UTF-8")
  reading <- regmatches(lines, regexpr("vignettes/[A-Za-z0-9_./-]+[.]Rmd", lines))
  if (length(reading)) reading[length(reading)] else NA_character_
}

build_site <- function() {
  if (!requireNamespace("resolveR", quietly = TRUE) ||
      as.character(packageVersion("resolveR")) != read.dcf("DESCRIPTION", "Version")[[1]]) {
    stop("Install this tree first: R CMD INSTALL r", call. = FALSE)
  }
  if (!resolveR::resolve.available()) stop(resolveR:::.resolve_backend_message(), call. = FALSE)

  python <- Sys.getenv("PYTHON", unset = "python")
  if (run_logged(python, "tools/python_twins.py") != 0L) {
    stop("tools/python_twins.py failed under ", python,
         ". Set PYTHON to the interpreter resolve_core is installed in.", call. = FALSE)
  }

  source("tools/build_readme.R")

  # The changelog is the engine's, at the repository root. pkgdown reads one release per top-level
  # heading naming the package and version, where the root file has `## v1.0.3 (2026-10-25)` under a
  # `# RESOLVE Changelog` title, so each heading moves up a level on the way. `## Unreleased`, the
  # changes since the last release, becomes pkgdown's `(development version)`.
  news <- readLines("../NEWS.md", encoding = "UTF-8")
  news <- news[!grepl("^# ", news)]
  news <- sub("^## v?([0-9][^ ]*)( .*)?$", "# resolveR \\1", news)
  news <- sub("^## Unreleased$", "# resolveR (development version)", news)
  news <- sub("^#(#{2,5}) ", "\\1 ", news)
  con <- file("NEWS.md", open = "wb")
  writeLines(enc2utf8(news), con, sep = "\n", useBytes = TRUE)
  close(con)

  # Marks the site build for the articles; pkgdown/docs_cache.R holds their chunk settings.
  Sys.setenv(RESOLVE_DOCS_CACHE = normalizePath("pkgdown/docs_cache.R"))
  pkgdown::clean_site(quiet = TRUE, force = TRUE)
  build_pkgdown_site()
  # The digest the site workflow checks the committed build against.
  if (run_logged("Rscript", c("tools/site_digest.R", "--write")) != 0L) {
    stop("tools/site_digest.R failed.", call. = FALSE)
  }
}

acquire_lock()
log_con <- file(log_path, open = "wt", encoding = "UTF-8")
sink(log_con, split = TRUE)
log_line(paste("build started", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "process", Sys.getpid()))
failure <- tryCatch(
  withCallingHandlers(
    {
      build_site()
      NULL
    },
    message = function(m) {
      log_line(conditionMessage(m))
      invokeRestart("muffleMessage")
    },
    warning = function(w) {
      log_line(paste("Warning:", conditionMessage(w)))
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) e
)
log_line(paste("build ended", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
sink()
close(log_con)
unlink(lock_dir, recursive = TRUE)
if (!is.null(failure)) {
  article <- article_in_flight()
  where <- if (is.na(article)) "" else paste0(" while rendering ", article)
  stop("Site build failed", where, ": ", conditionMessage(failure), "\nFull log: ",
       normalizePath(log_path), call. = FALSE)
}
