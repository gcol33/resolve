# Chunk settings for the site build, switched on by build_site.R through RESOLVE_DOCS_CACHE.
#
# The engine's datasets, trainers and predictors are external pointers into the resolve_c
# library, which knitr's chunk cache cannot restore, so no chunk is cached on its own. The
# article is cached whole instead: its key is the code of every chunk (headers and bodies, in
# order) together with the engine binary, the `resolve` executable on PATH, the example data,
# the package version and this file. When the key matches the last knitted run, no chunk is
# evaluated and each chunk's rendered output is replayed from that run; prose edits therefore
# rebuild without running the engine. Any other change runs every chunk and records the
# outputs together with the figures the chunks drew, which the replay copies back into the
# article's figure folder. RESOLVE_DOCS_REFRESH=1 runs every chunk regardless.

knitr::opts_chunk$set(cache = FALSE)

local({
  input <- knitr::current_input(dir = TRUE)
  if (is.null(input) || !file.exists(input)) return(invisible())

  md5_files <- function(paths) {
    paths <- paths[nzchar(paths) & file.exists(paths)]
    if (!length(paths)) return("")
    paste(basename(paths), unname(tools::md5sum(paths)), collapse = ";")
  }

  chunk_code <- function(lines) {
    open <- grepl("^\\s*```\\{", lines)
    fence <- grepl("^\\s*```\\s*$", lines)
    inside <- FALSE
    keep <- logical(length(lines))
    for (i in seq_along(lines)) {
      if (!inside && open[i]) {
        inside <- TRUE
        keep[i] <- TRUE
      } else if (inside) {
        keep[i] <- TRUE
        if (fence[i]) inside <- FALSE
      }
    }
    lines[keep]
  }

  extdata <- list.files(system.file("extdata", package = "resolveR"), full.names = TRUE,
                        recursive = TRUE)
  backend <- tryCatch(resolveR:::.resolve_find_backend(), error = function(e) "")
  cache_file <- Sys.getenv("RESOLVE_DOCS_CACHE")
  key_parts <- c(
    chunk_code(readLines(input, encoding = "UTF-8", warn = FALSE)),
    as.character(utils::packageVersion("resolveR")),
    md5_files(backend),
    md5_files(unname(Sys.which("resolve"))),
    md5_files(extdata),
    md5_files(cache_file)
  )
  scratch <- tempfile()
  writeLines(enc2utf8(key_parts), scratch, useBytes = TRUE)
  key <- unname(tools::md5sum(scratch))
  unlink(scratch)

  store_dir <- file.path(dirname(cache_file), "_cache", "articles")
  name <- tools::file_path_sans_ext(basename(input))
  store <- file.path(store_dir, paste0(name, ".rds"))
  store_figures <- file.path(store_dir, paste0(name, "_figures"))
  fig_path <- knitr::opts_chunk$get("fig.path")
  fig_dir <- dirname(paste0(fig_path, "x"))
  if (!grepl("^(/|[A-Za-z]:)", fig_dir)) fig_dir <- file.path(dirname(input), fig_dir)
  previous <- if (file.exists(store)) tryCatch(readRDS(store), error = function(e) NULL)
  refresh <- identical(Sys.getenv("RESOLVE_DOCS_REFRESH"), "1")
  replay <- !refresh && is.list(previous) && identical(previous$key, key)

  say <- function(...) cat("  docs cache:", name, ..., "\n", file = stderr())
  render_chunk <- knitr::knit_hooks$get("chunk")
  render_document <- knitr::knit_hooks$get("document")

  if (replay) {
    say("replayed")
    if (length(previous$figures)) {
      dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
      file.copy(file.path(store_figures, previous$figures), fig_dir, overwrite = TRUE)
    }
    knitr::opts_hooks$set(eval = function(options) {
      options$eval <- FALSE
      options
    })
    knitr::knit_hooks$set(chunk = function(x, options) {
      recorded <- previous$chunks[[options$label]]
      if (is.null(recorded)) render_chunk(x, options) else recorded
    })
    return(invisible())
  }

  say(if (refresh) "refreshed" else "run")
  recorded <- new.env(parent = emptyenv())
  recorded$chunks <- list()
  knitr::knit_hooks$set(
    chunk = function(x, options) {
      out <- render_chunk(x, options)
      recorded$chunks[[options$label]] <- out
      out
    },
    document = function(x) {
      figures <- list.files(fig_dir)
      unlink(store_figures, recursive = TRUE)
      if (length(figures)) {
        dir.create(store_figures, recursive = TRUE, showWarnings = FALSE)
        file.copy(file.path(fig_dir, figures), store_figures, overwrite = TRUE)
      }
      dir.create(store_dir, recursive = TRUE, showWarnings = FALSE)
      saveRDS(list(key = key, chunks = recorded$chunks, figures = figures), store)
      render_document(x)
    }
  )
})
