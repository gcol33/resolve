# Builds the resolveR pkgdown site.
#
#   Rscript r/build_site.R          into the _pkgdown.yml destination (r/docs/)
#   Rscript r/build_site.R <dir>    into <dir>; build_docs.py --r passes site/r
#
# Sourced from r/, it builds into r/docs/. Reference examples are not run: they
# call the engine, and a docs build has no resolve_c backend.

script <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
pkg <- if (length(script)) dirname(normalizePath(script)) else normalizePath(".")
args <- commandArgs(trailingOnly = TRUE)
override <- if (length(args)) list(destination = normalizePath(args[[1]], mustWork = FALSE)) else list()

pkgdown::build_site(pkg, examples = FALSE, override = override, preview = FALSE)

# pkgdown renders every root .md except README, NEWS, LICENSE and
# cran-comments as a page. One that R CMD build leaves out (.Rbuildignore) is
# a working note, so the build stops if such a file became a page.
handled <- c("README.md", "NEWS.md", "LICENSE.md", "LICENCE.md", "cran-comments.md")
ignore_patterns <- readLines(file.path(pkg, ".Rbuildignore"), warn = FALSE)
ignore_patterns <- ignore_patterns[nzchar(ignore_patterns) & !startsWith(ignore_patterns, "#")]
is_ignored <- function(name) any(vapply(ignore_patterns, grepl, logical(1), x = name, perl = TRUE))
notes <- Filter(is_ignored, setdiff(list.files(pkg, pattern = "\\.md$"), handled))
destination <- pkgdown::as_pkgdown(pkg, override = override)$dst_path
leaked <- file.path(destination, sub("\\.md$", ".html", notes))
leaked <- leaked[file.exists(leaked)]
if (length(leaked)) {
  stop("working notes were rendered as site pages: ", paste(leaked, collapse = ", "),
       call. = FALSE)
}
