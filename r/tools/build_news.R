# Writes the package's NEWS.md from the engine changelog at the repository root. pkgdown and R both
# read one release per top-level heading naming the package and version, where the root file has
# `## v1.0.3 (2026-10-25)` under a `# RESOLVE Changelog` title, so each heading moves up a level on
# the way. `## Unreleased`, the changes since the last release, becomes `(development version)`.
#
#   Rscript tools/build_news.R      run from r/
#
# The site build and the source-tarball build both call build_news(), so the tarball ships the same
# changelog the site shows.

build_news <- function(pkg_root = ".") {
  news <- readLines(file.path(pkg_root, "..", "NEWS.md"), encoding = "UTF-8")
  news <- news[!grepl("^# ", news)]
  news <- sub("^## v?([0-9][^ ]*)( .*)?$", "# resolveR \\1", news)
  news <- sub("^## Unreleased$", "# resolveR (development version)", news)
  news <- sub("^#(#{2,5}) ", "\\1 ", news)
  con <- file(file.path(pkg_root, "NEWS.md"), open = "wb")
  on.exit(close(con))
  writeLines(enc2utf8(news), con, sep = "\n", useBytes = TRUE)
  invisible(file.path(pkg_root, "NEWS.md"))
}

if (sys.nframe() == 0L) build_news()
