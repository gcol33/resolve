#!/usr/bin/env Rscript
# One digest over every source the site is rendered from. build_site.R writes it into docs/ after
# a build, and the site workflow recomputes it from the checkout and fails the push whose docs/
# were built from other sources. The site is built locally, by a tool that post-processes the
# figures, and this is what holds the committed build to the sources it claims to render.
#
#   Rscript tools/site_digest.R            print the digest of the sources
#   Rscript tools/site_digest.R --write    write it to docs/site-digest.txt
#   Rscript tools/site_digest.R --check    compare docs/site-digest.txt with the sources

# The sources are the files git tracks under these paths, so a file present in one working tree
# only (a backup, an editor's scratch file) changes neither the local digest nor the workflow's.
site_sources <- function(root = ".") {
  roots <- c("R", "man", "vignettes", "pkgdown", "inst/CITATION", "../NEWS.md",
             "README.md", "_pkgdown.yml", "DESCRIPTION")
  files <- system2("git", c("-C", shQuote(root), "-c", "core.quotepath=off", "ls-files", "--", roots),
                   stdout = TRUE)
  if (!is.null(attr(files, "status"))) stop("git ls-files failed in ", root)
  files <- files[file.exists(file.path(root, files))]
  files <- files[!grepl("_cache/", files, fixed = TRUE)]
  sort(files, method = "radix")
}

site_digest <- function(root = ".") {
  files <- site_sources(root)
  sums <- unname(tools::md5sum(file.path(root, files)))
  listing <- tempfile()
  on.exit(unlink(listing), add = TRUE)
  con <- file(listing, open = "wb")
  writeBin(charToRaw(paste(files, sums, sep = " ", collapse = "\n")), con)
  close(con)
  unname(tools::md5sum(listing))
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  digest <- site_digest()
  stamp <- "../docs/site-digest.txt"
  if ("--write" %in% args) {
    writeLines(digest, stamp)
    cat("wrote", stamp, digest, "\n")
  } else if ("--check" %in% args) {
    built <- if (file.exists(stamp)) readLines(stamp, n = 1L, warn = FALSE) else ""
    if (!identical(built, digest)) {
      cat("docs/ was built from other sources: the committed digest is", built, "and the",
          "sources digest to", digest, ".\nRun `Rscript build_site.R` in r/ and commit docs/.\n")
      quit(save = "no", status = 1L)
    }
    cat("docs/ is built from the current sources,", digest, "\n")
  } else {
    cat(digest, "\n")
  }
}
