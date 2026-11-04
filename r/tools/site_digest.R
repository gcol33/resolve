#!/usr/bin/env Rscript
# One digest over every source the site is rendered from. build_site.R writes it into docs/ after
# a build, and the site workflow recomputes it from the checkout and fails the push whose docs/
# were built from other sources. The site is built locally, by a tool that post-processes the
# figures, and this is what holds the committed build to the sources it claims to render.
#
#   Rscript tools/site_digest.R            print the digest of the sources
#   Rscript tools/site_digest.R --write    write it to docs/site-digest.txt
#   Rscript tools/site_digest.R --check    compare docs/site-digest.txt with the sources

site_sources <- function(root = ".") {
  roots <- c("R", "man", "vignettes", "pkgdown", "inst/CITATION", "../NEWS.md",
             "README.md", "_pkgdown.yml", "DESCRIPTION")
  files <- unlist(lapply(file.path(root, roots), function(p) {
    if (dir.exists(p)) list.files(p, recursive = TRUE, full.names = TRUE) else p
  }))
  files <- files[file.exists(files)]
  files <- files[!grepl("_cache/", files, fixed = TRUE)]
  sort(sub("^[.]/", "", files), method = "radix")
}

site_digest <- function(root = ".") {
  files <- site_sources(root)
  sums <- unname(tools::md5sum(files))
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
