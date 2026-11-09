#!/usr/bin/env Rscript
# One digest over every source the site is rendered from. build_site.R writes it into docs/ after
# a build, and the site workflow recomputes it from the checkout and fails the push whose docs/
# were built from other sources. The site is built locally, by a tool that post-processes the
# figures, and this is what holds the committed build to the sources it claims to render.
#
#   Rscript tools/site_digest.R            print the digest of the sources
#   Rscript tools/site_digest.R --write    write it to docs/site-digest.txt
#   Rscript tools/site_digest.R --check    compare docs/site-digest.txt with the sources

# The sources are the files under these paths that git tracks or would track (not ignored), so a
# new article counts before its first commit, and an ignored file (an editor backup) never does.
site_sources <- function(root = ".") {
  roots <- c("R", "man", "vignettes", "pkgdown", "inst/CITATION", "../NEWS.md",
             "README.md", "_pkgdown.yml", "DESCRIPTION")
  files <- system2("git", c("-C", shQuote(root), "-c", "core.quotepath=off", "ls-files", "--cached", "--others",
                          "--exclude-standard", "--", roots),
                   stdout = TRUE)
  if (!is.null(attr(files, "status"))) stop("git ls-files failed in ", root)
  files <- files[file.exists(file.path(root, files))]
  files <- files[!grepl("_cache/", files, fixed = TRUE)]
  sort(files, method = "radix")
}

# A file's MD5 with every CRLF read as LF: git may check a file out with either line ending (a
# Windows working tree keeps CRLF where the index holds LF), and both are the same source.
content_md5 <- function(path) {
  bytes <- readBin(path, "raw", file.size(path))
  n <- length(bytes)
  if (n > 1L) {
    cr <- which(bytes[-n] == as.raw(13L) & bytes[-1L] == as.raw(10L))
    if (length(cr)) bytes <- bytes[-cr]
  }
  tools::md5sum(bytes = bytes)
}

site_digest <- function(root = ".") {
  files <- site_sources(root)
  sums <- vapply(file.path(root, files), content_md5, character(1), USE.NAMES = FALSE)
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
