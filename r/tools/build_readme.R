# Renders README.Rmd into README.md, the site's home page, and into the repository's README.md,
# the GitHub front page, which sees the package's files one directory down.
#
#   Rscript tools/build_readme.R      run from r/; needs the engine (its usage chunk runs)

rmarkdown::render("README.Rmd", quiet = TRUE)
unlink("README.html")
readme <- readLines("README.md", encoding = "UTF-8")
root <- gsub('src="man/figures/', 'src="r/man/figures/', readme, fixed = TRUE)
write_lf <- function(lines, path) {
  con <- file(path, open = "wb")
  on.exit(close(con))
  writeLines(enc2utf8(lines), con, sep = "\n", useBytes = TRUE)
}
write_lf(readme, "README.md")
write_lf(root, "../README.md")
