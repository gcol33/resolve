# Builds the RESOLVE site, gillescolling.com/resolve/, from this package into ../docs.
#
#   Rscript build_site.R            run from r/
#
# The articles run their code: the R chunks against the engine in RESOLVE_C_HOME (or the backend
# resolve.install_backend() installed), the Python blocks against the resolve_core that PYTHON
# names. Both have to be the version DESCRIPTION carries.

source("~/.R/build_pkgdown.R")

if (!requireNamespace("resolveR", quietly = TRUE) ||
    as.character(packageVersion("resolveR")) != read.dcf("DESCRIPTION", "Version")[[1]]) {
  stop("Install this tree first: R CMD INSTALL r", call. = FALSE)
}
if (!resolveR::resolve.available()) {
  stop("The resolve_c backend is not loaded. Set RESOLVE_C_HOME or run resolveR::resolve.install_backend().",
       call. = FALSE)
}

python <- Sys.getenv("PYTHON", unset = "python")
if (system2(python, "tools/python_twins.py") != 0L) {
  stop("tools/python_twins.py failed under ", python,
       ". Set PYTHON to the interpreter resolve_core is installed in.", call. = FALSE)
}

source("tools/build_readme.R")

# The changelog is the engine's, at the repository root. pkgdown reads one release per top-level
# heading naming the package and version, where the root file has `## v1.0.3 (2026-10-25)` under a
# `# RESOLVE Changelog` title, so each heading moves up a level on the way.
news <- readLines("../NEWS.md", encoding = "UTF-8")
news <- news[!grepl("^# ", news)]
news <- sub("^## v?([0-9][^ ]*)( .*)?$", "# resolveR \\1", news)
news <- sub("^#(#{2,5}) ", "\\1 ", news)
con <- file("NEWS.md", open = "wb")
writeLines(enc2utf8(news), con, sep = "\n", useBytes = TRUE)
close(con)

# Marks the site build for the articles; pkgdown/docs_cache.R holds their chunk settings.
Sys.setenv(RESOLVE_DOCS_CACHE = normalizePath("pkgdown/docs_cache.R"))
pkgdown::clean_site(quiet = TRUE, force = TRUE)
build_pkgdown_site()
# The digest the site workflow checks the committed build against.
system2("Rscript", c("tools/site_digest.R", "--write"))
