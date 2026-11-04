# Chunk settings for the site build, switched on by build_site.R through RESOLVE_DOCS_CACHE.
# The engine's datasets, trainers and predictors are external pointers into the resolve_c
# library, which a knitr cache cannot restore: a chunk replayed from the cache would hand later
# chunks a dead pointer. Every chunk therefore runs on every build.
knitr::opts_chunk$set(cache = FALSE)
