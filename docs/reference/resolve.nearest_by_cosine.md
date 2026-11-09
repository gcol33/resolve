# Nearest Rows by Cosine Similarity

The RESOLVE engine's exact nearest-neighbour search: for each row of
`queries`, the `k` rows of `reference` most similar to it by cosine
similarity, most similar first. The same search finds the reference
plots of
[`resolve.retrieval()`](https://gillescolling.com/resolve/reference/resolve.retrieval.md).
On rows of species presence (1 where a plot records a species, else 0)
the cosine similarity is the Ochiai index of two species lists. The
similarities are computed in blocks, so memory stays bounded whatever
the size of either matrix.

## Usage

``` r
resolve.nearest_by_cosine(
  queries,
  reference,
  k,
  excludeSelf = FALSE,
  device = "cpu"
)
```

## Arguments

- queries:

  Numeric (or logical) matrix, one row per query.

- reference:

  Numeric (or logical) matrix with as many columns as `queries`, one row
  per candidate neighbour.

- k:

  Neighbours per query; at most `nrow(reference)` are returned (one
  fewer with `excludeSelf`).

- excludeSelf:

  `TRUE` when `queries` and `reference` are the same rows in the same
  order (pass the same matrix twice): row i is then never returned as
  its own neighbour, while another row with identical values (a plot
  with the same species list) still is. This is the leave-one-out search
  over one set of plots.

- device:

  Where the similarities are computed: `"cpu"`, `"cuda"` or `"cuda:N"`.

## Value

A list with `indices`, an integer matrix of `nrow(queries)` rows and one
column per neighbour holding rows of `reference` (most similar first),
and `similarities`, the matching matrix of cosine similarities.

## Examples

``` r
set.seed(1)
plots <- matrix(rbinom(20 * 12, 1, 0.4), nrow = 20)
found <- resolve.nearest_by_cosine(plots, plots, k = 3L, excludeSelf = TRUE)
found$indices[1:4, ]
round(found$similarities[1:4, ], 3)
```
