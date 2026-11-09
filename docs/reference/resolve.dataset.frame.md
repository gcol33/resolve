# Load a ResolveDataset from in-memory data frames

In-memory analog of
[`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md).
Builds and encodes a dataset directly from data frames already in R,
without writing them to a CSV first, for a header filtered or subset
before a fit. The result is identical to
[`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md)
on the equivalent CSV.

## Usage

``` r
resolve.dataset.frame(
  header,
  species = NULL,
  roles = list(),
  targets = list(),
  config = list(),
  schemaSource = NULL,
  vocabs = NULL,
  names = NULL,
  maxEditDistance = 1L
)
```

## Arguments

- header:

  A data.frame of header data (one row per plot). In single-table mode
  (`species = NULL`) this is instead the long-format species frame,
  whose first row of each plot carries the plot's coordinates and
  targets (covariates need a header).

- species:

  A data.frame of species data (long format), a single file path string
  (the large species table is then read once from disk while the header
  stays in memory), or `NULL` for single-table mode.

- roles:

  Named list mapping roles to column names (see
  [`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md)).

- targets:

  Named list of target specifications.

- config:

  Named list of dataset configuration options.

- schemaSource:

  Optional ResolveDataset whose vocabularies / class mappings are reused
  (see
  [`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md)).

- vocabs:

  Optional vocabulary list from `predictor$vocabs()` (see
  [`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md)).
  Mutually exclusive with `schemaSource`.

- names, maxEditDistance:

  Resolve the species names first (see
  [`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md)).

## Value

A ResolveDataset object.

## See also

[`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md)
