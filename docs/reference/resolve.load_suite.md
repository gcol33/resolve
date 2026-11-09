# Load a Model Suite

Load a released set of models scored as one: a directory holding a
`manifest.json` and the weight files it names. Each target of the suite
is predicted by several members (the same recipe trained under different
seeds), and
[`resolve.predict.suite()`](https://gillescolling.com/resolve/reference/resolve.predict.suite.md)
reports their combined prediction beside how far they agree.

## Usage

``` r
resolve.load_suite(
  dir,
  device = "cpu",
  vramFraction = 1,
  verify = TRUE,
  targets = NULL
)
```

## Arguments

- dir:

  Path to the suite directory.

- device:

  `"cpu"` (default), `"cuda"` or `"cuda:N"`.

- vramFraction:

  Fraction of GPU memory the caching allocator may use when loading onto
  CUDA (default 1.0).

- verify:

  Compare every member's size and SHA-256 with the manifest before
  loading (default `TRUE`).

- targets:

  Character vector of suite targets to load; `NULL` (default) loads
  every target.

## Value

A `Suite` object (an Rcpp module class). `suite$manifest()` returns the
manifest as a nested list, `suite$targets()` the loaded targets.

## Details

Loading reads and validates the manifest, compares every member file's
size and SHA-256 with it, loads each member and checks it against the
manifest's input contract (its heads, and the columns it was trained to
read). Any mismatch stops the load.

## See also

[`resolve.predict.suite()`](https://gillescolling.com/resolve/reference/resolve.predict.suite.md),
[`resolve.verify_suite()`](https://gillescolling.com/resolve/reference/resolve.verify_suite.md)

## Examples

``` r
source(system.file("examples", "habitat-suite.R", package = "resolveR"))
resolve.seal_suite(suite_dir)
suite <- resolve.load_suite(suite_dir)
suite$targets()
suite$manifest()$inputs$species
```
