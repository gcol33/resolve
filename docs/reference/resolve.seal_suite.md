# Seal a Model Suite Being Built

The last step of building a suite: read `dir/manifest.json`, whose
members may still lack their `sha256` and `bytes`, fill both from the
weight files under `dir`, validate the whole manifest, and write it
back. Afterwards
[`resolve.load_suite()`](https://gillescolling.com/resolve/reference/resolve.load_suite.md)
refuses any member file that changes.

## Usage

``` r
resolve.seal_suite(dir)
```

## Arguments

- dir:

  Path to the suite directory.

## Value

The sealed manifest as a nested list, invisibly.

## Examples

``` r
source(system.file("examples", "habitat-suite.R", package = "resolveR"))
sealed <- resolve.seal_suite(suite_dir)
sealed$targets[[1]]$members[[1]]$sha256
```
