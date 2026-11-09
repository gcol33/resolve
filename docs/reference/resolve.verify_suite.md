# Check a Model Suite Against Its Manifest

Read and validate a suite's manifest without loading any model, then
compare every member file (presence, size, SHA-256) with it.

## Usage

``` r
resolve.verify_suite(dir)
```

## Arguments

- dir:

  Path to the suite directory.

## Value

A list with `manifest` (the manifest as a nested list) and `problems` (a
character vector, empty when every member is intact).

## Examples

``` r
source(system.file("examples", "habitat-suite.R", package = "resolveR"))
resolve.seal_suite(suite_dir)
check <- resolve.verify_suite(suite_dir)
length(check$problems) == 0
```
