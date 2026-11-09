# Seed the Engine's Random Numbers

Seed the random number generators of the RESOLVE engine's torch, which
draw a new model's starting weights and its dropout. Training holds out
its plots and shuffles them by its own `seed` argument, so a model built
and trained after the same `resolve.manual_seed()`, with the same data,
settings, device and RESOLVE engine build, ends at the same weights and
metrics. R's [`set.seed()`](https://rdrr.io/r/base/Random.html) does not
reach the RESOLVE engine.

## Usage

``` r
resolve.manual_seed(seed)
```

## Arguments

- seed:

  A whole number from 0 to 2^53.

## Value

No return value.

## Examples

``` r
resolve.manual_seed(42)
```
