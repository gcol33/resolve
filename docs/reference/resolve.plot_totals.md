# Give a model each plot's totals, or withhold them

For a model trained with `plotTotals = TRUE`, sets whether
[`resolve.predict.dataset`](https://gillescolling.com/resolve/reference/resolve.predict.dataset.md)
gives it each plot's combined cover and species count (`TRUE`, the
default) or withholds them (`FALSE`), as for a plot whose totals are
unknown. A model trained with `plotTotalsDropout` has learned to predict
both ways. Withholding them keeps a prediction from depending on how
many species a plot records, for an analysis of species richness. No
effect on a model without totals.

## Usage

``` r
resolve.plot_totals(predictor, use = TRUE)
```

## Arguments

- predictor:

  A predictor from
  [`resolve.load`](https://gillescolling.com/resolve/reference/resolve.load.md).

- use:

  `TRUE` to give the totals, `FALSE` to withhold them.

## Value

The predictor, invisibly.
