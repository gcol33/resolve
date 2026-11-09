# Prediction metrics

The RESOLVE engine's metric functions, computed over a vector of
predictions and the matching vector of observed values. They are the
metrics
[`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md)
reports for each target, made available so a prediction scored elsewhere
(a reloaded checkpoint, a held-out survey) can be evaluated the same
way. The `resolve_c` backend must be loaded
([`resolve.available()`](https://gillescolling.com/resolve/reference/resolve.available.md)).

## Usage

``` r
resolve_band_accuracy(pred, target, threshold = 0.25)

resolve_mae(pred, target)

resolve_rmse(pred, target)

resolve_smape(pred, target, eps = 1e-08)

resolve_accuracy(pred, target)

resolve_r_squared(pred, target)
```

## Arguments

- pred:

  Numeric vector of predicted values (for `resolve_accuracy()`,
  predicted class codes).

- target:

  Numeric vector of observed values, the same length as `pred` (for
  `resolve_accuracy()`, the true class codes).

- threshold:

  Relative half-width of the band for `resolve_band_accuracy()`: a
  prediction counts as correct when it lies within
  `target * (1 - threshold)` and `target * (1 + threshold)`.

- eps:

  Small constant added to the denominator of the symmetric mean absolute
  percentage error to avoid division by zero.

## Value

A single number: the mean absolute error (`resolve_mae()`), the root
mean squared error (`resolve_rmse()`), the symmetric mean absolute
percentage error in `[0, 2]` (`resolve_smape()`), the coefficient of
determination (`resolve_r_squared()`), the share of predictions inside
the relative band (`resolve_band_accuracy()`), or the share of exactly
matching class codes (`resolve_accuracy()`).

## Examples

``` r
pred <- c(1.0, 2.2, 2.9, 4.1)
obs <- c(1, 2, 3, 4)
resolve_mae(pred, obs)
resolve_rmse(pred, obs)
resolve_r_squared(pred, obs)
resolve_band_accuracy(pred, obs, threshold = 0.05)
```
