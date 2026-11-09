# Default Configuration Values

The RESOLVE engine's default for every field of a configuration: the
values a field takes when a config list or a `NULL` argument leaves it
out. Every verb that takes a config list reads it against these fields,
so a key outside them is refused, at any level of nesting.

## Usage

``` r
resolve.default_config(which)
```

## Arguments

- which:

  `"dataset"` (the `config` of
  [`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md)),
  `"model"` and `"train"` (`modelConfig` and `trainConfig` of
  [`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md)),
  `"spatial_block"`
  ([`resolve.cross_validate()`](https://gillescolling.com/resolve/reference/resolve.cross_validate.md)),
  `"retrieval"`
  ([`resolve.retrieval()`](https://gillescolling.com/resolve/reference/resolve.retrieval.md)),
  `"pretrain"` and `"masked_species"` (the `pretrain` argument of
  [`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md)),
  or `"vae"`
  ([`resolve.pretrain_vae()`](https://gillescolling.com/resolve/reference/resolve.pretrain_vae.md)).

## Value

A named list, one element per field.

## Examples

``` r
resolve.default_config("train")$max_epochs
names(resolve.default_config("model"))
```
