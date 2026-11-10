# Open a Trained Model to Fine-Tune From

A fine-tuned model starts from a trained one, its parent: a checkpoint,
or a member of a model suite such as the released EVA suites. It reads
the vegetation plots exactly as the parent does (the parent's species,
genus, family and categorical vocabularies, its loader settings and the
scaling of its coordinates and covariates), starts from the parent's
encoder, and trains heads for your own targets: a national habitat
typology, a soil or management variable, or a released target refitted
on regional plots.

## Usage

``` r
resolve.parent(source, target = NULL, seed = NULL)
```

## Arguments

- source:

  Path of a checkpoint, or of a suite directory (one holding
  `manifest.json`).

- target:

  For a suite: the target whose member to start from. May be left `NULL`
  for a suite of one target.

- seed:

  For a suite: the member's seed (`NULL`, the default, takes the
  target's first member).

## Value

A `Parent` object (an Rcpp module class): `vocabs()` and
`dataset_config()` are what the fine-tuning dataset is loaded with,
`model_config()` the configuration the fine-tuned model takes,
`schema()` and `scalers()` the parent's, and `record()` the record a
fine-tuned checkpoint keeps of it (path, SHA-256, size, engine version,
and the suite, target and seed it came from). A suite member's size and
SHA-256 are checked against the manifest before it is read.

## Details

Load the fine-tuning plots with the parent's vocabularies and loader
settings, then pass the parent to
[`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md)
as `initFrom`:

    parent <- resolve.parent(suite_dir, target = "habitat")
    dataset <- resolve.dataset.csv(header, species, roles, targets,
                                   vocabs = parent$vocabs(),
                                   config = parent$dataset_config())
    fit <- resolve.train.dataset(dataset, initFrom = parent, freezeEncoder = TRUE)

## Examples

``` r
source(system.file("examples", "habitat-suite.R", package = "resolveR"))
resolve.seal_suite(suite_dir)
parent <- resolve.parent(suite_dir, target = "habitat", seed = 1L)
parent$record()$suite_target
```
