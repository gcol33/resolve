# Release Fine-Tuned Models as a Suite

Release checkpoints fine-tuned from a suite's members (see
[`resolve.parent()`](https://gillescolling.com/resolve/reference/resolve.parent.md))
as a model suite of their own, which
[`resolve.load_suite()`](https://gillescolling.com/resolve/reference/resolve.load_suite.md)
and
[`resolve.predict.suite()`](https://gillescolling.com/resolve/reference/resolve.predict.suite.md)
read like any other. Each checkpoint is copied into `dir`, and the
manifest written beside them takes its input contract from the parent
suite (the part of it the targets' members read), names for each target
the parent suite and target its members started from, with that suite's
licence and training scope, and records per member the SHA-256 of its
parent checkpoint, which the checkpoint records too. The manifest is
sealed and the suite loaded once, so every member is checked against it.

## Usage

``` r
resolve.derive_suite(
  dir,
  parent,
  name,
  licence,
  trainingScope,
  targets,
  description = ""
)
```

## Arguments

- dir:

  Directory the suite is written to; must not hold a manifest yet.

- parent:

  Path of the suite directory the members were fine-tuned from.

- name:

  The suite's name.

- licence:

  The licence the suite is released under.

- trainingScope:

  The plots the fine-tuned members were trained on.

- targets:

  A named list with one element per target, named by the target: a list
  with `members`, a named vector or list of checkpoint paths named by
  their seed (`c("1" = "seed1.pt", "2" = "seed2.pt")`), and optionally
  `outputs` (the heads read; default the head named as the target),
  `combine` (`"vote"`, `"mean"`, `"circular_mean"`, `"spherical_mean"`;
  default a vote for a classification head, a mean for a regression
  head), `period`, `units`, `status` (`"released"` or `"experimental"`),
  `limit` (required for an experimental target), `training` and
  `validation` (named lists recorded as they are).

- description:

  What the suite imputes.

## Value

The manifest written, as a nested list, invisibly.

## Examples

``` r
source(system.file("examples", "habitat-suite.R", package = "resolveR"))
resolve.seal_suite(suite_dir)
parent <- resolve.parent(suite_dir, target = "habitat", seed = 1L)
tuned <- resolve.dataset.csv(plots, species, roles = roles,
                             targets = list(elevation = list(column = "elevation",
                                                             task = "regression")),
                             vocabs = parent$vocabs(), config = parent$dataset_config())
path <- file.path(tempdir(), "elevation_seed1.pt")
fit <- resolve.train.dataset(tuned, initFrom = parent, freezeEncoder = TRUE,
                             maxEpochs = 20L, savePath = path, verbose = FALSE)
derived <- file.path(tempdir(), "elevation_suite")
unlink(derived, recursive = TRUE)
manifest <- resolve.derive_suite(derived, parent = suite_dir, name = "elevation_example",
                                 licence = "CC-BY-4.0",
                                 trainingScope = "150 synthetic Alpine plots",
                                 targets = list(elevation = list(members = c("1" = path),
                                                                 units = "m")))
manifest$targets[[1]]$derived_from$target
```
