# Check Training Progress

Read the `progress.json` file that a training run with a checkpoint
directory (`checkpointDir` in
[`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md))
updates after every checkpoint, so a long run can be followed from
another R session.

## Usage

``` r
resolve.progress(checkpointDir)
```

## Arguments

- checkpointDir:

  Path to the checkpoint directory of a training run.

## Value

A named list with the fields of the progress file (the current `epoch`,
`maxEpochs`, the percentage `progressPct`, and the latest losses), or
`NULL` when the directory holds no progress file yet.

## Examples

``` r
# No run has written to this directory, so there is no progress to report.
resolve.progress(tempdir())

# A progress file as a training run writes it.
dir <- file.path(tempdir(), "resolve-progress-example")
dir.create(dir, showWarnings = FALSE)
writeLines('{"epoch": 12, "maxEpochs": 100, "progressPct": 12.0}',
           file.path(dir, "progress.json"))
progress <- resolve.progress(dir)
progress$epoch
unlink(dir, recursive = TRUE)
```
