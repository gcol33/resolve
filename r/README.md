
# resolveR <img src="man/figures/logo.svg" align="right" height="139" alt="resolveR logo" />

*predicting sample attributes from the species or items observed in it*

<!-- badges: start -->

[![R-CMD-check](https://github.com/gcol33/resolve/actions/workflows/r-cmd-check.yml/badge.svg)](https://github.com/gcol33/resolve/actions/workflows/r-cmd-check.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://gillescolling.com/resolve/r/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- badges: end -->

resolveR predicts attributes of a sample from the set of things observed
in it, such as the area, elevation or habitat type of a vegetation plot
from its species list.

The package is the R interface to
[RESOLVE](https://gillescolling.com/resolve/), a C++ engine on libtorch,
the C++ core of [PyTorch](https://pytorch.org/). The Python package and
the command line call the same engine, so a model trained from R loads
in either of them. A sample’s composition is a variable-length set, and
the model pools the effect of each entity, weighted by its abundance,
into one latent representation of the sample. Every target is read off
that representation, so a continuous target and a categorical one are
fitted together. The fit holds out a share of the samples and reports
metrics per target, so you can compare encodings and inputs on your own
data and choose the model to keep.

resolveR supports:

- **Several targets from one encoder.** Regression and classification
  targets share the latent representation. A target labelled on only
  some of the samples keeps the rest in the fit when its `missing` field
  is `"mask"`.
- **Five encodings of the species set.** Feature hashing, learned
  embeddings, an explicit abundance vector, rank pooling over shared
  species, genus and family tables, and a transformer over species
  tokens.
- **Covariates, coordinates and taxonomy.** Numeric covariates, string
  covariates and coordinates enter the same encoder. Genus and family
  give a species the model never saw a fallback identity.
- **Cross-validation.**
  [`resolve.cross_validate()`](https://gillescolling.com/resolve/r/reference/resolve.cross_validate.html)
  refits the model on each of k folds, with spatial blocks when the
  samples have coordinates.
- **GPU training.** A CUDA build of the engine trains on an NVIDIA card,
  and a run that exhausts device memory halves its batch size and
  retries.
- **Model suites.**
  [`resolve.predict.suite()`](https://gillescolling.com/resolve/r/reference/resolve.predict.suite.html)
  scores a table with several trained models at once and reports, per
  sample, the combined prediction and how far the models agree.

The R package holds no model code of its own: every call goes through
the engine’s C interface, the compiled code that the [tests
workflow](https://github.com/gcol33/resolve/actions/workflows/tests.yml)
checks with the engine’s C++ tests and the Python binding tests on every
push, and R CMD check runs on the package itself. The feature-hashing
encoder follows [Weinberger et
al. (2009)](https://doi.org/10.1145/1553374.1553516) and the transformer
encoder [Vaswani et
al. (2017)](https://doi.org/10.48550/arXiv.1706.03762), which a methods
section can cite alongside the package.

## Installation

The package compiles a small C++ interface when it is installed from
GitHub, so it needs a compiler:

| Operating system | Build tools |
|----|----|
| Windows | [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version |
| macOS | `xcode-select --install` |
| Debian / Ubuntu | `r-base-dev` |

Check that R finds them:

``` r
pkgbuild::has_build_tools(debug = TRUE)
```

Then install the package:

``` r
# install.packages("pak")
pak::pak("gcol33/resolve/r")
```

### The engine backend

The dataset, training and prediction functions need the engine, a shared
library that bundles libtorch and is not compiled into the package.
`resolve.install_backend()` downloads a prebuilt build into the
package’s user data directory, and `resolve.available()` reports whether
it is loaded:

``` r
library(resolveR)

resolve.install_backend()
resolve.available()
```

`resolve.install_backend(variant = "cuda")` installs the GPU build
instead. It picks the CUDA 12.8 or CUDA 13 line from the installed
NVIDIA driver and downloads the matching libtorch (2 to 4 GB) from
PyTorch’s server; it is available on Windows and Linux x86-64. Training
then runs on the card with `device = "cuda"`. A backend built elsewhere
is loaded from the directory named in the `RESOLVE_C_HOME` environment
variable.

## Usage

1.  Put one row per sample in a header data frame and one row per
    observation in a species data frame.
2.  Name the columns with `roles`, declare each target, and load both
    tables with `resolve.dataset.frame()`.
3.  Fit the model with `resolve.train.dataset()`.
4.  Save it with `resolve.save()`, load it with `resolve.load()` and
    score a dataset with `resolve.predict.dataset()`.

The example builds 120 synthetic plots whose target `y` is the sum of a
fixed value per species, so a model that learns the composition recovers
it:

``` r
library(resolveR)

values <- setNames(sin(0:11 * 1.7) * 2, paste0("sp", 0:11))
picks <- lapply(0:119, function(i) (i + 0:2 * (i %/% 12 + 1)) %% 12)
y <- vapply(picks, function(s) sum(values[s + 1]), numeric(1))

plots <- data.frame(plot = paste0("P", 0:119), y = y)
records <- data.frame(
  plot = rep(plots$plot, each = 3),
  species = paste0("sp", unlist(picks))
)

dataset <- resolve.dataset.frame(
  header = plots,
  species = records,
  roles = list(plot_id = "plot", species_id = "species"),
  targets = list(y = list(column = "y", task = "regression"))
)

fit <- resolve.train.dataset(dataset)
fit$result$final_metrics

resolve.save(fit$trainer, "model.pt")
predictor <- resolve.load("model.pt")
preds <- resolve.predict.dataset(predictor, dataset)
```

`resolve.train.dataset()` holds out 20 % of the plots for early stopping
and returns the trainer, the result and the dataset. `final_metrics`
holds, per target, the mean absolute error, root mean square error, R²,
symmetric mean absolute percentage error and the share of held-out
predictions within 10, 25 and 50 % of the truth. `preds$predictions` has
one prediction per plot for every target, in the order of
`preds$plot_ids`.

The header table holds the targets and any covariates and coordinates;
the species table has one row per observation of an entity in a sample.
The `roles` list maps your column names onto the sample identifier, the
species identifier and the optional abundance, coordinates, genus,
family and covariate columns. `resolve.dataset.csv()` reads the same
tables from disk. A sample missing a target label is dropped by default,
and `missing = "mask"` in the target’s list keeps it and fits it on its
other targets.

## Encoding the species set

The species set is encoded before the sample encoder sees it, and the
mode is set in the dataset’s `config`: `"hash"`, `"embed"`, `"sparse"`,
`"rank_pool"` or `"transformer"`.

``` r
dataset <- resolve.dataset.frame(
  header = plots,
  species = records,
  roles = list(plot_id = "plot", species_id = "species"),
  targets = list(y = list(column = "y", task = "regression")),
  config = list(species_encoding = "rank_pool", pool_weighting = "log1p")
)
```

[Getting Started with
RESOLVE](https://gillescolling.com/resolve/r/articles/getting-started.html)
shows the settings of each mode, and [Encoding
Modes](https://gillescolling.com/resolve/tutorials/encoding-modes/) on
the RESOLVE site describes each in detail.

## Documentation

Start with [Getting Started with
RESOLVE](https://gillescolling.com/resolve/r/articles/getting-started.html):
one path from two CSV files to a saved model and its predictions.

- [Function reference](https://gillescolling.com/resolve/r/reference/):
  every function of the R package.
- [RESOLVE documentation](https://gillescolling.com/resolve/): the
  engine, with tutorials on data preparation, training, encoding modes,
  prediction and embeddings.

## Getting help

Questions and bug reports go to the [GitHub
issues](https://github.com/gcol33/resolve/issues). A bug report is
easiest to act on with a small reproducible example
([reprex](https://reprex.tidyverse.org/)) and the output of
`resolve.version()`.

## Contributing

Pull requests are welcome; see the [contributing
guide](https://github.com/gcol33/resolve/blob/master/CONTRIBUTING.md).

## Support

> “Software is like sex: it’s better when it’s free.” — Linus Torvalds

I’m a PhD student who builds R packages in my free time because I
believe good tools should be free and open. I started these projects for
my own work and figured others might find them useful too.

If this package saved you some time, buying me a coffee is a nice way to
say thanks. It helps with my coffee addiction.

[![Buy Me A
Coffee](https://img.shields.io/badge/-Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/gcol33)

## License

resolveR is licensed under MIT (see [LICENSE.md](LICENSE.md)).

## Citation

If you use this software, please cite it as below.

``` text
Colling G (2026). _resolveR: Predicting Sample Outcomes from
Compositional Data_. R package version 1.0.3,
<https://github.com/gcol33/resolve>.
```

BibTeX:

``` bibtex
@Manual{colling2026resolver,
  title = {resolveR: Predicting Sample Outcomes from Compositional Data},
  author = {Gilles Colling},
  year = {2026},
  url = {https://github.com/gcol33/resolve},
  note = {R package version 1.0.3},
}
```
